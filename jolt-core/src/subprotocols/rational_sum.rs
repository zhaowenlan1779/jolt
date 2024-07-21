use super::sumcheck::{BatchedCubicSumcheckRationalSum, SumcheckInstanceProof};
use crate::field::{JoltField, OptimizedMul};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::ProofTranscript;
use ark_ff::Zero;
use ark_serialize::*;
use itertools::Itertools;
use rayon::prelude::*;
use std::iter::zip;
use std::mem::take;
use std::ops::{Add, Mul, Sub};

#[derive(CanonicalSerialize, CanonicalDeserialize, Copy, Clone, Debug, PartialEq)]
pub struct Pair<F: JoltField> {
    pub p: F,
    pub q: F,
}

impl<F: JoltField> Pair<F> {
    fn zero() -> Self {
        Self {
            p: F::zero(),
            q: F::one(),
        }
    }

    fn rational_add(a: Pair<F>, b: Pair<F>) -> Pair<F> {
        Pair {
            p: a.p.mul_01_optimized(b.q) + a.q.mul_01_optimized(b.p),
            q: a.q.mul_1_optimized(b.q),
        }
    }
}

impl<F: JoltField> Add<Pair<F>> for Pair<F> {
    type Output = Pair<F>;

    fn add(self, rhs: Pair<F>) -> Self::Output {
        Pair {
            p: self.p + rhs.p,
            q: self.q + rhs.q,
        }
    }
}

impl<F: JoltField> Sub<Pair<F>> for Pair<F> {
    type Output = Pair<F>;

    fn sub(self, rhs: Pair<F>) -> Self::Output {
        Pair {
            p: self.p - rhs.p,
            q: self.q - rhs.q,
        }
    }
}

impl<F: JoltField> Mul<Pair<F>> for Pair<F> {
    type Output = Pair<F>;

    fn mul(self, rhs: Pair<F>) -> Self::Output {
        Pair {
            p: self.p.mul_01_optimized(rhs.p),
            q: self.q.mul_1_optimized(rhs.q),
        }
    }
}

impl<F: JoltField> Mul<F> for Pair<F> {
    type Output = Pair<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Pair {
            p: self.p.mul_01_optimized(rhs),
            q: self.q.mul_1_optimized(rhs),
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedRationalSumLayerProof<F: JoltField> {
    pub proof: SumcheckInstanceProof<F>,
    pub left_claims: Vec<Pair<F>>,
    pub right_claims: Vec<Pair<F>>,
}

impl<F: JoltField> BatchedRationalSumLayerProof<F> {
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> (F, Vec<F>) {
        self.proof
            .verify(claim, num_rounds, degree_bound, transcript)
            .unwrap()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedRationalSumProof<C: CommitmentScheme> {
    pub layers: Vec<BatchedRationalSumLayerProof<C::Field>>,
}

pub trait BatchedRationalSum<F: JoltField, C: CommitmentScheme<Field = F>>: Sized {
    /// The bottom/input layer of the grand products
    type Leaves;

    /// Constructs the grand product circuit(s) from `leaves`
    fn construct(leaves: Self::Leaves) -> Self;
    /// The number of layers in the grand product
    fn num_layers(&self) -> usize;
    /// The claimed outputs of the rational sums (p, q)
    fn claims(&self) -> Vec<Pair<F>>;
    /// Returns an iterator over the layers of this batched grand product circuit.
    /// Each layer is mutable so that its polynomials can be bound over the course
    /// of proving.
    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedRationalSumLayer<F>>;

    /// Computes a batched grand product proof, layer by layer.
    #[tracing::instrument(skip_all, name = "BatchedRationalSum::prove_rational_sum")]
    fn prove_rational_sum(
        &mut self,
        transcript: &mut ProofTranscript,
        _setup: Option<&C::Setup>,
    ) -> (BatchedRationalSumProof<C>, Vec<F>) {
        let mut proof_layers = Vec::with_capacity(self.num_layers());
        let mut claims_to_verify = self.claims();
        let mut r_rational_sum = Vec::new();

        for layer in self.layers() {
            proof_layers.push(layer.prove_layer(
                &mut claims_to_verify,
                &mut r_rational_sum,
                transcript,
            ));
        }

        (
            BatchedRationalSumProof {
                layers: proof_layers,
            },
            r_rational_sum,
        )
    }

    /// Verifies that the `sumcheck_claim` output by sumcheck verification is consistent
    /// with the `left_claims` and `right_claims` of corresponding `BatchedRationalSumLayerProof`.
    /// This function may be overridden if the layer isn't just multiplication gates, e.g. in the
    /// case of `ToggledBatchedRationalSum`.
    fn verify_sumcheck_claim(
        layer_proofs: &[BatchedRationalSumLayerProof<F>],
        layer_index: usize,
        coeffs: &[F],
        sumcheck_claim: F,
        eq_eval: F,
        claims: &mut Vec<Pair<F>>,
        r_rational_sum: &mut Vec<F>,
        lambda_layer: F,
        transcript: &mut ProofTranscript,
    ) {
        let layer_proof = &layer_proofs[layer_index];
        let expected_sumcheck_claim: F = (0..claims.len())
            .map(|i| {
                coeffs[i]
                    * ((layer_proof.right_claims[i].p * layer_proof.left_claims[i].q
                        + (layer_proof.left_claims[i].p
                            + lambda_layer * layer_proof.left_claims[i].q)
                            * layer_proof.right_claims[i].q)
                        * eq_eval)
            })
            .sum();

        assert_eq!(expected_sumcheck_claim, sumcheck_claim);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

        *claims = layer_proof
            .left_claims
            .iter()
            .zip(layer_proof.right_claims.iter())
            .map(|(&left_claim, &right_claim)| (left_claim + (right_claim - left_claim) * r_layer))
            .collect();

        r_rational_sum.push(r_layer);
    }

    /// Function used for layer sumchecks in the generic batch verifier as well as the quark layered sumcheck hybrid
    fn verify_layers(
        proof_layers: &[BatchedRationalSumLayerProof<F>],
        claims: &Vec<Pair<F>>,
        transcript: &mut ProofTranscript,
        r_start: Vec<F>,
    ) -> (Vec<Pair<F>>, Vec<F>) {
        let mut claims_to_verify = claims.to_owned();
        // We allow a non empty start in this function call because the quark hybrid form provides prespecified random for
        // most of the positions and then we proceed with GKR on the remaining layers using the preset random values.
        // For default thaler '13 layered grand products this should be empty.
        let mut r_rational_sum = r_start.clone();
        let fixed_at_start = r_start.len();

        for (layer_index, layer_proof) in proof_layers.iter().enumerate() {
            // produce a fresh set of coeffs
            let coeffs: Vec<F> =
                transcript.challenge_vector(b"rand_coeffs_next_layer", claims_to_verify.len());

            // Random combination for p and q
            let lambda_layer = transcript.challenge_scalar(b"challenge_lambda_layer");

            // produce a joint claim
            let claim = claims_to_verify
                .iter()
                .zip(coeffs.iter())
                .map(|(Pair { p, q }, &coeff)| *p * coeff + *q * coeff * lambda_layer)
                .sum();

            let (sumcheck_claim, r_sumcheck) =
                layer_proof.verify(claim, layer_index + fixed_at_start, 3, transcript);
            assert_eq!(claims.len(), layer_proof.left_claims.len());
            assert_eq!(claims.len(), layer_proof.right_claims.len());

            for (left, right) in layer_proof
                .left_claims
                .iter()
                .zip(layer_proof.right_claims.iter())
            {
                transcript.append_scalar(b"sumcheck left claim p", &left.p);
                transcript.append_scalar(b"sumcheck left claim q", &left.q);
                transcript.append_scalar(b"sumcheck right claim p", &right.p);
                transcript.append_scalar(b"sumcheck right claim q", &right.q);
            }

            assert_eq!(r_rational_sum.len(), r_sumcheck.len());

            let eq_eval: F = r_rational_sum
                .iter()
                .zip_eq(r_sumcheck.iter().rev())
                .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::one() - r_gp) * (F::one() - r_sc))
                .product();

            r_rational_sum = r_sumcheck.into_iter().rev().collect();

            Self::verify_sumcheck_claim(
                proof_layers,
                layer_index,
                &coeffs,
                sumcheck_claim,
                eq_eval,
                &mut claims_to_verify,
                &mut r_rational_sum,
                lambda_layer,
                transcript,
            );
        }

        (claims_to_verify, r_rational_sum)
    }

    /// Verifies the given grand product proof.
    #[tracing::instrument(skip_all, name = "BatchedRationalSum::verify_rational_sum")]
    fn verify_rational_sum(
        proof: &BatchedRationalSumProof<C>,
        claims: &Vec<Pair<F>>,
        transcript: &mut ProofTranscript,
        _setup: Option<&C::Setup>,
    ) -> (Vec<Pair<F>>, Vec<F>) {
        // Pass the inputs to the layer verification function, by default we have no quarks and so we do not
        // use the quark proof fields.
        let r_start = Vec::<F>::new();
        Self::verify_layers(&proof.layers, claims, transcript, r_start)
    }
}

pub trait BatchedRationalSumLayer<F: JoltField>:
    BatchedCubicSumcheckRationalSum<F, Pair<F>>
{
    /// Proves a single layer of a batched grand product circuit
    fn prove_layer(
        &mut self,
        claims: &mut Vec<Pair<F>>,
        r_rational_sum: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> BatchedRationalSumLayerProof<F> {
        // produce a fresh set of coeffs
        let coeffs: Vec<F> = transcript.challenge_vector(b"rand_coeffs_next_layer", claims.len());
        // Random combination for p and q
        let lambda_layer: F = transcript.challenge_scalar(b"challenge_lambda_layer");
        // produce a joint claim
        let claim = claims
            .iter()
            .zip(coeffs.iter())
            .map(|(Pair { p, q }, &coeff)| *p * coeff + *q * coeff * lambda_layer)
            .sum();

        let mut eq_poly = DensePolynomial::new(EqPolynomial::<F>::evals(r_rational_sum));

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_sumcheck(&claim, &coeffs, &mut eq_poly, transcript, &lambda_layer);

        drop_in_background_thread(eq_poly);

        let (left_claims, right_claims) = sumcheck_claims;
        for (left, right) in left_claims.iter().zip(right_claims.iter()) {
            transcript.append_scalar(b"sumcheck left claim p", &left.p);
            transcript.append_scalar(b"sumcheck left claim q", &left.q);
            transcript.append_scalar(b"sumcheck right claim p", &right.p);
            transcript.append_scalar(b"sumcheck right claim q", &right.q);
        }

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_rational_sum);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

        *claims = left_claims
            .iter()
            .zip(right_claims.iter())
            .map(|(&left_claim, &right_claim)| left_claim + (right_claim - left_claim) * r_layer)
            .collect();

        r_rational_sum.push(r_layer);

        BatchedRationalSumLayerProof {
            proof: sumcheck_proof,
            left_claims,
            right_claims,
        }
    }
}

/// Represents a single layer of a single grand product circuit.
/// A layer is assumed to be arranged in "interleaved" order, i.e. the natural
/// order in the visual representation of the circuit:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      / \
///   L0   R0  L1   R1  L2   R2  L3   R3   <- This is layer would be represented as [L0, R0, L1, R1, L2, R2, L3, R3]
///                                           (as opposed to e.g. [L0, L1, L2, L3, R0, R1, R2, R3])
pub type DenseRationalSumLayer<F> = Vec<Pair<F>>;

/// Represents a batch of `DenseRationalSumLayer`, all of the same length `layer_len`.
#[derive(Debug, Clone)]
pub struct BatchedDenseRationalSumLayer<F: JoltField, const C: usize> {
    pub layers_p: Vec<Vec<F>>,
    pub layers_q: Vec<Vec<F>>,
    pub layer_len: usize,
}

impl<F: JoltField, const C: usize> BatchedDenseRationalSumLayer<F, C> {
    pub fn new(layers_p: Vec<Vec<F>>, layers_q: Vec<Vec<F>>) -> Self {
        let layer_len = layers_p[0].len();
        Self {
            layers_p,
            layers_q,
            layer_len,
        }
    }
}

impl<F: JoltField, const C: usize> BatchedRationalSumLayer<F>
    for BatchedDenseRationalSumLayer<F, C>
{
}
impl<F: JoltField, const C: usize> BatchedCubicSumcheckRationalSum<F, Pair<F>>
    for BatchedDenseRationalSumLayer<F, C>
{
    fn num_rounds(&self) -> usize {
        self.layer_len.log_2() - 1
    }

    /// Incrementally binds a variable of this batched layer's polynomials.
    /// Even though each layer is backed by a single Vec<F>, it represents two polynomials
    /// one for the left nodes in the circuit, one for the right nodes in the circuit.
    /// These two polynomials' coefficients are interleaved into one Vec<F>. To preserve
    /// this interleaved order, we bind values like this:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    #[tracing::instrument(skip_all, name = "BatchedDenseRationalSumLayer::bind")]
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        debug_assert!(self.layer_len % 4 == 0);
        let n = self.layer_len / 4;
        // TODO(moodlezoup): parallelize over chunks instead of over batch
        rayon::scope(|s| {
            s.spawn(|_| {
                self.layers_p.par_iter_mut().for_each(|layer: &mut Vec<F>| {
                    for i in 0..n {
                        // left
                        layer[2 * i] = layer[4 * i] + (layer[4 * i + 2] - layer[4 * i]) * *r;
                        // right
                        layer[2 * i + 1] =
                            layer[4 * i + 1] + (layer[4 * i + 3] - layer[4 * i + 1]) * *r;
                    }
                })
            });
            s.spawn(|_| {
                self.layers_q.par_iter_mut().for_each(|layer: &mut Vec<F>| {
                    for i in 0..n {
                        // left
                        layer[2 * i] = layer[4 * i] + (layer[4 * i + 2] - layer[4 * i]) * *r;
                        // right
                        layer[2 * i + 1] =
                            layer[4 * i + 1] + (layer[4 * i + 3] - layer[4 * i + 1]) * *r;
                    }
                })
            });
            s.spawn(|_| eq_poly.bound_poly_var_bot(r));
        });
        self.layer_len /= 2;
    }

    /// We want to compute the evaluations of the following univariate cubic polynomial at
    /// points {0, 1, 2, 3}:
    ///     Σ coeff[batch_index] * (Σ eq(r, x) * (right(x).p * left(x).q + (left(x).p + lambda * left(x).q) * right(x).q))
    /// where the inner summation is over all but the "least significant bit" of the multilinear
    /// polynomials `eq`, `left`, and `right`. We denote this "least significant" variable x_b.
    ///
    /// Computing these evaluations requires processing pairs of adjacent coefficients of
    /// `eq`, `left`, and `right`.
    /// Recall that the `left` and `right` polynomials are interleaved in each layer of `self.layers`,
    /// so we process each layer 4 values at a time:
    ///                  layer = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    #[tracing::instrument(skip_all, name = "BatchedDenseRationalSumLayer::compute_cubic")]
    fn compute_cubic(
        &self,
        coeffs: &[F],
        eq_poly: &DensePolynomial<F>,
        previous_round_claim: F,
        lambda: &F,
    ) -> UniPoly<F> {
        let inner_func = |left: Pair<F>, right: Pair<F>| {
            right.p * left.q + (left.p + left.q * *lambda) * right.q
        };

        let evals = (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = {
                    let eval_point_0 = eq_poly[2 * i];
                    let m_eq = eq_poly[2 * i + 1] - eq_poly[2 * i];
                    let eval_point_2 = eq_poly[2 * i + 1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                };
                let mut evals = (F::zero(), F::zero(), F::zero());

                self.layers_p
                    .iter()
                    .enumerate()
                    .for_each(|(batch_index, layer)| {
                        let subtable_index = batch_index / C;

                        let left = (
                            Pair {
                                p: layer[4 * i],
                                q: self.layers_q[subtable_index][4 * i],
                            },
                            Pair {
                                p: layer[4 * i + 2],
                                q: self.layers_q[subtable_index][4 * i + 2],
                            },
                        );
                        let right = (
                            Pair {
                                p: layer[4 * i + 1],
                                q: self.layers_q[subtable_index][4 * i + 1],
                            },
                            Pair {
                                p: layer[4 * i + 3],
                                q: self.layers_q[subtable_index][4 * i + 3],
                            },
                        );

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        evals.0 += coeffs[batch_index] * inner_func(left.0, right.0);
                        evals.1 += coeffs[batch_index] * inner_func(left_eval_2, right_eval_2);
                        evals.2 += coeffs[batch_index] * inner_func(left_eval_3, right_eval_3);
                    });

                evals.0 *= eq_evals.0;
                evals.1 *= eq_evals.1;
                evals.2 *= eq_evals.2;
                evals
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            );

        let evals = [evals.0, previous_round_claim - evals.0, evals.1, evals.2];
        UniPoly::from_evals(&evals)
    }

    fn final_claims(&self) -> (Vec<Pair<F>>, Vec<Pair<F>>) {
        assert_eq!(self.layer_len, 2);
        let (left_claims, right_claims) = self
            .layers_p
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                (
                    Pair {
                        p: layer[0],
                        q: self.layers_q[i / C][0],
                    },
                    Pair {
                        p: layer[1],
                        q: self.layers_q[i / C][1],
                    },
                )
            })
            .unzip();
        (left_claims, right_claims)
    }
}

/// A batched grand product circuit.
/// Note that the circuit roots are not included in `self.layers`
///        o
///      /   \
///     o     o  <- layers[layers.len() - 1]
///    / \   / \
///   o   o o   o  <- layers[layers.len() - 2]
///       ...
pub struct BatchedDenseRationalSum<F: JoltField, const C: usize> {
    layers: Vec<BatchedDenseRationalSumLayer<F, C>>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, const C: usize> BatchedRationalSum<F, PCS>
    for BatchedDenseRationalSum<F, C>
{
    type Leaves = (Vec<Vec<F>>, Vec<Vec<F>>);

    #[tracing::instrument(skip_all, name = "BatchedDenseRationalSum::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let (leaves_p, leaves_q) = leaves;
        let num_layers = leaves_p[0].len().log_2();
        let mut layers: Vec<BatchedDenseRationalSumLayer<F, C>> = Vec::with_capacity(num_layers);
        layers.push(BatchedDenseRationalSumLayer::new(leaves_p, leaves_q));

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers.layer_len / 2;
            // TODO(moodlezoup): parallelize over chunks instead of over batch

            // Parallelize new_layers_p & new_layers_q ?
            let new_layers_p = previous_layers
                .layers_p
                .par_iter()
                .enumerate()
                .map(|(memory_idx, previous_layer_p)| {
                    let subtable_idx = memory_idx / C;
                    let previous_layer_q = &previous_layers.layers_q[subtable_idx];

                    (0..len)
                        .map(|i| {
                            previous_layer_p[2 * i] * previous_layer_q[2 * i + 1]
                                + previous_layer_p[2 * i + 1] * previous_layer_q[2 * i]
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let new_layers_q = previous_layers
                .layers_q
                .par_iter()
                .map(|previous_layer| {
                    (0..len)
                        .map(|i| previous_layer[2 * i] * previous_layer[2 * i + 1])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            layers.push(BatchedDenseRationalSumLayer::new(
                new_layers_p,
                new_layers_q,
            ));
        }

        Self { layers }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claims(&self) -> Vec<Pair<F>> {
        let num_layers =
            <BatchedDenseRationalSum<F, C> as BatchedRationalSum<F, PCS>>::num_layers(self);
        let last_layers = &self.layers[num_layers - 1];
        assert_eq!(last_layers.layer_len, 2);
        last_layers
            .layers_p
            .iter()
            .enumerate()
            .map(|(i, layer_p)| {
                let layer_q = &last_layers.layers_q[i / C];
                Pair {
                    p: layer_p[0] * layer_q[1] + layer_p[1] * layer_q[0],
                    q: layer_q[0] * layer_q[1],
                }
            })
            .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedRationalSumLayer<F>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedRationalSumLayer<F>)
            .rev()
    }
}

/// Represents a single layer of a single grand product circuit using a sparse vector,
/// i.e. a vector containing (index, value) pairs.
/// with values {p: 0, q: 1} omitted
pub type SparseRationalSumLayer<F> = Vec<(usize, Pair<F>)>;

/// A "dynamic density" grand product layer can switch from sparse representation
/// to dense representation once it's no longer sparse (after binding).
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicDensityRationalSumLayer<F: JoltField> {
    Sparse(SparseRationalSumLayer<F>),
    Dense(DenseRationalSumLayer<F>),
}

/// This constant determines:
///     - whether the `layer_output` of a `DynamicDensityRationalSumLayer` is dense
///       or sparse
///     - when to switch from sparse to dense representation during the binding of a
///       `DynamicDensityRationalSumLayer`
/// If the layer has >DENSIFICATION_THRESHOLD fraction of non-1 values, it'll switch
/// to the dense representation. Value tuned experimentally.
const DENSIFICATION_THRESHOLD: f64 = 0.8;

impl<F: JoltField> DynamicDensityRationalSumLayer<F> {
    /// Computes the grand product layer that is output by this layer.
    ///     L0'      R0'      L1'      R1'     <- output layer
    ///      Λ        Λ        Λ        Λ
    ///     / \      / \      / \      / \
    ///   L0   R0  L1   R1  L2   R2  L3   R3   <- this layer
    ///
    /// If the current layer is dense, the output layer will be dense.
    /// If the current layer is sparse, but already not very sparse (as parametrized by
    /// `DENSIFICATION_THRESHOLD`), the output layer will be dense.
    /// Otherwise, the output layer will be sparse.
    pub fn layer_output<const C: usize>(
        &self,
        output_len: usize,
        memory_index: usize,
        preprocessing: &Vec<Vec<F>>,
        preprocessing_next_layer: &Vec<Vec<F>>,
    ) -> Self {
        let subtable_index = memory_index / C;
        let default = &preprocessing[subtable_index];
        let get_default = |index| Pair {
            p: F::zero(),
            q: default[index],
        };

        match self {
            DynamicDensityRationalSumLayer::Sparse(sparse_layer) => {
                if (sparse_layer.len() as f64 / (output_len * 2) as f64) > DENSIFICATION_THRESHOLD {
                    // Current layer is already not very sparse, so make the next layer dense
                    let mut output_layer: DenseRationalSumLayer<F> = preprocessing_next_layer
                        [subtable_index]
                        .iter()
                        .map(|q| Pair {
                            p: F::zero(),
                            q: *q,
                        })
                        .collect();
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, get_default(index + 1)));
                            if right.0 == index + 1 {
                                output_layer[index / 2] = Pair::rational_add(right.1, *value);
                            } else {
                                output_layer[index / 2] =
                                    Pair::rational_add(get_default(index + 1), *value);
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration
                            output_layer[index / 2] =
                                Pair::rational_add(get_default(index - 1), *value);
                            next_index_to_process = index + 1;
                        }
                    }
                    DynamicDensityRationalSumLayer::Dense(output_layer)
                } else {
                    // Current layer is still pretty sparse, so make the next layer sparse
                    let mut output_layer: SparseRationalSumLayer<F> =
                        Vec::with_capacity(output_len);
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, get_default(index + 1)));
                            if right.0 == index + 1 {
                                // Corresponding right node was found; multiply them together
                                output_layer.push((index / 2, Pair::rational_add(right.1, *value)));
                            } else {
                                // Corresponding right node not found
                                output_layer.push((
                                    index / 2,
                                    Pair::rational_add(get_default(index + 1), *value),
                                ));
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration
                            output_layer.push((
                                index / 2,
                                Pair::rational_add(get_default(index - 1), *value),
                            ));
                            next_index_to_process = index + 1;
                        }
                    }
                    DynamicDensityRationalSumLayer::Sparse(output_layer)
                }
            }
            DynamicDensityRationalSumLayer::Dense(dense_layer) => {
                #[cfg(test)]
                let product = dense_layer.iter().copied().reduce(Pair::rational_add);

                // If current layer is dense, next layer should also be dense.
                let output_layer: DenseRationalSumLayer<F> = (0..output_len)
                    .map(|i| Pair::rational_add(dense_layer[2 * i], dense_layer[2 * i + 1]))
                    .collect();
                #[cfg(test)]
                {
                    let output_product = output_layer.iter().copied().reduce(Pair::rational_add);
                    assert_eq!(product, output_product);
                }
                DynamicDensityRationalSumLayer::Dense(output_layer)
            }
        }
    }
}

/// Represents a batch of `DynamicDensityRationalSumLayer`, all of which have the same
/// size `layer_len`. Note that within a single batch, some layers may be represented by
/// sparse vectors and others by dense vectors.
#[derive(Debug, Clone)]
pub struct BatchedSparseRationalSumLayer<F: JoltField, const C: usize> {
    pub layer_len: usize,
    pub layers: Vec<DynamicDensityRationalSumLayer<F>>,
    pub preprocessing: Vec<Vec<F>>,
    pub preprocessing_next: Vec<Vec<F>>,
}

impl<F: JoltField, const C: usize> BatchedSparseRationalSumLayer<F, C> {
    fn memory_to_subtable_index(i: usize) -> usize {
        i / C
    }
}

impl<F: JoltField, const C: usize> BatchedRationalSumLayer<F>
    for BatchedSparseRationalSumLayer<F, C>
{
}
impl<F: JoltField, const C: usize> BatchedCubicSumcheckRationalSum<F, Pair<F>>
    for BatchedSparseRationalSumLayer<F, C>
{
    fn num_rounds(&self) -> usize {
        self.layer_len.log_2() - 1
    }

    /// Incrementally binds a variable of this batched layer's polynomials.
    /// If `self` is dense, we bind as in `BatchedDenseRationalSumLayer`,
    /// processing nodes 4 at a time to preserve the interleaved order:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    /// If `self` is sparse, we basically do the same thing but with more
    /// cases to check 😬
    #[tracing::instrument(skip_all, name = "BatchedSparseRationalSumLayer::bind")]
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        debug_assert!(self.layer_len % 4 == 0);

        self.preprocessing_next = vec![vec![]; self.preprocessing.len()];

        rayon::join(
            || {
                (
                    self.layers.par_chunks_mut(C),
                    &mut self.preprocessing_next,
                    &self.preprocessing,
                )
                    .into_par_iter()
                    .for_each(|(layers, preprocessing_next, preprocessing)| {
                        let has_sparse = layers.iter().any(|layer| {
                            if let DynamicDensityRationalSumLayer::Sparse(_) = layer {
                                true
                            } else {
                                false
                            }
                        });
                        if has_sparse {
                            let n = preprocessing.len() / 4;
                            *preprocessing_next = vec![F::zero(); preprocessing.len() / 2];
                            for i in 0..n {
                                // left
                                preprocessing_next[2 * i] = preprocessing[4 * i]
                                    + (preprocessing[4 * i + 2] - preprocessing[4 * i]) * *r;
                                // right
                                preprocessing_next[2 * i + 1] = preprocessing[4 * i + 1]
                                    + (preprocessing[4 * i + 3] - preprocessing[4 * i + 1]) * *r;
                            }
                        }

                        layers.iter_mut().for_each(|layer| match layer {
                            DynamicDensityRationalSumLayer::Sparse(sparse_layer) => {
                                let default = &preprocessing;
                                let get_default = |index| Pair {
                                    p: F::zero(),
                                    q: default[index],
                                };

                                let mut dense_bound_layer = if (sparse_layer.len() as f64
                                    / self.layer_len as f64)
                                    > DENSIFICATION_THRESHOLD
                                {
                                    // Current layer is already not very sparse, so make the next layer dense
                                    Some(
                                        preprocessing_next
                                            .iter()
                                            .map(|x| Pair {
                                                p: F::zero(),
                                                q: *x,
                                            })
                                            .collect::<Vec<_>>(),
                                    )
                                } else {
                                    None
                                };

                                let mut num_bound = 0usize;
                                let mut push_to_bound_layer =
                                    |sparse_layer: &mut Vec<(usize, Pair<F>)>,
                                     dense_index: usize,
                                     value: Pair<F>| {
                                        match &mut dense_bound_layer {
                                            Some(ref mut dense_vec) => {
                                                debug_assert_eq!(
                                                    dense_vec[dense_index].p,
                                                    F::zero()
                                                );
                                                dense_vec[dense_index] = value;
                                            }
                                            None => {
                                                sparse_layer[num_bound] = (dense_index, value);
                                            }
                                        };
                                        num_bound += 1;
                                    };

                                let mut next_left_node_to_process = 0usize;
                                let mut next_right_node_to_process = 0usize;

                                for j in 0..sparse_layer.len() {
                                    let (index, value) = sparse_layer[j];
                                    if index % 2 == 0 && index < next_left_node_to_process {
                                        // This left node was already bound with its sibling in a previous iteration
                                        continue;
                                    }
                                    if index % 2 == 1 && index < next_right_node_to_process {
                                        // This right node was already bound with its sibling in a previous iteration
                                        continue;
                                    }

                                    let mut neighbors = vec![None, None];
                                    for k in [j + 1, j + 2] {
                                        if let Some((idx, val)) = sparse_layer.get(k) {
                                            if *idx == index + 1 {
                                                neighbors[0] = Some(*val);
                                            } else if *idx == index + 2 {
                                                neighbors[1] = Some(*val);
                                            } else {
                                                break;
                                            }
                                        }
                                    }

                                    match index % 4 {
                                        0 => {
                                            // Find sibling left node
                                            let sibling_value =
                                                neighbors[1].unwrap_or(get_default(index + 2));
                                            push_to_bound_layer(
                                                sparse_layer,
                                                index / 2,
                                                value + (sibling_value - value) * *r,
                                            );
                                            next_left_node_to_process = index + 4;
                                        }
                                        1 => {
                                            // Edge case: If this right node's neighbor is not 1 and has _not_
                                            // been bound yet, we need to bind the neighbor first to preserve
                                            // the monotonic ordering of the bound layer.
                                            if next_left_node_to_process <= index + 1 {
                                                if let Some(left_neighbor) = neighbors[0] {
                                                    let sibling_value = get_default(index - 1);
                                                    push_to_bound_layer(
                                                        sparse_layer,
                                                        index / 2,
                                                        sibling_value
                                                            + (left_neighbor - sibling_value) * *r,
                                                    );
                                                }
                                                next_left_node_to_process = index + 3;
                                            }

                                            // Find sibling right node
                                            let sibling_value =
                                                neighbors[1].unwrap_or(get_default(index + 2));
                                            push_to_bound_layer(
                                                sparse_layer,
                                                index / 2 + 1,
                                                value + (sibling_value - value) * *r,
                                            );
                                            next_right_node_to_process = index + 4;
                                        }
                                        2 => {
                                            // Sibling left node wasn't encountered in previous iteration,
                                            // so sibling must have value 1.
                                            let sibling_value = get_default(index - 2);
                                            push_to_bound_layer(
                                                sparse_layer,
                                                index / 2 - 1,
                                                sibling_value + (value - sibling_value) * *r,
                                            );
                                            next_left_node_to_process = index + 2;
                                        }
                                        3 => {
                                            // Sibling right node wasn't encountered in previous iteration,
                                            // so sibling must have value 1.
                                            let sibling_value = get_default(index - 2);
                                            push_to_bound_layer(
                                                sparse_layer,
                                                index / 2,
                                                sibling_value + (value - sibling_value) * *r,
                                            );
                                            next_right_node_to_process = index + 2;
                                        }
                                        _ => unreachable!("?_?"),
                                    }
                                }
                                if let Some(dense_vec) = dense_bound_layer {
                                    *layer = DynamicDensityRationalSumLayer::Dense(dense_vec);
                                } else {
                                    sparse_layer.truncate(num_bound);
                                }
                            }
                            DynamicDensityRationalSumLayer::Dense(dense_layer) => {
                                // If current layer is dense, next layer should also be dense.
                                let n = self.layer_len / 4;
                                for i in 0..n {
                                    // left
                                    dense_layer[2 * i] = dense_layer[4 * i]
                                        + (dense_layer[4 * i + 2] - dense_layer[4 * i]) * *r;
                                    // right
                                    dense_layer[2 * i + 1] = dense_layer[4 * i + 1]
                                        + (dense_layer[4 * i + 3] - dense_layer[4 * i + 1]) * *r;
                                }
                            }
                        })
                    });
            },
            || eq_poly.bound_poly_var_bot(r),
        );
        self.layer_len /= 2;

        self.preprocessing = take(&mut self.preprocessing_next);
    }

    /// We want to compute the evaluations of the following univariate cubic polynomial at
    /// points {0, 1, 2, 3}:
    ///     Σ coeff[batch_index] * (Σ eq(r, x) * (right(x).p * left(x).q + (left(x).p + lambda * left(x).q) * right(x).q))
    /// where the inner summation is over all but the "least significant bit" of the multilinear
    /// polynomials `eq`, `left`, and `right`. We denote this "least significant" variable x_b.
    ///
    /// Computing these evaluations requires processing pairs of adjacent coefficients of
    /// `eq`, `left`, and `right`.
    /// If `self` is dense, we process each layer 4 values at a time:
    ///                  layer = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    /// If `self` is sparse, we basically do the same thing but with some fancy optimizations and
    /// more cases to check 😬
    #[tracing::instrument(skip_all, name = "BatchedSparseRationalSumLayer::compute_cubic")]
    fn compute_cubic(
        &self,
        coeffs: &[F],
        eq_poly: &DensePolynomial<F>,
        previous_round_claim: F,
        lambda: &F,
    ) -> UniPoly<F> {
        let inner_func = |left: Pair<F>, right: Pair<F>| {
            right.p * left.q + (left.p + left.q * *lambda) * right.q
        };
        let inner_func_delta = |left: Pair<F>, right: Pair<F>| right.p * left.q + left.p * right.q;

        // Maybe into_par_iter this; but I found that it causes too much overhead
        let eq_evals: Vec<(F, F, F)> = (0..eq_poly.len() / 2)
            .map(|i| {
                let eval_point_0 = eq_poly[2 * i];
                let m_eq = eq_poly[2 * i + 1] - eq_poly[2 * i];
                let eval_point_2 = eq_poly[2 * i + 1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .collect();

        let total_evals: (F, F, F) = (
            self.layers.par_chunks(C),
            coeffs.par_chunks(C),
            &self.preprocessing,
        )
            .into_par_iter()
            .map(|(layers, coeffs, preprocessing)| {
                let default_sum = if preprocessing.len() != 0 {
                    let mut sum = (F::zero(), F::zero(), F::zero());
                    for i in 0..preprocessing.len() / 4 {
                        let (eq_eval_0, eq_eval_2, eq_eval_3) = eq_evals[i];

                        let left = (preprocessing[4 * i], preprocessing[4 * i + 2]);
                        let right = (preprocessing[4 * i + 1], preprocessing[4 * i + 3]);

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        sum.0 += eq_eval_0 * left.0 * right.0;
                        sum.1 += eq_eval_2 * left_eval_2 * right_eval_2;
                        sum.2 += eq_eval_3 * left_eval_3 * right_eval_3;
                    }
                    (sum.0 * *lambda, sum.1 * *lambda, sum.2 * *lambda)
                } else {
                    (F::zero(), F::zero(), F::zero())
                };

                let mut total_eval = (F::zero(), F::zero(), F::zero());
                zip(layers, coeffs).for_each(|(layer, coeff)| match layer {
                    // If sparse, we use the pre-emptively computed `eq_eval_sums` as a starting point:
                    //     eq_eval_sum := Σ eq_evals[i]
                    // What we ultimately want to compute:
                    //     Σ coeff[batch_index] * (Σ eq(r, x) * (right(x).p * left(x).q + (left(x).p + lambda * left(x).q) * right(x).q))
                    // Note that if left[i] and right[i] are all 0s, the inner sum is:
                    //     Σ eq_evals[i] = eq_eval_sum * lambda
                    // To recover the actual inner sum, we find all the non-0
                    // left[i] and right[i] terms and compute the delta:
                    //     ∆ := Σ eq_evals[j] * ((right(x).p * left(x).q + (left(x).p + lambda * left(x).q) * right(x).q) - lambda)
                    // Then we can compute:
                    //    coeff[batch_index] * (eq_eval_sum + ∆)
                    // ...which is exactly the summand we want.
                    DynamicDensityRationalSumLayer::Sparse(sparse_layer) => {
                        let default = &preprocessing;
                        let get_default = |index| Pair {
                            p: F::zero(),
                            q: default[index],
                        };

                        // Computes:
                        //     ∆ := Σ eq_evals[j] * (left[j] * right[j] - 1)    ∀j where left[j] ≠ 1 or right[j] ≠ 1
                        // for the evaluation points {0, 2, 3}
                        let mut sum = default_sum;

                        let mut next_index_to_process = 0usize;
                        for (j, (index, value)) in sparse_layer.iter().enumerate() {
                            if *index < next_index_to_process {
                                // This node was already processed in a previous iteration
                                continue;
                            }

                            let mut neighbors = vec![None, None, None];
                            for k in [j + 1, j + 2, j + 3] {
                                if let Some((idx, val)) = sparse_layer.get(k) {
                                    if *idx == index + 1 {
                                        neighbors[0] = Some(*val);
                                    } else if *idx == index + 2 {
                                        neighbors[1] = Some(*val);
                                    } else if *idx == index + 3 {
                                        neighbors[2] = Some(*val);
                                        break;
                                    } else {
                                        break;
                                    }
                                }
                            }
                            for k in 0..3 {
                                if index + k + 1 >= default.len() {
                                    break;
                                }
                                if neighbors[k] == None {
                                    neighbors[k] = Some(get_default(index + k + 1));
                                }
                            }

                            let find_neighbor = |i: usize| neighbors[i - index - 1].unwrap();

                            // Recall that in the dense case, we process four values at a time:
                            //                  layer = [L, R, L, R, L, R, ...]
                            //                           |  |  |  |
                            //    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
                            //     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
                            //
                            // In the sparse case, we do something similar, but some of the four
                            // values may be omitted from the sparse vector.
                            // We match on `index % 4` to determine which of the four values are
                            // present in the sparse vector, and infer the rest are 1.
                            let (left, right) = match index % 4 {
                                0 => {
                                    let left = (*value, find_neighbor(index + 2));
                                    let right =
                                        (find_neighbor(index + 1), find_neighbor(index + 3));
                                    next_index_to_process = index + 4;
                                    (left, right)
                                }
                                1 => {
                                    let left = (get_default(index - 1), find_neighbor(index + 1));
                                    let right = (*value, find_neighbor(index + 2));
                                    next_index_to_process = index + 3;
                                    (left, right)
                                }
                                2 => {
                                    let left = (get_default(index - 2), *value);
                                    let right = (get_default(index - 1), find_neighbor(index + 1));
                                    next_index_to_process = index + 2;
                                    (left, right)
                                }
                                3 => {
                                    let left = (get_default(index - 3), get_default(index - 1));
                                    let right = (get_default(index - 2), *value);
                                    next_index_to_process = index + 1;
                                    (left, right)
                                }
                                _ => unreachable!("?_?"),
                            };

                            let m_left = left.1 - left.0;
                            let m_right = right.1 - right.0;

                            let left_eval_2 = left.1 + m_left;
                            let left_eval_3 = left_eval_2 + m_left;

                            let right_eval_2 = right.1 + m_right;
                            let right_eval_3 = right_eval_2 + m_right;

                            let (eq_eval_0, eq_eval_2, eq_eval_3) = eq_evals[index / 4];

                            sum.0 += eq_eval_0.mul_0_optimized(inner_func_delta(left.0, right.0));
                            sum.1 += eq_eval_2
                                .mul_0_optimized(inner_func_delta(left_eval_2, right_eval_2));
                            sum.2 += eq_eval_3
                                .mul_0_optimized(inner_func_delta(left_eval_3, right_eval_3));
                        }

                        total_eval.0 += *coeff * sum.0;
                        total_eval.1 += *coeff * sum.1;
                        total_eval.2 += *coeff * sum.2;
                    }
                    // If dense, we just compute
                    //     Σ coeff[batch_index] * (Σ eq_evals[i] * left[i] * right[i])
                    // directly in `self.compute_cubic`, without using `eq_eval_sums`.
                    DynamicDensityRationalSumLayer::Dense(dense_layer) => {
                        // Computes:
                        //     coeff[batch_index] * (Σ eq_evals[i] * left[i] * right[i])
                        // for the evaluation points {0, 2, 3}
                        let evals = eq_evals
                            .iter()
                            .zip(dense_layer.chunks_exact(4))
                            .map(|(eq_evals, chunk)| {
                                let left = (chunk[0], chunk[2]);
                                let right = (chunk[1], chunk[3]);

                                let m_left = left.1 - left.0;
                                let m_right = right.1 - right.0;

                                let left_eval_2 = left.1 + m_left;
                                let left_eval_3 = left_eval_2 + m_left;

                                let right_eval_2 = right.1 + m_right;
                                let right_eval_3 = right_eval_2 + m_right;

                                (
                                    eq_evals.0 * inner_func(left.0, right.0),
                                    eq_evals.1 * inner_func(left_eval_2, right_eval_2),
                                    eq_evals.2 * inner_func(left_eval_3, right_eval_3),
                                )
                            })
                            .fold(
                                (F::zero(), F::zero(), F::zero()),
                                |(sum_0, sum_2, sum_3), (a, b, c)| {
                                    (sum_0 + a, sum_2 + b, sum_3 + c)
                                },
                            );
                        total_eval.0 += *coeff * evals.0;
                        total_eval.1 += *coeff * evals.1;
                        total_eval.2 += *coeff * evals.2;
                    }
                });
                total_eval
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
            );

        let cubic_evals = [
            total_evals.0,
            previous_round_claim - total_evals.0,
            total_evals.1,
            total_evals.2,
        ];
        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self) -> (Vec<Pair<F>>, Vec<Pair<F>>) {
        assert_eq!(self.layer_len, 2);
        self.layers
            .iter()
            .enumerate()
            .map(|(batch_index, layer)| match layer {
                DynamicDensityRationalSumLayer::Sparse(layer) => {
                    let subtable_index = Self::memory_to_subtable_index(batch_index);

                    let default = &self.preprocessing[subtable_index];
                    let get_default = |index| Pair {
                        p: F::zero(),
                        q: default[index],
                    };

                    match layer.len() {
                        0 => (get_default(0), get_default(1)), // Neither left nor right claim is present, so they must both be 1
                        1 => {
                            if layer[0].0.is_zero() {
                                // Only left claim is present, so right claim must be 1
                                (layer[0].1, get_default(1))
                            } else {
                                // Only right claim is present, so left claim must be 1
                                (get_default(0), layer[0].1)
                            }
                        }
                        2 => (layer[0].1, layer[1].1), // Both left and right claim are present
                        _ => panic!("Sparse layer length > 2"),
                    }
                }
                DynamicDensityRationalSumLayer::Dense(layer) => (layer[0], layer[1]),
            })
            .unzip()
    }
}

pub fn sparse_preprocess<F: JoltField>(leaves: Vec<Vec<F>>) -> Vec<Vec<Vec<F>>> {
    let num_layers = leaves[0].len().log_2();
    let mut layers: Vec<Vec<Vec<F>>> = Vec::with_capacity(num_layers);
    layers.push(leaves);

    // One more layer than usually present - so that we can always take a preprocessing
    for i in 0..num_layers {
        let previous_layers = &layers[i];
        let len = previous_layers[0].len() / 2;
        // TODO(moodlezoup): parallelize over chunks instead of over batch
        let new_layers = previous_layers
            .par_iter()
            .map(|previous_layer| {
                (0..len)
                    .map(|i| previous_layer[2 * i] * previous_layer[2 * i + 1])
                    .collect::<Vec<_>>()
            })
            .collect();
        layers.push(new_layers);
    }
    layers
}

/// A batched grand product circuit.
/// Note that the circuit roots are not included in `self.layers` but included
/// in preprocessing
///        o
///      /   \
///     o     o  <- layers[layers.len() - 1]
///    / \   / \
///   o   o o   o  <- layers[layers.len() - 2]
///       ...
pub struct BatchedSparseRationalSum<F: JoltField, const C: usize> {
    layers: Vec<BatchedSparseRationalSumLayer<F, C>>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, const C: usize> BatchedRationalSum<F, PCS>
    for BatchedSparseRationalSum<F, C>
{
    // (indices, p, q). p corresponds to indices (sparse), q is dense
    type Leaves = (Vec<Vec<usize>>, Vec<Vec<F>>, Vec<Vec<F>>);

    #[tracing::instrument(skip_all, name = "BatchedSparseRationalSum::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let (indices_p, values_p, leaves_q) = leaves;
        let num_subtables = leaves_q.len();
        let leaves_len = leaves_q[0].len();
        let num_layers = leaves_q[0].len().log_2();
        let mut preprocessings = sparse_preprocess(leaves_q);

        let mut layers: Vec<BatchedSparseRationalSumLayer<F, C>> = Vec::with_capacity(num_layers);
        layers.push(BatchedSparseRationalSumLayer {
            layer_len: leaves_len,
            layers: (0..num_subtables * C)
                .into_par_iter()
                .map(|batch_index| {
                    let subtable_index = batch_index / C;
                    let dimension_index = batch_index % C;
                    DynamicDensityRationalSumLayer::Sparse(
                        zip(&indices_p[dimension_index], &values_p[dimension_index])
                            .map(|(index, p)| {
                                (
                                    *index,
                                    Pair {
                                        p: *p,
                                        q: preprocessings[0][subtable_index][*index],
                                    },
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
            preprocessing: take(&mut preprocessings[0]),
            preprocessing_next: vec![],
        });

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers.layer_len / 2;
            let new_layers = previous_layers
                .layers
                .par_iter()
                .enumerate()
                .map(|(memory_index, previous_layer)| {
                    previous_layer.layer_output::<C>(
                        len,
                        memory_index,
                        &previous_layers.preprocessing,
                        &preprocessings[i + 1],
                    )
                })
                .collect();
            layers.push(BatchedSparseRationalSumLayer {
                layer_len: len,
                layers: new_layers,
                preprocessing: take(&mut preprocessings[i + 1]),
                preprocessing_next: vec![],
            });
        }

        Self { layers }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claims(&self) -> Vec<Pair<F>> {
        let last_layers = &self.layers.last().unwrap();
        let (left_claims, right_claims) = last_layers.final_claims();
        left_claims
            .iter()
            .zip(right_claims.iter())
            .map(|(left_claim, right_claim)| Pair::rational_add(*left_claim, *right_claim))
            .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedRationalSumLayer<F>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedRationalSumLayer<F>)
            .rev()
    }
}

#[cfg(test)]
mod rational_sum_tests {
    use super::*;
    use crate::poly::commitment::zeromorph::Zeromorph;
    use ark_bn254::{Bn254, Fr};
    use ark_std::{test_rng, One};
    use rand_core::RngCore;

    #[test]
    fn dense_prove_verify() {
        const LAYER_SIZE: usize = 1 << 8;
        const BATCH_SIZE: usize = 4;
        const C: usize = 4;
        let mut rng = test_rng();
        let leaves_p: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect()
        })
        .take(BATCH_SIZE * C)
        .collect();
        let leaves_q: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect()
        })
        .take(BATCH_SIZE)
        .collect();

        let expected_claims: Vec<Pair<Fr>> = leaves_p
            .iter()
            .enumerate()
            .map(|(i, layer_p)| {
                let layer_q = &leaves_q[i / C];
                zip(layer_p.iter(), layer_q.iter())
                    .map(|(p, q)| Pair { p: *p, q: *q })
                    .reduce(Pair::rational_add)
                    .unwrap()
            })
            .collect();

        let mut batched_circuit = <BatchedDenseRationalSum<Fr, C> as BatchedRationalSum<
            Fr,
            Zeromorph<Bn254>,
        >>::construct((leaves_p, leaves_q));
        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

        // I love the rust type system
        let claims =
            <BatchedDenseRationalSum<Fr, C> as BatchedRationalSum<Fr, Zeromorph<Bn254>>>::claims(
                &batched_circuit,
            );
        assert_eq!(expected_claims, claims);

        let (proof, r_prover) = <BatchedDenseRationalSum<Fr, C> as BatchedRationalSum<
            Fr,
            Zeromorph<Bn254>,
        >>::prove_rational_sum(
            &mut batched_circuit, &mut transcript, None
        );

        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
        let (_, r_verifier) = BatchedDenseRationalSum::<Fr, C>::verify_rational_sum(
            &proof,
            &claims,
            &mut transcript,
            None,
        );
        assert_eq!(r_prover, r_verifier);
    }

    #[test]
    fn sparse_prove_verify() {
        const LAYER_SIZE: usize = 1 << 8;
        const BATCH_SIZE: usize = 4;
        const C: usize = 4;
        let mut rng = test_rng();

        let baseline: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect()
        })
        .take(BATCH_SIZE)
        .collect();

        let mut indices_p = vec![vec![]; C];
        let mut values_p = vec![vec![]; C];

        let leaves_p: Vec<Vec<Fr>> = (0..C)
            .map(|layer_idx| {
                (0..LAYER_SIZE)
                    .map(|index| {
                        if rng.next_u32() % 4 == 0 {
                            let mut p = Fr::random(&mut rng);
                            while p == Fr::zero() {
                                p = Fr::random(&mut rng);
                            }
                            indices_p[layer_idx].push(index);
                            values_p[layer_idx].push(p);
                            p
                        } else {
                            Fr::zero()
                        }
                    })
                    .collect()
            })
            .collect();

        let expected_claims: Vec<Pair<Fr>> = (0..BATCH_SIZE * C)
            .map(|i| {
                let layer_p = &leaves_p[i % C];
                let layer_q = &baseline[i / C];
                zip(layer_p.iter(), layer_q.iter())
                    .map(|(p, q)| Pair { p: *p, q: *q })
                    .reduce(Pair::rational_add)
                    .unwrap()
            })
            .collect();

        let mut batched_circuit = <BatchedSparseRationalSum<Fr, C> as BatchedRationalSum<
            Fr,
            Zeromorph<Bn254>,
        >>::construct((indices_p, values_p, baseline.clone()));
        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

        // I love the rust type system
        let claims = <BatchedSparseRationalSum<Fr, C> as BatchedRationalSum<
            Fr,
            Zeromorph<Bn254>,
        >>::claims(&batched_circuit);
        assert_eq!(expected_claims, claims);

        let (proof, r_prover) = <BatchedSparseRationalSum<Fr, C> as BatchedRationalSum<
            Fr,
            Zeromorph<Bn254>,
        >>::prove_rational_sum(
            &mut batched_circuit, &mut transcript, None
        );

        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
        let (_, r_verifier) = BatchedSparseRationalSum::<Fr, C>::verify_rational_sum(
            &proof,
            &claims,
            &mut transcript,
            None,
        );
        assert_eq!(r_prover, r_verifier);
    }

    // #[test]
    // fn dense_sparse_bind_parity() {
    //     const LAYER_SIZE: usize = 1 << 4;
    //     const BATCH_SIZE: usize = 1;
    //     const C: usize = 2;
    //     let mut rng = test_rng();

    //     let baseline: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
    //         std::iter::repeat_with(|| Fr::random(&mut rng))
    //         .take(LAYER_SIZE)
    //         .collect()
    //     })
    //     .take(BATCH_SIZE)
    //     .collect();

    //     let preprocessing = sparse_preprocess(baseline.clone());

    //     let leaves_p: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
    //         std::iter::repeat_with(|| {
    //             if rng.next_u32() % 4 == 0 {
    //                 let mut p = Fr::random(&mut rng);
    //                 while p == Fr::zero() {
    //                     p = Fr::random(&mut rng);
    //                 }
    //                 p
    //             } else {
    //                 Fr::zero()
    //             }
    //         })
    //         .take(LAYER_SIZE)
    //         .collect()
    //     })
    //     .take(BATCH_SIZE * C)
    //     .collect();
    //     let mut batched_dense_layer = BatchedDenseRationalSumLayer::new(leaves_p.clone(), baseline.clone());

    //     let sparse_layers: Vec<DynamicDensityRationalSumLayer<Fr>> = leaves_p
    //         .iter()
    //         .enumerate()
    //         .map(|(memory_idx, dense_layer)| {
    //             let subtable_idx = memory_idx / C;
    //             let mut sparse_layer = vec![];
    //             for (i, val) in dense_layer.iter().enumerate() {
    //                 if *val != Fr::zero() {
    //                     sparse_layer.push((i, Pair{p : *val, q: baseline[subtable_idx][i]} ));
    //                 }
    //             }
    //             DynamicDensityRationalSumLayer::Sparse(sparse_layer)
    //         })
    //         .collect();
    //     let mut batched_sparse_layer = BatchedSparseRationalSumLayer {
    //         layer_len: LAYER_SIZE,
    //         layers: sparse_layers,
    //         preprocessing: preprocessing[0].clone(),
    //         preprocessing_next: vec![],
    //     };

    //     let condense = |sparse_layers: BatchedSparseRationalSumLayer<Fr, 1>| {
    //         sparse_layers
    //             .layers
    //             .iter()
    //             .enumerate()
    //             .map(|(idx, layer)| match layer {
    //                 DynamicDensityRationalSumLayer::Sparse(sparse_layer) => {
    //                     let mut densified = sparse_layers.preprocessing[idx].iter()
    //                     .map(|x| Pair{p: Fr::zero(), q: *x})
    //                     .collect::<Vec<_>>();
    //                     for (index, value) in sparse_layer {
    //                         densified[*index] = *value;
    //                     }
    //                     densified
    //                 }
    //                 DynamicDensityRationalSumLayer::Dense(dense_layer) => dense_layer.clone(),
    //             })
    //             .collect::<Vec<_>>()
    //     };

    //     assert_eq!(
    //         batched_dense_layer.layer_len,
    //         batched_sparse_layer.layer_len
    //     );
    //     let len = batched_dense_layer.layer_len;
    //     for (dense, sparse) in batched_dense_layer
    //         .layers
    //         .iter()
    //         .zip(condense(batched_sparse_layer.clone()).iter())
    //     {
    //         assert_eq!(dense[..len], sparse[..len]);
    //     }

    //     for _ in 0..LAYER_SIZE.log_2() - 1 {
    //         let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
    //             .take(4)
    //             .collect::<Vec<_>>();
    //         let mut eq_poly_dense = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_eq));
    //         let mut eq_poly_sparse = eq_poly_dense.clone();

    //         let r = Fr::random(&mut rng);
    //         batched_dense_layer.bind(&mut eq_poly_dense, &r);
    //         batched_sparse_layer.bind(&mut eq_poly_sparse, &r);

    //         assert_eq!(eq_poly_dense, eq_poly_sparse);
    //         assert_eq!(
    //             batched_dense_layer.layer_len,
    //             batched_sparse_layer.layer_len
    //         );
    //         let len = batched_dense_layer.layer_len;
    //         for (dense, sparse) in batched_dense_layer
    //             .layers
    //             .iter()
    //             .zip(condense(batched_sparse_layer.clone()).iter())
    //         {
    //             assert_eq!(dense[..len], sparse[..len]);
    //         }
    //     }
    // }

    // #[test]
    // fn dense_sparse_compute_cubic_parity() {
    //     const LAYER_SIZE: usize = 1 << 10;
    //     const BATCH_SIZE: usize = 4;
    //     let mut rng = test_rng();

    //     let coeffs: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
    //         .take(BATCH_SIZE)
    //         .collect();

    //     let baseline: Vec<Vec<Pair<Fr>>> = std::iter::repeat_with(|| {
    //         std::iter::repeat_with(|| Pair {
    //             p: Fr::zero(),
    //             q: Fr::random(&mut rng),
    //         })
    //         .take(LAYER_SIZE)
    //         .collect()
    //     })
    //     .take(BATCH_SIZE)
    //     .collect();

    //     let preprocessing = sparse_preprocess(baseline.clone());

    //     let dense_layers: Vec<DenseRationalSumLayer<Fr>> = (0..BATCH_SIZE)
    //         .map(|i| {
    //             (0..LAYER_SIZE)
    //                 .map(|j| {
    //                     if rng.next_u32() % 4 == 0 {
    //                         let mut p = Fr::random(&mut rng);
    //                         while p == Fr::zero() {
    //                             p = Fr::random(&mut rng);
    //                         }
    //                         Pair {
    //                             p,
    //                             q: baseline[i][j].q,
    //                         }
    //                     } else {
    //                         baseline[i][j]
    //                     }
    //                 })
    //                 .collect()
    //         })
    //         .collect();
    //     let mut batched_dense_layer = BatchedDenseRationalSumLayer::new(dense_layers.clone());

    //     let sparse_layers: Vec<DynamicDensityRationalSumLayer<Fr>> = dense_layers
    //         .iter()
    //         .map(|dense_layer| {
    //             let mut sparse_layer = vec![];
    //             for (i, val) in dense_layer.iter().enumerate() {
    //                 if val.p != Fr::zero() {
    //                     sparse_layer.push((i, *val));
    //                 }
    //             }
    //             DynamicDensityRationalSumLayer::Sparse(sparse_layer)
    //         })
    //         .collect();
    //     let mut batched_sparse_layer = BatchedSparseRationalSumLayer::<Fr, 1> {
    //         layer_len: LAYER_SIZE,
    //         layers: sparse_layers,
    //         preprocessing: preprocessing[0].clone(),
    //         preprocessing_next: vec![],
    //     };

    //     let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
    //         .take(LAYER_SIZE.log_2() - 1)
    //         .collect::<Vec<_>>();
    //     let eq_poly = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_eq));
    //     let claim = Fr::random(&mut rng);
    //     let lambda = Fr::random(&mut rng);
    //     let dense_evals = batched_dense_layer.compute_cubic(&coeffs, &eq_poly, claim, &lambda);
    //     let sparse_evals = batched_sparse_layer.compute_cubic(&coeffs, &eq_poly, claim, &lambda);
    //     assert_eq!(dense_evals, sparse_evals);
    // }
}
