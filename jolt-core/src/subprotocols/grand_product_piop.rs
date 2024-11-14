// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements useful functions for the product check protocol.

use crate::utils::transcript::AppendToTranscript;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::{BatchType, CommitmentScheme},
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, transcript::ProofTranscript},
};
use ark_std::{end_timer, start_timer};
use rayon::prelude::*;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

// Given a vector of field elements {v_i}, compute the vector {v_i^(-1)}
pub fn batch_inversion<F: JoltField>(v: &mut [F]) {
    batch_inversion_and_mul(v, &F::one());
}

// Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}
pub fn batch_inversion_and_mul<F: JoltField>(v: &mut [F], coeff: &F) {
    // Divide the vector v evenly between all available cores
    let min_elements_per_thread = 1;
    let num_cpus_available = rayon::current_num_threads();
    let num_elems = v.len();
    let num_elem_per_thread =
        std::cmp::max(num_elems / num_cpus_available, min_elements_per_thread);

    // Batch invert in parallel, without copying the vector
    v.par_chunks_mut(num_elem_per_thread).for_each(|mut chunk| {
        serial_batch_inversion_and_mul(&mut chunk, coeff);
    });
}

/// Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}.
/// This method is explicitly single-threaded.
fn serial_batch_inversion_and_mul<F: JoltField>(v: &mut [F], coeff: &F) {
    // Montgomery’s Trick and Fast Implementation of Masked AES
    // Genelle, Prouff and Quisquater
    // Section 3.2
    // but with an optimization to multiply every element in the returned vector by
    // coeff

    // First pass: compute [a, ab, abc, ...]
    let mut prod = Vec::with_capacity(v.len());
    let mut tmp = F::one();
    for f in v.iter().filter(|f| !f.is_zero()) {
        tmp.mul_assign(*f);
        prod.push(tmp);
    }

    // Invert `tmp`.
    tmp = tmp.inverse().unwrap(); // Guaranteed to be nonzero.

    // Multiply product by coeff, so all inverses will be scaled by coeff
    tmp *= *coeff;

    // Second pass: iterate backwards to compute inverses
    for (f, s) in v.iter_mut()
        // Backwards
        .rev()
        // Ignore normalized elements
        .filter(|f| !f.is_zero())
        // Backwards, skip last element, fill in one for last term.
        .zip(prod.into_iter().rev().skip(1).chain(Some(F::one())))
    {
        // tmp := tmp * f; f := tmp * s = 1/f
        let new_tmp = tmp * *f;
        *f = tmp * &s;
        tmp = new_tmp;
    }
}

/// Compute multilinear fractional polynomial s.t. frac(x) = f1(x) * ... * fk(x)
/// / (g1(x) * ... * gk(x)) for all x \in {0,1}^n
///
/// The caller needs to sanity-check that the number of polynomials and
/// variables match in fxs and gxs; and gi(x) has no zero entries.
pub fn compute_frac_poly<F: JoltField>(
    fx: &DensePolynomial<F>,
    gx: &DensePolynomial<F>,
) -> DensePolynomial<F> {
    let start = start_timer!(|| "compute frac(x)");

    let mut f_evals = fx.Z.clone();

    let mut g_evals = gx.Z.clone();
    batch_inversion(&mut g_evals[..]);

    for (f_eval, g_eval) in f_evals.iter_mut().zip(g_evals.iter()) {
        if *g_eval == F::zero() {
            panic!("gxs has zero entries in the boolean hypercube");
        }
        *f_eval *= *g_eval;
    }

    end_timer!(start);
    DensePolynomial::new(f_evals)
}

/// Project a little endian binary vector into an integer.
#[inline]
fn project(input: &[bool]) -> u64 {
    let mut res = 0;
    for &e in input.iter().rev() {
        res <<= 1;
        res += e as u64;
    }
    res
}

/// Decompose an integer into a binary vector in little endian.
fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(i & 1 == 1);
        i >>= 1;
    }
    res
}

// Input index
// - `i := (i_0, ...i_{n-1})`,
// - `num_vars := n`
// return three elements:
// - `x0 := (i_1, ..., i_{n-1}, 0)`
// - `x1 := (i_1, ..., i_{n-1}, 1)`
// - `sign := i_0`
#[inline]
fn get_index(i: usize, num_vars: usize) -> (usize, usize, bool) {
    let bit_sequence = bit_decompose(i as u64, num_vars);

    // the last bit comes first here because of LE encoding
    let x0 = project(&[[false].as_ref(), bit_sequence[..num_vars - 1].as_ref()].concat()) as usize;
    let x1 = project(&[[true].as_ref(), bit_sequence[..num_vars - 1].as_ref()].concat()) as usize;

    (x0, x1, bit_sequence[num_vars - 1])
}

/// Compute the product polynomial `prod(x)` such that
/// `prod(x) = [(1-x1)*frac(x2, ..., xn, 0) + x1*prod(x2, ..., xn, 0)] *
/// [(1-x1)*frac(x2, ..., xn, 1) + x1*prod(x2, ..., xn, 1)]` on the boolean
/// hypercube {0,1}^n
///
/// The caller needs to check num_vars matches in f and g
/// Cost: linear in N.
pub fn compute_product_poly<F: JoltField>(frac_poly: &DensePolynomial<F>) -> DensePolynomial<F> {
    let start = start_timer!(|| "compute evaluations of prod polynomial");
    let num_vars = frac_poly.get_num_vars();
    let frac_evals = &frac_poly.Z;

    // ===================================
    // prod(x)
    // ===================================
    //
    // `prod(x)` can be computed via recursing the following formula for 2^n-1
    // times
    //
    // `prod(x_1, ..., x_n) :=
    //      [(1-x1)*frac(x2, ..., xn, 0) + x1*prod(x2, ..., xn, 0)] *
    //      [(1-x1)*frac(x2, ..., xn, 1) + x1*prod(x2, ..., xn, 1)]`
    //
    // At any given step, the right hand side of the equation
    // is available via either frac_x or the current view of prod_x
    let mut prod_x_evals = vec![];
    for x in 0..(1 << num_vars) - 1 {
        // sign will decide if the evaluation should be looked up from frac_x or
        // prod_x; x_zero_index is the index for the evaluation (x_2, ..., x_n,
        // 0); x_one_index is the index for the evaluation (x_2, ..., x_n, 1);
        let (x_zero_index, x_one_index, sign) = get_index(x, num_vars);
        if !sign {
            prod_x_evals.push(frac_evals[x_zero_index] * frac_evals[x_one_index]);
        } else {
            // sanity check: if we are trying to look up from the prod_x_evals table,
            // then the target index must already exist
            if x_zero_index >= prod_x_evals.len() || x_one_index >= prod_x_evals.len() {
                panic!("Unreachable");
            }
            prod_x_evals.push(prod_x_evals[x_zero_index] * prod_x_evals[x_one_index]);
        }
    }

    // prod(1, 1, ..., 1) := 0
    prod_x_evals.push(F::zero());
    end_timer!(start);

    DensePolynomial::new(prod_x_evals)
}

/// generate the zerocheck proof for the virtual polynomial
///    prod(x) - p1(x) * p2(x) + alpha * [frac(x) * g1(x) * ... * gk(x) - f1(x)
/// * ... * fk(x)] where p1(x) = (1-x1) * frac(x2, ..., xn, 0) + x1 * prod(x2,
///   ..., xn, 0), p2(x) = (1-x1) * frac(x2, ..., xn, 1) + x1 * prod(x2, ...,
///   xn, 1)
///
/// Returns proof.
///
/// Cost: O(N)
pub fn prove_zero_check_batched<F: JoltField>(
    mut fxs: Vec<DensePolynomial<F>>,
    mut gxs: Vec<DensePolynomial<F>>,
    mut frac_poly: Vec<DensePolynomial<F>>,
    mut prod_x: Vec<DensePolynomial<F>>,
    alpha: &F,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F>, Vec<F>, Vec<F>) {
    let start = start_timer!(|| "zerocheck in product check");
    let num_vars = frac_poly[0].get_num_vars();

    // compute p1(x) = (1-x1) * frac(x2, ..., xn, 0) + x1 * prod(x2, ..., xn, 0)
    // compute p2(x) = (1-x1) * frac(x2, ..., xn, 1) + x1 * prod(x2, ..., xn, 1)
    let (mut p1_polys, mut p2_polys): (Vec<_>, Vec<_>) = (&frac_poly, &prod_x)
        .into_par_iter()
        .map(|(frac_poly, prod_x)| {
            let mut p1_evals = vec![F::zero(); 1 << num_vars];
            let mut p2_evals = vec![F::zero(); 1 << num_vars];
            for x in 0..1 << num_vars {
                let (x0, x1, sign) = get_index(x, num_vars);
                if !sign {
                    p1_evals[x] = frac_poly.Z[x0];
                    p2_evals[x] = frac_poly.Z[x1];
                } else {
                    p1_evals[x] = prod_x.Z[x0];
                    p2_evals[x] = prod_x.Z[x1];
                }
            }
            let p1 = DensePolynomial::new(p1_evals);
            let p2 = DensePolynomial::new(p2_evals);
            (p1, p2)
        })
        .unzip();

    let fx_len = fxs.len();

    //   prod(x)
    // - p1(x) * p2(x)
    // + alpha * frac(x) * g1(x) * ... * gk(x)
    // - alpha * f1(x) * ... * fk(x)]
    fxs.append(&mut gxs);
    fxs.append(&mut p1_polys);
    fxs.append(&mut p2_polys);
    fxs.append(&mut frac_poly);
    fxs.append(&mut prod_x);

    let r_zerocheck = transcript.challenge_vector(num_vars);
    let eq_poly = DensePolynomial::new(EqPolynomial::evals(&r_zerocheck));
    fxs.push(eq_poly);

    let coeffs: Vec<F> = transcript.challenge_vector(fx_len);

    let proof = SumcheckInstanceProof::<F>::prove_arbitrary(
        &F::zero(),
        num_vars,
        &mut fxs,
        |evals| {
            let eq_eval = evals[evals.len() - 1];
            coeffs
                .par_iter()
                .enumerate()
                .map(|(i, coeff)| {
                    let fx_eval = evals[i];
                    let gx_eval = evals[fx_len + i];
                    let p1_eval = evals[2 * fx_len + i];
                    let p2_eval = evals[3 * fx_len + i];
                    let frac_eval = evals[4 * fx_len + i];
                    let prod_eval = evals[5 * fx_len + i];
                    *coeff
                        * (prod_eval - p1_eval * p2_eval + *alpha * (frac_eval * gx_eval - fx_eval))
                })
                .sum::<F>()
                * eq_eval
        },
        3,
        transcript,
    );

    end_timer!(start);

    proof
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct GrandProductProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub zero_check_proof: SumcheckInstanceProof<F>,
    pub frac_comm: Vec<PCS::Commitment>,
    pub prod_x_comm: Vec<PCS::Commitment>,
}

pub struct GrandProductSubClaim<F: JoltField> {
    pub zero_check_point: Vec<F>,
    pub zero_check_claim: F,
    pub prod_final_query: Vec<F>,
    pub alpha: F,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> GrandProductProof<F, PCS> {
    pub fn prove(
        setup: &PCS::Setup,
        fxs: Vec<DensePolynomial<F>>,
        gxs: Vec<DensePolynomial<F>>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>, Vec<F>) {
        let start = start_timer!(|| "prod_check prove");

        if fxs.is_empty() {
            panic!("fxs is empty");
        }
        if fxs.len() != gxs.len() {
            panic!("fxs and gxs have different number of polynomials");
        }

        // compute the fractional polynomial frac_p s.t.
        // frac_p(x) = f1(x) * ... * fk(x) / (g1(x) * ... * gk(x))
        let (frac_polys, prod_x_polys): (Vec<_>, Vec<_>) = fxs
            .par_iter()
            .zip(gxs.par_iter())
            .map(|(fx, gx)| {
                let frac_poly = compute_frac_poly(&fx, &gx);
                let prod_x = compute_product_poly(&frac_poly);
                (frac_poly, prod_x)
            })
            .unzip();

        let claims = prod_x_polys
            .iter()
            .map(|prod_x| prod_x.Z[prod_x.Z.len() - 1])
            .collect::<Vec<_>>();

        // generate challenge
        let frac_comm = PCS::batch_commit_polys(&frac_polys, &setup, BatchType::Big);
        let prod_x_comm = PCS::batch_commit_polys(&prod_x_polys, &setup, BatchType::Big);
        for com in &frac_comm {
            com.append_to_transcript(transcript);
        }
        for com in &prod_x_comm {
            com.append_to_transcript(transcript);
        }
        let alpha = transcript.challenge_scalar();

        // build the zero-check proof
        let (zero_check_proof, zero_check_point, zero_check_expected_evaluations) =
            prove_zero_check_batched(fxs, gxs, frac_polys, prod_x_polys, &alpha, transcript);
        end_timer!(start);
        return (
            Self {
                zero_check_proof,
                prod_x_comm,
                frac_comm,
            },
            claims,
            zero_check_point,
            zero_check_expected_evaluations,
        );
    }

    pub fn verify(
        &self,
        transcript: &mut ProofTranscript,
    ) -> Result<GrandProductSubClaim<F>, ProofVerifyError> {
        let start = start_timer!(|| "prod_check verify");

        // update transcript and generate challenge
        for com in &self.frac_comm {
            com.append_to_transcript(transcript);
        }
        for com in &self.prod_x_comm {
            com.append_to_transcript(transcript);
        }
        let alpha = transcript.challenge_scalar();

        let num_rounds = self.zero_check_proof.compressed_polys.len();
        let _r_zerocheck : Vec<F> = transcript.challenge_vector(num_rounds);

        // invoke the zero check on the iop_proof
        // the virtual poly info for Q(x)
        let (zero_check_claim, zero_check_point) =
            self.zero_check_proof
                .verify(F::zero(), num_rounds, 3, transcript)?;

        // the final query is on prod_x
        let mut final_query = vec![F::one(); num_rounds];
        final_query[num_rounds - 1] = F::zero();

        end_timer!(start);

        Ok(GrandProductSubClaim {
            zero_check_point,
            zero_check_claim,
            prod_final_query: final_query,
            alpha,
        })
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::field::JoltField;
//     use crate::{
//         poly::{
//             commitment::{commitment_scheme::CommitmentScheme, mock::MockCommitScheme},
//             dense_mlpoly::DensePolynomial,
//         },
//         utils::errors::ProofVerifyError,
//     };
//     use ark_bn254::Fr;
//     use rand_core::SeedableRng;

//     // fs and gs are guaranteed to have the same product
//     // fs and hs doesn't have the same product
//     fn test_product_check_helper<F, PCS>(
//         fs: Vec<DensePolynomial<F>>,
//         gs: Vec<DensePolynomial<F>>,
//         hs: Vec<DensePolynomial<F>>,
//         pcs_param: &PCS::Setup,
//     ) -> Result<(), ProofVerifyError>
//     where
//         F: JoltField,
//         PCS: CommitmentScheme<Field = F>,
//     {
//         let mut transcript = ProofTranscript::new(b"testing");

//         let nv = fs[0].get_num_vars();
//         let num_polys = fs.len();

//         let (proof, claim, prod_x, frac_poly) =
//             GrandProductProof::<F, PCS>::prove(pcs_param, fs.clone(), gs, &mut transcript);
//         assert_eq!(claim, F::one());

//         let mut transcript = ProofTranscript::new(b"testing");

//         // what's aux_info for?
//         let prod_subclaim = proof.verify(nv, num_polys, &mut transcript)?;
//         assert_eq!(
//             prod_x.unwrap().evaluate(&prod_subclaim.prod_final_query),
//             F::one(),
//             "different product"
//         );

//         // bad path
//         let mut transcript = ProofTranscript::new(b"testing");

//         let (bad_proof, final_claim_2, prod_x_bad, frac_poly) =
//             GrandProductProof::<F, PCS>::prove(pcs_param, fs, hs, &mut transcript);

//         let mut transcript = ProofTranscript::new(b"testing");
//         let bad_subclaim = bad_proof.verify(nv, num_polys, &mut transcript)?;
//         let prod_x_bad = prod_x_bad.unwrap();
//         assert_ne!(
//             prod_x_bad.evaluate(&bad_subclaim.prod_final_query),
//             F::one(),
//             "can't detect wrong proof"
//         );
//         assert_eq!(
//             prod_x_bad.evaluate(&bad_subclaim.prod_final_query),
//             final_claim_2,
//         );
//         Ok(())
//     }

//     fn test_product_check(nv: usize) -> Result<(), ProofVerifyError> {
//         let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(69420 as u64);

//         let f1: DensePolynomial<Fr> = DensePolynomial::random(nv, &mut rng);
//         let mut g1 = f1.clone();
//         g1.Z.reverse();
//         let f2: DensePolynomial<Fr> = DensePolynomial::random(nv, &mut rng);
//         let mut g2 = f2.clone();
//         g2.Z.reverse();
//         let fs = vec![f1, f2];
//         let gs = vec![g1, g2];
//         let mut hs = vec![];
//         for _ in 0..fs.len() {
//             hs.push(DensePolynomial::random(fs[0].get_num_vars(), &mut rng));
//         }

//         let srs = MockCommitScheme::<Fr>::setup(&[]);

//         test_product_check_helper::<Fr, MockCommitScheme<Fr>>(fs, gs, hs, &srs)?;

//         Ok(())
//     }

//     #[test]
//     fn test_trivial_polynomial() -> Result<(), ProofVerifyError> {
//         test_product_check(1)
//     }
//     #[test]
//     fn test_normal_polynomial() -> Result<(), ProofVerifyError> {
//         test_product_check(10)
//     }
// }
