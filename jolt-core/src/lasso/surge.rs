use crate::field::{JoltField, OptimizedMul};
use crate::lasso::memory_checking::NoPreprocessing;
use crate::poly;
use crate::poly::commitment::commitment_scheme::BatchType;
use crate::subprotocols::rational_sum::{
    BatchedDenseRationalSum, BatchedRationalSum, BatchedRationalSumProof, BatchedSparseRationalSum,
    Pair,
};
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::AppendToTranscript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::marker::{PhantomData, Sync};
use std::mem::take;

use crate::{
    jolt::instruction::JoltInstruction,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, hyrax::matrix_dimensions},
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{StructuredCommitment, StructuredOpeningProof},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math, transcript::ProofTranscript},
};

// These are polynomials computed before beta and gamma are selected
pub struct SurgePolysPrimary<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    _marker: PhantomData<PCS>,
    pub dim: Vec<DensePolynomial<F>>,     // Size C
    pub E_polys: Vec<DensePolynomial<F>>, // Size NUM_MEMORIES
    pub m: Vec<DensePolynomial<F>>,       // Size C

    // Sparse representation of m
    pub m_indices: Vec<Vec<usize>>,
    pub m_values: Vec<Vec<F>>,
}

pub struct SurgePolysLogup<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    _marker: PhantomData<PCS>,
    pub f: Vec<DensePolynomial<F>>, // Size NUM_MEMORIES
    pub g: Vec<DensePolynomial<F>>, // Size NUM_MEMORIES
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitmentPrimary<CS: CommitmentScheme> {
    pub dim_commitment: Vec<CS::Commitment>, // Size C
    pub E_commitment: Vec<CS::Commitment>,   // Size NUM_MEMORIES
    pub m_commitment: Vec<CS::Commitment>,   // Size C
}

impl<F, PCS> StructuredCommitment<PCS> for SurgePolysPrimary<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Commitment = SurgeCommitmentPrimary<PCS>;

    #[tracing::instrument(skip_all, name = "SurgePolysPrimary::commit")]
    fn commit(&self, generators: &PCS::Setup) -> Self::Commitment {
        let dim_commitment =
            PCS::batch_commit_polys(&self.dim, generators, BatchType::SurgeReadWrite);
        let E_commitment =
            PCS::batch_commit_polys(&self.E_polys, generators, BatchType::SurgeReadWrite);
        let m_commitment = PCS::batch_commit_polys(&self.m, generators, BatchType::SurgeInitFinal);

        Self::Commitment {
            dim_commitment,
            E_commitment,
            m_commitment,
        }
    }
}

impl<F, PCS> AppendToTranscript for SurgeCommitmentPrimary<PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        [&self.dim_commitment, &self.E_commitment, &self.m_commitment]
            .iter()
            .for_each(|commitments| {
                commitments.iter().for_each(|commitment| {
                    commitment.append_to_transcript(label, transcript);
                })
            });
    }
}

type PrimarySumcheckOpenings<F> = Vec<F>;

impl<F, PCS> StructuredOpeningProof<F, PCS, SurgePolysPrimary<F, PCS>>
    for PrimarySumcheckOpenings<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = PCS::BatchedProof;
    type Preprocessing = NoPreprocessing;

    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::open")]
    fn open(polynomials: &SurgePolysPrimary<F, PCS>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        polynomials
            .E_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect()
    }

    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::prove_openings")]
    fn prove_openings(
        generators: &PCS::Setup,
        polynomials: &SurgePolysPrimary<F, PCS>,
        opening_point: &[F],
        E_poly_openings: &Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        PCS::batch_prove(
            generators,
            &polynomials.E_polys.iter().collect::<Vec<_>>(),
            opening_point,
            E_poly_openings,
            BatchType::SurgeReadWrite,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        generators: &PCS::Setup,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitmentPrimary<PCS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        PCS::batch_verify(
            opening_proof,
            generators,
            opening_point,
            self,
            &commitment.E_commitment.iter().collect::<Vec<_>>(),
            transcript,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgePrimarySumcheck<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    sumcheck_proof: SumcheckInstanceProof<F>,
    num_rounds: usize,
    claimed_evaluation: F,
    openings: PrimarySumcheckOpenings<F>,
    opening_proof: PCS::BatchedProof,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeLogupFPolyOpenings<F, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    Instruction: JoltInstruction + Default,
{
    _marker: PhantomData<Instruction>,
    m_openings: Vec<F>,
}

impl<F, Instruction, const C: usize, const M: usize> SurgeLogupFPolyOpenings<F, Instruction, C, M>
where
    F: JoltField,
    Instruction: JoltInstruction + Default,
{
    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::compute_sid_t")]
    fn compute_sid_t(&self, opening_point: &[F]) -> (F, Vec<F>) {
        rayon::join(
            || IdentityPolynomial::new(opening_point.len()).evaluate(opening_point),
            || {
                Instruction::default()
                    .subtables(C, M)
                    .par_iter()
                    .map(|(subtable, _)| subtable.evaluate_mle(opening_point))
                    .collect()
            },
        )
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize>
    StructuredOpeningProof<F, PCS, SurgePolysPrimary<F, PCS>>
    for SurgeLogupFPolyOpenings<F, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    type Proof = PCS::BatchedProof;
    type Preprocessing = SurgePreprocessing<F, Instruction, C, M>;

    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::open")]
    fn open(polynomials_primary: &SurgePolysPrimary<F, PCS>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        let evaluate = |poly: &DensePolynomial<F>| -> F { poly.evaluate_at_chi(&chis) };
        Self {
            _marker: PhantomData,
            m_openings: polynomials_primary.m.par_iter().map(evaluate).collect(),
        }
    }

    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::prove_openings")]
    fn prove_openings(
        generators: &PCS::Setup,
        polynomials_primary: &SurgePolysPrimary<F, PCS>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let polys = polynomials_primary.m.iter().collect::<Vec<_>>();

        PCS::batch_prove(
            generators,
            &polys,
            opening_point,
            &openings.m_openings,
            BatchType::SurgeInitFinal,
            transcript,
        )
    }

    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::verify_openings")]
    fn verify_openings(
        &self,
        generators: &PCS::Setup,
        opening_proof: &Self::Proof,
        commitment_primary: &SurgeCommitmentPrimary<PCS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        PCS::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &self.m_openings,
            &commitment_primary.m_commitment.iter().collect::<Vec<_>>(),
            transcript,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeLogupGPolyOpenings<F>
where
    F: JoltField,
{
    dim_openings: Vec<F>,
    e_poly_openings: Vec<F>,
}

impl<F, PCS> StructuredOpeningProof<F, PCS, SurgePolysPrimary<F, PCS>>
    for SurgeLogupGPolyOpenings<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = PCS::BatchedProof;

    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::open")]
    fn open(polynomials_primary: &SurgePolysPrimary<F, PCS>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        let evaluate = |poly: &DensePolynomial<F>| -> F { poly.evaluate_at_chi(&chis) };
        Self {
            dim_openings: polynomials_primary.dim.par_iter().map(evaluate).collect(),
            e_poly_openings: polynomials_primary
                .E_polys
                .par_iter()
                .map(evaluate)
                .collect(),
        }
    }

    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::prove_openings")]
    fn prove_openings(
        generators: &PCS::Setup,
        polynomials_primary: &SurgePolysPrimary<F, PCS>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let polys = polynomials_primary
            .dim
            .iter()
            .chain(polynomials_primary.E_polys.iter())
            .collect::<Vec<_>>();
        let openings = [
            openings.dim_openings.as_slice(),
            openings.e_poly_openings.as_slice(),
        ]
        .concat();

        PCS::batch_prove(
            generators,
            &polys,
            opening_point,
            &openings,
            BatchType::SurgeReadWrite,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        generators: &PCS::Setup,
        opening_proof: &Self::Proof,
        commitment_primary: &SurgeCommitmentPrimary<PCS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let openings: Vec<F> = [
            self.dim_openings.as_slice(),
            self.e_poly_openings.as_slice(),
        ]
        .concat();

        PCS::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &openings,
            &commitment_primary
                .dim_commitment
                .iter()
                .chain(commitment_primary.E_commitment.iter())
                .collect::<Vec<_>>(),
            transcript,
        )
    }
}

trait SurgeCommons<F, PCS, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    fn num_memories() -> usize {
        C * Instruction::default().subtables::<F>(C, M).len()
    }

    fn num_subtables() -> usize {
        Instruction::default().subtables::<F>(C, M).len()
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, NUM_SUBTABLES)
    fn memory_to_subtable_index(i: usize) -> usize {
        i / C
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, C)
    fn memory_to_dimension_index(i: usize) -> usize {
        i % C
    }

    #[tracing::instrument(skip_all, name = "SurgeCommons::polys_from_evals")]
    fn polys_from_evals(all_evals: &Vec<Vec<F>>) -> Vec<DensePolynomial<F>> {
        all_evals
            .par_iter()
            .map(|evals| DensePolynomial::new(evals.to_vec()))
            .collect()
    }

    #[tracing::instrument(skip_all, name = "SurgeCommons::polys_from_evals_usize")]
    fn polys_from_evals_usize(all_evals_usize: &Vec<Vec<usize>>) -> Vec<DensePolynomial<F>> {
        all_evals_usize
            .par_iter()
            .map(|evals_usize| DensePolynomial::from_usize(&evals_usize))
            .collect()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LogupCheckingProof<F, PCS, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    pub f_final_claims: Vec<Pair<F>>,
    pub f_proof: BatchedRationalSumProof<PCS>,
    pub g_final_claims: Vec<Pair<F>>,
    pub g_proof: BatchedRationalSumProof<PCS>,
    pub f_openings: SurgeLogupFPolyOpenings<F, Instruction, C, M>,
    pub f_openings_proof:
        <SurgeLogupFPolyOpenings<F, Instruction, C, M> as StructuredOpeningProof<
            F,
            PCS,
            SurgePolysPrimary<F, PCS>,
        >>::Proof,
    pub g_openings: SurgeLogupGPolyOpenings<F>,
    pub g_openings_proof: <SurgeLogupGPolyOpenings<F> as StructuredOpeningProof<
        F,
        PCS,
        SurgePolysPrimary<F, PCS>,
    >>::Proof,
}

impl<F, PCS, Instruction, const C: usize, const M: usize> SurgeCommons<F, PCS, Instruction, C, M>
    for LogupCheckingProof<F, PCS, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
}

impl<F, PCS, Instruction, const C: usize, const M: usize>
    LogupCheckingProof<F, PCS, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    fn protocol_name() -> &'static [u8] {
        b"logup"
    }

    #[tracing::instrument(skip_all, name = "LogupCheckingProof::compute_leaves")]
    fn compute_leaves(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        polynomials: &mut SurgePolysPrimary<F, PCS>,
        beta: &F,
        gamma: &F,
    ) -> (
        (Vec<Vec<usize>>, Vec<Vec<F>>, Vec<Vec<F>>),
        (Vec<Vec<F>>, Vec<Vec<F>>),
    ) {
        let num_lookups = polynomials.dim[0].len();
        let (g_leaves, f_leaves_q) = rayon::join(
            || {
                (0..Self::num_memories())
                    .into_par_iter()
                    .map(|memory_index| -> (Vec<F>, Vec<F>) {
                        let dim_index = Self::memory_to_dimension_index(memory_index);
                        (0..num_lookups)
                            .map(|i| {
                                (
                                    F::one(),
                                    polynomials.E_polys[memory_index][i].mul_0_optimized(*gamma)
                                        + polynomials.dim[dim_index][i]
                                        + *beta,
                                )
                            })
                            .unzip()
                    })
                    .unzip()
            },
            || {
                preprocessing
                    .materialized_subtables
                    .par_iter()
                    .map(|subtable| {
                        subtable
                            .iter()
                            .enumerate()
                            .map(|(i, t_eval)| {
                                t_eval.mul_0_optimized(*gamma)
                                    + F::from_u64(i as u64).unwrap()
                                    + *beta
                            })
                            .collect()
                    })
                    .collect()
            },
        );
        (
            (
                take(&mut polynomials.m_indices),
                take(&mut polynomials.m_values),
                f_leaves_q,
            ),
            g_leaves,
        )
    }

    #[tracing::instrument(skip_all, name = "LogupCheckingProof::prove_logup_checking")]
    fn prove_logup_checking(
        generators: &PCS::Setup,
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        polynomials_primary: &mut SurgePolysPrimary<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> LogupCheckingProof<F, PCS, Instruction, C, M> {
        transcript.append_protocol_name(Self::protocol_name());

        // We assume that primary commitments are already appended to the transcript
        let beta: F = transcript.challenge_scalar(b"logup_beta");
        let gamma: F = transcript.challenge_scalar(b"logup_gamma");

        let (f_leaves, g_leaves) =
            Self::compute_leaves(preprocessing, polynomials_primary, &beta, &gamma);

        let ((mut f_batched_circuit, f_claims), (mut g_batched_circuit, g_claims)) = rayon::join(
            || {
                let f_batched_circuit = <BatchedSparseRationalSum<F, C> as BatchedRationalSum<
                    F,
                    PCS,
                >>::construct(f_leaves);
                let f_claims =
                    <BatchedSparseRationalSum<F, C> as BatchedRationalSum<F, PCS>>::claims(
                        &f_batched_circuit,
                    );
                (f_batched_circuit, f_claims)
            },
            || {
                let g_batched_circuit = <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<
                    F,
                    PCS,
                >>::construct(g_leaves);
                let g_claims =
                    <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F, PCS>>::claims(
                        &g_batched_circuit,
                    );
                (g_batched_circuit, g_claims)
            },
        );

        let (f_proof, r_f) =
            <BatchedSparseRationalSum<F, C> as BatchedRationalSum<F, PCS>>::prove_rational_sum(
                &mut f_batched_circuit,
                transcript,
                Some(generators),
            );

        let (g_proof, r_g) =
            <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F, PCS>>::prove_rational_sum(
                &mut g_batched_circuit,
                transcript,
                Some(generators),
            );

        drop_in_background_thread(f_batched_circuit);
        drop_in_background_thread(g_batched_circuit);

        let (f_openings, g_openings) = rayon::join(
            || SurgeLogupFPolyOpenings::<F, Instruction, C, M>::open(polynomials_primary, &r_f),
            || SurgeLogupGPolyOpenings::<F>::open(polynomials_primary, &r_g),
        );

        let f_openings_proof = SurgeLogupFPolyOpenings::<F, Instruction, C, M>::prove_openings(
            generators,
            &polynomials_primary,
            &r_f,
            &f_openings,
            transcript,
        );
        let g_openings_proof = SurgeLogupGPolyOpenings::<F>::prove_openings(
            generators,
            &polynomials_primary,
            &r_g,
            &g_openings,
            transcript,
        );

        LogupCheckingProof {
            f_final_claims: f_claims,
            f_proof,
            g_final_claims: g_claims,
            g_proof,
            f_openings,
            f_openings_proof,
            g_openings,
            g_openings_proof,
        }
    }

    #[tracing::instrument(skip_all, name = "LogupCheckingProof::check_openings")]
    fn check_openings(
        proof: &LogupCheckingProof<F, PCS, Instruction, C, M>,
        claims_f: Vec<Pair<F>>,
        claims_g: Vec<Pair<F>>,
        sid: F,
        t: Vec<F>,
        beta: &F,
        gamma: &F,
    ) {
        rayon::join(
            || {
                claims_f.into_par_iter().enumerate().for_each(|(i, claim)| {
                    assert_eq!(claim.p, proof.f_openings.m_openings[i]);
                    let subtable_idx = Self::memory_to_subtable_index(i);
                    assert_eq!(
                        claim.q,
                        *beta + sid + t[subtable_idx].mul_01_optimized(*gamma)
                    );
                })
            },
            || {
                claims_g.into_par_iter().enumerate().for_each(|(i, claim)| {
                    assert_eq!(claim.p, F::one());
                    assert_eq!(
                        claim.q,
                        *beta
                            + proof.g_openings.dim_openings[i]
                            + proof.g_openings.e_poly_openings[i] * *gamma
                    );
                })
            },
        );
    }

    #[tracing::instrument(skip_all, name = "LogupCheckingProof::verify_logup_checking")]
    fn verify_logup_checking(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        mut proof: LogupCheckingProof<F, PCS, Instruction, C, M>,
        commitment_primary: &SurgeCommitmentPrimary<PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        transcript.append_protocol_name(Self::protocol_name());

        let num_memories = Self::num_memories();

        // Assumes that primary commitments have been added to transcript
        let beta: F = transcript.challenge_scalar(b"logup_beta");
        let gamma: F = transcript.challenge_scalar(b"logup_gamma");

        // Check that the final claims are equal
        rayon::join(
            || {
                (0..num_memories).into_par_iter().for_each(|i| {
                    assert_eq!(
                        proof.f_final_claims[i].p * proof.g_final_claims[i].q,
                        proof.f_final_claims[i].q * proof.g_final_claims[i].p,
                        "Final claims are inconsistent"
                    );
                });
            },
            || {
                let (claims_f, r_f) = <BatchedSparseRationalSum<F, C> as BatchedRationalSum<
                    F,
                    PCS,
                >>::verify_rational_sum(
                    &proof.f_proof,
                    &proof.f_final_claims,
                    transcript,
                    Some(generators),
                );
                let ((sid, t), result) = rayon::join(
                    || proof.f_openings.compute_sid_t(&r_f),
                    || {
                        let (claims_g, r_g) = <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<
                    F,
                    PCS,
                >>::verify_rational_sum(
                    &proof.g_proof,
                    &proof.g_final_claims,
                    transcript,
                    Some(generators),
                );


                        proof.f_openings.verify_openings(
                            generators,
                            &proof.f_openings_proof,
                            commitment_primary,
                            &r_f,
                            transcript,
                        )?;
                        proof.g_openings.verify_openings(
                            generators,
                            &proof.g_openings_proof,
                            commitment_primary,
                            &r_g,
                            transcript,
                        )?;
                        Ok((claims_f, claims_g))
                    },
                );

                let (claims_f, claims_g) = result?;
                Self::check_openings(&proof, claims_f, claims_g, sid, t, &beta, &gamma);
                Ok(())
            },
        )
        .1
    }
}

pub struct SurgePreprocessing<F, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,
    pub materialized_subtables: Vec<Vec<F>>,
}

#[allow(clippy::type_complexity)]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeProof<F, PCS, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    /// Commitments to all polynomials
    commitment: SurgeCommitmentPrimary<PCS>,

    /// Primary collation sumcheck proof
    primary_sumcheck: SurgePrimarySumcheck<F, PCS>,

    logup_checking: LogupCheckingProof<F, PCS, Instruction, C, M>,
}

impl<F, PCS, Instruction, const C: usize, const M: usize> SurgeCommons<F, PCS, Instruction, C, M>
    for SurgeProof<F, PCS, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
}

impl<F, Instruction, const C: usize, const M: usize> SurgePreprocessing<F, Instruction, C, M>
where
    F: JoltField,
    Instruction: JoltInstruction + Default + Sync,
{
    #[tracing::instrument(skip_all, name = "Surge::preprocess")]
    pub fn preprocess() -> Self {
        let instruction = Instruction::default();

        let materialized_subtables = instruction
            .subtables(C, M)
            .par_iter()
            .map(|(subtable, _)| subtable.materialize(M))
            .collect();

        Self {
            _instruction: PhantomData,
            materialized_subtables,
        }
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize> SurgeProof<F, PCS, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    fn protocol_name() -> &'static [u8] {
        b"Surge"
    }

    /// Computes the maximum number of group generators needed to commit to Surge polynomials
    /// using Hyrax, given `M` and the maximum number of lookups.
    pub fn num_generators(max_num_lookups: usize) -> usize {
        let max_num_lookups = max_num_lookups.next_power_of_two();
        let num_read_write_generators = matrix_dimensions(max_num_lookups.log_2(), 16).1;
        let num_init_final_generators =
            matrix_dimensions((M * Self::num_memories()).next_power_of_two().log_2(), 4).1;
        std::cmp::max(num_read_write_generators, num_init_final_generators)
    }

    #[tracing::instrument(skip_all, name = "Surge::prove")]
    pub fn prove(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        ops: Vec<Instruction>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        transcript.append_protocol_name(Self::protocol_name());

        let num_lookups = ops.len().next_power_of_two();
        let mut polynomials = Self::construct_polys(preprocessing, &ops);
        let commitment = polynomials.commit(generators);
        commitment.append_to_transcript(b"primary_commitment", transcript);

        let num_rounds = num_lookups.log_2();
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let r_primary_sumcheck = transcript.challenge_vector(b"primary_sumcheck", num_rounds);
        let eq: DensePolynomial<F> = DensePolynomial::new(EqPolynomial::evals(&r_primary_sumcheck));
        let sumcheck_claim: F = Self::compute_primary_sumcheck_claim(&polynomials, &eq);

        transcript.append_scalar(b"sumcheck_claim", &sumcheck_claim);
        let mut combined_sumcheck_polys = polynomials.E_polys.clone();
        combined_sumcheck_polys.push(eq);

        let combine_lookups_eq = |vals: &[F]| -> F {
            let vals_no_eq: &[F] = &vals[0..(vals.len() - 1)];
            let eq = vals[vals.len() - 1];
            instruction.combine_lookups(vals_no_eq, C, M) * eq
        };

        let (primary_sumcheck_proof, r_z, _) = SumcheckInstanceProof::<F>::prove_arbitrary::<_>(
            &sumcheck_claim,
            num_rounds,
            &mut combined_sumcheck_polys,
            combine_lookups_eq,
            instruction.g_poly_degree(C) + 1, // combined degree + eq term
            transcript,
        );

        let sumcheck_openings = PrimarySumcheckOpenings::open(&polynomials, &r_z); // TODO: use return value from prove_arbitrary?
        let sumcheck_opening_proof = PrimarySumcheckOpenings::prove_openings(
            generators,
            &polynomials,
            &r_z,
            &sumcheck_openings,
            transcript,
        );

        let primary_sumcheck = SurgePrimarySumcheck {
            claimed_evaluation: sumcheck_claim,
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            openings: sumcheck_openings,
            opening_proof: sumcheck_opening_proof,
        };

        let logup_checking = LogupCheckingProof::<F, PCS, Instruction, C, M>::prove_logup_checking(
            generators,
            preprocessing,
            &mut polynomials,
            transcript,
        );

        SurgeProof {
            commitment,
            primary_sumcheck,
            logup_checking,
        }
    }

    #[tracing::instrument(skip_all, name = "Surge::verify")]
    pub fn verify(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        proof: SurgeProof<F, PCS, Instruction, C, M>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        transcript.append_protocol_name(Self::protocol_name());

        proof
            .commitment
            .append_to_transcript(b"primary_commitment", transcript);

        let instruction = Instruction::default();

        let r_primary_sumcheck =
            transcript.challenge_vector(b"primary_sumcheck", proof.primary_sumcheck.num_rounds);

        transcript.append_scalar(
            b"sumcheck_claim",
            &proof.primary_sumcheck.claimed_evaluation,
        );
        let primary_sumcheck_poly_degree = instruction.g_poly_degree(C) + 1;
        let (claim_last, r_z) = proof.primary_sumcheck.sumcheck_proof.verify(
            proof.primary_sumcheck.claimed_evaluation,
            proof.primary_sumcheck.num_rounds,
            primary_sumcheck_poly_degree,
            transcript,
        )?;

        rayon::join(
            || {
                proof.primary_sumcheck.openings.verify_openings(
                    generators,
                    &proof.primary_sumcheck.opening_proof,
                    &proof.commitment,
                    &r_z,
                    transcript,
                )?;

                LogupCheckingProof::<F, PCS, Instruction, C, M>::verify_logup_checking(
                    preprocessing,
                    generators,
                    proof.logup_checking,
                    &proof.commitment,
                    transcript,
                )
            },
            || {
                let eq_eval = EqPolynomial::new(r_primary_sumcheck.to_vec()).evaluate(&r_z);
                assert_eq!(
                    eq_eval * instruction.combine_lookups(&proof.primary_sumcheck.openings, C, M),
                    claim_last,
                    "Primary sumcheck check failed."
                );
            },
        )
        .0
    }

    #[tracing::instrument(skip_all, name = "Surge::construct_polys")]
    fn construct_polys(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        ops: &[Instruction],
    ) -> SurgePolysPrimary<F, PCS> {
        let num_memories = Self::num_memories();
        let num_lookups = ops.len().next_power_of_two();

        // Construct dim, m
        let mut dim_usize: Vec<Vec<usize>> = vec![vec![0; num_lookups]; C];

        let mut m_evals = vec![vec![0usize; M]; C];
        let log_M = ark_std::log2(M) as usize;

        for (op_index, op) in ops.iter().enumerate() {
            let access_sequence = op.to_indices(C, log_M);
            assert_eq!(access_sequence.len(), C);

            for dimension_index in 0..C {
                let memory_address = access_sequence[dimension_index];
                debug_assert!(memory_address < M);

                dim_usize[dimension_index][op_index] = memory_address;
                m_evals[dimension_index][memory_address] += 1;
            }
        }

        // num_ops is padded to the nearest power of 2 for the usage of DensePolynomial. We cannot just fill
        // in zeros for m_evals as this implicitly specifies a read at address 0.
        for _fake_ops_index in ops.len()..num_lookups {
            for dimension_index in 0..C {
                let memory_address = 0;
                m_evals[dimension_index][memory_address] += 1;
            }
        }

        let mut m_indices = vec![];
        let mut m_values = vec![];
        let mut dim_poly = vec![];
        let mut m_poly = vec![];
        let mut E_poly = vec![];
        rayon::scope(|s| {
            s.spawn(|_| {
                (m_indices, m_values) = m_evals
                    .iter()
                    .map(|m_evals_it| {
                        let mut indices = vec![];
                        let mut values = vec![];
                        for (i, m) in m_evals_it.iter().enumerate() {
                            if *m != 0 {
                                indices.push(i);
                                values.push(F::from_u64(*m as u64).unwrap());
                            }
                        }
                        (indices, values)
                    })
                    .unzip();
            });
            s.spawn(|_| {
                dim_poly = Self::polys_from_evals_usize(&dim_usize);
            });
            s.spawn(|_| {
                m_poly = Self::polys_from_evals_usize(&m_evals);
            });
            s.spawn(|_| {
                // Construct E
                let mut E_i_evals = Vec::with_capacity(num_memories);
                for E_index in 0..num_memories {
                    let mut E_evals = Vec::with_capacity(num_lookups);
                    for op_index in 0..num_lookups {
                        let dimension_index = Self::memory_to_dimension_index(E_index);
                        let subtable_index = Self::memory_to_subtable_index(E_index);

                        let eval_index = dim_usize[dimension_index][op_index];
                        let E_eval =
                            preprocessing.materialized_subtables[subtable_index][eval_index];
                        E_evals.push(E_eval);
                    }
                    E_i_evals.push(E_evals);
                }
                E_poly = Self::polys_from_evals(&E_i_evals);
            });
        });

        SurgePolysPrimary {
            _marker: PhantomData,
            dim: dim_poly,
            E_polys: E_poly,
            m: m_poly,
            m_indices,
            m_values,
        }
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_primary_sumcheck_claim")]
    fn compute_primary_sumcheck_claim(
        polys: &SurgePolysPrimary<F, PCS>,
        eq: &DensePolynomial<F>,
    ) -> F {
        let g_operands = &polys.E_polys;
        let hypercube_size = g_operands[0].len();
        g_operands
            .iter()
            .for_each(|operand| assert_eq!(operand.len(), hypercube_size));

        let instruction = Instruction::default();

        (0..hypercube_size)
            .into_par_iter()
            .map(|eval_index| {
                let g_operands: Vec<F> = (0..Self::num_memories())
                    .map(|memory_index| g_operands[memory_index][eval_index])
                    .collect();
                eq[eval_index] * instruction.combine_lookups(&g_operands, C, M)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::SurgePreprocessing;
    use crate::{
        jolt::instruction::xor::XORInstruction,
        lasso::surge::SurgeProof,
        poly::commitment::{
            hyrax::HyraxScheme, mock::MockCommitScheme, pedersen::PedersenGenerators,
        },
        utils::transcript::ProofTranscript,
    };
    use ark_bn254::{Fr, G1Projective};

    #[test]
    fn e2e() {
        let ops = vec![
            XORInstruction(12, 12),
            XORInstruction(12, 82),
            XORInstruction(12, 12),
            XORInstruction(25, 12),
        ];
        const C: usize = 8;
        const M: usize = 1 << 8;

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let preprocessing = SurgePreprocessing::preprocess();
        // let generators = PedersenGenerators::new(
        //     SurgeProof::<Fr, HyraxScheme<G1Projective>, XORInstruction, C, M>::num_generators(128),
        //     b"LassoV1",
        // );
        let proof = SurgeProof::<Fr, MockCommitScheme<Fr>, XORInstruction, C, M>::prove(
            &preprocessing,
            &(),
            ops,
            &mut transcript,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        SurgeProof::verify(&preprocessing, &(), proof, &mut transcript).expect("should work");
    }

    #[test]
    fn e2e_non_pow_2() {
        let ops = vec![
            XORInstruction(0, 1),
            XORInstruction(101, 101),
            XORInstruction(202, 1),
            XORInstruction(220, 1),
            XORInstruction(220, 1),
        ];
        const C: usize = 2;
        const M: usize = 1 << 8;

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let preprocessing = SurgePreprocessing::preprocess();
        // let generators = PedersenGenerators::new(
        //     SurgeProof::<Fr, HyraxScheme<G1Projective>, XORInstruction, C, M>::num_generators(128),
        //     b"LassoV1",
        // );
        let proof = SurgeProof::<Fr, MockCommitScheme<Fr>, XORInstruction, C, M>::prove(
            &preprocessing,
            &(),
            ops,
            &mut transcript,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        SurgeProof::verify(&preprocessing, &(), proof, &mut transcript).expect("should work");
    }
}
