use crate::field::JoltField;
use crate::lasso::memory_checking::NoPreprocessing;
use crate::poly::commitment::commitment_scheme::BatchType;
use crate::utils::transcript::AppendToTranscript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::{izip, Itertools};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::marker::{PhantomData, Sync};

use crate::{
    jolt::instruction::JoltInstruction,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, hyrax::matrix_dimensions},
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{StructuredCommitment, StructuredOpeningProof, StructuredOpeningProof2},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    subprotocols::zerocheck::ZerocheckInstanceProof,
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
    pub T_polys: Vec<DensePolynomial<F>>, // Size NUM_MEMORIES
    pub E_polys: Vec<DensePolynomial<F>>, // Size NUM_MEMORIES
    pub m: Vec<DensePolynomial<F>>,       // Size C
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
        let m_commitment = PCS::batch_commit_polys(&self.m, generators, BatchType::SurgeReadWrite);

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

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitmentLogup<CS: CommitmentScheme> {
    pub f_commitment: Vec<CS::Commitment>, // Size NUM_MEMORIES
    pub g_commitment: Vec<CS::Commitment>, // Size NUM_MEMORIES
}

impl<F, PCS> StructuredCommitment<PCS> for SurgePolysLogup<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Commitment = SurgeCommitmentLogup<PCS>;

    #[tracing::instrument(skip_all, name = "SurgePolysLogup::commit")]
    fn commit(&self, generators: &PCS::Setup) -> Self::Commitment {
        let f_commitment = PCS::batch_commit_polys(&self.f, generators, BatchType::SurgeReadWrite);
        let g_commitment = PCS::batch_commit_polys(&self.g, generators, BatchType::SurgeReadWrite);

        Self::Commitment {
            f_commitment,
            g_commitment,
        }
    }
}

impl<F, PCS> AppendToTranscript for SurgeCommitmentLogup<PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        [&self.f_commitment, &self.g_commitment]
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
    f_openings: Vec<F>,
    m_openings: Vec<F>,
    sid: Option<F>,    // Computed by verifier
    t: Option<Vec<F>>, // Computed by verifier
}

impl<F, PCS, Instruction, const C: usize, const M: usize>
    StructuredOpeningProof2<F, PCS, SurgePolysPrimary<F, PCS>, SurgePolysLogup<F, PCS>>
    for SurgeLogupFPolyOpenings<F, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    type Proof = PCS::BatchedProof;
    type Preprocessing = SurgePreprocessing<F, Instruction, C, M>;

    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::open")]
    fn open(
        polynomials_primary: &SurgePolysPrimary<F, PCS>,
        polynomials_logup: &SurgePolysLogup<F, PCS>,
        opening_point: &[F],
    ) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        let evaluate = |poly: &DensePolynomial<F>| -> F { poly.evaluate_at_chi(&chis) };
        Self {
            _marker: PhantomData,
            f_openings: polynomials_logup.f.par_iter().map(evaluate).collect(),
            m_openings: polynomials_primary.m.par_iter().map(evaluate).collect(),
            sid: None,
            t: None,
        }
    }

    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::prove_openings")]
    fn prove_openings(
        generators: &PCS::Setup,
        polynomials_primary: &SurgePolysPrimary<F, PCS>,
        polynomials_logup: &SurgePolysLogup<F, PCS>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let polys = polynomials_logup
            .f
            .iter()
            .chain(polynomials_primary.m.iter())
            .collect::<Vec<_>>();
        let openings = [
            openings.f_openings.as_slice(),
            openings.m_openings.as_slice(),
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

    fn compute_verifier_openings(&mut self, _: &Self::Preprocessing, opening_point: &[F]) {
        self.sid = Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
        self.t = Some(
            Instruction::default()
                .subtables(C, M)
                .iter()
                .map(|(subtable, _)| subtable.evaluate_mle(opening_point))
                .collect(),
        );
    }

    fn verify_openings(
        &self,
        generators: &PCS::Setup,
        opening_proof: &Self::Proof,
        commitment_primary: &SurgeCommitmentPrimary<PCS>,
        commitment_logup: &SurgeCommitmentLogup<PCS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let openings: Vec<F> = [self.f_openings.as_slice(), self.m_openings.as_slice()].concat();

        PCS::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &openings,
            &commitment_logup
                .f_commitment
                .iter()
                .chain(commitment_primary.m_commitment.iter())
                .collect::<Vec<_>>(),
            transcript,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeLogupGPolyOpenings<F>
where
    F: JoltField,
{
    g_openings: Vec<F>,
    dim_openings: Vec<F>,
    e_poly_openings: Vec<F>,
}

impl<F, PCS> StructuredOpeningProof2<F, PCS, SurgePolysPrimary<F, PCS>, SurgePolysLogup<F, PCS>>
    for SurgeLogupGPolyOpenings<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = PCS::BatchedProof;

    #[tracing::instrument(skip_all, name = "SurgeLogupOpenings::open")]
    fn open(
        polynomials_primary: &SurgePolysPrimary<F, PCS>,
        polynomials_logup: &SurgePolysLogup<F, PCS>,
        opening_point: &[F],
    ) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        let evaluate = |poly: &DensePolynomial<F>| -> F { poly.evaluate_at_chi(&chis) };
        Self {
            g_openings: polynomials_logup.g.par_iter().map(evaluate).collect(),
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
        polynomials_logup: &SurgePolysLogup<F, PCS>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let polys = polynomials_logup
            .g
            .iter()
            .chain(polynomials_primary.dim.iter())
            .chain(polynomials_primary.E_polys.iter())
            .collect::<Vec<_>>();
        let openings = [
            openings.g_openings.as_slice(),
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
        commitment_logup: &SurgeCommitmentLogup<PCS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let openings: Vec<F> = [
            self.g_openings.as_slice(),
            self.dim_openings.as_slice(),
            self.e_poly_openings.as_slice(),
        ]
        .concat();

        PCS::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &openings,
            &commitment_logup
                .g_commitment
                .iter()
                .chain(commitment_primary.dim_commitment.iter())
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

    fn polys_from_evals(all_evals: &Vec<Vec<F>>) -> Vec<DensePolynomial<F>> {
        all_evals
            .iter()
            .map(|evals| DensePolynomial::new(evals.to_vec()))
            .collect()
    }

    fn polys_from_evals_usize(all_evals_usize: &Vec<Vec<usize>>) -> Vec<DensePolynomial<F>> {
        all_evals_usize
            .iter()
            .map(|evals_usize| DensePolynomial::from_usize(&evals_usize))
            .collect()
    }
}

pub struct LogupCheckingProof<F, PCS, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    pub commitment_logup: SurgeCommitmentLogup<PCS>,

    pub sumcheck_f: SumcheckInstanceProof<F>,
    pub sumcheck_g: SumcheckInstanceProof<F>,
    pub primary_sumcheck_claim: F,
    pub sumcheck_f_openings: Vec<F>,
    pub sumcheck_f_openings_proof: PCS::BatchedProof,
    pub sumcheck_g_openings: Vec<F>,
    pub sumcheck_g_openings_proof: PCS::BatchedProof,
    pub m: usize,
    pub n: usize,
    pub zerocheck_f: ZerocheckInstanceProof<F>,
    pub zerocheck_g: ZerocheckInstanceProof<F>,
    pub f_openings: SurgeLogupFPolyOpenings<F, Instruction, C, M>,
    pub f_openings_proof:
        <SurgeLogupFPolyOpenings<F, Instruction, C, M> as StructuredOpeningProof2<
            F,
            PCS,
            SurgePolysPrimary<F, PCS>,
            SurgePolysLogup<F, PCS>,
        >>::Proof,
    pub g_openings: SurgeLogupGPolyOpenings<F>,
    pub g_openings_proof: <SurgeLogupGPolyOpenings<F> as StructuredOpeningProof2<
        F,
        PCS,
        SurgePolysPrimary<F, PCS>,
        SurgePolysLogup<F, PCS>,
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

    fn prove_logup_checking(
        generators: &PCS::Setup,
        // preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        polynomials_primary: &SurgePolysPrimary<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> LogupCheckingProof<F, PCS, Instruction, C, M> {
        transcript.append_protocol_name(Self::protocol_name());

        let num_memories = Self::num_memories();
        let num_subtables = Self::num_subtables();

        // We assume that primary commitments are already appended to the transcript
        let beta = transcript.challenge_scalar(b"logup_beta");
        let gamma = transcript.challenge_scalar(b"logup_gamma");

        let polynomials_logup = Self::compute_logup_polys(&polynomials_primary, beta, gamma);
        let commitment_logup = polynomials_logup.commit(generators);
        commitment_logup.append_to_transcript(b"logup_commitment", transcript);

        let r = transcript.challenge_vector(b"logup_linear_comb", num_memories);
        let sumcheck_comb =
            |vals: &[F]| -> F { vals.iter().enumerate().map(|(i, val)| r[i] * val).sum() };

        // Primary sumchecks
        let primary_sumcheck_claim = Self::calculate_logup_sumcheck_claim(&polynomials_logup, &r);
        let n = polynomials_logup.f[0].get_num_vars();
        let m = polynomials_logup.g[0].get_num_vars();
        let (sumcheck_f, r_f, f_evals) = SumcheckInstanceProof::prove_arbitrary(
            &primary_sumcheck_claim,
            n,
            &mut polynomials_logup.f.clone(),
            sumcheck_comb,
            1,
            transcript,
        );
        let (sumcheck_g, r_g, g_evals) = SumcheckInstanceProof::prove_arbitrary(
            &primary_sumcheck_claim,
            m,
            &mut polynomials_logup.g.clone(),
            sumcheck_comb,
            1,
            transcript,
        );

        let sumcheck_f_openings_proof = PCS::batch_prove(
            generators,
            &polynomials_logup.f.iter().collect::<Vec<_>>(),
            &r_f,
            &f_evals,
            BatchType::SurgeReadWrite,
            transcript,
        );
        let sumcheck_g_openings_proof = PCS::batch_prove(
            generators,
            &polynomials_logup.g.iter().collect::<Vec<_>>(),
            &r_g,
            &g_evals,
            BatchType::SurgeReadWrite,
            transcript,
        );

        let mut zerocheck_f_polys = [
            &polynomials_logup.f[..],
            &polynomials_primary.m[..],
            &polynomials_primary.T_polys[..],
        ]
        .concat();

        // TODO: This is really stupid
        // The correct way is probably to properly specialize this sumcheck
        zerocheck_f_polys.push(DensePolynomial::new(
            (0..polynomials_logup.f[0].len())
                .map(|i| F::from_u64(i as u64).unwrap())
                .collect_vec(),
        ));

        let zerocheck_f_comb = |vals: &[F]| -> F {
            let f_vals = &vals[..num_memories];
            let m_vals = &vals[num_memories..num_memories + C];
            let t_vals = &vals[num_memories + C..num_memories + C + num_subtables];
            let sid_val = vals[vals.len() - 1];
            (0..num_memories)
                .map(|i| -> F {
                    let subtable_index = Self::memory_to_subtable_index(i);
                    let dimension_index = Self::memory_to_dimension_index(i);
                    let val = (beta + sid_val + gamma * t_vals[subtable_index]) * f_vals[i]
                        - m_vals[dimension_index];
                    r[i] * val
                })
                .sum()
        };
        let (zerocheck_f, r_f_2, f_evals_2) = ZerocheckInstanceProof::prove_arbitrary(
            n,
            &mut zerocheck_f_polys,
            zerocheck_f_comb,
            2,
            transcript,
        );

        let mut zerocheck_g_polys = [
            &polynomials_logup.g[..],
            &polynomials_primary.E_polys[..],
            &polynomials_primary.dim[..],
        ]
        .concat();

        let zerocheck_g_comb = |vals: &[F]| -> F {
            let g_vals = &vals[..num_memories];
            let E_vals = &vals[num_memories..2 * num_memories];
            let dim_vals = &vals[2 * num_memories..];
            (0..num_memories)
                .map(|i| -> F {
                    let dimension_index = Self::memory_to_dimension_index(i);
                    let val = (beta + dim_vals[dimension_index] + gamma * E_vals[i]) * g_vals[i]
                        - F::one();
                    r[i] * val
                })
                .sum()
        };
        let (zerocheck_g, r_g_2, g_evals_2) = ZerocheckInstanceProof::prove_arbitrary(
            m,
            &mut zerocheck_g_polys,
            zerocheck_g_comb,
            2,
            transcript,
        );

        let f_openings = SurgeLogupFPolyOpenings::<F, Instruction, C, M> {
            _marker: PhantomData,
            f_openings: f_evals_2[..num_memories].to_vec(),
            m_openings: f_evals_2[num_memories..num_memories + C].to_vec(),
            sid: None,
            t: None,
        };
        let f_openings_proof = SurgeLogupFPolyOpenings::<F, Instruction, C, M>::prove_openings(
            generators,
            &polynomials_primary,
            &polynomials_logup,
            &r_f_2,
            &f_openings,
            transcript,
        );

        let g_openings = SurgeLogupGPolyOpenings {
            g_openings: g_evals_2[..num_memories].to_vec(),
            e_poly_openings: g_evals_2[num_memories..2 * num_memories].to_vec(),
            dim_openings: g_evals_2[2 * num_memories..].to_vec(),
        };
        let g_openings_proof = SurgeLogupGPolyOpenings::prove_openings(
            generators,
            &polynomials_primary,
            &polynomials_logup,
            &r_g_2,
            &g_openings,
            transcript,
        );

        LogupCheckingProof {
            commitment_logup,
            sumcheck_f,
            sumcheck_g,
            primary_sumcheck_claim,
            zerocheck_f,
            zerocheck_g,
            sumcheck_f_openings: f_evals,
            sumcheck_f_openings_proof,
            sumcheck_g_openings: g_evals,
            sumcheck_g_openings_proof,
            m,
            n,
            f_openings,
            f_openings_proof,
            g_openings,
            g_openings_proof,
        }
    }

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

        proof
            .commitment_logup
            .append_to_transcript(b"logup_commitment", transcript);

        let r: Vec<F> = transcript.challenge_vector(b"logup_linear_comb", num_memories);
        let sumcheck_comb =
            |vals: &[F]| -> F { vals.iter().enumerate().map(|(i, val)| r[i] * val).sum() };

        let (claims_sumcheck_f, r_f) =
            proof
                .sumcheck_f
                .verify(proof.primary_sumcheck_claim, proof.n, 1, transcript)?;
        let (claims_sumcheck_g, r_g) =
            proof
                .sumcheck_g
                .verify(proof.primary_sumcheck_claim, proof.m, 1, transcript)?;
        assert_eq!(
            claims_sumcheck_f,
            sumcheck_comb(&proof.sumcheck_f_openings),
            "Primary sumcheck failed f"
        );
        assert_eq!(
            claims_sumcheck_g,
            sumcheck_comb(&proof.sumcheck_g_openings),
            "Primary sumcheck failed g"
        );

        PCS::batch_verify(
            &proof.sumcheck_f_openings_proof,
            generators,
            &r_f,
            &proof.sumcheck_f_openings,
            &proof
                .commitment_logup
                .f_commitment
                .iter()
                .collect::<Vec<_>>(),
            transcript,
        )?;
        PCS::batch_verify(
            &proof.sumcheck_g_openings_proof,
            generators,
            &r_g,
            &proof.sumcheck_g_openings,
            &proof
                .commitment_logup
                .g_commitment
                .iter()
                .collect::<Vec<_>>(),
            transcript,
        )?;

        // Zerochecks
        let (claims_zerocheck_f, eq_f, r_f_2) = proof.zerocheck_f.verify(proof.n, 2, transcript)?;
        let (claims_zerocheck_g, eq_g, r_g_2) = proof.zerocheck_g.verify(proof.m, 2, transcript)?;
        StructuredOpeningProof2::<
            F,
            PCS,
            SurgePolysPrimary<F, PCS>,
            SurgePolysLogup<F, PCS>,
        >::compute_verifier_openings(&mut proof.f_openings, preprocessing, &r_f_2);

        let zerocheck_f_comb = |openings: &SurgeLogupFPolyOpenings<F, Instruction, C, M>| -> F {
            let sid = openings.sid.unwrap();
            let t = openings.t.as_ref().unwrap();

            (0..num_memories)
                .map(|i| -> F {
                    let subtable_index = Self::memory_to_subtable_index(i);
                    let dimension_index = Self::memory_to_dimension_index(i);
                    let val = (beta + sid + gamma * t[subtable_index]) * openings.f_openings[i]
                        - openings.m_openings[dimension_index];
                    r[i] * val
                })
                .sum()
        };

        assert_eq!(
            claims_zerocheck_f,
            zerocheck_f_comb(&proof.f_openings) * eq_f
        );

        let zerocheck_g_comb = |openings: &SurgeLogupGPolyOpenings<F>| -> F {
            (0..num_memories)
                .map(|i| -> F {
                    let dimension_index = Self::memory_to_dimension_index(i);
                    let val = (beta
                        + openings.dim_openings[dimension_index]
                        + gamma * openings.e_poly_openings[i])
                        * openings.g_openings[i]
                        - F::one();
                    r[i] * val
                })
                .sum()
        };
        assert_eq!(
            claims_zerocheck_g,
            zerocheck_g_comb(&proof.g_openings) * eq_g
        );

        proof.f_openings.verify_openings(
            generators,
            &proof.f_openings_proof,
            commitment_primary,
            &proof.commitment_logup,
            &r_f_2,
            transcript,
        )?;
        proof.g_openings.verify_openings(
            generators,
            &proof.g_openings_proof,
            commitment_primary,
            &proof.commitment_logup,
            &r_g_2,
            transcript,
        )
    }

    fn calculate_logup_sumcheck_claim(polys: &SurgePolysLogup<F, PCS>, r: &[F]) -> F {
        polys
            .g
            .iter()
            .enumerate()
            .map(|(i, g_poly)| r[i] * g_poly.Z.iter().sum::<F>())
            .sum()
    }

    fn compute_logup_polys(
        polys: &SurgePolysPrimary<F, PCS>,
        beta: F,
        gamma: F,
    ) -> SurgePolysLogup<F, PCS> {
        let num_memories = Self::num_memories();
        let num_lookups = polys.E_polys[0].len();

        let mut g_i_evals = Vec::with_capacity(num_memories);
        for E_index in 0..num_memories {
            let mut g_evals = Vec::with_capacity(num_lookups);
            for op_index in 0..num_lookups {
                let dimension_index = Self::memory_to_dimension_index(E_index);
                g_evals.push(
                    F::one()
                        / (beta
                            + polys.dim[dimension_index][op_index]
                            + gamma * polys.E_polys[E_index][op_index]),
                );
            }
            g_i_evals.push(g_evals);
        }
        let g_poly = Self::polys_from_evals(&g_i_evals);

        // Construct f
        let f_i_evals = (0..num_memories)
            .map(|f_index| {
                let dimension_index = Self::memory_to_dimension_index(f_index);
                let subtable_index = Self::memory_to_subtable_index(f_index);

                izip!(
                    &polys.m[dimension_index].Z,
                    &polys.T_polys[subtable_index].Z
                )
                .enumerate()
                .map(|(i, (m_val, t_val))| {
                    (*m_val) / (beta + F::from_u64(i as u64).unwrap() + gamma * (*t_val))
                })
                .collect()
            })
            .collect();
        let f_poly = Self::polys_from_evals(&f_i_evals);

        SurgePolysLogup {
            _marker: PhantomData,
            f: f_poly,
            g: g_poly,
        }
    }
}

pub struct SurgePreprocessing<F, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,
    materialized_subtables: Vec<Vec<F>>,
}

#[allow(clippy::type_complexity)]
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
        let polynomials = Self::construct_polys(preprocessing, &ops);
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
            &polynomials,
            transcript,
        );

        SurgeProof {
            commitment,
            primary_sumcheck,
            logup_checking,
        }
    }

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

        let eq_eval = EqPolynomial::new(r_primary_sumcheck.to_vec()).evaluate(&r_z);
        assert_eq!(
            eq_eval * instruction.combine_lookups(&proof.primary_sumcheck.openings, C, M),
            claim_last,
            "Primary sumcheck check failed."
        );

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

        let dim_poly = Self::polys_from_evals_usize(&dim_usize);
        let m_poly = Self::polys_from_evals_usize(&m_evals);

        // Construct E
        let mut E_i_evals = Vec::with_capacity(num_memories);
        for E_index in 0..num_memories {
            let mut E_evals = Vec::with_capacity(num_lookups);
            for op_index in 0..num_lookups {
                let dimension_index = Self::memory_to_dimension_index(E_index);
                let subtable_index = Self::memory_to_subtable_index(E_index);

                let eval_index = dim_usize[dimension_index][op_index];
                let E_eval = preprocessing.materialized_subtables[subtable_index][eval_index];
                E_evals.push(E_eval);
            }
            E_i_evals.push(E_evals);
        }
        let E_poly = Self::polys_from_evals(&E_i_evals);

        // Construct T
        let T_poly = preprocessing
            .materialized_subtables
            .iter()
            .map(|subtable| DensePolynomial::new(subtable.to_vec()))
            .collect::<Vec<_>>();

        SurgePolysPrimary {
            _marker: PhantomData,
            dim: dim_poly,
            T_polys: T_poly,
            E_polys: E_poly,
            m: m_poly,
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
