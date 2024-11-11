use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    field::JoltField,
    poly::dense_mlpoly::DensePolynomial,
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};
use crate::utils::math::Math;
use super::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ec::VariableBaseMSM;

#[derive(Clone)]
pub struct MockCommitScheme<E: Pairing>
    where E::ScalarField : JoltField {
    _marker: PhantomData<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Default, Debug, PartialEq, Clone)]
pub struct MockCommitment {
}

impl AppendToTranscript for MockCommitment {
    fn append_to_transcript(&self, transcript: &mut ProofTranscript) {
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MockProof<F: JoltField> {
    opening_point: Vec<F>,
}

/// Evaluations over {0,1}^n for G1 or G2
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct Evaluations<C: AffineRepr> {
    /// The evaluations.
    pub evals: Vec<C>,
}

/// Prover Parameters
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MultilinearProverParam<E: Pairing> {
    /// number of variables
    pub num_vars: usize,
    /// `pp_{0}`, `pp_{1}`, ...,pp_{nu_vars} defined
    /// by XZZPD19 where pp_{nv-0}=g and
    /// pp_{nv-i}=g^{eq((t_1,..t_i),(X_1,..X_i))}
    pub powers_of_g: Vec<Evaluations<E::G1Affine>>,
    /// generator for G1
    pub g: E::G1Affine,
    /// generator for G2
    pub h: E::G2Affine,
}

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
/// A commitment is an Affine point.
pub struct Commitment<C: AffineRepr>(
    /// the actual commitment is an affine point.
    pub C,
);

impl<C: AffineRepr> AppendToTranscript for Commitment<C> {
    fn append_to_transcript(&self, transcript: &mut ProofTranscript) {
        transcript.append_point(&self.0.into());
    }
}

impl<E: Pairing> CommitmentScheme for MockCommitScheme<E>
    where E::ScalarField: JoltField {
    type Field = E::ScalarField;
    type Setup = MultilinearProverParam<E>;
    type Commitment = Commitment<E::G1Affine>;
    type Proof = ();
    type BatchedProof = ();

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        let max_len = shapes.iter().map(|shape| shape.input_length).max().unwrap().log_2();
        
        let prover_setup_filepath = format!(
            "mkzg-max{}.paras",
            max_len
        );
        let prover_setup = {
            let file = std::fs::File::open(prover_setup_filepath).unwrap();
            MultilinearProverParam::deserialize_uncompressed_unchecked(std::io::BufReader::new(file)).unwrap()
        };
        prover_setup
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    fn commit(
        poly: &DensePolynomial<Self::Field>,
        prover_param: &Self::Setup,
    ) -> Self::Commitment {
        if prover_param.num_vars < poly.get_num_vars() {
            panic!(
                "MlE length ({}) exceeds param limit ({})",
                poly.get_num_vars(), prover_param.num_vars
            );
        }
        let ignored = prover_param.num_vars - poly.get_num_vars();
        let commitment =
            E::G1MSM::msm_unchecked_par_auto(&prover_param.powers_of_g[ignored].evals, &poly.Z)
                .into();
        Commitment(commitment)
    }
    fn batch_commit(
        evals: &[&[Self::Field]],
        prover_param: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        evals.iter().map(|eval| Self::commit_slice(eval, prover_param))
            .collect()
    }
    fn commit_slice(evals: &[Self::Field], prover_param: &Self::Setup) -> Self::Commitment {
        if prover_param.num_vars < evals.len().log_2() {
            panic!(
                "MlE length ({}) exceeds param limit ({})",
                evals.len(), prover_param.num_vars
            );
        }
        let ignored = prover_param.num_vars - evals.len().log_2();
        let commitment =
            E::G1MSM::msm_unchecked_par_auto(&prover_param.powers_of_g[ignored].evals, evals)
                .into();
        Commitment(commitment)
    }
    fn prove(
        _setup: &Self::Setup,
        _poly: &DensePolynomial<Self::Field>,
        _opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
    }
    fn batch_prove(
        _setup: &Self::Setup,
        _polynomials: &[&DensePolynomial<Self::Field>],
        _opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _batch_type: BatchType,
        _transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        Commitment(E::G1MSM::msm_unchecked_par_auto(&commitments.iter().map(|comm| comm.0.clone()).collect::<Vec<_>>(), coeffs)
            .into())
    }

    fn verify(
        proof: &Self::Proof,
        _setup: &Self::Setup,
        _transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        Ok(())
    }

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        _setup: &Self::Setup,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"mock_commit"
    }
}
