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

use super::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};

#[derive(Clone)]
pub struct MockCommitScheme<F: JoltField> {
    _marker: PhantomData<F>,
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

impl<F: JoltField> CommitmentScheme for MockCommitScheme<F> {
    type Field = F;
    type Setup = ();
    type Commitment = MockCommitment;
    type Proof = ();
    type BatchedProof = ();

    fn setup(_shapes: &[CommitShape]) -> Self::Setup {}
    fn commit(poly: &DensePolynomial<Self::Field>, _setup: &Self::Setup) -> Self::Commitment {
        MockCommitment {}
    }
    fn batch_commit(
        evals: &[&[Self::Field]],
        _gens: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        vec![MockCommitment {}; evals.len()]
    }
    fn commit_slice(evals: &[Self::Field], _setup: &Self::Setup) -> Self::Commitment {
        MockCommitment {}
    }
    fn prove(
        _setup: &Self::Setup,
        _poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
    }
    fn batch_prove(
        _setup: &Self::Setup,
        _polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _batch_type: BatchType,
        _transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        MockCommitment {}
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
