#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::jolt::vm::{JoltCommitments, JoltPolynomials, JoltStuff};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::utils::errors::ProofVerifyError;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::ProofTranscript;
use crate::{
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::grand_product::{
        BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductProof,
    },
};

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::interleave;
use rayon::prelude::*;
use std::iter::zip;
use super::memory_checking::{MultisetHashes, ExogenousOpenings, NoExogenousOpenings, StructuredPolynomialData, Initializable};
use crate::subprotocols::grand_product_piop::{GrandProductProof};

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct MemoryCheckingPIOPProof<F, PCS, Openings, OtherOpenings>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Openings: StructuredPolynomialData<F> + Sync + CanonicalSerialize + CanonicalDeserialize + Clone,
    OtherOpenings: ExogenousOpenings<F> + Sync + Clone,
{
    /// Read/write/init/final multiset hashes for each memory
    // pub multiset_hashes: MultisetHashes<F>,
    /// The read and write grand products for every memory has the same size,
    /// so they can be batched.
    pub read_write_grand_product: GrandProductProof<F, PCS>,
    /// The init and final grand products for every memory has the same size,
    /// so they can be batched.
    pub init_final_grand_product: GrandProductProof<F, PCS>,
    /// The openings associated with the grand products.
    pub openings: Openings,
    pub exogenous_openings: OtherOpenings,
}

// Empty struct to represent that no preprocessing data is used.
pub struct NoPreprocessing;

pub trait MemoryCheckingPIOPProver<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Self: Sync,
{
    type Polynomials: StructuredPolynomialData<DensePolynomial<F>>;
    type Openings: StructuredPolynomialData<F> + Sync + Initializable<F, Self::Preprocessing> + Clone;
    type Commitments: StructuredPolynomialData<PCS::Commitment>;
    type ExogenousOpenings: ExogenousOpenings<F> + Sync + Clone = NoExogenousOpenings;

    type Preprocessing = NoPreprocessing;

    /// The data associated with each memory slot. A triple (a, v, t) by default.
    type MemoryTuple = (F, F, F);

    #[tracing::instrument(skip_all, name = "MemoryCheckingPIOPProver::prove_memory_checking")]
    /// Generates a memory checking proof for the given committed polynomials.
    fn prove_memory_checking(
        pcs_setup: &PCS::Setup,
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> MemoryCheckingPIOPProof<F, PCS, Self::Openings, Self::ExogenousOpenings> {
        let (
            read_write_grand_product,
            init_final_grand_product,
        ) = Self::prove_grand_products(
            preprocessing,
            polynomials,
            jolt_polynomials,
            opening_accumulator,
            transcript,
            pcs_setup,
        );

        let (openings, exogenous_openings) = Self::compute_openings(
            preprocessing,
            opening_accumulator,
            polynomials,
            jolt_polynomials,
            &[],
            &[],
            transcript,
        );

        MemoryCheckingPIOPProof {
            // multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
            openings,
            exogenous_openings,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingPIOPProver::prove_grand_products")]
    /// Proves the grand products for the memory checking multisets (init, read, write, final).
    fn prove_grand_products(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        pcs_setup: &PCS::Setup,
    ) -> (
        GrandProductProof<F, PCS>,
        GrandProductProof<F, PCS>,
    ) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        transcript.append_protocol_name(Self::protocol_name());

        let ((read_leaves, write_leaves), (init_leaves, final_leaves)) =
            Self::compute_leaves(preprocessing, polynomials, jolt_polynomials, &gamma, &tau);
        let (read_write_grand_product, read_write_hashes, _, _) = GrandProductProof::prove(pcs_setup, 
            read_leaves, write_leaves, transcript);
        let (init_final_grand_product, init_final_hashes, _, _) = GrandProductProof::prove(pcs_setup, 
            init_leaves, final_leaves, transcript);
        assert_eq!(read_write_hashes, init_final_hashes);

        (
            read_write_grand_product,
            init_final_grand_product,
        )
    }

    fn compute_openings(
        preprocessing: &Self::Preprocessing,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        r_read_write: &[F],
        r_init_final: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self::Openings, Self::ExogenousOpenings) {
        let mut openings = Self::Openings::initialize(preprocessing);
        let mut exogenous_openings = Self::ExogenousOpenings::default();

        // let eq_read_write = EqPolynomial::evals(r_read_write);
        // polynomials
        //     .read_write_values()
        //     .par_iter()
        //     .zip_eq(openings.read_write_values_mut().into_par_iter())
        //     .chain(
        //         Self::ExogenousOpenings::exogenous_data(jolt_polynomials)
        //             .par_iter()
        //             .zip_eq(exogenous_openings.openings_mut().into_par_iter()),
        //     )
        //     .for_each(|(poly, opening)| {
        //         let claim = poly.evaluate_at_chi(&eq_read_write);
        //         *opening = claim;
        //     });

        // let read_write_polys: Vec<_> = [
        //     polynomials.read_write_values(),
        //     Self::ExogenousOpenings::exogenous_data(jolt_polynomials),
        // ]
        // .concat();
        // let read_write_claims: Vec<_> =
        //     [openings.read_write_values(), exogenous_openings.openings()].concat();
        // opening_accumulator.append(
        //     &read_write_polys,
        //     DensePolynomial::new(eq_read_write),
        //     r_read_write.to_vec(),
        //     &read_write_claims,
        //     transcript,
        // );

        // let eq_init_final = EqPolynomial::evals(r_init_final);
        // polynomials
        //     .init_final_values()
        //     .par_iter()
        //     .zip_eq(openings.init_final_values_mut().into_par_iter())
        //     .for_each(|(poly, opening)| {
        //         let claim = poly.evaluate_at_chi(&eq_init_final);
        //         *opening = claim;
        //     });

        // opening_accumulator.append(
        //     &polynomials.init_final_values(),
        //     DensePolynomial::new(eq_init_final),
        //     r_init_final.to_vec(),
        //     &openings.init_final_values(),
        //     transcript,
        // );

        (openings, exogenous_openings)
    }

    /// Computes the MLE of the leaves of the read, write, init, and final grand product circuits,
    /// one of each type per memory.
    /// Returns: (interleaved read/write leaves, interleaved init/final leaves)
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        exogenous_polynomials: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> (
        (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>),
        (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>),
    );

    /// Computes the Reed-Solomon fingerprint (parametrized by `gamma` and `tau`) of the given memory `tuple`.
    /// Each individual "leaf" of a grand product circuit (as computed by `read_leaves`, etc.) should be
    /// one such fingerprint.
    fn fingerprint(tuple: &Self::MemoryTuple, gamma: &F, tau: &F) -> F;
    /// Name of the memory checking instance, used for Fiat-Shamir.
    fn protocol_name() -> &'static [u8];
}

pub trait MemoryCheckingPIOPVerifier<F, PCS>: MemoryCheckingPIOPProver<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Verifies a memory checking proof, given its associated polynomial `commitment`.
    fn verify_memory_checking(
        preprocessing: &Self::Preprocessing,
        pcs_setup: &PCS::Setup,
        mut proof: MemoryCheckingPIOPProof<F, PCS, Self::Openings, Self::ExogenousOpenings>,
        commitments: &Self::Commitments,
        jolt_commitments: &JoltCommitments<PCS>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        transcript.append_protocol_name(Self::protocol_name());

        // assert_eq!(proof.)
        // proof.multiset_hashes.append_to_transcript(transcript);

        // let (read_write_hashes, init_final_hashes) =
        //     Self::interleave_hashes(preprocessing, &proof.multiset_hashes);

        // let (claims_read_write, r_read_write) = s(
        //     &proof.read_write_grand_product,
        //     &read_write_hashes,
        //     Some(opening_accumulator),
        //     transcript,
        //     Some(pcs_setup),
        // );
        // let (claims_init_final, r_init_final) = Self::InitFinalGrandProduct::verify_grand_product(
        //     &proof.init_final_grand_product,
        //     &init_final_hashes,
        //     Some(opening_accumulator),
        //     transcript,
        //     Some(pcs_setup),
        // );
        proof.read_write_grand_product.verify(transcript)?;
        proof.init_final_grand_product.verify(transcript)?;

        // let read_write_commits: Vec<_> = [
        //     commitments.read_write_values(),
        //     Self::ExogenousOpenings::exogenous_data(jolt_commitments),
        // ]
        // .concat();
        // let read_write_claims: Vec<_> = [
        //     proof.openings.read_write_values(),
        //     proof.exogenous_openings.openings(),
        // ]
        // .concat();
        // opening_accumulator.append(
        //     &read_write_commits,
        //     r_read_write.to_vec(),
        //     &read_write_claims,
        //     transcript,
        // );

        // opening_accumulator.append(
        //     &commitments.init_final_values(),
        //     r_init_final.to_vec(),
        //     &proof.openings.init_final_values(),
        //     transcript,
        // );

        // Self::compute_verifier_openings(
        //     &mut proof.openings,
        //     preprocessing,
        //     &r_read_write,
        //     &r_init_final,
        // );

        // Self::check_fingerprints(
        //     preprocessing,
        //     claims_read_write,
        //     claims_init_final,
        //     &proof.openings,
        //     &proof.exogenous_openings,
        //     &gamma,
        //     &tau,
        // );

        Ok(())
    }

    /// Often some of the openings do not require an opening proof provided by the prover, and
    /// instead can be efficiently computed by the verifier by itself. This function populates
    /// any such fields in `self`.
    fn compute_verifier_openings(
        _openings: &mut Self::Openings,
        _preprocessing: &Self::Preprocessing,
        _r_read_write: &[F],
        _r_init_final: &[F],
    ) {
    }

    /// Computes "read" memory tuples (one per memory) from the given `openings`.
    fn read_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        exogenous_openings: &Self::ExogenousOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "write" memory tuples (one per memory) from the given `openings`.
    fn write_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        exogenous_openings: &Self::ExogenousOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "init" memory tuples (one per memory) from the given `openings`.
    fn init_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        exogenous_openings: &Self::ExogenousOpenings,
    ) -> Vec<Self::MemoryTuple>;
    /// Computes "final" memory tuples (one per memory) from the given `openings`.
    fn final_tuples(
        preprocessing: &Self::Preprocessing,
        openings: &Self::Openings,
        exogenous_openings: &Self::ExogenousOpenings,
    ) -> Vec<Self::MemoryTuple>;

    // Checks that the claimed multiset hashes (output by grand product) are consistent with the
    // openings given by `read_write_openings` and `init_final_openings`.
    // fn check_fingerprints(
    //     preprocessing: &Self::Preprocessing,
    //     claims_read_write: Vec<F>,
    //     claims_init_final: Vec<F>,
    //     openings: &Self::Openings,
    //     exogenous_openings: &Self::ExogenousOpenings,
    //     gamma: &F,
    //     tau: &F,
    // ) {
    //     let read_hashes: Vec<_> = Self::read_tuples(preprocessing, openings, exogenous_openings)
    //         .iter()
    //         .map(|tuple| Self::fingerprint(tuple, gamma, tau))
    //         .collect();
    //     let write_hashes: Vec<_> = Self::write_tuples(preprocessing, openings, exogenous_openings)
    //         .iter()
    //         .map(|tuple| Self::fingerprint(tuple, gamma, tau))
    //         .collect();
    //     let init_hashes: Vec<_> = Self::init_tuples(preprocessing, openings, exogenous_openings)
    //         .iter()
    //         .map(|tuple| Self::fingerprint(tuple, gamma, tau))
    //         .collect();
    //     let final_hashes: Vec<_> = Self::final_tuples(preprocessing, openings, exogenous_openings)
    //         .iter()
    //         .map(|tuple| Self::fingerprint(tuple, gamma, tau))
    //         .collect();
    //     assert_eq!(
    //         read_hashes.len() + write_hashes.len(),
    //         claims_read_write.len()
    //     );
    //     assert_eq!(
    //         init_hashes.len() + final_hashes.len(),
    //         claims_init_final.len()
    //     );

    //     let multiset_hashes = MultisetHashes {
    //         read_hashes,
    //         write_hashes,
    //         init_hashes,
    //         final_hashes,
    //     };
    //     let (read_write_hashes, init_final_hashes) =
    //         Self::interleave_hashes(preprocessing, &multiset_hashes);

    //     for (claim, fingerprint) in zip(claims_read_write, read_write_hashes) {
    //         assert_eq!(claim, fingerprint);
    //     }
    //     for (claim, fingerprint) in zip(claims_init_final, init_final_hashes) {
    //         assert_eq!(claim, fingerprint);
    //     }
    // }
}
