// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::UniformRand;
use ark_std::rand::SeedableRng;
use jolt_core::{field::JoltField, poly::{commitment::{commitment_scheme::{BatchType, CommitShape, CommitmentScheme}, hyperkzg::HyperKZG, hyrax::HyraxScheme, zeromorph::Zeromorph}, dense_mlpoly::DensePolynomial},
utils::{errors::ProofVerifyError, transcript::ProofTranscript}};
use std::time::Instant;

fn main() {
    bench_pcs::<Fr, HyperKZG<Bn254>>("HyperKZG", 24).unwrap();
    bench_pcs::<Fr, HyraxScheme<G1Projective>>("Hyrax", 24).unwrap();
    bench_pcs::<Fr, Zeromorph<Bn254>>("Zeromorph", 24).unwrap();
}

fn bench_pcs<
    F: JoltField + UniformRand,
    PCS: CommitmentScheme<Field = F>,
>(
    name: &str,
    supported_nv: usize,
) -> Result<(), ProofVerifyError> {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(69420u64);

    // normal polynomials
    let setup = PCS::setup(&[CommitShape::new(1 << supported_nv, BatchType::Big)]);

    for nv in [20, 22, 24] {
        let repetition = if nv < 10 {
            10
        } else if nv < 20 {
            5
        } else {
            2
        };

        let poly = DensePolynomial::random(nv, &mut rng);

        let point: Vec<_> = (0..nv).map(|_| F::rand(&mut rng)).collect();

        // commit
        let com = {
            let start = Instant::now();
            for _ in 0..repetition {
                let _commit = PCS::commit(&poly, &setup);
            }

            println!(
                "{} commit for {} variables: {} ns",
                name,
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );

            PCS::commit(&poly, &setup)
        };

        // open
        let (proof, value) = {
            let start = Instant::now();
            for _ in 0..repetition {
                let mut transcript = ProofTranscript::new(b"test_transcript");
                let _open = PCS::prove(&setup, &poly, &point, &mut transcript);
            }

            println!(
                "{} open for {} variables: {} ns",
                name,
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
            let mut transcript = ProofTranscript::new(b"test_transcript");
            let proof = PCS::prove(&setup, &poly, &point, &mut transcript);
            let value = poly.evaluate(&point);
            (proof, value)
        };

        // verify
        {
            let start = Instant::now();
            for _ in 0..repetition {
                let mut transcript = ProofTranscript::new(b"test_transcript");
                PCS::verify(&proof, &setup, &mut transcript, &point, &value, &com).unwrap();
            }
            println!(
                "{} verify for {} variables: {} ns",
                name,
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
        }

        println!("====================================");
    }

    Ok(())
}
