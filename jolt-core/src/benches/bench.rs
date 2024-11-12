use crate::field::JoltField;
use crate::host;
use crate::jolt::instruction::xor::XORInstruction;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M};
use crate::jolt::vm::Jolt;
use crate::lasso::surge::{SurgePreprocessing, SurgeProof};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::poly::commitment::hyrax::HyraxScheme;
use crate::poly::commitment::mock::MockCommitScheme;
use crate::poly::commitment::zeromorph::Zeromorph;
use crate::utils::transcript::ProofTranscript;
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_std::test_rng;
use rand_core::RngCore;
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum PCSType {
    Hyrax,
    Zeromorph,
    HyperKZG,
    Mock,
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    pcs_type: PCSType,
    bench_type: BenchType,
    _num_cycles: Option<usize>,
    _memory_size: Option<usize>,
    _bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match pcs_type {
        PCSType::Hyrax => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Sha3 => sha3::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Fibonacci => fibonacci::<Fr, HyraxScheme<G1Projective>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::Zeromorph => match bench_type {
            BenchType::Sha2 => sha2::<Fr, Zeromorph<Bn254>>(),
            BenchType::Sha3 => sha3::<Fr, Zeromorph<Bn254>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, Zeromorph<Bn254>>(),
            BenchType::Fibonacci => fibonacci::<Fr, Zeromorph<Bn254>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::HyperKZG => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyperKZG<Bn254>>(),
            BenchType::Sha3 => sha3::<Fr, HyperKZG<Bn254>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, HyperKZG<Bn254>>(),
            BenchType::Fibonacci => fibonacci::<Fr, HyperKZG<Bn254>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::Mock => match bench_type {
            BenchType::Sha2 => sha2::<Fr, MockCommitScheme<Bn254>>(),
            BenchType::Sha3 => sha3::<Fr, MockCommitScheme<Bn254>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, MockCommitScheme<Bn254>>(),
            BenchType::Fibonacci => fibonacci::<Fr, MockCommitScheme<Bn254>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        _ => panic!("PCS Type does not have a mapping"),
    }
}

fn fibonacci<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<u32, PCS, F>("fibonacci-guest", &9u32)
}

fn sha2<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<Vec<u8>, PCS, F>("sha2-guest", &vec![5u8; 2048])
}

fn sha3<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<Vec<u8>, PCS, F>("sha3-guest", &vec![5u8; 2048])
}

#[allow(dead_code)]
fn serialize_and_print_size(name: &str, item: &impl ark_serialize::CanonicalSerialize) {
    use std::fs::File;
    let mut file = File::create("temp_file").unwrap();
    item.serialize_compressed(&mut file).unwrap();
    let file_size_bytes = file.metadata().unwrap().len();
    println!("{:<30} : {} bytes", name, file_size_bytes);
}

fn prove_example<T: Serialize, PCS, F>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut tasks = Vec::new();
    // let mut program = host::Program::new(example_name);
    // program.set_input(input);

    let task = move || {
        let generators = PCS::setup(&[CommitShape {
            input_length: 1 << 24,
            batch_type: BatchType::Big,
        }]);
        for nv in 20..=24 {
            let num_ops = 1 << nv;
            let preprocessing = SurgePreprocessing::preprocess();
            let mut total_time = 0;
            let mut func = |_| {
                let mut rng = test_rng();
                const C: usize = 4;
                const M: usize = 1 << 16;
                let operand_size = (M * M) as u64;
                let ops = std::iter::repeat_with(|| {
                    XORInstruction(
                        (rng.next_u32() as u64) % operand_size,
                        (rng.next_u32() as u64) % operand_size,
                    )
                })
                .take(num_ops)
                .collect::<Vec<_>>();

                let before = Instant::now();
                let proof = SurgeProof::<F, PCS, XORInstruction<32>, C, M>::prove(
                    &preprocessing,
                    &generators,
                    ops,
                );
                total_time += before.elapsed().as_micros();
                proof
            };

            
            (0..9).for_each(|i| {
                func(i);
            });
            let (proof, _) = func(0);
            println!(
                "prover time for {}: {} us",
                nv,
                total_time / 10
            );
            println!("Proof sizing for {}:", nv);
            // serialize_and_print_size("jolt_commitments", &jolt_commitments);
            serialize_and_print_size("proof", &proof);

            let before = Instant::now();
            for i in 0..50 {
                SurgeProof::verify(&preprocessing, &generators, proof.clone(), None).expect("should work");
            }
            println!(
                "verifier time for {}: {} us",
                nv,
                before.elapsed().as_micros() / 50
            );
        }
    };

    // let task = move || {
    //     let (bytecode, memory_init) = program.decode();
    //     let (io_device, trace, circuit_flags) = program.trace();

    //     let preprocessing: crate::jolt::vm::JoltPreprocessing<F, PCS> =
    //         RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 22);

    //     let (jolt_proof, jolt_commitments) = <RV32IJoltVM as Jolt<_, PCS, C, M>>::prove(
    //         io_device,
    //         trace,
    //         circuit_flags,
    //         preprocessing.clone(),
    //     );

    //     let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
    //     assert!(
    //         verification_result.is_ok(),
    //         "Verification failed with error: {:?}",
    //         verification_result.err()
    //     );
    // };

    task();

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn sha2chain<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new("sha2-chain-guest");
    program.set_input(&[5u8; 32]);
    program.set_input(&1000u32);

    let task = move || {
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace) = program.trace();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<C, F, PCS> =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 22);

        let (jolt_proof, jolt_commitments, _) =
            <RV32IJoltVM as Jolt<_, PCS, C, M>>::prove(io_device, trace, preprocessing.clone());
        let verification_result =
            RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments, None);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}
