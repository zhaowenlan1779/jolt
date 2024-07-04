use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::ProofTranscript;
use ark_serialize::*;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct ZerocheckInstanceProof<F: JoltField> {
    sumcheck_proof: SumcheckInstanceProof<F>,
}

impl<F: JoltField> ZerocheckInstanceProof<F> {
    pub fn new(sumcheck_proof: SumcheckInstanceProof<F>) -> ZerocheckInstanceProof<F> {
        ZerocheckInstanceProof { sumcheck_proof }
    }

    /// Create a zerocheck proof for polynomial(s) of arbitrary degree.
    ///
    /// Params
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `polys`: Dense polynomials to combine and sumcheck
    /// - `comb_func`: Function used to combine each polynomial evaluation
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (SumcheckInstanceProof, r_eval_point, final_evals)
    /// - `r_eval_point`: Final random point of evaluation
    /// - `final_evals`: Each of the polys evaluated at `r_eval_point`
    #[tracing::instrument(skip_all, name = "Zerocheck.prove")]
    pub fn prove_arbitrary<Func>(
        num_rounds: usize,
        polys: &mut Vec<DensePolynomial<F>>,
        comb_func: Func,
        combined_degree: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let r_primary_zerocheck = transcript.challenge_vector(b"primary_zerocheck", num_rounds);
        polys.push(DensePolynomial::new(EqPolynomial::evals(
            &r_primary_zerocheck,
        )));

        let new_comb_func = |vals: &[F]| -> F {
            match vals {
                [head @ .., last] => comb_func(head) * last,
                _ => panic!("Unexpected slice length"),
            }
        };

        let claim = F::zero();
        let (sumcheck, challenges, mut final_eval) = SumcheckInstanceProof::prove_arbitrary(
            &claim,
            num_rounds,
            polys,
            new_comb_func,
            combined_degree + 1,
            transcript,
        );
        final_eval.pop();

        (
            ZerocheckInstanceProof::new(sumcheck),
            challenges,
            final_eval,
        )
    }

    /// Verify this zerocheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, eq, r)
    /// - `e`: Claimed evaluation at random point
    /// - `eq`: Evaluation of the eq polynomial
    /// - `r`: Evaluation point
    /// Should verify that oracle(r) * eq = e
    pub fn verify(
        &self,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, F, Vec<F>), ProofVerifyError> {
        let r_primary_zerocheck = transcript.challenge_vector(b"primary_zerocheck", num_rounds);
        let eq_poly = EqPolynomial::new(r_primary_zerocheck);

        match self
            .sumcheck_proof
            .verify(F::zero(), num_rounds, degree_bound + 1, transcript)
        {
            Ok((e, r)) => Ok((e, eq_poly.evaluate(&r), r)),
            Err(err) => Err(err),
        }
    }
}
