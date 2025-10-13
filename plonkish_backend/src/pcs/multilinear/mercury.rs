//! Implementation of https://eprint.iacr.org/2025/385

use crate::{
    pcs::{
        multilinear::additive,
        univariate::{err_too_large_deree, UnivariateKzg, UnivariateKzgCommitment},
        Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{
        multilinear::{merge_into, MultilinearPolynomial},
        univariate::UnivariatePolynomial,
    },
    util::{
        arithmetic::{
            horner_univariate_div, inner_product, radix2_fft, root_of_unity, root_of_unity_inv,
            squares, transpose, Field, MultiMillerLoop,
        },
        chain, izip_eq,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use halo2_curves::{ff::PrimeField, CurveAffine};
use rand::RngCore;
use std::{iter::once, marker::PhantomData, ops::Neg};

#[derive(Clone, Debug)]
pub struct Mercury<Pcs>(PhantomData<Pcs>);

impl<M> PolynomialCommitmentScheme<M::Fr> for Mercury<UnivariateKzg<M>>
where
    M: MultiMillerLoop,
    M::Fr: Serialize + DeserializeOwned,
    M::G1Affine: Serialize + DeserializeOwned + CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: Serialize + DeserializeOwned + CurveAffine<ScalarExt = M::Fr>,
{
    type Param = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Fr>>::Param;
    type ProverParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Fr>>::ProverParam;
    type VerifierParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Fr>>::VerifierParam;
    type Polynomial = MultilinearPolynomial<M::Fr>;
    type Commitment = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Fr>>::Commitment;
    type CommitmentChunk = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Fr>>::CommitmentChunk;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        UnivariateKzg::<M>::setup(poly_size, batch_size, rng)
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        UnivariateKzg::<M>::trim(param, poly_size, batch_size)
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("commit", pp.degree(), got));
        }

        Ok(UnivariateKzg::commit_monomial(pp, poly.evals()))
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<M::Fr, Self::Polynomial>,
        eval: &M::Fr,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, M::Fr>,
    ) -> Result<(), Error> {
        let num_vars = point.len();
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("open", pp.degree(), got));
        }

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        let t = num_vars >> 1 + 1;
        let b: usize = 1 << t; // Folding factor
        let f = UnivariatePolynomial::monomial(poly.evals().to_vec());

        // Compute `h(X)`
        let h = {
            let mut h = f.clone();
            for x_i in &point[..t] {
                let f_i_minus_one = h.coeffs();
                let mut f_i = Vec::with_capacity(f_i_minus_one.len() >> 1);
                merge_into(&mut f_i, f_i_minus_one, x_i, 1, 0);
                h = UnivariatePolynomial::monomial(f_i);
            }

            h
        };

        // Commit to `h(X)`
        let h_comm = UnivariateKzg::commit_and_write(pp, &h, transcript)?;

        // Compute `g(X) = f(X) (mod (X^b - \alpha))`
        let alpha = transcript.squeeze_challenge();
        let f_is = {
            let f_is = transpose(&f.coeffs().chunks(b).collect_vec());
            f_is.into_iter()
                .map(|coeffs| UnivariatePolynomial::monomial(coeffs))
                .collect_vec()
        };
        let (g, q) = {
            // f(X) = ∑ X^i • f_i(X^b)
            //      = ∑ X^i • ((X^b - \alpha) • q_i(X^b) + f_i(\alpha))
            //      = (X^b - \alpha) • ∑ X^i • q_i(X^b)  + ∑ X^i • f_i(\alpha)
            //      = (X^b - \alpha) • q(X)              + g(X)
            let mut g_coeffs = Vec::with_capacity(b);
            for f_i in &f_is {
                g_coeffs.push(f_i.evaluate(&alpha));
            }
            let g = UnivariatePolynomial::monomial(g_coeffs);
            let mut q_is = Vec::with_capacity(b);
            for f_i in f_is {
                q_is.push(horner_univariate_div(f_i.coeffs(), &alpha));
            }
            let q_coeffs = transpose(&q_is.iter().map(|q_i| q_i.as_slice()).collect_vec())
                .into_iter()
                .flatten()
                .collect_vec();
            let q = UnivariatePolynomial::monomial(q_coeffs);
            (g, q)
        };

        // Commit to `g(X)`, and `q(X)`
        let comms = UnivariateKzg::<M>::batch_commit_and_write(pp, vec![&g, &q], transcript)?;
        let [g_comm, q_comm] = comms.try_into().unwrap();

        let gamma = transcript.squeeze_challenge();

        // IPA polynomial for `˜g(u_1), ˜h(u_2)`
        // X^{b-1} • (
        //      g(X) • P_{u_1}(1 / X) + g(1 / X) • P_{u_1}(X) +
        //      \gamma • (h(X) • P_{u_2}(1 / X) + h(1 / X) • P_{u_2}(X))
        // )
        let mut batched_ipa_poly = {
            let omega = root_of_unity(2 * b);
            // X^{b-1} • ( g(X) • P_{u_1}(1 / X) + g(1 / X) • P_{u_1}(X) )
            let mut lhs = {
                // g(X) • ( X^{b-1} • P_{u_1}(1 / X) ) + ( X^{b-1} • g(1 / X) ) • P_{u_1}(X)
                let p_u1 = MultilinearPolynomial::eq_xy(&point[..t]).into_evals();
                // fft ( X^{b-1} • P_{u_1}(1 / X) )
                let mut p_u1_inv_evals = p_u1.iter().rev().copied().collect_vec();
                p_u1_inv_evals.resize(2 * b, M::Fr::ZERO);
                radix2_fft(&mut p_u1_inv_evals, omega, t + 1);
                // fft ( g(X) )
                let mut g_evals = g.coeffs().to_vec();
                g_evals.resize(2 * b, M::Fr::ZERO);
                radix2_fft(&mut g_evals, omega, t + 1);
                // fft ( g(X) • ( X^{b-1} • P_{u_1}(1 / X) ) )
                let g_times_p_u1_inv = izip_eq!(g_evals, p_u1_inv_evals)
                    .map(|(g_eval, p_u1_inv_eval)| g_eval * p_u1_inv_eval)
                    .collect_vec();

                // fft ( X^{b-1} • g(1 / X) )
                let mut g_inv_evals = g.coeffs().iter().rev().copied().collect_vec();
                g_inv_evals.resize(2 * b, M::Fr::ZERO);
                radix2_fft(&mut g_inv_evals, omega, t + 1);
                // fft ( P_{u_1}(X) )
                let mut p_u1_evals = p_u1.clone();
                p_u1_evals.resize(2 * b, M::Fr::ZERO);
                radix2_fft(&mut p_u1_evals, omega, t + 1);
                // fft ( ( X^{b-1} • g(1 / X) ) • P_{u_1}(X) )
                let g_inv_times_p_u1 = izip_eq!(g_inv_evals, p_u1_evals)
                    .map(|(g_inv_eval, p_u1_eval)| g_inv_eval * p_u1_eval)
                    .collect_vec();

                izip_eq!(g_times_p_u1_inv, g_inv_times_p_u1)
                    .map(|(a, b)| a + b)
                    .collect_vec()
            };
            // X^{b-1} • ( h(X) • P_{u_2}(1 / X) + h(1 / X) • P_{u_2}(X) )
            let rhs = {
                // h(X) • ( X^{b-1} • P_{u_2}(1 / X) ) + ( X^{b-1} • h(1 / X) ) • P_{u_2}(X)
                let mut p_u2 = MultilinearPolynomial::eq_xy(&point[t..]).into_evals();
                p_u2.resize(b, M::Fr::ZERO);
                // fft ( X^{b-1} • P_{u_2}(1 / X) )
                let mut p_u2_inv_evals = p_u2.iter().rev().copied().collect_vec();
                p_u2_inv_evals.resize(2 * b, M::Fr::ZERO);
                radix2_fft(&mut p_u2_inv_evals, omega, t + 1);
                // fft ( h(X) )
                let mut h_evals = h.coeffs().to_vec();
                h_evals.resize(2 * b, M::Fr::ZERO);
                radix2_fft(&mut h_evals, omega, t + 1);
                // fft ( h(X) • ( X^{b-1} • P_{u_1}(1 / X) ) )
                let h_times_p_u1_inv = izip_eq!(h_evals, p_u2_inv_evals)
                    .map(|(h_eval, p_u2_inv_eval)| h_eval * p_u2_inv_eval)
                    .collect_vec();

                // fft ( X^{b-1} • h(1 / X) )
                let mut h = h.coeffs().to_vec();
                h.resize(b, M::Fr::ZERO);
                let mut h_inv_evals = h.iter().rev().copied().collect_vec();
                h_inv_evals.resize(2 * b, M::Fr::ZERO);
                radix2_fft(&mut h_inv_evals, omega, t + 1);
                // fft ( P_{u_2}(X) )
                let mut p_u2_evals = p_u2.clone();
                p_u2_evals.resize(2 * b, M::Fr::ZERO);
                radix2_fft(&mut p_u2_evals, omega, t + 1);
                // fft ( ( X^{b-1} • h(1 / X) ) • P_{u_2}(X) )
                let h_inv_times_p_u2 = izip_eq!(h_inv_evals, p_u2_evals)
                    .map(|(h_inv_eval, p_u2_eval)| h_inv_eval * p_u2_eval)
                    .collect_vec();

                izip_eq!(h_times_p_u1_inv, h_inv_times_p_u2)
                    .map(|(a, b)| a + b)
                    .collect_vec()
            };
            lhs.iter_mut()
                .zip(rhs)
                .for_each(|(lhs, rhs)| *lhs += gamma * rhs);
            lhs
        };

        // `batched_ipa_poly` = X^{b-1} • (2(˜g(u_1) + \gamma • ˜h(u_2)) + X • s(X) + (1 / X) • s(1 / X))
        let s = {
            // size `2b` ifft for `batched_ipa_poly`
            let omega_inv = root_of_unity_inv(t + 1);
            let n_inv = M::Fr::TWO_INV.pow_vartime([(t + 1) as u64]);
            radix2_fft(&mut batched_ipa_poly, omega_inv, t + 1);
            batched_ipa_poly
                .iter_mut()
                .for_each(|coeff| *coeff *= n_inv);
            UnivariatePolynomial::monomial(batched_ipa_poly[b..=2 * b - 2].to_vec())
        };

        // degree of `g(X)` <= b
        let d = {
            let d_coeffs = g.coeffs().iter().rev().copied().collect::<Vec<_>>();
            UnivariatePolynomial::monomial(d_coeffs)
        };

        // Commit to `s(X)`, and `d(X)`
        let comms = UnivariateKzg::<M>::batch_commit_and_write(pp, vec![&s, &d], transcript)?;
        let [s_comm, d_comm] = comms.try_into().unwrap();

        let zeta = transcript.squeeze_challenge();
        let zeta_inv = zeta.invert().unwrap();

        let polys = vec![&g, &h, &s, &d];
        let comms = vec![&g_comm, &h_comm, &s_comm, &d_comm];
        let points = vec![zeta.clone(), zeta_inv.clone(), alpha.clone()];
        let evals = vec![
            Evaluation::new(0, 0, g.evaluate(&zeta)),     // g(ζ)
            Evaluation::new(0, 1, g.evaluate(&zeta_inv)), // g(1/ζ)
            Evaluation::new(1, 0, h.evaluate(&zeta)),     // h(ζ)
            Evaluation::new(1, 1, h.evaluate(&zeta_inv)), // h(1/ζ)
            Evaluation::new(2, 0, s.evaluate(&zeta)),     // s(ζ)
            Evaluation::new(2, 1, s.evaluate(&zeta_inv)), // s(1/ζ)
            Evaluation::new(3, 0, d.evaluate(&zeta)),     // d(ζ)
            Evaluation::new(1, 2, h.evaluate(&alpha)),    // h(α)
        ];
        transcript.write_field_elements(evals[..5].iter().map(Evaluation::value))?;

        // [H(x)]
        let phi_zeta = {
            let numerator = {
                let zeta_pow = zeta.pow(&[b as u64]);
                let g_zeta = evals[0].value().clone();
                let mut acc = f.clone();
                acc -= &q * &(zeta_pow - alpha);
                acc -= (g_zeta, UnivariatePolynomial::monomial(vec![M::Fr::ONE]));
                acc
            };
            let divisor = UnivariatePolynomial::monomial(vec![-zeta, M::Fr::ONE]); // (X - ζ)
            let (quotient, remainder) = numerator.div_rem(&divisor);
            assert!(remainder.is_empty());
            UnivariateKzg::commit_monomial(pp, &quotient.coeffs())
        };

        transcript.write_commitment(&phi_zeta.0)?;

        UnivariateKzg::batch_open(pp, polys, comms, &points, &evals, transcript)?;

        Ok(())
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Fr, Self::Polynomial>],
        evals: &[Evaluation<M::Fr>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, M::Fr>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a,
    {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        additive::batch_open::<_, Self>(pp, num_vars, polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Fr>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        UnivariateKzg::read_commitments(vp, num_polys, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Fr, Self::Polynomial>,
        eval: &M::Fr,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Fr>,
    ) -> Result<(), Error> {
        let num_vars = point.len();
        todo!()
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Fr, Self::Polynomial>],
        evals: &[Evaluation<M::Fr>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Fr>,
    ) -> Result<(), Error> {
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        let comms = comms.into_iter().collect_vec();
        additive::batch_verify::<_, Self>(vp, num_vars, comms, points, evals, transcript)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::mercury::Mercury,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
            univariate::UnivariateKzg,
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::bn256::Bn256;

    type Pcs = Mercury<UnivariateKzg<Bn256>>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
