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
            horner_univariate_div, radix2_fft, squares, transpose, Field, MultiMillerLoop,
        },
        chain,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use halo2_curves::CurveAffine;
use rand::RngCore;
use std::{marker::PhantomData, ops::Neg};

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
        let mut f = UnivariatePolynomial::monomial(poly.evals().to_vec());

        // Compute `h(X)`
        let h = {
            for x_i in &point[..t] {
                let f_i_minus_one = f.coeffs();
                let mut f_i = Vec::with_capacity(f_i_minus_one.len() >> 1);
                merge_into(&mut f_i, f_i_minus_one, x_i, 1, 0);
                f = UnivariatePolynomial::monomial(f_i);
            }

            f
        };

        // Compute `g(X) = f(X) (mod (X^b - \alpha))`
        let alpha = transcript.squeeze_challenge();
        let fis = {
            let mut fis = Vec::with_capacity(b);
            for _ in 0..b {
                fis.push(Vec::with_capacity(1 << (num_vars - t)));
            }
            f.coeffs().chunks(b).for_each(|chunk| {
                chunk.iter().enumerate().for_each(|(i, coeff)| {
                    fis[i].push(*coeff);
                });
            });
            fis.into_iter()
                .map(|coeffs| UnivariatePolynomial::monomial(coeffs))
                .collect_vec()
        };
        let (g, q) = {
            // f(X) = ∑ X^i • f_i(X^b)
            //      = ∑ X^i • ((X^b - \alpha) • q_i(X^b) + f_i(\alpha))
            //      = (X^b - \alpha) • ∑ X^i • q_i(X^b) + ∑ X^i • f_i(\alpha)
            //      = (X^b - \alpha) • q(X)             + g(X)
            let mut g_coeffs = Vec::with_capacity(b);
            for f_i in &fis {
                g_coeffs.push(f_i.evaluate(&alpha));
            }
            let g = UnivariatePolynomial::monomial(g_coeffs);
            let mut qis = Vec::with_capacity(b);
            for f_i in fis {
                qis.push(horner_univariate_div(f_i.coeffs(), &alpha));
            }
            let q_coeffs = transpose(&qis).into_iter().flatten().collect_vec();
            let q = UnivariatePolynomial::monomial(q_coeffs);
            (g, q)
        };

        // Commit to `h(X)`, `g(X)`, and `q(X)`
        UnivariateKzg::<M>::batch_commit_and_write(pp, &[h, g, q], transcript)?;

        let gamma = transcript.squeeze_challenge();

        // IPA for `˜g(u_1), ˜h(u_2)`
        let batched_ipa_poly = {
            // X^{b-1} (
            //  g(X) P_{u_1}(1/X) + g(1/X) P_{u_1}(X) +
            //  \gamma • (h(X) P_{u_2}(1/X) + h(1/X) P_{u_2}(X))
            // )
            let pu1 = UnivariatePolynomial::monomial(
                MultilinearPolynomial::eq_xy(&point[..t]).into_evals(),
            );
            let pu2 = UnivariatePolynomial::monomial(
                MultilinearPolynomial::eq_xy(&point[t..]).into_evals(),
            );

            
        };

        // degree of `g(X)` <= b
        let d = {
            let d_coeffs = g.coeffs().iter().rev().collect::<Vec<_>>();
            UnivariatePolynomial::monomial(d_coeffs)
        };

        let zeta = transcript.squeeze_challenge();

        todo!()
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
