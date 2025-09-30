use crate::{
    backend::serde::SerdeParam, pcs::{
        multilinear::{additive, err_too_many_variates, quotients, validate_input},
        Additive, Evaluation, Point, PolynomialCommitmentScheme,
    }, poly::multilinear::MultilinearPolynomial, util::{
        arithmetic::{
            batch_projective_to_affine, fixed_base_msm, variable_base_msm, window_size,
            window_table, Curve, CurveAffine, Field, MultiMillerLoop,
        },
        chain, izip,
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    }, Error
};
use halo2_curves::serde::SerdeObject;
use halo2_proofs::SerdeCurveAffine;
use pasta_curves::group::prime::PrimeCurveAffine;
use rand::RngCore;
use std::{iter, marker::PhantomData, ops::Neg, slice};

#[derive(Clone, Debug)]
pub struct MultilinearKzg<M>(PhantomData<M>)
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearKzgParam<M>
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr>,
{
    g1: M::G1Affine,
    eqs: Vec<Vec<M::G1Affine>>,
    g2: M::G2Affine,
    ss: Vec<M::G2Affine>,
}

impl<M> MultilinearKzgParam<M>
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr>,
{
    pub fn num_vars(&self) -> usize {
        self.eqs.len()
    }

    pub fn g1(&self) -> M::G1Affine {
        self.g1
    }

    pub fn eqs(&self) -> &[Vec<M::G1Affine>] {
        &self.eqs
    }

    pub fn g2(&self) -> M::G2Affine {
        self.g2
    }

    pub fn ss(&self) -> &[M::G2Affine] {
        &self.ss
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearKzgProverParam<M>
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr>,
{
    g1: M::G1Affine,
    eqs: Vec<Vec<M::G1Affine>>,
}

impl<M> MultilinearKzgProverParam<M>
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr>,
{
    pub fn num_vars(&self) -> usize {
        self.eqs.len() - 1
    }

    pub fn g1(&self) -> M::G1Affine {
        self.g1
    }

    pub fn eqs(&self) -> &[Vec<M::G1Affine>] {
        &self.eqs
    }

    pub fn eq(&self, num_vars: usize) -> &[M::G1Affine] {
        &self.eqs[num_vars]
    }
}

impl<M> SerdeParam for MultilinearKzgProverParam<M>
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr> + SerdeCurveAffine,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr> + SerdeCurveAffine,
{
    fn write_param<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.g1.write_raw(writer)?;
        writer.write_all(&(self.eqs.len() as u32).to_le_bytes())?;
        for eq in &self.eqs {
            writer.write_all(&(eq.len() as u32).to_le_bytes())?;
            for g1 in eq {
                g1.write_raw(writer)?;
            }
        }
        Ok(())
    }

    fn read_param<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let g1 = M::G1Affine::read_raw(reader)?;
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let num_eqs = u32::from_le_bytes(len_bytes) as usize;
        let mut eqs = Vec::with_capacity(num_eqs);
        for _ in 0..num_eqs {
            reader.read_exact(&mut len_bytes)?;
            let eq_len = u32::from_le_bytes(len_bytes) as usize;
            let mut eq = Vec::with_capacity(eq_len);
            for _ in 0..eq_len {
                eq.push(M::G1Affine::read_raw(reader)?);
            }
            eqs.push(eq);
        }
        Ok(Self { g1, eqs })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearKzgVerifierParam<M>
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr>,
{
    g1: M::G1Affine,
    g2: M::G2Affine,
    ss: Vec<M::G2Affine>,
}

impl<M> MultilinearKzgVerifierParam<M>
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr>,
{
    pub fn num_vars(&self) -> usize {
        self.ss.len()
    }

    pub fn g1(&self) -> M::G1Affine {
        self.g1
    }

    pub fn g2(&self) -> M::G2Affine {
        self.g2
    }

    pub fn ss(&self, num_vars: usize) -> &[M::G2Affine] {
        &self.ss[..num_vars]
    }
}

impl<M> SerdeParam for MultilinearKzgVerifierParam<M>
where
    M: MultiMillerLoop,
    M::G1Affine: CurveAffine<ScalarExt = M::Fr> + SerdeCurveAffine,
    M::G2Affine: CurveAffine<ScalarExt = M::Fr> + SerdeCurveAffine,
{
    fn write_param<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.g1.write_raw(writer)?;
        self.g2.write_raw(writer)?;
        writer.write_all(&(self.ss.len() as u32).to_le_bytes())?;
        for g2 in &self.ss {
            g2.write_raw(writer)?;
        }
        Ok(())
    }

    fn read_param<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let g1 = M::G1Affine::read_raw(reader)?;
        let g2 = M::G2Affine::read_raw(reader)?;
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let ss_len = u32::from_le_bytes(len_bytes) as usize;
        let mut ss = Vec::with_capacity(ss_len);
        for _ in 0..ss_len {
            ss.push(M::G2Affine::read_raw(reader)?);
        }
        Ok(Self { g1, g2, ss })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearKzgCommitment<C: CurveAffine>(pub C);

impl<C: SerdeCurveAffine> SerdeObject for MultilinearKzgCommitment<C> {
    fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_raw(writer)
    }

    fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        C::read_raw(reader).map(MultilinearKzgCommitment)
    }

    fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
        MultilinearKzgCommitment(C::read_raw_unchecked(reader))
    }

    fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
        C::from_raw_bytes(bytes).map(MultilinearKzgCommitment)
    }

    fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
        MultilinearKzgCommitment(C::from_raw_bytes_unchecked(bytes))
    }

    fn to_raw_bytes(&self) -> Vec<u8> {
        self.0.to_raw_bytes()
    }
}

impl<C: CurveAffine> Default for MultilinearKzgCommitment<C> {
    fn default() -> Self {
        Self(C::identity())
    }
}

impl<C: CurveAffine> PartialEq for MultilinearKzgCommitment<C> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<C: CurveAffine> Eq for MultilinearKzgCommitment<C> {}

impl<C: CurveAffine> AsRef<[C]> for MultilinearKzgCommitment<C> {
    fn as_ref(&self) -> &[C] {
        slice::from_ref(&self.0)
    }
}

impl<C: CurveAffine> AsRef<C> for MultilinearKzgCommitment<C> {
    fn as_ref(&self) -> &C {
        &self.0
    }
}

impl<C: CurveAffine> From<C> for MultilinearKzgCommitment<C> {
    fn from(comm: C) -> Self {
        Self(comm)
    }
}

impl<C: CurveAffine> Additive<C::Scalar> for MultilinearKzgCommitment<C> {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a C::Scalar>,
        bases: impl IntoIterator<Item = &'b Self>,
    ) -> Self {
        let scalars = scalars.into_iter().collect_vec();
        let bases = bases.into_iter().map(|base| &base.0).collect_vec();
        MultilinearKzgCommitment(variable_base_msm(scalars, bases).to_affine())
    }
}

impl<M> PolynomialCommitmentScheme<M::Fr> for MultilinearKzg<M>
where
    M: MultiMillerLoop,
    M::Fr: Serialize + DeserializeOwned,
    M::G1Affine: Serialize + DeserializeOwned + CurveAffine<ScalarExt = M::Fr>,
    M::G2Affine: Serialize + DeserializeOwned + CurveAffine<ScalarExt = M::Fr>,
{
    type Param = MultilinearKzgParam<M>;
    type ProverParam = MultilinearKzgProverParam<M>;
    type VerifierParam = MultilinearKzgVerifierParam<M>;
    type Polynomial = MultilinearPolynomial<M::Fr>;
    type Commitment = MultilinearKzgCommitment<M::G1Affine>;
    type CommitmentChunk = M::G1Affine;

    fn setup(poly_size: usize, _: usize, mut rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());

        let num_vars = poly_size.ilog2() as usize;
        let ss = iter::repeat_with(|| M::Fr::random(&mut rng))
            .take(num_vars)
            .collect_vec();

        let g1 = <M::G1Affine as PrimeCurveAffine>::generator();
        let eqs = {
            let mut eqs = Vec::with_capacity(1 << (num_vars + 1));
            eqs.push(vec![M::Fr::ONE]);

            for s_i in ss.iter() {
                let last_evals = eqs.last().unwrap();
                let mut evals = vec![M::Fr::ZERO; 2 * last_evals.len()];

                let (evals_lo, evals_hi) = evals.split_at_mut(last_evals.len());

                parallelize(evals_hi, |(evals_hi, start)| {
                    izip!(evals_hi, &last_evals[start..])
                        .for_each(|(eval_hi, last_eval)| *eval_hi = *s_i * last_eval);
                });
                parallelize(evals_lo, |(evals_lo, start)| {
                    izip!(evals_lo, &evals_hi[start..], &last_evals[start..])
                        .for_each(|(eval_lo, eval_hi, last_eval)| *eval_lo = *last_eval - eval_hi);
                });

                eqs.push(evals)
            }

            let window_size = window_size((2 << num_vars) - 2);
            let window_table = window_table(window_size, g1);
            let mut eqs = batch_projective_to_affine(&fixed_base_msm(
                window_size,
                &window_table,
                eqs.iter().flat_map(|evals| evals.iter()),
            ));

            let eqs = &mut eqs.drain(..);
            (0..num_vars + 1)
                .map(move |idx| eqs.take(1 << idx).collect_vec())
                .collect_vec()
        };

        let g2 = <M::G2Affine as PrimeCurveAffine>::generator();
        let ss = {
            let window_size = window_size(num_vars);
            let window_table = window_table(window_size, M::G2Affine::generator());
            batch_projective_to_affine(&fixed_base_msm(window_size, &window_table, &ss))
        };

        Ok(Self::Param { g1, eqs, g2, ss })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        if param.num_vars() < num_vars {
            return Err(err_too_many_variates("trim", param.num_vars(), num_vars));
        }
        let pp = Self::ProverParam {
            g1: param.g1,
            eqs: param.eqs[..num_vars + 1].to_vec(),
        };
        let vp = Self::VerifierParam {
            g1: param.g1,
            g2: param.g2,
            ss: param.ss[..num_vars].to_vec(),
        };
        Ok((pp, vp))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        Ok(variable_base_msm(poly.evals(), pp.eq(poly.num_vars())).into())
            .map(MultilinearKzgCommitment)
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let polys = polys.into_iter().collect_vec();
        if polys.is_empty() {
            return Ok(Vec::new());
        }
        validate_input("batch commit", pp.num_vars(), polys.iter().copied(), None)?;

        Ok(polys
            .iter()
            .map(|poly| variable_base_msm(poly.evals(), pp.eq(poly.num_vars())).into())
            .map(MultilinearKzgCommitment)
            .collect())
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<M::Fr, Self::Polynomial>,
        eval: &M::Fr,
        transcript: &mut impl TranscriptWrite<M::G1Affine, M::Fr>,
    ) -> Result<(), Error> {
        validate_input("open", pp.num_vars(), [poly], [point])?;

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        let (quotient_comms, remainder) = quotients(poly, point, |num_vars, quotient| {
            variable_base_msm(&quotient, pp.eq(num_vars)).into()
        });

        if cfg!(feature = "sanity-check") {
            assert_eq!(&remainder, eval);
        }

        transcript.write_commitments(&quotient_comms)?;

        Ok(())
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Fr, Self::Polynomial>],
        evals: &[Evaluation<M::Fr>],
        transcript: &mut impl TranscriptWrite<M::G1Affine, M::Fr>,
    ) -> Result<(), Error> {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        additive::batch_open::<_, Self>(pp, pp.num_vars(), polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        _: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Fr>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        transcript.read_commitments(num_polys).map(|comms| {
            comms
                .into_iter()
                .map(MultilinearKzgCommitment)
                .collect_vec()
        })
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Fr, Self::Polynomial>,
        eval: &M::Fr,
        transcript: &mut impl TranscriptRead<M::G1Affine, M::Fr>,
    ) -> Result<(), Error> {
        validate_input("verify", vp.num_vars(), [], [point])?;

        let quotients = transcript.read_commitments(point.len())?;

        let window_size = window_size(point.len());
        let window_table = window_table(window_size, vp.g2);
        let rhs = chain![
            [vp.g2.neg()],
            vp.ss(point.len())
                .iter()
                .cloned()
                .zip_eq(fixed_base_msm(window_size, &window_table, point))
                .map(|(s_i, x_i)| (s_i - x_i.into()).into()),
        ]
        .map_into()
        .collect_vec();
        let lhs = chain![
            [(comm.0.to_curve() - vp.g1 * eval).into()],
            quotients.iter().cloned()
        ]
        .collect_vec();
        M::pairings_product_is_identity(&lhs.iter().zip_eq(rhs.iter()).collect_vec())
            .then_some(())
            .ok_or_else(|| Error::InvalidPcsOpen("Invalid multilinear KZG open".to_string()))
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Fr, Self::Polynomial>],
        evals: &[Evaluation<M::Fr>],
        transcript: &mut impl TranscriptRead<M::G1Affine, M::Fr>,
    ) -> Result<(), Error> {
        let comms = comms.into_iter().collect_vec();
        additive::batch_verify::<_, Self>(vp, vp.num_vars(), comms, points, evals, transcript)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::kzg::MultilinearKzg,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::bn256::Bn256;

    type Pcs = MultilinearKzg<Bn256>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
