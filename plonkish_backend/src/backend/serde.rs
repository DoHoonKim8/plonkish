use std::io;

use halo2_curves::{ff::{FromUniformBytes, PrimeField}, serde::SerdeObject};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    backend::{
        hyperplonk::{preprocessor::compose, HyperPlonk, HyperPlonkProverParam, HyperPlonkVerifierParam},
        PlonkishCircuit,
    },
    frontend::halo2::{CircuitExt, Halo2Circuit},
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::{read_polynomial_vec, write_polynomial_slice, MultilinearPolynomial},
    util::{expression::Expression, SerdeFormat, SerdePrimeField},
};

fn write_expression<W: io::Write, F: Serialize + DeserializeOwned>(writer: &mut W, expression: &Expression<F>) -> io::Result<()> {
    // Serialize the expression to bytes
    let expr_bytes = bincode::serialize(expression)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    writer.write_all(&(expr_bytes.len() as u32).to_le_bytes())?;
    // Write the serialized expression
    writer.write_all(&expr_bytes)?;
    Ok(())
}

fn read_expression<R: io::Read, F: Serialize + DeserializeOwned>(reader: &mut R) -> io::Result<Expression<F>> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let expr_len = u32::from_le_bytes(len_bytes) as usize;
    let mut expr_bytes = vec![0u8; expr_len];
    reader.read_exact(&mut expr_bytes)?;
    let expression: Expression<F> = bincode::deserialize(&expr_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(expression)
}

#[derive(Clone, Debug)]
pub(crate) struct HyperPlonkProvingKey<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) vk: HyperPlonkVerifyingKey<F, Pcs>,
    pub(crate) preprocess_polys: Vec<MultilinearPolynomial<F>>,
    pub(crate) permutation_polys: Vec<(usize, MultilinearPolynomial<F>)>,
}

impl<F, Pcs> HyperPlonkProvingKey<F, Pcs>
where
    F: PrimeField + SerdePrimeField + FromUniformBytes<64> + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F>,
    Pcs::Commitment: SerdeObject,
{
    fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write vk
        self.vk.write(writer)?;
        // Write preprocess_polys
        write_polynomial_slice(&self.preprocess_polys, writer, SerdeFormat::RawBytes)?;
        writer.write_all(&(self.permutation_polys.len() as u32).to_le_bytes())?;
        // Write permutation_polys
        for (idx, poly) in &self.permutation_polys {
            writer.write_all(&(*idx as u32).to_le_bytes())?;
            poly.write(writer, SerdeFormat::RawBytes)?;
        }
        Ok(())
    }

    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        // Read vk
        let vk = HyperPlonkVerifyingKey::read(reader)?;
        // Read preprocess_polys
        let preprocess_polys = read_polynomial_vec(reader, SerdeFormat::RawBytes)?;
        // Read permutation_comms
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let perm_len = u32::from_le_bytes(len_bytes) as usize;
        let mut permutation_polys = Vec::with_capacity(perm_len);
        for _ in 0..perm_len {
            let mut poly_idx_bytes = [0u8; 4];
            reader.read_exact(&mut poly_idx_bytes)?;
            let poly_idx = u32::from_le_bytes(poly_idx_bytes) as usize;
            let poly = MultilinearPolynomial::read(reader, SerdeFormat::RawBytes)?;
            permutation_polys.push((poly_idx, poly));
        }
        Ok(Self {
            vk,
            preprocess_polys,
            permutation_polys,
        })
    }
}

impl<F, Pcs> HyperPlonkProverParam<F, Pcs>
where
    F: PrimeField + SerdePrimeField + FromUniformBytes<64> + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F>,
    Pcs::Commitment: SerdeObject,
{
    fn to_pk(&self) -> HyperPlonkProvingKey<F, Pcs> {
        let permutation_polys_indices = self
            .permutation_polys
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>();
        HyperPlonkProvingKey::<F, Pcs> {
            vk: HyperPlonkVerifyingKey {
                num_vars: self.num_vars,
                preprocess_comms: self.preprocess_comms.clone(),
                permutation_comms: self
                    .permutation_comms
                    .iter()
                    .zip(permutation_polys_indices.iter())
                    .map(|(comm, idx)| (*idx, comm.clone()))
                    .collect(),
                num_instances: self.num_instances.clone(),
                num_witness_polys: self.num_witness_polys.clone(),
                num_challenges: self.num_challenges.clone(),
                num_permutation_z_polys: self.num_permutation_z_polys,
                expression: self.expression.clone(),
            },
            preprocess_polys: self.preprocess_polys.clone(),
            permutation_polys: self.permutation_polys.clone(),
        }
    }

    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.to_pk().write(writer)?;

        // Write lookups
        writer.write_all(&(self.lookups.len() as u32).to_le_bytes())?;
        for lookup in &self.lookups {
            writer.write_all(&(lookup.len() as u32).to_le_bytes())?;
            for (input, table) in lookup {
                write_expression(writer, input)?;
                write_expression(writer, table)?;
            }
        }
        Ok(())
    }

    pub fn read<R: io::Read>(
        reader: &mut R,
        pcs: Pcs::ProverParam,
    ) -> io::Result<Self> {
        let pk = HyperPlonkProvingKey::<F, Pcs>::read(reader)?;

        let lookups = {
            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut lookups = Vec::with_capacity(len);
            for _ in 0..len {
                let mut lookup_len_bytes = [0u8; 4];
                reader.read_exact(&mut lookup_len_bytes)?;
                let lookup_len = u32::from_le_bytes(lookup_len_bytes) as usize;
                let mut lookup = Vec::with_capacity(lookup_len);
                for _ in 0..lookup_len {
                    let input = read_expression(reader)?;
                    let table = read_expression(reader)?;
                    lookup.push((input, table));
                }
                lookups.push(lookup);
            }
            lookups
        };

        Ok(Self {
            pcs,
            num_vars: pk.vk.num_vars,
            preprocess_polys: pk.preprocess_polys,
            preprocess_comms: pk.vk.preprocess_comms,
            permutation_comms: pk
                .vk
                .permutation_comms
                .iter()
                .map(|(_, comm)| comm.clone())
                .collect(),
            permutation_polys: pk.permutation_polys,
            num_permutation_z_polys: pk.vk.num_permutation_z_polys,
            expression: pk.vk.expression,
            num_witness_polys: pk.vk.num_witness_polys,
            num_instances: pk.vk.num_instances,
            num_challenges: pk.vk.num_challenges,
            lookups,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.write(&mut bytes).unwrap();
        bytes
    }
}

/// Stores the commitments of the polynomials that are also known to the verifier
#[derive(Clone, Debug)]
pub struct HyperPlonkVerifyingKey<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) num_vars: usize,
    // commitments to fixed polynomials and selectors
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    // commitments to sigma polynomials
    pub(crate) permutation_comms: Vec<(usize, Pcs::Commitment)>,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) expression: Expression<F>,
}

impl<F, Pcs> HyperPlonkVerifyingKey<F, Pcs>
where
    F: PrimeField + SerdePrimeField + FromUniformBytes<64> + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F>,
    Pcs::Commitment: SerdeObject,
{
    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write num_vars
        writer.write_all(&(self.num_vars as u32).to_be_bytes())?;
        // Write preprocess_comms
        writer.write_all(&(self.preprocess_comms.len() as u32).to_le_bytes())?;
        for comm in &self.preprocess_comms {
            comm.write_raw(writer)?;
        }
        // Write permutation_comms
        writer.write_all(&(self.permutation_comms.len() as u32).to_le_bytes())?;
        for (poly_idx, comm) in &self.permutation_comms {
            writer.write_all(&(*poly_idx as u32).to_le_bytes())?;
            comm.write_raw(writer)?;
        }
        // write num_instances
        writer.write_all(&(self.num_instances.len() as u32).to_le_bytes())?;
        for num in &self.num_instances {
            writer.write_all(&(*num as u32).to_le_bytes())?;
        }
        // write num_witness_polys
        writer.write_all(&(self.num_witness_polys.len() as u32).to_le_bytes())?;
        for num in &self.num_witness_polys {
            writer.write_all(&(*num as u32).to_le_bytes())?;
        }
        // write num_challenges
        writer.write_all(&(self.num_challenges.len() as u32).to_le_bytes())?;
        for num in &self.num_challenges {
            writer.write_all(&(*num as u32).to_le_bytes())?;
        }
        // write num_permutation_z_polys
        writer.write_all(&(self.num_permutation_z_polys as u32).to_le_bytes())?;
        // write expression
        write_expression(writer, &self.expression)?;
        Ok(())
    }

    pub fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let mut num_vars_bytes = [0u8; 4];
        reader.read_exact(&mut num_vars_bytes)?;
        let stored_num_vars = u32::from_be_bytes(num_vars_bytes) as usize;
        // Read preprocess_comms
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let preprocess_len = u32::from_le_bytes(len_bytes) as usize;
        let mut preprocess_comms = Vec::with_capacity(preprocess_len);
        for _ in 0..preprocess_len {
            let comm = Pcs::Commitment::read_raw(reader)?;
            preprocess_comms.push(comm);
        }
        // Read permutation_comms
        reader.read_exact(&mut len_bytes)?;
        let perm_len = u32::from_le_bytes(len_bytes) as usize;
        let mut permutation_comms = Vec::with_capacity(perm_len);
        for _ in 0..perm_len {
            let mut poly_idx_bytes = [0u8; 4];
            reader.read_exact(&mut poly_idx_bytes)?;
            let poly_idx = u32::from_le_bytes(poly_idx_bytes) as usize;
            let comm = Pcs::Commitment::read_raw(reader)?;
            permutation_comms.push((poly_idx, comm));
        }
        // read num_instances
        let num_instances = {
            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut nums = Vec::with_capacity(len);
            for _ in 0..len {
                let mut num_bytes = [0u8; 4];
                reader.read_exact(&mut num_bytes)?;
                nums.push(u32::from_le_bytes(num_bytes) as usize);
            }
            nums
        };
        // read num_witness_polys
        let num_witness_polys = {
            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut nums = Vec::with_capacity(len);
            for _ in 0..len {
                let mut num_bytes = [0u8; 4];
                reader.read_exact(&mut num_bytes)?;
                nums.push(u32::from_le_bytes(num_bytes) as usize);
            }
            nums
        };

        // read num_challenges
        let num_challenges = {
            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut nums = Vec::with_capacity(len);
            for _ in 0..len {
                let mut num_bytes = [0u8; 4];
                reader.read_exact(&mut num_bytes)?;
                nums.push(u32::from_le_bytes(num_bytes) as usize);
            }
            nums
        };

        // read num_permutation_z_polys
        let num_permutation_z_polys = {
            let mut num_bytes = [0u8; 4];
            reader.read_exact(&mut num_bytes)?;
            u32::from_le_bytes(num_bytes) as usize
        };

        // read expression
        let expression = read_expression(reader)?;

        Ok(Self {
            num_vars: stored_num_vars,
            preprocess_comms,
            permutation_comms,
            num_instances,
            num_witness_polys,
            num_challenges,
            num_permutation_z_polys,
            expression,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.write(&mut bytes).unwrap();
        bytes
    }
}

impl<F, Pcs> HyperPlonkVerifierParam<F, Pcs>
where
    F: PrimeField + SerdePrimeField + FromUniformBytes<64> + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F>,
    Pcs::Commitment: SerdeObject,
{
    fn to_vk(&self) -> HyperPlonkVerifyingKey<F, Pcs> {
        HyperPlonkVerifyingKey {
            num_vars: self.num_vars,
            preprocess_comms: self.preprocess_comms.clone(),
            permutation_comms: self.permutation_comms.clone(),
            num_instances: self.num_instances.clone(),
            num_witness_polys: self.num_witness_polys.clone(),
            num_challenges: self.num_challenges.clone(),
            num_permutation_z_polys: self.num_permutation_z_polys,
            expression: self.expression.clone(),
        }
    }

    pub fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.to_vk().write(writer)
    }

    pub fn read<R: io::Read, ConcreteCircuit: CircuitExt<F>>(
        reader: &mut R,
        circuit: ConcreteCircuit,
        circuit_params: ConcreteCircuit::Params,
        pcs: Pcs::VerifierParam,
    ) -> io::Result<Self> {
        let vk = HyperPlonkVerifyingKey::<F, Pcs>::read(reader)?;

        // re-generate circuit-specific information
        let circuit = Halo2Circuit::new_with_params::<HyperPlonk<Pcs>>(
            vk.num_vars,
            circuit,
            circuit_params,
        );
        let circuit_info = circuit.circuit_info().map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("circuit_info error: {e:?}"))
        })?;

        let (num_permutation_z_polys, expression) = compose(&circuit_info);

        Ok(Self {
            pcs,
            num_vars: vk.num_vars,
            preprocess_comms: vk.preprocess_comms,
            permutation_comms: vk.permutation_comms.clone(),
            num_permutation_z_polys,
            expression,
            num_witness_polys: circuit_info.num_witness_polys,
            num_instances: circuit_info.num_instances,
            num_challenges: circuit_info.num_challenges,
            num_lookups: circuit_info.lookups.len(),
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_vk().to_bytes()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::{
        backend::{
            hyperplonk::{
                util::rand_vanilla_plonk_circuit,
                HyperPlonk,
            },
            PlonkishBackend,
        },
        pcs::multilinear::MultilinearKzg,
        util::{
            expression::rotate::BinaryField,
            test::seeded_std_rng,
        },
    };
    use halo2_curves::bn256::{Bn256, Fr};



    type TestPcs = MultilinearKzg<Bn256>;
    type TestBackend = HyperPlonk<TestPcs>;

    #[test]
    fn test_hyperplonk_prover_param_serde_roundtrip() {
        let num_vars = 4;
        let mut rng = seeded_std_rng();
        // Create a test circuit and get circuit info
        let (circuit_info, _) = rand_vanilla_plonk_circuit::<Fr, BinaryField>(num_vars, seeded_std_rng(), seeded_std_rng());

        // Setup PCS parameters
        let pcs_param = TestBackend::setup(&circuit_info, &mut rng).unwrap();
        // Generate prover and verifier parameters
        let (prover_param, _verifier_param): (HyperPlonkProverParam<Fr, TestPcs>, HyperPlonkVerifierParam<Fr, TestPcs>) = TestBackend::preprocess(&pcs_param, &circuit_info).unwrap();

        // Test serialization roundtrip
        let mut buffer = Vec::new();
        prover_param.write(&mut buffer).expect("Failed to write prover param");

        let mut cursor = Cursor::new(&buffer);
        let deserialized_param: HyperPlonkProverParam<Fr, TestPcs> = HyperPlonkProverParam::read(
            &mut cursor,
            prover_param.pcs.clone()
        ).expect("Failed to read prover param");
        // Verify all fields are identical
        assert_eq!(prover_param.num_vars, deserialized_param.num_vars);
        assert_eq!(prover_param.num_instances, deserialized_param.num_instances);
        assert_eq!(prover_param.num_witness_polys, deserialized_param.num_witness_polys);
        assert_eq!(prover_param.num_challenges, deserialized_param.num_challenges);
        assert_eq!(prover_param.lookups, deserialized_param.lookups);
        assert_eq!(prover_param.num_permutation_z_polys, deserialized_param.num_permutation_z_polys);
        // Verify expression serialization
        assert_eq!(prover_param.expression, deserialized_param.expression);
        // Verify polynomial data
        assert_eq!(prover_param.preprocess_polys.len(), deserialized_param.preprocess_polys.len());
        for (orig, deser) in prover_param.preprocess_polys.iter().zip(deserialized_param.preprocess_polys.iter()) {
            assert_eq!(orig, deser, "Preprocess polynomial mismatch");
        }
        assert_eq!(prover_param.permutation_polys.len(), deserialized_param.permutation_polys.len());
        for (orig, deser) in prover_param.permutation_polys.iter().zip(deserialized_param.permutation_polys.iter()) {
            assert_eq!(orig.0, deser.0, "Permutation polynomial index mismatch");
            assert_eq!(orig.1, deser.1, "Permutation polynomial data mismatch");
        }
        // Verify commitment data
        assert_eq!(prover_param.preprocess_comms.len(), deserialized_param.preprocess_comms.len());
        for (orig, deser) in prover_param.preprocess_comms.iter().zip(deserialized_param.preprocess_comms.iter()) {
            // Note: We can't directly compare commitments as they might have different internal representations
            // but we can verify they serialize to the same bytes
            let orig_bytes = {
                let mut buf = Vec::new();
                orig.write_raw(&mut buf).unwrap();
                buf
            };
            let deser_bytes = {
                let mut buf = Vec::new();
                deser.write_raw(&mut buf).unwrap();
                buf
            };
            assert_eq!(orig_bytes, deser_bytes, "Preprocess commitment mismatch");
        }
        
        assert_eq!(prover_param.permutation_comms.len(), deserialized_param.permutation_comms.len());
        for (orig, deser) in prover_param.permutation_comms.iter().zip(deserialized_param.permutation_comms.iter()) {
            let orig_bytes = {
                let mut buf = Vec::new();
                orig.write_raw(&mut buf).unwrap();
                buf
            };
            let deser_bytes = {
                let mut buf = Vec::new();
                deser.write_raw(&mut buf).unwrap();
                buf
            };
            assert_eq!(orig_bytes, deser_bytes, "Permutation commitment mismatch");
        }
    }
}
