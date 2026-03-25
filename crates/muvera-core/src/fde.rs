use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::retriever::MuveraError;

pub type Vector = Vec<f32>;
pub type Document = Vec<Vector>;

pub fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
}

fn validate_vector_dimensions(vector: &[f32], expected: usize) -> Result<(), MuveraError> {
    if vector.len() != expected {
        return Err(MuveraError::DimensionMismatch {
            expected,
            found: vector.len(),
        });
    }
    Ok(())
}

pub(crate) fn validate_document(document: &[Vector], expected: usize) -> Result<(), MuveraError> {
    if document.is_empty() {
        return Err(MuveraError::EmptyDocument);
    }
    for vector in document {
        validate_vector_dimensions(vector, expected)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SimHash {
    dimensions: usize,
    k_sim: usize,
    hyperplanes: Vec<Vector>,
}

impl SimHash {
    pub fn new(dimensions: usize, k_sim: usize, seed: u64) -> Result<Self, MuveraError> {
        if k_sim == 0 || k_sim >= u32::BITS as usize {
            return Err(MuveraError::InvalidKSim(k_sim));
        }

        let normal = Normal::<f32>::new(0.0, 1.0).expect("valid gaussian parameters");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut hyperplanes = Vec::with_capacity(k_sim);
        for _ in 0..k_sim {
            let plane = (0..dimensions)
                .map(|_| normal.sample(&mut rng))
                .collect::<Vector>();
            hyperplanes.push(plane);
        }

        Ok(Self {
            dimensions,
            k_sim,
            hyperplanes,
        })
    }

    pub fn compute_hash(&self, vector: &[f32]) -> Result<u32, MuveraError> {
        validate_vector_dimensions(vector, self.dimensions)?;

        let mut hash = 0u32;
        for (idx, hyperplane) in self.hyperplanes.iter().enumerate().take(self.k_sim) {
            if dot_product(hyperplane, vector) >= 0.0 {
                hash |= 1u32 << idx;
            }
        }
        Ok(hash)
    }
}

#[derive(Debug, Clone)]
pub struct ExactChamferSimilarity {
    dimensions: usize,
}

impl ExactChamferSimilarity {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    pub fn compute_similarity(&self, lhs: &[Vector], rhs: &[Vector]) -> Result<f32, MuveraError> {
        validate_document(lhs, self.dimensions)?;
        validate_document(rhs, self.dimensions)?;

        let mut total = 0.0f32;
        for left in lhs {
            let best = rhs
                .iter()
                .map(|right| dot_product(left, right))
                .fold(f32::NEG_INFINITY, f32::max);
            total += best;
        }
        Ok(total)
    }
}

#[derive(Debug, Clone)]
pub struct FdeSimilarity {
    dimensions: usize,
    d_proj: usize,
    d_final: usize,
    d_fde: usize,
    b: usize,
    k_sim: usize,
    r_reps: usize,
    use_ams: bool,
    all_s: Vec<Vec<Vector>>,
    all_s_sparse: Vec<(Vec<usize>, Vec<i8>)>,
    all_simhash: Vec<SimHash>,
    countsketch_index: Vec<usize>,
    countsketch_sign: Vec<i8>,
}

impl FdeSimilarity {
    pub fn new(
        dimensions: usize,
        d_proj: usize,
        d_final: usize,
        k_sim: usize,
        r_reps: usize,
        seed: u64,
    ) -> Result<Self, MuveraError> {
        if d_proj == 0 {
            return Err(MuveraError::InvalidProjectionDimension);
        }
        if d_final == 0 {
            return Err(MuveraError::InvalidFinalDimension);
        }
        if r_reps == 0 {
            return Err(MuveraError::InvalidRepetitions);
        }
        if k_sim == 0 || k_sim >= usize::BITS as usize || k_sim >= u32::BITS as usize {
            return Err(MuveraError::InvalidKSim(k_sim));
        }

        let b = 1usize << k_sim;
        let mut all_simhash = Vec::with_capacity(r_reps);
        let mut all_s = Vec::with_capacity(r_reps);
        let mut all_s_sparse = Vec::with_capacity(r_reps);
        let scale = 1.0f32 / (d_proj as f32).sqrt();
        let sign_dist = Uniform::from(0u8..=1u8);
        let index_dist = Uniform::from(0..d_proj);

        for rep in 0..r_reps {
            all_simhash.push(SimHash::new(dimensions, k_sim, seed + rep as u64)?);

            let mut dense_rng = StdRng::seed_from_u64(seed + 103 + rep as u64);
            let dense = (0..d_proj)
                .map(|_| {
                    (0..dimensions)
                        .map(|_| if dense_rng.sample(sign_dist) == 1 { scale } else { -scale })
                        .collect::<Vector>()
                })
                .collect::<Vec<_>>();
            all_s.push(dense);

            let mut sparse_rng = StdRng::seed_from_u64(seed + 200 + rep as u64);
            let mut s_index = Vec::with_capacity(dimensions);
            let mut s_sign = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                s_index.push(sparse_rng.sample(index_dist));
                s_sign.push(if sparse_rng.sample(sign_dist) == 1 { 1 } else { -1 });
            }
            all_s_sparse.push((s_index, s_sign));
        }

        let d_fde = b * d_proj * r_reps;
        let mut cs_rng = StdRng::seed_from_u64(seed + 107);
        let final_index_dist = Uniform::from(0..d_final);
        let mut countsketch_index = Vec::with_capacity(d_fde);
        let mut countsketch_sign = Vec::with_capacity(d_fde);
        for _ in 0..d_fde {
            countsketch_index.push(cs_rng.sample(final_index_dist));
            countsketch_sign.push(if cs_rng.sample(sign_dist) == 1 { 1 } else { -1 });
        }

        Ok(Self {
            dimensions,
            d_proj,
            d_final,
            d_fde,
            b,
            k_sim,
            r_reps,
            use_ams: true,
            all_s,
            all_s_sparse,
            all_simhash,
            countsketch_index,
            countsketch_sign,
        })
    }

    pub fn d_fde(&self) -> usize {
        self.d_fde
    }

    pub fn embedding_dim(&self) -> usize {
        self.d_final
    }

    pub fn encode_document(&self, document: &[Vector]) -> Result<Vector, MuveraError> {
        validate_document(document, self.dimensions)?;

        let mut result = Vec::with_capacity(self.d_fde);
        for idx in 0..self.r_reps {
            let trial = self.encode_document_once(idx, document)?;
            result.extend(trial);
        }
        Ok(self.apply_countsketch(&result))
    }

    pub fn encode_query(&self, query: &[Vector]) -> Result<Vector, MuveraError> {
        validate_document(query, self.dimensions)?;

        let mut result = Vec::with_capacity(self.d_fde);
        for idx in 0..self.r_reps {
            let trial = self.encode_query_once(idx, query)?;
            result.extend(trial);
        }
        Ok(self.apply_countsketch(&result))
    }

    pub fn compute_similarity(&self, lhs: &[Vector], rhs: &[Vector]) -> Result<f32, MuveraError> {
        let left = self.encode_document(lhs)?;
        let right = self.encode_query(rhs)?;
        Ok(dot_product(&left, &right))
    }

    fn apply_countsketch(&self, vector: &[f32]) -> Vector {
        debug_assert_eq!(vector.len(), self.d_fde);

        let mut out = vec![0.0f32; self.d_final];
        for (idx, value) in vector.iter().enumerate() {
            let target = self.countsketch_index[idx];
            out[target] += self.countsketch_sign[idx] as f32 * value;
        }
        out
    }

    fn apply_ams(&self, vector: &[f32], rep_id: usize) -> Vector {
        debug_assert_eq!(vector.len(), self.dimensions);

        let (indices, signs) = &self.all_s_sparse[rep_id];
        let mut out = vec![0.0f32; self.d_proj];
        let scale = 1.0f32 / (self.d_proj as f32).sqrt();
        for idx in 0..self.dimensions {
            out[indices[idx]] += signs[idx] as f32 * vector[idx] * scale;
        }
        out
    }

    fn compute_proj_from_rep_idx(&self, rep_id: usize, vector: &[f32]) -> Vector {
        self.all_s[rep_id]
            .iter()
            .map(|row| dot_product(row, vector))
            .collect()
    }

    fn compute_hash_from_rep_idx(&self, rep_id: usize, vector: &[f32]) -> Result<u32, MuveraError> {
        self.all_simhash[rep_id].compute_hash(vector)
    }

    fn encode_document_once(&self, rep_id: usize, document: &[Vector]) -> Result<Vector, MuveraError> {
        let mut grouped = vec![vec![0.0f32; self.dimensions]; self.b];
        let mut bucket_counts = vec![0usize; self.b];

        for vector in document {
            let hash_value = self.compute_hash_from_rep_idx(rep_id, vector)? as usize;
            bucket_counts[hash_value] += 1;
            let count = bucket_counts[hash_value] as f32;
            for (dst, src) in grouped[hash_value].iter_mut().zip(vector.iter()) {
                *dst = ((*dst * (count - 1.0)) + src) / count;
            }
        }

        let mut phi = Vec::with_capacity(self.b * self.d_proj);
        for bucket in grouped {
            let projection = if self.use_ams {
                self.apply_ams(&bucket, rep_id)
            } else {
                self.compute_proj_from_rep_idx(rep_id, &bucket)
            };
            phi.extend(projection);
        }
        Ok(phi)
    }

    fn encode_query_once(&self, rep_id: usize, query: &[Vector]) -> Result<Vector, MuveraError> {
        let mut grouped = vec![vec![0.0f32; self.dimensions]; self.b];

        for vector in query {
            let hash_value = self.compute_hash_from_rep_idx(rep_id, vector)? as usize;
            for (dst, src) in grouped[hash_value].iter_mut().zip(vector.iter()) {
                *dst += src;
            }
        }

        let mut phi = Vec::with_capacity(self.b * self.d_proj);
        for bucket in grouped {
            let projection = if self.use_ams {
                self.apply_ams(&bucket, rep_id)
            } else {
                self.compute_proj_from_rep_idx(rep_id, &bucket)
            };
            phi.extend(projection);
        }
        Ok(phi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, -5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 12.0).abs() < 1e-6);
    }

    #[test]
    fn exact_chamfer_similarity_simple() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![1.0, -2.0, 3.0]];
        let b = vec![vec![4.0, 5.0, 6.0], vec![4.0, -5.0, 6.0]];
        let similarity = ExactChamferSimilarity::new(3)
            .compute_similarity(&a, &b)
            .unwrap();
        assert!((similarity - 64.0).abs() < 1e-6);
    }

    #[test]
    fn simhash_basic() {
        let simhash = SimHash::new(3, 10, 42).unwrap();
        let hash = simhash.compute_hash(&[1.0, 0.0, -1.0]).unwrap();
        assert!(hash < (1 << 10));
    }

    #[test]
    fn fde_basic() {
        let fde = FdeSimilarity::new(3, 128, 10_240, 10, 5, 42).unwrap();
        let a = vec![vec![1.0, 2.0, 3.0], vec![1.0, -2.0, 3.0]];
        let b = vec![vec![4.0, 5.0, 6.0], vec![4.0, -5.0, 6.0]];
        let encoded = fde.encode_document(&a).unwrap();
        assert_eq!(encoded.len(), 10_240);
        let similarity = fde.compute_similarity(&a, &b).unwrap();
        assert!(similarity.is_finite());
    }
}
