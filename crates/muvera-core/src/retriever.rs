use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

use diskann::graph::{self, DiskANNIndex};
use diskann::provider::DefaultContext;
use diskann::graph::search_output_buffer;
use diskann_providers::index::diskann_async;
use diskann_providers::model::graph::provider::async_::common::{FullPrecision, NoDeletes};
use diskann_providers::model::graph::provider::async_::inmem::{
    DefaultProviderParameters, FullPrecisionProvider, SetStartPoints,
};
use diskann_vector::distance::Metric;
use thiserror::Error;
use tokio::runtime::Runtime;

use crate::fde::{dot_product, validate_document, Document, ExactChamferSimilarity, FdeSimilarity, Vector};

// ---------- errors ----------

#[derive(Debug, Error)]
pub enum MuveraError {
    #[error("vector dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },
    #[error("documents must contain at least one vector")]
    EmptyDocument,
    #[error("dataset size does not match doc id count")]
    DatasetDocIdLengthMismatch,
    #[error("doc id 0 is reserved and cannot be used")]
    InvalidDocId,
    #[error("duplicate doc id {0}")]
    DuplicateDocId(u32),
    #[error("retriever is not initialized")]
    NotInitialized,
    #[error("top_k must be greater than zero")]
    InvalidTopK,
    #[error("max_points exceeded: capacity {capacity}, attempted {attempted}")]
    MaxPointsExceeded { capacity: usize, attempted: usize },
    #[error("k_sim must be between 1 and 31, found {0}")]
    InvalidKSim(usize),
    #[error("d_proj must be greater than zero")]
    InvalidProjectionDimension,
    #[error("d_final must be greater than zero")]
    InvalidFinalDimension,
    #[error("r_reps must be greater than zero")]
    InvalidRepetitions,
    #[error("DiskANN error: {0}")]
    DiskAnn(String),
}

impl From<diskann::ANNError> for MuveraError {
    fn from(e: diskann::ANNError) -> Self {
        MuveraError::DiskAnn(e.to_string())
    }
}

// ---------- shared trait ----------

pub trait Retriever {
    fn index_dataset(&mut self, dataset: &[Document], doc_ids: &[u32]) -> Result<(), MuveraError>;
    fn add_document(&mut self, document: Document, doc_id: u32) -> Result<(), MuveraError>;
    fn get_top_k(&self, query: &[Vector], top_k: usize) -> Result<Vec<u32>, MuveraError>;
}

// ---------- ExactChamferRetriever ----------

#[derive(Debug, Clone)]
pub struct ExactChamferRetriever {
    dimensions: usize,
    max_points: usize,
    initialized: bool,
    similarity: ExactChamferSimilarity,
    dataset: Vec<Document>,
    doc_ids: Vec<u32>,
    doc_id_set: HashSet<u32>,
}

impl ExactChamferRetriever {
    pub fn new(dimensions: usize, max_points: usize) -> Self {
        Self {
            dimensions,
            max_points,
            initialized: false,
            similarity: ExactChamferSimilarity::new(dimensions),
            dataset: Vec::new(),
            doc_ids: Vec::new(),
            doc_id_set: HashSet::new(),
        }
    }
}

impl Retriever for ExactChamferRetriever {
    fn index_dataset(&mut self, dataset: &[Document], doc_ids: &[u32]) -> Result<(), MuveraError> {
        validate_dataset(self.dimensions, self.max_points, dataset, doc_ids)?;
        self.dataset = dataset.to_vec();
        self.doc_ids = doc_ids.to_vec();
        self.doc_id_set = doc_ids.iter().copied().collect();
        self.initialized = true;
        Ok(())
    }

    fn add_document(&mut self, document: Document, doc_id: u32) -> Result<(), MuveraError> {
        ensure_initialized(self.initialized)?;
        validate_document_id(doc_id, &self.doc_id_set)?;
        validate_document(&document, self.dimensions)?;
        ensure_capacity(self.max_points, self.dataset.len() + 1)?;
        self.doc_id_set.insert(doc_id);
        self.doc_ids.push(doc_id);
        self.dataset.push(document);
        Ok(())
    }

    fn get_top_k(&self, query: &[Vector], top_k: usize) -> Result<Vec<u32>, MuveraError> {
        ensure_initialized(self.initialized)?;
        if top_k == 0 { return Err(MuveraError::InvalidTopK); }
        validate_document(query, self.dimensions)?;
        Ok(rank_by_score(
            self.doc_ids.iter().copied(),
            self.dataset.iter().map(|doc| {
                self.similarity.compute_similarity(doc, query).expect("validated")
            }),
            top_k,
        ))
    }
}

// ---------- MuveraRetriever (exact FDE) ----------

#[derive(Debug, Clone)]
pub struct MuveraRetriever {
    dimensions: usize,
    max_points: usize,
    initialized: bool,
    doc_id_set: HashSet<u32>,
    doc_ids: Vec<u32>,
    fde_engine: FdeSimilarity,
    embeddings: Vec<Vector>,
}

impl MuveraRetriever {
    pub fn new(
        dimensions: usize, max_points: usize,
        d_proj: usize, d_final: usize, k_sim: usize, r_reps: usize, seed: u64,
    ) -> Result<Self, MuveraError> {
        Ok(Self {
            dimensions, max_points,
            initialized: false,
            doc_id_set: HashSet::new(),
            doc_ids: Vec::new(),
            fde_engine: FdeSimilarity::new(dimensions, d_proj, d_final, k_sim, r_reps, seed)?,
            embeddings: Vec::new(),
        })
    }

    pub fn embedding_dim(&self) -> usize { self.fde_engine.embedding_dim() }
}

impl Retriever for MuveraRetriever {
    fn index_dataset(&mut self, dataset: &[Document], doc_ids: &[u32]) -> Result<(), MuveraError> {
        validate_dataset(self.dimensions, self.max_points, dataset, doc_ids)?;
        self.embeddings = self.fde_engine.batch_encode_documents(dataset)?;
        self.doc_ids = doc_ids.to_vec();
        self.doc_id_set = doc_ids.iter().copied().collect();
        self.initialized = true;
        Ok(())
    }

    fn add_document(&mut self, document: Document, doc_id: u32) -> Result<(), MuveraError> {
        ensure_initialized(self.initialized)?;
        validate_document_id(doc_id, &self.doc_id_set)?;
        validate_document(&document, self.dimensions)?;
        ensure_capacity(self.max_points, self.embeddings.len() + 1)?;
        let enc = self.fde_engine.encode_document(&document)?;
        self.doc_id_set.insert(doc_id);
        self.doc_ids.push(doc_id);
        self.embeddings.push(enc);
        Ok(())
    }

    fn get_top_k(&self, query: &[Vector], top_k: usize) -> Result<Vec<u32>, MuveraError> {
        ensure_initialized(self.initialized)?;
        if top_k == 0 { return Err(MuveraError::InvalidTopK); }
        validate_document(query, self.dimensions)?;
        let qe = self.fde_engine.encode_query(query)?;
        Ok(rank_by_score(
            self.doc_ids.iter().copied(),
            self.embeddings.iter().map(|e| dot_product(e, &qe)),
            top_k,
        ))
    }
}

// ---------- DiskAnnRetriever (FDE + DiskANN graph) ----------

type FpIndex = Arc<DiskANNIndex<FullPrecisionProvider<f32>>>;

pub struct DiskAnnRetriever {
    dimensions: usize,
    max_points: usize,
    initialized: bool,
    doc_id_set: HashSet<u32>,
    doc_ids: Vec<u32>,
    fde_engine: FdeSimilarity,
    embeddings: Vec<Vector>,
    graph: Option<FpIndex>,
    runtime: Runtime,
    max_degree: usize,
    l_build: usize,
    search_l: usize,
}

impl DiskAnnRetriever {
    /// Create a new approximate retriever backed by a DiskANN in-memory graph.
    ///
    /// * `max_degree` – DiskANN graph out-degree R (e.g. 32)
    /// * `l_build`    – build search-list size (e.g. 50)
    /// * `search_l`   – search-time list size (≥ top_k, e.g. 64)
    pub fn new(
        dimensions: usize, max_points: usize,
        d_proj: usize, d_final: usize, k_sim: usize, r_reps: usize, seed: u64,
        max_degree: usize, l_build: usize, search_l: usize,
    ) -> Result<Self, MuveraError> {
        Ok(Self {
            dimensions, max_points,
            initialized: false,
            doc_id_set: HashSet::new(),
            doc_ids: Vec::new(),
            fde_engine: FdeSimilarity::new(dimensions, d_proj, d_final, k_sim, r_reps, seed)?,
            embeddings: Vec::new(),
            graph: None,
            runtime: Runtime::new().map_err(|e| MuveraError::DiskAnn(e.to_string()))?,
            max_degree,
            l_build,
            search_l,
        })
    }

    pub fn embedding_dim(&self) -> usize { self.fde_engine.embedding_dim() }
    pub fn search_l(&self) -> usize { self.search_l }

    fn rebuild_graph(&mut self) -> Result<(), MuveraError> {
        let n = self.embeddings.len();
        if n == 0 { self.graph = None; return Ok(()); }

        let emb_dim = self.fde_engine.embedding_dim();
        // Clamp against max_points (full capacity), not n (current count),
        // so the graph degree is consistent whether built from 1 doc or 1000.
        let max_degree = self.max_degree.min(self.max_points.saturating_sub(1)).max(1);
        let l_build = self.l_build.max(max_degree * 2);

        let config = graph::config::Builder::new(
            max_degree,
            graph::config::MaxDegree::same(),
            l_build,
            Metric::InnerProduct.into(),
        )
        .build()
        .map_err(|e| MuveraError::DiskAnn(e.to_string()))?;

        // Use self.max_points (not n) so the graph pre-allocates its full
        // capacity upfront. Without this, incremental add_document calls
        // produce IDs beyond the initial allocated range and DiskANN panics
        // with "Vector id is out of boundary".
        let params = DefaultProviderParameters::simple(
            self.max_points,
            emb_dim,
            Metric::InnerProduct,
            max_degree as u32,
        );

        let index = diskann_async::new_index::<f32, _>(config, params, NoDeletes)
            .map_err(MuveraError::from)?;

        index
            .provider()
            .set_start_points(std::iter::once(self.embeddings[0].as_slice()))
            .map_err(MuveraError::from)?;

        let embeddings = self.embeddings.clone();
        self.runtime.block_on(async {
            for (i, emb) in embeddings.iter().enumerate() {
                index
                    .insert(FullPrecision, &DefaultContext, &(i as u32), emb.as_slice())
                    .await
                    .map_err(MuveraError::from)?;
            }
            Ok::<_, MuveraError>(())
        })?;

        self.graph = Some(index);
        Ok(())
    }
}

impl Retriever for DiskAnnRetriever {
    fn index_dataset(&mut self, dataset: &[Document], doc_ids: &[u32]) -> Result<(), MuveraError> {
        validate_dataset(self.dimensions, self.max_points, dataset, doc_ids)?;
        self.embeddings = self.fde_engine.batch_encode_documents(dataset)?;
        self.doc_ids = doc_ids.to_vec();
        self.doc_id_set = doc_ids.iter().copied().collect();
        self.initialized = true;
        self.rebuild_graph()
    }

    fn add_document(&mut self, document: Document, doc_id: u32) -> Result<(), MuveraError> {
        ensure_initialized(self.initialized)?;
        validate_document_id(doc_id, &self.doc_id_set)?;
        validate_document(&document, self.dimensions)?;
        ensure_capacity(self.max_points, self.embeddings.len() + 1)?;
        let enc = self.fde_engine.encode_document(&document)?;
        let internal_id = self.embeddings.len() as u32;
        self.doc_id_set.insert(doc_id);
        self.doc_ids.push(doc_id);
        self.embeddings.push(enc.clone());
        // Incrementally insert into the existing graph rather than rebuilding.
        // Falls back to a full rebuild only if no graph exists yet (empty initial dataset).
        match self.graph.clone() {
            Some(graph) => self.runtime.block_on(
                graph.insert(FullPrecision, &DefaultContext, &internal_id, enc.as_slice())
            ).map_err(MuveraError::from),
            None => self.rebuild_graph(),
        }
    }

    fn get_top_k(&self, query: &[Vector], top_k: usize) -> Result<Vec<u32>, MuveraError> {
        ensure_initialized(self.initialized)?;
        if top_k == 0 { return Err(MuveraError::InvalidTopK); }
        validate_document(query, self.dimensions)?;

        let qe = self.fde_engine.encode_query(query)?;
        let graph_opt = self.graph.as_deref();
        let graph = match graph_opt {
            Some(graph) => graph,
            None => return Ok(Vec::new()),
        };

        let search_l = self.search_l.max(top_k);
        let params = diskann::graph::search::Knn::new_default(top_k, search_l)
            .map_err(|e| MuveraError::DiskAnn(e.to_string()))?;
        let mut ids = vec![0_u32; top_k];
        let mut distances = vec![0.0_f32; top_k];
        let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        self.runtime.block_on(graph.search(
            params,
            &FullPrecision,
            &DefaultContext,
            qe.as_slice(),
            &mut output,
        ))
        .map_err(|e| MuveraError::DiskAnn(e.to_string()))?;

        Ok(ids.iter()
            .filter_map(|id| self.doc_ids.get(*id as usize).copied())
            .collect())
    }
}

// ---------- helpers ----------

fn ensure_initialized(initialized: bool) -> Result<(), MuveraError> {
    if initialized { Ok(()) } else { Err(MuveraError::NotInitialized) }
}

fn ensure_capacity(capacity: usize, attempted: usize) -> Result<(), MuveraError> {
    if attempted > capacity {
        Err(MuveraError::MaxPointsExceeded { capacity, attempted })
    } else {
        Ok(())
    }
}

fn validate_document_id(doc_id: u32, seen: &HashSet<u32>) -> Result<(), MuveraError> {
    if doc_id == 0 { return Err(MuveraError::InvalidDocId); }
    if seen.contains(&doc_id) { return Err(MuveraError::DuplicateDocId(doc_id)); }
    Ok(())
}

fn validate_dataset(
    dimensions: usize, max_points: usize,
    dataset: &[Document], doc_ids: &[u32],
) -> Result<(), MuveraError> {
    if dataset.len() != doc_ids.len() { return Err(MuveraError::DatasetDocIdLengthMismatch); }
    ensure_capacity(max_points, dataset.len())?;
    let mut seen = HashSet::with_capacity(doc_ids.len());
    for (doc, id) in dataset.iter().zip(doc_ids.iter().copied()) {
        validate_document_id(id, &seen)?;
        validate_document(doc, dimensions)?;
        seen.insert(id);
    }
    Ok(())
}

fn rank_by_score<I, S>(doc_ids: I, scores: S, top_k: usize) -> Vec<u32>
where
    I: Iterator<Item = u32>,
    S: Iterator<Item = f32>,
{
    let mut scored: Vec<(u32, f32)> = doc_ids.zip(scores).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().take(top_k).map(|(id, _)| id).collect()
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_chamfer_retriever_simple() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![1.0, -2.0, 3.0]];
        let b = vec![vec![4.0, 5.0, 6.0], vec![4.0, -5.0, 6.0]];
        let mut r = ExactChamferRetriever::new(3, 500);
        r.index_dataset(&[a.clone(), b], &[1, 2]).unwrap();
        assert_eq!(r.get_top_k(&a, 1).unwrap(), vec![2]);
    }

    #[test]
    fn muvera_retriever_basic() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![1.0, -2.0, 3.0]];
        let mut r = MuveraRetriever::new(3, 500, 128, 10_240, 10, 5, 42).unwrap();
        r.index_dataset(&[a.clone()], &[1]).unwrap();
        assert_eq!(r.get_top_k(&a, 1).unwrap(), vec![1]);
    }

    #[test]
    fn muvera_retriever_large_100d_top50() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let dim = 100usize;
        let n = 500usize;
        let mut r = MuveraRetriever::new(dim, n, 64, 4096, 7, 10, 42).unwrap();
        let mut rng = StdRng::seed_from_u64(12345);
        let dataset: Vec<_> = (0..n).map(|_|
            (0..4).map(|_| (0..dim).map(|_| rng.gen_range(-3.0f32..3.0f32)).collect::<Vec<_>>()).collect::<Vec<_>>()
        ).collect();
        let doc_ids: Vec<u32> = (1..=n as u32).collect();
        r.index_dataset(&dataset, &doc_ids).unwrap();
        let result = r.get_top_k(&dataset[100], 50).unwrap();
        assert_eq!(result.len(), 50);
        assert!(result.contains(&101u32));
    }

    #[test]
    fn diskann_retriever_basic() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![1.0, -2.0, 3.0]];
        let mut r = DiskAnnRetriever::new(3, 500, 128, 1024, 5, 3, 42, 4, 20, 30).unwrap();
        r.index_dataset(&[a.clone()], &[1]).unwrap();
        assert_eq!(r.get_top_k(&a, 1).unwrap(), vec![1]);
    }

    #[test]
    fn diskann_retriever_multi_doc() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let dim = 32usize;
        let n = 50usize;
        let mut r = DiskAnnRetriever::new(dim, n, 32, 512, 5, 4, 42, 16, 30, 50).unwrap();
        let mut rng = StdRng::seed_from_u64(99);
        let dataset: Vec<_> = (0..n).map(|_|
            (0..2).map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect::<Vec<_>>()).collect::<Vec<_>>()
        ).collect();
        let doc_ids: Vec<u32> = (1..=n as u32).collect();
        r.index_dataset(&dataset, &doc_ids).unwrap();
        let result = r.get_top_k(&dataset[10], 10).unwrap();
        assert_eq!(result.len(), 10);
        assert!(result.contains(&11u32));
    }
}
