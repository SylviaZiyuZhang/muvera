use std::sync::Mutex;

use muvera_core::{
    DiskAnnRetriever, ExactChamferRetriever, ExactChamferSimilarity, MuveraError, MuveraRetriever,
    Retriever,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn poisoned_lock_error(type_name: &str) -> PyErr {
    PyValueError::new_err(format!("{type_name} internal lock poisoned"))
}

fn lock_inner<'a, T>(mutex: &'a Mutex<T>, type_name: &str) -> PyResult<std::sync::MutexGuard<'a, T>> {
    mutex.lock().map_err(|_| poisoned_lock_error(type_name))
}

fn to_py_error(error: MuveraError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
fn py_exact_chamfer_similarity(
    dimensions: usize,
    lhs: Vec<Vec<f32>>,
    rhs: Vec<Vec<f32>>,
) -> PyResult<f32> {
    ExactChamferSimilarity::new(dimensions)
        .compute_similarity(&lhs, &rhs)
        .map_err(to_py_error)
}

#[pyclass(name = "ExactChamferRetriever")]
struct PyExactChamferRetriever {
    inner: Mutex<ExactChamferRetriever>,
}

#[pymethods]
impl PyExactChamferRetriever {
    #[new]
    fn new(dimensions: usize, max_points: usize) -> Self {
        Self {
            inner: Mutex::new(ExactChamferRetriever::new(dimensions, max_points)),
        }
    }

    fn index_dataset(&self, dataset: Vec<Vec<Vec<f32>>>, doc_ids: Vec<u32>) -> PyResult<()> {
        let mut inner = lock_inner(&self.inner, "ExactChamferRetriever")?;
        inner
            .index_dataset(&dataset, &doc_ids)
            .map_err(to_py_error)
    }

    fn add_document(&self, document: Vec<Vec<f32>>, doc_id: u32) -> PyResult<()> {
        let mut inner = lock_inner(&self.inner, "ExactChamferRetriever")?;
        inner
            .add_document(document, doc_id)
            .map_err(to_py_error)
    }

    fn get_top_k(&self, query: Vec<Vec<f32>>, top_k: usize) -> PyResult<Vec<u32>> {
        lock_inner(&self.inner, "ExactChamferRetriever")?
            .get_top_k(&query, top_k)
            .map_err(to_py_error)
    }
}

#[pyclass(name = "MuveraRetriever")]
struct PyMuveraRetriever {
    inner: Mutex<MuveraRetriever>,
}

#[pymethods]
impl PyMuveraRetriever {
    #[new]
    fn new(
        dimensions: usize,
        max_points: usize,
        d_proj: usize,
        d_final: usize,
        k_sim: usize,
        r_reps: usize,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(
                MuveraRetriever::new(
                    dimensions,
                    max_points,
                    d_proj,
                    d_final,
                    k_sim,
                    r_reps,
                    seed,
                )
                .map_err(to_py_error)?,
            ),
        })
    }

    #[getter]
    fn embedding_dim(&self) -> PyResult<usize> {
        Ok(lock_inner(&self.inner, "MuveraRetriever")?
            .embedding_dim()
        )
    }

    fn index_dataset(&self, dataset: Vec<Vec<Vec<f32>>>, doc_ids: Vec<u32>) -> PyResult<()> {
        lock_inner(&self.inner, "MuveraRetriever")?
            .index_dataset(&dataset, &doc_ids)
            .map_err(to_py_error)
    }

    fn add_document(&self, document: Vec<Vec<f32>>, doc_id: u32) -> PyResult<()> {
        lock_inner(&self.inner, "MuveraRetriever")?
            .add_document(document, doc_id)
            .map_err(to_py_error)
    }

    fn get_top_k(&self, query: Vec<Vec<f32>>, top_k: usize) -> PyResult<Vec<u32>> {
        lock_inner(&self.inner, "MuveraRetriever")?
            .get_top_k(&query, top_k)
            .map_err(to_py_error)
    }
}

#[pyclass(name = "DiskAnnRetriever")]
struct PyDiskAnnRetriever {
    inner: Mutex<DiskAnnRetriever>,
}

#[pymethods]
impl PyDiskAnnRetriever {
    #[new]
    fn new(
        dimensions: usize,
        max_points: usize,
        d_proj: usize,
        d_final: usize,
        k_sim: usize,
        r_reps: usize,
        seed: u64,
        max_degree: usize,
        l_build: usize,
        search_l: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(
                DiskAnnRetriever::new(
                    dimensions,
                    max_points,
                    d_proj,
                    d_final,
                    k_sim,
                    r_reps,
                    seed,
                    max_degree,
                    l_build,
                    search_l,
                )
                .map_err(to_py_error)?,
            ),
        })
    }

    #[getter]
    fn embedding_dim(&self) -> PyResult<usize> {
        Ok(lock_inner(&self.inner, "DiskAnnRetriever")?
            .embedding_dim())
    }

    #[getter]
    fn search_l(&self) -> PyResult<usize> {
        Ok(lock_inner(&self.inner, "DiskAnnRetriever")?
            .search_l())
    }

    fn index_dataset(&self, dataset: Vec<Vec<Vec<f32>>>, doc_ids: Vec<u32>) -> PyResult<()> {
        let mut inner = lock_inner(&self.inner, "DiskAnnRetriever")?;
        inner
            .index_dataset(&dataset, &doc_ids)
            .map_err(to_py_error)
    }

    fn add_document(&self, document: Vec<Vec<f32>>, doc_id: u32) -> PyResult<()> {
        let mut inner = lock_inner(&self.inner, "DiskAnnRetriever")?;
        inner
            .add_document(document, doc_id)
            .map_err(to_py_error)
    }

    fn get_top_k(&self, query: Vec<Vec<f32>>, top_k: usize) -> PyResult<Vec<u32>> {
        lock_inner(&self.inner, "DiskAnnRetriever")?
            .get_top_k(&query, top_k)
            .map_err(to_py_error)
    }
}

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_exact_chamfer_similarity, module)?)?;
    module.add_class::<PyExactChamferRetriever>()?;
    module.add_class::<PyMuveraRetriever>()?;
    module.add_class::<PyDiskAnnRetriever>()?;
    Ok(())
}
