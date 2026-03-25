use std::sync::Mutex;

use muvera_core::{
    dot_product, ExactChamferRetriever, ExactChamferSimilarity, MuveraError, MuveraRetriever,
    Retriever,
    DiskAnnRetriever,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_py_error(error: MuveraError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
fn py_dot_product(lhs: Vec<f32>, rhs: Vec<f32>) -> PyResult<f32> {
    if lhs.len() != rhs.len() {
        return Err(PyValueError::new_err("lhs and rhs must have the same length"));
    }
    Ok(dot_product(&lhs, &rhs))
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
        self.inner
            .lock()
            .expect("mutex poisoned")
            .index_dataset(&dataset, &doc_ids)
            .map_err(to_py_error)
    }

    fn add_document(&self, document: Vec<Vec<f32>>, doc_id: u32) -> PyResult<()> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .add_document(document, doc_id)
            .map_err(to_py_error)
    }

    fn get_top_k(&self, query: Vec<Vec<f32>>, top_k: usize) -> PyResult<Vec<u32>> {
        self.inner
            .lock()
            .expect("mutex poisoned")
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
    fn embedding_dim(&self) -> usize {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .embedding_dim()
    }

    fn index_dataset(&self, dataset: Vec<Vec<Vec<f32>>>, doc_ids: Vec<u32>) -> PyResult<()> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .index_dataset(&dataset, &doc_ids)
            .map_err(to_py_error)
    }

    fn add_document(&self, document: Vec<Vec<f32>>, doc_id: u32) -> PyResult<()> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .add_document(document, doc_id)
            .map_err(to_py_error)
    }

    fn get_top_k(&self, query: Vec<Vec<f32>>, top_k: usize) -> PyResult<Vec<u32>> {
        self.inner
            .lock()
            .expect("mutex poisoned")
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
    fn embedding_dim(&self) -> usize {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .embedding_dim()
    }

    #[getter]
    fn search_l(&self) -> usize {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .search_l()
    }

    fn index_dataset(&self, dataset: Vec<Vec<Vec<f32>>>, doc_ids: Vec<u32>) -> PyResult<()> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .index_dataset(&dataset, &doc_ids)
            .map_err(to_py_error)
    }

    fn add_document(&self, document: Vec<Vec<f32>>, doc_id: u32) -> PyResult<()> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .add_document(document, doc_id)
            .map_err(to_py_error)
    }

    fn get_top_k(&self, query: Vec<Vec<f32>>, top_k: usize) -> PyResult<Vec<u32>> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .get_top_k(&query, top_k)
            .map_err(to_py_error)
    }
}

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_dot_product, module)?)?;
    module.add_function(wrap_pyfunction!(py_exact_chamfer_similarity, module)?)?;
    module.add_class::<PyExactChamferRetriever>()?;
    module.add_class::<PyMuveraRetriever>()?;
    module.add_class::<PyDiskAnnRetriever>()?;
    Ok(())
}
