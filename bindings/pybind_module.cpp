#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fde.h"
#include "retriever.h"

namespace py = pybind11;

PYBIND11_MODULE(muvera_pybind, m) {
    m.doc() = "Python bindings for Muvera and ExactChamfer retrievers";

    py::class_<ExactChamferRetriever>(m, "ExactChamferRetriever")
        .def(py::init<size_t, size_t>()) // _dimensions, _max_points
        .def("index_dataset", &ExactChamferRetriever::index_dataset)
        .def("add_document", &ExactChamferRetriever::add_document)
        .def("get_top_k", &ExactChamferRetriever::get_top_k);

    py::class_<MuveraRetriever>(m, "MuveraRetriever")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, uint64_t>())
        .def("get_embedding_dim", &MuveraRetriever::get_embedding_dim)
        .def("index_dataset", &MuveraRetriever::index_dataset)
        .def("add_document", &MuveraRetriever::add_document)
        .def("get_top_k", &MuveraRetriever::get_top_k);
}
