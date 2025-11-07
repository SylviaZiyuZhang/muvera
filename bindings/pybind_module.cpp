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
        .def("load_index", &ExactChamferRetriever::load_index)
        .def("save_index", &ExactChamferRetriever::save_index)
        .def("add_document", &ExactChamferRetriever::add_document)
        .def("get_top_k", &ExactChamferRetriever::get_top_k);
    
    py::class_<RelaxedChamferRetriever>(m, "RelaxedChamferRetriever")
        .def(py::init<size_t, size_t, size_t>()) // _dimensions, _max_points, _softmax_s
        .def("index_dataset", &RelaxedChamferRetriever::index_dataset)
        .def("load_index", &RelaxedChamferRetriever::load_index)
        .def("save_index", &RelaxedChamferRetriever::save_index)
        .def("add_document", &RelaxedChamferRetriever::add_document)
        .def("get_top_k", &RelaxedChamferRetriever::get_top_k)
        .def("get_softmax_s", &RelaxedChamferRetriever::get_softmax_s);

    py::class_<MuveraRetriever>(m, "MuveraRetriever")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, uint64_t>())
        .def("get_embedding_dim", &MuveraRetriever::get_embedding_dim)
        .def("index_dataset", &MuveraRetriever::index_dataset)
        .def("load_index", &MuveraRetriever::load_index)
        .def("save_index", &MuveraRetriever::save_index)
        .def("add_document", &MuveraRetriever::add_document)
        .def("get_top_k", &MuveraRetriever::get_top_k);
}
