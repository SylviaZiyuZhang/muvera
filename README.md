# muvera
Reproducing MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings and experiments.

# Warning
Due to compatibility issues between `scikib-build-core` and Intel MKL libraries, this library is highly unstable and may undergo major dependency/API redesigns. Some parameter setting/features (e.g. query clustering, empty bucket handling) described in the paper have not been tested and released yet as a result. Please use and refer to at your own discretion.

# Setup

This project depends on Microsoft DiskANN and its dependencies. You may need to modify `POSSIBLE_MKL_LIB_PATHS`, `POSSIBLE_OMP_PATHS`, and `POSSIBLE_MKL_INCLUDE_PATHS` in `CMakeLists.txt`. To build and test MUVERA, run the following steps from the project root.

In a conda environment, run
```
conda install -c conda-forge boost-cpp
```

```
# Initialize submodules
# Note that the submodule uses the author's patched version of DiskANN instead of the latest Microsoft commit
git submodule update --init --recursive

# make
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cd build
make -j
make muvera_pybind -j
ctest
cd bindings
cp ../../bindings/tests/pybind_test.py .
python pybind_test.py
```
Note that running `pip install` likely does not work due to mkl library link ordering issues that the author has yet to resolve.

# Known Issues
You may need to apply the following local patches to DiskANN and/or sync the submodule to point to the author's patched fork via `git submodule sync; git submodule update`:
1. In `DiskANN/src/index_factory.cpp`, in the `std::unique_ptr<AbstractIndex> IndexFactory::create_instance()` function, `_config->num_frozen_pts` is being added twice to `_num_points` for constructing the graph store and the data store. This will cause assertions to fail since we are currently using `create_instance` in `retriever.h`. This should be fixed by the author's patched commit.

# Contributing / Troubleshooting
This repository is being actively monitored. Please feel free to submit PRs/Issues.

# Acknowledgements
I wholeheartedly appreciate the authors of [the paper](https://arxiv.org/abs/2405.19504) for helping clarify experiment recipes and releasing the Google code patches [here](https://github.com/google/graph-mining/tree/main/sketching/point_cloud).
