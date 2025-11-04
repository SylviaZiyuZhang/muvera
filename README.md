# muvera
A C++ implementation MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings and experiments.

# Warning
Due to some issues between `scikib-build-core` and Intel MKL libraries described in the Setup section below, this library is highly unstable and may undergo major dependency/API redesigns. The Python binding also cannot be installed via pip at the moment. As a result, thorough testing with retrieval benchmarks have not been conducted and some parameter setting/features (e.g. query clustering, empty bucket handling) described in the paper have not gone through basic testing and been released yet. Please use and refer to at your own discretion.

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

## Roadmap
- [ ] ANNS index dependency resolution so that installation and Python compatibility work out-of-the-box.

- [ ] Integration with `datasets` to improve usability with BEIR and other retrieval benchmarks.

- [ ] Performance optimizations to improve scalability.

# Acknowledgements
I wholeheartedly appreciate the authors of [the original paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/b71cfefae46909178603b5bc6c11d3ae-Paper-Conference.pdf) for helping clarify experiment recipes and releasing the Google code patches [here](https://github.com/google/graph-mining/tree/main/sketching/point_cloud).

If you find this work useful, please cite the NIPS paper. Please use the GitHub citation tool to cite this implementation when applicable.