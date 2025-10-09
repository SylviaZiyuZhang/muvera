# muvera
Reproducing MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings and experiments.

# Setup

This project depends on Microsoft DiskANN and its dependencies. You may need to modify `POSSIBLE_MKL_LIB_PATHS`, `POSSIBLE_OMP_PATHS`, and `POSSIBLE_MKL_INCLUDE_PATHS` in `CMakeLists.txt`. To build and test MUVERA, run the following steps from the project root.

In a conda environment, run
```
conda install -c conda-forge boost-cpp
```

```
# Initialize submodules
git submodule update --init --recursive

# make
mkdir -p build && cd build
cmake ..
make
ctest
```

You may need to apply the following local patches to DiskANN:
1. Currently there is a small problem in compilation - you may need to add the following imports to include/utils.h to diskann.
```
#ifdef _WINDOWS
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <intrin.h>
#else
#include <immintrin.h>
#endif
```
2. In `DiskANN/src/index_factory.cpp`, in the `std::unique_ptr<AbstractIndex> IndexFactory::create_instance()` function, `_config->num_frozen_pts` is being added twice to `_num_points` for constructing the graph store and the data store. This will cause assertions to fail since we are currently using `create_instance` in `retriever.h`.
