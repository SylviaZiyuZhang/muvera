# muvera
Reproducing MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings and experiments.

# Setup

This project depends on Microsoft DiskANN and its dependencies. You may need to modify `POSSIBLE_MKL_LIB_PATHS`, `POSSIBLE_OMP_PATHS`, and `POSSIBLE_MKL_INCLUDE_PATHS` in `CMakeLists.txt`. To build and test MUVERA, run the following steps from the project root.

```
mkdir -p build && cd build
cmake ..
make
ctest
```

Currently there is a small problem in compilation - you need to add the following imports to include/utils.h to diskann.
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
