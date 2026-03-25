# muvera

Rust-native MUVERA: multi-vector retrieval via fixed dimensional encodings, plus Python bindings for wheel builds.

## Status

The active implementation is now Rust-first:

- core library: [crates/muvera-core](crates/muvera-core)
- Python extension: [crates/muvera-py](crates/muvera-py)
- Python package shim: [python/muvera](python/muvera)

The legacy C++ and CMake code remains in the repository as reference code, but it is no longer the primary build path.

## Build the Rust library

From the repository root:

```bash
cargo test
```

## Build the Python wheel locally

```bash
python -m pip install maturin
maturin build --release
```

To install into the current environment:

```bash
maturin develop --release
```

Then in Python:

```python
from muvera import MuveraRetriever

retriever = MuveraRetriever(
	dimensions=3,
	max_points=500,
	d_proj=128,
	d_final=10240,
	k_sim=10,
	r_reps=5,
	seed=42,
)

dataset = [
	[[1.0, 2.0, 3.0], [1.0, -2.0, 3.0]],
	[[4.0, 5.0, 6.0], [4.0, -5.0, 6.0]],
]

retriever.index_dataset(dataset, [1, 2])
print(retriever.get_top_k(dataset[0], 1))
```

## Notes

- The Rust port includes the full FDE pipeline and both retrievers.
- The active `MuveraRetriever` is Rust-native and no longer compiles the legacy C++ DiskANN sources.
- The Python package is configured through [pyproject.toml](pyproject.toml) for PyPI-compatible wheel builds.
