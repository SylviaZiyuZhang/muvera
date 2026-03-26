# muvera

Rust-native MUVERA: multi-vector retrieval via fixed dimensional encodings, plus Python bindings for wheel builds.

## Status

The active implementation is now Rust-first:

- core library: [crates/muvera-core](crates/muvera-core)
- Python extension: [crates/muvera-py](crates/muvera-py)
- Python package shim: [python/muvera](python/muvera)

The legacy C++ remains in the repository on branch `legacy_cpp` for reference.

## Build the Rust library

Create and activate the Conda environment first:

```bash
conda env create -f environment.yml
conda activate muvera-rust
```

If the environment already exists and you only want to refresh dependencies:

```bash
conda env update -f environment.yml --prune
conda activate muvera-rust
```

Then from the repository root:

```bash
cargo test
```

## Build the Python wheel locally

```bash
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

## Python build, test, and release

### Local compile/test/package checks

After activating `muvera-rust`, from the repository root:

```bash
cargo test -p muvera-core
maturin develop --release
python -m pytest -q python/tests

rm -rf dist
maturin build --release --sdist -o dist
python -m twine check dist/*
```

### GitHub Actions workflows

- CI workflow: `.github/workflows/python-ci.yml`
	- Runs Rust tests, Python smoke tests, and package metadata checks on push/PR.
	- Builds a wheel with `maturin build` and installs it with `pip` for testing; it does not use `maturin develop`.
- Release workflow: `.github/workflows/python-release.yml`
	- Manual only via GitHub Actions `workflow_dispatch`.
	- Builds wheels (Linux/macOS/Windows) and sdist.
	- Publishes to TestPyPI via manual dispatch (`repository=testpypi`).
	- Publishes to PyPI via manual dispatch (`repository=pypi`).

For publishing, configure trusted publishing in PyPI/TestPyPI for this repository so
`pypa/gh-action-pypi-publish` can use OIDC (`id-token: write`).

Typical release flow:

```bash
conda activate muvera-rust

cargo test -p muvera-core
maturin develop --release
python -m pytest -q python/tests

rm -rf dist
maturin build --release --sdist -o dist
python -m twine check dist/*

git tag v0.1.1
git push origin v0.1.1
```

Then trigger `.github/workflows/python-release.yml` manually in GitHub Actions and choose
`repository=testpypi` or `repository=pypi`.

## Notes

- The Rust port includes the full FDE pipeline and both retrievers.
- The active `MuveraRetriever` is Rust-native and no longer compiles the legacy C++ DiskANN sources.
- The Python package is configured through [pyproject.toml](pyproject.toml) for PyPI-compatible wheel builds.
- The legacy C++ code is available on the `legacy_cpp` branch.
