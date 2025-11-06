import time
import random
import numpy as np
from muvera_pybind import ExactChamferRetriever, MuveraRetriever

def test_exact_chamfer_retriever_large_100D_top50():
    dimensions = 100
    num_docs = 500
    vectors_per_doc = 4
    top_k = 50

    exact_retriever = ExactChamferRetriever(dimensions, num_docs)
    # === Generate synthetic dataset ===
    rng = np.random.default_rng(12345)
    dataset = [
        [rng.uniform(-3.0, 3.0, dimensions).astype(np.float32).tolist() for _ in range(vectors_per_doc)]
        for _ in range(num_docs)
    ]
    doc_ids = [d + 1 for d in range(num_docs)]

    # === Index dataset (timed) ===
    t0 = time.perf_counter()
    exact_retriever.index_dataset(dataset, doc_ids)
    index_time_ms = (time.perf_counter() - t0) * 1000

    # === Query (timed) ===
    query_idx = 100
    query_doc = dataset[query_idx]

    q0 = time.perf_counter()
    result = exact_retriever.get_top_k(query_doc, top_k)
    query_time_ms = (time.perf_counter() - q0) * 1000

    # === Assertions ===
    assert len(result) == top_k, f"Expected {top_k} results, got {len(result)}"
    print(result)
    assert (query_idx + 1) in result, "Query document should appear in its own top_k results"

    # === Report ===
    print("✅ test_exact_chamfer_retriever_large_100D_top50 passed")
    print(f"   Indexing time: {index_time_ms:.2f} ms")
    print(f"   Query time:    {query_time_ms:.2f} ms")

def test_muvera_retriever_large_100D_top50():
    dimensions = 100
    num_docs = 500
    vectors_per_doc = 4
    top_k = 50

    muvera = MuveraRetriever(
        dimensions,   # dimensions
        num_docs,     # max_points
        64,           # d_proj
        4096,         # d_final
        7,            # k_sim
        10,           # r_reps
        42            # seed
    )

    # === Generate synthetic dataset ===
    rng = np.random.default_rng(12345)
    dataset = [
        [rng.uniform(-3.0, 3.0, dimensions).astype(np.float32).tolist() for _ in range(vectors_per_doc)]
        for _ in range(num_docs)
    ]
    doc_ids = [d + 1 for d in range(num_docs)]

    # === Index dataset (timed) ===
    t0 = time.perf_counter()
    muvera.index_dataset(dataset, doc_ids)
    index_time_ms = (time.perf_counter() - t0) * 1000

    # === Query (timed) ===
    query_idx = 100
    query_doc = dataset[query_idx]

    q0 = time.perf_counter()
    result = muvera.get_top_k(query_doc, top_k)
    query_time_ms = (time.perf_counter() - q0) * 1000

    # === Assertions ===
    assert len(result) == top_k, f"Expected {top_k} results, got {len(result)}"
    assert (query_idx + 1) in result, "Query document should appear in its own top_k results"

    # === Report ===
    print("✅ test_muvera_retriever_large_100D_top50 passed")
    print(f"   Indexing time: {index_time_ms:.2f} ms")
    print(f"   Query time:    {query_time_ms:.2f} ms")


if __name__ == "__main__":
    test_exact_chamfer_retriever_large_100D_top50()
    test_muvera_retriever_large_100D_top50()
