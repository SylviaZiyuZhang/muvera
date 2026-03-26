from muvera import (
    DiskAnnRetriever,
    ExactChamferRetriever,
    MuveraRetriever,
    dot_product,
    exact_chamfer_similarity,
)


def test_math_helpers_smoke() -> None:
    assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0

    lhs = [[1.0, 0.0], [0.0, 1.0]]
    rhs = [[1.0, 0.0], [0.0, 1.0]]
    score = exact_chamfer_similarity(2, lhs, rhs)
    assert score >= 2.0


def test_retrievers_smoke() -> None:
    dataset = [
        [[1.0, 0.1, 0.0], [0.9, 0.0, 0.1]],
        [[0.0, 1.0, 0.1], [0.1, 0.9, 0.0]],
        [[0.1, 0.0, 1.0], [0.0, 0.1, 0.9]],
    ]
    doc_ids = [1, 2, 3]

    exact = ExactChamferRetriever(dimensions=3, max_points=16)
    exact.index_dataset(dataset, doc_ids)
    assert 1 in exact.get_top_k(dataset[0], 1)

    muvera = MuveraRetriever(
        dimensions=3,
        max_points=16,
        d_proj=16,
        d_final=128,
        k_sim=4,
        r_reps=3,
        seed=42,
    )
    muvera.index_dataset(dataset, doc_ids)
    assert muvera.get_top_k(dataset[0], 1)[0] == 1

    diskann = DiskAnnRetriever(
        dimensions=3,
        max_points=16,
        d_proj=16,
        d_final=128,
        k_sim=4,
        r_reps=3,
        seed=42,
        max_degree=8,
        l_build=20,
        search_l=30,
    )
    diskann.index_dataset(dataset, doc_ids)
    assert 1 in diskann.get_top_k(dataset[0], 3)
