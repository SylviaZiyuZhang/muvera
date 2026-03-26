//! Stress and quality tests for muvera-core retrievers.
//!
//! Run with:
//!   cargo test -p muvera-core --test stress -- --include-ignored
//! Or a single group:
//!   cargo test -p muvera-core --test stress recall -- --include-ignored

use std::time::Instant;

use muvera_core::{DiskAnnRetriever, ExactChamferRetriever, MuveraRetriever, Retriever};
use rand::{rngs::StdRng, Rng, SeedableRng};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn make_dataset(n: usize, vecs_per_doc: usize, dim: usize, seed: u64) -> Vec<Vec<Vec<f32>>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..vecs_per_doc)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
                .collect()
        })
        .collect()
}

fn make_doc_ids(n: usize) -> Vec<u32> {
    (1..=(n as u32)).collect()
}

/// Recall@k: fraction of exact top-k ids that appear in the approximate result.
fn recall(exact: &[u32], approx: &[u32]) -> f64 {
    let k = exact.len();
    let hits = exact.iter().filter(|id| approx.contains(id)).count();
    hits as f64 / k as f64
}

// ---------------------------------------------------------------------------
// 1. Throughput — MuveraRetriever index + bulk search
// ---------------------------------------------------------------------------

/// Index 5 000 documents (dim=64, 4 vecs each) then issue 100 top-50 queries.
/// Asserts that the correct doc is rank-1 for self-queries and measures wall-clock time.
#[test]
#[ignore]
fn stress_muvera_bulk_index_and_search() {
    let n = 5_000usize;
    let dim = 64usize;
    let dataset = make_dataset(n, 4, dim, 1);
    let doc_ids = make_doc_ids(n);

    let t0 = Instant::now();
    let mut r = MuveraRetriever::new(dim, n, 64, 4_096, 10, 5, 42).unwrap();
    r.index_dataset(&dataset, &doc_ids).unwrap();
    let index_ms = t0.elapsed().as_millis();

    let t1 = Instant::now();
    for i in (0..n).step_by(50) {
        let result = r.get_top_k(&dataset[i], 5).unwrap();
        assert_eq!(result.len(), 5, "should return 5 results");
        assert_eq!(result[0], (i + 1) as u32, "self-query should rank first (doc {})", i + 1);
    }
    let search_ms = t1.elapsed().as_millis();

    eprintln!("[stress_muvera_bulk] index={index_ms}ms, 100×top-5 search={search_ms}ms");
}

// ---------------------------------------------------------------------------
// 2. Throughput — DiskAnnRetriever index + bulk search
// ---------------------------------------------------------------------------

/// Index 2 000 documents with DiskANN (dim=32, 3 vecs each) then issue 100 top-10 queries.
/// Verifies that self-queries still return the correct document and measures wall-clock time.
#[test]
#[ignore]
fn stress_diskann_bulk_index_and_search() {
    let n = 2_000usize;
    let dim = 32usize;
    let dataset = make_dataset(n, 3, dim, 2);
    let doc_ids = make_doc_ids(n);

    let t0 = Instant::now();
    let mut r = DiskAnnRetriever::new(dim, n, 32, 1_024, 8, 4, 42, 32, 64, 80).unwrap();
    r.index_dataset(&dataset, &doc_ids).unwrap();
    let index_ms = t0.elapsed().as_millis();

    let t1 = Instant::now();
    for i in (0..n).step_by(20) {
        let result = r.get_top_k(&dataset[i], 10).unwrap();
        assert_eq!(result.len(), 10);
        assert!(result.contains(&((i + 1) as u32)), "doc {} missing from self-query result", i + 1);
    }
    let search_ms = t1.elapsed().as_millis();

    eprintln!("[stress_diskann_bulk] index={index_ms}ms, 100×top-10 search={search_ms}ms");
}

// ---------------------------------------------------------------------------
// 3. Recall quality — DiskANN vs exact over 500 documents
// ---------------------------------------------------------------------------

/// Index 500 documents with both MuveraRetriever (exact FDE dot-product) and DiskAnnRetriever.
/// Measures average top-20 recall@20 across 50 random queries; asserts ≥ 70 %.
#[test]
#[ignore]
fn stress_diskann_recall_vs_exact() {
    let n = 500usize;
    let dim = 64usize;
    let dataset = make_dataset(n, 4, dim, 3);
    let doc_ids = make_doc_ids(n);

    let mut exact = MuveraRetriever::new(dim, n, 64, 4_096, 10, 5, 42).unwrap();
    exact.index_dataset(&dataset, &doc_ids).unwrap();

    let mut approx = DiskAnnRetriever::new(dim, n, 64, 4_096, 10, 5, 42, 32, 64, 80).unwrap();
    approx.index_dataset(&dataset, &doc_ids).unwrap();

    let k = 20usize;
    let query_indices: Vec<usize> = (0..n).step_by(n / 50).collect();
    let total_queries = query_indices.len();
    let total_recall: f64 = query_indices
        .iter()
        .map(|&i| {
            let e = exact.get_top_k(&dataset[i], k).unwrap();
            let a = approx.get_top_k(&dataset[i], k).unwrap();
            recall(&e, &a)
        })
        .sum();

    let avg_recall = total_recall / total_queries as f64;
    eprintln!("[stress_recall] avg recall@{k} over {total_queries} queries = {avg_recall:.3}");
    assert!(
        avg_recall >= 0.70,
        "expected ≥70% recall@{k}, got {:.1}%",
        avg_recall * 100.0
    );
}

// ---------------------------------------------------------------------------
// 4. Incremental add_document pressure
// ---------------------------------------------------------------------------

/// Add 300 documents one-at-a-time via add_document (after an initial empty index_dataset),
/// then verify that 20 self-queries all return the correct top-1 result.
#[test]
#[ignore]
fn stress_muvera_incremental_add_document() {
    let n = 300usize;
    let dim = 48usize;
    let dataset = make_dataset(n, 3, dim, 4);
    let doc_ids = make_doc_ids(n);

    // Seed with the first doc so the index is "initialized"
    let mut r = MuveraRetriever::new(dim, n, 48, 2_048, 8, 4, 42).unwrap();
    r.index_dataset(&dataset[..1], &doc_ids[..1]).unwrap();

    // Incrementally add the rest
    for i in 1..n {
        r.add_document(dataset[i].clone(), doc_ids[i]).unwrap();
    }

    // Verify 20 self-queries
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..20 {
        let i = rng.gen_range(0..n);
        let result = r.get_top_k(&dataset[i], 1).unwrap();
        assert_eq!(result[0], (i + 1) as u32, "self-query failed for doc {}", i + 1);
    }
}

/// Same for DiskAnnRetriever — each add_document triggers a graph rebuild.
#[test]
#[ignore]
fn stress_diskann_incremental_add_document() {
    let n = 100usize;
    let dim = 32usize;
    let dataset = make_dataset(n, 2, dim, 5);
    let doc_ids = make_doc_ids(n);

    let mut r = DiskAnnRetriever::new(dim, n, 32, 512, 5, 3, 99, 16, 30, 50).unwrap();
    r.index_dataset(&dataset[..1], &doc_ids[..1]).unwrap();

    let t0 = Instant::now();
    for i in 1..n {
        r.add_document(dataset[i].clone(), doc_ids[i]).unwrap();
    }
    eprintln!("[stress_diskann_incremental] {} add_document calls = {}ms", n - 1, t0.elapsed().as_millis());

    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..10 {
        let i = rng.gen_range(0..n);
        let result = r.get_top_k(&dataset[i], 5).unwrap();
        assert!(result.contains(&((i + 1) as u32)), "doc {} missing from self-query result", i + 1);
    }
}

// ---------------------------------------------------------------------------
// 5. ExactChamferRetriever at scale
// ---------------------------------------------------------------------------

/// Index 2 000 documents with ExactChamferRetriever and verify 50 self-queries are rank-1.
#[test]
#[ignore]
fn stress_exact_chamfer_large_scale() {
    let n = 2_000usize;
    let dim = 32usize;
    let dataset = make_dataset(n, 5, dim, 7);
    let doc_ids = make_doc_ids(n);

    let mut r = ExactChamferRetriever::new(dim, n);
    let t0 = Instant::now();
    r.index_dataset(&dataset, &doc_ids).unwrap();
    eprintln!("[stress_exact_chamfer] index={}ms", t0.elapsed().as_millis());

    let t1 = Instant::now();
    let mut rng = StdRng::seed_from_u64(8);
    for _ in 0..50 {
        let i = rng.gen_range(0..n);
        let result = r.get_top_k(&dataset[i], 1).unwrap();
        // ExactChamferSimilarity is inner-product based, so doc b (index 1) ties with itself —
        // the closest match to any query is always the query document itself.
        assert!(result.contains(&((i + 1) as u32)), "self-query failed for doc {}", i + 1);
    }
    eprintln!("[stress_exact_chamfer] 50 queries={}ms", t1.elapsed().as_millis());
}

// ---------------------------------------------------------------------------
// 6. Varied document cardinality
// ---------------------------------------------------------------------------

/// Documents vary from 1 to 16 token vectors. For all three retriever types
/// the correct doc should still be rank-1 for self-queries.
#[test]
#[ignore]
fn stress_varied_doc_cardinality() {
    let n = 200usize;
    let dim = 32usize;
    let mut rng = StdRng::seed_from_u64(9);

    let dataset: Vec<Vec<Vec<f32>>> = (0..n)
        .map(|_| {
            let nvecs = rng.gen_range(1usize..=16);
            (0..nvecs)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
                .collect()
        })
        .collect();
    let doc_ids = make_doc_ids(n);

    // MuveraRetriever
    let mut mr = MuveraRetriever::new(dim, n, 32, 1_024, 8, 4, 42).unwrap();
    mr.index_dataset(&dataset, &doc_ids).unwrap();

    // DiskAnnRetriever
    let mut dr = DiskAnnRetriever::new(dim, n, 32, 1_024, 8, 4, 42, 16, 30, 50).unwrap();
    dr.index_dataset(&dataset, &doc_ids).unwrap();

    for i in (0..n).step_by(20) {
        let mr_result = mr.get_top_k(&dataset[i], 1).unwrap();
        assert_eq!(mr_result[0], (i + 1) as u32, "MuveraRetriever: self-query failed for doc {}", i + 1);

        let dr_result = dr.get_top_k(&dataset[i], 5).unwrap();
        assert!(dr_result.contains(&((i + 1) as u32)), "DiskAnnRetriever: doc {} missing", i + 1);
    }
}
