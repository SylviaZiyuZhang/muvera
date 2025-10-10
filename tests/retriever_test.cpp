#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cassert>

#include "fde.h"
#include "retriever.h"

void test_exact_chamfer_retriever_simple() {
    std::vector<float> a_1 = {1.0, 2.0, 3.0};
    std::vector<float> a_2 = {1.0, -2.0, 3.0};
    std::vector<float> b_1 = {4.0, 5.0, 6.0};
    std::vector<float> b_2 = {4.0, -5.0, 6.0};
    std::vector<std::vector<float>> A;
    A.resize(2);
    A[0] = a_1;
    A[1] = a_2;
    std::vector<std::vector<float>> B;
    B.resize(2);
    B[0] = b_1;
    B[1] = b_2;
    std::vector<std::vector<std::vector<float>>> dataset;
    dataset.push_back(A);
    dataset.push_back(B);
    std::vector<uint32_t> doc_ids = {1, 2};
    ExactChamferRetriever exactChamferRetriever(3, 500);
    exactChamferRetriever.index_dataset(dataset, doc_ids);
    std::vector<uint32_t> result = exactChamferRetriever.get_top_k(A, 1);
    assert(result.size() == 1);
    assert(result[0] == 1);
    std::cout << "✅ test_exact_chamfer_retriever_simple passed" << std::endl;
}

void test_muvera_retriever_basic() {
    std::vector<float> a_1 = {1.0, 2.0, 3.0};
    std::vector<float> a_2 = {1.0, -2.0, 3.0};
    std::vector<std::vector<float>> A;
    A.resize(2);
    A[0] = a_1;
    A[1] = a_2;
    std::vector<std::vector<std::vector<float>>> dataset;
    dataset.push_back(A);
    std::vector<uint32_t> doc_ids = {1};
    MuveraRetriever muveraRetriever(3, 500, 128, 10240, 10, 5, 42);
    muveraRetriever.index_dataset(dataset, doc_ids);
    std::vector<uint32_t> result = muveraRetriever.get_top_k(A, 1);
    assert(result.size() == 1);
    assert(result[0] == 1);
    std::cout << "✅ test_muvera_retriever_basic passed" << std::endl;
}

void test_muvera_retriever_large_100D_top50() {
    const size_t dimensions = 100;
    const size_t num_docs = 500;
    const size_t vectors_per_doc = 4;
    const size_t top_k = 50;

    MuveraRetriever muveraRetriever(
        dimensions,   // dimensions
        num_docs,     // max_points
        64,          // d_proj
        4096,        // d_final
        7,           // k_sim
        10,            // r_reps
        42            // seed
    );

    // === Generate synthetic dataset ===
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<std::vector<std::vector<float>>> dataset;
    dataset.reserve(num_docs);
    std::vector<uint32_t> doc_ids;
    doc_ids.reserve(num_docs);

    for (size_t d = 0; d < num_docs; d++) {
        std::vector<std::vector<float>> doc;
        doc.reserve(vectors_per_doc);
        for (size_t v = 0; v < vectors_per_doc; ++v) {
            std::vector<float> vec(dimensions);
            for (size_t i = 0; i < dimensions; i++)
                vec[i] = dist(gen);
            doc.push_back(std::move(vec));
        }
        dataset.push_back(std::move(doc));
        doc_ids.push_back(static_cast<uint32_t>(d + 1));
    }

    // === Index dataset (timed) ===
    auto t0 = std::chrono::high_resolution_clock::now();
    muveraRetriever.index_dataset(dataset, doc_ids);
    auto t1 = std::chrono::high_resolution_clock::now();
    double index_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // === Query (timed) ===
    const size_t query_idx = 100;
    const auto& query_doc = dataset[query_idx];

    auto q0 = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> result = muveraRetriever.get_top_k(query_doc, top_k);
    auto q1 = std::chrono::high_resolution_clock::now();
    double query_time_ms = std::chrono::duration<double, std::milli>(q1 - q0).count();

    // === Assertions ===
    assert(result.size() == top_k);

    bool found_self = std::find(result.begin(), result.end(), query_idx + 1) != result.end();
    assert(found_self && "Query document should appear in its own top_k results");

    // === Report ===
    std::cout << "✅ test_muvera_retriever_large_100D_top50 passed" << std::endl;
    std::cout << "   Indexing time: " << index_time_ms << " ms" << std::endl;
    std::cout << "   Query time:    " << query_time_ms << " ms" << std::endl;
}

int main() {
    test_exact_chamfer_retriever_simple();
    test_muvera_retriever_basic();
    test_muvera_retriever_large_100D_top50();
    return 0;
}
