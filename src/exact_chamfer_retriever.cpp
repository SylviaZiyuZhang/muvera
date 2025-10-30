#include <immintrin.h>

#include <bitset>
#include <cstdint>
#include <memory>
#include <queue>
#include <random>
#include <vector>

#include <sstream>
#include <stdexcept>

#include <cmath>

#include "fde.h"
#include "retriever.h"



ExactChamferRetriever::ExactChamferRetriever(const size_t _dimensions,
    const size_t _max_points): AbstractRetriever(_dimensions, _max_points) {
    similarity_engine = std::make_unique<ExactChamferSimilarity>(_dimensions);
    dataset = std::vector<std::vector<std::vector<float>>>();
    doc_ids = std::vector<uint32_t>();
};


void ExactChamferRetriever::index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<uint32_t> _doc_ids)
{
    if (dataset.size() != doc_ids.size()) {
        throw std::runtime_error("ExactChamferRetriever.index_dataset: dataset and doc_ids have different sizes.");
    }
    dataset = _dataset;
    doc_ids = _doc_ids;
    initialized = true;
};

void ExactChamferRetriever::add_document(const std::vector<std::vector<float>>& P, const uint32_t doc_id) {
    if (!initialized) {
        throw std::runtime_error("ExactChamferRetriever add_document on uninitialized index!");
    }
    dataset.push_back(P);
    doc_ids.push_back(doc_id);
};

std::vector<uint32_t> ExactChamferRetriever::get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const {
    if (!initialized) {
        throw std::runtime_error("ExactChamferRetriever get_top_k on uninitialized index!");
    }
    std::priority_queue<std::pair<float, uint32_t>> pq;
    for (size_t i = 0; i < dataset.size(); i++) {
        float similarity = similarity_engine->compute_similarity(dataset[i], Q);
        pq.push({similarity, doc_ids[i]});
        if (pq.size() > top_k) pq.pop();
    }
    std::vector<uint32_t> results = std::vector<uint32_t>();
    while (!pq.empty()) {
        auto t = pq.top();
        pq.pop();
        results.push_back(t.second);
    }
    return results;
};