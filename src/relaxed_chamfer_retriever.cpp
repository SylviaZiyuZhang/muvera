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


// Cosine similarity is hardcoded into RelaxedChamferRetrievers
RelaxedChamferRetriever::RelaxedChamferRetriever(const size_t _dimensions,
    const size_t _max_points, const size_t _softmax_s): AbstractRetriever(_dimensions, _max_points) {
    similarity_engine = std::make_unique<RelaxedChamferSimilarity>(_dimensions, _softmax_s);
    dataset = std::vector<std::vector<std::vector<float>>>();
    doc_ids = std::vector<std::string>();
};


void RelaxedChamferRetriever::index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<std::string> _doc_ids)
{
    if (dataset.size() != doc_ids.size()) {
        throw std::runtime_error("RelaxedChamferRetriever.index_dataset: dataset and doc_ids have different sizes.");
    }
    dataset = _dataset;
    doc_ids.insert(doc_ids.end(), _doc_ids.begin(), _doc_ids.end());
    initialized = true;
};

void RelaxedChamferRetriever::load_index(const std::string &checkpoint_dir) {

    throw std::logic_error("RelaxedChamferRetriever::load_index() is not yet implemented.");

}

void RelaxedChamferRetriever::save_index(const std::string &checkpoint_dir) {

    throw std::logic_error("RelaxedChamferRetriever::save_index() is not yet implemented.");

}

void RelaxedChamferRetriever::add_document(const std::vector<std::vector<float>>& P, const std::string doc_id) {
    if (!initialized) {
        throw std::runtime_error("RelaxedChamferRetriever add_document on uninitialized index!");
    }
    dataset.push_back(P);
    doc_ids.push_back(doc_id);
};

std::vector<std::string> RelaxedChamferRetriever::get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const {
    if (!initialized) {
        throw std::runtime_error("RelaxedChamferRetriever get_top_k on uninitialized index!");
    }
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<std::pair<float, uint32_t>>> pq;
    for (size_t i = 0; i < dataset.size(); i++) {
        float similarity = similarity_engine->compute_similarity(dataset[i], Q);
        pq.push({similarity, i});
        if (pq.size() > top_k) pq.pop();
    }
    std::vector<std::string> results = std::vector<std::string>();
    while (!pq.empty()) {
        auto t = pq.top();
        pq.pop();
        results.push_back(doc_ids[t.second]);
    }
    return results;
};