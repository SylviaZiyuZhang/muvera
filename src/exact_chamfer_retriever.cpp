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


class ExactChamferRetriever : public AbstractRetriever {
    private:
    std::unique_ptr<ExactChamferSimilarity> similarity_engine;
    std::vector<std::vector<std::vector<double>>> dataset;
    std::vector<uint64_t> doc_ids;

    public:
    ExactChamferRetriever(const size_t _dimensions): AbstractRetriever(_dimensions) {
        
        similarity_engine = std::make_unique<ExactChamferSimilarity>(_dimensions);
        dataset = std::vector<std::vector<std::vector<double>>>();
        doc_ids = std::vector<uint64_t>();
    };


    void index_dataset(const std::vector<std::vector<std::vector<double>>>& _dataset, const std::vector<uint64_t> _doc_ids) override
    {
        if (dataset.size() != doc_ids.size()) {
            throw std::runtime_error("ExactChamferRetriever.index_dataset: dataset and doc_ids have different sizes.");
        }
        dataset = _dataset;
        doc_ids = _doc_ids;
        initialized = true;
    }

    void add_document(const std::vector<std::vector<double>>& P, const uint64_t doc_id) override {
        if (!initialized) {
            throw std::runtime_error("ExactChamferRetriever add_document on uninitialized index!");
        }
        dataset.push_back(P);
        doc_ids.push_back(doc_id);
    }

    std::vector<uint64_t> get_top_k(const std::vector<std::vector<double>>& Q, const size_t top_k) const override {
        if (!initialized) {
            throw std::runtime_error("ExactChamferRetriever get_top_k on uninitialized index!");
        }
        std::priority_queue<std::pair<double, uint64_t>> pq;
        for (size_t i = 0; i < dataset.size(); i++) {
            double similarity = similarity_engine->compute_similarity(dataset[i], Q);
            pq.push({similarity, doc_ids[i]});
        }
        std::vector<uint64_t> results = std::vector<uint64_t>();
        while (!pq.empty()) {
            auto t = pq.top();
            pq.pop();
            results.push_back(t.second);
        }
        return results;
    }
};