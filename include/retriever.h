#include <iostream>
#include <unordered_set>
#include <vector>
#include <string>

#include "abstract_index.h"
#include "index.h"
#include "index_config.h"
#include "index_factory.h"
#include "ann_exception.h"
#include "utils.h"

class AbstractRetriever {
    protected:
    size_t dimensions;
    size_t max_points;
    bool initialized;
    std::vector<std::string> doc_ids;

    public:
    AbstractRetriever(const size_t _dimensions, const size_t _max_points)
    :dimensions(_dimensions), max_points(_max_points) {
        initialized = false;
        doc_ids = std::vector<std::string>();
    };
    virtual ~AbstractRetriever() = default;

    // Initializes the retriever with the dataset
    virtual void index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<std::string> _doc_ids) = 0;
    // REQUIRES: dataset.size() == doc_ids.size()
    // ENSURES: initialized

    // Adds a document into the retriever.
    virtual void add_document(const std::vector<std::vector<float>>& P, const std::string doc_id) = 0;

    // Retrieves the top k documents based on a query.
    virtual std::vector<std::string> get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const = 0;
};

class ExactChamferRetriever : public AbstractRetriever {
    private:
    std::unique_ptr<ExactChamferSimilarity> similarity_engine;
    std::vector<std::vector<std::vector<float>>> dataset;

    public:
    ExactChamferRetriever(const size_t _dimensions, const size_t _max_points);

    void index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<std::string> _doc_ids) override;

    void add_document(const std::vector<std::vector<float>>& P, const std::string doc_id) override;
    std::vector<std::string> get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const override;
};

class MuveraRetriever : public AbstractRetriever {
    private:
    std::unique_ptr<FDESimilarity> fde_engine;
    std::unique_ptr<diskann::AbstractIndex> diskann_index;
    size_t embedding_dim;

    public:
    MuveraRetriever(const size_t _dimensions, const size_t _max_points, const size_t _d_proj, const size_t _d_final,
        const size_t _k_sim, const size_t _r_reps, const uint64_t _seed
    );

    size_t get_embedding_dim() {
        return embedding_dim;
    }

    void index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<std::string> _doc_ids) override;

    void add_document(const std::vector<std::vector<float>>& P, const std::string doc_id) override;

    std::vector<std::string> get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const override;
};