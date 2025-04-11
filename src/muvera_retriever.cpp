#include <bitset>
#include <memory>
#include <random>
#include <vector>

#include <sstream>
#include <stdexcept>

#include <cmath>

#include "fde.h"
#include "retriever.h"

#include "index.h"
#include "index_config.h"
#include "index_factory.h"
#include "ann_exception.h"
#include "utils.h"

double dot_product(const std::vector<double>& h, const std::vector<double>& p, size_t dimensions) {
    // REQUIRES: h.size() == dimensions && p.size() == dimensions
    double result = 0.0;
    for (size_t i = 0; i < dimensions; i++)
        result += h[i] * p[i];
    return result;
};

class MuveraRetriever : public AbstractRetriever {
    private:
    std::unique_ptr<FDESimilarity> similarity_engine;
    std::unique_ptr<diskann::AbstractIndex> diskann_index;

    public:
    MuveraRetriever(const size_t dimensions, const size_t d_proj,
        const size_t B, const size_t k_sim, const size_t r_reps
    ): AbstractChamferSimilarity(dimensions) {
        
        similarity_engine = std::make_unique<FDESimilarity>(dimensions, d_proj, B, k_sim, r_reps);

        diskann::IndexConfig config = diskann::IndexConfigBuilder()
            .with_metric(diskann::Metric::COSINE)
            .with_dimension(B * d_proj * r_reps)
            .with_max_points(100000)
            .is_dynamic_index(true)
            .is_enable_tags(true)
            .with_data_type("float")
            .build();
        diskann::IndexFactory index_factory(config);
        diskann_index = index_factory.create_instance();
    };

    size_t get_d_fde() {
        return B * d_proj * r_reps;
    }

    void index_document(const std::vector<std::vector<double>>& P, const uint64_t doc_id) const override {
        std::vector<double> encoding = encode_document(P);
        diskann_index->insert_point(encoding.data(), doc_id);
    }

    double compute_similarity(
        // TODO: implement final projections
        const std::vector<std::vector<double>>& P,
        const std::vector<std::vector<double>>& Q) const override {
        return dot_product(encode_document(P), encode_query(Q), B * d_proj * r_reps);
    }
};