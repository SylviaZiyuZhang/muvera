#include <bitset>
#include <cstdint>
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

class MuveraRetriever : public AbstractRetriever {
    private:
    std::unique_ptr<FDESimilarity> fde_engine;
    std::unique_ptr<diskann::AbstractIndex> diskann_index;
    size_t embedding_dim;

    public:
    MuveraRetriever(const size_t _dimensions, const size_t _d_proj,
        const size_t _B, const size_t _k_sim, const size_t _r_reps
    ): AbstractRetriever(_dimensions) {
        
        fde_engine = std::make_unique<FDESimilarity>(_dimensions, _d_proj, _B, _k_sim, _r_reps);
        embedding_dim = fde_engine->get_d_fde();

        diskann::IndexConfig config = diskann::IndexConfigBuilder()
            .with_metric(diskann::Metric::COSINE)
            .with_dimension(embedding_dim) // TODO: change this to final projection dimension after final projection is implemented
            .with_max_points(100000)
            .is_dynamic_index(true)
            .is_enable_tags(true)
            .with_data_type("float")
            .build();
        diskann::IndexFactory index_factory(config);
        diskann_index = index_factory.create_instance();
    };

    size_t get_embedding_dim() {
        return embedding_dim;
    }

    void index_dataset(const std::vector<std::vector<std::vector<double>>>& _dataset, const std::vector<uint64_t> _doc_ids) override
    {
        if (_dataset.size() != _doc_ids.size()) {
            throw std::runtime_error("MuveraRetriever.index_dataset: dataset and doc_ids have different sizes.");
        }
        std::vector<std::vector<double>> fdes = std::vector<std::vector<double>>();
        fdes.reserve(_dataset.size());
        for(auto P : _dataset) {
            fdes.emplace_back(fde_engine->encode_document(P));
        }
        diskann_index->build(fdes.data(), _dataset.size(), _doc_ids);

        initialized = true;
    }

    void add_document(const std::vector<std::vector<double>>& P, const uint64_t doc_id) override {
        if (!initialized) {
            throw std::runtime_error("MuveraRetriever add_document on uninitialized index!");
        }
        std::vector<double> encoding = fde_engine->encode_document(P);
        diskann_index->insert_point(encoding.data(), doc_id);
    }

    std::vector<uint64_t> get_top_k(const std::vector<std::vector<double>>& Q, const size_t top_k) const override {
        if (!initialized) {
            throw std::runtime_error("MuveraRetriever get_top_k on uninitialized index!");
        }
        std::vector<double> query_encoding = fde_engine->encode_query(Q);
        // TODO: search diskann_index with the query encoding and return corresponding tags
        std::vector<uint64_t> tags(top_k);
        std::vector<float> distances(top_k);
        std::vector<double*> result_vectors;
        for (size_t i = 0; i < top_k; i++) {
            double* v = new double(embedding_dim);
            result_vectors.push_back(v);
        }
        diskann_index->search_with_tags(
            query_encoding.data(),
            static_cast<const uint64_t>(top_k),
            static_cast<const uint32_t>(75), // beam width L
            tags.data(),
            distances.data(),
            result_vectors
        );
        return tags;
    }
};