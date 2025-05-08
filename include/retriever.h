#include <iostream>
#include <unordered_set>
#include <vector>

#include "abstract_index.h"
#include "index.h"
#include "index_config.h"
#include "index_factory.h"
#include "ann_exception.h"
#include "utils.h"

class AbstractRetriever {
    protected:
    size_t dimensions;
    bool initialized;
    std::unordered_set<uint32_t> document_ids;

    public:
    AbstractRetriever(const size_t _dimensions): dimensions(_dimensions) {
        initialized = false;
        document_ids = std::unordered_set<uint32_t>(0); // DiskANN uses 0 as start point.
    };
    virtual ~AbstractRetriever() = default;

    // Initializes the retriever with the dataset
    virtual void index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<uint32_t> _doc_ids) = 0;
    // REQUIRES: doc_id > 0 for each doc_id : doc_ids
    // REQUIRES: dataset.size() == doc_ids.size()
    // ENSURES: initialized

    // Adds a document into the retriever.
    virtual void add_document(const std::vector<std::vector<float>>& P, const uint32_t doc_id) = 0;
    // REQUIRES: doc_id > 0

    // Retrieves the top k documents based on a query.
    virtual std::vector<uint32_t> get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const = 0;
};

class ExactChamferRetriever : public AbstractRetriever {
    private:
    std::unique_ptr<ExactChamferSimilarity> similarity_engine;
    std::vector<std::vector<std::vector<float>>> dataset;
    std::vector<uint32_t> doc_ids;

    public:
    ExactChamferRetriever(const size_t _dimensions);


    void index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<uint32_t> _doc_ids);

    void add_document(const std::vector<std::vector<float>>& P, const uint32_t doc_id);
    std::vector<uint32_t> get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const;
};

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
            .with_max_points(100)
            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
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

    void index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<uint32_t> _doc_ids) override
    {
        if (_dataset.size() != _doc_ids.size()) {
            throw std::runtime_error("MuveraRetriever.index_dataset: dataset and doc_ids have different sizes.");
        }
        std::vector<float> fdes_flat = std::vector<float>();
        fdes_flat.reserve(_dataset.size() * embedding_dim);
        for(auto P : _dataset) {
            std::vector<float> embedding = fde_engine->encode_document(P);
            fdes_flat.insert(fdes_flat.end(), embedding.begin(), embedding.end());
        }
        std::cout << _dataset.size() << std::endl;
        std::cout << "Stored pointer: " << fdes_flat.data() << std::endl;
        std::cout << "Stored type: " << typeid(fdes_flat.data()).name() << std::endl;

        std::any any_data = std::any(static_cast<const float*>(fdes_flat.data()));  // Store in std::any
        std::cout << "Stored in any_data, attempting cast..." << std::endl;

        try {
            const float* casted_data = std::any_cast<const float*>(any_data);
            std::cout << "Success" << std::endl;
        } catch (const std::bad_any_cast &e) {
            std::cout << "Bad any cast: " << e.what() << std::endl;
        }
        // TODO: The following should work
        diskann_index->build(static_cast<const float*>(fdes_flat.data()), static_cast<size_t>(_dataset.size()), _doc_ids);

        initialized = true;
    }

    void add_document(const std::vector<std::vector<float>>& P, const uint32_t doc_id) override {
        if (!initialized) {
            throw std::runtime_error("MuveraRetriever add_document on uninitialized index!");
        }
        std::vector<float> encoding = fde_engine->encode_document(P);
        diskann_index->insert_point(encoding.data(), doc_id);
    }

    std::vector<uint32_t> get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const override {
        if (!initialized) {
            throw std::runtime_error("MuveraRetriever get_top_k on uninitialized index!");
        }
        std::vector<float> query_encoding = fde_engine->encode_query(Q);
        // TODO: search diskann_index with the query encoding and return corresponding tags
        std::vector<uint32_t> tags(top_k);
        std::vector<float> distances(top_k);
        std::vector<float*> result_vectors;
        for (size_t i = 0; i < top_k; i++) {
            float* v = new float(embedding_dim);
            result_vectors.push_back(v);
        }
        diskann_index->search_with_tags(
            query_encoding.data(),
            static_cast<const uint32_t>(top_k),
            static_cast<const uint32_t>(75), // beam width L
            tags.data(),
            distances.data(),
            result_vectors
        );
        return tags;
    }
};