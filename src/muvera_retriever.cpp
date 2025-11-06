#include <immintrin.h>

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "fde.h"
#include "retriever.h"


MuveraRetriever::MuveraRetriever(const size_t _dimensions, const size_t _max_points, const size_t _d_proj, const size_t _d_final,
    const size_t _k_sim, const size_t _r_reps, const uint64_t _seed
): AbstractRetriever(_dimensions, _max_points) {
    const size_t L = 128;
    const size_t R = 64;
    const size_t Lf = 128;
    const float alpha = 1.2;
    const size_t num_threads = 8;
    
    fde_engine = std::make_unique<FDESimilarity>(_dimensions, _d_proj, _d_final, _k_sim, _r_reps, _seed);
    embedding_dim = _d_final; // Without the final projection, this would be fde_engine->get_d_fde()

    diskann::IndexWriteParameters index_build_params =
        diskann::IndexWriteParametersBuilder(L, R)
            .with_filter_list_size(Lf)
            .with_alpha(alpha)
            .with_saturate_graph(false)
            .with_num_threads(num_threads)
            .build();

    diskann::IndexConfig config = diskann::IndexConfigBuilder()
        .with_metric(diskann::Metric::COSINE)
        .with_dimension(embedding_dim) // TODO: change this to final projection dimension after final projection is implemented
        .with_max_points(max_points)
        .is_dynamic_index(true)
        .with_index_write_params(index_build_params)
        .is_enable_tags(true)
        .is_use_opq(true)
        .is_pq_dist_build(false)
        .with_data_type("float")
        .build();
    diskann::IndexFactory index_factory(config);
    diskann_index = index_factory.create_instance();
};


void MuveraRetriever::index_dataset(const std::vector<std::vector<std::vector<float>>>& _dataset, const std::vector<std::string> _doc_ids)
{
    if (_dataset.size() != _doc_ids.size()) {
        throw std::runtime_error("MuveraRetriever.index_dataset: dataset and doc_ids have different sizes.");
    }
    const size_t total_size = _dataset.size() * embedding_dim;

    auto deleter = [](float* p){ std::free(p); };
    std::unique_ptr<float[], decltype(deleter)> fdes_aligned(
        static_cast<float*>(std::aligned_alloc(32, total_size * sizeof(float))),
        deleter
    );

    size_t offset = 0;
    for(const auto& P : _dataset) {
        std::vector<float> embedding = fde_engine->encode_document(P);
        if (embedding.size() != embedding_dim) {
            throw std::runtime_error("MuveraRetriever.index_dataset: embedding dimension mismatch.");
        }
        std::memcpy(fdes_aligned.get() + offset, embedding.data(), embedding_dim * sizeof(float));
        offset += embedding_dim;
    }

    std::any any_data = std::any(static_cast<const float*>(fdes_aligned.get()));  // Store in std::any
    
    // Sanity check for std::any casting.
    // try {
    //     const float* casted_data = std::any_cast<const float*>(any_data);
    // } catch (const std::bad_any_cast &e) {
    //     std::cout << "Bad any cast: " << e.what() << std::endl;
    // }
    std::vector<uint32_t> num_doc_ids = std::vector<uint32_t>();
    num_doc_ids.reserve(_doc_ids.size());
    for (uint32_t i = doc_ids.size(); i < doc_ids.size() + _doc_ids.size(); i++) {
        num_doc_ids.push_back(i);
    }
    doc_ids.insert(doc_ids.end(), _doc_ids.begin(), _doc_ids.end());


    diskann_index->build(static_cast<const float*>(fdes_aligned.get()), static_cast<size_t>(_dataset.size()), num_doc_ids);

    initialized = true;
}

void MuveraRetriever::add_document(const std::vector<std::vector<float>>& P, const std::string doc_id) {
    if (!initialized) {
        throw std::runtime_error("MuveraRetriever add_document on uninitialized index!");
    }
    std::vector<float> encoding = fde_engine->encode_document(P);
    doc_ids.push_back(doc_id);
    diskann_index->insert_point(encoding.data(), doc_ids.size() - 1);
}

std::vector<std::string> MuveraRetriever::get_top_k(const std::vector<std::vector<float>>& Q, const size_t top_k) const {
    if (!initialized) {
        throw std::runtime_error("MuveraRetriever get_top_k on uninitialized index!");
    }
    std::vector<float> query_encoding = fde_engine->encode_query(Q);
    // TODO: search diskann_index with the query encoding and return corresponding tags
    std::vector<uint32_t> tags(top_k);
    std::vector<float> distances(top_k);
    std::vector<float*> result_vectors;
    for (size_t i = 0; i < top_k; i++) {
        float* v = new float[embedding_dim];
        result_vectors.push_back(v);
    }
    diskann_index->search_with_tags(
        query_encoding.data(),
        static_cast<const uint32_t>(top_k),
        static_cast<const uint32_t>(top_k), // beam width L
        tags.data(),
        distances.data(),
        result_vectors
    );
    for (auto v : result_vectors) delete[] v;
    std::vector<std::string> final_result = std::vector<std::string>();
    for (auto ri : tags) {
        final_result.push_back(doc_ids[ri]);
    }
    return final_result;
}