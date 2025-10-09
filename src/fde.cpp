#include <bitset>
#include <random>
#include <vector>

#include <sstream>
#include <stdexcept>

#include <cmath>

#include "fde.h"
#include "index.h"
#include "index_config.h"
#include "index_factory.h"
#include "ann_exception.h"
#include "utils.h"

float dot_product(const std::vector<float>& h, const std::vector<float>& p, size_t dimensions) {
    // REQUIRES: h.size() == dimensions && p.size() == dimensions
    float result = 0.0;
    for (size_t i = 0; i < dimensions; i++)
        result += h[i] * p[i];
    return result;
};

// TODO: Implement SIMD optimizations
// TODO: Use type template to support datatypes other than float floats.
std::vector<float> SimHash::generate_gaussian_vector(size_t d) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float>dist(0.0, 1.0); // standard Gaussian
    std::vector<float> result(d);
    for (size_t i = 0; i < d; i++)
        result[i] = dist(gen);
    return result;
};

SimHash::SimHash(size_t dimensions, size_t k_sim): AbstractLSH(dimensions, k_sim) {
    hyperplanes.reserve(k_sim);
    for (size_t i = 0; i < k_sim; i++) {
        hyperplanes.push_back(generate_gaussian_vector(dimensions));
    }
};

uint32_t SimHash::compute_hash(const std::vector<float>& v) const {
    uint32_t hash = 0;
    for (size_t i = 0; i < k_sim; i++) {
        if (dot_product(hyperplanes[i], v, dimensions) >= 0) {
            hash |= (1ULL << i); // Little Endian
        }
    }
    return hash;
};

ExactChamferSimilarity::ExactChamferSimilarity(size_t dimensions): AbstractChamferSimilarity(dimensions) {};

// TODO: SIMD optimizations
// TODO: Make sure that embeddings are normalized
float ExactChamferSimilarity::compute_similarity(
    const std::vector<std::vector<float>>& P,
    const std::vector<std::vector<float>>& Q) const {
    float result = 0.0;
    // TODO: [fine grained performance engineering] :change the ordering of
    //       this iteration based on the relative size of P and Q
    for (auto p: P) {
        float best = 0.0;
        for (auto q: Q) {
            float c = dot_product(p, q, dimensions);
            best = c > best ? c : best;
        }
        result += best;
    }
    return result;
};


std::vector<std::vector<float>> FDESimilarity::get_scaled_S() { // (1 / sqrt(d_proj))S
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> binary_dist(0, 1);

    float scale = 1.0 / std::sqrt(d_proj);
    std::vector<std::vector<float>> result(d_proj, std::vector<float>(dimensions));

    for (int i = 0; i < d_proj; i++) {
        for (int j = 0; j < dimensions; j++) {
            int sign = binary_dist(gen) ? 1 : -1;
            result[i][j] = sign * scale;
        }
    }
    return result;
};

void FDESimilarity::initialize_scaled_S_AMS(uint64_t base_seed) {
    all_S_sparse.clear();
    all_S_sparse.reserve(r_reps);

    for (size_t rep_id = 0; rep_id < r_reps; rep_id++) {
        std::vector<int32_t> S_index(dimensions);
        std::vector<int8_t>  S_sign(dimensions);

        std::mt19937 gen(base_seed + rep_id);
        std::uniform_int_distribution<int32_t> index_dist(0, static_cast<int32_t>(d_proj - 1));
        std::uniform_int_distribution binary_dist(0, 1);

        for (size_t i = 0; i < dimensions; ++i) {
            S_index[i] = index_dist(gen);
            S_sign[i]  = binary_dist(gen) ? 1 : -1;
        }

        all_S_sparse.emplace_back(std::move(S_index), std::move(S_sign));
    }
}

std::vector<float> FDESimilarity::apply_countsketch(const std::vector<float>& v) const {
    assert(v.size() == d_fde);

    std::vector<float> v_final(d_final, 0.0f);

    for (size_t i = 0; i < d_fde; i++) {
        v_final[countsketch_index[i]] += countsketch_sign[i] * v[i];
    }
    return v_final;
}

std::vector<float> FDESimilarity::apply_ams(
    const std::vector<float>& v, size_t rep_id) const
{
    assert(rep_id < all_S_sparse.size());
    assert(v.size() == dimensions);
    const auto& [S_index, S_sign] = all_S_sparse[rep_id];

    std::vector<float> result(d_proj, 0.0f);

    for (size_t i = 0; i < dimensions; ++i) {
        result[i] = S_sign[i] * v[S_index[i]] / std::sqrt(static_cast<float>(d_proj));
    }

    return result;
}

uint32_t FDESimilarity::compute_hash_from_rep_idx(size_t idx, const std::vector<float>& v) const {
    // REQUIRES: 0 <= idx < r_reps && v.size() == dimensions
    return all_simhash[idx].compute_hash(v);
};

std::vector<float> FDESimilarity::compute_proj_from_rep_idx(size_t idx, const std::vector<float> &v) const {
    // TODO: Perf engineer this
    // REQUIRES: d_proj == all_S[idx].size();
    // REQUIRES: dimensions == v.size() == all_S[0][0].size()

    std::vector<float> result(d_proj, 0.0);
    for (size_t i = 0; i < d_proj; i++) {
        result[i] = dot_product(all_S[idx][i], v, dimensions);
    }
    return result;
};


std::vector<float> FDESimilarity::encode_document_once(size_t idx, const std::vector<std::vector<float>> &P) const {
    // idx is the repetition index
    std::vector<std::vector<float>> P_hash_grouped;
    std::vector<size_t> bucket_counts(B, 0);
    P_hash_grouped.resize(B);
    for (size_t i = 0; i < B; i++)
        P_hash_grouped[i] = std::vector<float>(dimensions, 0.0);
    for (auto p: P) {
        uint32_t hash_value = compute_hash_from_rep_idx(idx, p);
        bucket_counts[hash_value] ++;
        // TODO: float-check the type conversion here
        for (size_t j = 0; j < dimensions; j++) {
            P_hash_grouped[hash_value][j] = P_hash_grouped[hash_value][j] * (bucket_counts[hash_value] - 1 ) + p[j];
            P_hash_grouped[hash_value][j] /= bucket_counts[hash_value];
        }
    }
    std::vector<float> P_phi;
    P_phi.reserve(B * d_proj);
    for (size_t i = 0; i < B; i++) {
        std::vector<float> projection = use_ams ? apply_ams(P_hash_grouped[i], idx) : compute_proj_from_rep_idx(idx, P_hash_grouped[i]);
        P_phi.insert(P_phi.end(), projection.begin(), projection.end());
    }
    return P_phi;
};

std::vector<float> FDESimilarity::encode_query_once(size_t idx, const std::vector<std::vector<float>> &Q) const {
    // idx is the repetition index
    std::vector<std::vector<float>> Q_hash_grouped;
    Q_hash_grouped.resize(B);
    for (size_t i = 0; i < B; i++)
        Q_hash_grouped[i] = std::vector<float>(dimensions, 0.0);
    for (auto q: Q) {
        uint32_t hash_value = compute_hash_from_rep_idx(idx, q);
        // TODO: float-check the type conversion here
        for (size_t j = 0; j < dimensions; j++) Q_hash_grouped[hash_value][j] += q[j];
    }
    std::vector<float> Q_phi;
    Q_phi.reserve(B * d_proj);
    for (size_t i = 0; i < B; i++) {
        std::vector<float> projection = use_ams ? apply_ams(Q_hash_grouped[i], idx) : compute_proj_from_rep_idx(idx, Q_hash_grouped[i]);
        Q_phi.insert(Q_phi.end(), projection.begin(), projection.end());
    }
    return Q_phi;
}
    
size_t FDESimilarity::get_d_fde() {
    return d_fde;
};

std::vector<float> FDESimilarity::encode_document(const std::vector<std::vector<float>> &P) const {
    // TODO: implement fill_empty_clusters
    std::vector<float> result;
    result.reserve(d_fde);
    for(size_t idx = 0; idx < r_reps; idx++) {
        std::vector<float> trial = encode_query_once(idx, P);
        result.insert(result.end(), trial.begin(), trial.end());
    }
    return apply_countsketch(result);
};

std::vector<float> FDESimilarity::encode_query(const std::vector<std::vector<float>> &Q) const {
    std::vector<float> result;
    result.reserve(d_fde);
    for(size_t idx = 0; idx < r_reps; idx++) {
        std::vector<float> trial = encode_query_once(idx, Q);
        result.insert(result.end(), trial.begin(), trial.end());
    }
    return apply_countsketch(result);
};


FDESimilarity::FDESimilarity(const size_t _dimensions, const size_t _d_proj, const size_t _d_final,
    const size_t _B, const size_t _k_sim, const size_t _r_reps
): AbstractChamferSimilarity(_dimensions), d_proj(_d_proj), d_final(_d_final), B(_B),
k_sim(_k_sim), r_reps(_r_reps) {
    all_simhash.reserve(r_reps);
    all_S.reserve(r_reps);
    for (size_t i = 0; i < r_reps; i++) {
        all_simhash.emplace_back(dimensions, k_sim);
        all_S.push_back(get_scaled_S());
    }
    initialize_scaled_S_AMS();
    d_fde = B * d_proj * r_reps;

    // Initialize CountSketch
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> index_dist(0, d_final - 1);
    std::uniform_int_distribution<int> binary_dist(0, 1);

    countsketch_index.resize(d_fde);
    countsketch_sign.resize(d_fde);
    for (size_t i = 0; i < d_fde; i++) {
        countsketch_index[i] = index_dist(gen);
        countsketch_sign[i] = binary_dist(gen) ? 1 : -1;
    }
};

float FDESimilarity::compute_similarity(
    // TODO: implement final projections
    const std::vector<std::vector<float>>& P,
    const std::vector<std::vector<float>>& Q) const {
    return dot_product(encode_document(P), encode_query(Q), d_final);
};
