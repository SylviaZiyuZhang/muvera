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

double dot_product(const std::vector<double>& h, const std::vector<double>& p, size_t dimensions) {
    // REQUIRES: h.size() == dimensions && p.size() == dimensions
    double result = 0.0;
    for (size_t i = 0; i < dimensions; i++)
        result += h[i] * p[i];
    return result;
};

// TODO: Implement SIMD optimizations
// TODO: Use type template to support datatypes other than double floats.
class SimHash : public AbstractLSH {
    private:
    std::vector<std::vector<double>> hyperplanes;

    // Generates a random Gaussian vector
    std::vector<double> generate_gaussian_vector(size_t d) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double>dist(0.0, 1.0); // standard Gaussian
        std::vector<double> result(d);
        for (size_t i = 0; i < d; i++)
            result[i] = dist(gen);
        return result;
    }

    public:
    SimHash(size_t dimensions, size_t k_sim): AbstractLSH(dimensions, k_sim) {
        hyperplanes.reserve(k_sim);
        for (size_t i = 0; i < k_sim; i++) {
            hyperplanes.push_back(generate_gaussian_vector(dimensions));
        }
    }

    uint64_t compute_hash(const std::vector<double>& v) const override {
        uint64_t hash = 0;
        for (size_t i = 0; i < k_sim; i++) {
            if (dot_product(hyperplanes[i], v, dimensions) >= 0) {
                hash |= (1ULL << i); // Little Endian
            }
        }
        return hash;
    }
};

class ExactChamferSimilarity : public AbstractChamferSimilarity {
    public:
    ExactChamferSimilarity(size_t dimensions): AbstractChamferSimilarity(dimensions) {};

    // TODO: SIMD optimizations
    // TODO: Make sure that embeddings are normalized
    double compute_similarity(
        const std::vector<std::vector<double>>& P,
        const std::vector<std::vector<double>>& Q) const override {
        double result = 0.0;
        // TODO: [fine grained performance engineering] :change the ordering of
        //       this iteration based on the relative size of P and Q
        for (auto p: P) {
            double best = 0.0;
            for (auto q: Q) {
                double c = dot_product(p, q, dimensions);
                best = c > best ? c : best;
            }
            result += best;
        }
        return result;
    }
};

class FDESimilarity : public AbstractChamferSimilarity {
    private:
    // TODO: Setup hyperparameters in more intellligent ways
    size_t d_proj = 128;
    size_t B = 1024; // Number of buckets, 2^k_sim.
    size_t k_sim = 10;
    size_t r_reps = 5; // Number of trials for LSH.

    std::vector<std::vector<std::vector<double>>> all_S;
    std::vector<SimHash> all_simhash;

    std::vector<std::vector<double>> get_scaled_S() { // (1 / sqrt(d_proj))S
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> binary_dist(0, 1);

        double scale = 1.0 / std::sqrt(d_proj);
        std::vector<std::vector<double>> result(d_proj, std::vector<double>(dimensions));

        for (int i = 0; i < d_proj; ++i) {
            for (int j = 0; j < dimensions; ++j) {
                int sign = binary_dist(gen) ? 1 : -1;
                result[i][j] = sign * scale;
            }
        }
        return result;
    }

    uint64_t compute_hash_from_rep_idx(size_t idx, const std::vector<double>& v) const {
        // REQUIRES: 0 <= idx < r_reps && v.size() == dimensions
        return all_simhash[idx].compute_hash(v);
    }

    std::vector<double> compute_proj_from_rep_idx(size_t idx, const std::vector<double> &v) const {
        // TODO: Perf engineer this
        // REQUIRES: d_proj == all_S[idx].size();
        // REQUIRES: dimensions == v.size() == all_S[0][0].size()

        std::vector<double> result(d_proj, 0.0);
        for (size_t i = 0; i < d_proj; i++) {
            result[i] = dot_product(all_S[idx][i], v, dimensions);
        }
        return result;
    }


    std::vector<double> encode_document_once(size_t idx, const std::vector<std::vector<double>> &P) const {
        // idx is the repetition index
        std::vector<std::vector<double>> P_hash_grouped;
        std::vector<size_t> bucket_counts(B, 0);
        P_hash_grouped.resize(B);
        for (size_t i = 0; i < B; i++)
            P_hash_grouped[i] = std::vector<double>(dimensions, 0.0);
        for (auto p: P) {
            uint64_t hash_value = compute_hash_from_rep_idx(idx, p);
            bucket_counts[hash_value] ++;
            // TODO: Double-check the type conversion here
            for (size_t j = 0; j < dimensions; j++) {
                P_hash_grouped[hash_value][j] = P_hash_grouped[hash_value][j] * (bucket_counts[hash_value] - 1 ) + p[j];
                P_hash_grouped[hash_value][j] /= bucket_counts[hash_value];
            }
        }
        std::vector<double> P_phi;
        P_phi.reserve(B * d_proj);
        for (size_t i = 0; i < B; i++) {
            std::vector<double> projection = compute_proj_from_rep_idx(idx, P_hash_grouped[i]);
            P_phi.insert(P_phi.end(), projection.begin(), projection.end());
        }
        return P_phi;
    }

    std::vector<double> encode_query_once(size_t idx, const std::vector<std::vector<double>> &Q) const {
        // idx is the repetition index
        std::vector<std::vector<double>> Q_hash_grouped;
        Q_hash_grouped.resize(B);
        for (size_t i = 0; i < B; i++)
            Q_hash_grouped[i] = std::vector<double>(dimensions, 0.0);
        for (auto q: Q) {
            uint64_t hash_value = compute_hash_from_rep_idx(idx, q);
            // TODO: Double-check the type conversion here
            for (size_t j = 0; j < dimensions; j++) Q_hash_grouped[hash_value][j] += q[j];
        }
        std::vector<double> Q_phi;
        Q_phi.reserve(B * d_proj);
        for (size_t i = 0; i < B; i++) {
            std::vector<double> projection = compute_proj_from_rep_idx(idx, Q_hash_grouped[i]);
            Q_phi.insert(Q_phi.end(), projection.begin(), projection.end());
        }
        return Q_phi;
    }
    
    protected:
    size_t get_d_fde() {
        return B * d_proj * r_reps;
    }
    std::vector<double> encode_document(const std::vector<std::vector<double>> &P) const {
        // TODO: implement fill_empty_clusters
        std::vector<double> result;
        result.reserve(B * d_proj * r_reps);
        for(size_t idx = 0; idx < r_reps; idx++) {
            std::vector<double> trial = encode_query_once(idx, P);
            result.insert(result.end(), trial.begin(), trial.end());
        }
        return result;
    }

    std::vector<double> encode_query(const std::vector<std::vector<double>> &Q) const {
        std::vector<double> result;
        result.reserve(B * d_proj * r_reps);
        for(size_t idx = 0; idx < r_reps; idx++) {
            std::vector<double> trial = encode_query_once(idx, Q);
            result.insert(result.end(), trial.begin(), trial.end());
        }
        return result;
    }


    public:
    MuveraSimilarity(const size_t _dimensions, const size_t _d_proj,
        const size_t _B, const size_t _k_sim, const size_t r_reps
    ): AbstractChamferSimilarity(_dimensions), d_proj(_d_proj), B(_B),
    k_sim(_k_sim), r_reps(_r_reps) {
        all_simhash.reserve(r_reps);
        all_S.reserve(r_reps);
        for (size_t i = 0; i < r_reps; i++) {
            all_simhash.emplace_back(dimensions, k_sim);
            all_S.push_back(get_scaled_S());
        }
    };

    double compute_similarity(
        // TODO: implement final projections
        const std::vector<std::vector<double>>& P,
        const std::vector<std::vector<double>>& Q) const override {
        return dot_product(encode_document(P), encode_query(Q), B * d_proj * r_reps);
    }
};