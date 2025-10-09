#include <vector>

float dot_product(const std::vector<float>& h, const std::vector<float>& p, size_t dimensions);

class AbstractLSH {
    protected:
    size_t dimensions;
    size_t k_sim;

    public:
    AbstractLSH(size_t _dimensions, size_t _k_sim): dimensions(_dimensions), k_sim(_k_sim) {};
    virtual ~AbstractLSH() = default;
    virtual uint32_t compute_hash(const std::vector<float>& v) const = 0;
        // REQUIRES: v.size() == dimensions
        // ENSURES: 0 <= result < 2^k_sim
};

class SimHash : public AbstractLSH {
    private:
    std::vector<std::vector<float>> hyperplanes;

    // Generates a random Gaussian vector
    std::vector<float> generate_gaussian_vector(size_t d);

    public:
    SimHash(size_t dimensions, size_t k_sim);
    uint32_t compute_hash(const std::vector<float>& v) const;
};

// TODO: Add type templating and PQ
class AbstractChamferSimilarity {
    protected:
    size_t dimensions; // d

    public:
    AbstractChamferSimilarity(size_t _dimensions): dimensions(_dimensions) {};
    virtual ~AbstractChamferSimilarity() = default;

    
    // Computes (an approximation for) the chamfer similartiy between P and Q
    virtual float compute_similarity(const std::vector<std::vector<float>>& P, const std::vector<std::vector<float>>& Q) const = 0;
        // REQUIRES: p.size() == dimensions && ||p|| == 1 for any p in P
        // REQUIRES: q.size() == dimensions && ||q|| == 1 for any q in Q
};

class FDESimilarity : public AbstractChamferSimilarity {
    private:
        size_t d_proj;
        size_t d_final;
        size_t d_fde;
        size_t B;
        size_t k_sim;
        size_t r_reps;
        bool use_ams = true;
    
        std::vector<std::vector<std::vector<float>>> all_S; // dense random matrices

         // sparse random matrices if using AMS per-block projection
        std::vector<std::pair<std::vector<int32_t>, std::vector<int8_t>>> all_S_sparse;
        std::vector<SimHash> all_simhash;

        std::vector<int32_t> countsketch_index; // d_fde -> d_final
        std::vector<int8_t> countsketch_sign; // ±1
    
        std::vector<std::vector<float>> get_scaled_S();
        void initialize_scaled_S_AMS(uint64_t base_seed = 42);
        
        // Use CountSketch for the final projection as in the google graph mining
        // implementation. The paper describes a dense random matrix but CountSketch
        // also preserves the necessary theoretical guarantees.
        std::vector<float> apply_countsketch(const std::vector<float>& v) const;

        // For using AMS instead of dense random matrix during per-block projection.
        std::vector<float> apply_ams(const std::vector<float>& v, size_t rep_id) const;

        uint32_t compute_hash_from_rep_idx(size_t idx, const std::vector<float>& v) const;
        std::vector<float> compute_proj_from_rep_idx(size_t idx, const std::vector<float>& v) const;
    
        std::vector<float> encode_document_once(size_t idx, const std::vector<std::vector<float>>& P) const;
        std::vector<float> encode_query_once(size_t idx, const std::vector<std::vector<float>>& Q) const;
    
    public:
        FDESimilarity(size_t _dimensions, size_t _d_proj, size_t _d_final, size_t _B, size_t _k_sim, size_t _r_reps);
    
        size_t get_d_fde();

        std::vector<float> encode_document(const std::vector<std::vector<float>>& P) const;
        std::vector<float> encode_query(const std::vector<std::vector<float>>& Q) const;
        float compute_similarity(
            const std::vector<std::vector<float>>& P,
            const std::vector<std::vector<float>>& Q) const;
};

class ExactChamferSimilarity : public AbstractChamferSimilarity {
    public:
    ExactChamferSimilarity(size_t dimensions);

    // TODO: SIMD optimizations
    // TODO: Make sure that embeddings are normalized
    float compute_similarity(const std::vector<std::vector<float>>& P, const std::vector<std::vector<float>>& Q) const;
};