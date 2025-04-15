#include <vector>

double dot_product(const std::vector<double>& h, const std::vector<double>& p, size_t dimensions);

class AbstractLSH {
    protected:
    size_t dimensions;
    size_t k_sim;

    public:
    AbstractLSH(size_t _dimensions, size_t _k_sim): dimensions(_dimensions), k_sim(_k_sim) {};
    virtual ~AbstractLSH() = default;
    virtual uint64_t compute_hash(const std::vector<double>& v) const = 0;
        // REQUIRES: v.size() == dimensions
        // ENSURES: 0 <= result < 2^k_sim
};

class SimHash : public AbstractLSH {
    private:
    std::vector<std::vector<double>> hyperplanes;

    // Generates a random Gaussian vector
    std::vector<double> generate_gaussian_vector(size_t d);

    public:
    SimHash(size_t dimensions, size_t k_sim);
    uint64_t compute_hash(const std::vector<double>& v) const;
};

// TODO: Add type templating and PQ
class AbstractChamferSimilarity {
    protected:
    size_t dimensions;

    public:
    AbstractChamferSimilarity(size_t _dimensions): dimensions(_dimensions) {};
    virtual ~AbstractChamferSimilarity() = default;

    
    // Computes (an approximation for) the chamfer similartiy between P and Q
    virtual double compute_similarity(const std::vector<std::vector<double>>& P, const std::vector<std::vector<double>>& Q) const = 0;
        // REQUIRES: p.size() == dimensions && ||p|| == 1 for any p in P
        // REQUIRES: q.size() == dimensions && ||q|| == 1 for any q in Q
};

class FDESimilarity : public AbstractChamferSimilarity {
    private:
        size_t d_proj;
        size_t B;
        size_t k_sim;
        size_t r_reps;
    
        std::vector<std::vector<std::vector<double>>> all_S;
        std::vector<SimHash> all_simhash;
    
        std::vector<std::vector<double>> get_scaled_S();
        uint64_t compute_hash_from_rep_idx(size_t idx, const std::vector<double>& v) const;
        std::vector<double> compute_proj_from_rep_idx(size_t idx, const std::vector<double>& v) const;
    
        std::vector<double> encode_document_once(size_t idx, const std::vector<std::vector<double>>& P) const;
        std::vector<double> encode_query_once(size_t idx, const std::vector<std::vector<double>>& Q) const;
    
    public:
        FDESimilarity(size_t _dimensions, size_t _d_proj, size_t _B, size_t _k_sim, size_t _r_reps);
    
        size_t get_d_fde();

        std::vector<double> encode_document(const std::vector<std::vector<double>>& P) const;
        std::vector<double> encode_query(const std::vector<std::vector<double>>& Q) const;
        double compute_similarity(
            const std::vector<std::vector<double>>& P,
            const std::vector<std::vector<double>>& Q) const;
    
        // Optional: expose this for DiskANN search
        std::vector<uint64_t> get_top_k(const std::vector<std::vector<double>>& Q, const size_t top_k) const;
};

class ExactChamferSimilarity : public AbstractChamferSimilarity {
    public:
    ExactChamferSimilarity(size_t dimensions);

    // TODO: SIMD optimizations
    // TODO: Make sure that embeddings are normalized
    double compute_similarity(const std::vector<std::vector<double>>& P, const std::vector<std::vector<double>>& Q) const;
};