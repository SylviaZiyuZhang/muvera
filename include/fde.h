#include <vector>

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

// TODO: Add type templating and PQ
class AbstractChamferSimilarity {
    protected:
    size_t dimensions;

    public:
    AbstractChamferSimilarity(size_t _dimensions): dimensions(_dimensions) {};
    virtual ~AbstractChamferSimilarity() = default;

    void index_document(const std::vector<std::vector<double>>& P, const uint64_t doc_id) const = 0;
    
    // Computes (an approximation for) the chamfer similartiy between P and Q
    virtual double compute_similarity(const std::vector<std::vector<double>>& P, const std::vector<std::vector<double>>& Q) const = 0;
        // REQUIRES: p.size() == dimensions && ||p|| == 1 for any p in P
        // REQUIRES: q.size() == dimensions && ||q|| == 1 for any q in Q
};