#include <vector>

class AbstractLSH {
    protected:
    size_t dimensions;
    size_t k_sim;

    public:
    LSH(size_t _dimensions, size_t _k_sim): dimensios(_dimensions), k_sim(_k_sim) {};
    virtual ~LSH() = default;
    virtual uint64_t compute_hash(const std::vector<double>& v) const = 0;
        // REQUIRES: v.size() == dimensions
        // ENSURES: 0 <= result < 2^k_sim
}

// TODO: Add type templating and PQ
class AbstractChamferSimilarity {
    protected:
    size_t dimensions;

    public:
    SetSimilarity(size_t _dimensions): dimensions(_dimensions) {};
    virtual ~SetSimilarity() = default;
    
    // Computes (an approximation for) the chamfer similartiy between P and Q
    virtual double compute_similarity(const std::vector<std::vector<double>>& P, const std::vector<std::vector<double>>& Q) const = 0;
        // REQUIRES: p.size() == dimensions && ||p|| == 1 for any p in P
        // REQUIRES: q.size() == dimensions && ||q|| == 1 for any q in Q
}