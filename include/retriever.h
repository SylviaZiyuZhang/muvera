#include <unordered_set>
#include <vector>

class AbstractRetriever {
    protected:
    size_t dimensions;
    bool initialized;
    std::unordered_set<uint64_t> document_ids;

    public:
    AbstractRetriever(const size_t _dimensions): dimensions(_dimensions) {
        initialized = false;
        document_ids = std::unordered_set<uint64_t>(0); // DiskANN uses 0 as start point.
    };
    virtual ~AbstractRetriever() = default;

    // Initializes the retriever with the dataset
    virtual void index_dataset(const std::vector<std::vector<std::vector<double>>>& dataset, const std::vector<uint64_t> doc_ids) const = 0;
    // REQUIRES: doc_id > 0 for each doc_id : doc_ids
    // REQUIRES: dataset.size() == doc_ids.size()
    // ENSURES: initialized

    // Adds a document into the retriever.
    virtual void add_document(const std::vector<std::vector<double>>& P, const uint64_t doc_id) const = 0;
    // REQUIRES: doc_id > 0

    // Retrieves the top k documents based on a query.
    virtual std::vector<uint64_t> get_top_k(const std::vector<std::vector<double>>& Q, const size_t top_k) const = 0;
};
