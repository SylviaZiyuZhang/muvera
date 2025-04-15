#include <iostream>
#include <vector>
#include <cassert>

#include "fde.h"
#include "retriever.h"

void test_exact_chamfer_retriever_simple() {
    std::vector<double> a_1 = {1.0, 2.0, 3.0};
    std::vector<double> a_2 = {1.0, -2.0, 3.0};
    std::vector<double> b_1 = {4.0, 5.0, 6.0};
    std::vector<double> b_2 = {4.0, -5.0, 6.0};
    std::vector<std::vector<double>> A;
    A.resize(2);
    A[0] = a_1;
    A[1] = a_2;
    std::vector<std::vector<double>> B;
    B.resize(2);
    B[0] = b_1;
    B[1] = b_2;
    std::vector<std::vector<std::vector<double>>> dataset;
    dataset.push_back(A);
    dataset.push_back(B);
    std::vector<uint64_t> doc_ids = {1, 2};
    ExactChamferRetriever exactChamferRetriever(3);
    exactChamferRetriever.index_dataset(dataset, doc_ids);
    std::vector<uint64_t> result = exactChamferRetriever.get_top_k(A, 1);
    assert(result.size() == 1);
    assert(result[0] == 1);
    std::cout << "✅ test_exact_chamfer_retriever_simple passed\n";
}

int main() {
    test_exact_chamfer_retriever_simple();
    return 0;
}
