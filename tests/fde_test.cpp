#include <iostream>
#include <vector>
#include <cassert>
#include "../src/fde.cpp"

void test_dot_product_simple() {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, -5.0, 6.0};
    double result = dot_product(a, b, 3);
    assert(std::abs(result - (1*4 + 2*(-5) + 3*6)) < 1e-9);
    std::cout << "✅ test_dot_product_simple passed\n";
}

void test_exact_chamfer_similarity_simple() {
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
    ExactChamferSimilarity exactChamferSimilarityEngine(3);
    double result = exactChamferSimilarityEngine.compute_similarity(A, B);
    assert(std::abs(result - 64.0) < 1e-9);
    std::cout << "✅ test_exact_chamfer_similarity_simple passed\n";
}

void test_simhash_basic() {
    SimHash simhash(3, 10);
    std::vector<double> v = {1.0, 0.0, -1.0};
    uint64_t h = simhash.compute_hash(v);
    std::cout << "Hash: " << h << "\n";
    std::cout << "✅ test_simhash_basic passed\n";
}

void test_muvera_basic() {
    MuveraSimilarity muveraSimilarityEngine(3);
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
    double result = muveraSimilarityEngine.compute_similarity(A, B);
    std::cout << "Similarity: " << result << "\n";
    std::cout << "✅ test_muvera_basic passed\n";
}

int main() {
    test_dot_product_simple();
    test_exact_chamfer_similarity_simple();
    test_simhash_basic();
    test_muvera_basic();
    return 0;
}
