#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#include "fde.h"

void test_dot_product_simple() {
    std::vector<float> a = {1.0, 2.0, 3.0};
    std::vector<float> b = {4.0, -5.0, 6.0};
    float result = dot_product(a, b, 3);
    assert(std::abs(result - (1*4 + 2*(-5) + 3*6)) < 1e-9);
    std::cout << "✅ test_dot_product_simple passed\n";
}

void test_cosine_similarity_simple() {
    std::vector<float> a = {1.0, 2.0, 3.0};
    std::vector<float> b = {4.0, -5.0, 6.0};
    float result = cosine_similarity(a, b, 3);
    assert(std::abs(result - (1*4 + 2*(-5) + 3*6) / (std::sqrt(14.0) * std::sqrt(77.0))) < 1e-5);
    std::cout << "✅ test_cosine_similarity_simple passed\n";
}

void test_exact_chamfer_similarity_simple() {
    std::vector<float> a_1 = {1.0, 2.0, 3.0};
    std::vector<float> a_2 = {1.0, -2.0, 3.0};
    std::vector<float> b_1 = {4.0, 5.0, 6.0};
    std::vector<float> b_2 = {4.0, -5.0, 6.0};
    std::vector<std::vector<float>> A;
    A.resize(2);
    A[0] = a_1;
    A[1] = a_2;
    std::vector<std::vector<float>> B;
    B.resize(2);
    B[0] = b_1;
    B[1] = b_2;
    ExactChamferSimilarity exactChamferSimilarityEngine(3);
    float result = exactChamferSimilarityEngine.compute_similarity(A, B);
    assert(std::abs(result - 32.0 / (std::sqrt(14.0) * std::sqrt(77.0))) < 1e-5);
    std::cout << "✅ test_exact_chamfer_similarity_simple passed\n";
}

void test_simhash_basic() {
    SimHash simhash(3, 10, 42);
    std::vector<float> v = {1.0, 0.0, -1.0};
    uint32_t h = simhash.compute_hash(v);
    std::cout << "Hash: " << h << "\n";
    std::cout << "✅ test_simhash_basic passed\n";
}

void test_fde_basic() {
    FDESimilarity fdeSimilarityEngine(3, 128, 10240, 10, 5, 42);
    std::cout << "Initialized similarity engine" << std::endl;
    std::vector<float> a_1 = {1.0, 2.0, 3.0};
    std::vector<float> a_2 = {1.0, -2.0, 3.0};
    std::vector<float> b_1 = {4.0, 5.0, 6.0};
    std::vector<float> b_2 = {4.0, -5.0, 6.0};
    std::vector<std::vector<float>> A;
    A.resize(2);
    A[0] = a_1;
    A[1] = a_2;
    std::vector<std::vector<float>> B;
    B.resize(2);
    B[0] = b_1;
    B[1] = b_2;
    float result = fdeSimilarityEngine.compute_similarity(A, B);
    std::cout << "Similarity: " << result << "\n";
    std::cout << "✅ test_fde_basic passed\n";
}

int main() {
    test_dot_product_simple();
    test_exact_chamfer_similarity_simple();
    test_simhash_basic();
    test_fde_basic();
    return 0;
}
