#include <iostream>
#include <vector>
#include <cassert>
#include "../src/fde.cpp"

void test_dot_product() {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, -5.0, 6.0};
    double result = dot_product(a, b, 3);
    assert(std::abs(result - (1*4 + 2*(-5) + 3*6)) < 1e-9);
    std::cout << "✅ test_dot_product passed\n";
}

void test_simhash_basic() {
    SimHash simhash(3, 10);
    std::vector<double> v = {1.0, 0.0, -1.0};
    uint64_t h = simhash.compute_hash(v);
    std::cout << "Hash: " << h << "\n";
    std::cout << "✅ test_simhash_basic passed\n";
}

int main() {
    test_dot_product();
    test_simhash_basic();
    return 0;
}
