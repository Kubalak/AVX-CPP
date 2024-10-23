#include "test_utils.hpp"
#include <types/char256.hpp>
#include <types/short256.hpp>


int main(int argc, char* argv[]) {
    int result = testing::universal_test_add<avx::Char256, char>();
    result |= testing::universal_test_sub<avx::Char256, char>();
    result |= testing::universal_test_mul<avx::Char256, char>();
    result |= testing::universal_test_div<avx::Char256, char>();
    result |= testing::universal_test_mod<avx::Char256, char>();
    result |= testing::universal_test_and<avx::Char256, char>();
    result |= testing::universal_test_or<avx::Char256, char>();
    result |= testing::universal_test_xor<avx::Char256, char>();
    result |= testing::universal_test_not<avx::Char256, char>();
    result |= testing::universal_test_rshift<avx::Char256, char>();
    result |= testing::universal_test_lshift<avx::Char256, char>();
    result |= testing::universal_test_indexing<avx::Char256, char>();

    return result;
}