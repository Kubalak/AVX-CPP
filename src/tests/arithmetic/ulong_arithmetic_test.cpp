#include <iostream>
#include <types/ulong256.hpp>
#include <test_utils.hpp>


int main(int argc, char* argv[]) {

    int result = testing::universal_test_add<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_sub<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_mul<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_div<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_mod<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_lshift<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_rshift<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_and<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_or<avx::ULong256, unsigned long long>();
    result |= testing::universal_test_xor<avx::ULong256, unsigned long long>();

    return result;
}