#include <iostream>
#include <types/ushort256.hpp>
#include "../test_utils.hpp"

int main(int argc, char* argv[]) {

    int result = testing::universal_test_add<avx::UShort256, unsigned short>();
    result |= testing::universal_test_sub<avx::UShort256, unsigned short>();
    result |= testing::universal_test_mul<avx::UShort256, unsigned short>();
    result |= testing::universal_test_div<avx::UShort256, unsigned short>();
    result |= testing::universal_test_mod<avx::UShort256, unsigned short>();
    result |= testing::universal_test_and<avx::UShort256, unsigned short>();
    result |= testing::universal_test_or<avx::UShort256, unsigned short>();
    result |= testing::universal_test_xor<avx::UShort256, unsigned short>();
    result |= testing::universal_test_not<avx::UShort256, unsigned short>();
    result |= testing::universal_test_lshift<avx::UShort256, unsigned short>();
    result |= testing::universal_test_rshift<avx::UShort256, unsigned short>();

    return result;
}