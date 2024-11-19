#include <iostream>
#include <types/ushort256.hpp>
#include <test_utils.hpp>

int main(int argc, char* argv[]) {

    int result = 0;

    result |= testing::universalTestAdd<avx::UShort256>();
    result |= testing::universalTestSub<avx::UShort256>();
    result |= testing::universalTestMul<avx::UShort256>();
    result |= testing::universalTestDiv<avx::UShort256>();
    result |= testing::universalTestMod<avx::UShort256>();
    result |= testing::universalTestAND<avx::UShort256>();
    result |= testing::universalTestOR<avx::UShort256>();
    result |= testing::universalTestXOR<avx::UShort256>();
    result |= testing::universalTestNOT<avx::UShort256>();
    result |= testing::universalTestLshift<avx::UShort256>();
    result |= testing::universalTestRshift<avx::UShort256>();
    result |= testing::universalTestIndexing<avx::UShort256>();

    return result;
}