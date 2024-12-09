#include <iostream>
#include <types/ulong256.hpp>
#include <test_utils.hpp>


int main(int argc, char* argv[]) {

    int result = 0;
    result |= testing::universalTestAdd<avx::ULong256>();
    result |= testing::universalTestSub<avx::ULong256>();
    result |= testing::universalTestMul<avx::ULong256>();
    result |= testing::universalTestDiv<avx::ULong256>();
    result |= testing::universalTestMod<avx::ULong256>();
    result |= testing::universalTestAND<avx::ULong256>();
    result |= testing::universalTestOR<avx::ULong256>();
    result |= testing::universalTestXOR<avx::ULong256>();
    result |= testing::universalTestNOT<avx::ULong256>();
    result |= testing::universalTestLshift<avx::ULong256>();
    result |= testing::universalTestRshift<avx::ULong256>();
    result |= testing::universalTestIndexing<avx::ULong256>();
    result |= testing::universalTestCompare<avx::ULong256>();

    return result;
}