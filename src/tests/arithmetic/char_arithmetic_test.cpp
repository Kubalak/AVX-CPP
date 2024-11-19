#include "test_utils.hpp"
#include <types/char256.hpp>
#include <ops/avxmath.hpp>


int main(int argc, char* argv[]) {
    int result = testing::universalTestAdd<avx::Char256>();
    result |= testing::universalTestSub<avx::Char256>();
    result |= testing::universalTestMul<avx::Char256>();
    result |= testing::universalTestDiv<avx::Char256>();
    result |= testing::universalTestMod<avx::Char256>();
    result |= testing::universalTestAND<avx::Char256>();
    result |= testing::universalTestOR<avx::Char256>();
    result |= testing::universalTestXOR<avx::Char256>();
    result |= testing::universalTestNOT<avx::Char256>();
    result |= testing::universalTestRshift<avx::Char256>();
    result |= testing::universalTestLshift<avx::Char256>();
    result |= testing::universalTestIndexing<avx::Char256>();

    return result;
}