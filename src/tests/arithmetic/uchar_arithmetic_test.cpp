#include <iostream>
#include <types/uchar256.hpp>
#include <test_utils.hpp>

int main(int argc, char* argv[]) {

    int result = 0;

    result |= testing::universalTestAdd<avx::UChar256>();
    result |= testing::universalTestSub<avx::UChar256>();
    result |= testing::universalTestMul<avx::UChar256>();
    result |= testing::universalTestDiv<avx::UChar256>();
    result |= testing::universalTestMod<avx::UChar256>();
    result |= testing::universalTestAND<avx::UChar256>();
    result |= testing::universalTestOR<avx::UChar256>();
    result |= testing::universalTestXOR<avx::UChar256>();
    result |= testing::universalTestNOT<avx::UChar256>();
    result |= testing::universalTestLshift<avx::UChar256>();
    result |= testing::universalTestRshift<avx::UChar256>();
    result |= testing::universalTestIndexing<avx::UChar256>();
    result |= testing::universalTestCompare<avx::UChar256>();
    
    return result;
}