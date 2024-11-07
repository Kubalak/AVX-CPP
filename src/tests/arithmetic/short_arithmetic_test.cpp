#include <types/short256.hpp>
#include <test_utils.hpp>


int main(int argc, char* argv[]){
    int result = 0;

    result |= testing::universalTestAdd<avx::Short256>();
    result |= testing::universalTestSub<avx::Short256>();
    result |= testing::universalTestMul<avx::Short256>();
    result |= testing::universalTestDiv<avx::Short256>();
    result |= testing::universalTestMod<avx::Short256>();
    result |= testing::universalTestAND<avx::Short256>();
    result |= testing::universalTestOR<avx::Short256>();
    result |= testing::universalTestXOR<avx::Short256>();
    result |= testing::universalTestNOT<avx::Short256>();
    result |= testing::universalTestLshift<avx::Short256>();
    result |= testing::universalTestRshift<avx::Short256>();
    result |= testing::universalTestIndexing<avx::Short256>();

    return result;
}