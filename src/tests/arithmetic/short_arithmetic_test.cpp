#include <types/short256.hpp>
#include <test_utils.hpp>


int main(int argc, char* argv[]){
    int result = 0;

    result = testing::universal_test_add<avx::Short256, short>();
    result |= testing::universal_test_sub<avx::Short256, short>();
    result |= testing::universal_test_mul<avx::Short256, short>();
    result |= testing::universal_test_div<avx::Short256, short>();
    result |= testing::universal_test_mod<avx::Short256, short>();
    result |= testing::universal_test_and<avx::Short256, short>();
    result |= testing::universal_test_or<avx::Short256, short>();
    result |= testing::universal_test_xor<avx::Short256, short>();
    result |= testing::universal_test_not<avx::Short256, short>();
    result |= testing::universal_test_rshift<avx::Short256, short>();
    result |= testing::universal_test_lshift<avx::Short256, short>();
    result |= testing::universal_test_indexing<avx::Short256, short>();

    return result;
}