#include <iostream>
#include <types/uchar256.hpp>
#include <test_utils.hpp>

int main(int argc, char* argv[]) {

    int result = testing::universal_test_add<avx::UChar256, unsigned char>();
    result |= testing::universal_test_sub<avx::UChar256, unsigned char>();
    result |= testing::universal_test_mul<avx::UChar256, unsigned char>();
    result |= testing::universal_test_div<avx::UChar256, unsigned char>();
    
    
    return result;
}