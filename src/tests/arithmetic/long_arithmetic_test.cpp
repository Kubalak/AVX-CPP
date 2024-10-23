#include <iostream>
#include <array>
#include <types/long256.hpp>
#include <test_utils.hpp>

int data_load_save(){
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    long long data[] = {1, 2, 4, 5};
    long long dest[4];

    long long expected[] = {12, 14, 18, 20};

    avx::Long256 val(data);

    val+= 5;
    val*= 2;

    val.save(dest);

    for(unsigned i{0}; i < 4;++i)
        if(expected[i] != dest[i]){
            result = 1;
            printf("[%u] %lld <-> %lld\n", i, expected[i], dest[i]);
        }
    
    return result;
}

int data_load_save_aligned(){
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    long long data[] = {1, 2, 4, 5};

#ifdef _MSC_VER
    __declspec(align(32)) long long dest[4] ;
#else
    long long dest[4] __attribute__((aligned(32)));
#endif

    long long expected[] = {12, 14, 18, 20};

    avx::Long256 val(data);

    val+= 5;
    val*= 2;

    val.saveAligned(dest);

    for(unsigned i{0}; i < 4;++i)
        if(expected[i] != dest[i]){
            result = 1;
            printf("[%u] %lld <-> %lld\n", i, expected[i], dest[i]);
        }
    
    return result;
}

int main(int argc, char* argv[]) {
    int result = 0;

    result |= testing::universal_test_add<avx::Long256, long long>();
    result |= testing::universal_test_sub<avx::Long256, long long>();
    result |= testing::universal_test_mul<avx::Long256, long long>();
    result |= testing::universal_test_div<avx::Long256, long long>();
    result |= testing::universal_test_mod<avx::Long256, long long>();
    result |= testing::universal_test_and<avx::Long256, long long>();
    result |= testing::universal_test_or<avx::Long256, long long>();
    result |= testing::universal_test_xor<avx::Long256, long long>();
    result |= testing::universal_test_not<avx::Long256, long long>();
    result |= testing::universal_test_lshift<avx::Long256, long long>();
    result |= testing::universal_test_rshift<avx::Long256, long long>();
    result |= testing::universal_test_indexing<avx::Long256, long long>();
    
    result |= data_load_save();
    result |= data_load_save_aligned();

    return result;
}