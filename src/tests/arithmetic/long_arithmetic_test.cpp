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

    result |= testing::universalTestAdd<avx::Long256>();
    result |= testing::universalTestSub<avx::Long256>();
    result |= testing::universalTestMul<avx::Long256>();
    result |= testing::universalTestDiv<avx::Long256>();
    result |= testing::universalTestMod<avx::Long256>();
    result |= testing::universalTestAND<avx::Long256>();
    result |= testing::universalTestOR<avx::Long256>();
    result |= testing::universalTestXOR<avx::Long256>();
    result |= testing::universalTestNOT<avx::Long256>();
    result |= testing::universalTestLshift<avx::Long256>();
    result |= testing::universalTestRshift<avx::Long256>();
    result |= testing::universalTestIndexing<avx::Long256>();
    result |= testing::universalTestCompare<avx::Long256>();
    
    result |= data_load_save();
    result |= data_load_save_aligned();

    return result;
}