#include <iostream>
#include <array>
#include <string>
#include <climits>
#include <types/uint256.hpp> 
#include <test_utils.hpp>

int data_load_save(){
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    unsigned int data[] = {1, 2, 4, 5, 6, 10, 2, 5};
    unsigned int dest[8];

    int expected[] = {12, 14, 18, 20, 22, 30, 14, 20};

    avx::UInt256 val(data);

    val+= 5;
    val*= 2;

    val.save(dest);

    for(unsigned i{0}; i < 8;++i)
        if(expected[i] != dest[i]){
            result = 1;
            printf("[%u] %u <-> %u\n", i, expected[i], dest[i]);
        }
    
    return result;
}

int data_load_save_aligned(){
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    unsigned int data[] = {1, 2, 4, 5, 6, 10, 2, 5};

    alignas(32) unsigned int dest[8] ;

    unsigned expected[] = {12, 14, 18, 20, 22, 30, 14, 20};

    avx::UInt256 val(data);

    val+= 5;
    val*= 2;

    val.saveAligned(dest);

    for(unsigned i{0}; i < 8;++i)
        if(expected[i] != dest[i]){
            result = 1;
            printf("[%u] %u <-> %u\n", i, expected[i], dest[i]);
        }
    
    return result;
}

int main(int argc, char* argv[]) {
    int result = 0;

    #ifdef __AVX2__
        puts("AVX2 enabled");
    #else 
        puts("AVX2 disabled");
    #endif

    #ifdef __AVX512F__
        puts("AVX512F enabled");
    #else 
        puts("AVX512F disabled");
    #endif

    printf("Compiler %s %d.%d.%d, build date: %s %s on %s\n", 
        testing::getCompilerName(), 
        testing::getCompilerMajor(),
        testing::getCompilerMinor(), 
        testing::getCompilerPatchLevel(), 
        __DATE__, __TIME__, 
        testing::getPlatform()
    );

    result |= testing::universalTestAdd<avx::UInt256>();
    result |= testing::universalTestSub<avx::UInt256>();
    result |= testing::universalTestMul<avx::UInt256>();
    result |= testing::universalTestDiv<avx::UInt256>();
    result |= testing::universalTestMod<avx::UInt256>();
    result |= testing::universalTestAND<avx::UInt256>();
    result |= testing::universalTestOR<avx::UInt256>();
    result |= testing::universalTestXOR<avx::UInt256>();
    result |= testing::universalTestNOT<avx::UInt256>();
    result |= testing::universalTestLshift<avx::UInt256>();
    result |= testing::universalTestRshift<avx::UInt256>();
    result |= testing::universalTestIndexing<avx::UInt256>();

    result |= data_load_save();
    result |= data_load_save_aligned();


    return result;
}
