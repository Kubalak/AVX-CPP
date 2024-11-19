#include <types/int256.hpp>
#include <test_utils.hpp>
#include <iostream>


int data_load_save(){
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    int data[] = {1, 2, 4, 5, 6, 10, 2, 5};
    int dest[8];

    int expected[] = {12, 14, 18, 20, 22, 30, 14, 20};

    avx::Int256 val(data);

    val+= 5;
    val*= 2;

    val.save(dest);

    for(unsigned i{0}; i < 8;++i)
        if(expected[i] != dest[i]){
            result = 1;
            printf("[%u] %d <-> %d\n", i, expected[i], dest[i]);
        }
    
    return result;
}

int data_load_save_aligned(){
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    int data[] = {1, 2, 4, 5, 6, 10, 2, 5};

#ifdef _MSC_VER
    __declspec(align(32)) int dest[8] ;
#else
    int dest[8] __attribute__((aligned(32)));
#endif

    int expected[] = {12, 14, 18, 20, 22, 30, 14, 20};

    avx::Int256 val(data);

    val+= 5;
    val*= 2;

    val.saveAligned(dest);

    for(unsigned i{0}; i < 8;++i)
        if(expected[i] != dest[i]){
            result = 1;
            printf("[%u] %d <-> %d\n", i, expected[i], dest[i]);
        }
    
    return result;
}


int main(int argc, char* argv[]) {
    int result = 0;
    
    result |= testing::universalTestAdd<avx::Int256>();
    result |= testing::universalTestSub<avx::Int256>();
    result |= testing::universalTestMul<avx::Int256>();
    result |= testing::universalTestDiv<avx::Int256>();
    result |= testing::universalTestMod<avx::Int256>();
    result |= testing::universalTestAND<avx::Int256>();
    result |= testing::universalTestOR<avx::Int256>();
    result |= testing::universalTestXOR<avx::Int256>();
    result |= testing::universalTestNOT<avx::Int256>();
    result |= testing::universalTestLshift<avx::Int256>();
    result |= testing::universalTestRshift<avx::Int256>();
    result |= testing::universalTestIndexing<avx::Int256>();

    result |= data_load_save();
    result |= data_load_save_aligned();

    return result;
}
