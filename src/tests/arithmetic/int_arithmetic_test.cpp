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
    
    result |= testing::universal_test_add<avx::Int256, int>();
    result |= testing::universal_test_sub<avx::Int256, int>();
    result |= testing::universal_test_mul<avx::Int256, int>();
    result |= testing::universal_test_div<avx::Int256, int>();
    result |= testing::universal_test_mod<avx::Int256, int>();
    result |= testing::universal_test_and<avx::Int256, int>();
    result |= testing::universal_test_or<avx::Int256, int>();
    result |= testing::universal_test_xor<avx::Int256, int>();
    result |= testing::universal_test_not<avx::Int256, int>();
    result |= testing::universal_test_lshift<avx::Int256, int>();
    result |= testing::universal_test_rshift<avx::Int256, int>();
    result |= testing::universal_test_indexing<avx::Int256, int>();

    result |= data_load_save();
    result |= data_load_save_aligned();

    return result;
}
