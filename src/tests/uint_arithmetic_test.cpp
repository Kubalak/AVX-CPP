#include <iostream>
#include <array>
#include <string>
#include <climits>
#include <types/uint256.hpp> 
#include "test_utils.hpp"

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

#ifdef _MSC_VER
    __declspec(align(32)) unsigned int dest[8] ;
#else
    unsigned int dest[8] __attribute__((aligned(32)));
#endif

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
    #ifdef _WIN32
        #ifdef _WIN64
            constexpr char os[] = "Win64";
        #else
            constexpr char os[] = "Win32";
        #endif
    #elif __linux__
        constexpr char os[] = "Linux";
    #elif __unix__
        constexpr char os[] = "Unix";
    #elif __APPLE__
        constexpr char os[] = "Apple";
    #endif

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

    #ifdef _MSC_VER
        printf("Compiler MSC v.%d, build date: %s %s on %s\n", _MSC_VER, __DATE__, __TIME__, os);
    #elif __GNUC__
        printf("Compiler GCC %d.%d.%d, build date: %s %s on %s\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__, __DATE__, __TIME__, os);
    #endif
    result |= testing::universal_test_add<avx::UInt256, unsigned int>();
    result |= testing::universal_test_sub<avx::UInt256, unsigned int>();
    result |= testing::universal_test_mul<avx::UInt256, unsigned int>();
    result |= testing::universal_test_div<avx::UInt256, unsigned int>();
    result |= testing::universal_test_mod<avx::UInt256, unsigned int>();
    result |= testing::universal_test_and<avx::UInt256, unsigned int>();
    result |= testing::universal_test_or<avx::UInt256, unsigned int>();
    result |= testing::universal_test_xor<avx::UInt256, unsigned int>();
    result |= testing::universal_test_not<avx::UInt256, unsigned int>();
    result |= testing::universal_test_lshift<avx::UInt256, unsigned int>();
    result |= testing::universal_test_rshift<avx::UInt256, unsigned int>();

    result |= data_load_save();
    result |= data_load_save_aligned();


    return result;
}
