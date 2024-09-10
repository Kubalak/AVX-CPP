#include <iostream>
#include <array>
#include <string>
#include <climits>
#include <types/uint256.hpp> 
#include "test_utils.hpp"

int uint256_test_add() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 b({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 expected_add({2, 4, 6, 8, 10, 12, 14, 16});
    avx::UInt256 act_add = a + b;

    if (act_add != expected_add) {
        std::cerr << "Test " << __func__ << " UInt256 + UInt256 failed! Expected: " << expected_add.str() << " actual: " << act_add.str() << std::endl;
        result = 1;
    }

    a += b;
    if (a != expected_add) {
        std::cerr << "Test " << __func__ << " UInt256 += UInt256 failed! Expected: " << expected_add.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    avx::UInt256 c({1, 2, 3, 4, 5, 6, 7, 8});
    int d = 2;
    avx::UInt256 expected_add_int({3, 4, 5, 6, 7, 8, 9, 10});
    avx::UInt256 act_add_int = c + d;

    if (act_add_int != expected_add_int) {
        std::cerr << "Test " << __func__ << " UInt256 + int failed! Expected: " << expected_add_int.str() << " actual: " << act_add_int.str() << std::endl;
        result = 1;
    }

    c += d;
    if (c != expected_add_int) {
        std::cerr << "Test " << __func__ << " UInt256 += int failed! Expected: " << expected_add_int.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_sub() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({2, 4, 6, 8, 10, 12, 14, 16});
    avx::UInt256 b({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 expected_sub({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 act_sub = a - b;

    if (act_sub != expected_sub) {
        std::cerr << "Test " << __func__ << " UInt256 - UInt256 failed! Expected: " << expected_sub.str() << " actual: " << act_sub.str() << std::endl;
        result = 1;
    }

    a -= b;
    if (a != expected_sub) {
        std::cerr << "Test " << __func__ << " UInt256 -= UInt256 failed! Expected: " << expected_sub.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    avx::UInt256 c({3, 4, 5, 6, 7, 8, 9, 10});
    int d = 2;
    avx::UInt256 expected_sub_int({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 act_sub_int = c - d;

    if (act_sub_int != expected_sub_int) {
        std::cerr << "Test " << __func__ << " UInt256 - int failed! Expected: " << expected_sub_int.str() << " actual: " << act_sub_int.str() << std::endl;
        result = 1;
    }

    c -= d;
    if (c != expected_sub_int) {
        std::cerr << "Test " << __func__ << " UInt256 -= int failed! Expected: " << expected_sub_int.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_mul() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 b({2, 2, 2, 2, 2, 2, 2, 2});
    avx::UInt256 expected_mul({2, 4, 6, 8, 10, 12, 14, 16});
    avx::UInt256 act_mul = a * b;

    if (act_mul != expected_mul) {
        std::cerr << "Test " << __func__ << " UInt256 * UInt256 failed! Expected: " << expected_mul.str() << " actual: " << act_mul.str() << std::endl;
        result = 1;
    }

    a *= b;
    if (a != expected_mul) {
        std::cerr << "Test " << __func__ << " UInt256 *= UInt256 failed! Expected: " << expected_mul.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    avx::UInt256 c({1, 2, 3, 4, 5, 6, 7, 8});
    int d = 2;
    avx::UInt256 expected_mul_int({2, 4, 6, 8, 10, 12, 14, 16});
    avx::UInt256 act_mul_int = c * d;

    if (act_mul_int != expected_mul_int) {
        std::cerr << "Test " << __func__ << " UInt256 * int failed! Expected: " << expected_mul_int.str() << " actual: " << act_mul_int.str() << std::endl;
        result = 1;
    }

    c *= d;
    if (c != expected_mul_int) {
        std::cerr << "Test " << __func__ << " UInt256 *= int failed! Expected: " << expected_mul_int.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_div() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({2, 4, 6, 8, 10, 12, 14, 16});
    avx::UInt256 b({2, 2, 2, 2, 2, 2, 2, 2});
    avx::UInt256 expected_div({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 act_div = a / b;

    if (act_div != expected_div) {
        std::cerr << "Test " << __func__ << " UInt256 / UInt256 failed! Expected: " << expected_div.str() << " actual: " << act_div.str() << std::endl;
        result = 1;
    }

    a /= b;
    if (a != expected_div) {
        std::cerr << "Test " << __func__ << " UInt256 /= UInt256 failed! Expected: " << expected_div.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    avx::UInt256 c({2, 4, 6, 8, 10, 12, 14, 16});
    int d = 2;
    avx::UInt256 expected_div_int({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 act_div_int = c / d;

    if (act_div_int != expected_div_int) {
        std::cerr << "Test " << __func__ << " UInt256 / int failed! Expected: " << expected_div_int.str() << " actual: " << act_div_int.str() << std::endl;
        result = 1;
    }

    c /= d;
    if (c != expected_div_int) {
        std::cerr << "Test " << __func__ << " UInt256 /= int failed! Expected: " << expected_div_int.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_mod() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({5, 10, 15, 20, 25, 30, 35, 40});
    avx::UInt256 b({3, 3, 3, 3, 3, 3, 3, 3});
    avx::UInt256 expected_mod({2, 1, 0, 2, 1, 0, 2, 1});
    avx::UInt256 act_mod = a % b;

    if (act_mod != expected_mod) {
        std::cerr << "Test " << __func__ << " UInt256 % UInt256 failed! Expected: " << expected_mod.str() << " actual: " << act_mod.str() << std::endl;
        result = 1;
    }

    a %= b;
    if (a != expected_mod) {
        std::cerr << "Test " << __func__ << " UInt256 %= UInt256 failed! Expected: " << expected_mod.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    avx::UInt256 c({5, 10, 15, 20, 25, 30, 35, 40});
    int d = 3;
    avx::UInt256 expected_mod_int({2, 1, 0, 2, 1, 0, 2, 1});
    avx::UInt256 act_mod_int = c % d;

    if (act_mod_int != expected_mod_int) {
        std::cerr << "Test " << __func__ << " UInt256 % int failed! Expected: " << expected_mod_int.str() << " actual: " << act_mod_int.str() << std::endl;
        result = 1;
    }

    c %= d;
    if (c != expected_mod_int) {
        std::cerr << "Test " << __func__ << " UInt256 %= int failed! Expected: " << expected_mod_int.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_and() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;
    avx::UInt256 a({5, 6, 7, 8, 9, 10, 11, 12});
    avx::UInt256 b({3, 3, 3, 3, 3, 3, 3, 3});
    avx::UInt256 expected_and({1, 2, 3, 0, 1, 2, 3, 0});
    avx::UInt256 act_and = a & b;

    if (act_and != expected_and) {
        std::cerr << "Test " << __func__ << " UInt256 & UInt256 failed! Expected: " << expected_and.str() << " actual: " << act_and.str() << std::endl;
        result = 1;
    }

    a &= b;
    if (a != expected_and) {
        std::cerr << "Test " << __func__ << " UInt256 &= UInt256 failed! Expected: " << expected_and.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    avx::UInt256 c({5, 6, 7, 8, 9, 10, 11, 12});
    int d = 3;
    avx::UInt256 expected_and_int({1, 2, 3, 0, 1, 2, 3, 0});
    avx::UInt256 act_and_int = c & d;

    if (act_and_int != expected_and_int) {
        std::cerr << "Test " << __func__ << " UInt256 & int failed! Expected: " << expected_and_int.str() << " actual: " << act_and_int.str() << std::endl;
        result = 1;
    }

    c &= d;
    if (c != expected_and_int) {
        std::cerr << "Test " << __func__ << " UInt256 &= int failed! Expected: " << expected_and_int.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_or() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 b({2, 2, 2, 2, 2, 2, 2, 2});
    avx::UInt256 expected_or({3, 2, 3, 6, 7, 6, 7, 10});
    avx::UInt256 act_or = a | b;

    if (act_or != expected_or) {
        std::cerr << "Test " << __func__ << " UInt256 | UInt256 failed! Expected: " << expected_or.str() << " actual: " << act_or.str() << std::endl;
        result = 1;
    }

    a |= b;
    if (a != expected_or) {
        std::cerr << "Test " << __func__ << " UInt256 |= UInt256 failed! Expected: " << expected_or.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    avx::UInt256 c({1, 2, 3, 4, 5, 6, 7, 8});
    int d = 2;
    avx::UInt256 expected_or_int({3, 2, 3, 6, 7, 6, 7, 10});
    avx::UInt256 act_or_int = c | d;

    if (act_or_int != expected_or_int) {
        std::cerr << "Test " << __func__ << " UInt256 | int failed! Expected: " << expected_or_int.str() << " actual: " << act_or_int.str() << std::endl;
        result = 1;
    }

    c |= d;
    if (c != expected_or_int) {
        std::cerr << "Test " << __func__ << " UInt256 |= int failed! Expected: " << expected_or_int.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_xor() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 b({2, 2, 2, 2, 2, 2, 2, 2});
    avx::UInt256 expected_xor({3, 0, 1, 6, 7, 4, 5, 10});
    avx::UInt256 act_xor = a ^ b;

    if (act_xor != expected_xor) {
        std::cerr << "Test " << __func__ << " UInt256 ^ UInt256 failed! Expected: " << expected_xor.str() << " actual: " << act_xor.str() << std::endl;
        result = 1;
    }

    a ^= b;
    if (a != expected_xor) {
        std::cerr << "Test " << __func__ << " UInt256 ^= UInt256 failed! Expected: " << expected_xor.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    avx::UInt256 c({1, 2, 3, 4, 5, 6, 7, 8});
    int d = 2;
    avx::UInt256 expected_xor_int({3, 0, 1, 6, 7, 4, 5, 10});
    avx::UInt256 act_xor_int = c ^ d;

    if (act_xor_int != expected_xor_int) {
        std::cerr << "Test " << __func__ << " UInt256 ^ int failed! Expected: " << expected_xor_int.str() << " actual: " << act_xor_int.str() << std::endl;
        result = 1;
    }

    c ^= d;
    if (c != expected_xor_int) {
        std::cerr << "Test " << __func__ << " UInt256 ^= int failed! Expected: " << expected_xor_int.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_not() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({1, 2, 3, 4, 7, INT_MAX, 0, UINT_MAX});
    avx::UInt256 expected_not({0xFFFFFFFEu, 0xFFFFFFFDu, 0xFFFFFFFCu, 0xFFFFFFFBu, 0xFFFFFFF8u, 0x80000000u, 0xFFFFFFFFu, 0u});
    avx::UInt256 act_not = ~a;

    if (act_not != expected_not) {
        std::cerr << "Test " << __func__ << " UInt256 ~ failed! Expected: " << expected_not.str() << " actual: " << act_not.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_lshift() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 b(2);
    avx::UInt256 expected_lshift({4, 8, 12, 16, 20, 24, 28, 32});
    avx::UInt256 act_lshift = a << b;

    if (act_lshift != expected_lshift) {
        std::cerr << "Test " << __func__ << " UInt256 << UInt256 failed! Expected: " << expected_lshift.str() << " actual: " << act_lshift.str() << std::endl;
        result = 1;
    }

    a <<= b;
    if (a != expected_lshift) {
        std::cerr << "Test " << __func__ << " UInt256 <<= UInt256 failed! Expected: " << expected_lshift.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }
    a = avx::UInt256({1, 2, 3, 4, 5, 6, 7, 8});
    act_lshift = a << 2;

    if (act_lshift != expected_lshift) {
        std::cerr << "Test " << __func__ << " UInt256 << UInt256 failed! Expected: " << expected_lshift.str() << " actual: " << act_lshift.str() << std::endl;
        result = 1;
    }

    a <<= 2;
    if (a != expected_lshift) {
        std::cerr << "Test " << __func__ << " UInt256 <<= UInt256 failed! Expected: " << expected_lshift.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    return result;
}

int uint256_test_rshift() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::UInt256 a({4, 8, 12, 16, 20, 24, 28, 32});
    avx::UInt256 b(2);
    avx::UInt256 expected_rshift({1, 2, 3, 4, 5, 6, 7, 8});
    avx::UInt256 act_rshift = a >> b;

    if (act_rshift != expected_rshift) {
        std::cerr << "Test " << __func__ << " UInt256 >> UInt256 failed! Expected: " << expected_rshift.str() << " actual: " << act_rshift.str() << std::endl;
        result = 1;
    }

    a >>= b;
    if (a != expected_rshift) {
        std::cerr << "Test " << __func__ << " UInt256 >>= UInt256 failed! Expected: " << expected_rshift.str() << " actual: " << a.str() << std::endl;
        result = 1;
    }

    return result;
}

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
    std::cout << _MSC_VER << '\n';
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
    result |= uint256_test_add();
    result |= uint256_test_sub();
    result |= uint256_test_mul();
    result |= uint256_test_div();
    result |= uint256_test_mod();
    result |= uint256_test_and();
    result |= uint256_test_or();
    result |= uint256_test_xor();
    result |= uint256_test_not();
    result |= uint256_test_lshift();
    result |= uint256_test_rshift();

    result |= data_load_save();
    result |= data_load_save_aligned();


    return result;
}
