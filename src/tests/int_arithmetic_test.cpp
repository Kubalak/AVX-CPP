#include "types/int256.hpp"
#include <iostream>

int int256_test_add() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 b({8, 7, 6, 5, 4, 3, 2, 1});
    avx::Int256 expected_add({9, 9, 9, 9, 9, 9, 9, 9});
    avx::Int256 act_add = a + b;

    if (act_add != expected_add) {
        std::cerr << "Test " << __func__ << " Int256 + Int256 failed! Expected: " << expected_add.str() << " actual: " << act_add.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c += b;
    if (c != expected_add) {
        std::cerr << "Test " << __func__ << " Int256 += Int256 failed! Expected: " << expected_add.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    avx::Int256 d({1, 2, 3, 4, 5, 6, 7, 8});
    int e = 10;
    avx::Int256 expected_add_int({11, 12, 13, 14, 15, 16, 17, 18});
    avx::Int256 act_add_int = d + e;

    if (act_add_int != expected_add_int) {
        std::cerr << "Test " << __func__ << " Int256 + int failed! Expected: " << expected_add_int.str() << " actual: " << act_add_int.str() << std::endl;
        result = 1;
    }

    d += e;
    if (d != expected_add_int) {
        std::cerr << "Test " << __func__ << " Int256 += int failed! Expected: " << expected_add_int.str() << " actual: " << d.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_sub() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({9, 8, 7, 6, 5, 4, 3, 2});
    avx::Int256 b({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 expected_sub({8, 6, 4, 2, 0, -2, -4, -6});
    avx::Int256 act_sub = a - b;

    if (act_sub != expected_sub) {
        std::cerr << "Test " << __func__ << " Int256 - Int256 failed! Expected: " << expected_sub.str() << " actual: " << act_sub.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c -= b;
    if (c != expected_sub) {
        std::cerr << "Test " << __func__ << " Int256 -= Int256 failed! Expected: " << expected_sub.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    avx::Int256 d({9, 8, 7, 6, 5, 4, 3, 2});
    int e = 1;
    avx::Int256 expected_sub_int({8, 7, 6, 5, 4, 3, 2, 1});
    avx::Int256 act_sub_int = d - e;

    if (act_sub_int != expected_sub_int) {
        std::cerr << "Test " << __func__ << " Int256 - int failed! Expected: " << expected_sub_int.str() << " actual: " << act_sub_int.str() << std::endl;
        result = 1;
    }

    d -= e;
    if (d != expected_sub_int) {
        std::cerr << "Test " << __func__ << " Int256 -= int failed! Expected: " << expected_sub_int.str() << " actual: " << d.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_mul() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 b({8, 7, 6, 5, 4, 3, 2, 1});
    avx::Int256 expected_mul({8, 14, 18, 20, 20, 18, 14, 8});
    avx::Int256 act_mul = a * b;

    if (act_mul != expected_mul) {
        std::cerr << "Test " << __func__ << " Int256 * Int256 failed! Expected: " << expected_mul.str() << " actual: " << act_mul.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c *= b;
    if (c != expected_mul) {
        std::cerr << "Test " << __func__ << " Int256 *= Int256 failed! Expected: " << expected_mul.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    avx::Int256 d({1, 2, 3, 4, 5, 6, 7, 8});
    int e = 2;
    avx::Int256 expected_mul_int({2, 4, 6, 8, 10, 12, 14, 16});
    avx::Int256 act_mul_int = d * e;

    if (act_mul_int != expected_mul_int) {
        std::cerr << "Test " << __func__ << " Int256 * int failed! Expected: " << expected_mul_int.str() << " actual: " << act_mul_int.str() << std::endl;
        result = 1;
    }

    d *= e;
    if (d != expected_mul_int) {
        std::cerr << "Test " << __func__ << " Int256 *= int failed! Expected: " << expected_mul_int.str() << " actual: " << d.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_div() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({8, 16, 24, 32, 40, 48, 56, 64});
    avx::Int256 b({8, 8, 8, 8, 8, 8, 8, 8});
    avx::Int256 expected_div({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 act_div = a / b;

    if (act_div != expected_div) {
        std::cerr << "Test " << __func__ << " Int256 / Int256 failed! Expected: " << expected_div.str() << " actual: " << act_div.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c /= b;
    if (c != expected_div) {
        std::cerr << "Test " << __func__ << " Int256 /= Int256 failed! Expected: " << expected_div.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    avx::Int256 d({8, 16, 24, 32, 40, 48, 56, 64});
    int e = 8;
    avx::Int256 expected_div_int({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 act_div_int = d / e;

    if (act_div_int != expected_div_int) {
        std::cerr << "Test " << __func__ << " Int256 / int failed! Expected: " << expected_div_int.str() << " actual: " << act_div_int.str() << std::endl;
        result = 1;
    }

    d /= e;
    if (d != expected_div_int) {
        std::cerr << "Test " << __func__ << " Int256 /= int failed! Expected: " << expected_div_int.str() << " actual: " << d.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_mod() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({10, 20, 30, 40, 50, 60, 70, 80});
    avx::Int256 b({3, 3, 3, 3, 3, 3, 3, 3});
    avx::Int256 expected_mod({1, 2, 0, 1, 2, 0, 1, 2});
    avx::Int256 act_mod = a % b;

    if (act_mod != expected_mod) {
        std::cerr << "Test " << __func__ << " Int256 % Int256 failed! Expected: " << expected_mod.str() << " actual: " << act_mod.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c %= b;
    if (c != expected_mod) {
        std::cerr << "Test " << __func__ << " Int256 %= Int256 failed! Expected: " << expected_mod.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    avx::Int256 d({10, 20, 30, 40, 50, 60, 70, 80});
    int e = 3;
    avx::Int256 expected_mod_int({1, 2, 0, 1, 2, 0, 1, 2});
    avx::Int256 act_mod_int = d % e;

    if (act_mod_int != expected_mod_int) {
        std::cerr << "Test " << __func__ << " Int256 % int failed! Expected: " << expected_mod_int.str() << " actual: " << act_mod_int.str() << std::endl;
        result = 1;
    }

    d %= e;
    if (d != expected_mod_int) {
        std::cerr << "Test " << __func__ << " Int256 %= int failed! Expected: " << expected_mod_int.str() << " actual: " << d.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_and() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 b({8, 7, 6, 5, 4, 3, 2, 1});
    avx::Int256 expected_and({0, 2, 2, 4, 4, 2, 2, 0});
    avx::Int256 act_and = a & b;

    if (act_and != expected_and) {
        std::cerr << "Test " << __func__ << " Int256 & Int256 failed! Expected: " << expected_and.str() << " actual: " << act_and.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c &= b;
    if (c != expected_and) {
        std::cerr << "Test " << __func__ << " Int256 &= Int256 failed! Expected: " << expected_and.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    avx::Int256 d({1, 2, 3, 4, 5, 6, 7, 8});
    int e = 3;
    avx::Int256 expected_and_int({1, 2, 3, 0, 1, 2, 3, 0});
    avx::Int256 act_and_int = d & e;

    if (act_and_int != expected_and_int) {
        std::cerr << "Test " << __func__ << " Int256 & int failed! Expected: " << expected_and_int.str() << " actual: " << act_and_int.str() << std::endl;
        result = 1;
    }

    d &= e;
    if (d != expected_and_int) {
        std::cerr << "Test " << __func__ << " Int256 &= int failed! Expected: " << expected_and_int.str() << " actual: " << d.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_or() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 b({8, 7, 6, 5, 4, 3, 2, 1});
    avx::Int256 expected_or({9, 7, 7, 5, 5, 7, 7, 9});
    avx::Int256 act_or = a | b;

    if (act_or != expected_or) {
        std::cerr << "Test " << __func__ << " Int256 | Int256 failed! Expected: " << expected_or.str() << " actual: " << act_or.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c |= b;
    if (c != expected_or) {
        std::cerr << "Test " << __func__ << " Int256 |= Int256 failed! Expected: " << expected_or.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    avx::Int256 d({1, 2, 3, 4, 5, 6, 7, 8});
    int e = 3;
    avx::Int256 expected_or_int({3, 3, 3, 7, 7, 7, 7, 11});
    avx::Int256 act_or_int = d | e;

    if (act_or_int != expected_or_int) {
        std::cerr << "Test " << __func__ << " Int256 | int failed! Expected: " << expected_or_int.str() << " actual: " << act_or_int.str() << std::endl;
        result = 1;
    }

    d |= e;
    if (d != expected_or_int) {
        std::cerr << "Test " << __func__ << " Int256 |= int failed! Expected: " << expected_or_int.str() << " actual: " << d.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_xor() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 b({8, 7, 6, 5, 4, 3, 2, 1});
    avx::Int256 expected_xor({9, 5, 5, 1, 1, 5, 5, 9});
    avx::Int256 act_xor = a ^ b;

    if (act_xor != expected_xor) {
        std::cerr << "Test " << __func__ << " Int256 ^ Int256 failed! Expected: " << expected_xor.str() << " actual: " << act_xor.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c ^= b;
    if (c != expected_xor) {
        std::cerr << "Test " << __func__ << " Int256 ^= Int256 failed! Expected: " << expected_xor.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    avx::Int256 d({1, 2, 3, 4, 5, 6, 7, 8});
    int e = 3;
    avx::Int256 expected_xor_int({2, 1, 0, 7, 6, 5, 4, 11});
    avx::Int256 act_xor_int = d ^ e;

    if (act_xor_int != expected_xor_int) {
        std::cerr << "Test " << __func__ << " Int256 ^ int failed! Expected: " << expected_xor_int.str() << " actual: " << act_xor_int.str() << std::endl;
        result = 1;
    }

    d ^= e;
    if (d != expected_xor_int) {
        std::cerr << "Test " << __func__ << " Int256 ^= int failed! Expected: " << expected_xor_int.str() << " actual: " << d.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_not() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 expected_not({~1, ~2, ~3, ~4, ~5, ~6, ~7, ~8});
    avx::Int256 act_not = ~a;

    if (act_not != expected_not) {
        std::cerr << "Test " << __func__ << " ~Int256 failed! Expected: " << expected_not.str() << " actual: " << act_not.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_lshift() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({1, 2, 3, 4, 5, 6, 7, 8});
    int b = 2;
    avx::Int256 expected_lshift({4, 8, 12, 16, 20, 24, 28, 32});
    avx::Int256 act_lshift = a << b;

    if (act_lshift != expected_lshift) {
        std::cerr << "Test " << __func__ << " Int256 << Int256 failed! Expected: " << expected_lshift.str() << " actual: " << act_lshift.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c <<= b;
    if (c != expected_lshift) {
        std::cerr << "Test " << __func__ << " Int256 <<= int failed! Expected: " << expected_lshift.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int int256_test_rshift() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Int256 a({4, 8, 12, 16, 20, 24, 28, 32});
    int b = 2;
    avx::Int256 expected_rshift({1, 2, 3, 4, 5, 6, 7, 8});
    avx::Int256 act_rshift = a >> b;

    if (act_rshift != expected_rshift) {
        std::cerr << "Test " << __func__ << " Int256 >> Int256 failed! Expected: " << expected_rshift.str() << " actual: " << act_rshift.str() << std::endl;
        result = 1;
    }

    avx::Int256 c = a;
    c >>= b;
    if (c != expected_rshift) {
        std::cerr << "Test " << __func__ << " Int256 >>= int failed! Expected: " << expected_rshift.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

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
    std::cout << _MSC_VER << '\n';
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

    result |= int256_test_add();
    result |= int256_test_sub();
    result |= int256_test_mul();
    result |= int256_test_div();
    result |= int256_test_mod();
    result |= int256_test_and();
    result |= int256_test_or();
    result |= int256_test_xor();
    result |= int256_test_not();
    result |= int256_test_lshift();
    result |= int256_test_rshift();

    result |= data_load_save();
    result |= data_load_save_aligned();

    return result;
}
