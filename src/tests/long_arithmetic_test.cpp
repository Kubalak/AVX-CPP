#include <iostream>
#include <array>
#include "types/long256.hpp"
#include "test_utils.hpp"

int long256_test_add() {
    int result = 0;
       
    avx::Long256 zero;
    avx::Long256 one(1LL);
    avx::Long256 two({2LL, 2LL, 2LL, 2LL});
    avx::Long256 fromArray(std::array<long long, 4>{3LL, 3LL, 3LL, 3LL});

    std::cout <<"Starting test " << __func__ <<std::endl;

    if(zero != 0){
        std::cerr << "Default construction error!" << zero.str() << std::endl;
        result = 1;
    }
    
    avx::Long256 sum1 = one + two;
    
    if (sum1 != avx::Long256({3LL, 3LL, 3LL, 3LL})) {
        std::cerr << "Test failed: Long256 + Long256" << std::endl;
        result = 1;
    }

    avx::Long256 sum2 = one + 1LL;
    if (sum2 != avx::Long256({2LL, 2LL, 2LL, 2LL})) {
        std::cerr << "Test failed: Long256 + long long" << std::endl;
        result = 1;
    }

    avx::Long256 addResult = one;
    addResult += two;
    if (addResult != avx::Long256({3LL, 3LL, 3LL, 3LL})) {
        std::cerr << "Test failed: Long256 += Long256" << std::endl;
        result = 1;
    }

    avx::Long256 addLongResult = one;
    addLongResult += 1LL;
    if (addLongResult != avx::Long256({2LL, 2LL, 2LL, 2LL})) {
        std::cerr << "Test failed: Long256 += long long" << std::endl;
        result = 1;
    }

    return result;
}

int long256_test_multiply() {
    int result = 0;

    avx::Long256 one({1LL, 2LL, 3LL, 4LL});
    avx::Long256 two({2LL, 2LL, 2LL, 2LL});
    std::cout <<"Starting test " << __func__ <<std::endl;

    avx::Long256 product1 = one * two;
    if (product1 != avx::Long256({2LL, 4LL, 6LL, 8LL})) {
        std::cerr << "Test failed: Long256 * Long256" << one.str() << ' ' << two.str()  << ' ' << product1.str() << std::endl;
        result = 1;
    }

    avx::Long256 product2 = one * 2LL;
    if (product2 != avx::Long256({2LL, 4LL, 6LL, 8LL})) {
        std::cerr << "Test failed: Long256 * long long" << std::endl;
        result = 1;
    }

    product1 = one;
    product1 *= two;
    if (product1 != avx::Long256({2LL, 4LL, 6LL, 8LL})) {
        std::cerr << "Test failed: Long256 *= Long256" << one.str() << ' ' << two.str()  << ' ' << product1.str() << std::endl;
        result = 1;
    }

    product2 = one;
    product2 *= 2;
    if (product2 != avx::Long256({2LL, 4LL, 6LL, 8LL})) {
        std::cerr << "Test failed: Long256 *= long long" << std::endl;
        result = 1;
    }

    return result;
}


int long256_test_subtract() {
    int result = 0;

    avx::Long256 four(4LL);
    avx::Long256 two(2LL);
    avx::Long256 three({3LL, 3LL, 3LL, 3LL});
    std::cout <<"Starting test " << __func__ <<std::endl;
    
    avx::Long256 diff1 = four - two;
    if (diff1 != avx::Long256({2LL, 2LL, 2LL, 2LL})) {
        std::cerr << "Test failed: Long256 - Long256" << std::endl;
        result = 1;
    }

    avx::Long256 diff2 = four - 1LL;
    if (diff2 != avx::Long256({3LL, 3LL, 3LL, 3LL})) {
        std::cerr << "Test failed: Long256 - long long" << std::endl;
        result = 1;
    }

    diff1 = four;
    diff1 -= two;
    if (diff1 != avx::Long256({2LL, 2LL, 2LL, 2LL})) {
        std::cerr << "Test failed: Long256 -= Long256" << std::endl;
        result = 1;
    }

    diff2 = four;
    diff2 -= 1;
    if (diff2 != avx::Long256({3LL, 3LL, 3LL, 3LL})) {
        std::cerr << "Test failed: Long256 -= long long" << std::endl;
        result = 1;
    }

    return result;
}

int long256_test_divide() {
    int result = 0;
    
    avx::Long256 eight(8LL);
    avx::Long256 two(2LL);
    std::cout <<"Starting test " << __func__ <<std::endl;

    avx::Long256 quotient1 = eight / two;
    if (quotient1 != avx::Long256({4LL, 4LL, 4LL, 4LL})) {
        std::cerr << "Test failed: Long256 / Long256" << std::endl;
        result = 1;
    }

    avx::Long256 quotient2 = eight / 4LL;
    if (quotient2 != avx::Long256({2LL, 2LL, 2LL, 2LL})) {
        std::cerr << "Test failed: Long256 / long long" << std::endl;
        result = 1;
    }

    quotient1 = eight;
    quotient1 /= two;
    if (quotient1 != avx::Long256({4LL, 4LL, 4LL, 4LL})) {
        std::cerr << "Test failed: Long256 /= Long256" << std::endl;
        result = 1;
    }

    quotient2 = eight;
    quotient2 /= 4;
    if (quotient2 != avx::Long256({2LL, 2LL, 2LL, 2LL})) {
        std::cerr << "Test failed: Long256 /= long long" << std::endl;
        result = 1;
    }

    std::cout << quotient2.str() << std::endl;
    quotient2 /= 3;
    if(quotient2 != 0){
        std::cerr << "Division failed: " << quotient2.str() << std::endl;
        result = 1;
    }

    return result;
}

int long256_test_shift() {
    int result = 0;

    avx::Long256 one(1LL);
    avx::Long256 two(2LL);
    avx::Long256 four(4LL);
    std::cout <<"Starting test " << __func__ <<std::endl;

    avx::Long256 shiftLeft1 = one << two;
    if (shiftLeft1 != avx::Long256({4LL, 4LL, 4LL, 4LL})) {
        std::cerr << "Test failed: Long256 << Long256" << std::endl;
        result = 1;
    }

    avx::Long256 shiftLeft2 = one << 2LL;
    if (shiftLeft2 != avx::Long256({4LL, 4LL, 4LL, 4LL})) {
        std::cerr << "Test failed: Long256 << long long" << std::endl;
        result = 1;
    }

    shiftLeft1 = one;
    shiftLeft1 <<= two;
    if (shiftLeft1 != avx::Long256({4LL, 4LL, 4LL, 4LL})) {
        std::cerr << "Test failed: Long256 <<= Long256" << std::endl;
        result = 1;
    }

    shiftLeft2 = one;
    shiftLeft2 <<= 2LL;
    if (shiftLeft2 != avx::Long256({4LL, 4LL, 4LL, 4LL})) {
        std::cerr << "Test failed: Long256 <<= long long" << std::endl;
        result = 1;
    }

    avx::Long256 shiftRight1 = four >> two;
    if (shiftRight1 != avx::Long256({1LL, 1LL, 1LL, 1LL})) {
        std::cerr << "Test failed: Long256 >> Long256" << std::endl;
        result = 1;
    }

    avx::Long256 shiftRight2 = four >> 2LL;
    if (shiftRight2 != avx::Long256({1LL, 1LL, 1LL, 1LL})) {
        std::cerr << "Test failed: Long256 >> long long" << std::endl;
        result = 1;
    }

    shiftRight1 = four;
    shiftRight1 >>= two;
    if (shiftRight1 != avx::Long256({1LL, 1LL, 1LL, 1LL})) {
        std::cerr << "Test failed: Long256 >>= Long256" << std::endl;
        result = 1;
    }

    shiftRight2 = four;
    shiftRight2 >>= 2LL;
    if (shiftRight2 != avx::Long256({1LL, 1LL, 1LL, 1LL})) {
        std::cerr << "Test failed: Long256 >>= long long" << std::endl;
        result = 1;
    }

    return result;
}

int long256_test_bitwise_or() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;
    
    avx::Long256 a({1, 2, 3, 4});
    avx::Long256 b({4, 3, 2, 1});
    avx::Long256 expected_or({5, 3, 3, 5});
    avx::Long256 act_or = a | b;

    if (act_or != expected_or) {
        std::cerr << "Test " << __func__ << " Long256 | Long256 failed! Expected: "  << expected_or.str() << " actual: " << act_or.str() << std::endl;
        result = 1;
    }

    avx::Long256 c = a;
    c |= b;

    if (c != expected_or) {
        std::cerr << "Test " << __func__ << " |= Long256 |= Long256 failed! Expected: " << expected_or.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int long256_test_bitwise_xor() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Long256 a({1, 2, 3, 4});
    avx::Long256 b({4, 3, 2, 1});
    avx::Long256 expected_xor({5, 1, 1, 5});
    avx::Long256 act_xor = a ^ b;

    if (act_xor != expected_xor) {
        std::cerr << "Test " << __func__ << " Long256 ^ Long256 failed! Expected: " << expected_xor.str() << " actual: " << act_xor.str() << std::endl;
        result = 1;
    }

    avx::Long256 c = a;
    c ^= b;

    if (c != expected_xor) {
        std::cerr << "Test " << __func__ << " Long256 ^= Long256 failed! Expected: " << expected_xor.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int long256_test_bitwise_and() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Long256 a({1, 2, 3, 4});
    avx::Long256 b({4, 3, 2, 1});
    avx::Long256 expected_and({0, 2, 2, 0});
    avx::Long256 act_and = a & b;

    if (act_and != expected_and) {
        std::cerr << "Test " << __func__ << " Long256 & Long256 failed! Expected: " << expected_and.str() << " actual: " << act_and.str() << std::endl;
        result = 1;
    }

    avx::Long256 c = a;
    c &= b;

    if (c != expected_and) {
        std::cerr << "Test " << __func__ << " Long256 &= Long256 failed! Expected: " << expected_and.str() << " actual: " << c.str() << std::endl;
        result = 1;
    }

    return result;
}

int long256_test_bitwise_not() {
    std::cout << "Starting test: " << __func__ << std::endl;

    int result = 0;

    avx::Long256 a({0, 1, 2, 3});
    avx::Long256 expected_not({~0LL, ~1LL, ~2LL, ~3LL});
    avx::Long256 act_not = ~a;

    if (act_not != expected_not) {
        std::cerr << "Test " << __func__ << " ~Long256 failed! Expected: " << expected_not.str() << " actual: " << act_not.str() << std::endl;
        result = 1;
    }

    return result;
}


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

    
    result |= long256_test_add();
    result |= long256_test_subtract();
    result |= long256_test_multiply();
    result |= long256_test_divide();
    result |= long256_test_shift();    
    result |= long256_test_bitwise_and();
    result |= long256_test_bitwise_or();
    result |= long256_test_bitwise_xor();
    result |= long256_test_bitwise_not();

    /*result |= testing::universal_test_add<avx::Long256, long long>();
    result |= testing::universal_test_sub<avx::Long256, long long>();
    result |= testing::universal_test_mul<avx::Long256, long long>();
    result |= testing::universal_test_div<avx::Long256, long long>();
    result |= testing::universal_test_mod<avx::Long256, long long>();
    result |= testing::universal_test_lshift<avx::Long256, long long>();
    result |= testing::universal_test_rshift<avx::Long256, long long>();
    result |= testing::universal_test_and<avx::Long256, long long>();
    result |= testing::universal_test_or<avx::Long256, long long>();
    result |= testing::universal_test_xor<avx::Long256, long long>();*/
    
    result |= data_load_save();
    result |= data_load_save_aligned();

    return result;
}