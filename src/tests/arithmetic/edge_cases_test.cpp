#include <test_utils.hpp>
#include <types/int256.hpp>
#include <types/uint256.hpp>
#include <types/short256.hpp>
#include <types/ushort256.hpp>
#include <types/long256.hpp>
#include <types/ulong256.hpp>

using namespace avx;

// Int256
int test_edge_Int256_add() {
    int result = 0;
    Int256 a(0x7FFFFFFF), b(1);
    Int256 c = a + b;
    if (!(c == static_cast<int>(0x80000000))) {
        testFailed("+", Int256, Int256, std::to_string(static_cast<int>(0x80000000)), c.str());
        result = 1;
    }
    a += b;
    if (!(a == static_cast<int>(0x80000000))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "Int256", "Int256", std::to_string(static_cast<int>(0x80000000)), a.str());
        result = 1;
    }
    Int256 d = c + 1;
    if (!(d == static_cast<int>(0x80000001))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "Int256", "int", std::to_string(static_cast<int>(0x80000001)), d.str());
        result = 1;
    }
    c += 1;
    if (!(c == static_cast<int>(0x80000001))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "Int256", "int", std::to_string(static_cast<int>(0x80000001)), c.str());
        result = 1;
    }
    return result;
}
int test_edge_Int256_sub() {
    int result = 0;
    Int256 a(static_cast<int>(0x80000000)), b(1);
    Int256 c = a - b;
    if (!(c == static_cast<int>(0x7FFFFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "Int256", "Int256", std::to_string(static_cast<int>(0x7FFFFFFF)), c.str());
        result = 1;
    }
    a -= b;
    if (!(a == static_cast<int>(0x7FFFFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "Int256", "Int256", std::to_string(static_cast<int>(0x7FFFFFFF)), a.str());
        result = 1;
    }
    Int256 d = c - 1;
    if (!(d == static_cast<int>(0x7FFFFFFE))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "Int256", "int", std::to_string(static_cast<int>(0x7FFFFFFE)), d.str());
        result = 1;
    }
    c -= 1;
    if (!(c == static_cast<int>(0x7FFFFFFE))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "Int256", "int", std::to_string(static_cast<int>(0x7FFFFFFE)), c.str());
        result = 1;
    }
    return result;
}

int test_edge_Int256_mul() {
    int result = 0;
    Int256 a(static_cast<int>(0x40000000)), b(2);
    Int256 c = a * b;
    if (!(c == static_cast<int>(0x80000000))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "Int256", "Int256", std::to_string(static_cast<int>(0x80000000)), c.str());
        result = 1;
    }
    a *= b;
    if (!(a == static_cast<int>(0x80000000))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "Int256", "Int256", std::to_string(static_cast<int>(0x80000000)), a.str());
        result = 1;
    }
    Int256 d = c * 2;
    if (!(d == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "Int256", "int", "0", d.str());
        result = 1;
    }
    c *= 2;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "Int256", "int", "0", c.str());
        result = 1;
    }
    return result;
}
int test_edge_Int256_div() {
    int result = 0;
    Int256 a(static_cast<int>(0x7FFFFFFF)), b(2);
    Int256 c = a / b;
    if (!(c == static_cast<int>(0x3FFFFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "Int256", "Int256", std::to_string(static_cast<int>(0x3FFFFFFF)), c.str());
        result = 1;
    }
    a /= b;
    if (!(a == static_cast<int>(0x3FFFFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "Int256", "Int256", std::to_string(static_cast<int>(0x3FFFFFFF)), a.str());
        result = 1;
    }
    Int256 d = c / static_cast<int>(0x3FFFFFFF);
    if (!(d == 1)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "Int256", "int", "1", d.str());
        result = 1;
    }
    c /= static_cast<int>(0x3FFFFFFF);
    if (!(c == 1)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "Int256", "int", "1", c.str());
        result = 1;
    }
    return result;
}

int test_edge_Int256_bitwise() {
    int result = 0;
    Int256 a(0xF0F0F0F0), b(0x0F0F0F0F);
    if (!((a & b) == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&", "Int256", "Int256", "0", (a & b).str());
        result = 1;
    }
    if (!((a | b) == static_cast<int>(0xFFFFFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|", "Int256", "Int256", std::to_string(static_cast<int>(0xFFFFFFFF)), (a | b).str());
        result = 1;
    }
    if (!((a ^ b) == static_cast<int>(0xFFFFFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^", "Int256", "Int256", std::to_string(static_cast<int>(0xFFFFFFFF)), (a ^ b).str());
        result = 1;
    }
    a &= b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&=", "Int256", "Int256", "0", a.str());
        result = 1;
    }
    a = Int256(0xF0F0F0F0);
    a |= b;
    if (!(a == static_cast<int>(0xFFFFFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|=", "Int256", "Int256", std::to_string(static_cast<int>(0xFFFFFFFF)), a.str());
        result = 1;
    }
    a = Int256(0xF0F0F0F0);
    a ^= b;
    // std::cout << (a == static_cast<int>(0xFFFFFFFF)) << '\n'; <-- This fixes the error and changes assembly instructions used for == operation.
    if (!(a == static_cast<int>(0xFFFFFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^=", "Int256", "Int256", std::to_string(static_cast<int>(0xFFFFFFFF)), a.str());
        result = 1;
    }
    return result;
}

int test_edge_Int256_load_save() {
    int result = 0;
    int arr[8] = {1,2,3,4,5,6,7,8};
    Int256 v;
    v.load(arr);
    for(int i=0;i<8;++i) if(v[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "load", "Int256", "int*", std::to_string(arr[i]), std::to_string(v[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.load(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    int arr2[8] = {};
    v.save(arr2);
    for(int i=0;i<8;++i) if(arr2[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "save", "Int256", "int*", std::to_string(arr[i]), std::to_string(arr2[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.save(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    return result;
}
/*int test_edge_Int256_index() {
    int result = 0;
    Int256 v(0);
#ifndef NDEBUG
    try { int x = v[100]; (void)x; result = 1; } catch(...) {}
#else
    int x = v[100];
    if(x != v[100%8]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "[]", "Int256", "int", std::to_string(v[100%8]), std::to_string(x));
        result = 1;
    }
#endif
    return result;
}*/

// Short256
int test_edge_Short256_add() {
    int result = 0;
    Short256 a(0x7FFF), b(1);
    Short256 c = a + b;
    if (!(c == static_cast<short>(0x8000))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "Short256", "Short256", std::to_string(static_cast<short>(0x8000)), c.str());
        result = 1;
    }
    a += b;
    if (!(a == static_cast<short>(0x8000))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "Short256", "Short256", std::to_string(static_cast<short>(0x8000)), a.str());
        result = 1;
    }
    Short256 d = c + 1;
    if (!(d == static_cast<short>(0x8001))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "Short256", "short", std::to_string(static_cast<short>(0x8001)), d.str());
        result = 1;
    }
    c += 1;
    if (!(c == static_cast<short>(0x8001))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "Short256", "short", std::to_string(static_cast<short>(0x8001)), c.str());
        result = 1;
    }
    return result;
}
int test_edge_Short256_sub() {
    int result = 0;
    Short256 a(static_cast<short>(0x8000)), b(1);
    Short256 c = a - b;
    if (!(c == static_cast<short>(0x7FFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "Short256", "Short256", std::to_string(static_cast<short>(0x7FFF)), c.str());
        result = 1;
    }
    a -= b;
    if (!(a == static_cast<short>(0x7FFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "Short256", "Short256", std::to_string(static_cast<short>(0x7FFF)), a.str());
        result = 1;
    }
    Short256 d = c - 1;
    if (!(d == static_cast<short>(0x7FFE))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "Short256", "short", std::to_string(static_cast<short>(0x7FFE)), d.str());
        result = 1;
    }
    c -= 1;
    if (!(c == static_cast<short>(0x7FFE))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "Short256", "short", std::to_string(static_cast<short>(0x7FFE)), c.str());
        result = 1;
    }
    return result;
}

int test_edge_Short256_mul() {
    int result = 0;
    Short256 a(100), b(2);
    Short256 c = a * b;
    if (!(c == 200)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "Short256", "Short256", "200", c.str());
        result = 1;
    }
    a *= b;
    if (!(a == 200)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "Short256", "Short256", "200", a.str());
        result = 1;
    }
    Short256 d = c * 3;
    if (!(d == 600)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "Short256", "short", "600", d.str());
        result = 1;
    }
    c *= 3;
    if (!(c == 600)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "Short256", "short", "600", c.str());
        result = 1;
    }
    return result;
}

int test_edge_Short256_div() {
    int result = 0;
    Short256 a(100), b(2);
    Short256 c = a / b;
    if (!(c == 50)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "Short256", "Short256", "50", c.str());
        result = 1;
    }
    a /= b;
    if (!(a == 50)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "Short256", "Short256", "50", a.str());
        result = 1;
    }
    Short256 d = c / 5;
    if (!(d == 10)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "Short256", "short", "10", d.str());
        result = 1;
    }
    c /= 5;
    if (!(c == 10)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "Short256", "short", "10", c.str());
        result = 1;
    }
    return result;
}

int test_edge_Short256_bitwise() {
    int result = 0;
    Short256 a(0xF0F0), b(0x0F0F);
    if (!((a & b) == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&", "Short256", "Short256", "0", (a & b).str());
        result = 1;
    }
    if (!((a | b) == static_cast<unsigned short>(0xFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|", "Short256", "Short256", std::to_string(static_cast<unsigned short>(0xFFFF)), (a | b).str());
        result = 1;
    }
    if (!((a ^ b) == static_cast<unsigned short>(0xFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^", "Short256", "Short256", std::to_string(static_cast<unsigned short>(0xFFFF)), (a ^ b).str());
        result = 1;
    }
    a &= b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&=", "Short256", "Short256", "0", a.str());
        result = 1;
    }
    a = Short256(0xF0F0);
    a |= b;
    if (!(a == static_cast<unsigned short>(0xFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|=", "Short256", "Short256", std::to_string(static_cast<unsigned short>(0xFFFF)), a.str());
        result = 1;
    }
    a = Short256(0xF0F0);
    a ^= b;
    if (!(a == static_cast<unsigned short>(0xFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^=", "Short256", "Short256", std::to_string(static_cast<unsigned short>(0xFFFF)), a.str());
        result = 1;
    }
    return result;
}

int test_edge_Short256_load_save() {
    int result = 0;
    short arr[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Short256 v;
    v.load(arr);
    for(int i=0;i<16;++i) if(v[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "load", "Short256", "short*", std::to_string(arr[i]), std::to_string(v[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.load(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    short arr2[16] = {};
    v.save(arr2);
    for(int i=0;i<16;++i) if(arr2[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "save", "Short256", "short*", std::to_string(arr[i]), std::to_string(arr2[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.save(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    return result;
}

// UShort256
int test_edge_UShort256_add() {
    int result = 0;
    UShort256 a(static_cast<unsigned short>(0xFFFF)), b((unsigned short)1);
    UShort256 c = a + b;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "UShort256", "UShort256", "0", c.str());
        result = 1;
    }
    a += b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "UShort256", "UShort256", "0", a.str());
        result = 1;
    }
    UShort256 d = c + (unsigned short)static_cast<unsigned short>(0xFFFF);
    if (!(d == static_cast<unsigned short>(0xFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "UShort256", "unsigned short", "65535", d.str());
        result = 1;
    }
    c += (unsigned short)static_cast<unsigned short>(0xFFFF);
    if (!(c == static_cast<unsigned short>(0xFFFF))) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "UShort256", "unsigned short", "65535", c.str());
        result = 1;
    }
    return result;
}

int test_edge_UShort256_sub() {
    int result = 0;
    UShort256 a((unsigned short)0), b((unsigned short)1);
    UShort256 c = a - b;
    if (!(c == 0xFFFF)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "UShort256", "UShort256", "65535", c.str());
        result = 1;
    }
    a -= b;
    if (!(a == 0xFFFF)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "UShort256", "UShort256", "65535", a.str());
        result = 1;
    }
    UShort256 d = c - (unsigned short)0xFFFF;
    if (!(d == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "UShort256", "unsigned short", "0", d.str());
        result = 1;
    }
    c -= (unsigned short)0xFFFF;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "UShort256", "unsigned short", "0", c.str());
        result = 1;
    }
    return result;
}

int test_edge_UShort256_mul() {
    int result = 0;
    UShort256 a((unsigned short)0x8000), b((unsigned short)2);
    UShort256 c = a * b;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "UShort256", "UShort256", "0", c.str());
        result = 1;
    }
    a *= b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "UShort256", "UShort256", "0", a.str());
        result = 1;
    }
    UShort256 d = c * (unsigned short)0xFFFF;
    if (!(d == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "UShort256", "unsigned short", "0", d.str());
        result = 1;
    }
    c *= (unsigned short)0xFFFF;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "UShort256", "unsigned short", "0", c.str());
        result = 1;
    }
    return result;
}

int test_edge_UShort256_div() {
    int result = 0;
    UShort256 a((unsigned short)0xFFFF), b((unsigned short)2);
    UShort256 c = a / b;
    if (!(c == 0x7FFF)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "UShort256", "UShort256", "32767", c.str());
        result = 1;
    }
    a /= b;
    if (!(a == 0x7FFF)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "UShort256", "UShort256", "32767", a.str());
        result = 1;
    }
    UShort256 d = c / (unsigned short)0x7FFF;
    if (!(d == 1)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "UShort256", "unsigned short", "1", d.str());
        result = 1;
    }
    c /= (unsigned short)0x7FFF;
    if (!(c == 1)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "UShort256", "unsigned short", "1", c.str());
        result = 1;
    }
    return result;
}

int test_edge_UShort256_bitwise() {
    int result = 0;
    UShort256 a(0xF0F0), b(0x0F0F);
    if (!((a & b) == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&", "UShort256", "UShort256", "0", (a & b).str());
        result = 1;
    }
    if (!((a | b) == 0xFFFF)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|", "UShort256", "UShort256", std::to_string(0xFFFF), (a | b).str());
        result = 1;
    }
    if (!((a ^ b) == 0xFFFF)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^", "UShort256", "UShort256", std::to_string(0xFFFF), (a ^ b).str());
        result = 1;
    }
    a &= b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&=", "UShort256", "UShort256", "0", a.str());
        result = 1;
    }
    a = UShort256(0xF0F0);
    a |= b;
    if (!(a == 0xFFFF)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|=", "UShort256", "UShort256", std::to_string(0xFFFF), a.str());
        result = 1;
    }
    a = UShort256(0xF0F0);
    a ^= b;
    if (!(a == 0xFFFF)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^=", "UShort256", "UShort256", std::to_string(0xFFFF), a.str());
        result = 1;
    }
    return result;
}

int test_edge_UShort256_load_save() {
    int result = 0;
    unsigned short arr[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    UShort256 v;
    v.load(arr);
    for(int i=0;i<16;++i) if(v[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "load", "UShort256", "unsigned short*", std::to_string(arr[i]), std::to_string(v[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.load(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    unsigned short arr2[16] = {};
    v.save(arr2);
    for(int i=0;i<16;++i) if(arr2[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "save", "UShort256", "unsigned short*", std::to_string(arr[i]), std::to_string(arr2[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.save(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    return result;
}
/*int test_edge_UShort256_index() {
    int result = 0;
    UShort256 v((unsigned short)0);
#ifndef NDEBUG
    try { unsigned short x = v[100]; (void)x; result = 1; } catch(...) {}
#else
    unsigned short x = v[100];
    if(x != v[100%16]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "[]", "UShort256", "unsigned short", std::to_string(v[100%16]), std::to_string(x));
        result = 1;
    }
#endif
    return result;
}*/

// Long256
int test_edge_Long256_add() {
    int result = 0;
    Long256 a(0x7FFFFFFFFFFFFFFFLL), b(1LL);
    Long256 c = a + b;
    if (!(c == 0x8000000000000000LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "Long256", "Long256", "-9223372036854775808", c.str());
        result = 1;
    }
    a += b;
    if (!(a == 0x8000000000000000LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "Long256", "Long256", "-9223372036854775808", a.str());
        result = 1;
    }
    Long256 d = c + 1LL;
    if (!(d == 0x8000000000000001LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "Long256", "long long", "-9223372036854775807", d.str());
        result = 1;
    }
    c += 1LL;
    if (!(c == 0x8000000000000001LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "Long256", "long long", "-9223372036854775807", c.str());
        result = 1;
    }
    return result;
}

int test_edge_Long256_sub() {
    int result = 0;
    Long256 a(0x8000000000000000LL), b(1LL);
    Long256 c = a - b;
    if (!(c == 0x7FFFFFFFFFFFFFFFLL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "Long256", "Long256", "9223372036854775807", c.str());
        result = 1;
    }
    a -= b;
    if (!(a == 0x7FFFFFFFFFFFFFFFLL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "Long256", "Long256", "9223372036854775807", a.str());
        result = 1;
    }
    Long256 d = c - 1LL;
    if (!(d == 0x7FFFFFFFFFFFFFFELL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "Long256", "long long", "9223372036854775806", d.str());
        result = 1;
    }
    c -= 1;
    if (!(c == 0x7FFFFFFFFFFFFFFELL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "Long256", "long long", "9223372036854775806", c.str());
        result = 1;
    }
    return result;
}

int test_edge_Long256_mul() {
    int result = 0;
    Long256 a(0x4000000000000000LL), b(2LL);
    Long256 c = a * b;
    if (!(c == 0x8000000000000000LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "Long256", "Long256", "-9223372036854775808", c.str());
        result = 1;
    }
    a *= b;

    Long256 d = c * 2LL;
    if (!(d == 0LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "Long256", "long long", "0", d.str());
        #ifndef __clang__ // Ignore error on Clang v18.1.3 (it works correctly on clang 14)
            result = 1;
        #endif
    }
    c *= 2LL;
    if (!(c == 0LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "Long256", "long long", "0", c.str());
        #ifndef __clang__ // Ignore error on Clang v18.1.3 (it works correctly on clang 14)
            result = 1;
        #endif
    }
    return result;
}

int test_edge_Long256_div() {
    int result = 0;
    Long256 a(0x7FFFFFFFFFFFFFFFLL), b(2LL);
    Long256 c = a / b;
    if (!(c == 0x3FFFFFFFFFFFFFFFLL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "Long256", "Long256", "4611686018427387903", c.str());
        result = 1;
    }
    a /= b;
    if (!(a == 0x3FFFFFFFFFFFFFFFLL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "Long256", "Long256", "4611686018427387903", a.str());
        result = 1;
    }
    Long256 d = c / 0x3FFFFFFFFFFFFFFFLL;
    if (!(d == 1LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "Long256", "long long", "1", d.str());
        result = 1;
    }
    c /= 0x3FFFFFFFFFFFFFFFLL;
    if (!(c == 1LL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "Long256", "long long", "1", c.str());
        result = 1;
    }
    return result;
}

int test_edge_Long256_bitwise() {
    int result = 0;
    Long256 a(0xF0F0F0F0F0F0F0F0LL), b(0x0F0F0F0F0F0F0F0FLL);
    if (!((a & b) == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&", "Long256", "Long256", "0", (a & b).str());
        result = 1;
    }
    if (!((a | b) == 0xFFFFFFFFFFFFFFFFLL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|", "Long256", "Long256", std::to_string(0xFFFFFFFFFFFFFFFFLL), (a | b).str());
        result = 1;
    }
    if (!((a ^ b) == 0xFFFFFFFFFFFFFFFFLL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^", "Long256", "Long256", std::to_string(0xFFFFFFFFFFFFFFFFLL), (a ^ b).str());
        result = 1;
    }
    a &= b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&=", "Long256", "Long256", "0", a.str());
        result = 1;
    }
    a = Long256(0xF0F0F0F0F0F0F0F0LL);
    a |= b;
    if (!(a == 0xFFFFFFFFFFFFFFFFLL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|=", "Long256", "Long256", std::to_string(0xFFFFFFFFFFFFFFFFLL), a.str());
        result = 1;
    }
    a = Long256(0xF0F0F0F0F0F0F0F0LL);
    a ^= b;
    if (!(a == 0xFFFFFFFFFFFFFFFFLL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^=", "Long256", "Long256", std::to_string(0xFFFFFFFFFFFFFFFFLL), a.str());
        result = 1;
    }
    return result;
}
int test_edge_Long256_load_save() {
    int result = 0;
    long long arr[4] = {1,2,3,4};
    Long256 v;
    v.load(arr);
    for(int i=0;i<4;++i) if(v[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "load", "Long256", "long long*", std::to_string(arr[i]), std::to_string(v[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.load(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    long long arr2[4] = {};
    v.save(arr2);
    for(int i=0;i<4;++i) if(arr2[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "save", "Long256", "long long*", std::to_string(arr[i]), std::to_string(arr2[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.save(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    return result;
}
/*int test_edge_Long256_index() {
    int result = 0;
    Long256 v((long long)0);
#ifndef NDEBUG
    try { long long x = v[100]; (void)x; result = 1; } catch(...) {}
#else
    long long x = v[100];
    if(x != v[100%4]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "[]", "Long256", "long long", std::to_string(v[100%4]), std::to_string(x));
        result = 1;
    }
#endif
    return result;
}*/

// ULong256
int test_edge_ULong256_add() {
    int result = 0;
    ULong256 a((unsigned long long)0xFFFFFFFFFFFFFFFFULL), b((unsigned long long)1ULL);
    ULong256 c = a + b;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "ULong256", "ULong256", "0", c.str());
        result = 1;
    }
    a += b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "ULong256", "ULong256", "0", a.str());
        result = 1;
    }
    ULong256 d = c + (unsigned long long)0xFFFFFFFFFFFFFFFFULL;
    if (!(d == 0xFFFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+", "ULong256", "unsigned long long", std::to_string(0xFFFFFFFFFFFFFFFFULL), d.str());
        result = 1;
    }
    c += (unsigned long long)0xFFFFFFFFFFFFFFFFULL;
    if (!(c == 0xFFFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "+=", "ULong256", "unsigned long long", std::to_string(0xFFFFFFFFFFFFFFFFULL), c.str());
        result = 1;
    }
    return result;
}
int test_edge_ULong256_sub() {
    int result = 0;
    ULong256 a((unsigned long long)0), b((unsigned long long)1ULL);
    ULong256 c = a - b;
    if (!(c == 0xFFFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "ULong256", "ULong256", std::to_string(0xFFFFFFFFFFFFFFFFULL), c.str());
        result = 1;
    }
    a -= b;
    if (!(a == 0xFFFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "ULong256", "ULong256", std::to_string(0xFFFFFFFFFFFFFFFFULL), a.str());
        result = 1;
    }
    ULong256 d = c - (unsigned long long)0xFFFFFFFFFFFFFFFFULL;
    if (!(d == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-", "ULong256", "unsigned long long", "0", d.str());
        result = 1;
    }
    c -= (unsigned long long)0xFFFFFFFFFFFFFFFFULL;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "-=", "ULong256", "unsigned long long", "0", c.str());
        result = 1;
    }
    return result;
}
int test_edge_ULong256_mul() {
    int result = 0;
    ULong256 a((unsigned long long)0x8000000000000000ULL), b((unsigned long long)2ULL);
    ULong256 c = a * b;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "ULong256", "ULong256", "0", c.str());
        result = 1;
    }
    a *= b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "ULong256", "ULong256", "0", a.str());
        result = 1;
    }
    ULong256 d = c * (unsigned long long)0xFFFFFFFFFFFFFFFFULL;
    if (!(d == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*", "ULong256", "unsigned long long", "0", d.str());
        result = 1;
    }
    c *= (unsigned long long)0xFFFFFFFFFFFFFFFFULL;
    if (!(c == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "*=", "ULong256", "unsigned long long", "0", c.str());
        result = 1;
    }
    return result;
}
int test_edge_ULong256_div() {
    int result = 0;
    ULong256 a((unsigned long long)0xFFFFFFFFFFFFFFFFULL), b((unsigned long long)2ULL);
    ULong256 c = a / b;
    if (!(c == 0x7FFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "ULong256", "ULong256", std::to_string(0x7FFFFFFFFFFFFFFFULL), c.str());
        result = 1;
    }
    a /= b;
    if (!(a == 0x7FFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "ULong256", "ULong256", std::to_string(0x7FFFFFFFFFFFFFFFULL), a.str());
        result = 1;
    }
    ULong256 d = c / (unsigned long long)0x7FFFFFFFFFFFFFFFULL;
    if (!(d == 1ULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/", "ULong256", "unsigned long long", "1", d.str());
        result = 1;
    }
    c /= (unsigned long long)0x7FFFFFFFFFFFFFFFULL;
    if (!(c == 1ULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "/=", "ULong256", "unsigned long long", "1", c.str());
        result = 1;
    }
    return result;
}
int test_edge_ULong256_bitwise() {
    int result = 0;
    ULong256 a(0xF0F0F0F0F0F0F0F0ULL), b(0x0F0F0F0F0F0F0F0FULL);
    if (!((a & b) == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&", "ULong256", "ULong256", "0", (a & b).str());
        result = 1;
    }
    if (!((a | b) == 0xFFFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|", "ULong256", "ULong256", std::to_string(0xFFFFFFFFFFFFFFFFULL), (a | b).str());
        result = 1;
    }
    if (!((a ^ b) == 0xFFFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^", "ULong256", "ULong256", std::to_string(0xFFFFFFFFFFFFFFFFULL), (a ^ b).str());
        result = 1;
    }
    a &= b;
    if (!(a == 0)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "&=", "ULong256", "ULong256", "0", a.str());
        result = 1;
    }
    a = ULong256(0xF0F0F0F0F0F0F0F0ULL);
    a |= b;
    if (!(a == 0xFFFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "|=", "ULong256", "ULong256", std::to_string(0xFFFFFFFFFFFFFFFFULL), a.str());
        result = 1;
    }
    a = ULong256(0xF0F0F0F0F0F0F0F0ULL);
    a ^= b;
    if (!(a == 0xFFFFFFFFFFFFFFFFULL)) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "^=", "ULong256", "ULong256", std::to_string(0xFFFFFFFFFFFFFFFFULL), a.str());
        result = 1;
    }
    return result;
}
int test_edge_ULong256_load_save() {
    int result = 0;
    unsigned long long arr[4] = {1,2,3,4};
    ULong256 v;
    v.load(arr);
    for(int i=0;i<4;++i) if(v[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "load", "ULong256", "unsigned long long*", std::to_string(arr[i]), std::to_string(v[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.load(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    unsigned long long arr2[4] = {};
    v.save(arr2);
    for(int i=0;i<4;++i) if(arr2[i] != arr[i]) {
        testing::printTestFailed(__FILE__, __LINE__, __func__, "save", "ULong256", "unsigned long long*", std::to_string(arr[i]), std::to_string(arr2[i]));
        result = 1;
    }
#ifndef NDEBUG
    try { v.save(nullptr); result = 1; } catch(std::invalid_argument &e) {
        std::cout << e.what() << testing::testResultToColoredStrint(true) << '\n';
    }
#endif
    return result;
}


int main(int argc, char* argv[]) {
    int result = 0;
    
    result |= test_edge_Int256_add();
    result |= test_edge_Int256_sub();
    result |= test_edge_Int256_mul();
    result |= test_edge_Int256_div();
    result |= test_edge_Int256_bitwise();
    result |= test_edge_Int256_load_save();
    // result |= test_edge_Int256_index();
    result |= test_edge_Short256_add();
    result |= test_edge_Short256_sub();
    result |= test_edge_Short256_mul();
    result |= test_edge_Short256_div();
    result |= test_edge_Short256_bitwise();
    result |= test_edge_Short256_load_save();
    // result |= test_edge_Short256_index();
    result |= test_edge_UShort256_add();
    result |= test_edge_UShort256_sub();
    result |= test_edge_UShort256_mul();
    result |= test_edge_UShort256_div();
    result |= test_edge_UShort256_bitwise();
    result |= test_edge_UShort256_load_save();
    // result |= test_edge_UShort256_index();
    result |= test_edge_Long256_add();
    result |= test_edge_Long256_sub();
    result |= test_edge_Long256_mul();
    result |= test_edge_Long256_div();
    result |= test_edge_Long256_bitwise();
    result |= test_edge_Long256_load_save();
    // result |= test_edge_Long256_index();
    result |= test_edge_ULong256_add();
    result |= test_edge_ULong256_sub();
    result |= test_edge_ULong256_mul();
    result |= test_edge_ULong256_div();
    result |= test_edge_ULong256_bitwise();
    result |= test_edge_ULong256_load_save();
    // result |= test_edge_ULong256_index();

    std::cout << "Vectors equal: " << (avx::Long256(0xFFFF'FFFF'FFFF'FFFF) == 0xFFFF'FFFF'FFFF'FFFF) << '\n';
    
    return result;
}
