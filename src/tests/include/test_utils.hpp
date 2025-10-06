#pragma once
#ifndef _AVXCPP_TEST_UTILS_HPP
#define _AVXCPP_TEST_UTILS_HPP

#include <regex>
#include <array>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <cstdio>
#include <memory>
#include <utility>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <functional>
#include <type_traits>
#include <immintrin.h>

#ifdef __GNUG__
    #include <cxxabi.h>
#endif

#ifdef _WIN32
    constexpr const char path_regex[] = "^.+\\\\(?=src)";
#else 
    constexpr const char path_regex[] = "^.+\\/(?=src)";
#endif

/** 
* #define is used for getting correct __LINE__ and __FILE__ macros as well as __func__
* Prints values of test variables to stderr.
* @param first First variable.
* @param second Second variable.
*/
#define printTestVariables(first, second) std::cerr << __FILE__  << ':' << __LINE__ << testing::demangle(__func__) << '(' << first << ", " << second << ")\n";

#define testFailed(op, typeA, typeB, expectedVal, actualVal) \
    testing::printTestFailed(__FILE__, __LINE__, testing::demangle(__func__).c_str(), op, testing::demangle(typeid(typeA).name()).c_str(), testing::demangle(typeid(typeB).name()).c_str(), expectedVal, actualVal)

namespace testing
{   
    constexpr const char* getCompilerName() {
        #if defined(__clang__)
            return "Clang";
        #elif defined(__GNUC__) || defined(__GNUG__)
            return "GCC";
        #elif defined(_MSC_VER)
            return "MSVC";
        #else
            return "n/a";
        #endif
    }

    constexpr int getCompilerMajor() {
        #if defined(__clang__)
            return __clang_major__;
        #elif defined(__GNUC__) || defined(__GNUG__)
            return __GNUC__;
        #elif defined(_MSC_VER)
            return _MSC_VER / 100;
        #else
            return 0;
        #endif
    }

    constexpr int getCompilerMinor() {
        #if defined(__clang__)
            return __clang_minor__;
        #elif defined(__GNUC__) || defined(__GNUG__)
            return __GNUC_MINOR__;
        #elif defined(_MSC_VER)
            return _MSC_VER % 100;
        #else
            return 0;
        #endif
    }

    constexpr int getCompilerPatchLevel() {
        #if defined(__clang__)
            return __clang_patchlevel__;
        #elif defined(__GNUC__) || defined(__GNUG__)
            return __GNUC_PATCHLEVEL__;
        #elif defined(_MSC_VER)
            return 0;
        #else
            return 0;
        #endif
    }

    constexpr const char* getPlatform() {
        #if defined(_WIN32)
            return "Windows";
        #elif defined(__APPLE__) && defined(__MACH__)
            return "macOS";
        #elif defined(__linux__)
            return "Linux";
        #elif defined(__unix__)
            return "Unix";
        #else
            return "Unknown Platform";
        #endif
    }

    std::string getSIMDFlags(){
        std::string flags;

        #ifdef __AVX__ //SSE not included as AVX and AVX2 is min requirement for library to work
            flags += "AVX ";
        #endif

        #ifdef __AVX2__
            flags += "AVX2 ";
        #endif

        #ifdef __AVX512F__
            flags += "AVX512F ";
        #endif

        #ifdef __AVX512VL__	    
            flags += "AVX512VL ";
        #endif

        #ifdef __AVX512BW__	    
            flags += "AVX512BW ";
        #endif

        #ifdef __AVX512DQ__	    
            flags += "AVX512DQ ";
        #endif

        #ifdef __AVX512CD__	    
            flags += "AVX512CD ";
        #endif

        #ifdef __AVX512ER__	    
            flags += "AVX512ER ";
        #endif

        #ifdef __AVX512PF__	    
            flags += "AVX512PF ";
        #endif

        #ifdef __AVX512IFMA__	    
            flags += "AVX512IFMA ";
        #endif

        #ifdef __AVX512VBMI__	    
            flags += "AVX512VBMI ";
        #endif

        #ifdef __AVX512VBMI2__	    
            flags += "AVX512VBMI2 ";
        #endif

        #ifdef __AVX512VNNI__	    
            flags += "AVX512VNNI ";
        #endif

        #ifdef __AVX512BITALG__	
            flags += "AVX512BITALG ";
        #endif

        #ifdef __AVX5124VNNIW__	
            flags += "AVX5124VNNIW ";
        #endif

        #ifdef __AVX5124FMAPS__
            flags += "AVX5124FMAPS ";
        #endif

        #ifdef __AVX512BF16__
            flags += "AVX512BF16 ";
        #endif

        #ifdef __AVX512VP2INTERSECT__
            flags += "AVX512VP2INTERSECT ";
        #endif

        #ifdef __AVX512FP16__ 
            flags += "AVX512FP16 ";
        #endif

        #ifdef __AVX10_VER__
            flags += "AVX10.1 ";
        #endif

        return flags;
    }

    template <typename T>
    constexpr T getMaxBits(){
        T val = 0xFF;
        for(int i{0}; i < sizeof(T); ++i)
            val = (val<<8)|(static_cast<T>(0xFF));
        return val;
    }

    std::string demangle(const char* name) {
        #ifdef __GNUG__
            std::string res;
            int status;
            char* realname = abi::__cxa_demangle(name, 0, 0, &status);
            if(realname){
                res.assign(realname) ;
                free(realname);
            }
            return res;
        #else
            return std::string(name);
        #endif
    }

    void printTestFailed(const char* filename, const int line, const char* func, const char* op, const char* type_a, const char* type_b, const std::string& expected, const std::string& actual) {
        std::string tmp(filename);
        std::smatch match;

        if(std::regex_search(tmp, match, std::regex(path_regex)))
            tmp = match.suffix().str();
        
        fprintf(
            stderr,
            "%s:%d Test %s (%s %s %s) failed! Expected %s actual %s\n",
            tmp.c_str(),
            line,
            func, 
            type_a,
            op, 
            type_b,
            expected.c_str(),
            actual.c_str()
        );
    }

    std::pair<double, std::string> universalDuration(int64_t ticks) {
        static const std::array<std::string, 5> times{"ns", "us", "ms", "s", "m"};
        unsigned int i = 0;
        while(ticks / pow(1000., i) > 1000) ++i;

        if(i > 4)
            return {ticks / pow(1000., 4), "m"};

        return {ticks / pow(1000., i), times[i]};
    }

    void printTestDuration(const char* func, std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop) {
        std::pair<double, std::string> duration = universalDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count());

        if(duration.second == "ns")
            printf("Test %s finished in %ld %s\n", func, static_cast<long int>(duration.first), duration.second.c_str());
        else if(duration.second == "us")
            printf("Test %s finished in %.2lf %s\n", func, duration.first, duration.second.c_str());
        else 
            printf("Test %s finished in %.4lf %s\n", func, duration.first, duration.second.c_str());
    }


    /**
     Below are the functions for arithmetic testing. Each one of them tests one operator but using different types and assignement e.g.
     universalTest_add tests + and += operator where arguments are -> corresponding SIMD declared class and literal e.g. Int256 and int.
     So the order of testing is as follows (using universalTest_add with Int256 as an example):
     
     Int256 + Int256
     Int256 += Int256
     Int256 + int
     Int256 += int

     Those functions do not follow DRY principle as to not introduce fragmentation and due to the fact that it would complicate some things unnecessarily.
     */

    /**
     * Universal function for testing `+` and `+=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestAdd(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S maxval{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i){
            maxval = (maxval << 8) | static_cast<S>(0xFF);
        }

        std::uniform_int_distribution<std::mt19937::result_type> dist(1, maxval);

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] + bV[i];
            litV[i] = aV[i] + randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a + b;
        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "+", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c += b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "+=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a + randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "+", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c += randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "+=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }

    /**
     * Universal function for testing `-` and `-=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestSub(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S zero{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i){
            zero = (zero << 8) | static_cast<S>(0xFF);
        }
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] - bV[i];
            litV[i] = aV[i] - randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a - b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "-", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c -= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "-=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a - randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "-", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c -= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "-=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `*` and `*=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestMul(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S zero{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i){
            zero = (zero << 8) | static_cast<S>(0xFF);
        }
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] * bV[i];
            litV[i] = aV[i] * randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a * b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "*", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c *= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "*=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a * randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "*", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c *= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "*=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `/` and `/=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestDiv(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S zero{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i){
            zero = (zero << 8) | static_cast<S>(0xFF);
        }
        
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

        while(!(randLit = dist(rng)));
        
        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            while(!(bV[i] = dist(rng)));
            resV[i] = aV[i] / bV[i];
            litV[i] = aV[i] / randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a / b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "/", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c /= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "/=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a / randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "/", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c /= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "/=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }

    /**
     * Universal function for testing `%` and `%=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestMod(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S zero{static_cast<S>(0xFF)}, randLit;

        for(unsigned i = 0; i < sizeof(S);++i){
            zero = (zero << 8) | static_cast<S>(0xFF);
        }

        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

        while(!(randLit = dist(rng)));

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            while(!(bV[i] = dist(rng)));
            resV[i] = aV[i] % bV[i];
            litV[i] = aV[i] % randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a % b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "%", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c %= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "%=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a % randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "%", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c %= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "%=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `<`< and` <`<= operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestLshift(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        unsigned int randLit;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, sizeof(S) * 8 - 1); // -1 to avoid undefined behaviour.

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] << bV[i];
            litV[i] = aV[i] << randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a << b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "<<", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c <<= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "<<=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a << randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "<<", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c <<= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "<<=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `>`> and` >`>= operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestRshift(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        unsigned int randLit;
        S min_v = static_cast<S>(INT_MIN), max_v = static_cast<S>(INT_MAX);
        using sel_type = std::conditional_t<std::is_same_v<S, char> || std::is_same_v<S, unsigned char>,
            short,
            S>;
        if constexpr(std::is_same_v<S, char>){
            min_v = -128;
            max_v = 127;
        }
        else if constexpr(std::is_same_v<S, unsigned char>){
            min_v = 0;
            max_v = 255;
        }
        else {
            min_v = std::numeric_limits<S>::min();
            max_v = std::numeric_limits<S>::max();
        }

        std::uniform_int_distribution<sel_type> dist(min_v, max_v);


        randLit = std::rand() % (sizeof(S) * 8 - 1);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = static_cast<unsigned long>(dist(rng)) % (sizeof(S) * 8);
            resV[i] = aV[i] >> bV[i];
            litV[i] = aV[i] >> randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a >> b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c >>= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a >> randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c >>= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `|` and `|=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestOR(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S maxval{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i){
            maxval = (maxval << 8) | static_cast<S>(0xFF);
        }
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, maxval);

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] | bV[i];
            litV[i] = aV[i] | randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a | b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "|", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c |= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "|=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a | randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "|", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c |= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "|=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `&` and `&=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestAND(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S maxval{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i){
            maxval = (maxval << 8) | static_cast<S>(0xFF);
        }
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, maxval);

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] & bV[i];
            litV[i] = aV[i] & randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a & b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "&", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c &= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "&=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a & randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "&", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c &= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "&=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `^` and `^=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestXOR(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S maxval{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i){
            maxval = (maxval << 8) | static_cast<S>(0xFF);
        }
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, maxval);

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] ^ bV[i];
            litV[i] = aV[i] ^ randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a ^ b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "^", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c ^= b;

        if(c != expected){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "^=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a ^ randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "^", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c ^= randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                "^=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expectedLit.str(), 
                c.str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }



    /**
     * Universal function for testing `^` and `^=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S = typename T::storedType>
    int universalTestNOT(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S maxval{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i){
            maxval = (maxval << 8) | static_cast<S>(0xFF);
        }
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, maxval);


        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            resV[i] = aV[i] ^ maxval; // XOR because ~aV[i] may fail to compile for some integer types.
        }

        T a(aV.data()), c, expected(resV.data());
        c = ~a;
        std::string tmp(__FILE__);
        std::smatch match;

        if(std::regex_search(tmp, match, std::regex(path_regex)))
            tmp = match.suffix().str();
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            fprintf(
                stderr,
                "%s:%d Test %s (~%s) failed! Expected %s actual %s\n",
                tmp.c_str(), 
                __LINE__, 
                __func__, 
                demangle(typeid(T).name()).c_str(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " ~A: " << c.str() << " expected: " << expected.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }

    template<typename T, typename S = typename T::storedType>
    int universalTestIndexing(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        S maxval{static_cast<S>(0xFF)}, randLit;
        for(unsigned i = 0; i < sizeof(S);++i)
            maxval = (maxval << 8) | static_cast<S>(0xFF);
        
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, maxval);

        for(unsigned int i{0}; i < size; ++i)
            aV[i] = dist(rng);
        
        T a(aV.data());

        std::string tmp(__FILE__);
        std::smatch match;

        if(std::regex_search(tmp, match, std::regex(path_regex)))
            tmp = match.suffix().str();

        #ifndef NDEBUG
                for(unsigned int i{0}; i < size; ++i)
                {
                    try{
                        if(a[i] != aV[i]) { 
                            // std::cerr cause different argument types to avoid wrong values display.
                            std::cerr << tmp << ':' << __LINE__ << " Test " << __func__ << " (" << demangle(typeid(T).name()).c_str() << "[" << i << "]) failed! Expected " << aV[i] << " actual " << a[i] << '\n',
                            result = 1;    
                        }
                    } catch (std::out_of_range& e){
                        fprintf(
                            stderr,
                            "%s:%d Test %s: Exception std::out_of_range %s[%d] -> %s\n",
                            tmp.c_str(),
                            __LINE__,
                            __func__,
                            demangle(typeid(T).name()).c_str(),
                            i,
                            e.what()
                        );
                        result = 1;
                        break;
                        
                    }
                }
        #else
            for(unsigned int i{0}; i < size; ++i) {
                if(a[i] != aV[i]) {
                    // std::cerr cause different argument types to avoid wrong values display.
                    std::cerr << tmp << ':' << __LINE__ << " Test " << __func__ << " (" << demangle(typeid(T).name()).c_str() << "[" << i << "]) failed! Expected " << aV[i] << " actual " << a[i] << '\n',
                    result = 1;    
                }
            }
        #endif

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);

        return result;
    }


    template <typename T, typename S = typename T::storedType>
    int universalTestCompare(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();

        const std::string excpected = "true";
        const std::string actual = "false";

        std::vector<S> eqTest(size);
        std::vector<S> neqTest(size);
        std::vector<S> constVal(size);
        std::vector<S> zeros(size, 0) ;
        std::vector<S> ones(size, getMaxBits<S>());

        for(int i{0};i < size; ++i){
            eqTest[i] = i;
            neqTest[i] = i;
            constVal[i] = 3;
        }

        neqTest[size - 1] += 1;

        T a(eqTest.data()), b(neqTest.data()), c(zeros.data()), d(ones.data()), f(constVal.data());

        if(!(a == a)){
            printTestFailed(
                __FILE__,
                __LINE__,
                __func__,
                "==",
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                excpected,
                actual
            );
            result = 1;
        }

        if(!(a != b)){
            printTestFailed(
                __FILE__,
                __LINE__,
                __func__,
                "!=",
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                excpected,
                actual
            );
            result = 1;
        }

        if(!(c == c)){
            printTestFailed(
                __FILE__,
                __LINE__,
                __func__,
                "== (0)",
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                excpected,
                actual
            );
            result = 1;
        }

        if(!(d == d)){
            printTestFailed(
                __FILE__,
                __LINE__,
                __func__,
                "== (MAX)",
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(T).name()).c_str(),
                excpected,
                actual
            );
            result = 1;
        }

        if(!(f == 3)){
            printTestFailed(
                __FILE__,
                __LINE__,
                __func__,
                "==",
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                excpected,
                actual
            );
            result = 1;
        }

        if(!(f != 4)){
            printTestFailed(
                __FILE__,
                __LINE__,
                __func__,
                "!=",
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                excpected,
                actual
            );
            result = 1;
        }

        auto stop = std::chrono::steady_clock::now();

        printTestDuration(__func__, start, stop);        

        return result;
    }

    std::string testResultToColoredStrint(const bool &result) {
        if(result)
            return "[\033[32mOK\033[0m]";
        return "[\033[31mFAIL\033[0m]";
    }

    template<typename T, typename S = typename T::storedType>
    int universalTestBorderVal(S minval, S maxval, const unsigned int size = T::size) {
        T a, b, expected;
        S tmp;
        std::vector<S> buffer(size), results(size);
        int returnVal = 1;

        if(minval) {
            for(S i{0}; i < size; ++i) {
                buffer[i] = minval + i;
                results[i] = buffer[i] / 2;
            }

            a.load(buffer.data());
            b.load(buffer.data());

            if(a != b) 
                testFailed("/", T, T, b.str(), a.str());
        }

        return returnVal;
    }
};
#endif