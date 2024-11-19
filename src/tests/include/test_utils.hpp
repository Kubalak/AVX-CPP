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
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <functional>
#include <immintrin.h>

#ifdef __GNUG__
    #include <cxxabi.h>
#endif

#ifdef _WIN32
constexpr const char path_regex[] = "^.+\\\\(?=src)";
#else 
constexpr const char path_regex[] = "^.+//(?=src)";
#endif

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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
    int universalTestRshift(const unsigned int size = T::size) {
        int result = 0;
        auto start = std::chrono::steady_clock::now();
        std::vector<S> aV(size), bV(size), resV(size), litV(size);
        std::random_device dev;
        std::mt19937 rng(dev());
        unsigned int randLit;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, sizeof(S) * 8 - 1); // -1 to avoid undefined behaviour

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
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

        c = a >> (const unsigned int&)randLit;
        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c >>= (const unsigned int&)randLit;

        if(c != expectedLit){
            printTestFailed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>=", 
                demangle(typeid(T).name()).c_str(),
                demangle(typeid(S).name()).c_str(),
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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
                expected.str(), 
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
                expected.str(), 
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
    template <typename T, typename S = T::storedType>
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

    template<typename T, typename S = T::storedType>
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

    bool fileExists(const std::string &filename){
        return std::filesystem::exists(filename) && std::filesystem::is_regular_file(filename);
    }

    template <typename T>
    bool readFile(const std::string &filename, std::vector<T> &dest) {
        if(!fileExists(filename)) return false;
        
        uint64_t file_size = std::filesystem::file_size(filename);
        if(!file_size) return false;

        if(file_size / sizeof(T) != dest.size())
            dest.resize(file_size / sizeof(T));

        std::ifstream infile(filename, std::ios_base::binary);

        if(!infile.good()) return false;

        infile.read((char*)dest.data(), file_size);

        auto bytes_read = infile.gcount();
        
        infile.close();
        if(bytes_read != file_size) return false;
        
        return true;
    }   
};
#endif