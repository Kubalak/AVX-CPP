#pragma once
#ifndef _TEST_UTILS_HPP
#define _TEST_UTILS_HPP

#include <regex>
#include <array>
#include <vector>
#include <functional>
#include <random>
#include <string>
#include <chrono>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <fstream>

#ifdef _WIN32
constexpr const char path_regex[] = "^.+\\\\(?=src)";
#else 
constexpr const char path_regex[] = "^.+//(?=src)";
#endif

namespace testing
{   
    constexpr const char* get_compiler_name() {
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

    constexpr int get_compiler_major() {
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

    constexpr int get_compiler_minor() {
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

    constexpr int get_compiler_patch_level() {
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

    constexpr const char* get_platform() {
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

    void print_test_failed(const char* filename, const int line, const char* func, const char* op, const char* type_a, const char* type_b, const std::string& expected, const std::string& actual) {
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

    std::pair<double, std::string> universal_duration(int64_t ticks) {
        static const std::array<std::string, 5> times{"ns", "us", "ms", "s", "m"};
        unsigned int i = 0;
        while(ticks / pow(1000., i) > 1000) ++i;

        if(i > 4)
            return {ticks / pow(1000., 4), "m"};

        return {ticks / pow(1000., i), times[i]};
    }

    void print_test_duration(const char* func, std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop) {
        std::pair<double, std::string> duration = universal_duration(std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count());

        if(duration.second == "ns")
            printf("Test %s finished in %ld %s\n", func, static_cast<long int>(duration.first), duration.second.c_str());
        else if(duration.second == "us")
            printf("Test %s finished in %.2lf %s\n", func, duration.first, duration.second.c_str());
        else 
            printf("Test %s finished in %.4lf %s\n", func, duration.first, duration.second.c_str());
    }

    /**
     * Applies a function on each pair of elements from vectors.
     * @param va First vector.
     * @param vb Second vector.
     * @param f Function to be applied.
     * @return Vector of same size containing results of f(va[n], vb[n]) or vector of size 0 if sizes don't match.
     */
    template <typename T, typename Func>
    std::vector<T> apply_seq(const std::vector<T> &va, const std::vector<T> &vb, Func f)
    {
        static_assert(std::is_invocable_r_v<T, Func, T, T>, "Passed function needs to have 2 arguments!");

        if (va.size() != vb.size())
            return {};
        std::vector<T> result(va.size());

        for (size_t i = 0; i < va.size(); ++i)
            result[i] = f(va[i], vb[i]);

        return result;
    }

    /**
     * Applies a function on each elemnt of va and literal b.
     * @param va Vector.
     * @param b Literal.
     * @param f Function to be applied.
     * @return Vector of same size containing results of f(va[n], b).
     */
    template <typename T, typename Func>
    std::vector<T> apply_lit(const std::vector<T> &va, const T &b, Func f)
    {
        static_assert(std::is_invocable_r_v<T, Func, T, T>, "Passed function needs to have 2 arguments!");
        std::vector<T> result(va.size());

        for (size_t i = 0; i < va.size(); ++i)
            result[i] = f(va[i], b);

        return result;
    }

    /**
     * Adds a to b.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a + b.
     */
    template <typename T>
    T add(const T &a, const T &b)
    {
        return a + b;
    }

    /**
     * Subs b from a.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a - b.
     */
    template <typename T>
    T sub(const T &a, const T &b)
    {
        return a - b;
    }

    /**
     * Multiplys a by b.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a * b.
     */
    template <typename T>
    T mul(const T &a, const T &b)
    {
        return a * b;
    }

    /**
     * Divides a by b.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a / b.
     */
    template <typename T>
    T div(const T &a, const T &b)
    {
        return a / b;
    }

    /**
     * Performs a modulo division.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a % b.
     */
    template <typename T>
    T mod(const T &a, const T &b)
    {
        return a % b;
    }

    /**
     * Bitwise OR.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a | b.
     */
    template <typename T>
    T b_or(const T &a, const T &b)
    {
        return a | b;
    }

    /**
     * Bitwise AND.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a & b.
     */
    template <typename T>
    T b_and(const T &a, const T &b)
    {
        return a & b;
    }

    /**
     * Bitwise XOR.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a ^ b.
     */
    template <typename T>
    T b_xor(const T &a, const T &b)
    {
        return a ^ b;
    }

    /**
     * Left bits shift.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a << b.
     */
    template <typename T>
    T lshift(const T &a, const T &b)
    {
        return a << b;
    }

    /**
     * Right bits shift.
     * @param a First argument.
     * @param b Second argument.
     * @return The result of a >> b.
     */
    template <typename T>
    T rshift(const T &a, const T &b)
    {
        return a >> b;
    }

    /**
     Below are the functions for arithmetic testing. Each one of them tests one operator but using different types and assignement e.g.
     universal_test_add tests + and += operator where arguments are -> corresponding SIMD declared class and literal e.g. Int256 and int.
     So the order of testing is as follows (using universal_test_add with Int256 as an example):
     
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
    template <typename T, typename S>
    int universal_test_add(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "+", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c += b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "+=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a + randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "+", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c += randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "+=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }

    /**
     * Universal function for testing `-` and `-=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_sub(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "-", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c -= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "-=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a - randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "-", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c -= randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "-=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `*` and `*=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_mul(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "*", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c *= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "*=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a * randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "*", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c *= randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "*=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `/` and `/=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_div(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "/", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c /= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "/=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a / randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "/", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c /= randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "/=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }

    /**
     * Universal function for testing `%` and `%=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_mod(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "%", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c %= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "%=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a % randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "%", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c %= randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "%=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `<`< and` <`<= operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_lshift(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "<<", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c <<= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "<<=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a << randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "<<", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c <<= randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "<<=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `>`> and` >`>= operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_rshift(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c >>= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a >> (const unsigned int&)randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c >>= (const unsigned int&)randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                ">>=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `|` and `|=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_or(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "|", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c |= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "|=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a | randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "|", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c |= randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "|=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `&` and `&=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_and(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "&", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c &= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "&=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a & randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "&", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c &= randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "&=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }


    /**
     * Universal function for testing `^` and `^=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_xor(const unsigned int size = T::size) {
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
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "^", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c ^= b;

        if(c != expected){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "^=", 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a ^ randLit;
        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "^", 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str(), 
                c.str()
            );
            result = 1;
        }

        c = a;
        c ^= randLit;

        if(c != expectedLit){
            print_test_failed(
                __FILE__, 
                __LINE__, 
                __func__,
                "^=", 
                typeid(T).name(), 
                typeid(S).name(),
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

        print_test_duration(__func__, start, stop);

        return result;
    }



    /**
     * Universal function for testing `^` and `^=` operator of integer types.
     * Writes to `stderr` in case of failure.
     * 
     * @param size Elements count in type `T`.
     * @return 0 on success or 1 on failure.
     */
    template <typename T, typename S>
    int universal_test_not(const unsigned int size = T::size) {
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
                typeid(T).name(), 
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " ~A: " << c.str() << " expected: " << expected.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        print_test_duration(__func__, start, stop);

        return result;
    }

    template<typename T, typename S>
    int universal_test_indexing(const unsigned int size = T::size) {
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
                            std::cerr << tmp << ':' << __LINE__ << " Test " << __func__ << " (" << typeid(T).name() << "[" << i << "]) failed! Expected " << aV[i] << " actual " << a[i] << '\n',
                            result = 1;    
                        }
                    } catch (std::out_of_range& e){
                        fprintf(
                            stderr,
                            "%s:%d Test %s: Exception std::out_of_range %s[%d] -> %s\n",
                            tmp.c_str(),
                            __LINE__,
                            __func__,
                            typeid(T).name(),
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
                    std::cerr << tmp << ':' << __LINE__ << " Test " << __func__ << " (" << typeid(T).name() << "[" << i << "]) failed! Expected " << aV[i] << " actual " << a[i] << '\n',
                    result = 1;    
                }
            }
        #endif

        auto stop = std::chrono::steady_clock::now();

        print_test_duration(__func__, start, stop);

        return result;
    }

    /*template <typename T, typename S>
    int universal_test_limits(unsigned int size = T::size){
        S ffs = UINT64_MAX;
        S Offs = UINT64_MAX >> 1;
        return 0;
    }*/

   template<typename T, typename S>
   int universal_test_perf_avx(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV) {
    if(aV.size() != bV.size() || aV.size() != cV.size()){
        std::cerr << "Vector sizes don't match!\n";
        return 1;
    }

    /* // Warmup code should go here
    for(size_t i{0}; i < aV.size() / 4; ++i){

    }*/
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t index = 0;
    for(; index < aV.size() - T::size; index += T::size){
        T a(aV.data() + index);
        T b(bV.data() + index);
        T c = a + b;
        c += 3;
        c *= 2;
        c = c / 4;
        c <<= 2;
        c *= b;
        c -= a;
        c.save(cV.data() + index);
    }

    for(;index < aV.size(); ++index){
        cV[index] = aV[index] + bV[index];
        cV[index] += 3;
        cV[index] *= 2;
        cV[index] = cV[index] / 4;
        cV[index] <<= 2;
        cV[index] *= bV[index];
        cV[index] -= aV[index];
    } 


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    auto formatted_duration = universal_duration(duration);
    std::cout << "Test " << __func__ << " finished in " << formatted_duration.first << ' ' << formatted_duration.second <<'\n';

    return 0;
   }

    template<typename T, typename S>
    int universal_test_perf_seq(const std::vector<S>& aV, const std::vector<S>& bV, std::vector<S>& cV) {
        if(aV.size() != bV.size() || aV.size() != cV.size()){
            std::cerr << "Vector sizes don't match!\n";
            return 1;
        }

        /* // Warmup code should go here
        for(size_t i{0}; i < aV.size() / 4; ++i){

        }*/
        
        auto start = std::chrono::high_resolution_clock::now();
        const S *aP = aV.data(), *bP = bV.data();
        S *cP = cV.data();
        for(size_t index = 0;index < aV.size(); ++index){
            cP[index] = aP[index] + bP[index];
            cP[index] += 3;
            cP[index] *= 2;
            cP[index] = cP[index] / 4;
            cP[index] <<= 2;
            cP[index] *= bP[index];
            cP[index] -= aP[index];
        } 


        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        auto formatted_duration = universal_duration(duration);
        std::cout << "Test " << __func__ << " finished in " << formatted_duration.first << ' ' << formatted_duration.second <<'\n';

        return 0;
   }

    bool file_exists(const std::string &filename){
        return std::filesystem::exists(filename) && std::filesystem::is_regular_file(filename);
    }

    template <typename T>
    bool read_file(const std::string &filename, std::vector<T> &dest) {
        if(!file_exists(filename)) return false;
        
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