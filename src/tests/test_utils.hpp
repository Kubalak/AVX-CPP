#pragma once
#ifndef _TEST_UTILS_HPP
#define _TEST_UTILS_HPP
#include <vector>
#include <functional>
#include <random>
#include <string>
#include <chrono>

namespace testing
{
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
        S zero{0}, randLit;
        zero |= 0xFFFFFFFFFFFFFFFF;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

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
            // Fprintf bcz std::cout is pain in the 455.
            fprintf(
                stderr, 
                "%s:%d Test %s (%s + %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c += b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s += %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a + randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s + %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c += randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s += %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S zero{0}, randLit;
        zero |= 0xFFFFFFFFFFFFFFFF;
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
            fprintf(
                stderr, 
                "%s:%d Test %s (%s - %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c -= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s -= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a - randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s - %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c -= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s -= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S zero{0}, randLit;
        zero |= 0xFFFFFFFFFFFFFFFF;
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
            fprintf(
                stderr, 
                "%s:%d Test %s (%s * %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c *= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s *= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a * randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s * %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c *= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s *= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S zero{0}, randLit;
        zero |= 0xFFFFFFFFFFFFFFFF;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] / bV[i];
            litV[i] = aV[i] / randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a / b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            fprintf(
                stderr, 
                "%s:%d Test %s (%s / %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c /= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s /= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a / randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s / %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c /= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s /= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S zero{0}, randLit;
        zero |= 0xFFFFFFFFFFFFFFFF;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

        randLit = dist(rng);

        for(unsigned int i = 0; i < size; ++i){
            aV[i] = dist(rng);
            bV[i] = dist(rng);
            resV[i] = aV[i] % bV[i];
            litV[i] = aV[i] % randLit;
        }

        T a(aV.data()), b(bV.data()), c, expected(resV.data()), expectedLit(litV.data());

        c = a % b;
        if(c != expected){
            // Fprintf bcz std::cout is pain in the 455.
            fprintf(
                stderr, 
                "%s:%d Test %s (%s %% %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c %= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s %%= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a % randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s %% %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c %= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s %%= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S randLit;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, sizeof(S) * 8);

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
            fprintf(
                stderr, 
                "%s:%d Test %s (%s << %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c <<= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s <<= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a << randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s << %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c <<= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s <<= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S randLit;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, sizeof(S) * 8);

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
            fprintf(
                stderr, 
                "%s:%d Test %s (%s >> %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c >>= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s >>= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a >> randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s >> %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c >>= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s >>= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S zero{0}, randLit;
        zero |= 0xFFFFFFFFFFFFFFFF;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

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
            fprintf(
                stderr, 
                "%s:%d Test %s (%s | %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c |= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s |= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a | randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s | %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c |= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s |= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S zero{0}, randLit;
        zero |= 0xFFFFFFFFFFFFFFFF;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

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
            fprintf(
                stderr, 
                "%s:%d Test %s (%s & %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c &= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s &= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a & randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s & %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c &= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s &= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

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
        S zero{0}, randLit;
        zero |= 0xFFFFFFFFFFFFFFFF;
        std::uniform_int_distribution<std::mt19937::result_type> dist(1, zero);

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
            fprintf(
                stderr, 
                "%s:%d Test %s (%s ^ %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c ^= b;

        if(c != expected){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s ^= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(T).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a ^ randLit;
        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s ^ %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        c = a;
        c ^= randLit;

        if(c != expectedLit){
            fprintf(
                stderr, 
                "%s:%d Test %s (%s ^= %s) failed! Expected %s actual %s\n",
                __FILE__, 
                __LINE__, 
                __func__, 
                typeid(T).name(), 
                typeid(S).name(),
                expected.str().c_str(), 
                c.str().c_str()
            );
            result = 1;
        }

        if(result){
            std::cerr << "A: " << a.str() << " B: " << b.str() << " expected: " << expected.str() << '\n';
            std::cerr << "Literal: " << randLit << " expected: " << expectedLit.str() << '\n';
        }

        auto stop = std::chrono::steady_clock::now();

        printf("Test %s finished in %.4lf us\n", __func__, std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count()/1000.f);

        return result;
    }
};
#endif