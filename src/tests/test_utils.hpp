#pragma once
#ifndef _TEST_UTILS_HPP
#define _TEST_UTILS_HPP
#include <vector>
#include <functional>

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
};
#endif