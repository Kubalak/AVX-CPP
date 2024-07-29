#pragma once
#ifndef MATH_HPP_
#define MATH_HPP_
#include "types.hpp"
#include "int256.hpp"
#include <set>

namespace avx {
// Trigonometric functions

    Double256 sin(const Double256&);
    Double256 cos(const Double256&);
    Double256 tan(const Double256&);
    Double256 cot(const Double256&);
    Double256 sec(const Double256&);
    Double256 csc(const Double256&);

    Float256 sin(const Float256&);
    Float256 cos(const Float256&);
    Float256 tan(const Float256&);
    Float256 cot(const Float256&);
    Float256 sec(const Float256&);
    Float256 csc(const Float256&);

// Inverse trigonometric functions

    Double256 asin(const Double256&);
    Double256 acos(const Double256&);
    Double256 atan(const Double256&);
    Double256 acot(const Double256&);
    Double256 asec(const Double256&);
    Double256 acsc(const Double256&);

    Float256 asin(const Float256&);
    Float256 acos(const Float256&);
    Float256 atan(const Float256&);
    Float256 acot(const Float256&);
    Float256 asec(const Float256&);
    Float256 acsc(const Float256&);

// Hyperbolic functions

    Double256 sinh(const Double256&);
    Double256 cosh(const Double256&);
    Double256 tanh(const Double256&);
    Double256 coth(const Double256&);
    Double256 sech(const Double256&);
    Double256 csch(const Double256&);

    Float256 sinh(const Float256&);
    Float256 cosh(const Float256&);
    Float256 tanh(const Float256&);
    Float256 coth(const Float256&);
    Float256 sech(const Float256&);
    Float256 csch(const Float256&);

// Inverse hyperbolic functions

    Double256 asinh(const Double256&);
    Double256 acosh(const Double256&);
    Double256 atanh(const Double256&);
    Double256 acoth(const Double256&);
    Double256 acsch(const Double256&);
    Double256 asech(const Double256&);

    Float256 asinh(const Float256&);
    Float256 acosh(const Float256&);
    Float256 atanh(const Float256&);
    Float256 acoth(const Float256&);
    Float256 acsch(const Float256&);
    Float256 asech(const Float256&);

    Int256 abs(const Int256&);
    Short256 abs(const Short256&);
    Long256 abs(const Long256&);
    Char256 abs(const Char256&);
    Double256 abs(const Double256&);
    Float256 abs(const Float256&);

    template<typename T, typename Iter>
    T sum(Iter begin, Iter end) {
        T result;
        while(begin != end){
            result += begin;
            ++begin;
        }
        return result;
    }

};

#endif