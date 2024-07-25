#pragma once
#ifndef MATH_HPP_
#define MATH_HPP_
#include "types.hpp"

namespace avx {
// Trigonometric functions

    Double256 sin(Double256&);
    Double256 cos(Double256&);
    Double256 tan(Double256&);
    Double256 cot(Double256&);
    Double256 sec(Double256&);
    Double256 csc(Double256&);

    Float256 sin(Float256&);
    Float256 cos(Float256&);
    Float256 tan(Float256&);
    Float256 cot(Float256&);
    Float256 sec(Float256&);
    Float256 csc(Float256&);

// Inverse trigonometric functions

    Double256 asin(Double256&);
    Double256 acos(Double256&);
    Double256 atan(Double256&);
    Double256 acot(Double256&);
    Double256 asec(Double256&);
    Double256 acsc(Double256&);

    Float256 asin(Float256&);
    Float256 acos(Float256&);
    Float256 atan(Float256&);
    Float256 acot(Float256&);
    Float256 asec(Float256&);
    Float256 acsc(Float256&);

// Hyperbolic functions

    Double256 sinh(Double256&);
    Double256 cosh(Double256&);
    Double256 tanh(Double256&);
    Double256 coth(Double256&);
    Double256 sech(Double256&);
    Double256 csch(Double256&);

    Float256 sinh(Float256&);
    Float256 cosh(Float256&);
    Float256 tanh(Float256&);
    Float256 coth(Float256&);
    Float256 sech(Float256&);
    Float256 csch(Float256&);

// Inverse hyperbolic functions

    Double256 asinh(Double256&);
    Double256 acosh(Double256&);
    Double256 atanh(Double256&);
    Double256 acoth(Double256&);
    Double256 acsch(Double256&);
    Double256 asech(Double256&);

    Float256 asinh(Float256&);
    Float256 acosh(Float256&);
    Float256 atanh(Float256&);
    Float256 acoth(Float256&);
    Float256 acsch(Float256&);
    Float256 asech(Float256&);

    Int256 abs(Int256&);
    Short256 abs(Short256&);
    Long256 abs(Long256&);
    Char256 abs(Char256&);
    Double256 abs(Double256&);
    Float256 abs(Float256);
};


#endif