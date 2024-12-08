#pragma once
#ifndef MATH_HPP_
#define MATH_HPP_

#include "types/long256.hpp"
#include "types/int256.hpp"
#include "types/short256.hpp"
#include "types/char256.hpp"
#include "types/double256.hpp"
#include "types/float256.hpp"
#include <set>

namespace avx {

    Double256 sin(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_sin_pd(bV.get());
        #else 
            return bV;
        #endif
    }

    Double256 cos(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_cos_pd(bV.get());
        #else
            return bV;
        #endif
    }

    Double256 tan(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_tan_pd(bV.get());
        #else
            return bV;
        #endif
    }

    Double256 ctg(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_pd(
                constants::DOUBLE_ONE, 
                _mm256_tan_pd(bV.get())
            );
        #else
            return bV;
        #endif
    }

    Double256 sec(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_pd(
                constants::DOUBLE_ONE, 
                _mm256_cos_pd(bV.get())
            ); 
        #else
            return bV;
        #endif
    }

    Double256 cosec(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_pd(
                constants::DOUBLE_ONE, 
                _mm256_sin_pd(bV.get())
            );
        #else
            return bV;
        #endif
    }

    Float256 sin(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_sin_ps(bV.get());
        #else
            return bV;
        #endif
    }

    Float256 cos(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_cos_ps(bV.get());
        #else
            return bV;
        #endif
    }

    Float256 tan(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_tan_ps(bV.get());
        #else
            return bV;
        #endif
    }

    Float256 ctg(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_ps(
                constants::FLOAT_ONE, 
                _mm256_tan_ps(bV.get())
            );
        #else
            return bV;
        #endif
    }
    
    Float256 sec(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_ps(
                constants::FLOAT_ONE, 
                _mm256_cos_ps(bV.get())
            );
        #else
            return bV;
        #endif
    }

    Float256 cosec(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_ps(
                constants::FLOAT_ONE, 
                _mm256_sin_ps(bV.get())
            );
        #else
            return bV;
        #endif
    }

// Inverse trigonometric functions

    Double256 asin(const Double256& bV);
    Double256 acos(const Double256& bV);
    Double256 atan(const Double256& bV);
    Double256 acot(const Double256& bV);
    Double256 asec(const Double256& bV);
    Double256 acsc(const Double256& bV);

    Float256 asin(const Float256& bV);
    Float256 acos(const Float256& bV);
    Float256 atan(const Float256& bV);
    Float256 acot(const Float256& bV);
    Float256 asec(const Float256& bV);
    Float256 acsc(const Float256& bV);

// Hyperbolic functions

    Double256 sinh(const Double256& bV);
    Double256 cosh(const Double256& bV);
    Double256 tanh(const Double256& bV);
    Double256 coth(const Double256& bV);
    Double256 sech(const Double256& bV);
    Double256 csch(const Double256& bV);

    Float256 sinh(const Float256& bV);
    Float256 cosh(const Float256& bV);
    Float256 tanh(const Float256& bV);
    Float256 coth(const Float256& bV);
    Float256 sech(const Float256& bV);
    Float256 csch(const Float256& bV);

// Inverse hyperbolic functions

    Double256 asinh(const Double256& bV);
    Double256 acosh(const Double256& bV);
    Double256 atanh(const Double256& bV);
    Double256 acoth(const Double256& bV);
    Double256 acsch(const Double256& bV);
    Double256 asech(const Double256& bV);

    Float256 asinh(const Float256& bV);
    Float256 acosh(const Float256& bV);
    Float256 atanh(const Float256& bV);
    Float256 acoth(const Float256& bV);
    Float256 acsch(const Float256& bV);
    Float256 asech(const Float256& bV);

    Int256 abs(const Int256& bV) {
        __m256i toAbs = _mm256_srai_epi32(bV.get(), 31);
        __m256i absVal = _mm256_xor_si256(bV.get(), toAbs);
        return _mm256_add_epi32(_mm256_and_si256(toAbs, constants::EPI32_ONE), absVal);
    }

    Short256 abs(const Short256& bV) {
        __m256i toAbs = _mm256_srai_epi16(bV.get(), 15);
        __m256i absVal = _mm256_xor_si256(bV.get(), toAbs);
        return _mm256_add_epi16(_mm256_and_si256(toAbs, constants::EPI16_ONE), absVal);
    }

    Long256 abs(const Long256& bV) {
        __m256i toAbs = _mm256_srai_epi64(bV.get(), 63);
        __m256i absVal = _mm256_xor_si256(bV.get(), toAbs);
        return _mm256_add_epi64(_mm256_and_si256(toAbs, constants::EPI64_ONE), absVal);
    }

    Char256 abs(const Char256& bV) {
        __m256i toAbs = _mm256_cmpgt_epi8(_mm256_setzero_si256(), bV.get());
        __m256i absVal = _mm256_xor_si256(bV.get(), toAbs);
        return _mm256_add_epi8(_mm256_and_si256(toAbs, constants::EPI8_ONE), absVal);
    }

    Double256 abs(const Double256& bV) {
        return _mm256_and_pd(bV.get(), constants::DOUBLE_NO_SIGN);
    }

    Float256 abs(const Float256& bV) {
        return _mm256_and_ps(bV.get(), constants::FLOAT_NO_SIGN);
    }

    /**
     * Generic sum algorithm.
     * @param begin Start of container iterator.
     * @param end End of container iterator.
     * @returns Sum of elements.
     */
    template<typename T, typename Iter>
    T sum(Iter begin, Iter end) {
        T result;
        while(begin != end){
            result += *begin;
            ++begin;
        }
        return result;
    }

};

#endif