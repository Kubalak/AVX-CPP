#pragma once
#ifndef MATH_HPP_
#define MATH_HPP_

#include "types/long256.hpp"
#include "types/ulong256.hpp"
#include "types/int256.hpp"
#include "types/uint256.hpp"
#include "types/short256.hpp"
#include "types/ushort256.hpp"
#include "types/uint256.hpp"
#include "types/char256.hpp"
#include "types/uchar256.hpp"
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

    Int256 abs(const Int256& bV) {
        return _mm256_abs_epi32(bV.get());
    }

    Short256 abs(const Short256& bV) {
        return _mm256_abs_epi16(bV.get());
    }

    Long256 abs(const Long256& bV) {
        #if defined(__AVX512F__) && defined(__AVX512VL__)
            return _mm256_abs_epi64(bV.get());
        #else
            __m256i sign = _mm256_and_si256(bV.get(), constants::EPI64_SIGN);
            sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), sign);
            __m256i abs = _mm256_xor_si256(bV.get(), sign);
            return _mm256_add_epi64(abs, _mm256_and_si256(constants::EPI64_ONE, sign));
        #endif

    }

    Char256 abs(const Char256& bV) {
        return _mm256_abs_epi8(bV.get());
    }

    Double256 abs(const Double256& bV) {
        return _mm256_and_pd(bV.get(), constants::DOUBLE_NO_SIGN);
    }

    Float256 abs(const Float256& bV) {
        return _mm256_and_ps(bV.get(), constants::FLOAT_NO_SIGN);
    }

    int accumulate(const std::vector<int>& data, int initVal) {
        #ifdef __AVX512F__
            __m512i result = _mm512_setzero_si512();
            uint64_t i{0};
            for(; i + 16 < data.size(); i += 16)
                result = _mm512_add_epi32(result, _mm512_loadu_si512((const __m512i*)(data.data() + i)));
            
            for(uint8_t j{0}; j < 16; ++j)
                initVal += ((int*)&result)[j];
        #else
            __m256i result = _mm256_setzero_si256();
            uint64_t i{0};
            for(; i + 8 < data.size(); i += 8)
                result = _mm256_add_epi32(result, _mm256_lddqu_si256((const __m256i*)(data.data() + i)));
            
            for(uint8_t j{0}; j < 8; ++j)
                initVal += ((int*)&result)[j];
        #endif

        for(;i<data.size(); ++i)
            initVal += data[i];

        return initVal;
    }

    float accumulate(const std::vector<float>& data, float initVal) {
        #ifdef __AVX512F__
            __m512 result = _mm512_setzero_ps();
            uint64_t i{0};
            for(; i + 16 < data.size(); i += 16)
                result = _mm512_add_ps(result, _mm512_loadu_ps((data.data() + i)));
            
            for(uint8_t j{0}; j < 16; ++j)
                initVal += ((float*)&result)[j];
        #else
            __m256 result = _mm256_setzero_ps();
            uint64_t i{0};
            for(; i + 8 < data.size(); i += 8)
                result = _mm256_add_ps(result, _mm256_loadu_ps((data.data() + i)));
            
            for(uint8_t j{0}; j < 8; ++j)
                initVal += ((float*)&result)[j];
        #endif

        for(;i<data.size(); ++i)
            initVal += data[i];

        return initVal;
    }

    int64_t accumulate(const std::vector<int64_t>& data, int64_t initVal) {
        #ifdef __AVX512F__
            __m512i result = _mm512_setzero_si512();
            uint64_t i{0};
            for(; i + 8 < data.size(); i += 8)
                result = _mm512_add_epi64(result, _mm512_loadu_si512((const __m512i*)(data.data() + i)));
            
            for(uint8_t j{0}; j < 8; ++j)
                initVal += ((int64_t*)&result)[j];
        #else
            __m256i result = _mm256_setzero_si256();
            uint64_t i{0};
            for(; i + 4 < data.size(); i += 4)
                result = _mm256_add_epi64(result, _mm256_lddqu_si256((const __m256i*)(data.data() + i)));
            
            for(uint8_t j{0}; j < 4; ++j)
                initVal += ((int64_t*)&result)[j];
        #endif
        
        for(;i<data.size(); ++i)
            initVal += data[i];

        return initVal;
    }

    double accumulate(const std::vector<double>& data, double initVal) {
        #ifdef __AVX512F__
            __m512d result = _mm512_setzero_pd();
            uint64_t i{0};
            for(; i + 8 < data.size(); i += 8)
                result = _mm512_add_pd(result, _mm512_loadu_pd((data.data() + i)));
            
            for(uint8_t j{0}; j < 8; ++j)
                initVal += ((double*)&result)[j];
        #else
            __m256d result = _mm256_setzero_pd();
            uint64_t i{0};
            for(; i + 4 < data.size(); i += 4)
                result = _mm256_add_pd(result, _mm256_loadu_pd((data.data() + i)));
            
            for(uint8_t j{0}; j < 4; ++j)
                initVal += ((double*)&result)[j];
        #endif
        
        for(;i<data.size(); ++i)
            initVal += data[i];

        return initVal;
    }


};

/**
 * TODO: Implement inverse and hyperbolic trigonometric functions.

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

*/

#endif