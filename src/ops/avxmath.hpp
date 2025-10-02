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
#include <cmath>

#ifndef _MSC_VER
    #include <sleef.h>
#endif

namespace avx {

    /**
     * Calculates sine of the vector.
     * @param bV Vector which values will be used to calculate sine.
     * @return Vector containing sine of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 sin(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_sin_pd(bV.get());
        #else 
            return Sleef_sind4_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates cosine of the vector.
     * @param bV Vector which values will be used to calculate cosine.
     * @return Vector containing cosine of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 cos(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_cos_pd(bV.get());
        #else
            return Sleef_cosd4_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates tangent of the vector.
     * @param bV Vector which values will be used to calculate tangent.
     * @return Vector containing tangent of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 tan(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_tan_pd(bV.get());
        #else
            return Sleef_tand4_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates cotangent of the vector.
     * @param bV Vector which values will be used to calculate cotangent.
     * @return Vector containing cotangent of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 ctg(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_pd(
                constants::DOUBLE_ONE, 
                _mm256_tan_pd(bV.get())
            );
        #else
            return _mm256_div_pd(constants::DOUBLE_ONE, Sleef_tand4_u35avx2(bV.get()));
        #endif
    }

    /**
     * Calculates secant of the vector.
     * @param bV Vector which values will be used to calculate secant.
     * @return Vector containing secant of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 sec(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_pd(
                constants::DOUBLE_ONE, 
                _mm256_cos_pd(bV.get())
            ); 
        #else
            return _mm256_div_pd(constants::DOUBLE_ONE, Sleef_cosd4_u35avx2(bV.get()));
        #endif
    }

    /**
     * Calculates cosecant of the vector.
     * @param bV Vector which values will be used to calculate cosecant.
     * @return Vector containing cosecant of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 cosec(const Double256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_pd(
                constants::DOUBLE_ONE, 
                _mm256_sin_pd(bV.get())
            );
        #else
            return _mm256_div_pd(constants::DOUBLE_ONE, Sleef_sind4_u35avx2(bV.get()));
        #endif
    }

    /**
     * Calculates sine of the vector.
     * @param bV Vector which values will be used to calculate sine.
     * @return Vector containing sine of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 asin(const Double256 &bV) {
        #ifdef _MSC_VER
            return _mm256_asin_pd(bV.get());
        #else
            return Sleef_asind4_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates acos of the vector.
     * @param bV Vector which values will be used to calculate acos.
     * @return Vector containing acos of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 acos(const Double256 &bV) {
        #ifdef _MSC_VER
            return _mm256_acos_pd(bV.get());
        #else
            return Sleef_acosd4_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates atan of the vector.
     * @param bV Vector which values will be used to calculate atan.
     * @return Vector containing atan of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Double256 atan(const Double256 &bV) {
        #ifdef _MSC_VER
            return _mm256_atan_pd(bV.get());
        #else
            return Sleef_atand4_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates square root of the values inside vector.
     * @param bV Vector which values will be used to calculate square root.
     * @return New vector containing sqrt of each value inside `bV`.
     */
    Double256 sqrt(const Double256 &bV) {
        return _mm256_sqrt_pd(bV.get());
    }

    /**
     * Calculates sine of the vector.
     * @param bV Vector which values will be used to calculate sine.
     * @return Vector containing sine of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 sin(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_sin_ps(bV.get());
        #else
            return Sleef_sinf8_u10avx2(bV.get());
        #endif
    }

    /**
     * Calculates cosine of the vector.
     * @param bV Vector which values will be used to calculate cosine.
     * @return Vector containing cosine of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 cos(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_cos_ps(bV.get());
        #else
            return Sleef_cosf8_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates tangent of the vector.
     * @param bV Vector which values will be used to calculate tangent.
     * @return Vector containing tangent of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 tan(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_tan_ps(bV.get());
        #else
            return Sleef_tanf8_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates cotangent of the vector.
     * @param bV Vector which values will be used to calculate cotangent.
     * @return Vector containing cotangent of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 ctg(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_ps(
                constants::FLOAT_ONE, 
                _mm256_tan_ps(bV.get())
            );
        #else
            return _mm256_div_ps(constants::FLOAT_ONE, Sleef_tanf8_u35avx2(bV.get()));
        #endif
    }
    
    /**
     * Calculates secant of the vector.
     * @param bV Vector which values will be used to calculate secant.
     * @return Vector containing secant of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 sec(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_ps(
                constants::FLOAT_ONE, 
                _mm256_cos_ps(bV.get())
            );
        #else
            return _mm256_div_ps(constants::FLOAT_ONE, Sleef_cosf8_u35avx2(bV.get()));
        #endif
    }

    /**
     * Calculates cosecant of the vector.
     * @param bV Vector which values will be used to calculate cosecant.
     * @return Vector containing cosecant of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 cosec(const Float256& bV) {
        #ifdef _MSC_VER
            return _mm256_div_ps(
                constants::FLOAT_ONE, 
                _mm256_sin_ps(bV.get())
            );
        #else
            return _mm256_div_ps(constants::FLOAT_ONE, Sleef_sinf8_u35avx2(bV.get()));
        #endif
    }

    /**
     * Calculates asin of the vector.
     * @param bV Vector which values will be used to calculate cosecant.
     * @return Vector containing asin of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 asin(const Float256 &bV) {
        #ifdef _MSC_VER
            return _mm256_asin_ps(bV.get());
        #else
            return Sleef_asinf8_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates acos of the vector.
     * @param bV Vector which values will be used to calculate acos.
     * @return Vector containing acos of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 acos(const Float256 &bV) {
        #ifdef _MSC_VER
            return _mm256_acos_ps(bV.get());
        #else
            return Sleef_acosf8_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates atan of the vector.
     * @param bV Vector which values will be used to calculate atan.
     * @return Vector containing atan of each value in `bV`.
     * 
     * * Note: This function uses SVML library, which is available only on GCC and Clang <a href="https://github.com/shibatch/sleef">Sleef</a> library is used to perform computaion.
     */
    Float256 atan(const Float256 &bV) {
        #ifdef _MSC_VER
            return _mm256_atan_ps(bV.get());
        #else
            return Sleef_atanf8_u35avx2(bV.get());
        #endif
    }

    /**
     * Calculates square root of the values inside vector.
     * @param bV Vector which values will be used to calculate square root.
     * @return New vector containing sqrt of each value inside `bV`.
     */
    Float256 sqrt(const Float256 &bV) {
        return _mm256_sqrt_ps(bV.get());
    }

    /**
     * Calculates absolute value of the values inside vector.
     * @param bV Vector which values will be used to calculate absolute value.
     * @return New vector containing abs of each value inside `bV`.
     */
    Int256 abs(const Int256& bV) {
        return _mm256_abs_epi32(bV.get());
    }

    /**
     * Calculates absolute value of the values inside vector.
     * @param bV Vector which values will be used to calculate absolute value.
     * @return New vector containing abs of each value inside `bV`.
     */
    Short256 abs(const Short256& bV) {
        return _mm256_abs_epi16(bV.get());
    }

    /**
     * Calculates absolute value of the values inside vector.
     * @param bV Vector which values will be used to calculate absolute value.
     * @return New vector containing abs of each value inside `bV`.
     * 
     * Note: When avaliable function uses AVX512F and AVX512VL to calculate absolute value.
     */
    Long256 abs(const Long256& bV) {
        #if defined(__AVX512F__) && defined(__AVX512VL__)
            return _mm256_abs_epi64(bV.get());
        #else // If AVX512 not available, use abs algorithm.
            __m256i sign = _mm256_and_si256(bV.get(), constants::EPI64_SIGN); // Get mask for sub zero.
            sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), sign); // Expand mask to all bits - 0x8000'0000 will become 0xFFFF'FFFF and 0x0000'0000 will not change.
            __m256i abs = _mm256_xor_si256(bV.get(), sign); // Negate bits using XOR with mask (positive values will remain unaffected).
            return _mm256_add_epi64(abs, _mm256_and_si256(constants::EPI64_ONE, sign)); // Add 1 to the results using mask.
        #endif

    }

    /**
     * Calculates absolute value of the values inside vector.
     * @param bV Vector which values will be used to calculate absolute value.
     * @return New vector containing abs of each value inside `bV`.
     */
    Char256 abs(const Char256& bV) {
        return _mm256_abs_epi8(bV.get());
    }

    /**
     * Calculates absolute value of the values inside vector.
     * @param bV Vector which values will be used to calculate absolute value.
     * @return New vector containing abs of each value inside `bV`.
     */
    Double256 abs(const Double256& bV) {
        return _mm256_and_pd(bV.get(), constants::DOUBLE_NO_SIGN);
    }

    /**
     * Calculates absolute value of the values inside vector.
     * @param bV Vector which values will be used to calculate absolute value.
     * @return New vector containing abs of each value inside `bV`.
     */
    Float256 abs(const Float256& bV) {
        return _mm256_and_ps(bV.get(), constants::FLOAT_NO_SIGN);
    }

    /**
     * Accumulates values using AVX2 or AVX512 (if selected in CMake, no runtime detection).
     * @param data Vector containing data to be accumulated.
     * @param initVal Initial value to which all data will be added.
     * @return Accumulated value: `sum(data) + initVal`
     */
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

    /**
     * Accumulates values using AVX2 or AVX512 (if selected in CMake, no runtime detection).
     * @param data Vector containing data to be accumulated.
     * @param initVal Initial value to which all data will be added.
     * @return Accumulated value: `sum(data) + initVal`
     */
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

    /**
     * Accumulates values using AVX2 or AVX512 (if selected in CMake, no runtime detection).
     * @param data Vector containing data to be accumulated.
     * @param initVal Initial value to which all data will be added.
     * @return Accumulated value: `sum(data) + initVal`
     */
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

    /**
     * Accumulates values using AVX2 or AVX512 (if selected in CMake, no runtime detection).
     * @param data Vector containing data to be accumulated.
     * @param initVal Initial value to which all data will be added.
     * @return Accumulated value: `sum(data) + initVal`
     */
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

#endif