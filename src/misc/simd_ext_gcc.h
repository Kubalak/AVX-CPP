#pragma once
#ifndef __SIMD_EXT_GCC_H
#define __SIMD_EXT_GCC_H

#include <immintrin.h>

#ifndef _MSC_VER // If not MSVC provide function definitions for GCC/Clang

static const __m256d __DATA_0 = {2.0, 2.0, 2.0, 2.0};
static const __m256d __DATA_1 = {2.0000000000002274, 2.0000000000002274, 2.0000000000002274, 2.0000000000002274};
static const __m256i __DATA_2 = {0x00000000ffffffffLL, 0x00000000ffffffffLL, 0x00000000ffffffffLL, 0x00000000ffffffffLL};
static const __m256  __DATA_3 = {0.0f, 184.0f, 0.0f, 184.0f, 0.0f, 184.0f, 0.0f, 184.0f};
static const __m256i __DATA_4 = {0x4538000000000000LL, 0x4538000000000000LL, 0x4538000000000000LL, 0x4538000000000000LL}; 
static const __m256i __DATA_5 = {0x4338000000000000LL, 0x4338000000000000LL, 0x4338000000000000LL, 0x4338000000000000LL}; 
static const __m256i __DATA_6 = {0x3FEFFFFE00000000LL, 0x3FEFFFFE00000000LL, 0x3FEFFFFE00000000LL, 0x3FEFFFFE00000000LL}; 
static const __m256i __DATA_7 = {0x7FFFFFFFFFE00000LL, 0x7FFFFFFFFFE00000LL, 0x7FFFFFFFFFE00000LL, 0x7FFFFFFFFFE00000LL};
static const __m256i __DATA_8 = {0x7FFFFFFC00000000LL, 0x7FFFFFFC00000000LL, 0x7FFFFFFC00000000LL, 0x7FFFFFFC00000000LL};
static const __m256i __DATA_9 = {0x0003FFFFFFFFFFFFLL, 0x0003FFFFFFFFFFFFLL, 0x0003FFFFFFFFFFFFLL, 0x0003FFFFFFFFFFFFLL};
static const __m256i __DATA_10 = {0x0000000000000001LL, 0x0000000000000001LL, 0x0000000000000001LL, 0x0000000000000001LL};

#ifdef __cplusplus // Use for linking C lib inside C++
extern "C" { 
#endif
    /**
     * Reimplements missing division for two vectors containing `int` type.
     * Please note that this function is only intended to be used with GCC or Clang compilers.
     * MSVC provides implementation of this function by default.
     * 
     * @param a Vector with 8 `int` values to be divided.
     * @param b Vector containing 8 divisors.
     * @return Division results.
     */
    __m256i _mm256_div_epi32(__m256i a, __m256i b);

    /**
     * Reimplements missing division for two vectors containing `unsigned int` type.
     * Please note that this function is only intended to be used with GCC or Clang compilers.
     * MSVC provides implementation of this function by default.
     * 
     * @param a Vector with 8 `unsigned int` values to be divided.
     * @param b Vector containing 8 divisors.
     * @return Division results.
     */
    __m256i _mm256_div_epu32(__m256i a, __m256i b);

    /**
     * Reimplements missing for two vectors containing `long long` type.
     * Please note that this function is only intended to be used with GCC or Clang compilers.
     * MSVC provides implementation of this function by default.
     * @param a Vector with 4 `long long` values to be divided.
     * @param b Vector containing 4 divisors.
     * @return Division results.
     */
    __m256i _mm256_div_epi64(__m256i a, __m256i b);

    __m256 _mm256_sin_ps(__m256 a);

#ifdef __cplusplus // End of extern C block
}
#endif

#endif // _MSC_VER
#endif // __SIMD_EXT_GCC_H