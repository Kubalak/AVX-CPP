#pragma once
#ifndef __AVX_CONSTANTS_HPP
#define __AVX_CONSTANTS_HPP

#include <immintrin.h>

namespace avx {
    // Namespace containing constants shared across all types.
    namespace constants{

        static inline const __m256i ONES = _mm256_set1_epi8(0xFF);
        static inline const __m256i EPI8_ONE = _mm256_set1_epi8(1);
        static inline const __m256i EPI8_CRATE_EPI16 = _mm256_set1_epi16(0xFF);
        static inline const __m256i EPI8_CRATE_EPI16_INVERSE = _mm256_set1_epi16(0xFF00);
        static inline const __m256i EPI8_CRATE_EPI32 = _mm256_set1_epi32(0xFF);
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_I = _mm256_set1_epi32(0xFF00);
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_II = _mm256_set1_epi32(0xFF0000);
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_III = _mm256_set1_epi32(0xFF000000);
        static inline const __m256i EPI16_ONE = _mm256_set1_epi16(1);
        static inline const __m256i EPI16_CRATE_EPI32 = _mm256_set1_epi32(0xFFFF);
        static inline const __m256i EPI16_CRATE_EPI32_INVERSE = _mm256_set1_epi32(0xFFFF0000);
        static inline const __m256i EPI32_CRATE_EPI64 = _mm256_set1_epi64x(0xFFFFFFFF);
        static inline const __m256i EPI32_SIGN = _mm256_set1_epi32(0x8000'0000);
        static inline const __m256i EPI32_ONE = _mm256_set1_epi32(1);
        static inline const __m256i EPI64_ONE = _mm256_set1_epi64x(1);
        static inline const __m256i EPI64_SIGN = _mm256_set1_epi64x(0x8000'0000'0000'0000);
        static inline const __m256d DOUBLE_NO_SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF'FFFF'FFFF'FFFF));
        static inline const __m256 FLOAT_NO_SIGN = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF'FFFF));
        static inline const __m256d DOUBLE_ONE = _mm256_set1_pd(1.0);
        static inline const __m256 FLOAT_ONE = _mm256_set1_ps(1.0f);
    }
}

#endif