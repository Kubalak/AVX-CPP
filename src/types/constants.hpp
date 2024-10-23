#pragma once
#ifndef __AVX_CONSTANTS_HPP
#define __AVX_CONSTANTS_HPP

#include <immintrin.h>

namespace avx {
    // Namespace containing constants shared across all types.
    namespace constants{
        static inline const __m256i ONES = _mm256_set1_epi8(0xFF);
        static inline const __m256i EPI8_CRATE_EPI16 = _mm256_set1_epi16(0xFF);
        static inline const __m256i EPI8_CRATE_EPI16_INVERSE = _mm256_set1_epi16(0xFF00);
        static inline const __m256i EPI8_CRATE_EPI32 = _mm256_set1_epi32(0xFF);
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_I = _mm256_set1_epi32(0xFF00);
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_II = _mm256_set1_epi32(0xFF0000);
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_III = _mm256_set1_epi32(0xFF000000);
        static inline const __m256i EPI16_CRATE_EPI32 = _mm256_set1_epi32(0xFFFF);
        static inline const __m256i EPI16_CRATE_EPI32_INVERSE = _mm256_set1_epi32(0xFFFF0000);
        static inline const __m256i EPI32_CRATE_EPI64 = _mm256_set1_epi64x(0xFFFFFFFF);
    }
}

#endif