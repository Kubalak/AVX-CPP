#pragma once
#ifndef __AVX_CONTANTS_HPP
#define __AVX_CONTANTS_HPP

#include <immintrin.h>

namespace avx {
    // Namespace containing constants shared across all types.
    namespace constants{
        static const __m256i ONES = _mm256_set1_epi8(0xFF);
        static const __m256i EPI8_CRATE_EPI16 = _mm256_set1_epi16(0xFF);
        static const __m256i EPI8_CRATE_EPI16_INVERSE = _mm256_set1_epi16(0xFF00);
        static const __m256i EPI8_CRATE_EPI32 = _mm256_set1_epi32(0xFF);
        static const __m256i EPI8_CRATE_EPI32_SHIFT_I = _mm256_set1_epi32(0xFF00);
        static const __m256i EPI8_CRATE_EPI32_SHIFT_II = _mm256_set1_epi32(0xFF0000);
        static const __m256i EPI8_CRATE_EPI32_SHIFT_III = _mm256_set1_epi32(0xFF000000);
        static const __m256i EPI16_CRATE_EPI32 = _mm256_set1_epi32(0xFFFF);
        static const __m256i EPI16_CRATE_EPI32_INVERSE = _mm256_set1_epi32(0xFFFF0000);
    }
}

#endif