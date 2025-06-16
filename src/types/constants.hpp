#pragma once
#ifndef __AVX_CONSTANTS_HPP
#define __AVX_CONSTANTS_HPP

#include <immintrin.h>

#ifdef NDEBUG
    #define N_THROW_REL noexcept
#else
    #define N_THROW_REL
#endif

#define __AVX_LOCALIZED_NULL_STR std::string(__FILE__) + std::string(":") + std::to_string(__LINE__) + std::string(" ") + std::string(__func__) +"() Passed argument is nullptr!"

namespace avx {
    /**
     *  Namespace containing constants shared across all types.
     */
    namespace constants{

        /**
        * Vector containing all 1's - 0xFF's
        */
        static inline const __m256i ONES = _mm256_set1_epi8(0xFF);
        /**
         *  Vector of chars set to 1 - 0x01
         */
        static inline const __m256i EPI8_ONE = _mm256_set1_epi8(1);
        /**
         *  Vector containing 0xFF aligned in the following order: 0x00FF'00FF...00FF
         */
        static inline const __m256i EPI8_CRATE_EPI16 = _mm256_set1_epi16(0xFF);
        /**
         *  Vector containing 0xFF aligned in the following order: 0xFF00'FF00...FF00
         */
        static inline const __m256i EPI8_CRATE_EPI16_INVERSE = _mm256_set1_epi16(0xFF00);
        /**
         *  Vector containing 0xFF aligned in the following order: 0x000000FF'000000FF...000000FF
         */
        static inline const __m256i EPI8_CRATE_EPI32 = _mm256_set1_epi32(0xFF);
        /**
         *  Vector containing 0xFF aligned in the following order: 0x0000FF00'0000FF00...0000FF00
         */
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_I = _mm256_set1_epi32(0xFF00);
        /**
         *  Vector containing 0xFF aligned in the following order: 0x00FF0000'00FF00000...00FF0000
         */
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_II = _mm256_set1_epi32(0xFF0000);
        /**
         *  Vector containing 0xFF aligned in the following order: 0xFF000000'FF00000000...FF000000
         */
        static inline const __m256i EPI8_CRATE_EPI32_SHIFT_III = _mm256_set1_epi32(0xFF000000);
        /**
         *  Vector of short values set to 1
         */
        static inline const __m256i EPI16_ONE = _mm256_set1_epi16(1);
        /**
         *  Vector containing 0xFFFF aligned in the following order: 0x0000FFFF'0000FFFF...0000FFFF
         */
        static inline const __m256i EPI16_CRATE_EPI32 = _mm256_set1_epi32(0xFFFF);
        /**
         *  Vector containing 0xFFFF aligned in the following order: 0xFFFF0000'FFFF0000...FFFF0000
         */
        static inline const __m256i EPI16_CRATE_EPI32_INVERSE = _mm256_set1_epi32(0xFFFF0000);
        /**
         *  Vector containing 0xFFFF'FFFF aligned in the following order: 0x00000000FFFFFFFF'00000000FFFFFFFF...00000000FFFFFFFF
         */
        static inline const __m256i EPI32_CRATE_EPI64 = _mm256_set1_epi64x(0xFFFFFFFF);
        /**
         *  Vector of int values with sign bit set to 1
         */
        static inline const __m256i EPI32_SIGN = _mm256_set1_epi32(0x8000'0000);
        /**
         *  Vector of int values set to 1
         */
        static inline const __m256i EPI32_ONE = _mm256_set1_epi32(1);
        /**
         *  Vector of long long values set to 1
         */
        static inline const __m256i EPI64_ONE = _mm256_set1_epi64x(1);
        /**
         *  Vector of long long values with sign bit set to 1
         */
        static inline const __m256i EPI64_SIGN = _mm256_set1_epi64x(0x8000'0000'0000'0000);
        /**
         *  Mask for double values with sign bit set to 0
         */
        static inline const __m256d DOUBLE_NO_SIGN = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF'FFFF'FFFF'FFFF));
        /**
         *  Mask for float values with sign bit set to 0
         */
        static inline const __m256 FLOAT_NO_SIGN = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF'FFFF));
        /**
         *  Vector with double values set to 1.0
         */
        static inline const __m256d DOUBLE_ONE = _mm256_set1_pd(1.0);
        /**
         *  Vector with float values set to 1
         */
        static inline const __m256 FLOAT_ONE = _mm256_set1_ps(1.0f);
        /**
         *  Vector with value limiting float reliable casting aka 2^24
         */
        static inline const __m256i FLOAT_LIMIT = _mm256_set1_epi32(0x0100'0000);

        static inline const __m256i EPI32_SLOWEST = _mm256_set1_epi32(0x8000'0001);

        /**
         *  2^24
         */
        static constexpr int INT2FLOAT_LIMIT = 0x0100'0000;
    }
}

#endif