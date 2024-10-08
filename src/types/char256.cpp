#include "char256.hpp"

namespace avx {
    const __m256i Char256::ones = _mm256_set1_epi8(0xFF);
    const __m256i Char256::epi16_crate = _mm256_set1_epi16(0xFF);
    const __m256i Char256::epi16_crate_shift_1 = _mm256_slli_si256(Char256::epi16_crate, 1);
    const __m256i Char256::epi32_crate = _mm256_set1_epi32(0xFF);
    const __m256i Char256::epi32_epi16_crate = _mm256_set1_epi32(0xFFFF);
    const __m256i Char256::epi32_epi16_crate_inverse = _mm256_set1_epi32(0xFFFF0000);

    std::ostream& operator<<(std::ostream& os, const Char256& a) {
        alignas(32) char tmp[33];
        tmp[32] = '\0';

        _mm256_store_si256((__m256i*)tmp, a.v);
        
        os << tmp;
        return os;
    }
}