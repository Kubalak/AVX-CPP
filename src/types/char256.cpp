#include "char256.hpp"

namespace avx {
    const __m256i Char256::ones = _mm256_set1_epi8(0xFF);
    const __m256i Char256::epi16_crate = _mm256_set1_epi16(0xFF);
    const __m256i Char256::epi16_crate_shift_1 = _mm256_slli_si256(Char256::epi16_crate, 1);
}