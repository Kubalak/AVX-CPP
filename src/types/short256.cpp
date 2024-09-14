#include "short256.hpp"

namespace avx {
    static_assert(sizeof(short) == 2, "You are compiling to 32-bit. Please switch to x64 to avoid undefined behaviour.");
    const __m256i Short256::ones = _mm256_set1_epi8(0xFF);
    const __m256i Short256::crate = _mm256_set_epi16(0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0);
    const __m256i Short256::crate_inverse = _mm256_set_epi16(0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF);

    
};