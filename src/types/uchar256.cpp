#include "uchar256.hpp"


namespace avx {

    const __m256i UChar256::ones = _mm256_set1_epi8(0xFF);
    const __m256i UChar256::epi16_crate = _mm256_set1_epi16(0xFF);
    const __m256i UChar256::epi16_crate_shift_1 = _mm256_slli_si256(UChar256::epi16_crate, 1);
    const __m256i UChar256::epi32_crate = _mm256_set1_epi32(0xFF);

    std::ostream& operator<<(std::ostream& os, const UChar256& a) {
        alignas(32) unsigned char tmp[33];
        tmp[32] = '\0';

        _mm256_store_si256((__m256i*)tmp, a.v);
        
        os << tmp;
        return os;
    }

}