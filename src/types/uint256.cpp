#include "uint256.hpp"

namespace avx {
    const __m256i UInt256::ones = _mm256_set1_epi8(0xFF);
    const __m256i UInt256::crate = _mm256_set_epi32(0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF);


    
    UInt256 sum(std::vector<UInt256>& items){
        __m256i result = _mm256_setzero_si256();
        for(const UInt256& item : items)
            result = _mm256_add_epi32(result, item.v);
        
        return UInt256(result);
    }


    UInt256 sum(std::set<UInt256>& items){
        __m256i result = _mm256_setzero_si256();
        for(const UInt256& item : items)
            result = _mm256_add_epi32(result, item.v);
        
        return UInt256(result);
    }
}