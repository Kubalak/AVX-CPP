#include "long256.hpp"

namespace avx {

    Long256 sum(std::vector<Long256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Long256& item : a)
            result = _mm256_add_epi64(result, item.v);
        
        return Long256(result);
    }


    Long256 sum(std::set<Long256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Long256& item : a)
            result = _mm256_add_epi64(result, item.v);
        
        return Long256(result);
    }
};