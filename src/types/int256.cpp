#include "int256.hpp"
#include <string>
#include <stdexcept>

namespace avx {
    static_assert(sizeof(int) == 4, "You are compiling to 32-bit. Please switch to x64 to avoid undefined behaviour.");

    
    Int256 sum(std::vector<Int256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Int256& item : a)
            result = _mm256_add_epi32(result, item.v);
        
        return result;
    }


    Int256 sum(std::set<Int256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Int256& item : a)
            result = _mm256_add_epi32(result, item.v);
        
        return result;
    }
};