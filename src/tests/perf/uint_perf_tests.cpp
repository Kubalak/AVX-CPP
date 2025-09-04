#include <iostream>
#include "perf_utils.hpp"
#include <types/uint256.hpp>

int64_t raw_avx_add(const std::vector<unsigned int>& aV, const std::vector<unsigned int>& bV, std::vector<unsigned int>& cV, const bool print = true){
    auto start = std::chrono::steady_clock::now();

    uint64_t pos = 0;
    unsigned int cLit = bV[bV.size() / 2];
    __m256i c = _mm256_set1_epi32(cLit);
    while(pos + 8 < aV.size()){
        __m256i a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        __m256i b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_add_epi32(_mm256_add_epi32(a,b), c));
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = aV[pos] + bV[pos];
        cV[pos] += cLit;
        ++pos;
    }
    
    auto stop = std::chrono::steady_clock::now();
    if(print)
        testing::printTestDuration(__func__, start, stop);
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}

int64_t raw_avx_sub(const std::vector<unsigned int>& aV, const std::vector<unsigned int>& bV, std::vector<unsigned int>& cV, const bool print = true){
    auto start = std::chrono::steady_clock::now();

    uint64_t pos = 0;
    unsigned int cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    __m256i d = _mm256_set1_epi32(cLit);
    
    while(pos + 8 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_sub_epi32(a, b);
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_sub_epi32(c, d));
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) - *(bV.data() + pos);
        cV[pos] -= cLit;
        ++pos;
    }
    
    auto stop = std::chrono::steady_clock::now();
    if(print)
        testing::printTestDuration(__func__, start, stop);
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}



int main(int argc, char* argv[]) {
    std::vector<unsigned int> aV(268'435'456), bV(268'435'456), cV(268'435'456);
    testing::perf::TestConfig<unsigned int> config;
    config.avxFuncs.addRaw = raw_avx_add;
    config.avxFuncs.subRaw = raw_avx_sub;

    int result = testing::perf::allPerfTest<avx::UInt256>(aV, bV, cV, config);
    return (result & _AVX_IGNORE_LSH) != 0; // Ignore Lshift errors as SIMD behaves differently when crossing size of stored type.
}