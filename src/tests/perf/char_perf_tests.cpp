#include <iostream>
#include "perf_utils.hpp"
#include <types/char256.hpp>

int64_t raw_avx_add(const std::vector<char>& aV, const std::vector<char> &bV, std::vector<char> &cV, bool print){   
    __START_TIME

    uint64_t pos = 0;
    char cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    __m256i d = _mm256_set1_epi8(cLit);
    
    while(pos + 32 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_add_epi8(a, b);
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_add_epi8(c, d));
        pos += 32;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) + *(bV.data() + pos);
        cV[pos] += cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
};

int64_t raw_avx_sub(const std::vector<char>& aV, const std::vector<char> &bV, std::vector<char> &cV, bool print){   
    __START_TIME

    uint64_t pos = 0;
    char cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    __m256i d = _mm256_set1_epi8(cLit);
    
    while(pos + 32 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_sub_epi8(a, b);
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_sub_epi8(c, d));
        pos += 32;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) - *(bV.data() + pos);
        cV[pos] -= cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
};

int main(int argc, char *argv[]) {
    std::vector<char> aV(1'073'741'824), bV(1'073'741'824), cV(1'073'741'824);
    testing::perf::TestConfig<char> config;
    config.avxFuncs.addRaw = raw_avx_add;
    config.avxFuncs.subRaw = raw_avx_sub;
    return testing::perf::allPerfTest<avx::Char256>(aV, bV, cV, config);
}