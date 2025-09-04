#include <iostream>
#include "perf_utils.hpp"
#include <types/ushort256.hpp>

int64_t raw_avx_add(const std::vector<unsigned short>& aV, const std::vector<unsigned short> &bV, std::vector<unsigned short> &cV, bool print){   
    __START_TIME

    uint64_t pos = 0;
    unsigned short cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    __m256i d = _mm256_set1_epi16(cLit);
    
    while(pos + 16 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_add_epi16(a, b);
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_add_epi16(c, d));
        pos += 16;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) + *(bV.data() + pos);
        cV[pos] += cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
};

int64_t raw_avx_sub(const std::vector<unsigned short>& aV, const std::vector<unsigned short> &bV, std::vector<unsigned short> &cV, bool print){   
    __START_TIME

    uint64_t pos = 0;
    unsigned short cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    __m256i d = _mm256_set1_epi16(cLit);
    
    while(pos + 16 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_sub_epi16(a, b);
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_sub_epi16(c, d));
        pos += 16;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) - *(bV.data() + pos);
        cV[pos] -= cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
};

int main(int argc, char *argv[]) {
    std::vector<unsigned short> aV(536'870'912), bV(536'870'912), cV(536'870'912);
    testing::perf::TestConfig<unsigned short> config;
    config.warmupDuration = 20;

    config.avxFuncs.addRaw = raw_avx_add;
    config.avxFuncs.subRaw = raw_avx_sub;

    return testing::perf::allPerfTest<avx::UShort256>(aV, bV, cV, config);
}