#include <iostream>
#include "perf_utils.hpp"
#include <types/int256.hpp>
#include <sleef.h>

int64_t raw_avx_add(const std::vector<int>& aV, const std::vector<int>& bV, std::vector<int>& cV, const bool print = true){
    __START_TIME

    uint64_t pos = 0;
    int cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    __m256i d = _mm256_set1_epi32(cLit);
    
    while(pos + 8 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_add_epi32(a, b);
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_add_epi32(c, d));
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) + *(bV.data() + pos);
        cV[pos] += cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
}

int64_t raw_avx_sub(const std::vector<int>& aV, const std::vector<int>& bV, std::vector<int>& cV, const bool print = true){
    __START_TIME

    uint64_t pos = 0;
    int cLit = bV[bV.size() / 2];
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
    
    __FINALIZE_TEST
}

int64_t raw_avx_mul(const std::vector<int>& aV, const std::vector<int>& bV, std::vector<int>& cV, const bool print = true){
    __START_TIME

    uint64_t pos = 0;
    int cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    __m256i d = _mm256_set1_epi32(cLit);
    
    while(pos + 8 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_mullo_epi32(a, b);
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_mullo_epi32(c, d));
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) * *(bV.data() + pos);
        cV[pos] *= cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
}


int64_t raw_avx_div(const std::vector<int>& aV, const std::vector<int>& bV, std::vector<int>& cV, const bool print = true){
    __START_TIME

    uint64_t pos = 0;
    int cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    #ifdef __AVX512F__
        __m512d d = _mm512_set1_pd(static_cast<double>(cLit));
    #else
        __m256i d = _mm256_set1_epi32(cLit);
    #endif
    
    while(pos + 8 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        #ifdef __AVX512F__
            c = _mm512_cvttpd_epi32(
                _mm512_div_pd(
                    _mm512_cvtepi32_pd(a), 
                    _mm512_cvtepi32_pd(b)
                )
            );
            _mm256_storeu_si256((__m256i*)(cV.data() + pos), 
                _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(c), 
                        d
                    )
                )
        );
        #else
            c = _mm256_div_epi32(a, b); 
            _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_div_epi32(c, d));
        #endif
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) / *(bV.data() + pos);
        cV[pos] /= cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
}

int64_t raw_avx_mod(const std::vector<int>& aV, const std::vector<int>& bV, std::vector<int>& cV, const bool print = true){
    __START_TIME

    uint64_t pos = 0;
    int cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    #ifdef __AVX512F__
        __m512d d = _mm512_set1_pd(static_cast<double>(cLit));
        __m256i dLit = _mm256_set1_epi32(cLit);
    #else
        __m256i d = _mm256_set1_epi32(cLit);
    #endif
    
    while(pos + 8 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        #ifdef __AVX512F__
            c = _mm256_sub_epi32(a, _mm256_mullo_epi32(b, _mm512_cvttpd_epi32(
                _mm512_div_pd(
                    _mm512_cvtepi32_pd(a), 
                    _mm512_cvtepi32_pd(b)
                )
            )));
            _mm256_storeu_si256((__m256i*)(cV.data() + pos), 
                _mm256_sub_epi32(c, 
                    _mm256_mullo_epi32(dLit, _mm512_cvttpd_epi32(
                        _mm512_div_pd(
                            _mm512_cvtepi32_pd(c), 
                            d
                            )
                        )
                    )
                )
            );
        #else
            c = _mm256_sub_epi32(a, _mm256_mullo_epi32(b, _mm256_div_epi32(a, b))); 
            _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_sub_epi32(c, _mm256_mullo_epi32(d, _mm256_div_epi32(c, d))));
        #endif
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) % *(bV.data() + pos);
        cV[pos] %= cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
}

int64_t raw_avx_lsh(const std::vector<int>& aV, const std::vector<int>& bV, std::vector<int>& cV, const bool print = true){
    __START_TIME

    uint64_t pos = 0;
    int cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    
    while(pos + 8 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_sllv_epi32(a, b); 
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_slli_epi32(c, cLit));
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) << *(bV.data() + pos);
        cV[pos] <<= cLit;
        ++pos;
    }
    
    __FINALIZE_TEST
}

int main(int argc, char* argv[]) {
    std::vector<int> aV(268'435'456), bV(268'435'456), cV(268'435'456);
    testing::perf::TestConfig<int> config;
    
    config.avxFuncs.addRaw = raw_avx_add;
    config.avxFuncs.subRaw = raw_avx_sub;
    config.avxFuncs.mulRaw = raw_avx_mul;
    config.avxFuncs.divRaw = raw_avx_div;
    config.avxFuncs.modRaw = raw_avx_mod;
    config.avxFuncs.lshRaw = raw_avx_lsh;

    config.doWarmup = true;
    config.warmupDuration = 20;
    config.printWarmupInfo = true;
    int result = testing::perf::allPerfTest<avx::Int256>(aV, bV, cV, config);
    return (result & _AVX_IGNORE_LSH) != 0; // Ignore Lshift errors as SIMD behaves differently when crossing size of stored type.
}
