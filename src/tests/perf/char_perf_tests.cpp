#include <iostream>
#include "perf_utils.hpp"
#include <types/char256.hpp>

#define _mm256_sign_extend_epi8_epi16(var, half_one, half_two)\
    half_one = _mm256_and_si256(var, avx::constants::EPI8_CRATE_EPI16_INVERSE);\
    half_two = _mm256_and_si256(var, avx::constants::EPI8_CRATE_EPI16);\
    half_one = _mm256_srai_epi16(half_one, 8);\
    half_two = _mm256_slli_si256(half_two, 1);\
    half_two = _mm256_srai_epi16(half_two, 8);

#define _mm256_sign_extend_epi16_epi32(var, half_one, half_two)\
    half_one = _mm256_and_si256(var, avx::constants::EPI16_CRATE_EPI32_INVERSE);\
    half_two = _mm256_and_si256(var, avx::constants::EPI16_CRATE_EPI32);\
    half_one = _mm256_srai_epi32(half_one, 16);\
    half_two = _mm256_slli_si256(half_two, 2);\
    half_two = _mm256_srai_epi32(half_two, 16);

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

int64_t raw_avx_mul(const std::vector<char>& aV, const std::vector<char> &bV, std::vector<char> &cV, bool print){   
    __START_TIME

    uint64_t pos = 0;
    char cLit = bV[bV.size() / 2];
    __m256i a, b, c;
#if defined(__AVX512BW__)
    __m512i d = _mm512_set1_epi16(cLit);
#else
    __m256i d = _mm256_set1_epi8(cLit);
    __m256i d_16 = _mm256_set1_epi16(cLit);
#endif

    while(pos + 32 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
    #if defined(__AVX512BW__)
        c = _mm512_cvtepi16_epi8(
            _mm512_mullo_epi16(
                _mm512_mullo_epi16(
                    _mm512_cvtepi8_epi16(a), 
                    _mm512_cvtepi8_epi16(b)
                ),            
                d
            )
        );
    #else
        __m256i fhalf_a = _mm256_and_si256(a, avx::constants::EPI8_CRATE_EPI16);
        __m256i fhalf_b = _mm256_and_si256(b, avx::constants::EPI8_CRATE_EPI16);

        __m256i shalf_a = _mm256_and_si256(a, avx::constants::EPI8_CRATE_EPI16_INVERSE);
        __m256i shalf_b = _mm256_and_si256(b, avx::constants::EPI8_CRATE_EPI16_INVERSE);

        shalf_a = _mm256_srli_si256(shalf_a, 1);
        shalf_b = _mm256_srli_si256(shalf_b, 1);

        __m256i fresult = _mm256_mullo_epi16(fhalf_a, fhalf_b);
        fresult = _mm256_and_si256(fresult, avx::constants::EPI8_CRATE_EPI16);

        __m256i sresult = _mm256_mullo_epi16(shalf_a, shalf_b);
        sresult = _mm256_and_si256(sresult, avx::constants::EPI8_CRATE_EPI16);
        sresult = _mm256_slli_si256(sresult, 1);

        c = _mm256_or_si256(fresult, sresult);
        fhalf_a = _mm256_and_si256(c, avx::constants::EPI8_CRATE_EPI16);

        shalf_a = _mm256_and_si256(c, avx::constants::EPI8_CRATE_EPI16_INVERSE);

        shalf_a = _mm256_srli_si256(shalf_a, 1);

        fresult = _mm256_mullo_epi16(fhalf_a, d_16);
        fresult = _mm256_and_si256(fresult, avx::constants::EPI8_CRATE_EPI16);

        sresult = _mm256_mullo_epi16(shalf_a, d_16);
        sresult = _mm256_and_si256(sresult, avx::constants::EPI8_CRATE_EPI16);
        sresult = _mm256_slli_si256(sresult, 1);

        c = _mm256_or_si256(fresult, sresult);
        
    #endif
        
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), c);
        pos += 32;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) * *(bV.data() + pos);
        cV[pos] *= cLit;
        ++pos;
    }

    __FINALIZE_TEST
}


int64_t raw_avx_div(const std::vector<char>& aV, const std::vector<char> &bV, std::vector<char> &cV, bool print){   
    __START_TIME

    uint64_t pos = 0;
    char cLit = bV[bV.size() / 2];
    __m256i a, b, c;
#if defined(__AVX512BW__)
    __m512 d = _mm512_set1_ps(cLit);
#else
    __m256i d = _mm256_set1_epi8(cLit);
    __m256 d_ps = _mm256_set1_ps(cLit);
#endif

    while(pos + 32 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
    #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        __m512i first16 = _mm512_cvtepi8_epi16(a);
        __m512i second16 = _mm512_cvtepi8_epi16(b);
        
        __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
        __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

        __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second16)));
        __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second16, 1)));

        firstfp = _mm512_div_ps(_mm512_div_ps(firstfp, secondfp), d);
        firstfp_1 = _mm512_div_ps(_mm512_div_ps(firstfp_1, secondfp_1), d);

        __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

        c = _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
    #else
        __m256i v_fhalf_epi16, v_shalf_epi16;
        _mm256_sign_extend_epi8_epi16(a, v_fhalf_epi16, v_shalf_epi16);
         
        __m256i b_fhalf_epi16, b_shalf_epi16;
        _mm256_sign_extend_epi8_epi16(b, b_fhalf_epi16, b_shalf_epi16);

        __m256i v_first_half, v_second_half;
        
        _mm256_sign_extend_epi16_epi32(v_fhalf_epi16, v_first_half, v_second_half);
        __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
        __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

        __m256i bv_first_half, bv_second_half;
        _mm256_sign_extend_epi16_epi32(b_fhalf_epi16, bv_first_half, bv_second_half);

        __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
        __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

        __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), d_ps));
        __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), d_ps));
        
        fresult = _mm256_and_si256(fresult, avx::constants::EPI8_CRATE_EPI32);
        fresult = _mm256_slli_si256(fresult, 3);
        sresult = _mm256_and_si256(sresult, avx::constants::EPI8_CRATE_EPI32);
        sresult = _mm256_slli_si256(sresult, 1);

        __m256i half_res = _mm256_or_si256(fresult, sresult);

        _mm256_sign_extend_epi16_epi32(v_shalf_epi16, v_first_half, v_second_half);
        v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
        v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

        _mm256_sign_extend_epi16_epi32(b_shalf_epi16, bv_first_half, bv_second_half);
        bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
        bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

        fresult = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), d_ps));
        sresult = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), d_ps));
        
        fresult = _mm256_and_si256(fresult, avx::constants::EPI8_CRATE_EPI32);
        fresult = _mm256_slli_si256(fresult, 2);
        sresult = _mm256_and_si256(sresult, avx::constants::EPI8_CRATE_EPI32);

        __m256i shalf_res = _mm256_or_si256(fresult, sresult);

        c = _mm256_or_si256(half_res, shalf_res);
        
    #endif
        
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), c);
        pos += 32;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) / *(bV.data() + pos);
        cV[pos] /= cLit;
        ++pos;
    }

    __FINALIZE_TEST
}

int64_t raw_avx_mod(const std::vector<char>& aV, const std::vector<char> &bV, std::vector<char> &cV, bool print){   
    __START_TIME

    uint64_t pos = 0;
    char cLit = bV[bV.size() / 2];
    __m256i a, b, c;
#if defined(__AVX512BW__)
    __m512 d = _mm512_set1_ps(cLit);
#else
    __m256i d = _mm256_set1_epi8(cLit);
    __m256 d_ps = _mm256_set1_ps(cLit);
#endif

    while(pos + 32 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
    #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        __m512i first16 = _mm512_cvtepi8_epi16(a);
        __m512i second16 = _mm512_cvtepi8_epi16(b);
        
        __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
        __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

        __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second16)));
        __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second16, 1)));

        firstfp = _mm512_div_ps(firstfp, secondfp);
        firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

        __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
        result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

        result = _mm512_mullo_epi16(result, second16);
        result = _mm512_sub_epi16(first16, result);
        first16 = result;

        firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(result)));
        firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(result, 1)));

        firstfp = _mm512_div_ps(firstfp, d);
        firstfp_1 = _mm512_div_ps(firstfp_1, d);

        result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
        result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);
        
        c = _mm256_sub_epi8(_mm512_cvtepi16_epi8(first16), _mm512_cvtepi16_epi8(result));
    #else
        __m256i v_fhalf_epi16, v_shalf_epi16;
        _mm256_sign_extend_epi8_epi16(a, v_fhalf_epi16, v_shalf_epi16);
         
        __m256i b_fhalf_epi16, b_shalf_epi16;
        _mm256_sign_extend_epi8_epi16(b, b_fhalf_epi16, b_shalf_epi16);

        __m256i v_first_half, v_second_half;
        
        _mm256_sign_extend_epi16_epi32(v_fhalf_epi16, v_first_half, v_second_half);
        __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
        __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

        __m256i bv_first_half, bv_second_half;
        _mm256_sign_extend_epi16_epi32(b_fhalf_epi16, bv_first_half, bv_second_half);

        __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
        __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

        __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), d_ps));
        __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), d_ps));
        
        fresult = _mm256_and_si256(fresult, avx::constants::EPI8_CRATE_EPI32);
        fresult = _mm256_slli_si256(fresult, 3);
        sresult = _mm256_and_si256(sresult, avx::constants::EPI8_CRATE_EPI32);
        sresult = _mm256_slli_si256(sresult, 1);

        __m256i half_res = _mm256_or_si256(fresult, sresult);

        _mm256_sign_extend_epi16_epi32(v_shalf_epi16, v_first_half, v_second_half);
        v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
        v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

        _mm256_sign_extend_epi16_epi32(b_shalf_epi16, bv_first_half, bv_second_half);
        bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
        bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

        fresult = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), d_ps));
        sresult = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), d_ps));
        
        fresult = _mm256_and_si256(fresult, avx::constants::EPI8_CRATE_EPI32);
        fresult = _mm256_slli_si256(fresult, 2);
        sresult = _mm256_and_si256(sresult, avx::constants::EPI8_CRATE_EPI32);

        __m256i shalf_res = _mm256_or_si256(fresult, sresult);

        c = _mm256_or_si256(half_res, shalf_res);
        
    #endif
        
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), c);
        pos += 32;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) / *(bV.data() + pos);
        cV[pos] /= cLit;
        ++pos;
    }

    __FINALIZE_TEST
}

int main(int argc, char *argv[]) {
    std::vector<char> aV(1'073'741'824), bV(1'073'741'824), cV(1'073'741'824);
    testing::perf::TestConfig<char> config;
    config.avxFuncs.addRaw = raw_avx_add;
    config.avxFuncs.subRaw = raw_avx_sub;
    config.avxFuncs.mulRaw = raw_avx_mul;
    config.avxFuncs.divRaw = raw_avx_div;
    config.avxFuncs.modRaw = raw_avx_mod;
    //config.printTestFailed = true;
    config.printVerificationFailed = true;
    int res = testing::perf::allPerfTest<avx::Char256>(aV, bV, cV, config);

    //std::cout << static_cast<int>(aV[13031]) << ' ' << static_cast<int>(bV[13031]) << ' ' << static_cast<int>(aV[13031] / bV[13031]) << '\n';
    return res;
}