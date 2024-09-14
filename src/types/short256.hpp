#pragma once
#ifndef SHORT256_HPP__
#define SHORT256_HPP__

#include <array>
#include <string>
#include <immintrin.h>

namespace avx {
    class Short256 {
        private:
            __m256i v;
            const static __m256i ones;
            const static __m256i crate;
            const static __m256i crate_inverse;
        
        public:
            static constexpr const int size = 32 / sizeof(short);
            Short256() noexcept : v(_mm256_setzero_si256()){}

            Short256(const Short256& init) noexcept : v(init.v){}

            Short256(const __m256i& init) noexcept : v(init){}
    
            Short256(const std::array<short, 16>& init) : v(_mm256_lddqu_si256((const __m256i*)init.data())){}

            explicit Short256(const short* addr) : v(_mm256_lddqu_si256((const __m256i*)addr)){}

            explicit Short256(const short b) noexcept : v(_mm256_set1_epi16(b)){}
            
            explicit Short256(const short &b) noexcept : v(_mm256_set1_epi16(b)){}

            bool operator==(const Short256 &bV) const {
                short* v1,* v2;
                v1 = (short*)&v;
                v2 = (short*)&bV.v;

                for(unsigned int i{0}; i < 16; ++i)
                    if(v1[i] != v2[i])
                        return false;

                return true;
            }

            bool operator==(const short &b) const {
                short* v1;
                v1 = (short*)&v;

                for(unsigned int i{0}; i < 16; ++i)
                    if(v1[i] != b)
                        return false;

                return true;
            }

            bool operator!=(const Short256 &bV) const {
                short* v1,* v2;
                v1 = (short*)&v;
                v2 = (short*)&bV.v;

                for(unsigned int i{0}; i < 16; ++i)
                    if(v1[i] != v2[i])
                        return true;

                return false;
            }

            bool operator!=(const short &b) const {
                short* v1;
                v1 = (short*)&v;

                for(unsigned int i{0}; i < 16; ++i)
                    if(v1[i] != b)
                        return true;

                return false;
            }
            
            Short256 operator+(const Short256& bV) const {
                return _mm256_add_epi16(v, bV.v);
            }

            Short256 operator+(const short& b) const {
                return _mm256_add_epi16(v, _mm256_set1_epi16(b));
            }

            Short256& operator+=(const Short256& bV) {
                v = _mm256_add_epi16(v, bV.v);
                return *this;
            }

            Short256& operator+=(const short& b) {
                v = _mm256_add_epi16(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator-(const Short256& bV) const {
                return _mm256_sub_epi16(v, bV.v);
            }

            Short256 operator-(const short& b) const {
                return _mm256_sub_epi16(v, _mm256_set1_epi16(b));
            }

            Short256& operator-=(const Short256& bV) {
                v = _mm256_sub_epi16(v, bV.v);
                return *this;
            }
            Short256& operator-=(const short& b) {
                v =_mm256_sub_epi16(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator*(const Short256& bV) const {
                return _mm256_mullo_epi16(v, bV.v);
            }

            Short256 operator*(const short& b) const {
                return _mm256_mullo_epi16(v,_mm256_set1_epi16(b));
            }

            Short256& operator*=(const Short256& bV) {
                v = _mm256_mullo_epi16(v, bV.v);
                return *this;
            }
            Short256& operator*=(const short& b) {
                v = _mm256_mullo_epi16(v,_mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator/(const Short256& bV) const {
                return v;
            }

            Short256 operator/(const short& b) const {
                return v;
            }

            Short256& operator/=(const Short256& bV) {
                return *this;
            }
            
            Short256& operator/=(const short& b) {
                return *this;
            }

            Short256 operator%(const Short256& bV) const {
                return v;
            }

            Short256 operator%(const short& b) {
                return v;
            }

            Short256& operator%=(const Short256& bV) {
                return *this;
            }

            Short256& operator%=(const short& b) {
                return *this;
            }

            Short256 operator|(const Short256& bV) const {
                return _mm256_or_si256(v, bV.v);
            }

            Short256 operator|(const short& b) const {
                return _mm256_or_si256(v, _mm256_set1_epi16(b));
            }

            Short256& operator|=(const Short256& bV) {
                v = _mm256_or_si256(v, bV.v);
                return *this;
            }

            Short256& operator|=(const short& b) {
                v = _mm256_or_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator&(const Short256& bV) const {
                return _mm256_and_si256(v, bV.v);
            }

            Short256 operator&(const short& b) const {
                return _mm256_and_si256(v, _mm256_set1_epi16(b));
            }

            Short256& operator&=(const Short256& bV) {
                v = _mm256_and_si256(v, bV.v);
                return *this;
            }

            Short256& operator&=(const short& b) {
                v = _mm256_and_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator^(const Short256& bV) const {
                return _mm256_xor_si256(v, bV.v);
            }

            Short256 operator^(const short& b) const {
                return _mm256_xor_si256(v, _mm256_set1_epi16(b));
            }

            Short256& operator^=(const Short256& bV) {
                v = _mm256_xor_si256(v, bV.v);
                return *this;
            }

            Short256& operator^=(const short& b) {
                v = _mm256_xor_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator<<(const Short256& bV) const {
                #if (defined __AVX512BW__ && defined __AVX512DQ__)
                    return _mm256_sllv_epi16(v,bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, crate);
                    __m256i bhalves = _mm256_and_si256(bV.v, crate);

                    __m256i first_res = _mm256_sllv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, crate_inverse);

                    halves = _mm256_and_si256(_mm256_srli_epi32(v, 16), crate_inverse);
                    bhalves = _mm256_and_si256(_mm256_srli_epi32(bV.v, 16), crate_inverse);

                    __m256i second_res = _mm256_sllv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, crate);
                    second_res = _mm256_slli_si256(second_res, 16);

                    return _mm256_or_si256(first_res, second_res);
                #endif
            }

            Short256 operator<<(const unsigned int& shift) const {
                return _mm256_slli_epi16(v, shift);
            }

            Short256& operator<<=(const Short256& bV) {
                #if (defined __AVX512BW__ && defined __AVX512DQ__)
                    v = _mm256_sllv_epi16(v,bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, crate);
                    __m256i bhalves = _mm256_and_si256(bV.v, crate);

                    __m256i first_res = _mm256_sllv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, crate_inverse);

                    halves = _mm256_and_si256(_mm256_srli_epi32(v, 16), crate_inverse);
                    bhalves = _mm256_and_si256(_mm256_srli_epi32(bV.v, 16), crate_inverse);

                    __m256i second_res = _mm256_sllv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, crate);
                    second_res = _mm256_slli_si256(second_res, 16);

                    v = _mm256_or_si256(first_res, second_res);
                #endif
                return *this;
            }

            Short256& operator<<=(const unsigned int& shift) {
                v = _mm256_slli_epi16(v, shift);
                return *this;
            }

            Short256 operator>>(const Short256& bV) {
                #if (defined __AVX512BW__ && defined __AVX512DQ__)
                    return _mm256_sllv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, crate);
                    __m256i bhalves = _mm256_and_si256(bV.v, crate);

                    __m256i first_res = _mm256_srlv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, crate_inverse);

                    halves = _mm256_and_si256(_mm256_srli_si256(v, 16), crate_inverse);
                    bhalves = _mm256_and_si256(_mm256_srli_si256(bV.v, 16), crate_inverse);

                    __m256i second_res = _mm256_srlv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, crate);
                    second_res = _mm256_slli_si256(second_res, 16);

                    return _mm256_or_si256(first_res, second_res);
                #endif
            }

            Short256 operator>>(const unsigned int& shift) const {
                return _mm256_srli_epi16(v, shift);
            }

            Short256& operator>>=(const Short256& bV) {
                #if (defined __AVX512BW__ && defined __AVX512DQ__)
                    v = _mm256_sllv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, crate);
                    __m256i bhalves = _mm256_and_si256(bV.v, crate);

                    __m256i first_res = _mm256_srlv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, crate_inverse);

                    halves = _mm256_and_si256(_mm256_srli_si256(v, 16), crate_inverse);
                    bhalves = _mm256_and_si256(_mm256_srli_si256(bV.v, 16), crate_inverse);

                    __m256i second_res = _mm256_srlv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, crate);
                    second_res = _mm256_slli_si256(second_res, 16);

                    v = _mm256_or_si256(first_res, second_res);
                #endif
                return *this;
            }

            Short256& operator>>=(const unsigned int& shift) {
                v = _mm256_srli_epi16(v, shift);
                return *this;
            }

            Short256 operator~() const noexcept{
                return _mm256_xor_si256(v, ones);
            }

            std::string str() const {
                std::string result = "Shortt256(";
                short* iv = (short*)&v; 
                for(unsigned i{0}; i < 15; ++i)
                    result += std::to_string(iv[i]) + ", ";
                
                result += std::to_string(iv[15]);
                result += ")";
                return result;
            }
    };
};

#endif