#pragma once
#ifndef SHORT256_HPP__
#define SHORT256_HPP__

#include <array>
#include <string>
#include <stdexcept>
#include <immintrin.h>

namespace avx {
    class Short256 {
        private:
            __m256i v;
            const static __m256i ones;
            const static __m256i crate;
            const static __m256i crate_inverse;
        
        public:
            static constexpr const int size = 16;
            Short256() noexcept : v(_mm256_setzero_si256()){}

            Short256(const Short256& init) noexcept : v(init.v){}

            Short256(const __m256i& init) noexcept : v(init){}
    
            Short256(const std::array<short, 16>& init) noexcept : v(_mm256_lddqu_si256((const __m256i*)init.data())){}

            explicit Short256(const short* addr) 
            #ifndef NDEBUG
                {
                    if(addr == nullptr)throw std::invalid_argument("Passed address is nullptr!");
                    v = _mm256_lddqu_si256((const __m256i*)addr);
                }
            #else
                : v(_mm256_lddqu_si256((const __m256i*)addr)){}
            #endif

            explicit Short256(const short b) noexcept : v(_mm256_set1_epi16(b)){}

            void save(short* dest) const {
                #ifndef NDEBUG
                    if(dest == nullptr) throw std::invalid_argument("Passed address is nullptr!");
                    _mm256_storeu_si256((__m256i*)dest, v);
                #else
                    _mm256_storeu_si256((__m256i*)dest, v);
                #endif
            }

            void save(std::array<short, 16>&dest) const {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            void saveAligned(short* dest) const {
                #ifndef NDEBUG
                    if(dest == nullptr) throw std::invalid_argument("Passed address is nullptr!");
                    _mm256_store_si256((__m256i*)dest, v);
                #else
                    _mm256_store_si256((__m256i*)dest, v);
                #endif
            }


            /**
             * Indexing operator.
             * @param index Position of desired element between 0 and 15.
             * @return Value of underlying element.
             * @throws `std::out_of_range` If index is not within the correct range.
             */
            const short operator[](const unsigned int& index) const {
                if(index > 15)
                    throw std::out_of_range("Range be within range 0-15! Got: " + std::to_string(index));
                return ((short*)&v)[index];
            }

            bool operator==(const Short256 &bV) const noexcept {
                short* v1,* v2;
                v1 = (short*)&v;
                v2 = (short*)&bV.v;

                for(unsigned int i{0}; i < 16; ++i)
                    if(v1[i] != v2[i])
                        return false;

                return true;
            }

            bool operator==(const short &b) const noexcept{
                short* v1;
                v1 = (short*)&v;

                for(unsigned int i{0}; i < 16; ++i)
                    if(v1[i] != b)
                        return false;

                return true;
            }

            bool operator!=(const Short256 &bV) const noexcept{
                short* v1,* v2;
                v1 = (short*)&v;
                v2 = (short*)&bV.v;

                for(unsigned int i{0}; i < 16; ++i)
                    if(v1[i] != v2[i])
                        return true;

                return false;
            }

            bool operator!=(const short &b) const noexcept{
                short* v1;
                v1 = (short*)&v;

                for(unsigned int i{0}; i < 16; ++i)
                    if(v1[i] != b)
                        return true;

                return false;
            }
            
            Short256 operator+(const Short256& bV) const noexcept{
                return _mm256_add_epi16(v, bV.v);
            }

            Short256 operator+(const short& b) const noexcept{
                return _mm256_add_epi16(v, _mm256_set1_epi16(b));
            }

            Short256& operator+=(const Short256& bV) noexcept {
                v = _mm256_add_epi16(v, bV.v);
                return *this;
            }

            Short256& operator+=(const short& b) noexcept {
                v = _mm256_add_epi16(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator-(const Short256& bV) const noexcept {
                return _mm256_sub_epi16(v, bV.v);
            }

            Short256 operator-(const short& b) const noexcept {
                return _mm256_sub_epi16(v, _mm256_set1_epi16(b));
            }

            Short256& operator-=(const Short256& bV) noexcept {
                v = _mm256_sub_epi16(v, bV.v);
                return *this;
            }

            Short256& operator-=(const short& b) noexcept {
                v =_mm256_sub_epi16(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator*(const Short256& bV) const noexcept {
                return _mm256_mullo_epi16(v, bV.v);
            }

            Short256 operator*(const short& b) const noexcept {
                return _mm256_mullo_epi16(v,_mm256_set1_epi16(b));
            }

            Short256& operator*=(const Short256& bV) noexcept {
                v = _mm256_mullo_epi16(v, bV.v);
                return *this;
            }

            Short256& operator*=(const short& b) noexcept {
                v = _mm256_mullo_epi16(v,_mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator/(const Short256& bV) const noexcept {
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m128i bv_first_half = _mm256_extracti128_si256(bV.v, 0);
                __m128i bv_second_half = _mm256_extracti128_si256(bV.v, 1);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_first_half));
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_second_half));

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                return _mm256_insert_epi64(combinedres, b1, 2);
            }

            Short256 operator/(const short& b) const noexcept {
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);

                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m256 bV = _mm256_set1_ps(b);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                return _mm256_insert_epi64(combinedres, b1, 2);
            }

            Short256& operator/=(const Short256& bV) noexcept {
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m128i bv_first_half = _mm256_extracti128_si256(bV.v, 0);
                __m128i bv_second_half = _mm256_extracti128_si256(bV.v, 1);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_first_half));
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_second_half));

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                v = _mm256_insert_epi64(combinedres, b1, 2);
                return *this;
            }
            
            Short256& operator/=(const short& b) noexcept {
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);

                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m256 bV = _mm256_set1_ps(b);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                v = _mm256_insert_epi64(combinedres, b1, 2);
                return *this;
            }

            Short256 operator%(const Short256& bV) const noexcept {
                return v;
            }

            Short256 operator%(const short& b) noexcept {
                return v;
            }

            Short256& operator%=(const Short256& bV) noexcept {
                return *this;
            }

            Short256& operator%=(const short& b) noexcept {
                return *this;
            }

            Short256 operator|(const Short256& bV) const noexcept {
                return _mm256_or_si256(v, bV.v);
            }

            Short256 operator|(const short& b) const noexcept {
                return _mm256_or_si256(v, _mm256_set1_epi16(b));
            }

            Short256& operator|=(const Short256& bV) noexcept {
                v = _mm256_or_si256(v, bV.v);
                return *this;
            }

            Short256& operator|=(const short& b) noexcept {
                v = _mm256_or_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator&(const Short256& bV) const noexcept {
                return _mm256_and_si256(v, bV.v);
            }

            Short256 operator&(const short& b) const noexcept {
                return _mm256_and_si256(v, _mm256_set1_epi16(b));
            }

            Short256& operator&=(const Short256& bV) noexcept {
                v = _mm256_and_si256(v, bV.v);
                return *this;
            }

            Short256& operator&=(const short& b) noexcept {
                v = _mm256_and_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            Short256 operator^(const Short256& bV) const noexcept{
                return _mm256_xor_si256(v, bV.v);
            }

            Short256 operator^(const short& b) const noexcept{
                return _mm256_xor_si256(v, _mm256_set1_epi16(b));
            }

            Short256& operator^=(const Short256& bV) noexcept{
                v = _mm256_xor_si256(v, bV.v);
                return *this;
            }

            Short256& operator^=(const short& b) noexcept{
                v = _mm256_xor_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            /**
             * Performs left bitwise shift of corresponding values.
             * @param bV Second vector that specifies number of bits to shift (for each 16-bit value).
             * @return New value of `v` shifted by number of bits specfied in `bV`.
             */
            Short256 operator<<(const Short256& bV) const noexcept {
                #if (defined __AVX512BW__ && defined __AVX512VL__)
                    // If compiler is using AVX-512BW and AVX-512DQ use available function.
                    return _mm256_sllv_epi16(v, bV.v);
                #else
                    // Perform bitwise left on first half of elements.
                    __m256i halves = _mm256_and_si256(v, crate_inverse);
                    __m256i bhalves = _mm256_and_si256(bV.v, crate_inverse);

                    __m256i first_res = _mm256_sllv_epi32(halves, bhalves);
                    // AND is used to get rid of unwanted bits that may happen as 32-bit mode is used.
                    first_res = _mm256_and_si256(first_res, crate_inverse);

                    // Performs AND operation on the rest of values and shifts to right place.
                    halves = _mm256_srli_si256(_mm256_and_si256(v, crate), 2);
                    bhalves = _mm256_srli_si256(_mm256_and_si256(bV.v, crate), 2);

                    // Performs the same operation as in first scenario
                    __m256i second_res = _mm256_sllv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, crate_inverse);
                    second_res = _mm256_slli_si256(second_res, 2);

                    // Joins results together.
                    return _mm256_or_si256(first_res, second_res);
                #endif
            }

            Short256 operator<<(const unsigned int& shift) const noexcept {
                return _mm256_slli_epi16(v, shift);
            }

            Short256& operator<<=(const Short256& bV) noexcept {
                #if (defined __AVX512BW__ && defined __AVX512VL__)
                    v = _mm256_sllv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, crate_inverse);
                    __m256i bhalves = _mm256_and_si256(bV.v, crate_inverse);

                    __m256i first_res = _mm256_sllv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, crate_inverse);

                    halves = _mm256_srli_si256(_mm256_and_si256(v, crate), 2);
                    bhalves = _mm256_srli_si256(_mm256_and_si256(bV.v, crate), 2);

                    __m256i second_res = _mm256_sllv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, crate_inverse);
                    second_res = _mm256_slli_si256(second_res, 2);

                    v = _mm256_or_si256(first_res, second_res);
                #endif
                return *this;
            }

            Short256& operator<<=(const unsigned int& shift) noexcept {
                v = _mm256_slli_epi16(v, shift);
                return *this;
            }

            Short256 operator>>(const Short256& bV) const noexcept {
                #if (defined __AVX512BW__ && defined __AVX512VL__)
                    return _mm256_srlv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, crate_inverse);
                    __m256i bhalves = _mm256_and_si256(bV.v, crate_inverse);

                    __m256i first_res = _mm256_srlv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, crate_inverse);

                    halves = _mm256_srli_si256(_mm256_and_si256(v, crate), 2);
                    bhalves = _mm256_srli_si256(_mm256_and_si256(bV.v, crate), 2);

                    __m256i second_res = _mm256_srlv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, crate_inverse);
                    second_res = _mm256_slli_si256(second_res, 2);

                    return _mm256_or_si256(first_res, second_res);
                #endif
            }

            Short256 operator>>(const unsigned int& shift) const noexcept{
                return _mm256_srli_epi16(v, shift);
            }

            Short256& operator>>=(const Short256& bV) noexcept {
                #if (defined __AVX512BW__ && defined __AVX512VL__)
                    v = _mm256_srlv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, crate_inverse);
                    __m256i bhalves = _mm256_and_si256(bV.v, crate_inverse);

                    __m256i first_res = _mm256_srlv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, crate_inverse);

                    halves = _mm256_srli_si256(_mm256_and_si256(v, crate), 2);
                    bhalves = _mm256_srli_si256(_mm256_and_si256(bV.v, crate), 2);

                    __m256i second_res = _mm256_srlv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, crate_inverse);
                    second_res = _mm256_slli_si256(second_res, 2);

                    v = _mm256_or_si256(first_res, second_res);
                #endif
                return *this;
            }

            Short256& operator>>=(const unsigned int& shift) noexcept {
                v = _mm256_srli_epi16(v, shift);
                return *this;
            }

            Short256 operator~() const noexcept{
                return _mm256_xor_si256(v, ones);
            }

            std::string str() const noexcept {
                std::string result = "Short256(";
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