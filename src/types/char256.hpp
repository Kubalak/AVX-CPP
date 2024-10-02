#pragma once
#ifndef CHAR256_HPP__
#define CHAR256_HPP__

#include <array>
#include <cstring>
#include <string>
#include <stdexcept>
#include <immintrin.h>
namespace avx {
    class Char256 {
        private:

            __m256i v;

            static const __m256i ones;
            static const __m256i epi16_crate;
            static const __m256i epi16_crate_shift_1;
            static const __m256i epi32_crate;

        public:
            
            static constexpr int size = 32;

            Char256() noexcept : v(_mm256_setzero_si256()){}

            Char256(const char init) noexcept : v(_mm256_set1_epi8(init)){}

            Char256(const __m256i& init) noexcept : v(init){}

            explicit Char256(const char* addr)
            #ifndef NDEBUG
                {
                    if(addr == nullptr)throw std::invalid_argument("Passed address is nullptr!");
                    v = _mm256_lddqu_si256((const __m256i*)addr);
                }
            #else
                : v(_mm256_lddqu_si256((const __m256i*)addr)){}
            #endif

            Char256(const std::string& init) noexcept {
                if(init.size() >= 32)
                    v = _mm256_lddqu_si256((const __m256i*)init.data());
                else {
                    alignas(32) char initV[32];
                    printf("0x%p\n", initV);
                    memset(initV, 0, 32);
                    #ifdef _MSC_VER
                        strncpy_s(initV, 32, init.data(), init.size());
                    #elif defined( __GNUC__)
                        strncpy(initV, init.data(), init.size());
                    #endif
                    v = _mm256_load_si256((const __m256i*) initV);
                }
            }


            Char256(const std::array<char, 32>& init) noexcept : v(_mm256_lddqu_si256((const __m256i*)init.data())){}

            /**
             * Saves data to destination in memory.
             * @param dest A valid pointer to a memory of at least 32 bytes (`char`).
             * @throws If in debug mode and `dest` is `nullptr` throws `std::invalid_argument`. Otherwise no exception will be thrown. 
             */
            void save(std::array<char, 32>& dest) const {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            void save(char* dest) const {
                #ifndef NDEBUG
                    if(dest == nullptr) throw std::invalid_argument("Passed address is nullptr!");
                    _mm256_storeu_si256((__m256i*)dest, v);
                #else
                    _mm256_storeu_si256((__m256i*)dest, v);
                #endif
            }

            /**
             * Saves data to destination in memory. The memory must be aligned at 32-byte boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param dest A valid pointer to a memory of at least 32 bytes (`char`).
             * @throws If in debug mode and `dest` is `nullptr` throws `std::invalid_argument`. Otherwise no exception will be thrown. 
             */
            void saveAligned(char* dest) const {
                #ifndef NDEBUG
                    if(dest == nullptr) throw std::invalid_argument("Passed address is nullptr!");
                    _mm256_store_si256((__m256i*)dest, v);
                #else
                    _mm256_store_si256((__m256i*)dest, v);
                #endif
            }

            /**
             * Get the internal vector value.
             * @returns The value of `__m256i` vector.
             */
            __m256i get() const noexcept{return v;}

            /**
             * Set the internal vector value.
             * @param value New value to be set.
             */
            void set(const __m256i value) noexcept {v = value;}

            /**
             * Indexing operator.
             * @param index Position of desired element between 0 and 31.
             * @return Value of underlying element.
             * @throws `std::out_of_range` If index is not within the correct range.
             */
            char operator[](const unsigned int& index) const {
                if(index > 31)
                    throw std::out_of_range("Range be within range 0-31! Got: " + std::to_string(index));
                return ((char*)&v)[index];
            }

            bool operator==(const Char256& bV) const noexcept {
                __m256i eq = _mm256_cmpeq_epi8(v, bV.v);
                unsigned long long* eqV = (unsigned long long*)&eq;
                for(uint8_t i = 0; i < 4; ++i)
                    if(eqV[i] != UINT64_MAX)
                        return false;
                return true;
            }

            bool operator==(const char b) const noexcept {
                char* v1,* v2;
                v1 = (char*)&v;

                for(unsigned int i{0}; i < 32; ++i)
                    if(v1[i] != b)
                        return false;

                return true;
            }

            bool operator!=(const Char256& bV) const noexcept {
                __m256i eq = _mm256_cmpeq_epi8(v, bV.v);
                unsigned long long* eqV = (unsigned long long*)&eq;
                for(uint8_t i = 0; i < 4; ++i)
                    if(eqV[i] != UINT64_MAX)
                        return true;
                return false;
            }

            bool operator!=(const char b) const noexcept {
                char* v1,* v2;
                v1 = (char*)&v;

                for(unsigned int i{0}; i < 32; ++i)
                    if(v1[i] != b)
                        return true;

                return false;
            }

            Char256 operator+(const Char256& bV) {
                return _mm256_add_epi8(v, bV.v);
            }


            Char256 operator+(const char& b) {
                return _mm256_add_epi8(v, _mm256_set1_epi8(b));
            }


            Char256& operator+=(const Char256& bV) {
                v = _mm256_add_epi8(v, bV.v);
                return *this;
            }


            Char256& operator+=(const char& b) {
                v =_mm256_add_epi8(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator-(const Char256& bV) {
                return _mm256_sub_epi8(v, bV.v);
            }


            Char256 operator-(const char& b) {
                return _mm256_sub_epi8(v, _mm256_set1_epi8(b));
            }


            Char256& operator-=(const Char256& bV) {
                v = _mm256_sub_epi8(v, bV.v);
                return *this;
            }


            Char256& operator-=(const char& b) {
                v = _mm256_sub_epi8(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator*(const Char256& bV) {
                __m256i fhalf_a = _mm256_and_si256(v, epi16_crate);
                __m256i fhalf_b = _mm256_and_si256(bV.v, epi16_crate);

                __m256i shalf_a = _mm256_and_si256(v, epi16_crate_shift_1);
                __m256i shalf_b = _mm256_and_si256(bV.v, epi16_crate_shift_1);

                shalf_a = _mm256_srli_si256(shalf_a, 1);
                shalf_b = _mm256_srli_si256(shalf_b, 1);

                __m256i fresult = _mm256_mullo_epi16(fhalf_a, fhalf_b);
                fresult = _mm256_and_si256(fresult, epi16_crate);

                __m256i sresult = _mm256_mullo_epi16(shalf_a, shalf_b);
                sresult = _mm256_and_si256(sresult, epi16_crate);
                sresult = _mm256_slli_si256(sresult, 1);

                return _mm256_or_si256(fresult, sresult);
            }


            Char256 operator*(const char& b) {
                __m256i fhalf = _mm256_and_si256(v, epi16_crate);
                __m256i bV = _mm256_set1_epi16(b); 

                __m256i shalf = _mm256_and_si256(v, epi16_crate_shift_1);

                shalf = _mm256_srli_si256(shalf, 1);

                __m256i fresult = _mm256_mullo_epi16(fhalf, bV);
                fresult = _mm256_and_si256(fresult, epi16_crate);

                __m256i sresult = _mm256_mullo_epi16(shalf, bV);
                sresult = _mm256_and_si256(sresult, epi16_crate);
                sresult = _mm256_slli_si256(sresult, 1);

                return _mm256_or_si256(fresult, sresult);
            }


            Char256& operator*=(const Char256& bV) {
                __m256i fhalf_a = _mm256_and_si256(v, epi16_crate);
                __m256i fhalf_b = _mm256_and_si256(bV.v, epi16_crate);

                __m256i shalf_a = _mm256_and_si256(v, epi16_crate_shift_1);
                __m256i shalf_b = _mm256_and_si256(bV.v, epi16_crate_shift_1);

                shalf_a = _mm256_srli_si256(shalf_a, 1);
                shalf_b = _mm256_srli_si256(shalf_b, 1);

                __m256i fresult = _mm256_mullo_epi16(fhalf_a, fhalf_b);
                fresult = _mm256_and_si256(fresult, epi16_crate);

                __m256i sresult = _mm256_mullo_epi16(shalf_a, shalf_b);
                sresult = _mm256_and_si256(sresult, epi16_crate);
                sresult = _mm256_slli_si256(sresult, 1);

                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }


            Char256& operator*=(const char& b) {
                __m256i fhalf = _mm256_and_si256(v, epi16_crate);
                __m256i bV = _mm256_set1_epi16(b); 

                __m256i shalf = _mm256_and_si256(v, epi16_crate_shift_1);

                shalf = _mm256_srli_si256(shalf, 1);

                __m256i fresult = _mm256_mullo_epi16(fhalf, bV);
                fresult = _mm256_and_si256(fresult, epi16_crate);

                __m256i sresult = _mm256_mullo_epi16(shalf, bV);
                sresult = _mm256_and_si256(sresult, epi16_crate);
                sresult = _mm256_slli_si256(sresult, 1);

                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }


            Char256 operator/(const Char256& bV) {
                alignas(32) char vP[32];
                alignas(32) char bP[32];

                _mm256_store_si256((__m256i*)vP, v);
                _mm256_store_si256((__m256i*)bP, bV.v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = bP[i] ? vP[i] / bP[i] : 0;

                return _mm256_load_si256((const __m256i*)vP); 
            }


            Char256 operator/(const char& b) {
                alignas(32) char vP[32];

                _mm256_store_si256((__m256i*)vP, v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = b ? vP[i] / b : 0;

                return _mm256_load_si256((const __m256i*)vP);
            }


            Char256& operator/=(const Char256& bV) {
                alignas(32) char vP[32];
                alignas(32) char bP[32];

                _mm256_store_si256((__m256i*)vP, v);
                _mm256_store_si256((__m256i*)bP, bV.v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = bP[i] ? vP[i] / bP[i] : 0;

                v = _mm256_load_si256((const __m256i*)vP); 
                return *this;
            }


            Char256& operator/=(const char& b) {
                alignas(32) char vP[32];

                _mm256_store_si256((__m256i*)vP, v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = b ? vP[i] / b : 0;

                v = _mm256_load_si256((const __m256i*)vP);
                return *this;
            }


            Char256 operator%(const Char256& bV) {
                alignas(32) char vP[32];
                alignas(32) char bP[32];

                _mm256_store_si256((__m256i*)vP, v);
                _mm256_store_si256((__m256i*)bP, bV.v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = bP[i] ? vP[i] % bP[i] : 0;

                return _mm256_load_si256((const __m256i*)vP); 
            }


            Char256 operator%(const char& b) {
                alignas(32) char vP[32];

                _mm256_store_si256((__m256i*)vP, v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = b ? vP[i] % b : 0;

                return _mm256_load_si256((const __m256i*)vP);
            }


            Char256& operator%=(const Char256& bV) {
                alignas(32) char vP[32];
                alignas(32) char bP[32];

                _mm256_store_si256((__m256i*)vP, v);
                _mm256_store_si256((__m256i*)bP, bV.v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = bP[i] ? vP[i] % bP[i] : 0;

                v = _mm256_load_si256((const __m256i*)vP); 
                return *this;
            }


            Char256& operator%=(const char& b) {
                alignas(32) char vP[32];

                _mm256_store_si256((__m256i*)vP, v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = b ? vP[i] % b : 0;

                v = _mm256_load_si256((const __m256i*)vP);
                return *this;
            }


            Char256 operator&(const Char256& bV) {
                return _mm256_and_si256(v, bV.v);
            }


            Char256 operator&(const char& b) {
                return _mm256_and_si256(v, _mm256_set1_epi8(b));
            }


            Char256& operator&=(const Char256& bV) {
                v = _mm256_and_si256(v, bV.v);
                return *this;
            }


            Char256& operator&=(const char& b) {
                v = _mm256_and_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator|(const Char256& bV) {
                return _mm256_or_si256(v, bV.v);
            }


            Char256 operator|(const char& b) {
                return _mm256_or_si256(v, _mm256_set1_epi8(b));
            }


            Char256& operator|=(const Char256& bV) {
                v = _mm256_or_si256(v, bV.v);
                return *this;
            }


            Char256& operator|=(const char& b) {
                v = _mm256_or_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator^(const Char256& bV) {
                return _mm256_xor_si256(v, bV.v);
            }


            Char256 operator^(const char& b) {
                return _mm256_xor_si256(v, _mm256_set1_epi8(b));
            }


            Char256& operator^=(const Char256& bV) {
                v = _mm256_xor_si256(v, bV.v);
                return *this;
            }


            Char256& operator^=(const char& b) {
                v = _mm256_xor_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator<<(const Char256& bV) {
                #ifdef __AVX512VL__
                    __m128i a_lo = _mm256_castsi256_si128(v);           // Pierwsze 128 bitów
                    __m128i a_hi = _mm256_extracti128_si256(v, 1);      // Drugie 128 bitów

                    __m128i b_lo = _mm256_castsi256_si128(bV.v);           // Pierwsze 128 bitów (shift values)
                    __m128i b_hi = _mm256_extracti128_si256(bV.v, 1);      // Drugie 128 bitów (shift values)

                    // Operacja przesunięcia bitowego dla dolnych 128 bitów
                    __m128i result_lo = _mm_sllv_epi16(a_lo, b_lo);     // SSE2/SSE3: zmienne przesunięcia

                    // Operacja przesunięcia bitowego dla górnych 128 bitów
                    __m128i result_hi = _mm_sllv_epi16(a_hi, b_hi);     // SSE2/SSE3: zmienne przesunięcia

                    // Łączymy wynik z powrotem w jeden wektor 256-bitowy
                    return _mm256_set_m128i(result_hi, result_lo);
                #else
                    __m256i q1_a = _mm256_and_si256(v, epi32_crate);
                    __m256i q1_b = _mm256_and_si256(bV.v, epi32_crate);

                    __m256i q2_a = _mm256_and_si256(_mm256_srli_si256(v, 1), epi32_crate);
                    __m256i q2_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), epi32_crate);

                    __m256i q3_a = _mm256_and_si256(_mm256_srli_si256(v, 2), epi32_crate);
                    __m256i q3_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), epi32_crate);

                    __m256i q4_a = _mm256_and_si256(_mm256_srli_si256(v, 3), epi32_crate);
                    __m256i q4_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), epi32_crate);

                    __m256i q1_res = _mm256_sllv_epi32(q1_a, q1_b);
                    __m256i q2_res = _mm256_sllv_epi32(q2_a, q2_b);
                    __m256i q3_res = _mm256_sllv_epi32(q3_a, q3_b);
                    __m256i q4_res = _mm256_sllv_epi32(q4_a, q4_b);

                    q1_res = _mm256_and_si256(q1_res, epi32_crate);
                    q2_res = _mm256_and_si256(q2_res, epi32_crate);
                    q3_res = _mm256_and_si256(q3_res, epi32_crate);
                    q4_res = _mm256_and_si256(q4_res, epi32_crate);

                    q2_res = _mm256_slli_si256(q2_res, 1);
                    q3_res = _mm256_slli_si256(q3_res, 2);
                    q4_res = _mm256_slli_si256(q4_res, 3);
                    
                    q1_res = _mm256_or_si256(q1_res, q2_res);
                    q2_res = _mm256_or_si256(q3_res, q4_res);
                    return _mm256_or_si256(q1_res, q2_res);
                #endif
            }


            Char256 operator<<(const unsigned int& b) {
                __m256i fhalf = _mm256_and_si256(v, epi16_crate);
                __m256i shalf = _mm256_and_si256(v, epi16_crate_shift_1);
                fhalf = _mm256_slli_epi16(fhalf, b);
                shalf = _mm256_slli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, epi16_crate);
                shalf = _mm256_and_si256(shalf, epi16_crate_shift_1);
                return _mm256_or_si256(fhalf, shalf);
            }


            Char256& operator<<=(const Char256& bV) {
                #ifdef __AVX512VL__
                    __m128i a_lo = _mm256_castsi256_si128(v);           // Pierwsze 128 bitów
                    __m128i a_hi = _mm256_extracti128_si256(v, 1);      // Drugie 128 bitów

                    __m128i b_lo = _mm256_castsi256_si128(bV.v);           // Pierwsze 128 bitów (shift values)
                    __m128i b_hi = _mm256_extracti128_si256(bV.v, 1);      // Drugie 128 bitów (shift values)

                    // Operacja przesunięcia bitowego dla dolnych 128 bitów
                    __m128i result_lo = _mm_sllv_epi16(a_lo, b_lo);     // SSE2/SSE3: zmienne przesunięcia

                    // Operacja przesunięcia bitowego dla górnych 128 bitów
                    __m128i result_hi = _mm_sllv_epi16(a_hi, b_hi);     // SSE2/SSE3: zmienne przesunięcia

                    // Łączymy wynik z powrotem w jeden wektor 256-bitowy
                    return _mm256_set_m128i(result_hi, result_lo);
                #else
                    __m256i q1_a = _mm256_and_si256(v, epi32_crate);
                    __m256i q1_b = _mm256_and_si256(bV.v, epi32_crate);

                    __m256i q2_a = _mm256_and_si256(_mm256_srli_si256(v, 1), epi32_crate);
                    __m256i q2_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), epi32_crate);

                    __m256i q3_a = _mm256_and_si256(_mm256_srli_si256(v, 2), epi32_crate);
                    __m256i q3_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), epi32_crate);

                    __m256i q4_a = _mm256_and_si256(_mm256_srli_si256(v, 3), epi32_crate);
                    __m256i q4_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), epi32_crate);

                    __m256i q1_res = _mm256_sllv_epi32(q1_a, q1_b);
                    __m256i q2_res = _mm256_sllv_epi32(q2_a, q2_b);
                    __m256i q3_res = _mm256_sllv_epi32(q3_a, q3_b);
                    __m256i q4_res = _mm256_sllv_epi32(q4_a, q4_b);

                    q1_res = _mm256_and_si256(q1_res, epi32_crate);
                    q2_res = _mm256_and_si256(q2_res, epi32_crate);
                    q3_res = _mm256_and_si256(q3_res, epi32_crate);
                    q4_res = _mm256_and_si256(q4_res, epi32_crate);

                    q2_res = _mm256_slli_si256(q2_res, 1);
                    q3_res = _mm256_slli_si256(q3_res, 2);
                    q4_res = _mm256_slli_si256(q4_res, 3);
                    
                    q1_res = _mm256_or_si256(q1_res, q2_res);
                    q2_res = _mm256_or_si256(q3_res, q4_res);
                    v = _mm256_or_si256(q1_res, q2_res);
                #endif
                return *this;
            }


            Char256& operator<<=(const unsigned int& b) {
                __m256i fhalf = _mm256_and_si256(v, epi16_crate);
                __m256i shalf = _mm256_and_si256(v, epi16_crate_shift_1);
                fhalf = _mm256_slli_epi16(fhalf, b);
                shalf = _mm256_slli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, epi16_crate);
                shalf = _mm256_and_si256(shalf, epi16_crate_shift_1);
                v = _mm256_or_si256(fhalf, shalf);
                return *this;
            }


            Char256 operator>>(const Char256& bV) {
                return v;
            }


            Char256 operator>>(const unsigned int& b) {
                __m256i fhalf = _mm256_and_si256(v, epi16_crate);
                __m256i shalf = _mm256_and_si256(v, epi16_crate_shift_1);
                fhalf = _mm256_srli_epi16(fhalf, b);
                shalf = _mm256_srli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, epi16_crate);
                shalf = _mm256_and_si256(shalf, epi16_crate_shift_1);
                return _mm256_or_si256(fhalf, shalf);
            }


            Char256& operator>>=(const Char256& bV) {
                return *this;
            }


            Char256& operator>>=(const unsigned int& b) {
                __m256i fhalf = _mm256_and_si256(v, epi16_crate);
                __m256i shalf = _mm256_and_si256(v, epi16_crate_shift_1);
                fhalf = _mm256_srli_epi16(fhalf, b);
                shalf = _mm256_srli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, epi16_crate);
                shalf = _mm256_and_si256(shalf, epi16_crate_shift_1);
                v = _mm256_or_si256(fhalf, shalf);
                return *this;
            }

            Char256 operator~(){
                return _mm256_xor_si256(v, ones);
            }

            std::string str() {
                std::string result = "Char256(";
                char* iv = (char*)&v; 
                for(unsigned i{0}; i < 31; ++i)
                    result += std::to_string(static_cast<int>(iv[i])) + ", ";
                
                result += std::to_string(static_cast<int>(iv[31]));
                result += ")";
                return result;
            }

    };
}

#endif