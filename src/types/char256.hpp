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
                return v;
            }


            Char256 operator*(const char& b) {
                return v;
            }


            Char256& operator*=(const Char256& bV) {
                return *this;
            }


            Char256& operator*=(const char& b) {
                return *this;
            }


            Char256 operator/(const Char256& bV) {
                return v;
            }


            Char256 operator/(const char& b) {
                return v;
            }


            Char256& operator/=(const Char256& bV) {
                return *this;
            }


            Char256& operator/=(const char& b) {
                return *this;
            }


            Char256 operator%(const Char256& bV) {
                return v;
            }


            Char256 operator%(const char& b) {
                return v;
            }


            Char256& operator%=(const Char256& bV) {
                return *this;
            }


            Char256& operator%=(const char& b) {
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
                return v;
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