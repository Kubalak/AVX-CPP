#pragma once
#ifndef UCHAR256_HPP__
#define UCHAR256_HPP__

#include <array>
#include <string>
#include <stdexcept>
#include <immintrin.h>

namespace avx {
    class UChar256{

        __m256i v;

        public:

            static constexpr int size = 32;

            UChar256() noexcept : v(_mm256_setzero_si256()){}

            UChar256(const unsigned char init) noexcept : v(_mm256_set1_epi8(init)){}

            UChar256(const __m256i& init) noexcept : v(init){}

            UChar256(const UChar256& init) noexcept : v(init.v){}

            explicit UChar256(const unsigned char* addr)
            #ifndef NDEBUG
                {
                    if(addr == nullptr)throw std::invalid_argument("Passed address is nullptr!");
                    v = _mm256_lddqu_si256((const __m256i*)addr);
                }
            #else
                : v(_mm256_lddqu_si256((const __m256i*)addr)){}
            #endif

            UChar256(const std::string& init) noexcept {
                if(init.size() >= 32)
                    v = _mm256_lddqu_si256((const __m256i*)init.data());
                else {
                    alignas(32) char initV[32];
                    memset(initV, 0, 32);
                    #ifdef _MSC_VER
                        strncpy_s(initV, 32, init.data(), init.size());
                    #elif defined( __GNUC__)
                        strncpy(initV, init.data(), init.size());
                    #endif
                    v = _mm256_load_si256((const __m256i*) initV);
                }
            }

            UChar256(const std::array<unsigned char, 32>& init) noexcept : v(_mm256_lddqu_si256((const __m256i*)init.data())){}

            UChar256(std::initializer_list<unsigned char> init) {
                alignas(32) unsigned char init_v[32];
                memset(init_v, 0, 32);
                if(init.size() < 32){
                    auto begin = init.begin();
                    for(int i{0}; i < init.size(); ++i)
                        init_v[i] = *begin++;
                }
                else {
                    auto begin = init.begin();
                    for(int i{0}; i < 32; ++i)
                        init_v[i] = *begin++;
                }
                v = _mm256_load_si256((const __m256i*)init_v);
            }


            /**
             * Saves data to destination in memory.
             * @param dest A valid pointer to a memory of at least 32 bytes (`char`).
             * @throws If in debug mode and `dest` is `nullptr` throws `std::invalid_argument`. Otherwise no exception will be thrown. 
             */
            void save(std::array<unsigned char, 32>& dest) const {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param dest A valid pointer to a memory of at least 32 bytes (`char`).
             * @throws If in debug mode and `dest` is `nullptr` throws `std::invalid_argument`. Otherwise no exception will be thrown. 
             */
            void save(unsigned char* dest) const {
                #ifndef NDEBUG
                    if(dest == nullptr) throw std::invalid_argument("Passed address is nullptr!");
                #endif

                _mm256_storeu_si256((__m256i*)dest, v);
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
                #endif

                _mm256_store_si256((__m256i*)dest, v);
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
            unsigned char operator[](const unsigned int& index) const {
                if(index > 31)
                    throw std::out_of_range("Range be within range 0-31! Got: " + std::to_string(index));
                return ((unsigned char*)&v)[index];
            }


            bool operator==(const UChar256& bV) const noexcept {
                __m256i eq = _mm256_cmpeq_epi8(v, bV.v);
                unsigned long long* eqV = (unsigned long long*)&eq;
                for(uint8_t i = 0; i < 4; ++i)
                    if(eqV[i] != UINT64_MAX)
                        return false;
                return true;
            }


            bool operator==(const unsigned char b) const noexcept {
                unsigned char* v1,* v2;
                v1 = (unsigned char*)&v;

                for(unsigned int i{0}; i < size; ++i)
                    if(v1[i] != b)
                        return false;

                return true;
            }


            bool operator!=(const UChar256& bV) const noexcept {
                __m256i eq = _mm256_cmpeq_epi8(v, bV.v);
                unsigned long long* eqV = (unsigned long long*)&eq;
                for(uint8_t i = 0; i < 4; ++i)
                    if(eqV[i] != UINT64_MAX)
                        return true;
                return false;
            }


            bool operator!=(const unsigned char b) const noexcept {
                unsigned char* v1,* v2;
                v1 = (unsigned char*)&v;

                for(unsigned int i{0}; i < size; ++i)
                    if(v1[i] != b)
                        return true;

                return false;
            }


            UChar256 operator+(const UChar256& bV) const noexcept {
                return _mm256_add_epi8(v, bV.v);
            }


            UChar256 operator+(const unsigned char& b) const noexcept{
                return _mm256_add_epi8(v, _mm256_set1_epi8(b));
            }


            UChar256& operator+=(const UChar256& bV) noexcept {
                v = _mm256_add_epi8(v, bV.v);
                return *this;
            }


            UChar256& operator+=(const unsigned char& b) noexcept {
                v =_mm256_add_epi8(v, _mm256_set1_epi8(b));
                return *this;
            }


            UChar256 operator-(const UChar256& bV) const noexcept {
                return _mm256_sub_epi8(v, bV.v);
            }


            UChar256 operator-(const unsigned char& b) const noexcept {
                return _mm256_sub_epi8(v, _mm256_set1_epi8(b));
            }


            UChar256& operator-=(const UChar256& bV) noexcept {
                v = _mm256_sub_epi8(v, bV.v);
                return *this;
            }


            UChar256& operator-=(const unsigned char& b) noexcept {
                v = _mm256_sub_epi8(v, _mm256_set1_epi8(b));
                return *this;
            }






            std::string str() const noexcept {
                std::string result = "UChar256(";
                unsigned char* iv = (unsigned char*)&v; 
                for(unsigned i{0}; i < 31; ++i)
                    result += std::to_string(static_cast<unsigned int>(iv[i])) + ", ";
                
                result += std::to_string(static_cast<unsigned int>(iv[31]));
                result += ")";
                return result;
            }

            std::string toString() const noexcept {
                alignas(32) unsigned char tmp[33];
                tmp[32] = '\0';

                _mm256_store_si256((__m256i*)tmp, v);

                return std::string(reinterpret_cast<char*>(tmp));
            }

    };
}

#endif