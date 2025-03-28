#pragma once
#ifndef CHAR256_HPP__
#define CHAR256_HPP__

#include <array>
#include <tuple>
#include <ostream>
#include <utility>
#include <cstring>
#include <string>
#include <stdexcept>
#include <immintrin.h>
#include "constants.hpp"

namespace avx {
    /**
     * Class representing vectorized version of `char`.
     * It can hold 32 individual `char` variables.
     * Provides support for arithmetic and bitwise operators.
     * `str()` method returns stored data as string.
     * Supports printing directly to stream (cout).
     */
    class Char256 {
        private:

            __m256i v;
            /**
             * Sign extend to two vectors containing 16 `short` values each.
             * 
             * @param vec Vector containing 32 `epi8` aka `char` values.
             * @return Pair of sign extended `epi16` aka `short` values.
             */
            static std::pair<__m256i,__m256i> _sig_ext_epi8_epi16(const __m256i& vec) noexcept {
                __m256i fhalf = _mm256_and_si256(vec, constants::EPI8_CRATE_EPI16_INVERSE);
                __m256i shalf = _mm256_and_si256(vec, constants::EPI8_CRATE_EPI16);
                
                fhalf = _mm256_srai_epi16(fhalf, 8);

                shalf = _mm256_slli_si256(shalf, 1);
                shalf = _mm256_srai_epi16(shalf, 8);

                return {fhalf, shalf};
            }

            /**
             * Sign extend to two vectors containing 16 `short` values each.
             * 
             * @param vec Vector containing 32 `epi8` aka `char` values.
             * @return Pair of sign extended `epi16` aka `short` values.
             */
            static std::pair<__m256i,__m256i> _sig_ext_epi16_epi32(const __m256i& vec) noexcept {
                __m256i fhalf = _mm256_and_si256(vec, constants::EPI16_CRATE_EPI32_INVERSE);
                __m256i shalf = _mm256_and_si256(vec, constants::EPI16_CRATE_EPI32);
                
                fhalf = _mm256_srai_epi32(fhalf, 16);

                shalf = _mm256_slli_si256(shalf, 2);
                shalf = _mm256_srai_epi32(shalf, 16);

                return {fhalf, shalf};
            }

        public:
            
            /**
             * Number of individual values stored by object. This value can be used to iterate over elements.
            */
            static constexpr int size = 32;
            
            /**
             * Type that is stored inside vector.
             */
            using storedType = char;

            /**
             * Creates object and fills with 0.
             */
            Char256() noexcept : v(_mm256_setzero_si256()){}

            /**
             * Initializes object with provided value.
             * 
             * @param init Value to be broadcasted to vector content.
             */
            Char256(const char init) noexcept : v(_mm256_set1_epi8(init)){}

            /**
             * Initializes object with `__m256i` vector.
             * @param init Vector from which values will be copied.
             */
            Char256(const __m256i& init) noexcept : v(init){}

            /**
             * Initializes object with another object.
             * @param init Object which content will be copied.
             */
            Char256(const Char256& init) noexcept : v(init.v){}

            /**
             * Initializes object with first 32 bytes of data stored under `addr`.
             * Data does not need to be aligned to a 32 byte boundary. 
             * 
             * @param addr Memory holding data (minimum 32 bytes).
             * @throw If used in debug mode if `addr` is `nullptr` then `std::invalid_argument` will be thrown. Otherwise if `nullptr` is passed it will initialize vector with 0s.
             */
            explicit Char256(const char* addr)
            #ifndef NDEBUG
                {
                    if(addr == nullptr)throw std::invalid_argument("Passed address is nullptr!");
                    v = _mm256_lddqu_si256((const __m256i*)addr);
                }
            #else
                noexcept{
                    if(addr != nullptr) 
                        v = _mm256_lddqu_si256((const __m256i*)addr);
                    else
                        v = _mm256_setzero_si256();
                }
            #endif

            /**
             * Initializes with first 32 bytes read from string. If `init` is less than 32 bytes long missing values will be set to 0.
             * 
             * @param init String containing initial data.
             */
            Char256(const std::string& init) noexcept {
                if(init.size() >= 32)
                    v = _mm256_lddqu_si256((const __m256i*)init.data());
                else {
                    alignas(32) char initV[32];
                    #ifdef _MSC_VER
                        memcpy_s(initV, 32, init.data(), init.size());
                    #elif defined( __GNUC__)
                        memcpy(initV, init.data(), init.size());
                    #endif
                    memset(initV + init.size(), 0, 32 - init.size()); // To make sure all other bytes are set to 0.
                    v = _mm256_load_si256((const __m256i*) initV);
                }
            }

            /**
             * Initializes object with array of 32 bytes.
             * 
             * @param init Array containing initial data.
             */
            Char256(const std::array<char, 32>& init) noexcept : v(_mm256_lddqu_si256((const __m256i*)init.data())){}

            /**
             * Initializes object using initializer list. The number of elements in list is not limited but only a maximum of first 32 will be used.
             * If size of list is less than 32 bytes missing values will be set to 0.
             * 
             * @param init Initializer list of values to assign.
             */
            Char256(std::initializer_list<char> init) {
                alignas(32) char init_v[32];
                memset(init_v, 0, 32);
                if(init.size() < 32){
                    auto begin = init.begin();
                    for(int i{0}; i < init.size(); ++i){
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                else {
                    auto begin = init.begin();
                    for(int i{0}; i < 32; ++i){
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                v = _mm256_load_si256((const __m256i*)init_v);
            }

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param sP Pointer to memory from which to load data.
             */
            void load(const char *sP) {
                if(sP != nullptr)
                    v = _mm256_lddqu_si256((const __m256i*)sP);
            }

            /**
             * Saves data to destination in memory.
             * @param dest A valid pointer to a memory of at least 32 bytes (`char`).
             * @throws If in debug mode and `dest` is `nullptr` throws `std::invalid_argument`. Otherwise no exception will be thrown. 
             */
            void save(std::array<char, 32>& dest) const {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param dest A valid pointer to a memory of at least 32 bytes (`char`).
             * @throws If in debug mode and `dest` is `nullptr` throws `std::invalid_argument`. Otherwise no exception will be thrown. 
             */
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
             * @throws If index is not within the correct range and build type is debug `std::out_of_range` will be thrown. Otherwise bitwise AND will prevent index to be out of range.
             */
            char operator[](const unsigned int& index) const 
            #ifndef NDEBUG
                {
                    if(index > 31)
                        throw std::out_of_range("Range be within range 0-31! Got: " + std::to_string(index));
                    return ((char*)&v)[index];
                }
            #else
                noexcept { return ((char*)&v)[index & 31]; }
            #endif 

            /**
             * Compare with other vector for equality.
             * 
             * @param bV Second object to compare.
             * @return If ALL values are the same then it will return `true`, otherwise `false`.
             */
            bool operator==(const Char256& bV) const noexcept {
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) != 0;
            }

            /**
             * Compare if ALL values in vector are the same as provided in `b`.
             * 
             * @param b Scalar value to compare with.
             * @return If ALL values in vector are equal to `b` then will return `true`, otherwise `false` will be returned.
             */
            bool operator==(const char b) const noexcept {
                __m256i bV = _mm256_set1_epi8(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) != 0;
            }

            /**
             * Compares vectors for inequality.
             * 
             * @param bV Vector to compare with.
             * @return If ANY value doesn't match then `true` will be returned. Otherwise will return `false`.
             */
            bool operator!=(const Char256& bV) const noexcept {
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) == 0;
            }

            /**
             * Compares vector with scalar for inequality.
             * 
             * @param b Scalar to compare with.
             * @return If ANY value doesn't match with `b` then `true` will be returned. Otherwise will return `false`.
             */
            bool operator!=(const char b) const noexcept {
                __m256i bV = _mm256_set1_epi8(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) == 0;
            }

            /**
             * Adds two vectors. Simple call to `_mm256_add_epi8`.
             * 
             * @param bV Vector to add.
             * @return New object being a result of addition of vectors.
             */
            Char256 operator+(const Char256& bV) const noexcept {
                return _mm256_add_epi8(v, bV.v);
            }

            /**
             * Adds a scalar to vector. Similar to one using Char256 but creates intermediate vector filled with value of `b`.
             * 
             * @param b Scalar to add.
             * @return New object being a result of adding value of `b` to vector.
             */
            Char256 operator+(const char& b) const noexcept{
                return _mm256_add_epi8(v, _mm256_set1_epi8(b));
            }

            /**
             * Adds second vector and returns reference to existing vector.
             * 
             * @param bV Vector to add.
             * @return Reference to the same object after performing addition (`*this`).
             */
            Char256& operator+=(const Char256& bV) noexcept {
                v = _mm256_add_epi8(v, bV.v);
                return *this;
            }


            /**
             * Adds a scalar to vector and returns reference to existing vector.
             * 
             * @param b Scalar to add.
             * @return Reference to the same object after performing addition (`*this`).
             */
            Char256& operator+=(const char& b) noexcept {
                v =_mm256_add_epi8(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator-(const Char256& bV) const noexcept {
                return _mm256_sub_epi8(v, bV.v);
            }


            Char256 operator-(const char& b) const noexcept {
                return _mm256_sub_epi8(v, _mm256_set1_epi8(b));
            }


            Char256& operator-=(const Char256& bV) noexcept {
                v = _mm256_sub_epi8(v, bV.v);
                return *this;
            }


            Char256& operator-=(const char& b) noexcept {
                v = _mm256_sub_epi8(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator*(const Char256& bV) const noexcept {
                __m256i fhalf_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i fhalf_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI16);

                __m256i shalf_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                __m256i shalf_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI16_INVERSE);

                shalf_a = _mm256_srli_si256(shalf_a, 1);
                shalf_b = _mm256_srli_si256(shalf_b, 1);

                __m256i fresult = _mm256_mullo_epi16(fhalf_a, fhalf_b);
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI16);

                __m256i sresult = _mm256_mullo_epi16(shalf_a, shalf_b);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI16);
                sresult = _mm256_slli_si256(sresult, 1);

                return _mm256_or_si256(fresult, sresult);
            }


            Char256 operator*(const char& b) const noexcept {
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i bV = _mm256_set1_epi16(b); 

                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);

                shalf = _mm256_srli_si256(shalf, 1);

                __m256i fresult = _mm256_mullo_epi16(fhalf, bV);
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI16);

                __m256i sresult = _mm256_mullo_epi16(shalf, bV);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI16);
                sresult = _mm256_slli_si256(sresult, 1);

                return _mm256_or_si256(fresult, sresult);
            }


            Char256& operator*=(const Char256& bV) noexcept {
                __m256i fhalf_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i fhalf_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI16);

                __m256i shalf_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                __m256i shalf_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI16_INVERSE);

                shalf_a = _mm256_srli_si256(shalf_a, 1);
                shalf_b = _mm256_srli_si256(shalf_b, 1);

                __m256i fresult = _mm256_mullo_epi16(fhalf_a, fhalf_b);
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI16);

                __m256i sresult = _mm256_mullo_epi16(shalf_a, shalf_b);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI16);
                sresult = _mm256_slli_si256(sresult, 1);

                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }


            Char256& operator*=(const char& b) noexcept {
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i bV = _mm256_set1_epi16(b); 

                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);

                shalf = _mm256_srli_si256(shalf, 1);

                __m256i fresult = _mm256_mullo_epi16(fhalf, bV);
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI16);

                __m256i sresult = _mm256_mullo_epi16(shalf, bV);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI16);
                sresult = _mm256_slli_si256(sresult, 1);

                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }


            Char256 operator/(const Char256& bV) const noexcept {
                /*alignas(32) char vP[size];
                alignas(32) char bP[size];

                _mm256_store_si256((__m256i*)vP, v);

                for(unsigned int i = 0; i < size; ++i)
                    vP[i] = bP[i] ? vP[i] / bP[i] : 0;

                return _mm256_load_si256((const __m256i*)vP);
                */

                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                auto [b_fhalf_epi16, b_shalf_epi16] = _sig_ext_epi8_epi16(bV.v);

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                auto [bv_first_half, bv_second_half] = _sig_ext_epi16_epi32(b_fhalf_epi16);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 3);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                std::tie(bv_first_half, bv_second_half) = _sig_ext_epi16_epi32(b_shalf_epi16);
                bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                return _mm256_or_si256(half_res, shalf_res);
            }


            Char256 operator/(const char& b) const noexcept {
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                __m256 bV = _mm256_set1_ps(static_cast<float>(b));

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 3);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                return _mm256_or_si256(half_res, shalf_res);
            }


            Char256& operator/=(const Char256& bV) noexcept {
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                auto [b_fhalf_epi16, b_shalf_epi16] = _sig_ext_epi8_epi16(bV.v);

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                auto [bv_first_half, bv_second_half] = _sig_ext_epi16_epi32(b_fhalf_epi16);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 3);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                std::tie(bv_first_half, bv_second_half) = _sig_ext_epi16_epi32(b_shalf_epi16);
                bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                v = _mm256_or_si256(half_res, shalf_res);
                return *this;
            }


            Char256& operator/=(const char& b) noexcept {
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                __m256 bV = _mm256_set1_ps(static_cast<float>(b));

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 3);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                v = _mm256_or_si256(half_res, shalf_res);
                return *this;
            }


            Char256 operator%(const Char256& bV) const noexcept {
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                auto [b_fhalf_epi16, b_shalf_epi16] = _sig_ext_epi8_epi16(bV.v);

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                auto [bv_first_half, bv_second_half] = _sig_ext_epi16_epi32(b_fhalf_epi16);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                std::tie(bv_first_half, bv_second_half) = _sig_ext_epi16_epi32(b_shalf_epi16);
                bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                fresult = _mm256_mullo_epi16(half_res, b_fhalf_epi16);
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI16);
                fresult = _mm256_slli_si256(fresult, 1);

                sresult = _mm256_mullo_epi16(shalf_res, b_shalf_epi16);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI16);
                
                __m256i result = _mm256_or_si256(fresult, sresult);

                return _mm256_sub_epi8(v, result);
            }


            Char256 operator%(const char& b) const noexcept {
                /*alignas(32) char vP[size];

                _mm256_store_si256((__m256i*)vP, v);

                for(unsigned int i = 0; i < 32; ++i)
                    vP[i] = b ? vP[i] % b : 0;

                return _mm256_load_si256((const __m256i*)vP);*/

                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);

                __m256 bV = _mm256_set1_ps(static_cast<float>(b));
                __m256i bV_epi16 = _mm256_set1_epi16(static_cast<short>(b));

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                fresult = _mm256_mullo_epi16(half_res, bV_epi16);
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI16);
                fresult = _mm256_slli_si256(fresult, 1);

                sresult = _mm256_mullo_epi16(shalf_res, bV_epi16);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI16);
                
                __m256i result = _mm256_or_si256(fresult, sresult);

                return _mm256_sub_epi8(v, result);
            }


            Char256& operator%=(const Char256& bV) noexcept {
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                auto [b_fhalf_epi16, b_shalf_epi16] = _sig_ext_epi8_epi16(bV.v);

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                auto [bv_first_half, bv_second_half] = _sig_ext_epi16_epi32(b_fhalf_epi16);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                std::tie(bv_first_half, bv_second_half) = _sig_ext_epi16_epi32(b_shalf_epi16);
                bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                fresult = _mm256_mullo_epi16(half_res, b_fhalf_epi16);
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI16);
                fresult = _mm256_slli_si256(fresult, 1);

                sresult = _mm256_mullo_epi16(shalf_res, b_shalf_epi16);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI16);
                
                __m256i result = _mm256_or_si256(fresult, sresult);

                v = _mm256_sub_epi8(v, result);
                return *this;
            }


            Char256& operator%=(const char& b) noexcept {
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);

                __m256 bV = _mm256_set1_ps(static_cast<float>(b));
                __m256i bV_epi16 = _mm256_set1_epi16(static_cast<short>(b));

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                fresult = _mm256_mullo_epi16(half_res, bV_epi16);
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI16);
                fresult = _mm256_slli_si256(fresult, 1);

                sresult = _mm256_mullo_epi16(shalf_res, bV_epi16);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI16);
                
                __m256i result = _mm256_or_si256(fresult, sresult);

                v = _mm256_sub_epi8(v, result);
                return *this;
            }


            Char256 operator&(const Char256& bV) const noexcept {
                return _mm256_and_si256(v, bV.v);
            }


            Char256 operator&(const char& b) const noexcept {
                return _mm256_and_si256(v, _mm256_set1_epi8(b));
            }


            Char256& operator&=(const Char256& bV) noexcept {
                v = _mm256_and_si256(v, bV.v);
                return *this;
            }


            Char256& operator&=(const char& b) noexcept {
                v = _mm256_and_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator|(const Char256& bV) const noexcept {
                return _mm256_or_si256(v, bV.v);
            }


            Char256 operator|(const char& b) const noexcept {
                return _mm256_or_si256(v, _mm256_set1_epi8(b));
            }


            Char256& operator|=(const Char256& bV) noexcept {
                v = _mm256_or_si256(v, bV.v);
                return *this;
            }


            Char256& operator|=(const char& b) noexcept {
                v = _mm256_or_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator^(const Char256& bV) const noexcept {
                return _mm256_xor_si256(v, bV.v);
            }


            Char256 operator^(const char& b) const noexcept {
                return _mm256_xor_si256(v, _mm256_set1_epi8(b));
            }


            Char256& operator^=(const Char256& bV) noexcept {
                v = _mm256_xor_si256(v, bV.v);
                return *this;
            }


            Char256& operator^=(const char& b) noexcept {
                v = _mm256_xor_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            Char256 operator<<(const Char256& bV) const noexcept {
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
                    __m256i q1_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                    __m256i q1_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);

                    __m256i q2_a = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                    __m256i q2_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);

                    __m256i q3_a = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                    __m256i q3_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);

                    __m256i q4_a = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                    __m256i q4_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), constants::EPI8_CRATE_EPI32);

                    __m256i q1_res = _mm256_sllv_epi32(q1_a, q1_b);
                    __m256i q2_res = _mm256_sllv_epi32(q2_a, q2_b);
                    __m256i q3_res = _mm256_sllv_epi32(q3_a, q3_b);
                    __m256i q4_res = _mm256_sllv_epi32(q4_a, q4_b);

                    q1_res = _mm256_and_si256(q1_res, constants::EPI8_CRATE_EPI32);
                    q2_res = _mm256_and_si256(q2_res, constants::EPI8_CRATE_EPI32);
                    q3_res = _mm256_and_si256(q3_res, constants::EPI8_CRATE_EPI32);
                    q4_res = _mm256_and_si256(q4_res, constants::EPI8_CRATE_EPI32);

                    q2_res = _mm256_slli_si256(q2_res, 1);
                    q3_res = _mm256_slli_si256(q3_res, 2);
                    q4_res = _mm256_slli_si256(q4_res, 3);
                    
                    q1_res = _mm256_or_si256(q1_res, q2_res);
                    q2_res = _mm256_or_si256(q3_res, q4_res);
                    return _mm256_or_si256(q1_res, q2_res);
                #endif
            }


            Char256 operator<<(const unsigned int& b) const noexcept {
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                fhalf = _mm256_slli_epi16(fhalf, b);
                shalf = _mm256_slli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                return _mm256_or_si256(fhalf, shalf);
            }


            Char256& operator<<=(const Char256& bV) noexcept {
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
                    v = _mm256_set_m128i(result_hi, result_lo);
                #else
                    __m256i q1_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                    __m256i q1_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);

                    __m256i q2_a = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                    __m256i q2_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);

                    __m256i q3_a = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                    __m256i q3_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);

                    __m256i q4_a = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                    __m256i q4_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), constants::EPI8_CRATE_EPI32);

                    __m256i q1_res = _mm256_sllv_epi32(q1_a, q1_b);
                    __m256i q2_res = _mm256_sllv_epi32(q2_a, q2_b);
                    __m256i q3_res = _mm256_sllv_epi32(q3_a, q3_b);
                    __m256i q4_res = _mm256_sllv_epi32(q4_a, q4_b);

                    q1_res = _mm256_and_si256(q1_res, constants::EPI8_CRATE_EPI32);
                    q2_res = _mm256_and_si256(q2_res, constants::EPI8_CRATE_EPI32);
                    q3_res = _mm256_and_si256(q3_res, constants::EPI8_CRATE_EPI32);
                    q4_res = _mm256_and_si256(q4_res, constants::EPI8_CRATE_EPI32);

                    q2_res = _mm256_slli_si256(q2_res, 1);
                    q3_res = _mm256_slli_si256(q3_res, 2);
                    q4_res = _mm256_slli_si256(q4_res, 3);
                    
                    q1_res = _mm256_or_si256(q1_res, q2_res);
                    q2_res = _mm256_or_si256(q3_res, q4_res);
                    v = _mm256_or_si256(q1_res, q2_res);
                #endif
                return *this;
            }


            Char256& operator<<=(const unsigned int& b) noexcept {
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                fhalf = _mm256_slli_epi16(fhalf, b);
                shalf = _mm256_slli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                v = _mm256_or_si256(fhalf, shalf);
                return *this;
            }


            Char256 operator>>(const Char256& bV) const noexcept {
                __m256i q1_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i q1_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);

                __m256i q2_a = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256i q2_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);

                __m256i q3_a = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                __m256i q3_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);

                __m256i q4_a = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                __m256i q4_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), constants::EPI8_CRATE_EPI32);

                __m256i q1_res = _mm256_srlv_epi32(q1_a, q1_b);
                __m256i q2_res = _mm256_srlv_epi32(q2_a, q2_b);
                __m256i q3_res = _mm256_srlv_epi32(q3_a, q3_b);
                __m256i q4_res = _mm256_srlv_epi32(q4_a, q4_b);

                q1_res = _mm256_and_si256(q1_res, constants::EPI8_CRATE_EPI32);
                q2_res = _mm256_and_si256(q2_res, constants::EPI8_CRATE_EPI32);
                q3_res = _mm256_and_si256(q3_res, constants::EPI8_CRATE_EPI32);
                q4_res = _mm256_and_si256(q4_res, constants::EPI8_CRATE_EPI32);

                q2_res = _mm256_slli_si256(q2_res, 1);
                q3_res = _mm256_slli_si256(q3_res, 2);
                q4_res = _mm256_slli_si256(q4_res, 3);
                
                q1_res = _mm256_or_si256(q1_res, q2_res);
                q2_res = _mm256_or_si256(q3_res, q4_res);
                return _mm256_or_si256(q1_res, q2_res);
            }


            Char256 operator>>(const unsigned int& b) const noexcept {
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                fhalf = _mm256_srli_epi16(fhalf, b);
                shalf = _mm256_srli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                return _mm256_or_si256(fhalf, shalf);
            }


            Char256& operator>>=(const Char256& bV) noexcept {
                __m256i q1_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i q1_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);

                __m256i q2_a = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256i q2_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);

                __m256i q3_a = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                __m256i q3_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);

                __m256i q4_a = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                __m256i q4_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), constants::EPI8_CRATE_EPI32);

                __m256i q1_res = _mm256_srlv_epi32(q1_a, q1_b);
                __m256i q2_res = _mm256_srlv_epi32(q2_a, q2_b);
                __m256i q3_res = _mm256_srlv_epi32(q3_a, q3_b);
                __m256i q4_res = _mm256_srlv_epi32(q4_a, q4_b);

                q1_res = _mm256_and_si256(q1_res, constants::EPI8_CRATE_EPI32);
                q2_res = _mm256_and_si256(q2_res, constants::EPI8_CRATE_EPI32);
                q3_res = _mm256_and_si256(q3_res, constants::EPI8_CRATE_EPI32);
                q4_res = _mm256_and_si256(q4_res, constants::EPI8_CRATE_EPI32);

                q2_res = _mm256_slli_si256(q2_res, 1);
                q3_res = _mm256_slli_si256(q3_res, 2);
                q4_res = _mm256_slli_si256(q4_res, 3);
                
                q1_res = _mm256_or_si256(q1_res, q2_res);
                q2_res = _mm256_or_si256(q3_res, q4_res);
                v = _mm256_or_si256(q1_res, q2_res);
                return *this;
            }


            Char256& operator>>=(const unsigned int& b) noexcept {
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                fhalf = _mm256_srli_epi16(fhalf, b);
                shalf = _mm256_srli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                v = _mm256_or_si256(fhalf, shalf);
                return *this;
            }

            /**
             * Bitwise NOT.
             * 
             * @return New object with negated bits.
             */
            Char256 operator~() const noexcept {
                return _mm256_xor_si256(v, constants::ONES);
            }

            /**
             * A string representation of internal vector contents. All 32 stored values will be printed out.
             * 
             * @return String in the following format (for default constructor): "Char256(0, 0, [...], 0)"
             */
            std::string str() const noexcept {
                std::string result = "Char256(";
                char* iv = (char*)&v; 
                for(unsigned i{0}; i < 31; ++i)
                    result += std::to_string(static_cast<short>(iv[i])) + ", ";
                
                result += std::to_string(static_cast<short>(iv[31]));
                result += ")";
                return result;
            }

            /**
             * Creates a string from internal vector.
             * This function is safe even if data is not null-terminated. 
             * 
             * @return String filled with contents of internal vector.
             */
            std::string toString() const noexcept {
                alignas(32) char tmp[33];
                tmp[32] = '\0';

                _mm256_store_si256((__m256i*)tmp, v);

                return std::string(tmp);
            }

            /**
             * Prints content of vector as raw string.
             * @param os Output stream, to which content will be written.
             * @param a Vector, whose value will be written to stream.
             * @return Reference to `os`.
             */
            friend std::ostream& operator<<(std::ostream& os, const Char256& a);

    };
}

#endif