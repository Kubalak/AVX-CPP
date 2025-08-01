#pragma once
#ifndef USHORT256_HPP__
#define USHORT256_HPP__

#include <array>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include "constants.hpp"

namespace avx {
    /**
     * Class providing vectorized version of `unsigned short`.
     * Can hold 16 individual `unsigned short` values.
     * Provides arithmetic and bitwise operators.
     * Provides comparison operators == !=.
     */
    class UShort256 {
        private:
            __m256i v;
                    
        public:
            static constexpr const int size = 16;
            using storedType = unsigned short;

            /**
             * Default constructor. Sets zero to whole vector.
             */
            UShort256() noexcept : v(_mm256_setzero_si256()){}


            /**
             * Initializes vector with value from other vector.
             * @param init Object which value will be copied.
             */
            UShort256(const UShort256& init) noexcept : v(init.v){}


            /**
             * Initializes vector but using `__m256i` type.
             * @param init Raw value to be set.
             */
            UShort256(const __m256i& init) noexcept : v(init){}


            /**
             * Initialize vector with values read from an array.
             * @param init Array from which values will be copied.
             */
            UShort256(const std::array<unsigned short, 16>& init) noexcept : v(_mm256_lddqu_si256((const __m256i*)init.data())){}

            UShort256(std::initializer_list<unsigned short> init) {
                alignas(32) unsigned short init_v[size];
                std::memset(init_v, 0, 32);
                if(init.size() < size){
                    auto begin = init.begin();
                    for(int i{0}; i < init.size(); ++i) {
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                else {
                    auto begin = init.begin();
                    for(int i{0}; i < size; ++i) {
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                v = _mm256_load_si256((const __m256i*)init_v);
            }


            /**
             * Initialize vector with values using pointer.
             * @param addr A valid address containing at least 16 `unsigned short` numbers.
             * @throws std::invalid_argument If in debug mode and `addr` is `nullptr`. Otherwise if `addr` is nullptr vector will be filled with 0's.
             */
            explicit UShort256(const unsigned short* addr) 
            #ifndef NDEBUG
                {
                    if(addr == nullptr)throw std::invalid_argument("Passed address is nullptr!");
                    v = _mm256_lddqu_si256((const __m256i*)addr);
                }
            #else
                {
                    if(addr)
                        v = _mm256_lddqu_si256((const __m256i*)addr);
                    else 
                        v = _mm256_setzero_si256();
                }
            #endif


            /**
             * Initializes all vector fields with single value.
             * @param b A literal value to be set.
             */
            explicit UShort256(const unsigned short b) noexcept : v(_mm256_set1_epi16(b)){}

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param pSrc Pointer to memory from which to load data.
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void load(const unsigned short *pSrc) N_THROW_REL {
                if(pSrc)
                    v = _mm256_lddqu_si256((const __m256i*)pSrc);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif
            }

            /**
             * Saves data to destination in memory.
             * @param dest Reference to the list to which vector will be saved. Array doesn't need to be aligned to any specific boundary.
             */
            void save(std::array<unsigned short, 16>& dest) const noexcept {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (16x `unsigned short`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void save(unsigned short *pDest) const N_THROW_REL {
                if(pDest)
                    _mm256_storeu_si256((__m256i*)pDest, v);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif
            }

            /**
             * Saves data to destination in memory. The memory must be aligned at 32-byte boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (16x `unsigned short`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void saveAligned(unsigned short *pDest) const N_THROW_REL {
                if(pDest)
                    _mm256_store_si256((__m256i*)pDest, v);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
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
             * @param index Position of desired element between 0 and 15.
             * @return Value of underlying element.
             * @throws std::out_of_range If index is not within the correct range.
             */
            unsigned short operator[](const unsigned int& index) const 
            #ifndef NDEBUG
                {
                    if(index > 15)
                        throw std::out_of_range("Range be within range 0-15! Got: " + std::to_string(index));
                    return ((unsigned short*)&v)[index];
                }
            #else
                noexcept { return ((unsigned short*)&v)[index & 15]; }
            #endif


            /**
             * Compares with second vector for equality.
             * @param bV Object to compare.
             * @returns `true` if all elements are equal or `false` if not.
             */
            bool operator==(const UShort256 &bV) const noexcept {
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) != 0;
            }


            /**
             * Compares with value for equality.
             * @param b Value to compare.
             * @returns `true` if all elements are equal to passed value `false` if not.
             */
            bool operator==(const unsigned short &b) const noexcept{
                __m256i bV = _mm256_set1_epi16(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) != 0;
            }


            /**
             * Compares with second vector for inequality.
             * @param bV Object to compare.
             * @returns `true` if any alement is not equal to corresponding element in `bV` otherwise `false`.
             */
            bool operator!=(const UShort256 &bV) const noexcept{
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) == 0;
            }


            /**
             * Compares with value for inequality.
             * @param b Value
             * @returns `true` if any alement is not equal to corresponding element in `bV` otherwise `false`.
             */
            bool operator!=(const unsigned short &b) const noexcept{
                __m256i bV = _mm256_set1_epi16(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) == 0;
            }
            
            UShort256 operator+(const UShort256& bV) const noexcept{
                return _mm256_add_epi16(v, bV.v);
            }

            UShort256 operator+(const unsigned short& b) const noexcept{
                return _mm256_add_epi16(v, _mm256_set1_epi16(b));
            }

            UShort256& operator+=(const UShort256& bV) noexcept {
                v = _mm256_add_epi16(v, bV.v);
                return *this;
            }

            UShort256& operator+=(const unsigned short& b) noexcept {
                v = _mm256_add_epi16(v, _mm256_set1_epi16(b));
                return *this;
            }

            UShort256 operator-(const UShort256& bV) const noexcept {
                return _mm256_sub_epi16(v, bV.v);
            }

            UShort256 operator-(const unsigned short& b) const noexcept {
                return _mm256_sub_epi16(v, _mm256_set1_epi16(b));
            }

            UShort256& operator-=(const UShort256& bV) noexcept {
                v = _mm256_sub_epi16(v, bV.v);
                return *this;
            }

            UShort256& operator-=(const unsigned short& b) noexcept {
                v =_mm256_sub_epi16(v, _mm256_set1_epi16(b));
                return *this;
            }

            UShort256 operator*(const UShort256& bV) const noexcept {
                __m256i fhalf_a = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE);
                fhalf_a = _mm256_srli_si256(fhalf_a, 2);
                __m256i fhalf_b = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                fhalf_b = _mm256_srli_si256(fhalf_b, 2);
                __m256i shalf_a = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);
                __m256i shalf_b = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);

                __m256i fresult = _mm256_mullo_epi32(fhalf_a, fhalf_b);
                __m256i sresult = _mm256_mullo_epi32(shalf_a, shalf_b);

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                fresult = _mm256_slli_si256(fresult, 2);

                return _mm256_or_si256(fresult, sresult);
            }

            UShort256 operator*(const unsigned short& b) const noexcept {
                __m256i bV = _mm256_set_epi16(0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b);
                __m256i fhalf = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                fhalf = _mm256_srli_si256(fhalf, 2);
                __m256i shalf= _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);

                __m256i fresult = _mm256_mullo_epi32(fhalf, bV);
                __m256i sresult = _mm256_mullo_epi32(shalf, bV);

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                fresult = _mm256_slli_si256(fresult, 2);

                return _mm256_or_si256(fresult, sresult);
            }

            UShort256& operator*=(const UShort256& bV) noexcept {
                __m256i fhalf_a = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE);
                fhalf_a = _mm256_srli_si256(fhalf_a, 2);
                __m256i fhalf_b = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                fhalf_b = _mm256_srli_si256(fhalf_b, 2);
                __m256i shalf_a = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);
                __m256i shalf_b = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);

                __m256i fresult = _mm256_mullo_epi32(fhalf_a, fhalf_b);
                __m256i sresult = _mm256_mullo_epi32(shalf_a, shalf_b);

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                fresult = _mm256_slli_si256(fresult, 2);

                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }

            UShort256& operator*=(const unsigned short& b) noexcept {
                __m256i bV = _mm256_set_epi16(0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b);
                __m256i fhalf = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                fhalf = _mm256_srli_si256(fhalf, 2);
                __m256i shalf= _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);

                __m256i fresult = _mm256_mullo_epi32(fhalf, bV);
                __m256i sresult = _mm256_mullo_epi32(shalf, bV);

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                fresult = _mm256_slli_si256(fresult, 2);

                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }

            /**
             * Performs an integer division. 
             * 
             * NOTE: Value is first casted to `int` and then to `float` and inverse to return integer result which has not been yet tested for performance.
             * @param bV Divisors vector.
             * @return Result of integer division with truncation.
             */
            UShort256 operator/(const UShort256& bV) const noexcept {
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                v_first_half = _mm256_srli_si256(v_first_half, 2);
                __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bv_first_half = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE);
                bv_first_half = _mm256_srli_si256(bv_first_half, 2);
                __m256i bv_second_half = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);

                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);
                
                return _mm256_or_si256(fresult, sresult);
            }

             /**
             * Performs an integer division. 
             * 
             * NOTE: Value is first casted to `int` and then to `float` and inverse to return integer result which has not been yet tested for performance.
             * @param bV Divisor value.
             * @return Result of integer division with truncation.
             */
            UShort256 operator/(const unsigned short& b) const noexcept {
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                v_first_half = _mm256_srli_si256(v_first_half, 2);
                __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256 bV = _mm256_set_ps(b, b, b, b, b, b, b, b);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);

                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);
                
                return _mm256_or_si256(fresult, sresult);
            }

            UShort256& operator/=(const UShort256& bV) noexcept {
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                v_first_half = _mm256_srli_si256(v_first_half, 2);
                __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bv_first_half = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE);
                bv_first_half = _mm256_srli_si256(bv_first_half, 2);
                __m256i bv_second_half = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);

                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);
                
                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }
            
            UShort256& operator/=(const unsigned short& b) noexcept {
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                v_first_half = _mm256_srli_si256(v_first_half, 2);
                __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256 bV = _mm256_set_ps(b, b, b, b, b, b, b, b);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);

                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);
                
                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }

            /**
             * Performs a modulo operation. The implemented algorithm works as shown below:
             * 
             * mod(a, b) -> a - b * (a / b) where `/` is an integer division.
             * Due to SIMD (AVX2) limitations values are casted to two float vectors and then divided.
             * 
             * NOTE: Analogously as in `/` and `/=` operators values are casted before performing a division.
             * @param bV Divisor.
             * @return Modulo result.
             */
            UShort256 operator%(const UShort256& bV) const noexcept {
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                v_first_half = _mm256_srli_si256(v_first_half, 2);
                __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bv_first_half = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE);
                bv_first_half = _mm256_srli_si256(bv_first_half, 2);
                __m256i bv_second_half = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);

                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                fresult = _mm256_sub_epi32(v_first_half, _mm256_mullo_epi32(bv_first_half, fresult));
                sresult = _mm256_sub_epi32(v_second_half, _mm256_mullo_epi32(bv_second_half, sresult));
                
                fresult = _mm256_slli_si256(fresult, 2);
                
                return _mm256_or_si256(fresult, sresult);
            }


            /**
             * Performs a modulo operation. The implemented algorithm works as shown below:
             * 
             * mod(a, b) -> a - b * (a / b) where `/` is an integer division.
             * Due to SIMD (AVX2) limitations values are casted to two float vectors and then divided.
             * 
             * NOTE: Analogously as in `/` and `/=` operators values are casted before performing a division.
             * @param bV Divisor.
             * @return Modulo result.
             */
            UShort256 operator%(const unsigned short& b) noexcept {
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                v_first_half = _mm256_srli_si256(v_first_half, 2);
                __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bV = _mm256_set_epi16(0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b);
                __m256 bVf = _mm256_set1_ps(b);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bVf), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bVf), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                fresult = _mm256_sub_epi32(v_first_half, _mm256_mullo_epi32(bV, fresult));
                sresult = _mm256_sub_epi32(v_second_half, _mm256_mullo_epi32(bV, sresult));
                
                fresult = _mm256_slli_si256(fresult, 2);
                
                return _mm256_or_si256(fresult, sresult);
            }

            /**
             * Performs a modulo operation. The implemented algorithm works as shown below:
             * 
             * mod(a, b) -> a - b * (a / b) where `/` is an integer division.
             * Due to SIMD (AVX2) limitations values are casted to two float vectors and then divided.
             * 
             * NOTE: Analogously as in `/` and `/=` operators values are casted before performing a division.
             * @param bV Divisor.
             * @return Reference to modified object.
             */
            UShort256& operator%=(const UShort256& bV) noexcept {
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                v_first_half = _mm256_srli_si256(v_first_half, 2);
                __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bv_first_half = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE);
                bv_first_half = _mm256_srli_si256(bv_first_half, 2);
                __m256i bv_second_half = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);

                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                fresult = _mm256_sub_epi32(v_first_half, _mm256_mullo_epi32(bv_first_half, fresult));
                sresult = _mm256_sub_epi32(v_second_half, _mm256_mullo_epi32(bv_second_half, sresult));
                
                fresult = _mm256_slli_si256(fresult, 2);
                
                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }

            /**
             * Performs a modulo operation. The implemented algorithm works as shown below:
             * 
             * mod(a, b) -> a - b * (a / b) where `/` is an integer division.
             * Due to SIMD (AVX2) limitations values are casted to two float vectors and then divided.
             * 
             * NOTE: Analogously as in `/` and `/=` operators values are casted before performing a division.
             * @param bV Divisor.
             * @return Reference to modified object.
             */
            UShort256& operator%=(const unsigned short& b) noexcept {
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
                v_first_half = _mm256_srli_si256(v_first_half, 2);
                __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bV = _mm256_set_epi16(0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b);
                __m256 bVf = _mm256_set1_ps(b);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bVf), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bVf), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);

                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                fresult = _mm256_sub_epi32(v_first_half, _mm256_mullo_epi32(bV, fresult));
                sresult = _mm256_sub_epi32(v_second_half, _mm256_mullo_epi32(bV, sresult));
                
                fresult = _mm256_slli_si256(fresult, 2);
                
                v = _mm256_or_si256(fresult, sresult);
                return *this;
            }

            UShort256 operator|(const UShort256& bV) const noexcept {
                return _mm256_or_si256(v, bV.v);
            }

            UShort256 operator|(const unsigned short& b) const noexcept {
                return _mm256_or_si256(v, _mm256_set1_epi16(b));
            }

            UShort256& operator|=(const UShort256& bV) noexcept {
                v = _mm256_or_si256(v, bV.v);
                return *this;
            }

            UShort256& operator|=(const unsigned short& b) noexcept {
                v = _mm256_or_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            UShort256 operator&(const UShort256& bV) const noexcept {
                return _mm256_and_si256(v, bV.v);
            }

            UShort256 operator&(const unsigned short& b) const noexcept {
                return _mm256_and_si256(v, _mm256_set1_epi16(b));
            }

            UShort256& operator&=(const UShort256& bV) noexcept {
                v = _mm256_and_si256(v, bV.v);
                return *this;
            }

            UShort256& operator&=(const unsigned short& b) noexcept {
                v = _mm256_and_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            UShort256 operator^(const UShort256& bV) const noexcept{
                return _mm256_xor_si256(v, bV.v);
            }

            UShort256 operator^(const unsigned short& b) const noexcept{
                return _mm256_xor_si256(v, _mm256_set1_epi16(b));
            }

            UShort256& operator^=(const UShort256& bV) noexcept{
                v = _mm256_xor_si256(v, bV.v);
                return *this;
            }

            UShort256& operator^=(const unsigned short& b) noexcept{
                v = _mm256_xor_si256(v, _mm256_set1_epi16(b));
                return *this;
            }

            /**
             * Performs left bitwise shift of corresponding values.
             * @param bV Second vector that specifies number of bits to shift (for each 16-bit value).
             * @return New value of `v` shifted by number of bits specfied in `bV`.
             */
            UShort256 operator<<(const UShort256& bV) const noexcept {
                #if (defined __AVX512BW__ && defined __AVX512VL__)
                    // If compiler is using AVX-512BW and AVX-512DQ use available function.
                    return _mm256_sllv_epi16(v, bV.v);
                #else
                    // Perform bitwise left on first half of elements.
                    __m256i halves = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                    __m256i bhalves = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);

                    __m256i first_res = _mm256_sllv_epi32(halves, bhalves);
                    // AND is used to get rid of unwanted bits that may happen as 32-bit mode is used.
                    first_res = _mm256_and_si256(first_res, constants::EPI16_CRATE_EPI32);

                    // Performs AND operation on the rest of values and shifts to right place.
                    halves = _mm256_srli_si256(_mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE), 2);
                    bhalves = _mm256_srli_si256(_mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE), 2);

                    // Performs the same operation as in first scenario
                    __m256i second_res = _mm256_sllv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, constants::EPI16_CRATE_EPI32);
                    second_res = _mm256_slli_si256(second_res, 2);

                    // Joins results together.
                    return _mm256_or_si256(first_res, second_res);
                #endif
            }

            UShort256 operator<<(const unsigned int& shift) const noexcept {
                return _mm256_slli_epi16(v, shift);
            }

            UShort256& operator<<=(const UShort256& bV) noexcept {
                #if (defined __AVX512BW__ && defined __AVX512VL__)
                    v = _mm256_sllv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                    __m256i bhalves = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);

                    __m256i first_res = _mm256_sllv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, constants::EPI16_CRATE_EPI32);

                    halves = _mm256_srli_si256(_mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE), 2);
                    bhalves = _mm256_srli_si256(_mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE), 2);

                    __m256i second_res = _mm256_sllv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, constants::EPI16_CRATE_EPI32);
                    second_res = _mm256_slli_si256(second_res, 2);

                    v = _mm256_or_si256(first_res, second_res);
                #endif
                return *this;
            }

            UShort256& operator<<=(const unsigned int& shift) noexcept {
                v = _mm256_slli_epi16(v, shift);
                return *this;
            }

            UShort256 operator>>(const UShort256& bV) const noexcept {
                #if (defined __AVX512BW__ && defined __AVX512VL__)
                    return _mm256_srlv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                    __m256i bhalves = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);

                    __m256i first_res = _mm256_srlv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, constants::EPI16_CRATE_EPI32);

                    halves = _mm256_srli_si256(_mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE), 2);
                    bhalves = _mm256_srli_si256(_mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE), 2);

                    __m256i second_res = _mm256_srlv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, constants::EPI16_CRATE_EPI32);
                    second_res = _mm256_slli_si256(second_res, 2);

                    return _mm256_or_si256(first_res, second_res);
                #endif
            }

            UShort256 operator>>(const unsigned int& shift) const noexcept{
                return _mm256_srli_epi16(v, shift);
            }

            UShort256& operator>>=(const UShort256& bV) noexcept {
                #if (defined __AVX512BW__ && defined __AVX512VL__)
                    v = _mm256_srlv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                    __m256i bhalves = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);

                    __m256i first_res = _mm256_srlv_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, constants::EPI16_CRATE_EPI32);

                    halves = _mm256_srli_si256(_mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE), 2);
                    bhalves = _mm256_srli_si256(_mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE), 2);

                    __m256i second_res = _mm256_srlv_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, constants::EPI16_CRATE_EPI32);
                    second_res = _mm256_slli_si256(second_res, 2);

                    v = _mm256_or_si256(first_res, second_res);
                #endif
                return *this;
            }

            UShort256& operator>>=(const unsigned int& shift) noexcept {
                v = _mm256_srli_epi16(v, shift);
                return *this;
            }

            UShort256 operator~() const noexcept{
                return _mm256_xor_si256(v, constants::ONES);
            }

            std::string str() const noexcept {
                std::string result = "UShort256(";
                unsigned short* iv = (unsigned short*)&v; 
                for(unsigned i{0}; i < 15; ++i)
                    result += std::to_string(iv[i]) + ", ";
                
                result += std::to_string(iv[15]);
                result += ")";
                return result;
            }
    };
};


#endif