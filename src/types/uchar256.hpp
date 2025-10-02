#pragma once
#ifndef UCHAR256_HPP__
#define UCHAR256_HPP__

#include <array>
#include <string>
#include <ostream>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include "constants.hpp"

namespace avx {
    /**
     * Class representing vectorized version of `unsigned char`.
     * It can hold 32 individual `unsigned char` variables.
     * Provides support for arithmetic and bitwise operators.
     * `str()` method returns stored data as string.
     * Supports printing directly to stream (cout).
     */
    class UChar256{
        // Internal vector containing stored values.
        __m256i v;

        public:
            
            /**
             * Number of individual values stored by object. This value can be used to iterate over elements.
            */
            static constexpr int size = 32;

            /**
             * Type that is stored inside vector.
             */
            using storedType = unsigned char;

            /**
             * Default constructor. Initializes vector with zeros.
             */
            UChar256() noexcept : v(_mm256_setzero_si256()){}

            /**
             * Initializes all vector fields with single value.
             * @param b A literal value to be set.
             */
            UChar256(const unsigned char init) noexcept : v(_mm256_set1_epi8(init)){}

            /**
             * Initializes vector but using `__m256i` type.
             * @param init Raw value to be set.
             */
            UChar256(const __m256i& init) noexcept : v(init){}
            
            /**
             * Initializes vector with value from other vector.
             * @param init Object which value will be copied.
             */
            UChar256(const UChar256& init) noexcept : v(init.v){}

            /**
             * Initializes object with first 32 bytes of data stored under `addr`.
             * Data does not need to be aligned to any specific boundary. 
             * 
             * @param pSrc Memory holding data (minimum 32 bytes).
             * @throws std::invalid_argument If in Debug and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            explicit UChar256(const unsigned char* pSrc){
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else                
            #endif                
                    v = _mm256_lddqu_si256((const __m256i*)pSrc);
            }
            
            /**
             * Initializes with first 32 bytes read from string. If `init` is less than 32 bytes long missing values will be set to 0.
             * 
             * @param init String containing initial data.
             */
            UChar256(const std::string& init) noexcept {
                if(init.size() >= 32)
                    v = _mm256_lddqu_si256((const __m256i*)init.data());
                else {
                    alignas(32) unsigned char initV[32];
                    memset(initV, 0, 32);
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
             * Initialize vector with values read from an array.
             * @param init Array from which values will be copied.
             */
            UChar256(const std::array<unsigned char, 32>& init) noexcept : v(_mm256_lddqu_si256((const __m256i*)init.data())){}

            UChar256(std::initializer_list<unsigned char> init) {
                alignas(32) unsigned char init_v[32];
                memset(init_v, 0, 32);
                if(init.size() < 32){
                    auto begin = init.begin();
                    for(int i{0}; i < init.size(); ++i) {
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                else {
                    auto begin = init.begin();
                    for(int i{0}; i < 32; ++i) {
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                v = _mm256_load_si256((const __m256i*)init_v);
            }

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param pSrc Pointer to memory from which to load data.
             * @throws std::invalid_argument If in Debug and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void load(const unsigned char *pSrc) {
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else                
            #endif                
                    v = _mm256_lddqu_si256((const __m256i*)pSrc);
            }

            /**
             * Saves data to destination in memory.
             * @param dest Reference to the list to which vector will be saved. Array doesn't need to be aligned to any specific boundary.
             */
            void save(std::array<unsigned char, 32>& dest) const noexcept {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (32x `unsigned char`).
             * @throws std::invalid_argument If in Debug and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void save(unsigned char *pDest) const {
            #ifndef NDEBUG
                if(!pDest)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else                
            #endif                
                    _mm256_storeu_si256((__m256i*)pDest, v);
            }

            /**
             * Saves data to destination in memory. The memory must be aligned at 32-byte boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (32x `unsigned char`).
             * @throws std::invalid_argument If in Debug and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void saveAligned(unsigned char *pDest) const {
            #ifndef NDEBUG
                if(!pDest)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else                
            #endif                
                    _mm256_store_si256((__m256i*)pDest, v);
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
             * @throws std::out_of_range If index is not within the correct range and using Debug mode. If not in Debug mode no exception will be thrown (bitwise AND ensures index stays withing bounds 0-31).
             */
            unsigned char operator[](const unsigned int& index) const 
            #ifndef NDEBUG
                {
                    if(index > 31)
                        throw std::out_of_range("Range be within range 0-31! Got: " + std::to_string(index));
                    return ((unsigned char*)&v)[index];
                }
            #else
                noexcept { return ((unsigned char*)&v)[index & 31]; }
            #endif


            bool operator==(const UChar256& bV) const noexcept {
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) != 0;
            }

            bool operator==(const char b) const noexcept {
                __m256i bV = _mm256_set1_epi8(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) != 0;
            }

            bool operator!=(const UChar256& bV) const noexcept {
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) == 0;
            }

            bool operator!=(const char b) const noexcept {
                __m256i bV = _mm256_set1_epi8(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) == 0;
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


            UChar256 operator*(const UChar256& bV) const noexcept {
            #ifdef __AVX512BW__
                return _mm512_cvtepi16_epi8(
                    _mm512_mullo_epi16(
                        _mm512_cvtepu8_epi16(v), 
                        _mm512_cvtepu8_epi16(bV.v)
                    )
                );
            #else
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
            #endif
            }


            UChar256 operator*(const char& b) const noexcept {
            #ifdef __AVX512BW__
                return _mm512_cvtepi16_epi8(
                    _mm512_mullo_epi16(
                        _mm512_cvtepu8_epi16(v), 
                        _mm512_set1_epi16(static_cast<unsigned short>(b))
                    )
                );
            #else
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
            #endif
            }


            UChar256& operator*=(const UChar256& bV) noexcept {
            #ifdef __AVX512BW__
                v = _mm512_cvtepi16_epi8(
                    _mm512_mullo_epi16(
                        _mm512_cvtepu8_epi16(v), 
                        _mm512_cvtepu8_epi16(bV.v)
                    )
                );
            #else
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
            #endif
                return *this;
            }


            UChar256& operator*=(const char& b) noexcept {
            #ifdef __AVX512BW__
                v = _mm512_cvtepi16_epi8(
                    _mm512_mullo_epi16(
                        _mm512_cvtepu8_epi16(v), 
                        _mm512_set1_epi16(static_cast<unsigned short>(b))
                    )
                );
            #else
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
            #endif
                return *this;
            }


            UChar256 operator/(const UChar256& bV) const noexcept {
            #ifdef __AVX512BW__
                __m512i first_16 = _mm512_cvtepu8_epi16(v);
                __m512i second_16 = _mm512_cvtepu8_epi16(bV.v); // No way of checking FP16 support lol

                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first_16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first_16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second_16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second_16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

                return _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
            #else
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bv_first_half = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);
                __m256i bv_second_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                v_first_half = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                bv_first_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);
                bv_second_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), constants::EPI8_CRATE_EPI32);

                bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 3);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                return _mm256_or_si256(half_res, shalf_res);
            #endif
            }

            UChar256 operator/(const unsigned char b) const noexcept {
            #ifdef __AVX512BW__
                __m512i first_16 = _mm512_cvtepu8_epi16(v);
                __m512i second_16 = _mm512_set1_epi16(static_cast<short>(b));

                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first_16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first_16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second_16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second_16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

                return _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
            #else
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256 bV = _mm256_set1_ps(static_cast<float>(b));

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                v_first_half = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 3);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                return _mm256_or_si256(half_res, shalf_res);
            #endif
            }

            UChar256 operator/=(const UChar256& bV) noexcept {
            #ifdef __AVX512BW__
                __m512i first_16 = _mm512_cvtepu8_epi16(v);
                __m512i second_16 = _mm512_cvtepu8_epi16(bV.v);

                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first_16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first_16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second_16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second_16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

                v = _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
            #else
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bv_first_half = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);
                __m256i bv_second_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                v_first_half = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                bv_first_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);
                bv_second_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), constants::EPI8_CRATE_EPI32);

                bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 3);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                v = _mm256_or_si256(half_res, shalf_res);
            #endif
                return *this;
            }

            UChar256 operator/=(const unsigned char b) noexcept {
            #ifdef __AVX512BW__
                __m512i first_16 = _mm512_cvtepu8_epi16(v);
                __m512i second_16 = _mm512_set1_epi16(static_cast<short>(b));
 
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first_16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first_16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second_16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second_16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

                v = _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
            #else
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256 bV = _mm256_set1_ps(static_cast<float>(b));

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                v_first_half = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 3);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                v = _mm256_or_si256(half_res, shalf_res);
            #endif
                return *this;
            }


            UChar256 operator%(const UChar256& bV) const noexcept {
            #if defined(__AVX512BW__) && defined(__AVX512F__)
                __m512i first_16 = _mm512_cvtepu8_epi16(v);
                __m512i second_16 = _mm512_cvtepu8_epi16(bV.v);

                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first_16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first_16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second_16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second_16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
                result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

                return _mm512_cvtepi16_epi8(_mm512_sub_epi8(first_16, _mm512_mullo_epi16(second_16, result)));
            #else
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bv_first_half = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);
                __m256i bv_second_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bv_fhalf_f));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_sub_epi8(v_first_half, fresult);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_sub_epi8(v_second_half, sresult);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                v_first_half = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                bv_first_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);
                bv_second_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), constants::EPI8_CRATE_EPI32);

                bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bv_fhalf_f));
                sresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_sub_epi8(v_first_half, fresult);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_sub_epi8(v_second_half, sresult);
                sresult = _mm256_slli_si256(sresult, 3);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                return _mm256_or_si256(half_res, shalf_res);
            #endif
            }

            UChar256 operator%(const unsigned char b) const noexcept {
                if(!b) return _mm256_setzero_si256();
            #if defined(__AVX512BW__) && defined(__AVX512F__)
                __m512i first_16 = _mm512_cvtepu8_epi16(v);
                __m512i second_16 = _mm512_set1_epi16(static_cast<unsigned short>(b));
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first_16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first_16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second_16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second_16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
                result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

                return _mm512_cvtepi16_epi8(_mm512_sub_epi8(first_16, _mm512_mullo_epi16(second_16, result)));
            #else
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256 bV = _mm256_set1_ps(static_cast<float>(b));

                __m256i fresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bV));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_sub_epi8(v_first_half, fresult);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_sub_epi8(v_second_half, sresult);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                v_first_half = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bV));
                sresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_sub_epi8(v_first_half, fresult);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_sub_epi8(v_second_half, sresult);
                sresult = _mm256_slli_si256(sresult, 3);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                return _mm256_or_si256(half_res, shalf_res);
            #endif
            }

            UChar256 operator%=(const UChar256& bV) noexcept {
            #if defined(__AVX512BW__) && defined(__AVX512F__)
                __m512i first_16 = _mm512_cvtepu8_epi16(v);
                __m512i second_16 = _mm512_cvtepu8_epi16(bV.v);

                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first_16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first_16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second_16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second_16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
                result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

                v = _mm512_cvtepi16_epi8(_mm512_sub_epi8(first_16, _mm512_mullo_epi16(second_16, result)));
            #else
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i bv_first_half = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);
                __m256i bv_second_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bv_fhalf_f));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_sub_epi8(v_first_half, fresult);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_sub_epi8(v_second_half, sresult);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                v_first_half = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                bv_first_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);
                bv_second_half = _mm256_and_si256(_mm256_srli_si256(bV.v, 3), constants::EPI8_CRATE_EPI32);

                bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bv_fhalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bv_fhalf_f));
                sresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bv_shalf_f), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_sub_epi8(v_first_half, fresult);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_sub_epi8(v_second_half, sresult);
                sresult = _mm256_slli_si256(sresult, 3);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                v = _mm256_or_si256(half_res, shalf_res);
            #endif
                return *this;
            }




            UChar256 operator%=(const unsigned char b) noexcept {
                if(!b) {
                    v = _mm256_setzero_si256();
                    return *this;
                }

            #if defined(__AVX512BW__) && defined(__AVX512F__)
                __m512i first_16 = _mm512_cvtepu8_epi16(v);
                __m512i second_16 = _mm512_set1_epi16(static_cast<unsigned short>(b));
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first_16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first_16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second_16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second_16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
                result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

                v = _mm512_cvtepi16_epi8(_mm512_sub_epi8(first_16, _mm512_mullo_epi16(second_16, result)));
            #else
                __m256i v_first_half = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                __m256i v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256 bV = _mm256_set1_ps(static_cast<float>(b));

                __m256i fresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bV));
                __m256i sresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_sub_epi8(v_first_half, fresult);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_sub_epi8(v_second_half, sresult);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                v_first_half = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                v_second_half = _mm256_and_si256(_mm256_srli_si256(v, 3), constants::EPI8_CRATE_EPI32);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_fhalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bV));
                sresult = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_round_ps(_mm256_div_ps(v_shalf_f, bV), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC), bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_sub_epi8(v_first_half, fresult);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_sub_epi8(v_second_half, sresult);
                sresult = _mm256_slli_si256(sresult, 3);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                v = _mm256_or_si256(half_res, shalf_res);
            #endif
                return *this;
            }


            UChar256 operator&(const UChar256& bV) const noexcept {
                return _mm256_and_si256(v, bV.v);
            }


            UChar256 operator&(const unsigned char& b) const noexcept {
                return _mm256_and_si256(v, _mm256_set1_epi8(b));
            }


            UChar256& operator&=(const UChar256& bV) noexcept {
                v = _mm256_and_si256(v, bV.v);
                return *this;
            }


            UChar256& operator&=(const unsigned char& b) noexcept {
                v = _mm256_and_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            UChar256 operator|(const UChar256& bV) const noexcept {
                return _mm256_or_si256(v, bV.v);
            }


            UChar256 operator|(const unsigned char& b) const noexcept {
                return _mm256_or_si256(v, _mm256_set1_epi8(b));
            }


            UChar256& operator|=(const UChar256& bV) noexcept {
                v = _mm256_or_si256(v, bV.v);
                return *this;
            }


            UChar256& operator|=(const unsigned char& b) noexcept {
                v = _mm256_or_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            UChar256 operator^(const UChar256& bV) const noexcept {
                return _mm256_xor_si256(v, bV.v);
            }


            UChar256 operator^(const unsigned char& b) const noexcept {
                return _mm256_xor_si256(v, _mm256_set1_epi8(b));
            }


            UChar256& operator^=(const UChar256& bV) noexcept {
                v = _mm256_xor_si256(v, bV.v);
                return *this;
            }


            UChar256& operator^=(const unsigned char& b) noexcept {
                v = _mm256_xor_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            UChar256 operator<<(const UChar256& bV) const noexcept {
                 #ifdef __AVX512BW__
                    __m512i fV = _mm512_cvtepu8_epi16(v);
                    __m512i sV = _mm512_cvtepu8_epi16(bV.v);
                    return _mm512_cvtepi16_epi8(_mm512_sllv_epi16(fV, sV));
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


            UChar256 operator<<(const unsigned int& b) const noexcept {
                #ifdef __AVX512BW__
                    __m512i fV = _mm512_cvtepu8_epi16(v);
                    return _mm512_cvtepi16_epi8(_mm512_slli_epi16(fV, static_cast<short>(b)));
                #else
                    __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                    __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                    fhalf = _mm256_slli_epi16(fhalf, b);
                    shalf = _mm256_slli_epi16(shalf, b);
                    fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                    shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                    return _mm256_or_si256(fhalf, shalf);
                #endif
            }


            UChar256& operator<<=(const UChar256& bV) noexcept {
                 #ifdef __AVX512BW__
                    __m512i fV = _mm512_cvtepu8_epi16(v);
                    __m512i sV = _mm512_cvtepu8_epi16(bV.v);
                    v = _mm512_cvtepi16_epi8(_mm512_sllv_epi16(fV, sV));
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


            UChar256& operator<<=(const unsigned int& b) noexcept {
                #ifdef __AVX512BW__
                    __m512i fV = _mm512_cvtepu8_epi16(v);
                    v = _mm512_cvtepi16_epi8(_mm512_slli_epi16(fV, static_cast<short>(b)));
                #else
                    __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                    __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                    fhalf = _mm256_slli_epi16(fhalf, b);
                    shalf = _mm256_slli_epi16(shalf, b);
                    fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                    shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                    v = _mm256_or_si256(fhalf, shalf);
                #endif
                return *this;
            }


            UChar256 operator>>(const UChar256& bV) const noexcept {
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


            UChar256 operator>>(const unsigned int& b) const noexcept {
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                fhalf = _mm256_srli_epi16(fhalf, b);
                shalf = _mm256_srli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                return _mm256_or_si256(fhalf, shalf);
            }


            UChar256& operator>>=(const UChar256& bV) noexcept {
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


            UChar256& operator>>=(const unsigned int& b) noexcept {
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                fhalf = _mm256_srli_epi16(fhalf, b);
                shalf = _mm256_srli_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                v = _mm256_or_si256(fhalf, shalf);
                return *this;
            }



            UChar256 operator~() const noexcept {
                return _mm256_xor_si256(v, constants::ONES);
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

           friend std::ostream& operator<<(std::ostream& os, const UChar256& a) {
                alignas(32) unsigned char tmp[33];
                tmp[32] = '\0';

                _mm256_store_si256((__m256i*)tmp, a.v);
                
                os << tmp;
                return os;
            }


    };
}

#endif