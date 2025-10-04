#pragma once
#ifndef UINT256_HPP__
#define UINT256_HPP__

#include <set>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <immintrin.h>
#include "constants.hpp"
#include "../misc/simd_ext_gcc.h"

namespace avx {
    /**
     * Class providing vectorized version of `unsigned int`.
     * It can hold 8 individual values.
     * Supports arithmetic and bitwise operators.
     * Provides comparison operators == !=. 
     */
    class UInt256 {
        private:
            __m256i v; 
        
        public:
            
            /**
             * Number of individual values stored by object. This value can be used to iterate over elements.
            */
            static constexpr int size = 8;

            /**
             * Type that is stored inside vector.
             */
            using storedType = unsigned int;

            /**
             * Default constructor.
             * Initializes vector with zeros using `_mm256_setzero_si256()`.
             */
            UInt256():v(_mm256_setzero_si256()) {}

            /** Initializes vector by loading data from memory (via `_mm256_lddq_si256`). 
             * @param pSrc Valid memory addres of minimal size of 256-bits (32 bytes).
             * @throws std::invalid_argument If in Debug and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
            */
            UInt256(const unsigned int* pSrc) {
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                    v = _mm256_lddqu_si256((const __m256i*)pSrc);
            }

            /**
             * Fills vector with passed value using `_mm256_set1_epi32()`.
             * @param init Value to be set.
             */
            UInt256(const unsigned int init) noexcept : v(_mm256_set1_epi32(init)) {}

            /**
             * Just sets vector value using passed `__m256i`.
             * @param init Vector value to be set.
             */
            UInt256(__m256i init) noexcept : v(init) {}

            /**
             * Set value using reference.
             * @param init Reference to object which value will be copied.
             */
            UInt256(UInt256& init) noexcept : v(init.v) {}

            /**
             * Set value using const reference.
             * @param init Const reference to object which value will be copied.
             */
            UInt256(const UInt256& init) noexcept : v(init.v){};

            /** Sets vector values using array of unsigned integers.
             * 
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned integers which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned int, 8> init) noexcept :
                v(_mm256_lddqu_si256((const __m256i*)init.data()))
            {}

            /** Sets vector values using array of unsigned shorts.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned shorts which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned short, 8> init) noexcept :
                v(_mm256_set_epi32(
                    init[0], 
                    init[1], 
                    init[2], 
                    init[3], 
                    init[4], 
                    init[5], 
                    init[6], 
                    init[7]
                    )
                )
            {}

            /** Sets vector values using array of unsigned chars.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned chars which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned char, 8> init) noexcept :
                v(_mm256_set_epi32(
                    init[0], 
                    init[1], 
                    init[2], 
                    init[3], 
                    init[4], 
                    init[5], 
                    init[6], 
                    init[7]
                    )
                )
            {}

            /** Sets vector values using array of unsigned integers.
             * If list is longer than 8 other values will be ignored.
             * If the list contains fewer than 8 elements other vector fields will be set to 0.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Initlizer list containing unsigned integers which values will be assigned to vector fields.
             */
            UInt256(std::initializer_list<unsigned int> init) noexcept {
                alignas(32) unsigned int init_v[size];
                std::memset(init_v, 0, 32);
                if(init.size() < size){
                    auto begin = init.begin();
                    for(int i{0}; i < init.size(); ++i){
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                else {
                    auto begin = init.begin();
                    for(int i{0}; i < size; ++i){
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                v = _mm256_load_si256((const __m256i*)init_v);
            }

            /**
             * Get underlying `__m256i` vector.
             * @returns Copy of internal `__m256i` vector.
             */
            __m256i get() const noexcept {return v;}

            /**
             * Set underlying `__m256i` vector value.
             * @param val Vector which value will be copied.
             */
            void set(__m256i val) noexcept {v = val;}

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param pSrc Pointer to memory from which to load data.
             * @throws std::invalid_argument If in Debug and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void load(const unsigned int *pSrc) {
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
            void save(std::array<unsigned int, 8>& dest) const noexcept {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (8x `unsigned int`).
             * @throws std::invalid_argument If in Debug and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void save(unsigned int *pDest) const {
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
             * @param pDest A valid pointer to a memory of at least 32 bytes (8x `unsigned int`).
             * @throws std::invalid_argument If in Debug and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void saveAligned(unsigned int *pDest) const {
            #ifndef NDEBUG
                if(!pDest)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                _mm256_store_si256((__m256i*)pDest, v);
            }

            /**
             * Compares vectors for equality.
             * @param bV Second vector.
             * @returns true if all values in both vectors are equal, false if any value doesn't match.
             */
            bool operator==(const UInt256 &bV) const {
                _mm256_zeroall();
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) != 0;
            }

            /**
             * Compares vector to scalar for equality.
             * @param b Value to compare.
             * @returns true if all values in vector are equal to `b`, otherwise false.
             */
            bool operator==(const int b) const {
                _mm256_zeroall();
                __m256i bV = _mm256_set1_epi32(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) != 0;
            }

            /**
             * Compares vectors for inequality.
             * @param bV Second vector.
             * @returns true if ANY value is different between vectors.
             */
            bool operator!=(const UInt256 &bV) const {
                _mm256_zeroall();
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) == 0;
            }

            /**
             * Compares vector to scalar for inequality.
             * @param b Value to compare.
             * @returns true if ANY value in vector is different than `b`, otherwise false.
             */
            bool operator!=(const int b) const {
                _mm256_zeroall();
                __m256i bV = _mm256_set1_epi32(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) == 0;
            }

            unsigned int operator[](const unsigned int index) const 
            #ifndef NDEBUG
                {
                    if(index > 7) 
                        throw std::out_of_range("Range be within range 0-7! Got: " + std::to_string(index));
                    
                    return ((unsigned int*)&v)[index];
                }
            #else
                noexcept { return ((unsigned int*)&v)[index & 7]; }
            #endif

            /**
             * Adds values from other vector and returns new vector.
             * @param b Second vector.
             * @return UInt256 New vector being a sum of this vector and `b`.
             */
            UInt256 operator+(const UInt256& b) const noexcept {return _mm256_add_epi32(v, b.v);}

            /**
             * Adds single value across all vector fields.
             * @param b Value to add to vector.
             * @return UInt256 New vector being a sum of this vector and `b`.
             */
            UInt256 operator+(const unsigned int b) const noexcept {return _mm256_add_epi32(v, _mm256_set1_epi32(b));}

            /**
             * Adds two vectors together and stores result inside original vector.
             * @param b Second vector.
             * @returns Reference to same vector after adding `b` to vector.
             */
            UInt256& operator+=(const UInt256& b) noexcept {
                v = _mm256_add_epi32(v, b.v);
                return *this;
            }

            /**
             * Adds scalar to vector and stores result inside original vector.
             * @param b Scalar to be added.
             * @returns Reference to same vector after adding `b` to vector.
             */
            UInt256& operator+=(const unsigned int b) noexcept {
                v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
                return *this;
            }

            /**
             * Subtracts values from vector.
             * @param b Second vector.
             * @return UInt256 New vector being result of subtracting `b` from vector.
             */
            UInt256 operator-(const UInt256& b) const noexcept {return _mm256_sub_epi32(v, b.v);}

            /**
             * Subtracts a single value from all vector fields.
             * @param b Value to subtract from vector.
             * @return UInt256 New vector being result of subtracting `b` from vector.
             */
            UInt256 operator-(const unsigned int b) const noexcept {return _mm256_sub_epi32(v, _mm256_set1_epi32(b));}

            /**
             * Subtracts two vectors and stores result inside original vector.
             * @param b Second vector.
             * @returns Reference to same vector after subtracting `b` from vector.
             */
            UInt256& operator-=(const UInt256& b) noexcept {
                v = _mm256_sub_epi32(v,b.v);
                return *this;
            }

            /**
             * Subtracts scalar from vector and stores result inside original vector.
             * @param b Scalar to be subtracted.
             * @returns Reference to same vector after subtracting `b` from vector.
             */
            UInt256& operator-=(const unsigned int b) noexcept {
                v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
                return *this;
            }

            /**
             * Multiplies two vectors.
             * @param b Second vector.
             * @return UInt256 New vector being result of multiplying vector by `b`.
             */
            UInt256 operator*(const UInt256& b) const noexcept {
                __m256i first = _mm256_mul_epu32(v, b.v);
                __m256i av = _mm256_srli_si256(v, sizeof(unsigned int));
                __m256i bv = _mm256_srli_si256(b.v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(av, bv);

                // Full AVX version
                second = _mm256_and_si256(second, constants::EPI32_CRATE_EPI64);
                first = _mm256_and_si256(first, constants::EPI32_CRATE_EPI64);
                second = _mm256_slli_si256(second, sizeof(unsigned int));

                return _mm256_or_si256(first, second);
            }

            /**
             * Multiplies all vector fields by a single value.
             * @param b Value to multiply by.
             * @return UInt256 New vector being result of multiplying vector by `b`.
             */
            UInt256 operator*(const unsigned int b) const noexcept {
                __m256i bv = _mm256_set1_epi32(b);
                __m256i first = _mm256_mul_epu32(v, bv);
                __m256i av = _mm256_srli_si256(v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(av, bv);

                second = _mm256_and_si256(second, constants::EPI32_CRATE_EPI64);
                first = _mm256_and_si256(first, constants::EPI32_CRATE_EPI64);
                second = _mm256_slli_si256(second, sizeof(unsigned int));


                return _mm256_or_si256(first, second);
            }

            /**
             * Multiplies two vectors and stores result inside original vector.
             * @param b Second vector.
             * @returns Reference to same vector after multiplying by `b`.
             */
            UInt256& operator*=(const UInt256& b) noexcept {
                __m256i first = _mm256_mul_epu32(v, b.v);
                v = _mm256_srli_si256(v, sizeof(unsigned int));
                
                __m256i bv = _mm256_srli_si256(b.v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(v, bv);

                second = _mm256_and_si256(second, constants::EPI32_CRATE_EPI64);
                first = _mm256_and_si256(first, constants::EPI32_CRATE_EPI64);
                second = _mm256_slli_si256(second, sizeof(unsigned int));

                v =_mm256_or_si256(first, second);
                return *this;
            }

            /**
             * Multiplies vector by scalar and stores result inside original vector.
             * @param b Scalar to multiply by.
             * @returns Reference to same vector after multiplying by `b`.
             */
            UInt256& operator*=(const unsigned int b) noexcept {
                __m256i bv = _mm256_set1_epi32(b);
                __m256i first = _mm256_mul_epu32(v, bv);

                v = _mm256_srli_si256(v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(v, bv);

                second = _mm256_and_si256(second, constants::EPI32_CRATE_EPI64);
                first = _mm256_and_si256(first, constants::EPI32_CRATE_EPI64);
                second = _mm256_slli_si256(second, sizeof(unsigned int));
                
                v =_mm256_or_si256(first, second);

                return *this;
            }

            /**
             * Divides two vectors.
             * @param bV Second vector (divisor).
             * @return UInt256 New vector being result of dividing vector by `bV`.
             */
            UInt256 operator/(const UInt256& bV) const noexcept {
            #ifdef __AVX512F__
                __m512d first = _mm512_cvtepu32_pd(v);
                __m512d second = _mm512_cvtepu32_pd(bV.v);

                return _mm512_cvttpd_epu32(_mm512_div_pd(first,second));
            #else
                return _mm256_div_epu32(v, bV.v);
            #endif
            }

            /* NOTE: Faster than using only div function however ends up failing on VERY edge cases due to float casting.
             int div_lt_limit = _mm256_testz_si256(b.v, constants::EPI32_SIGN); // b.v AND 0b1000'0000...0b1000'0000 == 0
                    int num_lt_limit = _mm256_testz_si256(v, constants::EPI32_SIGN);
                    switch((num_lt_limit << 1) | div_lt_limit) {
                        case 0b00: // If both aV an bV are greater than 2^31                           
                        case 0b01: // If only nominal is greater than 2^31                            
                            return _mm256_div_epu32(v, b.v);
                        case 0b10: // If only nominal is smaller than 2^31
                            {  
                                __m256i msbVal = _mm256_and_si256(v, constants::EPI32_SIGN);
                                __m256i restVal = _mm256_xor_si256(v, msbVal);

                                __m256 divisor = _mm256_cvtepi32_ps(b.v);

                                __m256 result = _mm256_add_ps(_mm256_div_ps(_mm256_cvtepi32_ps(msbVal), divisor), _mm256_div_ps(_mm256_cvtepi32_ps(restVal), divisor));

                                __m256i subZeroMask = _mm256_castps_si256(_mm256_cmp_ps(result, _mm256_setzero_ps(), _CMP_GT_OS));

                                return _mm256_and_si256(_mm256_cvtps_epi32(result), subZeroMask);
                            }
                        case 0b11: // If both are smaller (need to check for float approximation)
                            __m256i approxMask = _mm256_cmpgt_epi32(v, b.v);
                            approxMask = _mm256_or_si256(_mm256_cmpeq_epi32(v, b.v), approxMask);   
                            return _mm256_and_si256(
                                approxMask, 
                                _mm256_cvttps_epi32(
                                    _mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(b.v))
                                )
                            );
                    }

                return _mm256_setzero_si256();
             */

            /**
             * Divides all vector fields by a single value.
             * @param b Value (divisor).
             * @return UInt256 New vector being result of dividing vector by `b`.
             */
            UInt256 operator/(const unsigned int b) const noexcept {
                if(!b) return _mm256_setzero_si256();
                if(b < 2)
                    return v;
            #ifdef __AVX512F__
                __m512d first = _mm512_cvtepu32_pd(v);
                __m512d second = _mm512_set1_pd(static_cast<double>(b));

                return _mm512_cvttpd_epu32(_mm512_div_pd(first,second));
            #else
                return _mm256_div_epu32(v, _mm256_set1_epi32(b));
            #endif
            }

            /**
             * Divides two vectors and stores result inside original vector.
             * @param bV Second vector (divisor).
             * @returns Reference to same vector after dividing by `bV`.
             */
            UInt256& operator/=(const UInt256& bV) noexcept {
            #ifdef __AVX512F__
                __m512d first = _mm512_cvtepu32_pd(v);
                __m512d second = _mm512_cvtepu32_pd(bV.v);

                v = _mm512_cvttpd_epu32(_mm512_div_pd(first,second));
            #else
                v = _mm256_div_epu32(v, bV.v);
            #endif
                return *this;
            }

            /**
             * Divides vector by scalar and stores result inside original vector.
             * @param b Scalar value (divisor).
             * @returns Reference to same vector after dividing by `b`.
             */
            UInt256& operator/=(const unsigned int b) noexcept {
                if(!b) {
                    v = _mm256_setzero_si256();
                    return *this;
                }
                    
                if(b < 2)
                    return *this;
            #ifdef __AVX512F__
                __m512d first = _mm512_cvtepu32_pd(v);
                __m512d second = _mm512_set1_pd(static_cast<double>(b));

                v = _mm512_cvttpd_epu32(_mm512_div_pd(first,second));
            #else
                v =_mm256_div_epu32(v, _mm256_set1_epi32(b));
            #endif
                return *this;
            }
            
            /**
             * Calculates element-wise modulo of two vectors.
             * @param bV Second vector (divisor).
             * @return UInt256 New vector being result of modulo operation.
             */
            UInt256 operator%(const UInt256& bV) const noexcept {        
            #ifdef __AVX512F__
                __m512d first = _mm512_cvtepu32_pd(v);
                __m512d second = _mm512_cvtepu32_pd(bV.v);

                __m512i result = _mm512_cvttpd_epu64(_mm512_div_pd(first,second));
                return _mm256_sub_epi32(v, _mm512_cvtepi64_epi32(_mm512_mullo_epi64(result, _mm512_cvtepu32_epi64(bV.v))));
            #else
                __m256i divisor = _mm256_div_epu32(v, bV.v);

                __m256i first = _mm256_mul_epu32(bV.v, divisor);
                divisor = _mm256_srli_si256(divisor, sizeof(unsigned int));
                
                __m256i bv = _mm256_srli_si256(bV.v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(divisor, bv);

                second = _mm256_and_si256(second, constants::EPI32_CRATE_EPI64);
                first = _mm256_and_si256(first, constants::EPI32_CRATE_EPI64);
                second = _mm256_slli_si256(second, sizeof(unsigned int));

                __m256i multiplied = _mm256_or_si256(first, second);

                return _mm256_sub_epi32(v, multiplied);
            #endif
            }

            /**
             * Calculates element-wise modulo of vector and scalar.
             * @param b Value (divisor).
             * @return UInt256 New vector being result of modulo operation.
             */
            UInt256 operator%(const unsigned int b) const noexcept {
                if(b > 1) {
            #ifdef __AVX512F__
                    __m512d first = _mm512_cvtepu32_pd(v);
                    __m512d second = _mm512_set1_pd(static_cast<double>(b));

                    __m512i result = _mm512_cvttpd_epu64(_mm512_div_pd(first,second));
                    return _mm256_sub_epi32(v, _mm512_cvtepi64_epi32(_mm512_mullo_epi64(result, _mm512_set1_epi64(static_cast<int64_t>(b)))));
            #else
                    __m256i bV = _mm256_set1_epi32(b);
                    __m256i divisor = _mm256_div_epu32(v, bV);

                    __m256i first = _mm256_mul_epu32(bV, divisor);
                    divisor = _mm256_srli_si256(divisor, sizeof(unsigned int));
                    
                    __m256i second = _mm256_mul_epu32(divisor, bV);

                    second = _mm256_and_si256(second, constants::EPI32_CRATE_EPI64);
                    first = _mm256_and_si256(first, constants::EPI32_CRATE_EPI64);
                    second = _mm256_slli_si256(second, sizeof(unsigned int));

                    __m256i multiplied = _mm256_or_si256(first, second);

                    return _mm256_sub_epi32(v, multiplied);
            #endif
                }
                
                return  _mm256_setzero_si256();
            }

            /**
             * Performs modulo operation. It does so by dividing vectors, multiplying result and subtracting from vector.
             * @param bV Second modulo operand (divisor)
             * @return Reference to the original vector holding modulo operation results.
             */
            UInt256& operator%=(const UInt256& bV) noexcept {
            #ifdef __AVX512F__
                __m512d first = _mm512_cvtepu32_pd(v);
                __m512d second = _mm512_cvtepu32_pd(bV.v);

                __m512i result = _mm512_cvttpd_epu64(_mm512_div_pd(first,second));
                v = _mm256_sub_epi32(v, _mm512_cvtepi64_epi32(_mm512_mullo_epi64(result, _mm512_cvtepu32_epi64(bV.v))));
            #else
                __m256i divisor = _mm256_div_epu32(v, bV.v);

                __m256i first = _mm256_mul_epu32(bV.v, divisor);
                divisor = _mm256_srli_si256(divisor, sizeof(unsigned int));
                
                __m256i bv = _mm256_srli_si256(bV.v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(divisor, bv);

                second = _mm256_and_si256(second, constants::EPI32_CRATE_EPI64);
                first = _mm256_and_si256(first, constants::EPI32_CRATE_EPI64);
                second = _mm256_slli_si256(second, sizeof(unsigned int));

                __m256i multiplied = _mm256_or_si256(first, second);

                v = _mm256_sub_epi32(v, multiplied);
            #endif
                return *this;
            }

            /**
             * Performs modulo operation. It does so by dividing vectors, multiplying result and subtracting from vector.
             * @param b Second modulo operand (divisor)
             * @return Reference to the original vector holding modulo operation results.
             */
            UInt256& operator%=(const unsigned int b) noexcept {
                if(b) {
            #ifdef __AVX512F__
                    __m512d first = _mm512_cvtepu32_pd(v);
                    __m512d second = _mm512_set1_pd(static_cast<double>(b));

                    __m512i result = _mm512_cvttpd_epu64(_mm512_div_pd(first,second));
                    v = _mm256_sub_epi32(v, _mm512_cvtepi64_epi32(_mm512_mullo_epi64(result, _mm512_set1_epi64(static_cast<int64_t>(b)))));
            #else
                    __m256i bV = _mm256_set1_epi32(b);
                    __m256i divisor = _mm256_div_epu32(v, bV);

                    __m256i first = _mm256_mul_epu32(bV, divisor);
                    divisor = _mm256_srli_si256(divisor, sizeof(unsigned int));
                    
                    __m256i second = _mm256_mul_epu32(divisor, bV);

                    second = _mm256_and_si256(second, constants::EPI32_CRATE_EPI64);
                    first = _mm256_and_si256(first, constants::EPI32_CRATE_EPI64);
                    second = _mm256_slli_si256(second, sizeof(unsigned int));

                    __m256i multiplied = _mm256_or_si256(first, second);

                    v = _mm256_sub_epi32(v, multiplied);
            #endif
                }
                else 
                    v =  _mm256_setzero_si256();
                return *this;
            }

            /**
             * Bitwise XOR operator.
             * @param bV Second vector.
             * @return UInt256 New vector being result of bitwise XOR with `bV`.
             */
            UInt256 operator^(const UInt256& bV) const noexcept {
                return _mm256_xor_si256(v, bV.v);
            }

            /**
             * Bitwise XOR operator with scalar.
             * @param b Value to XOR with.
             * @return UInt256 New vector being result of bitwise XOR with `b`.
             */
            UInt256 operator^(const unsigned int b) const noexcept {
                return _mm256_xor_si256(v, _mm256_set1_epi32(b));
            }

            /**
             * Bitwise XOR assignment operator.
             * Applies bitwise XOR between this vector and the given vector, storing the result in this vector.
             * @param b Second vector.
             * @return Reference to the modified object.
             */
            UInt256& operator^=(const UInt256& b) noexcept {
                v = _mm256_xor_si256(v, b.v);
                return *this;
            }

            /**
             * Bitwise XOR assignment operator.
             * Applies bitwise XOR between this vector and the given integer value, storing the result in this vector.
             * @param b Integer value.
             * @return Reference to the modified object.
             */
            UInt256& operator^=(const unsigned int b) noexcept {
                v = _mm256_xor_si256(v, _mm256_set1_epi32(b));
                return *this;
            }

            /**
             * Bitwise OR operator.
             * @param b Second vector.
             * @return UInt256 New vector being result of bitwise OR with `b`.
             */
            UInt256 operator|(const UInt256& b) const noexcept {
                return _mm256_or_si256(v, b.v);
            }

            /**
             * Bitwise OR operator with scalar.
             * @param b Value to OR with.
             * @return UInt256 New vector being result of bitwise OR with `b`.
             */
            UInt256 operator|(const unsigned int b) const noexcept {
                return _mm256_or_si256(v, _mm256_set1_epi32(b));
            }

            /**
             * Bitwise OR assignment operator.
             * Applies bitwise OR between this vector and the given vector, storing the result in this vector.
             * @param b Second vector.
             * @return Reference to the modified object.
             */
            UInt256& operator|=(const UInt256& b) noexcept {
                v = _mm256_or_si256(v, b.v);
                return *this;
            }

            /**
             * Bitwise OR assignment operator.
             * Applies bitwise OR between this vector and the given integer value, storing the result in this vector.
             * @param b Integer value.
             * @return Reference to the modified object.
             */
            UInt256& operator|=(const unsigned int b) noexcept {
                v = _mm256_or_si256(v, _mm256_set1_epi32(b));
                return *this;
            }

            /**
             * Bitwise AND operator.
             * @param b Second vector.
             * @return UInt256 New vector being result of bitwise AND with `b`.
             */
            UInt256 operator&(const UInt256& b) const noexcept {
                return _mm256_and_si256(v, b.v);
            }

            /**
             * Bitwise AND operator with scalar.
             * @param b Value to AND with.
             * @return UInt256 New vector being result of bitwise AND with `b`.
             */
            UInt256 operator&(const unsigned int b) const noexcept {
                return _mm256_and_si256(v, _mm256_set1_epi32(b));
            }

            /**
             * Bitwise AND assignment operator.
             * Applies bitwise AND between this vector and the given vector, storing the result in this vector.
             * @param b Second vector.
             * @return Reference to the modified object.
             */
            UInt256& operator&=(const UInt256& b) noexcept {
                v = _mm256_and_si256(v, b.v);
                return *this;
            }

            /**
             * Bitwise AND assignment operator.
             * Applies bitwise AND between this vector and the given integer value, storing the result in this vector.
             * @param b Integer value.
             * @return Reference to the modified object.
             */
            UInt256& operator&=(const unsigned int b) noexcept {
                v = _mm256_and_si256(v, _mm256_set1_epi32(b));
                return *this;
            }

            /**
             * Bitwise left shift operator (element-wise).
             * @param b Vector containing number of bits for which each corresponding element should be shifted.
             * @return UInt256 New vector after left shift.
             */
            UInt256 operator<<(const UInt256& b) const noexcept {
                return _mm256_sllv_epi32(v, b.v);
            }

            /**
             * Bitwise left shift operator by scalar.
             * @param b Number of bits by which values should be shifted.
             * @return UInt256 New vector after left shift.
             */
            UInt256 operator<<(const unsigned int b) const noexcept {
                return _mm256_slli_epi32(v, b);
            }

            /**
             * Bitwise left shift assignment operator (element-wise).
             * @param b Vector containing number of bits for which each corresponding element should be shifted.
             * @returns Reference to modified object.
             */
            UInt256& operator<<=(const UInt256& b) noexcept {
                v = _mm256_sllv_epi32(v, b.v);
                return *this;
            }

            /**
             * Bitwise left shift assignment operator by scalar.
             * @param b Number of bits by which values should be shifted.
             * @returns Reference to modified object.
             */
            UInt256& operator<<=(const unsigned int b) noexcept {
                v = _mm256_slli_epi32(v,b);
                return *this;
            }

            /**
             * Bitwise right shift operator (element-wise, logical shift).
             * @param b Vector containing number of bits for which each corresponding element should be shifted.
             * @return UInt256 New vector after right shift.
             */
            UInt256 operator>>(const UInt256& b) const noexcept {
                return _mm256_srlv_epi32(v, b.v);
            }

            /**
             * Bitwise right shift operator by scalar (logical shift).
             * @param b Number of bits by which values should be shifted.
             * @return UInt256 New vector after right shift.
             */
            UInt256 operator>>(const unsigned int b) const noexcept {
                return _mm256_srli_epi32(v, b);
            }

            /**
             * Bitwise right shift assignment operator (element-wise, logical shift).
             * @param b Vector containing number of bits for which each corresponding element should be shifted.
             * @returns Reference to modified object.
             */
            UInt256& operator>>=(const UInt256& b) noexcept {
                v = _mm256_srlv_epi32(v, b.v);
                return *this;
            }
            /**
             * Bitwise right shift assignment operator by scalar (logical shift).
             * @param b Number of bits by which values should be shifted.
             * @returns Reference to modified object.
             */
            UInt256& operator>>=(const unsigned int b) noexcept {
                v = _mm256_srli_epi32(v, b);
                return *this;
            }
            

            /**
             * Bitwise NOT operator.
             * @return UInt256 New vector with all bits inverted.
             */
            UInt256 operator~() const noexcept { return _mm256_xor_si256(v, constants::ONES);}


            /**
             * Return string representation of vector.
             * 
             * @returns String representation of vector.
             */
            std::string str() const noexcept {
                std::string result = "UInt256(";
                unsigned int* iv = (unsigned int*)&v; 
                for(unsigned i{0}; i < 7; ++i)
                    result += std::to_string(iv[i]) + ", ";
                
                result += std::to_string(iv[7]);
                result += ")";
                return result;
            }
        };
};


#endif