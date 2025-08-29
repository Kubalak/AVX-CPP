#pragma once
#ifndef SHORT256_HPP__
#define SHORT256_HPP__

#include <array>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include "constants.hpp"

namespace avx {
    /**
     * Class providing vectorized version of `short`.
     * Can hold 16 individual `short` values.
     * Provides arithmetic and bitwise operators.
     * Provides comparison operators == !=.
     */
    class Short256 {
        private:
            __m256i v;
       
        public:

            /**
             * Number of individual values stored by object. This value can be used to iterate over elements.
            */
            static constexpr const int size = 16;

            /**
             * Type that is stored inside vector.
             */
            using storedType = short;

            /**
             * Default constructor. Sets zero to whole vector.
             */
            Short256() noexcept : v(_mm256_setzero_si256()){}


            /**
             * Initializes vector with value from other vector.
             * @param init Object which value will be copied.
             */
            Short256(const Short256& init) noexcept : v(init.v){}


            /**
             * Initializes vector but using `__m256i` type.
             * @param init Raw value to be set.
             */
            Short256(const __m256i& init) noexcept : v(init){}


            /**
             * Initialize vector with values read from an array.
             * @param init Array from which values will be copied.
             */
            Short256(const std::array<short, 16>& init) noexcept : v(_mm256_lddqu_si256((const __m256i*)init.data())){}


            /**
             * Initialize vector with values using pointer.
             * @param addr A valid address containing at least 16 `short` numbers.
             * @throws If in debug mode and `addr` is `nullptr` throws `std::invalid_argument`. Otherwise no exception will be thrown.
             */
            explicit Short256(const short* addr) N_THROW_REL {
                if(addr)
                    v = _mm256_lddqu_si256((const __m256i*)addr);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif
            }


            /**
             * Initializes all vector fields with single value.
             * @param b A literal value to be set.
             */
            explicit Short256(const short b) noexcept : v(_mm256_set1_epi16(b)){}

            Short256(std::initializer_list<short> init) {
                alignas(32) short init_v[size];
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
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param pSrc Pointer to memory from which to load data.
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void load(const short *pSrc) N_THROW_REL {
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
            void save(std::array<short, 16>& dest) const noexcept {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (16x `short`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void save(short *pDest) const N_THROW_REL {
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
             * @param pDest A valid pointer to a memory of at least 32 bytes (16x `short`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void saveAligned(short *pDest) const N_THROW_REL {
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
            short operator[](const unsigned int& index) const 
            #ifndef NDEBUG
                {
                    if(index > 15)
                        throw std::out_of_range("Range be within range 0-15! Got: " + std::to_string(index));
                    return ((short*)&v)[index];
                }
            #else
                noexcept { return ((short*)&v)[index & 15]; }
            #endif


            /**
             * Compares with second vector for equality.
             * @param bV Object to compare.
             * @returns `true` if all elements are equal or `false` if not.
             */
            bool operator==(const Short256 &bV) const noexcept {
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) != 0;
            }


            /**
             * Compares with value for equality.
             * @param b Value to compare.
             * @returns `true` if all elements are equal to passed value `false` if not.
             */
            bool operator==(const short b) const noexcept{
                __m256i bV = _mm256_set1_epi16(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) != 0;
            }


            /**
             * Compares with second vector for inequality.
             * @param bV Object to compare.
             * @returns `true` if any alement is not equal to corresponding element in `bV` otherwise `false`.
             */
            bool operator!=(const Short256 &bV) const noexcept{
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) == 0;
            }


            /**
             * Compares with value for inequality.
             * @param b Value
             * @returns `true` if any alement is not equal to corresponding element in `bV` otherwise `false`.
             */
            bool operator!=(const short b) const noexcept{
                __m256i bV = _mm256_set1_epi16(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) == 0;
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

            /**
             * Performs an integer division. Utilizes casting to `float` to compensate for lack of native integer division in AVX and AVX2.
             * 
             * @param bV Divisors vector.
             * @return Result of integer division with truncation.
             */
            Short256 operator/(const Short256& bV) const noexcept {
            #ifdef __AVX512F__
                __m512 first = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                __m512 second = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(bV.v));
                return _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(first, second)));
            #else
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m128i bv_first_half = _mm256_extracti128_si256(bV.v, 0);
                __m128i bv_second_half = _mm256_extracti128_si256(bV.v, 1);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_first_half));
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_second_half));

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                return _mm256_insert_epi64(combinedres, b1, 2);
            #endif
            }

             /**
             * Performs an integer division. 
             * 
             * @param bV Divisor value.
             * @return Result of integer division with truncation.
             */
            Short256 operator/(const short& b) const noexcept {
            #ifdef __AVX512F__
                __m512 first = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                __m512 second = _mm512_set1_ps(static_cast<float>(b));
                return _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(first, second)));
            #else
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);

                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m256 bV = _mm256_set1_ps(b);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                return _mm256_insert_epi64(combinedres, b1, 2);
            #endif
            }

            Short256& operator/=(const Short256& bV) noexcept {
            #ifdef __AVX512F__
                __m512 first = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                __m512 second = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(bV.v));
                v = _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(first, second)));
            #else
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m128i bv_first_half = _mm256_extracti128_si256(bV.v, 0);
                __m128i bv_second_half = _mm256_extracti128_si256(bV.v, 1);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_first_half));
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_second_half));

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                v = _mm256_insert_epi64(combinedres, b1, 2);
            #endif
                return *this;
            }
            
            Short256& operator/=(const short& b) noexcept {
            #ifdef __AVX512F__
                __m512 first = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                __m512 second = _mm512_set1_ps(static_cast<float>(b));
                v = _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(first, second)));
            #else
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);

                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m256 bV = _mm256_set1_ps(b);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                v = _mm256_insert_epi64(combinedres, b1, 2);
            #endif
                return *this;
            }

            /**
             * Performs a modulo operation.
             * 
             * NOTE: Analogously as in `/` and `/=` operators values are casted before performing a division.
             * @param bV Divisors vector.
             * @return Modulo result.
             */
            Short256 operator%(const Short256& bV) const noexcept {
            #ifdef __AVX512F__
                __m512 first = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                __m512 second = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(bV.v));
                __m256i result = _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(first, second)));
                return _mm256_sub_epi16(v, _mm256_mullo_epi16(bV.v, result));
            #else
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m128i bv_first_half = _mm256_extracti128_si256(bV.v, 0);
                __m128i bv_second_half = _mm256_extracti128_si256(bV.v, 1);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_first_half));
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_second_half));

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                combinedres = _mm256_insert_epi64(combinedres, b1, 2);  
                return _mm256_sub_epi16(v, _mm256_mullo_epi16(bV.v, combinedres));
            #endif
            }


            /**
             * Performs a modulo operation.
             * 
             * NOTE: Analogously as in `/` and `/=` operators values are casted before performing a division.
             * @param bV Divisor.
             * @return Modulo result.
             */
            Short256 operator%(const short& b) noexcept {
            #ifdef __AVX512F__
                __m512 first = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                __m512 second = _mm512_set1_ps(static_cast<float>(b));
                __m256i result = _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(first, second)));
                return _mm256_sub_epi16(v, _mm256_mullo_epi16(_mm256_set1_epi16(b), result));
            #else
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);

                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m256 bV = _mm256_set1_ps(b);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                combinedres = _mm256_insert_epi64(combinedres, b1, 2);  
                return _mm256_sub_epi16(v, _mm256_mullo_epi16(_mm256_set1_epi16(b), combinedres));
            #endif
            }


            Short256& operator%=(const Short256& bV) noexcept {
            #ifdef __AVX512F__
                __m512 first = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                __m512 second = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(bV.v));
                __m256i result = _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(first, second)));
                v = _mm256_sub_epi16(v, _mm256_mullo_epi16(bV.v, result));
            #else
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m128i bv_first_half = _mm256_extracti128_si256(bV.v, 0);
                __m128i bv_second_half = _mm256_extracti128_si256(bV.v, 1);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_first_half));
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(bv_second_half));

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                combinedres = _mm256_insert_epi64(combinedres, b1, 2);  
                v = _mm256_sub_epi16(v, _mm256_mullo_epi16(bV.v, combinedres));
            #endif
                return *this;
            }

            Short256& operator%=(const short& b) noexcept {
            #ifdef __AVX512F__
                __m512 first = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                __m512 second = _mm512_set1_ps(static_cast<float>(b));
                __m256i result = _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(_mm512_div_ps(first, second)));
                v = _mm256_sub_epi16(v, _mm256_mullo_epi16(_mm256_set1_epi16(b), result));
            #else
                __m128i v_first_half = _mm256_extracti128_si256(v, 0);
                __m128i v_second_half = _mm256_extracti128_si256(v, 1);

                __m256 v_fhalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_first_half));
                __m256 v_shalf_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v_second_half));

                __m256 bV = _mm256_set1_ps(b);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                __m256i combinedres = _mm256_packs_epi32(fresult, sresult);
                long long a2, b1, *vP;
                vP = (long long*)&combinedres;
                b1 = vP[1];
                a2 = vP[2];
                combinedres = _mm256_insert_epi64(combinedres, a2, 1);
                combinedres = _mm256_insert_epi64(combinedres, b1, 2);  
                v = _mm256_sub_epi16(v, _mm256_mullo_epi16(_mm256_set1_epi16(b), combinedres));
            #endif
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
                #if defined(__AVX512BW__) && defined(__AVX512VL__)
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

            Short256 operator<<(const unsigned int& shift) const noexcept {
                return _mm256_slli_epi16(v, shift);
            }

            Short256& operator<<=(const Short256& bV) noexcept {
                #if defined(__AVX512BW__) && defined(__AVX512VL__)
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

            Short256& operator<<=(const unsigned int& shift) noexcept {
                v = _mm256_slli_epi16(v, shift);
                return *this;
            }

            Short256 operator>>(const Short256& bV) const noexcept {
                #if defined(__AVX512BW__) && defined(__AVX512VL__)
                    return _mm256_srlv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                    halves = _mm256_slli_si256(halves, 2);
                    halves = _mm256_srai_epi32(halves, 16);
                    __m256i bhalves = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);
                    bhalves = _mm256_slli_si256(bhalves, 2);
                    bhalves = _mm256_srai_epi32(bhalves, 16);

                    __m256i first_res = _mm256_srav_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, constants::EPI16_CRATE_EPI32);

                    halves = _mm256_srai_epi32(_mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE), 16);
                    bhalves = _mm256_srai_epi32(_mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE), 16);

                    __m256i second_res = _mm256_srav_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, constants::EPI16_CRATE_EPI32);
                    second_res = _mm256_slli_si256(second_res, 2);

                    return _mm256_or_si256(first_res, second_res);
                #endif
            }

            Short256 operator>>(const unsigned int& shift) const noexcept{
                return _mm256_srai_epi16(v, shift);
            }

            Short256& operator>>=(const Short256& bV) noexcept {
                #if defined(__AVX512BW__) && defined(__AVX512VL__)
                    v = _mm256_srlv_epi16(v, bV.v);
                #else
                    __m256i halves = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
                    halves = _mm256_slli_si256(halves, 2);
                    halves = _mm256_srai_epi32(halves, 16);
                    __m256i bhalves = _mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32);
                    bhalves = _mm256_slli_si256(bhalves, 2);
                    bhalves = _mm256_srai_epi32(bhalves, 16);

                    __m256i first_res = _mm256_srav_epi32(halves, bhalves);
                    first_res = _mm256_and_si256(first_res, constants::EPI16_CRATE_EPI32);

                    halves = _mm256_srai_epi32(_mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE), 16);
                    bhalves = _mm256_srai_epi32(_mm256_and_si256(bV.v, constants::EPI16_CRATE_EPI32_INVERSE), 16);

                    __m256i second_res = _mm256_srav_epi32(halves, bhalves);
                    second_res = _mm256_and_si256(second_res, constants::EPI16_CRATE_EPI32);
                    second_res = _mm256_slli_si256(second_res, 2);

                    v = _mm256_or_si256(first_res, second_res);
                #endif
                return *this;
            }

            Short256& operator>>=(const unsigned int& shift) noexcept {
                v = _mm256_srai_epi16(v, shift);
                return *this;
            }

            Short256 operator~() const noexcept{
                return _mm256_xor_si256(v, constants::ONES);
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