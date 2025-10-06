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
             * @param pSrc Memory holding data (minimum 32 bytes).
             * @throws std::invalid_argument If in Debug and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            explicit Char256(const char* pSrc) {
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else                
            #endif                
                    v = _mm256_lddqu_si256((const __m256i*)pSrc);
                }
            
            /**
             * Initializes object using first `count` bytes from `pSrc` or 32 bytes if `count` > 32.
             * Data does not need to be aligned to 32 byte boundary.
             * 
             * @param pSrc Valid pointer from which data will be loaded.
             * @param count Number of bytes held by `pSrc`. For less than 32 remaining bytes will be filled with 0's. Otherwise first 32 bytes will be used.
             * @throws std::invalid_argument if in debug mode and `pSrc` is `nullptr`. If in release, then no pointer checks are performed to improve performance.
            */
            Char256(const char* pSrc, unsigned int count) {
            #ifndef NDEBUG
                if(!pSrc) 
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif
                if(count >= 32)
                    v = _mm256_lddqu_si256((const __m256i*)pSrc);
                else {
                    char tmp[32];
                    #if defined(_MSC_VER) || defined(__STDC_WANT_LIB_EXT1__)
                        memcpy_s(tmp, sizeof(tmp), pSrc, count);
                    #else
                        memcpy(tmp, pSrc, count);
                    #endif
                    memset(tmp + count, 0, 32 - count);
                    v = _mm256_lddqu_si256((const __m256i*)tmp);
                }
            }

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
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary.
             * @param pSrc Pointer to memory from which to load data.
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. If in Release mode no pointer checks are performed to improve performance.
             */
            void load(const char *pSrc) {
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
            void save(std::array<char, 32>& dest) const noexcept {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (32x `char`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve performance.
             */
            void save(char *pDest) const {
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
             * @param pDest A valid pointer to a memory of at least 32 bytes (32x `char`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve performance.
             */
            void saveAligned(char *pDest) const {
            #ifndef NDEBUG
                if(!pDest)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                _mm256_store_si256((__m256i*)pDest, v);
            }

            /**
             * Get the internal vector value.
             * @return The value of `__m256i` vector.
             */
            __m256i get() const noexcept{return v;}

            /**
             * Set the internal vector value.
             * @param value New value to be set.
             */
            void set(const __m256i value) noexcept {v = value;}

            /**
             * Indexing operator. Returns element under given `index`.
             * Does not support value assignment through this method (e.g. aV[0] = 1 won't work).
             * @param index Position of desired element between 0 and 31.
             * @return Value of underlying element.
             * @throws If index is not within the correct range and build type is debug `std::out_of_range` will be thrown. Otherwise bitwise AND will prevent index to be out of range. Side effect is that only 5 LSBs are used from `index`.
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
            #if defined(__AVX512F__) || defined(__AVX512VL__)
                _mm256_zeroupper();
            #endif
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
            #if defined(__AVX512F__) || defined(__AVX512VL__)
                _mm256_zeroupper();
            #endif
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
            #if defined(__AVX512F__) || defined(__AVX512VL__)
                _mm256_zeroupper();
            #endif
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
            #if defined(__AVX512F__) || defined(__AVX512VL__)
                _mm256_zeroupper();
            #endif
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


            /**
             * Subtracts values from vector.
             * @param bV Second vector.
             * @return Char256 New vector being result of subtracting `bV` from vector.
             */
            Char256 operator-(const Char256& bV) const noexcept {
                return _mm256_sub_epi8(v, bV.v);
            }

            /**
             * Subtracts a single value from all vector fields.
             * @param b Value to subtract from vector.
             * @return Char256 New vector being result of subtracting `b` from vector.
             */
            Char256 operator-(const char& b) const noexcept {
                return _mm256_sub_epi8(v, _mm256_set1_epi8(b));
            }

            /**
             * Subtracts two vectors and stores result inside original vector.
             * @param bV Second vector.
             * @return Reference to same vector after subtracting `bV` from vector.
             */
            Char256& operator-=(const Char256& bV) noexcept {
                v = _mm256_sub_epi8(v, bV.v);
                return *this;
            }

            /**
             * Subtracts scalar from vector and stores result inside original vector.
             * @param b Scalar to be subtracted.
             * @return Reference to same vector after subtracting `b` from vector.
             */
            Char256& operator-=(const char& b) noexcept {
                v = _mm256_sub_epi8(v, _mm256_set1_epi8(b));
                return *this;
            }


            /**
             * Multiplies two vectors.
             * @param bV Second vector.
             * @return Char256 New vector being result of multiplying vector by `bV`.
             */
            Char256 operator*(const Char256& bV) const noexcept {
                #if defined(__AVX512BW__)
                    return _mm512_cvtepi16_epi8(
                        _mm512_mullo_epi16(
                            _mm512_cvtepi8_epi16(v), 
                            _mm512_cvtepi8_epi16(bV.v)
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

            /**
             * Multiplies all vector fields by a single value.
             * @param b Value to multiply by.
             * @return Char256 New vector being result of multiplying vector by `b`.
             */
            Char256 operator*(const char& b) const noexcept {
                #if defined(__AVX512BW__)
                    return _mm512_cvtepi16_epi8(
                        _mm512_mullo_epi16(
                            _mm512_cvtepi8_epi16(v), 
                            _mm512_set1_epi16(static_cast<short>(b))
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

            /**
             * Multiplies two vectors and stores result inside original vector.
             * @param bV Second vector.
             * @return Reference to same vector after multiplying by `bV`.
             */
            Char256& operator*=(const Char256& bV) noexcept {
                #if defined(__AVX512BW__)
                    v = _mm512_cvtepi16_epi8(
                        _mm512_mullo_epi16(
                            _mm512_cvtepi8_epi16(v), 
                            _mm512_cvtepi8_epi16(bV.v)
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

            /**
             * Multiplies vector by scalar and stores result inside original vector.
             * @param b Scalar to multiply by.
             * @return Reference to same vector after multiplying by `b`.
             */
            Char256& operator*=(const char& b) noexcept {
                #if defined(__AVX512BW__)
                    v = _mm512_cvtepi16_epi8(
                        _mm512_mullo_epi16(
                            _mm512_cvtepi8_epi16(v), 
                            _mm512_set1_epi16(static_cast<short>(b))
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


            /**
             * Divides two vectors.
             * @param bV Second vector (divisor).
             * @return Char256 New vector being result of dividing vector by `bV`.
             */
            Char256 operator/(const Char256& bV) const noexcept {
            #if defined(__AVX512_FP16__) && defined(__AVX512BW__) && defined(__AVX512F__) // If supports FP16
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                __m512i second16 = _mm512_cvtepi8_epi16(second.v);
                __m512h firstFp16 = _mm512_cvtepi16_ph(first16);
                __m512h secondFp16 = _mm512_cvtepi16_ph(bV16);
                firstFp16 = _mm512_div_ph(firstFp16, secondFp16);
                first16 = _mm512_cvttph_epi16(firstFp16);
                return _mm512_cvtepi16_epi8(first16);
            #elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                __m512i second16 = _mm512_cvtepi8_epi16(bV.v);
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

                return _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
            #else
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                auto [b_fhalf_epi16, b_shalf_epi16] = _sig_ext_epi8_epi16(bV.v);

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                auto [bv_first_half, bv_second_half] = _sig_ext_epi16_epi32(b_fhalf_epi16);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
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

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                return _mm256_or_si256(half_res, shalf_res);                
            #endif
            }

            /* // AVX2(512)version. Apparently _mm256_cvtepi16_epi8 and _mm256_cvtepi32_epi16 for some reason didn't make it to AVX2 lol ->_<'-
                __m256i v_first_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
                __m256i v_second_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));

                __m256i bV_first_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bV.v));
                __m256i bV_second_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bV.v, 1));

                __m256 v_f_first = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_first_16)));
                __m256 v_s_first = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_first_16, 1)));

                __m256 v_f_second = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_second_16)));
                __m256 v_s_second = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_second_16, 1)));

                __m256 bV_f_first = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(bV_first_16)));
                __m256 bV_s_first = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bV_first_16, 1)));

                __m256 bV_f_second = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(bV_second_16)));
                __m256 bV_s_second = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bV_second_16, 1)));

                v_f_first = _mm256_div_ps(v_f_first, bV_f_first);
                v_s_first = _mm256_div_ps(v_s_first, bV_s_first);
                v_f_second = _mm256_div_ps(v_f_second, bV_f_second);
                v_s_second = _mm256_div_ps(v_s_second, bV_s_second);

                v_first_16 = _mm256_castsi128_si256(_mm256_cvtepi32_epi16(_mm256_cvttps_epi32(v_f_first)));
                v_first_16 = _mm256_inserti128_si256(v_first_16, _mm256_cvtepi32_epi16(_mm256_cvttps_epi32(v_s_first)), 1);

                v_second_16 = _mm256_castsi128_si256(_mm256_cvtepi32_epi16(_mm256_cvttps_epi32(v_f_second)));
                v_second_16 = _mm256_inserti128_si256(v_second_16, _mm256_cvtepi32_epi16(_mm256_cvttps_epi32(v_s_second)), 1);

                v_first_16 = _mm256_castsi128_si256(_mm256_cvtepi16_epi8(v_first_16));
                return _mm256_inserti128_si256(v_first_16, _mm256_cvtepi16_epi8(v_second_16), 1);
             */


            /**
             * Divides all vector fields by a single value.
             * @param b Value (divisor).
             * @return Char256 New vector being result of dividing vector by `b`.
             */
            Char256 operator/(const char& b) const noexcept {
            #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

                __m512 secondfp = _mm512_set1_ps(static_cast<float>(b));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp);

                __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

                return _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
            #else
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                __m256 bV = _mm256_set1_ps(static_cast<float>(b));

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 3);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                return _mm256_or_si256(half_res, shalf_res);
                #endif
            }

            /**
             * Divides two vectors and stores result inside original vector.
             * @param bV Second vector (divisor).
             * @return Reference to same vector after dividing by `bV`.
             */
            Char256& operator/=(const Char256& bV) noexcept {
            #if defined(__AVX512_FP16__) && defined(__AVX512BW__) && defined(__AVX512F__) // If supports FP16
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                __m512i second16 = _mm512_cvtepi8_epi16(bV.v);
                __m512h firstFp16 = _mm512_cvtepi16_ph(first16);
                __m512h secondFp16 = _mm512_cvtepi16_ph(bV16);
                firstFp16 = _mm512_div_ph(firstFp16, secondFp16);
                first16 = _mm512_cvttph_epi16(firstFp16);
                v = _mm512_cvtepi16_epi8(first16);
            #elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                __m512i second16 = _mm512_cvtepi8_epi16(bV.v);
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

                v = _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
            #else
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                auto [b_fhalf_epi16, b_shalf_epi16] = _sig_ext_epi8_epi16(bV.v);

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                auto [bv_first_half, bv_second_half] = _sig_ext_epi16_epi32(b_fhalf_epi16);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
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

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                v = _mm256_or_si256(half_res, shalf_res);
            #endif
                return *this;
            }

            /**
             * Divides vector by scalar and stores result inside original vector.
             * @param b Scalar value (divisor).
             * @return Reference to same vector after dividing by `b`.
             */
            Char256& operator/=(const char& b) noexcept {
            #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

                __m512 secondfp = _mm512_set1_ps(static_cast<float>(b));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp);

                __m256i result = _mm256_castsi128_si256(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp)));

                v = _mm256_inserti128_si256(result, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(firstfp_1)), 1);
            #else
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                __m256 bV = _mm256_set1_ps(static_cast<float>(b));

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 3);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);
                sresult = _mm256_slli_si256(sresult, 1);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI8_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI8_CRATE_EPI32);

                __m256i shalf_res = _mm256_or_si256(fresult, sresult);

                v = _mm256_or_si256(half_res, shalf_res);
            #endif
                return *this;
            }

            /**
             * Performs modulo operation. It does so by dividing vectors, multiplying result and subtracting from vector.
             * @param bV Second modulo operand (divisor)
             * @return Result of modulo operation.
             */
            Char256 operator%(const Char256& bV) const noexcept {
            #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                __m512i second16 = _mm512_cvtepi8_epi16(bV.v);
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
                result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

                result = _mm512_mullo_epi16(result, second16);
                
                return _mm256_sub_epi8(v, _mm512_cvtepi16_epi8(result));
            #else
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                auto [b_fhalf_epi16, b_shalf_epi16] = _sig_ext_epi8_epi16(bV.v);

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                auto [bv_first_half, bv_second_half] = _sig_ext_epi16_epi32(b_fhalf_epi16);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
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

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
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
            #endif
            }

            /**
             * Calculates element-wise modulo of vector and scalar.
             * @param b Value (divisor).
             * @return Char256 New vector being result of modulo operation.
             */
            Char256 operator%(const char& b) const noexcept {
            #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                __m512i second16 = _mm512_set1_epi16(static_cast<short>(b));
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

                __m512 secondfp = _mm512_set1_ps(static_cast<float>(b));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp);

                __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
                result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

                result = _mm512_mullo_epi16(result, second16);
                
                return _mm256_sub_epi8(v, _mm512_cvtepi16_epi8(result));
            #else
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);

                __m256 bV = _mm256_set1_ps(static_cast<float>(b));
                __m256i bV_epi16 = _mm256_set1_epi16(static_cast<short>(b));

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
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
            #endif
            }

            /**
             * Performs modulo operation. It does so by dividing vectors, multiplying result and subtracting from vector.
             * @param bV Second modulo operand (divisor)
             * @return Reference to the original vector holding modulo operation results.
             */
            Char256& operator%=(const Char256& bV) noexcept {
            #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                __m512i second16 = _mm512_cvtepi8_epi16(bV.v);
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

                __m512 secondfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(second16)));
                __m512 secondfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(second16, 1)));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp_1);

                __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
                result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

                result = _mm512_mullo_epi16(result, second16);
                
                v = _mm256_sub_epi8(v, _mm512_cvtepi16_epi8(result));
            #else
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);
                auto [b_fhalf_epi16, b_shalf_epi16] = _sig_ext_epi8_epi16(bV.v);

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                auto [bv_first_half, bv_second_half] = _sig_ext_epi16_epi32(b_fhalf_epi16);
                __m256 bv_fhalf_f = _mm256_cvtepi32_ps(bv_first_half);
                __m256 bv_shalf_f = _mm256_cvtepi32_ps(bv_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
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

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));
                
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
            #endif
                return *this;
            }

            /**
             * Performs modulo operation. It does so by dividing vectors, multiplying result and subtracting from vector.
             * @param b Scalar (divisor).
             * @return Reference to the original vector holding modulo operation results.
             */
            Char256& operator%=(const char& b) noexcept {
            #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
                __m512i first16 = _mm512_cvtepi8_epi16(v);
                __m512i second16 = _mm512_set1_epi16(static_cast<short>(b));
                
                __m512 firstfp = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(first16)));
                __m512 firstfp_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(first16, 1)));

                __m512 secondfp = _mm512_set1_ps(static_cast<float>(b));

                firstfp = _mm512_div_ps(firstfp, secondfp);
                firstfp_1 = _mm512_div_ps(firstfp_1, secondfp);

                __m512i result = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp)));
                result = _mm512_inserti64x4(result, _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(firstfp_1)), 1);

                result = _mm512_mullo_epi16(result, second16);
                
                v = _mm256_sub_epi8(v, _mm512_cvtepi16_epi8(result));
            #else
                auto [v_fhalf_epi16, v_shalf_epi16] = _sig_ext_epi8_epi16(v);

                __m256 bV = _mm256_set1_ps(static_cast<float>(b));
                __m256i bV_epi16 = _mm256_set1_epi16(static_cast<short>(b));

                auto [v_first_half, v_second_half] = _sig_ext_epi16_epi32(v_fhalf_epi16);
                __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);
                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

                __m256i half_res = _mm256_or_si256(fresult, sresult);

                std::tie(v_first_half, v_second_half) = _sig_ext_epi16_epi32(v_shalf_epi16);
                
                v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
                v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

                fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
                sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));
                
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
            #endif
                return *this;
            }

            /**
             * Bitwise AND operator.
             * @param bV Second vector.
             * @return Char256 New vector being result of bitwise AND with `bV`.
             */
            Char256 operator&(const Char256& bV) const noexcept {
                return _mm256_and_si256(v, bV.v);
            }

            /**
             * Bitwise AND operator with scalar.
             * @param b Value to AND with.
             * @return Char256 New vector being result of bitwise AND with `b`.
             */
            Char256 operator&(const char& b) const noexcept {
                return _mm256_and_si256(v, _mm256_set1_epi8(b));
            }

            /**
             * Bitwise AND assignment operator.
             * Applies bitwise AND between this vector and the given vector, storing the result in this vector.
             * @param bV Second vector.
             * @return Reference to the modified object.
             */
            Char256& operator&=(const Char256& bV) noexcept {
                v = _mm256_and_si256(v, bV.v);
                return *this;
            }

            /**
             * Bitwise AND assignment operator.
             * Applies bitwise AND between this vector and the given value, storing the result in this vector.
             * @param b Value to AND with.
             * @return Reference to the modified object.
             */
            Char256& operator&=(const char& b) noexcept {
                v = _mm256_and_si256(v, _mm256_set1_epi8(b));
                return *this;
            }

            /**
             * Bitwise OR operator.
             * @param bV Second vector.
             * @return Char256 New vector being result of bitwise OR with `bV`.
             */
            Char256 operator|(const Char256& bV) const noexcept {
                return _mm256_or_si256(v, bV.v);
            }

            /**
             * Bitwise OR operator with scalar.
             * @param b Value to OR with.
             * @return Char256 New vector being result of bitwise OR with `b`.
             */
            Char256 operator|(const char& b) const noexcept {
                return _mm256_or_si256(v, _mm256_set1_epi8(b));
            }

            /**
             * Bitwise OR assignment operator.
             * Applies bitwise OR between this vector and the given vector, storing the result in this vector.
             * @param bV Second vector.
             * @return Reference to the modified object.
             */
            Char256& operator|=(const Char256& bV) noexcept {
                v = _mm256_or_si256(v, bV.v);
                return *this;
            }

            /**
             * Bitwise OR assignment operator.
             * Applies bitwise OR between this vector and the given scalar value, storing the result in this vector.
             * @param b Scalar value.
             * @return Reference to the modified object.
             */
            Char256& operator|=(const char& b) noexcept {
                v = _mm256_or_si256(v, _mm256_set1_epi8(b));
                return *this;
            }

            /**
             * Bitwise XOR operator.
             * @param bV Second vector.
             * @return Char256 New vector being result of bitwise XOR with `bV`.
             */
            Char256 operator^(const Char256& bV) const noexcept {
                return _mm256_xor_si256(v, bV.v);
            }

            /**
             * Bitwise XOR operator with scalar.
             * @param b Value to XOR with.
             * @return Char256 New vector being result of bitwise XOR with `b`.
             */
            Char256 operator^(const char& b) const noexcept {
                return _mm256_xor_si256(v, _mm256_set1_epi8(b));
            }

            /**
             * Bitwise XOR assignment operator.
             * Applies bitwise XOR between this vector and the given vector, storing the result in this vector.
             * @param bV Second vector.
             * @return Reference to the modified object.
             */
            Char256& operator^=(const Char256& bV) noexcept {
                v = _mm256_xor_si256(v, bV.v);
                return *this;
            }

            /**
             * Bitwise XOR assignment operator.
             * Applies bitwise XOR between this vector and the given scalar value, storing the result in this vector.
             * @param b Scalar value.
             * @return Reference to the modified object.
             */
            Char256& operator^=(const char& b) noexcept {
                v = _mm256_xor_si256(v, _mm256_set1_epi8(b));
                return *this;
            }


            /**
             * Bitwise left shift operator (element-wise).
             * @param bV Vector containing number of bits for which each corresponding element should be shifted.
             * @return Char256 New vector after left shift.
             */
            Char256 operator<<(const Char256& bV) const noexcept {
                #ifdef __AVX512BW__
                    __m512i fV = _mm512_cvtepi8_epi16(v);
                    __m512i sV = _mm512_cvtepi8_epi16(bV.v);
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


            /**
             * Bitwise left shift operator by scalar.
             * @param b Number of bits by which values should be shifted.
             * @return Char256 New vector after left shift.
             */
            Char256 operator<<(const unsigned int& b) const noexcept {
                #ifdef __AVX512BW__
                    __m512i fV = _mm512_cvtepi8_epi16(v);
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


            /**
             * Shifts values left while shifting in 0.
             * @param bV Vector containing number of bits for which each corresponding element should be shifted.
             * @return Reference to modified object.
             */
            Char256& operator<<=(const Char256& bV) noexcept {
                #ifdef __AVX512BW__
                    v = _mm512_cvtepi16_epi8(
                        _mm512_sllv_epi16(
                            _mm512_cvtepi8_epi16(v), 
                            _mm512_cvtepi8_epi16(bV.v)
                        )
                    );
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


            /**
             * Shifts values left while shifting in 0.
             * @param b Number of bits by which values should be shifted.
             * @return Reference to modified object.
             */
            Char256& operator<<=(const unsigned int& b) noexcept {
                #ifdef __AVX512BW__
                    v = _mm512_cvtepi16_epi8(
                        _mm512_slli_epi16(
                            _mm512_cvtepi8_epi16(v), 
                            static_cast<short>(b)
                        )
                    );
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


            /**
             * Bitwise right shift operator (element-wise, arithmetic shift).
             * @param bV Vector containing number of bits for which each corresponding element should be shifted.
             * @return Char256 New vector after right shift.
             */
            Char256 operator>>(const Char256& bV) const noexcept {
            #if defined(__AVX512BW__)
                return _mm512_cvtepi16_epi8(
                    _mm512_srav_epi16(
                        _mm512_cvtepi8_epi16(v), 
                        _mm512_cvtepi8_epi16(bV.v)
                    )
                );
            #else
                /*__m256i a_1 = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                a_1 = _mm256_slli_si256(a_1, 1);
                a_1 = _mm256_srai_epi16(a_1, 8);
                __m256i a_2 = _mm256_srai_epi16(_mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE), 8);
                
                __m256i b_1 = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI16);
                b_1 = _mm256_slli_si256(b_1, 1);
                b_1 = _mm256_srai_epi16(b_1, 8);
                __m256i b_2 = _mm256_srai_epi16(_mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI16_INVERSE), 8);

                __m128i res_1_1 = _mm_sra_epi16(_mm256_castsi256_si128(a_1), _mm256_castsi256_si128(b_1));
                __m128i res_1_2 = _mm_sra_epi16(_mm256_extracti128_si256(a_1, 1), _mm256_extracti128_si256(b_1, 1));

                res_1_1 = _mm_and_si128(res_1_1, _mm256_castsi256_si128(constants::EPI8_CRATE_EPI16));
                res_1_2 = _mm_slli_si128(_mm_and_si128(res_1_2, _mm256_castsi256_si128(constants::EPI8_CRATE_EPI16)), 1);
                res_1_1 = _mm_or_si128(res_1_1, res_1_2);
                
                
                __m128i res_2_1 = _mm_sra_epi16(_mm256_castsi256_si128(a_2), _mm256_castsi256_si128(b_2));
                __m128i res_2_2 = _mm_sra_epi16(_mm256_extracti128_si256(a_2, 1), _mm256_extracti128_si256(b_2, 1));

                res_2_1 = _mm_and_si128(res_2_1, _mm256_castsi256_si128(constants::EPI8_CRATE_EPI16));
                res_2_2 = _mm_slli_si128(_mm_and_si128(res_2_2, _mm256_castsi256_si128(constants::EPI8_CRATE_EPI16)), 1);
                res_2_1 = _mm_or_si128(res_2_1, res_2_2);
                
                return _mm256_inserti128_si256(_mm256_castsi128_si256(res_1_1), res_2_1, 1);*/

                __m256i q1_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                q1_a = _mm256_slli_si256(q1_a, 3);
                q1_a = _mm256_srai_epi32(q1_a, 24);
                __m256i q1_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);
                q1_b = _mm256_slli_si256(q1_b, 3);
                q1_b = _mm256_srai_epi32(q1_b, 24);
                
                __m256i q2_a = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                q2_a = _mm256_slli_si256(q2_a, 3);
                q2_a = _mm256_srai_epi32(q2_a, 24);
                
                __m256i q2_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);
                q2_b = _mm256_slli_si256(q2_b, 3);
                q2_b = _mm256_srai_epi32(q2_b, 24);
                
                __m256i q3_a = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                q3_a = _mm256_slli_si256(q3_a, 3);
                q3_a = _mm256_srai_epi32(q3_a, 24);
                
                __m256i q3_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);
                q3_b = _mm256_slli_si256(q3_b, 3);
                q3_b = _mm256_srai_epi32(q3_b, 24);
                
                __m256i q4_a = _mm256_srai_epi32(v, 24);

                __m256i q4_b = _mm256_srai_epi32(bV.v, 24);

                __m256i q1_res = _mm256_srav_epi32(q1_a, q1_b);
                __m256i q2_res = _mm256_srav_epi32(q2_a, q2_b);
                __m256i q3_res = _mm256_srav_epi32(q3_a, q3_b);
                __m256i q4_res = _mm256_srav_epi32(q4_a, q4_b);

                q1_res = _mm256_and_si256(q1_res, constants::EPI8_CRATE_EPI32);
                q2_res = _mm256_and_si256(q2_res, constants::EPI8_CRATE_EPI32);
                q3_res = _mm256_and_si256(q3_res, constants::EPI8_CRATE_EPI32);
                q4_res = _mm256_and_si256(q4_res, constants::EPI8_CRATE_EPI32);

                q2_res = _mm256_slli_si256(q2_res, 1);
                q3_res = _mm256_slli_si256(q3_res, 2);
                q4_res = _mm256_slli_si256(q4_res, 3);
                
                q1_res = _mm256_or_si256(q1_res, q2_res);
                q2_res = _mm256_or_si256(q3_res, q4_res);
                return  _mm256_or_si256(q1_res, q2_res);
            #endif
            }


            /**
             * Bitwise right shift operator by scalar (arithmetic shift).
             * @param b Number of bits by which values should be shifted.
             * @return Char256 New vector after right shift.
             */
            Char256 operator>>(const unsigned int& b) const noexcept {
            #ifdef __AVX512BW__
                return _mm512_cvtepi16_epi8(_mm512_srai_epi16(_mm512_cvtepi8_epi16(v), b));
            #else
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                fhalf = _mm256_slli_si256(fhalf, 1);
                fhalf = _mm256_srai_epi16(fhalf, 8);
                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                fhalf = _mm256_srai_epi16(fhalf, b);
                shalf = _mm256_srai_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                return _mm256_or_si256(fhalf, shalf);
            #endif
            }


            /**
             * Shifts values right while shifting in sign bit.
             * @param bV Vector containing number of bits for which each corresponding element should be shifted.
             * @return Reference to modified object.
             */
            Char256& operator>>=(const Char256& bV) noexcept {
            #if defined(__AVX512BW__)
                v = _mm512_cvtepi16_epi8(
                    _mm512_srav_epi16(
                        _mm512_cvtepi8_epi16(v), 
                        _mm512_cvtepi8_epi16(bV.v)
                    )
                );
            #else
                // TODO: Use _mm256_sra_epi16
                __m256i q1_a = _mm256_and_si256(v, constants::EPI8_CRATE_EPI32);
                q1_a = _mm256_slli_si256(q1_a, 3);
                q1_a = _mm256_srai_epi32(q1_a, 24);
                __m256i q1_b = _mm256_and_si256(bV.v, constants::EPI8_CRATE_EPI32);
                q1_b = _mm256_slli_si256(q1_b, 3);
                q1_b = _mm256_srai_epi32(q1_b, 24);
                
                __m256i q2_a = _mm256_and_si256(_mm256_srli_si256(v, 1), constants::EPI8_CRATE_EPI32);
                q2_a = _mm256_slli_si256(q2_a, 3);
                q2_a = _mm256_srai_epi32(q2_a, 24);
                
                __m256i q2_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 1), constants::EPI8_CRATE_EPI32);
                q2_b = _mm256_slli_si256(q2_b, 3);
                q2_b = _mm256_srai_epi32(q2_b, 24);
                
                __m256i q3_a = _mm256_and_si256(_mm256_srli_si256(v, 2), constants::EPI8_CRATE_EPI32);
                q3_a = _mm256_slli_si256(q3_a, 3);
                q3_a = _mm256_srai_epi32(q3_a, 24);
                
                __m256i q3_b = _mm256_and_si256(_mm256_srli_si256(bV.v, 2), constants::EPI8_CRATE_EPI32);
                q3_b = _mm256_slli_si256(q3_b, 3);
                q3_b = _mm256_srai_epi32(q3_b, 24);
                
                __m256i q4_a = _mm256_srai_epi32(v, 24);

                __m256i q4_b = _mm256_srai_epi32(bV.v, 24);

                __m256i q1_res = _mm256_srav_epi32(q1_a, q1_b);
                __m256i q2_res = _mm256_srav_epi32(q2_a, q2_b);
                __m256i q3_res = _mm256_srav_epi32(q3_a, q3_b);
                __m256i q4_res = _mm256_srav_epi32(q4_a, q4_b);

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


            /**
             * Shifts values right while shifting in sign bit.
             * @param b Number of bits by which values should be shifted.
             * @return Reference to modified object.
             */
            Char256& operator>>=(const unsigned int& b) noexcept {
            #ifdef __AVX512BW__
                v = _mm512_cvtepi16_epi8(_mm512_srai_epi16(_mm512_cvtepi8_epi16(v), b));
            #else
                __m256i fhalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16);
                fhalf = _mm256_slli_si256(fhalf, 1);
                fhalf = _mm256_srai_epi16(fhalf, 8);
                __m256i shalf = _mm256_and_si256(v, constants::EPI8_CRATE_EPI16_INVERSE);
                fhalf = _mm256_srai_epi16(fhalf, b);
                shalf = _mm256_srai_epi16(shalf, b);
                fhalf = _mm256_and_si256(fhalf, constants::EPI8_CRATE_EPI16);
                shalf = _mm256_and_si256(shalf, constants::EPI8_CRATE_EPI16_INVERSE);
                v = _mm256_or_si256(fhalf, shalf);
            #endif
                return *this;
            }

            /**
             * Bitwise NOT operator.
             * @return Char256 New vector with all bits inverted.
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
                
                _mm256_store_si256((__m256i*)tmp, v);
                
                tmp[32] = '\0';
                return std::string(tmp);
            }

            /**
             * Prints content of vector as raw string.
             * @param os Output stream, to which content will be written.
             * @param a Vector, whose value will be written to stream.
             * @return Reference to `os`.
             */
            friend std::ostream& operator<<(std::ostream& os, const Char256& a) {
                alignas(32) char tmp[33];
                tmp[32] = '\0';

                _mm256_store_si256((__m256i*)tmp, a.v);
                
                os << tmp;
                return os;
            }

    };
}

#endif