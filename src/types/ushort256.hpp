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
        
        /**
         * Number of individual values stored by object. This value can be used to iterate over elements.
         */
        static constexpr const int size = 16;

        /**
         * Type that is stored inside vector.
         */
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

        /**
         * Initializes vector from initializer_list of int values.
         * If the list contains fewer than 16 elements, remaining elements are set to zero.
         * If the list contains more than 16 elements, only the first 16 are used.
         * @param init Initializer list of int values.
         */
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
         * @param pSrc A valid address containing at least 16 `unsigned short` numbers.
         * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
         */
        explicit UShort256(const unsigned short* pSrc) {
        #ifndef NDEBUG
            if(!pSrc)
                throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            else
        #endif
            v = _mm256_lddqu_si256((const __m256i*)pSrc);
        }

        /**
         * Initializes all vector fields with single value.
         * @param b A literal value to be set.
         */
        explicit UShort256(const unsigned short b) noexcept : v(_mm256_set1_epi16(b)){}

        /**
         * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
         * @param pSrc Pointer to memory from which to load data.
         * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
         */
        void load(const unsigned short *pSrc) {
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
        void save(std::array<unsigned short, 16> &dest) const noexcept {
            _mm256_storeu_si256((__m256i*)dest.data(), v);
        }

        /**
         * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
         * 
         * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
         * @param pDest A valid pointer to a memory of at least 32 bytes (16x `unsigned short`).
         * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
         */
        void save(unsigned short *pDest) const {
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
         * @param pDest A valid pointer to a memory of at least 32 bytes (16x `unsigned short`).
         * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
         */
        void saveAligned(unsigned short *pDest) const {
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
        
        /**
         * Adds values from other vector and returns new vector.
         * @param bV Second vector.
         * @return UShort256 New vector being a sum of this vector and `bV`.
         */
        UShort256 operator+(const UShort256& bV) const noexcept{
            return _mm256_add_epi16(v, bV.v);
        }

        /**
         * Adds single value across all vector fields.
         * @param b Value to add to vector.
         * @return UShort256 New vector being a sum of this vector and `b`.
         */
        UShort256 operator+(const unsigned short& b) const noexcept{
            return _mm256_add_epi16(v, _mm256_set1_epi16(b));
        }

        /**
         * Adds two vectors together and stores result inside original vector.
         * @param bV Second vector.
         * @returns Reference to same vector after adding `bV` to vector.
         */
        UShort256& operator+=(const UShort256& bV) noexcept {
            v = _mm256_add_epi16(v, bV.v);
            return *this;
        }

        /**
         * Adds scalar to vector and stores result inside original vector.
         * @param b Scalar to be added.
         * @returns Reference to same vector after adding `b` to vector.
         */
        UShort256& operator+=(const unsigned short& b) noexcept {
            v = _mm256_add_epi16(v, _mm256_set1_epi16(b));
            return *this;
        }

        /**
         * Subtracts values from vector.
         * @param bV Second vector.
         * @return UShort256 New vector being result of subtracting `bV` from vector.
         */
        UShort256 operator-(const UShort256& bV) const noexcept {
            return _mm256_sub_epi16(v, bV.v);
        }

        /**
         * Subtracts a single value from all vector fields.
         * @param b Value to subtract from vector.
         * @return UShort256 New vector being result of subtracting `b` from vector.
         */
        UShort256 operator-(const unsigned short& b) const noexcept {
            return _mm256_sub_epi16(v, _mm256_set1_epi16(b));
        }

        /**
         * Subtracts two vectors and stores result inside original vector.
         * @param bV Second vector.
         * @returns Reference to same vector after subtracting `bV` from vector.
         */
        UShort256& operator-=(const UShort256& bV) noexcept {
            v = _mm256_sub_epi16(v, bV.v);
            return *this;
        }

        /**
         * Subtracts scalar from vector and stores result inside original vector.
         * @param b Scalar to be subtracted.
         * @returns Reference to same vector after subtracting `b` from vector.
         */
        UShort256& operator-=(const unsigned short& b) noexcept {
            v =_mm256_sub_epi16(v, _mm256_set1_epi16(b));
            return *this;
        }

        /**
         * Multiplies two vectors.
         * @param bV Second vector.
         * @return UShort256 New vector being result of multiplying vector by `bV`.
         */
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

        /**
         * Multiplies all vector fields by a single value.
         * @param b Value to multiply by.
         * @return UShort256 New vector being result of multiplying vector by `b`.
         */
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

        /**
         * Multiplies two vectors and stores result inside original vector.
         * @param bV Second vector.
         * @returns Reference to same vector after multiplying by `bV`.
         */
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

        /**
         * Multiplies vector by scalar and stores result inside original vector.
         * @param b Scalar to multiply by.
         * @returns Reference to same vector after multiplying by `b`.
         */
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
         * NOTE: Value is first casted to `int` and then to `float` and inverse using `_mm256_cvttps_epi32` to return integer.
         * @param bV Divisors vector.
         * @return Result of integer division with truncation.
         */
        UShort256 operator/(const UShort256 &bV) const noexcept {
            #ifdef __AVX512F__
                return _mm512_cvtepi32_epi16(
                    _mm512_cvttps_epi32(
                        _mm512_div_ps(
                            _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v)), 
                            _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(bV.v))
                        )
                    )
                );
            #else
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

                __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
                __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));

                fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
                fresult = _mm256_slli_si256(fresult, 2);

                sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);
                
                return _mm256_or_si256(fresult, sresult);
            #endif
        }

        /**
         * Performs an integer division by scalar. 
         * 
         * NOTE: Value is first casted to `int` and then to `float` and inverse using `_mm256_cvttps_epi32` to return integer.
         * @param b Divisor value.
         * @return Result of integer division with truncation.
         */
        UShort256 operator/(const unsigned short &b) const noexcept {
        #ifdef __AVX512F__
            return _mm512_cvtepi32_epi16(
                _mm512_cvttps_epi32(
                    _mm512_div_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v)), 
                        _mm512_set1_ps(static_cast<float>(b))
                    )
                )
            );
        #else
            __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
            v_first_half = _mm256_srli_si256(v_first_half, 2);
            __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
            __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
            __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

            __m256 bV = _mm256_set_ps(b, b, b, b, b, b, b, b);

            __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
            __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));

            fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
            fresult = _mm256_slli_si256(fresult, 2);

            sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);
            
            return _mm256_or_si256(fresult, sresult);
        #endif
        }

        /**
         * Performs an integer division with assignment. 
         * 
         * NOTE: Value is first casted to `int` and then to `float` and inverse using `_mm256_cvttps_epi32` to return integer.
         * @param bV Divisors vector.
         * @return Reference to modified object.
         */
        UShort256& operator/=(const UShort256& bV) noexcept {
        #ifdef __AVX512F__
            v = _mm512_cvtepi32_epi16(
                    _mm512_cvttps_epi32(
                        _mm512_div_ps(
                            _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v)), 
                            _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(bV.v))
                        )
                    )
                );
        #else
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

            __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
            __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));

            fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
            fresult = _mm256_slli_si256(fresult, 2);

            sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);
            
            v = _mm256_or_si256(fresult, sresult);
        #endif
            return *this;
        }
        
        /**
         * Performs an integer division by scalar with assignment. 
         * 
         * NOTE: Value is first casted to `int` and then to `float` and inverse using `_mm256_cvttps_epi32` to return integer.
         * @param b Divisor value.
         * @return Reference to modified object.
         */
        UShort256& operator/=(const unsigned short& b) noexcept {
        #ifdef __AVX512F__
            v = _mm512_cvtepi32_epi16(
                _mm512_cvttps_epi32(
                    _mm512_div_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v)), 
                        _mm512_set1_ps(static_cast<float>(b))
                    )
                )
            );
        #else
            __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
            v_first_half = _mm256_srli_si256(v_first_half, 2);
            __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
            __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
            __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

            __m256 bV = _mm256_set_ps(b, b, b, b, b, b, b, b);

            __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bV));
            __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bV));

            fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);
            fresult = _mm256_slli_si256(fresult, 2);

            sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);
            
            v = _mm256_or_si256(fresult, sresult);
        #endif
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
        UShort256 operator%(const UShort256 &bV) const noexcept {
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

            __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
            __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));

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
         * @param b Divisor.
         * @return Modulo result.
         */
        UShort256 operator%(const unsigned short &b) noexcept {
            __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
            v_first_half = _mm256_srli_si256(v_first_half, 2);
            __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
            __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
            __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

            __m256i bV = _mm256_set_epi16(0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b);
            __m256 bVf = _mm256_set1_ps(b);

            __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bVf));
            __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bVf));

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

            __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bv_fhalf_f));
            __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bv_shalf_f));

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
         * @param b Divisor.
         * @return Reference to modified object.
         */
        UShort256& operator%=(const unsigned short &b) noexcept {
            __m256i v_first_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32_INVERSE);
            v_first_half = _mm256_srli_si256(v_first_half, 2);
            __m256i v_second_half = _mm256_and_si256(v, constants::EPI16_CRATE_EPI32);
            __m256 v_fhalf_f = _mm256_cvtepi32_ps(v_first_half);
            __m256 v_shalf_f = _mm256_cvtepi32_ps(v_second_half);

            __m256i bV = _mm256_set_epi16(0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b, 0, b);
            __m256 bVf = _mm256_set1_ps(b);

            __m256i fresult = _mm256_cvttps_epi32(_mm256_div_ps(v_fhalf_f, bVf));
            __m256i sresult = _mm256_cvttps_epi32(_mm256_div_ps(v_shalf_f, bVf));

            fresult = _mm256_and_si256(fresult, constants::EPI16_CRATE_EPI32);

            sresult = _mm256_and_si256(sresult, constants::EPI16_CRATE_EPI32);

            fresult = _mm256_sub_epi32(v_first_half, _mm256_mullo_epi32(bV, fresult));
            sresult = _mm256_sub_epi32(v_second_half, _mm256_mullo_epi32(bV, sresult));
            
            fresult = _mm256_slli_si256(fresult, 2);
            
            v = _mm256_or_si256(fresult, sresult);
            return *this;
        }
        
        /**
         * Bitwise OR operator.
         * @param bV Second vector.
         * @return UShort256 New vector being result of bitwise OR with `bV`.
         */
        UShort256 operator|(const UShort256& bV) const noexcept {
            return _mm256_or_si256(v, bV.v);
        }

        /**
         * Bitwise OR operator with scalar.
         * @param b Value to OR with.
         * @return UShort256 New vector being result of bitwise OR with `b`.
         */
        UShort256 operator|(const unsigned short& b) const noexcept {
            return _mm256_or_si256(v, _mm256_set1_epi16(b));
        }

        /**
         * Bitwise OR assignment operator.
         * Applies bitwise OR between this vector and the given value, storing the result in this vector.
         * @param bV Second vector.
         * @return Reference to the modified object.
         */
        UShort256& operator|=(const UShort256& bV) noexcept {
            v = _mm256_or_si256(v, bV.v);
            return *this;
        }

        /**
         * Bitwise AND operator with scalar.
         * @param b Value to AND with.
         * @return UShort256 New vector being result of bitwise AND with `b`.
         */
        UShort256& operator|=(const unsigned short& b) noexcept {
            v = _mm256_or_si256(v, _mm256_set1_epi16(b));
            return *this;            
        }

        /**
         * Bitwise AND operator.
         * @param bV Second vector.
         * @return UShort256 New vector being result of bitwise AND with `bV`.
         */
        UShort256 operator&(const UShort256& bV) const noexcept {
            return _mm256_and_si256(v, bV.v);
        }

        /**
         * Bitwise AND operator with scalar.
         * @param b Value to AND with.
         * @return UShort256 New vector being result of bitwise AND with `bV`.
         */
        UShort256 operator&(const unsigned short& b) const noexcept {
            return _mm256_and_si256(v, _mm256_set1_epi16(b));
        }
        
        /**
         * Bitwise AND assignment operator.
         * Applies bitwise AND between this vector and the given vector, storing the result in this vector.
         * @param bV Second vector.
         * @return Reference to the modified object.
         */
        UShort256& operator&=(const UShort256& bV) noexcept {
            v = _mm256_and_si256(v, bV.v);
            return *this;
        }
        
        /**
         * Bitwise AND assignment operator with scalar.
         * Applies bitwise AND between this vector and the given value, storing the result in this vector.
         * @param b Value to AND with.
         * @return Reference to the modified object.
         */
        UShort256& operator&=(const unsigned short& b) noexcept {
            v = _mm256_and_si256(v, _mm256_set1_epi16(b));
            return *this;
        }

        /**
         * Bitwise XOR operator.
         * @param bV Second vector.
         * @return UShort256 New vector being result of bitwise XOR with `bV`.
         */
        UShort256 operator^(const UShort256 &bV) const noexcept{
            return _mm256_xor_si256(v, bV.v);            
        }
        
        /**
         * Bitwise XOR operator with scalar.
         * @param b Value to XOR with.
         * @return UShort256 New vector being result of bitwise XOR with `b`.
         */
        UShort256 operator^(const unsigned short &b) const noexcept{
            return _mm256_xor_si256(v, _mm256_set1_epi16(b));
        }
        
        /**
         * Bitwise XOR assignment operator.
         * Applies bitwise XOR between this vector and the given vector, storing the result in this vector.
         * @param bV Second vector.
         * @return Reference to the modified object.
         */
        UShort256& operator^=(const UShort256 &bV) noexcept{
            v = _mm256_xor_si256(v, bV.v);
            return *this;
        }
        
        /**
         * Bitwise XOR assignment operator with scalar.
         * Applies bitwise XOR between this vector and the given value, storing the result in this vector.
         * @param b Value to XOR with.
         * @return Reference to the modified object.
         */
        UShort256& operator^=(const unsigned short &b) noexcept{
            v = _mm256_xor_si256(v, _mm256_set1_epi16(b));
            return *this;
        }

        /**
         * Performs left bitwise shift of corresponding values.
         * @param bV Second vector that specifies number of bits to shift (for each 16-bit value).
         * @return UShort256 New vector with each fiedld shifted by number of bits specfied in `bV`.
         */
        UShort256 operator<<(const UShort256 &bV) const noexcept {
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

        /**
         * Bitwise left shift operator by scalar.
         * @param b Number of bits by which values should be shifted.
         * @return UShort256 New vector after left shift.
         */
        UShort256 operator<<(const unsigned int &b) const noexcept {
            return _mm256_slli_epi16(v, b);
        }

        /**
         * Bitwise left shift assignment operator (element-wise).
         * Shifts values left by the number of bits specified in each element of the vector.
         * @param bV Vector containing number of bits for which each corresponding element should be shifted.
         * @return Reference to modified object.
         */
        UShort256& operator<<=(const UShort256 &bV) noexcept {
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

        /**
         * Bitwise left shift assignment operator by scalar.
         * Shifts values left by the number of bits specified.
         * @param b Number of bits by which values should be shifted.
         * @return Reference to modified object.
         */
        UShort256& operator<<=(const unsigned int &b) noexcept {
            v = _mm256_slli_epi16(v, b);
            return *this;
        }

        /**
         * Bitwise right shift operator (element-wise).
         * @param bV Vector containing number of bits for which each corresponding element should be shifted.
         * @return UShort256 New vector after right shift.
         */
        UShort256 operator>>(const UShort256 &bV) const noexcept {
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

        /**
         * Bitwise right shift operator by scalar.
         * @param b Number of bits by which values should be shifted.
         * @return UShort256 New vector after right shift.
         */
        UShort256 operator>>(const unsigned int &b) const noexcept{
            return _mm256_srli_epi16(v, b);
        }
        
        /**
         * Bitwise right shift assignment operator (element-wise).
         * Shifts values right by the number of bits specified in each element of the vector.
         * @param bV Vector containing number of bits for which each corresponding element should be shifted.
         * @return Reference to modified object.
         */
        UShort256& operator>>=(const UShort256 &bV) noexcept {
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

        /**
         * Bitwise right shift assignment operator by scalar.
         * Shifts values right by the number of bits specified.
         * @param b Number of bits by which values should be shifted.
         * @return Reference to modified object.
         */
        UShort256& operator>>=(const unsigned int &b) noexcept {
            v = _mm256_srli_epi16(v, b);
            return *this;
        }

        /**
         * Bitwise NOT operator.
         * @return UShort256 New vector with all bits inverted.
         */
        UShort256 operator~() const noexcept{
            return _mm256_xor_si256(v, constants::ONES);
        }

        /**
         * Returns string representation of vector.
         * Printing will result in Short256(<vector_values>) eg. Short256(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
         * @returns String representation of underlying vector.
         */
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