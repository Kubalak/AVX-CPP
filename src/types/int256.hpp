#pragma once
#ifndef INT256_HPP__
#define INT256_HPP__

#include <set>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include <unordered_set>
#include <chrono>

#ifndef _MSC_VER
    #include "../misc/simd_ext_gcc.h"
#endif

#include "constants.hpp"

/**
 * Namespace containing type definitions and basic functions.
 */
namespace avx
{
    /**
     * Class providing vectorized version of `int`.
     * Can hold 8 individual `int` values.
     * Provides arithmetic and bitwise operators.
     * Provides comparison operators == !=.
     */
    class Int256
    
    {
    private:
        alignas(32) __m256i v;
    public:

        /**
         * Number of individual values stored by object. This value can be used to iterate over elements.
        */
        static constexpr int size = 8;

        /**
        * Type that is stored inside vector.
        */
        using storedType = int;

        /**
         * Default constructor. Initializes vector with zeros.
         */
        Int256() : v(_mm256_setzero_si256()) {}

        /** Initializes vector by loading data from memory (via `_mm256_lddq_si256`).
         * @param pSrc Valid memory addres of minimal size of 256-bits (32 bytes).
         * @throws std::invalid_argument If in Debug and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
         */
        Int256(const int *pSrc) {
        #ifndef NDEBUG
            if(!pSrc)
                throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            else
        #endif
            v = _mm256_lddqu_si256((const __m256i *)pSrc); 
        }

        /**
         * Initializes vector with const value. Each cell will be set with value of `init`.
         * @param init Value to be set.
         */
        Int256(const int &init) : v(_mm256_set1_epi32(init)) {};

        /**
         * Initializes vector from __m256i value.
         * @param init Value of type __m256i to initialize the vector.
         */
        Int256(const __m256i &init) : v(init) {};

        /**
         * Copy constructor.
         * Initializes vector from another Int256 vector.
         * @param init Another Int256 vector to copy from.
         */
        Int256(const Int256 &init) : v(init.v) {};

        /**
         * Initializes vector from std::array of 8 int values.
         * @param init Array of 8 int values to initialize the vector.
         */
        Int256(const std::array<int, 8> &init) : v(_mm256_loadu_si256((const __m256i *)init.data())) {};

        /**
         * Initializes vector from std::array of 8 short values.
         * Each short value is promoted to int.
         * @param init Array of 8 short values to initialize the vector.
         */
        Int256(const std::array<short, 8> &init) 
        : v(_mm256_set_epi32(
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

        /**
         * Initializes vector from std::array of 8 char values.
         * Each char value is promoted to int.
         * @param init Array of 8 char values to initialize the vector.
         */
        Int256(const std::array<char, 8> &init) : v(_mm256_set_epi32(
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

        /**
         * Initializes vector from initializer_list of int values.
         * If the list contains fewer than 8 elements, remaining elements are set to zero.
         * If the list contains more than 8 elements, only the first 8 are used.
         * @param init Initializer list of int values.
         */
        Int256(std::initializer_list<int> init) {
            alignas(32) int init_v[size];
            std::memset((char*)init_v, 0, 32);
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
         * Returns the internal __m256i value stored by the object.
         * @return The __m256i value.
         */
        __m256i get() const { return v; }

        /**
         * Sets the internal __m256i value stored by the object.
         * @param val New value of type __m256i.
         */
        void set(__m256i val) { v = val; }

        /**
         * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
         * @param pSrc Pointer to memory from which to load data.
         * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
         */
        void load(const int *pSrc) {
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
        void save(std::array<int, 8>& dest) const noexcept {
            _mm256_storeu_si256((__m256i*)dest.data(), v);
        }

        /**
         * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
         * 
         * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
         * @param pDest A valid pointer to a memory of at least 32 bytes (8x `int`).
         * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
         */
        void save(int *pDest) const {
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
         * @param pDest A valid pointer to a memory of at least 32 bytes (8x `int`).
         * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
         */
        void saveAligned(int *pDest) const {
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
        bool operator==(const Int256 &bV) const noexcept {
        #if (defined(__AVX512F__) || defined(__AVX512VL__)) && defined(__FIX_CMP) // Fix compare where output assembly included vpternlogq which produced UB
            _mm256_zeroupper();
        #endif
            __m256i eq = _mm256_xor_si256(v, bV.v);
            return _mm256_testz_si256(eq, eq) != 0;
        }

        /**
         * Compares vectors for equality.
         * @param b Value to compare.
         * @returns true if all values in vector is equal to `b`, otherwise false.
         */
        bool operator==(const int b) const noexcept {
        #if (defined(__AVX512F__) || defined(__AVX512VL__)) && defined(__FIX_CMP) // Fix compare where output assembly included vpternlogq which produced UB
            _mm256_zeroupper();
        #endif
            __m256i bV = _mm256_set1_epi32(b);
            __m256i eq = _mm256_xor_si256(v, bV);
            return _mm256_testz_si256(eq, eq) != 0;
        }

        /**
         * Compares vectors for inequality.
         * @param bV Second vector.
         * @returns true if ANY value is different between vectors.
         */
        bool operator!=(const Int256 &bV) const noexcept {
        #if (defined(__AVX512F__) || defined(__AVX512VL__)) && defined(__FIX_CMP) // Fix compare where output assembly included vpternlogq which produced UB
            _mm256_zeroupper();
        #endif
            __m256i eq = _mm256_xor_si256(v, bV.v);
            return _mm256_testz_si256(eq, eq) == 0;
        }

        /**
         * Compares vectors for inequality.
         * @param b Value to compare.
         * @returns true if ANY value in vector is different than `b`, otherwise false.
         */
        bool operator!=(const int b) const noexcept {
        #if (defined(__AVX512F__) || defined(__AVX512VL__)) && defined(__FIX_CMP) // Fix compare where output assembly included vpternlogq which produced UB
            _mm256_zeroupper();
        #endif
            __m256i bV = _mm256_set1_epi32(b);
            __m256i eq = _mm256_xor_si256(v, bV);
            return _mm256_testz_si256(eq, eq) == 0;
        }

        /**
        * Indexing operator.
        * Does not support value assignment through this method (e.g. aV[0] = 1 won't work).
        * @param index Position of desired element between 0 and 7.
        * @return Value of underlying element.
        * @throws std::out_of_range If index is not within the correct range and build type is debug will be thrown. Otherwise bitwise AND will prevent index to be out of range. Side effect is that only 3 LSBs are used from `index`.
        */
        int operator[](const unsigned int index) const 
        #ifndef NDEBUG
            {
                if(index > 7) 
                    throw std::out_of_range("Range be within range 0-7! Got: " + std::to_string(index));

                return ((int*)&v)[index];
            }
        #else
            noexcept { return ((int*)&v)[index & 7];}
        #endif 

        /**
         * Adds values from other vector and returns new vector.
         * @param bV Second vector.
         * @return Int256 New vector being a sum of this vector and `bv`.
         */
        Int256 operator+(const Int256 &bV) const noexcept { return _mm256_add_epi32(v, bV.v); };

        /**
         * Adds single value across all vector fields.
         * @param b Value to add to vector.
         * @return Int256 New vector being a sum of this vector and `b`.
         */
        Int256 operator+(const int b) const noexcept { return _mm256_add_epi32(v, _mm256_set1_epi32(b)); }

        /**
         * Subtracts values from vector.
         * @param bV Second vector.
         * @return Int256 New vector being result of subtracting `bV` from vector.
         */
        Int256 operator-(const Int256 &bV) const noexcept { return _mm256_sub_epi32(v, bV.v); };

        /**
         * Subtracts a single value from all vector fields.
         * @param b Value to subtract from vector.
         * @return Int256 New vector being result of subtracting `b` from vector.
         */
        Int256 operator-(const int b) const noexcept { return _mm256_sub_epi32(v, _mm256_set1_epi32(b)); }

        /**
         * Multiplies two vectors.
         * @param bV Second vector.
         * @return Int256 New vector being result of multiplying vector by `bV`.
         */
        Int256 operator*(const Int256 &bV) const noexcept { return _mm256_mullo_epi32(v, bV.v); }

        /**
         * Multiplies all vector fields by a single value.
         * @param b Value to multiply by.
         * @return Int256 New vector being result of multiplying vector by `b`.
         */
        Int256 operator*(const int b) const noexcept { return _mm256_mullo_epi32(v,_mm256_set1_epi32(b)); }

        /**
         * Divides two vectors.
         * @param bV Second vector (divisor).
         * @return Int256 New vector being result of dividing vector by `bV`.
         */
        Int256 operator/(const Int256 &bV) const noexcept { 
            // ASM goes brrr ≽^•⩊•^≼
            #ifdef __AVX512F__
                return _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                return _mm256_div_epi32(v, bV.v); 
            #endif
        }

        /**
         * Divides all vector fields by a single value.
         * @param b Value (divisor).
         * @return Int256 New vector being result of dividing vector by `b`.
         */
        Int256 operator/(const int b) const {

            if(!b) return _mm256_setzero_si256();

            #ifdef __AVX512F__
                return _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_set1_pd(static_cast<double>(b))
                    )
                );
            #else
                return _mm256_div_epi32(v, _mm256_set1_epi32(b));
            #endif
        }

        /**
         * Calculates element-wise modulo of two vectors.
         * @param bV Second vector (divisor).
         * @return Int256 New vector being result of modulo operation.
         */
        Int256 operator%(const Int256 &bV) const {
            #ifdef __AVX512F__
                __m256i divided = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                __m256i divided = _mm256_div_epi32(v, bV.v);
            #endif
            return _mm256_sub_epi32(v, _mm256_mullo_epi32(bV.v, divided));
        }

        /**
         * Calculates element-wise modulo of vector and scalar.
         * @param b Value (divisor).
         * @return Int256 New vector being result of modulo operation.
         */
        Int256 operator%(const int b) const {
            if(!b) return _mm256_setzero_si256();
            __m256i bV = _mm256_set1_epi32(b);
            #ifdef __AVX512F__
                __m256i divided = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV)
                    )
                );
            #else
                __m256i divided = _mm256_div_epi32(v, bV);
            #endif
            return _mm256_sub_epi32(v, _mm256_mullo_epi32(bV, divided));
        }

        /**
         * Bitwise XOR operator.
         * @param bV Second vector.
         * @return Int256 New vector being result of bitwise XOR with `bV`.
         */
        Int256 operator^(const Int256 &bV) const { return _mm256_xor_si256(v, bV.v); }

        /**
         * Bitwise XOR operator with scalar.
         * @param b Value to XOR with.
         * @return Int256 New vector being result of bitwise XOR with `b`.
         */
        Int256 operator^(const int b) const { return _mm256_xor_si256(v, _mm256_set1_epi32(b)); }

        /**
         * Bitwise OR operator.
         * @param bV Second vector.
         * @return Int256 New vector being result of bitwise OR with `bV`.
         */
        Int256 operator|(const Int256 &bV) const {return _mm256_or_si256(v, bV.v);}

        /**
         * Bitwise OR operator with scalar.
         * @param b Value to OR with.
         * @return Int256 New vector being result of bitwise OR with `b`.
         */
        Int256 operator|(const int b) const { return _mm256_or_si256(v, _mm256_set1_epi32(b)); }

        /**
         * Bitwise AND operator.
         * @param bV Second vector.
         * @return Int256 New vector being result of bitwise AND with `bV`.
         */
        Int256 operator&(const Int256 &bV) const { return _mm256_and_si256(v, bV.v);}

        /**
         * Bitwise AND operator with scalar.
         * @param b Value to AND with.
         * @return Int256 New vector being result of bitwise AND with `b`.
         */
        Int256 operator&(const int b) const { return _mm256_and_si256(v, _mm256_set1_epi32(b)); }

        /**
         * Bitwise NOT operator.
         * @return Int256 New vector with all bits inverted.
         */
        Int256 operator~() const { return _mm256_xor_si256(v, constants::ONES); }

        /**
         * Bitwise left shift operator (element-wise).
         * @param bV Vector containing number of bits for which each corresponding element should be shifted.
         * @return Int256 New vector after left shift.
         */
        Int256 operator<<(const Int256 &bV) const { return _mm256_sllv_epi32(v,bV.v); }

        /**
         * Bitwise left shift operator by scalar.
         * @param b Number of bits by which values should be shifted.
         * @return Int256 New vector after left shift.
         */
        Int256 operator<<(const int b) const { return _mm256_slli_epi32(v, b); }

        /**
         * Bitwise right shift operator (element-wise, arithmetic shift).
         * @param bV Vector containing number of bits for which each corresponding element should be shifted.
         * @return Int256 New vector after right shift.
         */
        Int256 operator>>(const Int256 &bV) const { return _mm256_srav_epi32(v, bV.v); }

        /**
         * Bitwise right shift operator by scalar (arithmetic shift).
         * @param b Number of bits by which values should be shifted.
         * @return Int256 New vector after right shift.
         */
        Int256 operator>>(const int b) const { return _mm256_srai_epi32(v, b); }

        /**
         * Adds two vectors together and stores result inside original vector.
         * @param bV Second vector.
         * @returns Reference to same vector after adding `bV` to vector.
         */
        Int256& operator+=(const Int256 &bV) {
            v = _mm256_add_epi32(v, bV.v);
            return *this;
        }
        
        /**
         * Adds scalar to vector and stores result inside original vector.
         * @param b Scalar to be added.
         * @returns Reference to same vector after adding `b` to vector.
         */
        Int256& operator+=(const int b) {
            v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
            return *this;
        }

        /**
         * Subtracts two vectors and stores result inside original vector.
         * @param bV Second vector.
         * @returns Reference to same vector after subtracting `bV` from vector.
         */
        Int256& operator-=(const Int256 &bV) {
            v = _mm256_sub_epi32(v, bV.v);
            return *this;
        }

        /**
         * Subtracts scalar from vector and stores result inside original vector.
         * @param b Scalar to be subtracted.
         * @returns Reference to same vector after subtracting `b` from vector.
         */
        Int256 &operator-=(const int b) {
            v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
            return *this;
        }

        /**
         * Multiplies two vectors and stores result inside original vector.
         * @param bV Second vector.
         * @returns Reference to same vector after multiplying by `bV`.
         */
        Int256 &operator*=(const Int256 &bV) {
            v = _mm256_mullo_epi32(v, bV.v);
            return *this;
        }

        /**
         * Multiplies vector by scalar and stores result inside original vector.
         * @param b Scalar to multiply by.
         * @returns Reference to same vector after multiplying by `b`.
         */
        Int256 &operator*=(const int b)
        {
            v = _mm256_mullo_epi32(v, _mm256_set1_epi32(b));
            return *this;
        };

        /**
         * Divides two vectors and stores result inside original vector.
         * @param bV Second vector (divisor).
         * @returns Reference to same vector after dividing by `bV`.
         */
        Int256 &operator/=(const Int256 &bV) {
            #ifdef __AVX512F__
                v = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                v = _mm256_div_epi32(v, bV.v); 
            #endif
            return *this;
        }

        /**
         * Divides vector by scalar and stores result inside original vector.
         * @param b Scalar value (divisor).
         * @returns Reference to same vector after dividing by `b`.
         */
        Int256 &operator/=(const int b) {
            #ifdef __AVX512F__
                v = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_set1_pd(static_cast<double>(b))
                    )
                );
            #else
                v = _mm256_div_epi32(v, _mm256_set1_epi32(b));
            #endif
            return *this;
        }


        /**
         * Performs modulo operation. It does so by dividing vectors, multiplying result and subtracting from vector.
         * @param bV Second modulo operand (divisor)
         * @return Result of modulo operation.
         */
        Int256& operator%=(const Int256 &bV) {
            #ifdef __AVX512F__
                __m256i divided = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                __m256i divided = _mm256_div_epi32(v, bV.v);
            #endif
            
            v = _mm256_sub_epi32(v, _mm256_mullo_epi32(bV.v, divided));
            
            return *this;
        }

        /**
         * Performs modulo operation. It does so by dividing vectors, multiplying result and subtracting from vector.
         * @param b Second modulo operand (divisor).
         * @return Reference to the original vector holding modulo operation results.
         */
        Int256& operator%=(const int b){
            if(!b) {
                v = _mm256_setzero_si256();
                return *this;
            }
            
            __m256i bV = _mm256_set1_epi32(b);

            #ifdef __AVX512F__
                __m256i divided = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV)
                    )
                );
            #else
                __m256i divided = _mm256_div_epi32(v, bV);
            #endif
            
            v = _mm256_sub_epi32(v, _mm256_mullo_epi32(bV, divided));
            
            return *this;
        }

        /**
         * Bitwise OR assignment operator.
         * Applies bitwise OR between this vector and the given vector, storing the result in this vector.
         * @param bV Second vector.
         * @return Reference to the modified object.
         */
        Int256 & operator|=(const Int256 &bV){
            v = _mm256_or_si256(v, bV.v);
            return *this;
        }

        /**
         * Bitwise OR assignment operator.
         * Applies bitwise OR between this vector and the given integer value, storing the result in this vector.
         * @param b Integer value.
         * @return Reference to the modified object.
         */
        Int256 & operator|=(const int b){
            v = _mm256_or_si256(v, _mm256_set1_epi32(b));
            return *this;
        }

        /**
         * Bitwise AND assignment operator.
         * Applies bitwise AND between this vector and the given vector, storing the result in this vector.
         * @param bV Second vector.
         * @return Reference to the modified object.
         */
        Int256 & operator&=(const Int256 &bV){
            v = _mm256_and_si256(v, bV.v);
            return *this;
        }

        /**
         * Bitwise AND assignment operator.
         * Applies bitwise AND between this vector and the given integer value, storing the result in this vector.
         * @param b Integer value.
         * @return Reference to the modified object.
         */
        Int256 & operator&=(const int b){
            v = _mm256_and_si256(v, _mm256_set1_epi32(b));
            return *this;
        }

        /**
         * Bitwise XOR assignment operator.
         * Applies bitwise XOR between this vector and the given vector, storing the result in this vector.
         * @param bV Second vector.
         * @return Reference to the modified object.
         */
        Int256 &operator^=(const Int256 &bV) {
            v = _mm256_xor_si256(v, bV.v);
            return *this;
        }

        /**
         * Bitwise XOR assignment operator.
         * Applies bitwise XOR between this vector and the given integer value, storing the result in this vector.
         * @param b Integer value.
         * @return Reference to the modified object.
         */
        Int256 &operator^=(const int b){
            v = _mm256_xor_si256(v, _mm256_set1_epi32(b));
            return *this;
        }

        /**
         * Shifts values left while shifting in 0.
         * @param bV Vector containing number of bits for which each corresponding element should be shifted.
         * @returns Reference to modified object.
         */
        Int256 &operator<<=(const Int256 &bV) {
            v = _mm256_sllv_epi32(v, bV.v);
            return *this;
        }
        
        /**
         * Shifts values right while shifting in 0.
         * @param b Number of bits by which values should be shifted.
         * @returns Reference to modified object.
         */
        Int256 &operator<<=(const int b) {
            v = _mm256_slli_epi32(v, b);
            return *this;
        }

        /**
         * Shifts values right while shifting in sign bit.
         * @param bV Vector containing number of bits for which each corresponding element should be shifted.
         * @returns Reference to modified object.
         */
        Int256 &operator>>=(const Int256 &bV) {
            v = _mm256_srav_epi32(v, bV.v);
            return *this;
        }

        /**
         * Shifts values right while shifting in sign bit.
         * @param b Number of bits by which values should be shifted.
         * @returns Reference to modified object.
         */
        Int256 &operator>>=(const int b) {
            v = _mm256_srai_epi32(v, b);
            return *this;
        }

        /**
         * Returns string representation of vector.
         * Printing will result in Int256(<vector_values>) eg. Int256(1, 2, 3, 4, 5, 6, 7, 8)
         * @returns String representation of underlying vector.
         */
        std::string str() const {
            std::string result = "Int256(";
            int* iv = (int*)&v; 
            for(unsigned i{0}; i < 7; ++i)
                result += std::to_string(iv[i]) + ", ";
            
            result += std::to_string(iv[7]);
            result += ")";
            return result;
        }

    };
};
#endif