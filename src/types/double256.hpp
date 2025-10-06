#pragma once
#ifndef DOUBLE256_HPP__
#define DOUBLE256_HPP__

#include <array>
#include <string>
#include <stdexcept>
#include <immintrin.h>
#include "constants.hpp"

namespace avx {
    /**
     * Class providing vectorized version of `double`.
     * Can hold 4 individual `double` values.
     * Provides arithmetic operators.
     * Provides comparison operators == !=.
     */
    class Double256 {

        private:
            __m256d v;
        
        public:
            
            /**
             * Number of individual values stored by object. This value can be used to iterate over elements.
            */
            static constexpr int size = 4;

            /**
             * Type that is stored inside vector.
             */
            using storedType = double;

            /**
             * Default constructor. Initializes vector values with zeros.
             */
            Double256() noexcept : v(_mm256_setzero_pd()){}

            /**
             * Initializes all vector fields with single value.
             * @param init A literal value to be set.
             */
            Double256(const double init) noexcept : v(_mm256_set1_pd(init)){}

            /**
             * Initializes vector but using `__m256d` type.
             * @param init Raw value to be set.
             */
            Double256(const __m256d init) noexcept : v(init){}

            /**
             * Initializes vector with value from other vector.
             * @param init Object which value will be copied.
             */
            Double256(const Double256 &init) noexcept : v(init.v){}

            /**
             * Initialize vector with values read from an array.
             * @param init Array from which values will be copied.
             */
            Double256(const std::array<double, 4> &init) noexcept : v(_mm256_loadu_pd(init.data())){}

            /** Initializes vector by loading data from memory (via `_mm256_loadu_pd`).
             * @param pSrc Valid memory addres of minimal size of 256-bits (32 bytes).
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            Double256(const double* pSrc) {    
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif    
                v = _mm256_loadu_pd(pSrc);
            }

            /**
             * Initializes vector with variable-length initializer list.
             * If list contains less then 16 values missing values will be filled with zeros.
             * Otherwise only first 16 values will be copied into vector.
             * @param init Initializer list from which values will be copied.
             */
            Double256(std::initializer_list<double> init) {
                alignas(32) double init_v[size]{0.0, 0.0, 0.0, 0.0};
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
                v = _mm256_load_pd(init_v);
            }

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param pSrc Pointer to memory from which to load data.
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void load(const double *pSrc) {
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif    
                v = _mm256_loadu_pd(pSrc);
            }

            /**
             * Saves data to destination in memory.
             * @param dest Reference to the list to which vector will be saved. Array doesn't need to be aligned to any specific boundary.
             */
            void save(std::array<double, 4>& dest) const noexcept {
                _mm256_storeu_pd(dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (4x `double`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void save(double *pDest) const {
            #ifndef NDEBUG
                if(!pDest)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                _mm256_storeu_pd(pDest, v);
            }

            /**
             * Saves data to destination in memory. The memory must be aligned at 32-byte boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (4x `double`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void saveAligned(double *pDest) const {
            #ifndef NDEBUG
                if(!pDest)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                _mm256_store_pd(pDest, v);
            }

            /**
             * Get the internal vector value.
             * @returns The value of `__m256i` vector.
             */
            const __m256d get() const noexcept { return v;}

            /**
             * Set the internal vector value.
             * @param value New value to be set.
             */
            void set(__m256d val) noexcept { v = val;}

            /**
             * Compares two vectors for equality.
             * This operator is secured against -0.0 == 0.0 comparison ensuring it will result `true`.
             * @returns bool `true` if ALL values in vectors are equal otherwise `false`.
             */
            bool operator==(const Double256& bV) {
            #if defined(__AVX512F__) || defined(__AVX512VL__)
                _mm256_zeroupper();
            #endif
                __m256d eq = _mm256_xor_pd(v, bV.v); // Bitwise XOR - equal values return field with 0.

                __m256d zerofx = _mm256_castsi256_pd(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(bV.v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256())
                ));

                // Fixes 0.0 == -0.0 mismatch by zeroing corresponding fields
                eq = _mm256_and_pd(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castpd_si256(eq), _mm256_castpd_si256(eq)) != 0;
            }

            /**
             * Compares vector with scalar for equality.
             * This operator is secured against -0.0 == 0.0 comparison ensuring it will return `true`.
             * @returns bool `true` if ALL values in vector are equal to `b` otherwise `false`.
             */
            bool operator==(const double b) {
            #if defined(__AVX512F__) || defined(__AVX512VL__)
                _mm256_zeroupper();
            #endif
                __m256d bV = _mm256_set1_pd(b);
                __m256d eq = _mm256_xor_pd(v, bV);

                __m256d zerofx = _mm256_castsi256_pd(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(bV, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_pd(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castpd_si256(eq), _mm256_castpd_si256(eq)) != 0;
            }

            /**
             * Compares two vectors for inequality.
             * This operator is secured against -0.0 == 0.0 comparison ensuring it will return `false`.
             * @returns bool `true` if ANY value in vectors is not equal otherwise `false`.
             */
            bool operator!=(const Double256& bV) {
            #if defined(__AVX512F__) || defined(__AVX512VL__)
                _mm256_zeroupper();
            #endif
                __m256d eq = _mm256_xor_pd(v, bV.v);

                __m256d zerofx = _mm256_castsi256_pd(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(bV.v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_pd(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castpd_si256(eq), _mm256_castpd_si256(eq)) == 0;
            }

            /**
             * Compares vector with scalar for inequality.
             * This operator is secured against -0.0 == 0.0 comparison ensuring it will return `false`.
             * @returns bool `true` if ANY value in vector is not equal to `b` otherwise `false`.
             */
            bool operator!=(const double b) {
            #if defined(__AVX512F__) || defined(__AVX512VL__)
                _mm256_zeroupper();
            #endif
                __m256d bV = _mm256_set1_pd(b);
                __m256d eq = _mm256_xor_pd(v, bV);

                __m256d zerofx = _mm256_castsi256_pd(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(bV, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_pd(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castpd_si256(eq), _mm256_castpd_si256(eq)) == 0;
            }


            /**
             * Adds two vectors together.
             * @param bV Second vector.
             * @returns New vector being result of adding `bV` to vector.
             */
            Double256 operator+(const Double256& bV) const noexcept {
                return Double256(_mm256_add_pd(v, bV.v));
            }

            /**
             * Adds scalar to all vector fields.
             * @param b Scalar value to be added.
             * @returns New vector being result of adding `b` to vector.
             */
            Double256 operator+(const double b) const noexcept {
                return Double256(_mm256_add_pd(v, _mm256_set1_pd(b)));
            }

            /**
             * Adds two vectors together and stores result inside original vector.
             * @param bV Second vector.
             * @returns Reference to same vector after adding `bV` to vector.
             */
            Double256& operator+=(const Double256& bV) noexcept {
                v = _mm256_add_pd(v, bV.v);
                return *this;
            }

            /**
             * Adds scalar to vector and stores result inside original vector.
             * @param b Scalar to be added.
             * @returns Reference to same vector after adding `b` to vector.
             */
            Double256& operator+=(const double b) noexcept {
                v = _mm256_add_pd(v, _mm256_set1_pd(b));
                return *this;
            }

            /**
             * Subtracts two vectors.
             * @param bV Second vector.
             * @returns New vector being result of subtracting `bV` from vector.
             */
            Double256 operator-(const Double256& bV) const noexcept {
                return Double256(_mm256_sub_pd(v, bV.v));
            }

            /**
             * Subtracts scalar from all vector fields.
             * @param b Scalar value to be subtracted.
             * @returns New vector being result of subtracting `b` from vector.
             */
            Double256 operator-(const double b) const noexcept {
                return Double256(_mm256_sub_pd(v, _mm256_set1_pd(b)));
            }

            /**
             * Subtracts two vectors and stores result inside original vector.
             * @param bV Second vector.
             * @returns Reference to same vector after subtracting `bV` from vector.
             */
            Double256& operator-=(const Double256& bV) noexcept {
                v = _mm256_sub_pd(v, bV.v);
                return *this;
            }

            /**
             * Subtracts scalar from vector and stores result inside original vector.
             * @param b Scalar to be subtracted.
             * @returns Reference to same vector after subtracting `b` from vector.
             */
            Double256& operator-=(const double b) noexcept {
                v = _mm256_sub_pd(v, _mm256_set1_pd(b));
                return *this;
            }

            /**
             * Multiplies two vectors.
             * @param bV Second vector.
             * @returns New vector being result of multiplying vector by `bV`.
             */
            Double256 operator*(const Double256& bV) const noexcept {
                return Double256(_mm256_mul_pd(v, bV.v));
            }

            /**
             * Multiplies all vector fields by scalar.
             * @param b Scalar value to multiply by.
             * @returns New vector being result of multiplying vector by `b`.
             */
            Double256 operator*(const double b) const noexcept {
                return Double256(_mm256_mul_pd(v, _mm256_set1_pd(b)));
            }

            /**
             * Multiplies two vectors and stores result inside original vector.
             * @param bV Second vector.
             * @returns Reference to same vector after multiplying by `bV`.
             */
            Double256& operator*=(const Double256& bV) noexcept {
                v = _mm256_mul_pd(v, bV.v);
                return *this;
            }

            /**
             * Multiplies vector by scalar and stores result inside original vector.
             * @param b Scalar to multiply by.
             * @returns Reference to same vector after multiplying by `b`.
             */
            Double256& operator*=(const double b) noexcept {
                v = _mm256_mul_pd(v, _mm256_set1_pd(b));
                return *this;
            }

            /**
             * Divides two vectors.
             * @param bV Second vector (divisor).
             * @returns New vector being result of dividing vector by `bV`.
             */
            Double256 operator/(const Double256& bV) const noexcept {
                return Double256(_mm256_div_pd(v, bV.v));
            }

            /**
             * Divides all vector fields by scalar.
             * @param b Scalar value (divisor).
             * @returns New vector being result of dividing vector by `b`.
             */
            Double256 operator/(const double b) const noexcept {
                return Double256(_mm256_div_pd(v, _mm256_set1_pd(b)));
            }

            /**
             * Divides two vectors and stores result inside original vector.
             * @param bV Second vector (divisor).
             * @returns Reference to same vector after dividing by `bV`.
             */
            Double256& operator/=(const Double256& bV) noexcept {
                v = _mm256_div_pd(v, bV.v);
                return *this;
            }

            /**
             * Divides vector by scalar and stores result inside original vector.
             * @param b Scalar value (divisor).
             * @returns Reference to same vector after dividing by `b`.
             */
            Double256& operator/=(const double b) noexcept {
                v = _mm256_div_pd(v, _mm256_set1_pd(b));
                return *this;
            }

            /**
            * Indexing operator.
            * Does not support value assignment through this method (e.g. aV[0] = 1 won't work).
            * @param index Position of desired element between 0 and 3.
            * @return Value of underlying element.
            * @throws std::out_of_range If index is not within the correct range and build type is debug will be thrown. Otherwise bitwise AND will prevent index to be out of range.
            */
            double operator[](const unsigned int index) const 
            #ifndef NDEBUG
                {
                    if(index > 3)
                        throw std::invalid_argument("Invalid index! Index should be within 0-3, passed: " + std::to_string(index));
                    return ((double*)&v)[index];
                }
            #else
                noexcept { return ((double*)&v)[index & 3]; }
            #endif 


            /**
             * Returns string representation of vector.
             * Printing will result in Double256(<vector_values>) eg. Double256(1.000000, 2.000000, 3.000000, 4.000000)
             * @returns String representation of underlying vector.
             */
            std::string str() const noexcept {
                std::string result = "Double256(";
                double* iv = (double*)&v; 
                for(unsigned i{0}; i < 3; ++i)
                    result += std::to_string(iv[i]) + ", ";
                
                result += std::to_string(iv[3]);
                result += ")";
                return result;
            }

            // Friend operators (used for double <op> Double256)

            /**
             * Provides support for `double` + Double256 operation.
             * @param a Scalar to which `bV` should be added.
             * @param bV Vector which will be added.
             * @returns Double256 Vector being a result of `a` + `bV`
             */
            friend Double256 operator+(double a, const Double256 &bV){
                return _mm256_add_pd(bV.v, _mm256_set1_pd(a));
            }

            /**
             * Provides support for `double` - Double256 operation.
             * @param a Scalar to which `bV` should be substracted.
             * @param bV Vector which will be substracted.
             * @returns Double256 Vector being a result of `a` - `bV`
             */
            friend Double256 operator-(double a, const Double256 &bV){
                return _mm256_sub_pd(_mm256_set1_pd(a), bV.v);
            }

            /**
             * Provides support for `double` * Double256 operation.
             * @param a Scalar which should be multiplied by `bV`.
             * @param bV Vector which will be multiplier.
             * @returns Double256 Vector being a result of `a` * `bV`
             */
            friend Double256 operator*(double a, const Double256 &bV){
                return _mm256_mul_pd(_mm256_set1_pd(a), bV.v);
            }

            /**
             * Provides support for `double` / Double256 operation.
             * @param a Scalar which should be divided by `bV`.
             * @param bV Vector which will be divisor.
             * @returns Double256 Vector being a result of `a` / `bV`
             */
            friend Double256 operator/(double a, const Double256 &bV){
                return _mm256_div_pd(_mm256_set1_pd(a), bV.v);
            }

    };
}


#endif