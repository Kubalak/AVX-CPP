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
     * * Provides comparison operators == !=.
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

            Double256() noexcept : v(_mm256_setzero_pd()){}

            Double256(const double val) noexcept : v(_mm256_set1_pd(val)){}

            Double256(const __m256d init) noexcept : v(init){}

            Double256(const Double256 &init) noexcept : v(init.v){}

            Double256(const std::array<double, 4> &init) noexcept : v(_mm256_loadu_pd(init.data())){}

            Double256(const double* pSrc) {    
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif    
                v = _mm256_loadu_pd(pSrc);
            }

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

            
            const __m256d get() const noexcept { return v;}
            void set(__m256d val) noexcept { v = val;}


            bool operator==(const Double256& bV) {
                // TODO: Check performance vs classic approach on MSVC and GCC.
                __m256d eq = _mm256_xor_pd(v, bV.v); // Bitwise XOR - equal values return field with 0.

                /*
                    Explanation - compute bitwise AND with value and sign bit set to 0.
                    Next is comparing result to 0 (remove sign bit from equation).
                    Compute AND of NOT v and NOT bV.v - if they are equal (0) then corresponding field will be set to 0.
                */
                __m256d zerofx = _mm256_castsi256_pd(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(bV.v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256())
                ));

                // Fixes 0.0 == -0.0 mismatch by zeroing corresponding fields
                eq = _mm256_and_pd(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castpd_si256(eq), _mm256_castpd_si256(eq)) != 0;
            }

            bool operator==(const double b) {
                __m256d bV = _mm256_set1_pd(b);
                __m256d eq = _mm256_xor_pd(v, bV);

                __m256d zerofx = _mm256_castsi256_pd(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(bV, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_pd(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castpd_si256(eq), _mm256_castpd_si256(eq)) != 0;
            }

            bool operator!=(const Double256& bV) {
                __m256d eq = _mm256_xor_pd(v, bV.v);

                __m256d zerofx = _mm256_castsi256_pd(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(bV.v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_pd(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castpd_si256(eq), _mm256_castpd_si256(eq)) == 0;
            }

            bool operator!=(const double b) {
                __m256d bV = _mm256_set1_pd(b);
                __m256d eq = _mm256_xor_pd(v, bV);

                __m256d zerofx = _mm256_castsi256_pd(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(v, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castpd_si256(_mm256_and_pd(bV, constants::DOUBLE_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_pd(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castpd_si256(eq), _mm256_castpd_si256(eq)) == 0;
            }


            Double256 operator+(const Double256& other) const noexcept {
                return Double256(_mm256_add_pd(v, other.v));
            }

            Double256 operator+(const double val) const noexcept {
                return Double256(_mm256_add_pd(v, _mm256_set1_pd(val)));
            }

            Double256& operator+=(const Double256& other) noexcept {
                v = _mm256_add_pd(v, other.v);
                return *this;
            }

            Double256& operator+=(const double val) noexcept {
                v = _mm256_add_pd(v, _mm256_set1_pd(val));
                return *this;
            }

            Double256 operator-(const Double256& other) const noexcept {
                return Double256(_mm256_sub_pd(v, other.v));
            }

            Double256 operator-(const double val) const noexcept {
                return Double256(_mm256_sub_pd(v, _mm256_set1_pd(val)));
            }

            Double256& operator-=(const Double256& other) noexcept {
                v = _mm256_sub_pd(v, other.v);
                return *this;
            }

            Double256& operator-=(const double val) noexcept {
                v = _mm256_sub_pd(v, _mm256_set1_pd(val));
                return *this;
            }

            Double256 operator*(const Double256& other) const noexcept {
                return Double256(_mm256_mul_pd(v, other.v));
            }

            Double256 operator*(const double val) const noexcept {
                return Double256(_mm256_mul_pd(v, _mm256_set1_pd(val)));
            }

            Double256& operator*=(const Double256& other) noexcept {
                v = _mm256_mul_pd(v, other.v);
                return *this;
            }

            Double256& operator*=(const double val) noexcept {
                v = _mm256_mul_pd(v, _mm256_set1_pd(val));
                return *this;
            }

            Double256 operator/(const Double256& other) const noexcept {
                return Double256(_mm256_div_pd(v, other.v));
            }

            Double256 operator/(const double val) const noexcept {
                return Double256(_mm256_div_pd(v, _mm256_set1_pd(val)));
            }

            Double256& operator/=(const Double256& other) noexcept {
                v = _mm256_div_pd(v, other.v);
                return *this;
            }

            Double256& operator/=(const double val) noexcept {
                v = _mm256_div_pd(v, _mm256_set1_pd(val));
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

    };
}


#endif