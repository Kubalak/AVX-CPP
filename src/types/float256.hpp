#pragma once
#ifndef FLOAT256_HPP__
#define FLOAT256_HPP__
#include <array>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include <types/constants.hpp>

namespace avx {
    /**
     * Class providing vectorized version of `float`.
     * Can hold 8 individual `float` values.
     * Provides arithmetic operators.
     * * Provides comparison operators == != (optimization on the way).
     */
    /**
     * Class providing vectorized version of `float`.
     * Can hold 8 individual `float` values.
     * Provides arithmetic operators.
     * * Provides comparison operators == != (optimization on the way).
     */
    class Float256 {
        private:
        __m256 v;

        public:
            static constexpr int size = 8;
            using storedType = float;

            Float256() noexcept : v(_mm256_setzero_ps()){}
            Float256(const Float256 &init) noexcept : v(init.v){}
            Float256(const float value) noexcept : v(_mm256_set1_ps(value)){}
            Float256(const __m256 init) noexcept : v(init){}
            Float256(const std::array<float, 8> &init) noexcept : v(_mm256_loadu_ps(init.data())){}

            Float256(std::initializer_list<float> init) noexcept {
                alignas(32) float init_v[size];
                memset((char*)init_v, 0, 32);
                if(init.size() < size){
                    auto begin = init.begin();
                    for(char i{0}; i < init.size(); ++i){
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                else {
                    auto begin = init.begin();
                    for(char i{0}; i < size; ++i){
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                v = _mm256_load_ps((const float*)init_v);
            }

            Float256(const float* pSrc) : v(_mm256_loadu_ps(pSrc)){}

            const __m256 get() const noexcept { return v;}
            void set(__m256 val) noexcept { v = val;}

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param pSrc Pointer to memory from which to load data.
             */
            void load(const float *pSrc) {
                if(pSrc != nullptr)
                    v = _mm256_loadu_ps(pSrc);
            }

            void save(float *pDest) const {
                _mm256_storeu_ps(pDest, v);
            }

            void save(std::array<float, 8> &dest) const noexcept{
                _mm256_storeu_ps(dest.data(), v);
            }

            void saveAligned(float* dest) const {
                _mm256_store_ps(dest, v);
            }
            

            /**
             * Compares with second vector for equality. This method is secured to return true when comparing 0.0f with -0.0f.
             * @param bV Second vector to compare.
             * @returns `true` if all fields in both vectors have the same value, `false` otherwise.
             */
            bool operator==(const Float256& bV) {
                __m256 eq = _mm256_xor_ps(v, bV.v); // Bitwise XOR - equal values return field with 0.

                /*
                    Explanation - compute bitwise AND with value and sign bit set to 0.
                    Next is comparing result to 0 (remove sign bit from equation).
                    Compute AND of NOT v and NOT bV.v - if they are equal (0) then corresponding field will be set to 0.
                */
                __m256 zerofx = _mm256_castsi256_ps(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(bV.v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256())
                ));

                // Fixes 0.0f == -0.0f mismatch by zeroing corresponding fields
                eq = _mm256_and_ps(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castps_si256(eq), _mm256_castps_si256(eq)) != 0;
            }


            /**
             * Compares with value for equality. This method is secured to return true when comparing 0.0f with -0.0f.
             * @param b Value to compare with.
             * @returns `true` if all fields in vectors have the same value as b, `false` otherwise.
             */
            bool operator==(const float b) {
                __m256 bV = _mm256_set1_ps(b);
                __m256 eq = _mm256_xor_ps(v, bV);

                __m256 zerofx = _mm256_castsi256_ps(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(bV, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_ps(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castps_si256(eq), _mm256_castps_si256(eq)) != 0;
            }

            /**
             * Compares with second vector for equality. This method is secured to return true when comparing 0.0f with -0.0f.
             * @param bV Second vector to compare.
             * @returns `true` if ANY field in one vector has different value than one in scond vector, `false` if vector are equal.
             */
            bool operator!=(const Float256& bV) {
                __m256 eq = _mm256_xor_ps(v, bV.v);

                __m256 zerofx = _mm256_castsi256_ps(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(bV.v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_ps(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castps_si256(eq), _mm256_castps_si256(eq)) == 0;
            }


            /**
             * Compares with second vector for equality. This method is secured to return true when comparing 0.0f with -0.0f.
             * @param b Value to compare with.
             * @returns `true` if ANY field in vector has different value than passed value, `false` if vector are equal.
             */
            bool operator!=(const float b) {
                __m256 bV = _mm256_set1_ps(b);
                __m256 eq = _mm256_xor_ps(v, bV);

                __m256 zerofx = _mm256_castsi256_ps(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(bV, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_ps(eq, zerofx);
                
                return _mm256_testz_si256(_mm256_castps_si256(eq), _mm256_castps_si256(eq)) == 0;
            }

            Float256 operator+(const Float256& bV) const noexcept{
                return _mm256_add_ps(v, bV.v);
            }

            Float256 operator+(const float b) const {
                return _mm256_add_ps(v, _mm256_set1_ps(b));
            }

            Float256& operator+=(const Float256& bV) {
                v = _mm256_add_ps(v, bV.v);
                return *this;
            }

            Float256& operator+=(const float b) {
                v = _mm256_add_ps(v, _mm256_set1_ps(b));
                return *this;
            }

            Float256 operator-(const Float256& bV) const {
                return _mm256_sub_ps(v, bV.v);
            }

            Float256 operator-(const float b) const {
                return _mm256_sub_ps(v, _mm256_set1_ps(b));
            }

            Float256& operator-=(const Float256& bV) {
                v = _mm256_sub_ps(v, bV.v);
                return *this;
            }

            Float256& operator-=(const float b) {
                v = _mm256_sub_ps(v, _mm256_set1_ps(b));
                return *this;
            }

            Float256 operator*(const Float256& bV) const {
                return _mm256_mul_ps(v, bV.v);
            }

            Float256 operator*(const float b) const {
                return _mm256_mul_ps(v, _mm256_set1_ps(b));
            }

            Float256& operator*=(const Float256& bV) {
                v = _mm256_mul_ps(v, bV.v);
                return *this;
            }

            Float256& operator*=(const float b) {
                v = _mm256_mul_ps(v, _mm256_set1_ps(b));
                return *this;
            }

            Float256 operator/(const Float256& bV) const {
                return _mm256_div_ps(v, bV.v);
            }

            Float256 operator/(const float b) const {
                return _mm256_div_ps(v, _mm256_set1_ps(b));
            }

            Float256& operator/=(const Float256& bV) {
                v = _mm256_div_ps(v, bV.v);
                return *this;
            }

            Float256& operator/=(const float b) {
                v = _mm256_div_ps(v, _mm256_set1_ps(b));
                return *this;
            }

            float operator[](const unsigned int index) const 
            #ifndef NDEBUG 
                {
                    if(index > 7)
                        throw std::invalid_argument("Invalid index! Index should be within 0-7, passed: " + std::to_string(index));
                    return ((float*)&v)[index];
                }
            #else 
                noexcept { return ((float*)&v)[index & 7]; }
            #endif

            std::string str() const noexcept {
                std::string result = "Float256(";
                float* iv = (float*)&v; 
                for(unsigned i{0}; i < 7; ++i)
                    result += std::to_string(iv[i]) + ", ";
                
                result += std::to_string(iv[7]);
                result += ")";
                return result;
            }

    };
} 


#endif