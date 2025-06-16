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
     * * Provides comparison operators == != (optimization on the way).
     */
    class Double256 {

        private:
            __m256d v;
        
        public:

            static constexpr int size = 4;
            using storedType = double;

            Double256() noexcept : v(_mm256_setzero_pd()){}

            Double256(const double val) noexcept : v(_mm256_set1_pd(val)){}

            Double256(const __m256d init) noexcept : v(init){}

            Double256(const Double256 &init) noexcept : v(init.v){}

            Double256(const std::array<double, 4> &init) noexcept : v(_mm256_loadu_pd(init.data())){}

            Double256(const double* addr) N_THROW_REL {
                if(addr)
                    v = _mm256_loadu_pd(addr);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif    
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
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void load(const double *pSrc) N_THROW_REL {
                if(pSrc)
                    v = _mm256_loadu_pd(pSrc);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif
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
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void save(double *pDest) const N_THROW_REL {
                if(pDest)
                    _mm256_storeu_pd(pDest, v);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif
            }

            /**
             * Saves data to destination in memory. The memory must be aligned at 32-byte boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (4x `double`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void saveAligned(double *pDest) const N_THROW_REL {
                if(pDest)
                    _mm256_store_pd(pDest, v);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif
            }

            
            const __m256d get() const noexcept { return v;}
            void set(__m256d val) noexcept { v = val;}


            bool operator==(const Double256& bV) {
                double* vP,* bP;
                vP = (double*)&v;
                bP = (double*)&bV.v;

                for(unsigned int i{0}; i < 4; ++i)
                    if(vP[i] != bP[i])
                        return false;

                return true;
            }

            bool operator==(const double b) {
                double* vP,* bP;
                vP = (double*)&v;

                for(unsigned int i{0}; i < 4; ++i)
                    if(vP[i] != b)
                        return false;

                return true;
            }

            bool operator!=(const Double256& bV) {
                double* vP,* bP;
                vP = (double*)&v;
                bP = (double*)&bV.v;

                for(unsigned int i{0}; i < 4; ++i)
                    if(vP[i] != bP[i])
                        return true;

                return false;
            }

            bool operator!=(const double b) {
                double* vP,* bP;
                vP = (double*)&v;

                for(unsigned int i{0}; i < 4; ++i)
                    if(vP[i] != b)
                        return true;

                return false;
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