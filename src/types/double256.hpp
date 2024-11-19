#pragma once
#ifndef DOUBLE256_HPP__
#define DOUBLE256_HPP__

#include <array>
#include <string>
#include <stdexcept>
#include <immintrin.h>

namespace avx {
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

            Double256(const double* addr)
            #ifndef NDEBUG
            {
                if(addr == nullptr)throw std::invalid_argument("Passed address is nullptr!");
                v = _mm256_loadu_pd(addr);
            }
            #else
            noexcept : v(_mm256_loadu_pd(addr)){} 
            #endif

            Double256(std::initializer_list<double> init) {
                alignas(32) double init_v[size]{0.0, 0.0, 0.0, 0.0};
                if(init.size() < size){
                    auto begin = init.begin();
                    for(int i{0}; i < init.size(); ++i)
                        init_v[i] = *begin++;
                }
                else {
                    auto begin = init.begin();
                    for(int i{0}; i < size; ++i)
                        init_v[i] = *begin++;
                }
                v = _mm256_load_pd(init_v);
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