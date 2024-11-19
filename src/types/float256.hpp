#pragma once
#ifndef FLOAT256_HPP__
#define FLOAT256_HPP__
#include <array>
#include <string>
#include <stdexcept>
#include <immintrin.h>

namespace avx {
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
            Float256(const float* addr) : v(_mm256_loadu_ps(addr)){}

            const __m256 get() const noexcept { return v;}
            void set(__m256 val) noexcept { v = val;}

            void save(float *dest) const {
                _mm256_storeu_ps(dest, v);
            }

            void save(std::array<float, 8> &dest) const noexcept{
                _mm256_storeu_ps(dest.data(), v);
            }

            void saveAligned(float* dest) const {
                _mm256_store_ps(dest, v);
            }

            bool operator==(const Float256& bV) {
                float* vP,* bP;
                vP = (float*)&v;
                bP = (float*)&bV.v;

                for(unsigned int i{0}; i < 8; ++i)
                    if(vP[i] != bP[i])
                        return false;

                return true;
            }

            bool operator==(const float b) {
                float* vP,* bP;
                vP = (float*)&v;

                for(unsigned int i{0}; i < 8; ++i)
                    if(vP[i] != b)
                        return false;

                return true;
            }

            bool operator!=(const Float256& bV) {
                float* vP,* bP;
                vP = (float*)&v;
                bP = (float*)&bV.v;

                for(unsigned int i{0}; i < 8; ++i)
                    if(vP[i] != bP[i])
                        return true;

                return false;
            }

            bool operator!=(const float b) {
                float* vP,* bP;
                vP = (float*)&v;

                for(unsigned int i{0}; i < 8; ++i)
                    if(vP[i] != b)
                        return true;

                return false;
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