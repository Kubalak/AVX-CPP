#pragma once
#ifndef INT256_HPP__
#define INT256_HPP__
/**
 * @author Jakub Jach (c) 2024
 */
#include <set>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include <unordered_set>
#include "constants.hpp"

/**
 * Namespace containing type definitions and basic functions.
 */
namespace avx
{
    /**
     * Vector containing 8 signed 32-bit integers.
     */
    class Int256
    {
    private:
        __m256i v;

    public:
        static constexpr int size = 8;
        using storedType = int;

        /**
         * Default constructor. Initializes vector with zeros.
         */
        Int256() : v(_mm256_setzero_si256()) {}

        /** Initializes vector by loading data from memory (via `_mm256_lddq_si256`).
         * @param init Valid memory addres of minimal size of 256-bits (32 bytes).
         */
        Int256(const int *init) : v(_mm256_lddqu_si256((const __m256i *)init)) {};

        /**
         * Initializes vector with const value. Each cell will be set with value of `init`.
         * @param init Value to be set.
         */
        Int256(const int &init) : v(_mm256_set1_epi32(init)) {};
        Int256(const __m256i &init) : v(init) {};
        Int256(const Int256 &init) : v(init.v) {};
        Int256(const std::array<int, 8> &init) : v(_mm256_loadu_si256((const __m256i *)init.data())) {};
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

        Int256(std::initializer_list<int> init) {
            alignas(32) int init_v[size];
            std::memset((char*)init_v, 0, 32);
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
            v = _mm256_load_si256((const __m256i*)init_v);
        }

        __m256i get() const { return v; }


        void set(__m256i val) { v = val; }

        /**
         * Saves vector data into an array.
         * @param dest Destination array.
         */
        void save(std::array<int, 8> &dest) const { _mm256_storeu_si256((__m256i *)dest.data(), v); };

        /**
         * Saves data into given memory address. Memory doesn't need to be aligned to any specific boundary.
         * @param dest A valid (non-nullptr) memory address with size of at least 32 bytes.
         */
        void save(int *dest) const { _mm256_storeu_si256((__m256i *)dest, v); };

        /**
         * Saves data from vector into given memory address. Memory needs to be aligned on 32 byte boundary.
         * See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html for more details.
         * @param dest A valid (non-NULL) memory address aligned to 32-byte boundary.
         */
        void saveAligned(int* dest) const {_mm256_store_si256((__m256i*)dest, v);};

        bool operator==(const Int256 &b) const {
            int* v1,* v2;
            v1 = (int*)&v;
            v2 = (int*)&b.v;

            for(unsigned short i{0}; i < 8; ++i)
                if(v1[i] != v2[i])
                    return false;

            return true;
        }

        bool operator==(const int &b) const {
            int* v1 = (int*)&v;

            for(unsigned short i{0}; i < 8; ++i)
                if(v1[i] != b)
                    return false;

            return true;
        }

        bool operator!=(const Int256 &b) const {
            int* v1,* v2;
            v1 = (int*)&v;
            v2 = (int*)&b.v;

            for(unsigned short i{0}; i < 8; ++i)
                if(v1[i] != v2[i])
                    return true;

            return false;
        }

        bool operator!=(const int &b) const {
            int* v1 = (int*)&v;

            for(unsigned short i{0}; i < 8; ++i)
                if(v1[i] != b)
                    return true;

            return false;
        }

        int operator[](unsigned int &index) const 
        #ifndef NDEBUG
            {
                if(index > 7) 
                    throw std::out_of_range("Range be within range 0-7! Got: " + std::to_string(index));

                return ((int*)&v)[index];
            }
        #else
            noexcept { return ((int*)&v)[index & 7];}
        #endif 

        // Plus operators
        /**
         * Adds values from other vector and returns new vector.
         * @return New vector being a sum of this vector and `bv`.
         */
        Int256 operator+(const Int256 &b) const { return _mm256_add_epi32(v, b.v); };

        /**
         * Adds single value across all vector fields.
         * @return New vector being a sum of this vector and `b`.
         */
        Int256 operator+(const int &b) const { return _mm256_add_epi32(v, _mm256_set1_epi32(b)); }

        // Minus operators
        Int256 operator-(const Int256 &b) const { return _mm256_sub_epi32(v, b.v); };
        Int256 operator-(const int &b) const { return _mm256_sub_epi32(v, _mm256_set1_epi32(b)); }

        // Multiplication operators
        Int256 operator*(const Int256 &b) const { return _mm256_mullo_epi32(v, b.v); }
        Int256 operator*(const int &b) const { return _mm256_mullo_epi32(v,_mm256_set1_epi32(b)); }

        // Division operators
        // TODO: Working division and modulo (no AVX2 native solution)

        Int256 operator/(const Int256 &b) const {
            return _mm256_cvtps_epi32(
                _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(b.v)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
            );
        }
        Int256 operator/(const int&b) const {
            return _mm256_cvtps_epi32(
                _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(b)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
            );
        }

        // Modulo operators
        Int256 operator%(const Int256 &b) const {
            __m256i divided = _mm256_cvtps_epi32(
                _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(b.v)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
            );

            return _mm256_sub_epi32(
                v,
                _mm256_mullo_epi32(b.v, divided)
            );
        }

        
        Int256 operator%(const int &b) const {
            if(b) {
                __m256i divisor = _mm256_set1_epi32(b);

                __m256i divided = _mm256_cvtps_epi32(
                    _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(divisor)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
                );

                return _mm256_sub_epi32(
                    v,
                    _mm256_mullo_epi32(divisor, divided)
                );
            }
            else 
                return _mm256_setzero_si256();
        }    

        // XOR operators
        Int256 operator^(const Int256 &b) const { return _mm256_xor_si256(v, b.v); }
        Int256 operator^(const int &b) const { return _mm256_xor_si256(v, _mm256_set1_epi32(b)); }

        // OR operators
        Int256 operator|(const Int256 &b) const {return _mm256_or_si256(v, b.v);}
        Int256 operator|(const int &b) const { return _mm256_or_si256(v, _mm256_set1_epi32(b)); }

        // AND operators
        Int256 operator&(const Int256 &b) const { return _mm256_and_si256(v, b.v);}
        Int256 operator&(const int &b) const { return _mm256_and_si256(v, _mm256_set1_epi32(b)); }

        // NOT operators
        Int256 operator~() const { return _mm256_xor_si256(v, constants::ONES); }

        // Bitwise shift operations
        Int256 operator<<(const Int256 &b) const { return _mm256_sllv_epi32(v,b.v); }
        Int256 operator<<(const int &b) const { return _mm256_slli_epi32(v, b); }

        Int256 operator>>(const Int256 &b) const { return _mm256_srav_epi32(v, b.v); }
        Int256 operator>>(const int &b) const { return _mm256_srai_epi32(v, b); }

        // Calc and store operators
        Int256& operator+=(const Int256 & b) {
            v = _mm256_add_epi32(v,b.v);
            return *this;
        }
        
        Int256& operator+=(const int& b) {
            v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
            return *this;
        }

        Int256& operator-=(const Int256 &b) {
            v = _mm256_sub_epi32(v, b.v);
            return *this;
        }

        Int256 &operator-=(const int &b) {
            v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
            return *this;
        }

        Int256 &operator*=(const Int256 &b) {
            v = _mm256_mullo_epi32(v, b.v);
            return *this;
        }

        Int256 &operator*=(const int &b)
        {
            v = _mm256_mullo_epi32(v, _mm256_set1_epi32(b));
            return *this;
        };

        Int256 &operator/=(const Int256 &b) {
            v = _mm256_cvtps_epi32(
                _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(b.v)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
            );
            return *this;
        }
        Int256 &operator/=(const int &b) {
            v = _mm256_cvtps_epi32(
                _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(b)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
            );
        return *this;
        }

        Int256 &operator%=(const Int256 &b) {
            __m256i divided = _mm256_cvtps_epi32(
                _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(b.v)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
            );

            v = _mm256_sub_epi32(
                v,
                _mm256_mullo_epi32(b.v, divided)
            );
            return *this;
        }

        Int256 &operator%=(const int &b){
            if(b) {
                __m256i divisor = _mm256_set1_epi32(b);

                __m256i divided = _mm256_cvtps_epi32(
                    _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(divisor)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
                );

                v = _mm256_sub_epi32(
                    v,
                    _mm256_mullo_epi32(divisor, divided)
                );
            }
            else 
                v = _mm256_setzero_si256();
            return *this;
        }

        Int256 & operator|=(const Int256 &b){
            v = _mm256_or_si256(v, b.v);
            return *this;
        }


        Int256 & operator|=(const int &b){
            v = _mm256_or_si256(v, _mm256_set1_epi32(b));
            return *this;
        }


        Int256 & operator&=(const Int256 &b){
            v = _mm256_and_si256(v, b.v);
            return *this;
        }

        Int256 &operator^=(const Int256 &b){
            v = _mm256_xor_si256(v, b.v);
            return *this;
        }


        Int256 &operator^=(const int &b){
            v = _mm256_xor_si256(v, _mm256_set1_epi32(b));
            return *this;
        }

        Int256 &operator<<=(const Int256 &b) {
            v = _mm256_sllv_epi32(v, b.v);
            return *this;
        }

        Int256 &operator<<=(const int &b) {
            v = _mm256_slli_epi32(v, b);
            return *this;
        }

        Int256 &operator>>=(const Int256 &b) {
            v = _mm256_srav_epi32(v, b.v);
            return *this;
        }

        Int256 &operator>>=(const int &b) {
            v = _mm256_srai_epi32(v, b);
            return *this;
        }

        std::string str() const {
            std::string result = "Int256(";
            int* iv = (int*)&v; 
            for(unsigned i{0}; i < 7; ++i)
                result += std::to_string(iv[i]) + ", ";
            
            result += std::to_string(iv[7]);
            result += ")";
            return result;
        }

        friend Int256 sum(std::vector<Int256> &);
        friend Int256 sum(std::set<Int256> &);
    };
};
#endif