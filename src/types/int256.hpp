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
#include <stdexcept>
#include <immintrin.h>
#include <unordered_set>

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
        const static __m256i ones;

    public:
        static constexpr int size = 8;

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
        Int256(std::array<int, 8> init) : v(_mm256_loadu_si256((const __m256i *)init.data())) {};
        Int256(std::array<short, 8> init) 
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

        Int256(std::array<char, 8> init) : v(_mm256_set_epi32(
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
        Int256(std::initializer_list<int> init);
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

        bool operator==(const Int256 &) const;
        bool operator==(const int &) const;
        bool operator!=(const Int256 &) const;
        bool operator!=(const int &) const;

        const int operator[](unsigned int &index) const {
            if(index > 7) {
                std::string error_text = "Invalid index! Valid range is [0-7] (was ";
                error_text += std::to_string(index);
                error_text += ").";
                throw std::out_of_range(error_text);
            }
            int* tmp = (int*)&v;
            return tmp[index];
        }

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

        Int256 operator/(const Int256&) const;
        Int256 operator/(const int&) const;

        // Modulo operators
        Int256 operator%(const Int256 &b) const {
            int* a = (int*)&v;
            int* bv = (int*)&b.v;


            return _mm256_set_epi32(
                a[7] % bv[7],
                a[6] % bv[6],
                a[5] % bv[5],
                a[4] % bv[4],
                a[3] % bv[3],
                a[2] % bv[2],
                a[1] % bv[1],
                a[0] % bv[0]  
            );
        }

        
        Int256 operator%(const int &b) const {
            int* a = (int*)&v;

            return _mm256_set_epi32(
                a[7] % b,
                a[6] % b,
                a[5] % b,
                a[4] % b,
                a[3] % b,
                a[2] % b,
                a[1] % b,
                a[0] % b
            );
        }    

        // XOR operators
        Int256 operator^(const Int256&b) const { return _mm256_xor_si256(v, b.v); }
        Int256 operator^(const int& b) const { return _mm256_xor_si256(v, _mm256_set1_epi32(b)); }

        // OR operators
        Int256 operator|(const Int256& b) const {return _mm256_or_si256(v, b.v);}
        Int256 operator|(const int& b) const { return _mm256_or_si256(v, _mm256_set1_epi32(b)); }

        // AND operators
        Int256 operator&(const Int256& b) const { return _mm256_and_si256(v, b.v);}
        Int256 operator&(const int& b) const { return _mm256_and_si256(v, _mm256_set1_epi32(b)); }

        // NOT operators
        Int256 operator~() const { return _mm256_xor_si256(v, ones); }

        // Bitwise shift operations
        Int256 operator<<(const Int256 &b) const { return _mm256_sllv_epi32(v,b.v); }
        Int256 operator<<(const int &b) const { return _mm256_slli_epi32(v, b); }

        Int256 operator>>(const Int256& b) const { return _mm256_srav_epi32(v, b.v); }
        Int256 operator>>(const int& b) const { return _mm256_srai_epi32(v, b); }

        // Calc and store operators
        Int256 &operator+=(const Int256& b) {
            v = _mm256_add_epi32(v,b.v);
            return *this;
        }
        Int256 &operator+=(const int& b) {
            v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
            return *this;
        }

        Int256 &operator-=(const Int256& b) {
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

        Int256 &operator*=(const int& b)
        {
            v = _mm256_mullo_epi32(v, _mm256_set1_epi32(b));
            return *this;
        };

        Int256 &operator/=(const Int256 &);
        Int256 &operator/=(const int &);

        Int256 &operator%=(const Int256 &b) {
            int* a = (int*)&v;
            int* bv = (int*)&b.v;

            v = _mm256_set_epi32(
                a[7] % bv[7],
                a[6] % bv[6],
                a[5] % bv[5],
                a[4] % bv[4],
                a[3] % bv[3],
                a[2] % bv[2],
                a[1] % bv[1],
                a[0] % bv[0]
            );
            return *this;
        }

        Int256 &operator%=(const int &b){
            int* a = (int*)&v;

            v = _mm256_set_epi32(
                a[7] % b,
                a[6] % b,
                a[5] % b,
                a[4] % b,
                a[3] % b,
                a[2] % b,
                a[1] % b,
                a[0] % b
            );
            return *this;
        }

        Int256& operator|=(const Int256& b){
            v = _mm256_or_si256(v, b.v);
            return *this;
        }


        Int256& operator|=(const int& b){
            v = _mm256_or_si256(v, _mm256_set1_epi32(b));
            return *this;
        }


        Int256& operator&=(const Int256& b){
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

        std::string str() const;

        friend Int256 sum(std::vector<Int256> &);
        friend Int256 sum(std::set<Int256> &);
    };
};
#endif