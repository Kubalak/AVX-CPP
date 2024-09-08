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
#include <immintrin.h>
#include <unordered_set>

#define INT256_SIZE 8

/**
 * Namespace containing type definitions and basic functions.
 */
namespace avx {
    /**
     * Vector containing 8 signed 32-bit integers.
     */
    class Int256 {
        private:
            __m256i v;
            const static __m256i ones;
        public:
            
            
            /**
             * Default constructor. Initializes vector with zeros.
            */
            Int256():v(_mm256_setzero_si256()){}
            
            
            /** Initializes vector by loading data from memory (via `_mm256_lddq_si256`). 
             * @param init Valid memory addres of minimal size of 256-bits (32 bytes).
            */
            Int256(const int*);


            /**
             * Initializes vector with const value. Each cell will be set with value of `init`.
             * @param init Value to be set.
             */
            Int256(const int& init);
            Int256(__m256i init);
            Int256(Int256& init);
            Int256(const Int256& init):v(init.v){};
            Int256(std::array<int, 8> init);
            Int256(std::array<short, 8> init);
            Int256(std::array<char, 8> init);
            Int256(std::initializer_list<int> init);
            __m256i get() const {return v;}
            void set(__m256i val){v = val;}

            /**
             * Saves vector data into an array.
             * @param dest Destination array.
             */
            void save(std::array<int, 8>&);

            
            /**
             * Saves data into given memory address. Memory doesn't need to be aligned to any specific boundary.
             * @param dest A valid (non-nullptr) memory address with size of at least 32 bytes.
             */
            void save(int*);

            /**
             * Saves data from vector into given memory address. Memory needs to be aligned on 32 byte boundary.
             * See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html for more details.
             * @param dest A valid (non-NULL) memory address aligned to 32-byte boundary.
             */
            void saveAligned(int*);

            bool operator==(const Int256&) const;
            bool operator==(const int&) const;
            bool operator!=(const Int256&) const;
            bool operator!=(const int&) const;
            const int operator[](unsigned int) const;


// Plus operators
            /**
             * Adds values from other vector and returns new vector.
             * @return New vector being a sum of this vector and `bv`.
             */
            Int256 operator+(const Int256&) const;


            /**
             * Adds single value across all vector fields.
             * @return New vector being a sum of this vector and `b`.
             */
            Int256 operator+(const int&) const;

// Minus operators
            Int256 operator-(const Int256&) const;
            Int256 operator-(const int&) const;

// Multiplication operators
            Int256 operator*(const Int256&) const;
            Int256 operator*(const int&) const;

// Division operators
//TODO: Working division and modulo (no AVX2 native solution)

            Int256 operator/(const Int256&) const;
            Int256 operator/(const int&) const;

// Modulo operators
            Int256 operator%(const Int256&) const;
            Int256 operator%(const int&) const;

// XOR operators
            Int256 operator^(const Int256&) const;
            Int256 operator^(const int&) const;

// OR operators
            Int256 operator|(const Int256&) const;
            Int256 operator|(const int&) const;

// AND operators
            Int256 operator&(const Int256&) const;
            Int256 operator&(const int&) const;

// NOT operators
            Int256 operator~() const;

// Bitwise shift operations
            Int256 operator<<(const Int256&) const;
            Int256 operator<<(const int&) const;

            Int256 operator>>(const Int256&) const;
            Int256 operator>>(const int&) const;

// Calc and store operators
            Int256& operator+=(const Int256&);
            Int256& operator+=(const int&);

            Int256& operator-=(const Int256&);
            Int256& operator-=(const int&);

            Int256& operator*=(const Int256&);
            Int256& operator*=(const int&);

            Int256& operator/=(const Int256&);
            Int256& operator/=(const int&);

            Int256& operator%=(const Int256&);
            Int256& operator%=(const int&);

            Int256& operator|=(const Int256&);
            Int256& operator|=(const int&);

            Int256& operator&=(const Int256&);
            Int256& operator&=(const int&);

            Int256& operator^=(const Int256&);
            Int256& operator^=(const int&);

            Int256& operator<<=(const Int256&);
            Int256& operator<<=(const int&);

            Int256& operator>>=(const Int256&);
            Int256& operator>>=(const int&);

            std::string str() const;

            friend Int256 sum(std::vector<Int256>&);
            friend Int256 sum(std::set<Int256>&);

    };
};
#endif