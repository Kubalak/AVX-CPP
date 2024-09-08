#pragma once
#ifndef UINT256_HPP__
#define UINT256_HPP__
#include <set>
#include <array>
#include <vector>
#include <string>
#include <immintrin.h>

#define UINT256_SIZE 8
namespace avx {
    class UInt256 {
        private:
            __m256i v;
            const static __m256i ones; 
            const static __m256i crate;
        
        public:
            /**
             * Default constructor.
             * Initializes vector with zeros using `_mm256_setzero_si256()`.
             */
            UInt256():v(_mm256_setzero_si256()){}

            /**
             * Fills vector with passed value using `_mm256_set1_epi32()`.
             * @param init Value to be set.
             */
            UInt256(const unsigned int& init);

            /**
             * Just sets vector value using passed `__m256i`.
             * @param init Vector value to be set.
             */
            UInt256(__m256i init);

            /**
             * Set value using reference.
             * @param init Reference to object which value will be copied.
             */
            UInt256(UInt256& init);

            /**
             * Set value using const reference.
             * @param init Const reference to object which value will be copied.
             */
            UInt256(const UInt256& init):v(init.v){};

            /** Sets vector values using array of unsigned integers.
             * 
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned integers which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned int, 8> init);

            /** Sets vector values using array of unsigned shorts.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned shorts which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned short, 8> init);

            /** Sets vector values using array of unsigned chars.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned chars which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned char, 8> init);

            /** Sets vector values using array of unsigned integers.
             * If list is longer than 8 other values will be ignored.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Initlizer list containing unsigned integers which values will be assigned to vector fields.
             * @throws `std::invalid_argument` When initializer list length is lower than 8.
             */
            UInt256(std::initializer_list<unsigned int> init);

            /**
             * Get underlying `__m256i` vector.
             * @returns Copy of internal `__m256i` vector.
             */
            __m256i get() const {return v;}

            /**
             * Set underlying `__m256i` vector value.
             * @param val Vector which value will be copied.
             */
            void set(__m256i val){v = val;}

            bool operator==(const UInt256&) const;
            bool operator==(const unsigned int&) const;
            bool operator!=(const UInt256&) const;
            bool operator!=(const unsigned int&) const;
            const unsigned int operator[](unsigned int) const;


// Plus operators
            UInt256 operator+(const UInt256&) const;
            UInt256 operator+(const unsigned int&) const;

// Minus operators
            UInt256 operator-(const UInt256&) const;
            UInt256 operator-(const unsigned int&) const;

// Multiplication operators
            UInt256 operator*(const UInt256&) const;
            UInt256 operator*(const unsigned int&) const;

// Division operators
//TODO: Working division and modulo (no AVX2 native solution)

            UInt256 operator/(const UInt256&) const;
            UInt256 operator/(const unsigned int&) const;

// Modulo operators
            UInt256 operator%(const UInt256&) const;
            UInt256 operator%(const unsigned int&) const;

// XOR operators
            UInt256 operator^(const UInt256&) const;
            UInt256 operator^(const unsigned int&) const;

// OR operators
            UInt256 operator|(const UInt256&) const;
            UInt256 operator|(const unsigned int&) const;

// AND operators
            UInt256 operator&(const UInt256&) const;
            UInt256 operator&(const unsigned int&) const;

// NOT operators
            UInt256 operator~() const;

// Bitwise shift operations
            UInt256 operator<<(const UInt256&) const;
            UInt256 operator<<(const unsigned int&) const;

            UInt256 operator>>(const UInt256&) const;
            UInt256 operator>>(const unsigned int&) const;

// Calc and store operators
            UInt256& operator+=(const UInt256&);
            UInt256& operator+=(const unsigned int&);

            UInt256& operator-=(const UInt256&);
            UInt256& operator-=(const unsigned int&);

            UInt256& operator*=(const UInt256&);
            UInt256& operator*=(const unsigned int&);

            UInt256& operator/=(const UInt256&);
            UInt256& operator/=(const unsigned int&);

            UInt256& operator%=(const UInt256&);
            UInt256& operator%=(const unsigned int&);

            UInt256& operator|=(const UInt256&);
            UInt256& operator|=(const unsigned int&);

            UInt256& operator&=(const UInt256&);
            UInt256& operator&=(const unsigned int&);

            UInt256& operator^=(const UInt256&);
            UInt256& operator^=(const unsigned int&);

            UInt256& operator<<=(const UInt256&);
            UInt256& operator<<=(const unsigned int&);

            UInt256& operator>>=(const UInt256&);
            UInt256& operator>>=(const unsigned int&);

            /**
             * Return string representation of vector.
             * 
             * @returns String representation of vector.
             */
            std::string str() const;


            /**
             * Saves vector data into an array.
             * @param dest Destination array.
             */
            void save(std::array<unsigned int, 8>&);

            
            /**
             * Saves data into given memory address. Memory doesn't need to be aligned to any specific boundary.
             * @param dest A valid (non-nullptr) memory address with size of at least 32 bytes.
             */
            void save(unsigned int*);

            /**
             * Saves data from vector into given memory address. Memory needs to be aligned on 32 byte boundary.
             * See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html for more details.
             * @param dest A valid (non-NULL) memory address aligned to 32-byte boundary.
             */
            void saveAligned(unsigned int*);

            
            /**
             * Sums all elements in vector.
             * @param items Vector containing `UInt256` values.
             * @return Sum of all elements in vector.
             */
            friend UInt256 sum(std::vector<UInt256>&);

            
            /**
             * Sums all elements in set.
             * @param items Set containing `UInt256` values.
             * @return Sum of all elements in set.
             */
            friend UInt256 sum(std::set<UInt256>&);

    };
};


#endif