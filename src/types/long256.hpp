#pragma once
#ifndef LONG256_HPP__
#define LONG256_HPP__
#include <set>
#include <array>
#include <vector>
#include <string>
#include <immintrin.h>
#include "constants.hpp"
namespace avx {
    class Long256 {
        private:
            __m256i v;
        
        public:
            static constexpr int size = 4;
            Long256():v(_mm256_setzero_si256()){};
            Long256(const long long*);
            Long256(const long long& init);
            Long256(__m256i init);
            Long256(Long256& init);
            Long256(const Long256& init):v(init.v){};
            Long256(std::array<long long, 4> init);
            Long256(std::array<int, 4> init);
            Long256(std::array<short, 4> init);
            Long256(std::array<char, 4> init);
            Long256(std::initializer_list<long long> init);
            __m256i get() const {return v;}
            void set(__m256i val){v = val;}

            bool operator==(const Long256&) const;
            bool operator==(const long long&) const;
            bool operator!=(const Long256&) const;
            bool operator!=(const long long&) const;
            long long operator[](unsigned long long) const
            #ifdef NDEBUG
                noexcept
            #endif
            ;


// Plus operators
            Long256 operator+(const Long256&) const;
            Long256 operator+(const long long&) const;

// Minus operators
            Long256 operator-(const Long256&) const;
            Long256 operator-(const long long&) const;

// Multiplication operators
            Long256 operator*(const Long256&) const;
            Long256 operator*(const long long&) const;

// Division operators
//TODO: Working division and modulo (no AVX2 native solution)

            Long256 operator/(const Long256&) const;
            Long256 operator/(const long long&) const;

// Modulo operators
            Long256 operator%(const Long256&) const;
            Long256 operator%(const long long&) const;

// XOR operators
            Long256 operator^(const Long256&) const;
            Long256 operator^(const long long&) const;

// OR operators
            Long256 operator|(const Long256&) const;
            Long256 operator|(const long long&) const;

// AND operators
            Long256 operator&(const Long256&) const;
            Long256 operator&(const long long&) const;

// NOT operators
            Long256 operator~() const;

// Bitwise shift operations
            Long256 operator<<(const Long256&) const;
            Long256 operator<<(const unsigned int&) const;

            Long256 operator>>(const Long256&) const;
            Long256 operator>>(const unsigned int&) const;

// Calc and store operators
            Long256& operator+=(const Long256&);
            Long256& operator+=(const long long&);

            Long256& operator-=(const Long256&);
            Long256& operator-=(const long long&);

            Long256& operator*=(const Long256&);
            Long256& operator*=(const long long&);

            Long256& operator/=(const Long256&);
            Long256& operator/=(const long long&);

            Long256& operator%=(const Long256&);
            Long256& operator%=(const long long&);

            Long256& operator|=(const Long256&);
            Long256& operator|=(const long long&);

            Long256& operator&=(const Long256&);
            Long256& operator&=(const long long&);

            Long256& operator^=(const Long256&);
            Long256& operator^=(const long long&);

            Long256& operator<<=(const Long256&);
            Long256& operator<<=(const unsigned int&);

            Long256& operator>>=(const Long256&);
            Long256& operator>>=(const unsigned int&);

            std::string str() const;

            /**
             * Saves vector data into an array.
             * @param dest Destination array.
             */
            void save(std::array<long long, 4>&) const;

            
            /**
             * Saves data into given memory address. Memory doesn't need to be aligned to any specific boundary.
             * @param dest A valid (non-nullptr) memory address with size of at least 32 bytes.
             */
            void save(const long long*) const;

            /**
             * Saves data from vector into given memory address. Memory needs to be aligned on 32 byte boundary.
             * See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html for more details.
             * @param dest A valid (non-NULL) memory address aligned to 32-byte boundary.
             */
            void saveAligned(const long long*) const;

            friend Long256 sum(std::vector<Long256>&);
            friend Long256 sum(std::set<Long256>&);

    };

};

#endif