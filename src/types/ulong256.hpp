#pragma once
#ifndef ULONG256_HPP__
#define ULONG256_HPP__
#include <set>
#include <string>
#include <vector>
#include <array>
#include <immintrin.h>
#include "constants.hpp"
namespace avx {
    class ULong256 {
        private:
            __m256i v;

        public:
            static constexpr int size = 4;

            ULong256():v(_mm256_setzero_si256()){}

            ULong256(__m256i init):v(init){};
            ULong256(const unsigned long long&);
            ULong256(const ULong256& init):v(init.v){};
            ULong256(const unsigned long long*);
            ULong256(const std::array<unsigned long long, 4>&);
            ULong256(const std::array<unsigned int, 4>&);
            ULong256(const std::array<unsigned short, 4>&);
            ULong256(const std::array<unsigned char, 4>&);
            ULong256(std::initializer_list<unsigned long long>);

            __m256i get() const {return v;}
            void set(__m256i val){v = val;}

            /**
             * Saves vector data into an array.
             * @param dest Destination array.
             */
            void save(std::array<unsigned long long, 4>&) const;

            
            /**
             * Saves data into given memory address. Memory doesn't need to be aligned to any specific boundary.
             * @param dest A valid (non-nullptr) memory address with size of at least 32 bytes.
             */
            void save(unsigned long long*) const;

            /**
             * Saves data from vector into given memory address. Memory needs to be aligned on 32 byte boundary.
             * See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html for more details.
             * @param dest A valid (non-NULL) memory address aligned to 32-byte boundary.
             */
            void saveAligned(unsigned long long*) const;

            bool operator==(const ULong256&) const;
            bool operator==(const unsigned long long&) const;
            bool operator!=(const ULong256&) const;
            bool operator!=(const unsigned long long&) const;
            unsigned long long operator[](unsigned int) const;


// Plus operators
            /**
             * Adds values from other vector and returns new vector.
             * @return New vector being a sum of this vector and `bv`.
             */
            ULong256 operator+(const ULong256&) const;


            /**
             * Adds single value across all vector fields.
             * @return New vector being a sum of this vector and `b`.
             */
            ULong256 operator+(const unsigned long long&) const;


            ULong256 operator-(const ULong256&) const;
            ULong256 operator-(const unsigned long long&) const;

            ULong256 operator*(const ULong256&) const;
            ULong256 operator*(const unsigned long long&) const;

            ULong256 operator/(const ULong256&) const;
            ULong256 operator/(const unsigned long long&) const;

            ULong256 operator%(const ULong256&) const;
            ULong256 operator%(const unsigned long long&) const;

            ULong256 operator&(const ULong256&) const;
            ULong256 operator&(const unsigned long long&) const;

            ULong256 operator|(const ULong256&) const;
            ULong256 operator|(const unsigned long long&) const;

            ULong256 operator^(const ULong256&) const;
            ULong256 operator^(const unsigned long long&) const;            

            ULong256 operator~() const;

            ULong256 operator<<(const ULong256&) const;
            ULong256 operator<<(const unsigned long long&) const;

            ULong256 operator>>(const ULong256&) const;
            ULong256 operator>>(const unsigned long long&) const;

            ULong256& operator+=(const ULong256&);
            ULong256& operator+=(const unsigned long long&);

            ULong256& operator-=(const ULong256&);
            ULong256& operator-=(const unsigned long long&);

            ULong256& operator*=(const ULong256&);
            ULong256& operator*=(const unsigned long long&);

            ULong256& operator/=(const ULong256&);
            ULong256& operator/=(const unsigned long long&);

            ULong256& operator%=(const ULong256&);
            ULong256& operator%=(const unsigned long long&);

            ULong256& operator|=(const ULong256&);
            ULong256& operator|=(const unsigned long long&);

            ULong256& operator&=(const ULong256&);
            ULong256& operator&=(const unsigned long long&);

            ULong256& operator^=(const ULong256&);
            ULong256& operator^=(const unsigned long long&);

            ULong256& operator<<=(const ULong256&);
            ULong256& operator<<=(const unsigned long long&);

            ULong256& operator>>=(const ULong256&);
            ULong256& operator>>=(const unsigned long long&);

            std::string str() const;

            friend ULong256 sum(const std::vector<ULong256>&);
            friend ULong256 sum(const std::set<ULong256>&);

    };
}

#endif