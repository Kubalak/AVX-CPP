#pragma once
#ifndef INT256_HPP__
#define INT256_HPP__
#include <set>
#include <array>
#include <vector>
#include <immintrin.h>
#include <unordered_set>


namespace avx {
    class Int256 {
        private:
            __m256i v;
            const static __m256i ones; 
        
        public:
            Int256():v(_mm256_setzero_si256()){}
            Int256(const int& init);
            Int256(__m256i init);
            Int256(Int256& init);
            Int256(const Int256& init):v(init.v){};
            Int256(std::array<int, 8> init);
            Int256(std::array<short, 8> init);
            Int256(std::array<char, 8> init);
            Int256(std::initializer_list<int> init);

            bool operator==(const Int256&) const;
            bool operator==(const int&) const;
            bool operator!=(const Int256&) const;
            bool operator!=(const int&) const;
            int operator[](unsigned int) const;


// Plus operators
            Int256 operator+(Int256&) const;
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

            std::string str();

            friend Int256 sum(std::vector<Int256>&);
            friend Int256 sum(std::set<Int256>&);

    };
};
#endif