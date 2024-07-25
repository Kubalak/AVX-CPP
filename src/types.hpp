#pragma once
#ifndef TYPES_HPP_
#define TYPES_HPP_
#include <immintrin.h>
#include <vector>
#include <array>

namespace avx {
    class Int256 {
        private:
            __m256i v;
        
        public:
            Int256(__m256i& init);
            Int256(Int256& init);
            Int256(std::array<int, 8> init);
            Int256(std::array<short, 16> init);
            Int256(std::array<char, 32> init);
            Int256(std::initializer_list<int> init);

            bool operator==(const Int256& b);
            bool operator!=(const Int256& b);


// Plus operators
            Int256 operator+(const Int256& b);
            Int256 operator+(const int& b);
            Int256 operator+(const short& b);
// Minus operators
            Int256 operator-(const Int256& b);
// Multiplication operators
            Int256 operator*(const Int256& b);
// Division operators
            Int256 operator/(const Int256& b);
// Modulo operators
            Int256 operator%(const Int256& b);
// XOR operators
            Int256 operator^(const Int256& b);
// OR operators
            Int256 operator|(const Int256& b);
// AND operators
            Int256 operator&(const Int256& b);
// NOT operators
            Int256 operator!();
    };

    class UInt256 {

    };


    class Short256 {

    };


    class UShort256 {

    };


    class Long256 {

    };


    class ULong256 {

    };


    class Char256 {

    };


    class UChar256{

    };


    class Float256 {

    };


    class Double256 {

    };
    
};


#endif