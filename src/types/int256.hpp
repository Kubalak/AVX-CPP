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
#include <chrono>

#ifndef _MSC_VER
    #include "../misc/simd_ext_gcc.h"
#endif

#include "constants.hpp"

/**
 * Namespace containing type definitions and basic functions.
 */
namespace avx
{
    /**
     * Class providing vectorized version of `int`.
     * Can hold 8 individual `int` values.
     * Provides arithmetic and bitwise operators.
     * Provides comparison operators == !=.
     * Class providing vectorized version of `int`.
     * Can hold 8 individual `int` values.
     * Provides arithmetic and bitwise operators.
     * Provides comparison operators == !=.
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

        /**
         * Initializes vector from __m256i value.
         * @param init Value of type __m256i to initialize the vector.
         */
        Int256(const __m256i &init) : v(init) {};

        /**
         * Copy constructor.
         * Initializes vector from another Int256 vector.
         * @param init Another Int256 vector to copy from.
         */
        Int256(const Int256 &init) : v(init.v) {};

        /**
         * Initializes vector from std::array of 8 int values.
         * @param init Array of 8 int values to initialize the vector.
         */
        Int256(const std::array<int, 8> &init) : v(_mm256_loadu_si256((const __m256i *)init.data())) {};

        /**
         * Initializes vector from std::array of 8 short values.
         * Each short value is promoted to int.
         * @param init Array of 8 short values to initialize the vector.
         */
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

        /**
         * Initializes vector from std::array of 8 char values.
         * Each char value is promoted to int.
         * @param init Array of 8 char values to initialize the vector.
         */
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

        /**
         * Initializes vector from initializer_list of int values.
         * If the list contains fewer than 8 elements, remaining elements are set to zero.
         * If the list contains more than 8 elements, only the first 8 are used.
         * @param init Initializer list of int values.
         */
        Int256(std::initializer_list<int> init) {
            alignas(32) int init_v[size];
            std::memset((char*)init_v, 0, 32);
            if(init.size() < size){
                auto begin = init.begin();
                for(int i{0}; i < init.size(); ++i){
                    init_v[i] = *begin;
                    begin++;
                }
            }
            else {
                auto begin = init.begin();
                for(int i{0}; i < size; ++i){
                    init_v[i] = *begin;
                    begin++;
                }
            }
            v = _mm256_load_si256((const __m256i*)init_v);
        }

        __m256i get() const { return v; }


        void set(__m256i val) { v = val; }

        /**
         * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
         * @param pSrc Pointer to memory from which to load data.
         * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
         */
        void load(const int *pSrc) N_THROW_REL {
            if(pSrc)
                v = _mm256_lddqu_si256((const __m256i*)pSrc);
        #ifndef NDEBUG
            else
                throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
        #endif
        }

        /**
         * Saves data to destination in memory.
         * @param dest Reference to the list to which vector will be saved. Array doesn't need to be aligned to any specific boundary.
         */
        void save(std::array<int, 8>& dest) const noexcept {
            _mm256_storeu_si256((__m256i*)dest.data(), v);
        }

        /**
         * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
         * 
         * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
         * @param pDest A valid pointer to a memory of at least 32 bytes (8x `int`).
         * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
         */
        void save(int *pDest) const N_THROW_REL {
            if(pDest)
                _mm256_storeu_si256((__m256i*)pDest, v);
        #ifndef NDEBUG
            else
                throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
        #endif
        }

        /**
         * Saves data to destination in memory. The memory must be aligned at 32-byte boundary.
         * 
         * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
         * @param pDest A valid pointer to a memory of at least 32 bytes (8x `int`).
         * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
         */
        void saveAligned(int *pDest) const N_THROW_REL{
            if(pDest)
                _mm256_store_si256((__m256i*)pDest, v);
        #ifndef NDEBUG
            else
                throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
        #endif
        }

        bool operator==(const Int256 &bV) const {
            __m256i eq = _mm256_xor_si256(v, bV.v);
            return _mm256_testz_si256(eq, eq) != 0;
        }

        bool operator==(const int &b) const {
            __m256i bV = _mm256_set1_epi32(b);
            __m256i eq = _mm256_xor_si256(v, bV);
            return _mm256_testz_si256(eq, eq) != 0;
        }

        bool operator!=(const Int256 &bV) const {
            __m256i eq = _mm256_xor_si256(v, bV.v);
            return _mm256_testz_si256(eq, eq) == 0;
        }

        bool operator!=(const int &b) const {
            __m256i bV = _mm256_set1_epi32(b);
            __m256i eq = _mm256_xor_si256(v, bV);
            return _mm256_testz_si256(eq, eq) == 0;
        }

        /**
        * Indexing operator.
        * Does not support value assignment through this method (e.g. aV[0] = 1 won't work).
        * @param index Position of desired element between 0 and 7.
        * @return Value of underlying element.
        * @throws std::out_of_range If index is not within the correct range and build type is debug will be thrown. Otherwise bitwise AND will prevent index to be out of range. Side effect is that in case of out of index it will behave like `index % 32`.
        */
        int operator[](const unsigned int index) const 
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
        Int256 operator+(const Int256 &bV) const { return _mm256_add_epi32(v, bV.v); };

        /**
         * Adds single value across all vector fields.
         * @return New vector being a sum of this vector and `b`.
         */
        Int256 operator+(const int &b) const { return _mm256_add_epi32(v, _mm256_set1_epi32(b)); }

        // Minus operators
        Int256 operator-(const Int256 &bV) const { return _mm256_sub_epi32(v, bV.v); };
        Int256 operator-(const int &b) const { return _mm256_sub_epi32(v, _mm256_set1_epi32(b)); }

        // Multiplication operators
        Int256 operator*(const Int256 &bV) const { return _mm256_mullo_epi32(v, bV.v); }
        Int256 operator*(const int &b) const { return _mm256_mullo_epi32(v,_mm256_set1_epi32(b)); }


        Int256 operator/(const Int256 &bV) const { 
            // ASM goes brrr ≽^•⩊•^≼
            #ifdef __AVX512F__
                return _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                return _mm256_div_epi32(v, bV.v); 
            #endif
        }
        
        Int256 operator/(const int&b) const {

            if(!b) return _mm256_setzero_si256();

            #ifdef __AVX512F__
                return _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_set1_pd(static_cast<double>(b))
                    )
                );
            #else
                return _mm256_div_epi32(v, _mm256_set1_epi32(b));
            #endif
        }

        // Modulo operators
        Int256 operator%(const Int256 &bV) const {
            #ifdef __AVX512F__
                __m256i divided = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                __m256i divided = _mm256_div_epi32(v, bV.v);
            #endif
            
            return _mm256_sub_epi32(v, _mm256_mullo_epi32(bV.v, divided));
        }


        // TODO: Fix float value limit resulting in wrong mod values.        
        Int256 operator%(const int &b) const {
            if(!b) return _mm256_setzero_si256();

            __m256i bV = _mm256_set1_epi32(b);

            #ifdef __AVX512F__
                __m256i divided = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                __m256i divided = _mm256_div_epi32(v, bV);
            #endif
            
            return _mm256_sub_epi32(v, _mm256_mullo_epi32(bV, divided));
        }    

        // XOR operators
        Int256 operator^(const Int256 &bV) const { return _mm256_xor_si256(v, bV.v); }
        Int256 operator^(const int &b) const { return _mm256_xor_si256(v, _mm256_set1_epi32(b)); }

        // OR operators
        Int256 operator|(const Int256 &bV) const {return _mm256_or_si256(v, bV.v);}
        Int256 operator|(const int &b) const { return _mm256_or_si256(v, _mm256_set1_epi32(b)); }

        // AND operators
        Int256 operator&(const Int256 &bV) const { return _mm256_and_si256(v, bV.v);}
        Int256 operator&(const int &b) const { return _mm256_and_si256(v, _mm256_set1_epi32(b)); }

        // NOT operators
        Int256 operator~() const { return _mm256_xor_si256(v, constants::ONES); }

        // Bitwise shift operations
        Int256 operator<<(const Int256 &bV) const { return _mm256_sllv_epi32(v,bV.v); }
        Int256 operator<<(const int &b) const { return _mm256_slli_epi32(v, b); }

        Int256 operator>>(const Int256 &bV) const { return _mm256_srav_epi32(v, bV.v); }
        Int256 operator>>(const int &b) const { return _mm256_srai_epi32(v, b); }

        // Calc and store operators
        Int256& operator+=(const Int256 &bV) {
            v = _mm256_add_epi32(v, bV.v);
            return *this;
        }
        
        Int256& operator+=(const int& b) {
            v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
            return *this;
        }

        Int256& operator-=(const Int256 &bV) {
            v = _mm256_sub_epi32(v, bV.v);
            return *this;
        }

        Int256 &operator-=(const int &b) {
            v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
            return *this;
        }

        Int256 &operator*=(const Int256 &bV) {
            v = _mm256_mullo_epi32(v, bV.v);
            return *this;
        }

        Int256 &operator*=(const int &b)
        {
            v = _mm256_mullo_epi32(v, _mm256_set1_epi32(b));
            return *this;
        };

        Int256 &operator/=(const Int256 &bV) {
            #ifdef __AVX512F__
                v = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                v = _mm256_div_epi32(v, bV.v); 
            #endif
            return *this;
        }

        Int256 &operator/=(const int &b) {
            #ifdef __AVX512F__
                v = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_set1_pd(static_cast<double>(b))
                    )
                );
            #else
                v = _mm256_div_epi32(v, _mm256_set1_epi32(b));
            #endif
            return *this;
        }


        /**
         * Performs integer division. IMPORTANT: Does not work for 0x8000'0000 aka -2 147 483 648
         * @param bV Second modulo operand (divisor)
         * @return Result of modulo operation.
         */
        Int256& operator%=(const Int256 &bV) {
            #ifdef __AVX512F__
                __m256i divided = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                __m256i divided = _mm256_div_epi32(v, bV.v);
            #endif
            
            v = _mm256_sub_epi32(v, _mm256_mullo_epi32(bV.v, divided));
            
            return *this;
        }

        Int256& operator%=(const int &b){
            if(!b) {
                v = _mm256_setzero_si256();
                return *this;
            }
            
            __m256i bV = _mm256_set1_epi32(b);

            #ifdef __AVX512F__
                __m256i divided = _mm512_cvttpd_epi32(
                    _mm512_div_pd(
                        _mm512_cvtepi32_pd(v), 
                        _mm512_cvtepi32_pd(bV.v)
                    )
                );
            #else
                __m256i divided = _mm256_div_epi32(v, bV);
            #endif
            
            v = _mm256_sub_epi32(v, _mm256_mullo_epi32(bV, divided));
            
            return *this;
        }

        Int256 & operator|=(const Int256 &bV){
            v = _mm256_or_si256(v, bV.v);
            return *this;
        }


        Int256 & operator|=(const int &b){
            v = _mm256_or_si256(v, _mm256_set1_epi32(b));
            return *this;
        }


        Int256 & operator&=(const Int256 &bV){
            v = _mm256_and_si256(v, bV.v);
            return *this;
        }

        Int256 &operator^=(const Int256 &bV){
            v = _mm256_xor_si256(v, bV.v);
            return *this;
        }


        Int256 &operator^=(const int &b){
            v = _mm256_xor_si256(v, _mm256_set1_epi32(b));
            return *this;
        }

        Int256 &operator<<=(const Int256 &bV) {
            v = _mm256_sllv_epi32(v, bV.v);
            return *this;
        }

        Int256 &operator<<=(const int &b) {
            v = _mm256_slli_epi32(v, b);
            return *this;
        }

        Int256 &operator>>=(const Int256 &bV) {
            v = _mm256_srav_epi32(v, bV.v);
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