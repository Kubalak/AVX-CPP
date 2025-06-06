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
             * @param sP Pointer to memory from which to load data.
             */
            void load(const int *sP) {
                if(sP != nullptr)
                    v = _mm256_lddqu_si256((const __m256i*)sP);
            }

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
        void saveAligned(int *dest) const {_mm256_store_si256((__m256i*)dest, v);};

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
            // Last number that can be represented by float without approx: 16777216 -> 0x100'0000
            // RIP to all who were victims of this ≽^•⩊•^≼
            __m256i vGLimit = _mm256_cmpgt_epi32(_mm256_abs_epi32(v), constants::FLOAT_LIMIT);
            __m256i bGLimit = _mm256_cmpgt_epi32(_mm256_abs_epi32(bV.v), constants::FLOAT_LIMIT);
            bGLimit = _mm256_or_si256(bGLimit, vGLimit); // Optimized for speed (see: _mm256_testz_si256)

            if(!_mm256_testz_si256(bGLimit, bGLimit)){
                #ifdef _MSC_VER
                    return _mm256_div_epi32(v, bV.v);
                #elif defined(__GNUC__)
                    /*
                    TODO: Implement faster divider - SVML not available for GCC/Clang
                    alignas(32) int result[8];

                    _mm256_store_si256(
                        (__m256i*)result, 
                        _mm256_cvttps_epi32(
                            _mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(bV.v))
                        )
                    );

                    for(int i{0}; i < 32; i+=4){
                        unsigned char index = i >> 2;
                        if(((unsigned char*)&bGLimit)[i])
                            result[index] = (((int*)&v)[index] / ((int*)&bV.v)[index]);
                    }

                    return _mm256_load_si256((const __m256i*)result);
                    */

                    return _mm256_cvttps_epi32(
                        _mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(bV.v))
                    );
                #else
                    static_assert(false, "Functionality for this compiler is not implemented! Verify if GNU solution works correctly!");
                #endif
            }
            
            return _mm256_cvttps_epi32(
                _mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(bV.v))
            );
        }
        
        Int256 operator/(const int&b) const {

            if(!b) return _mm256_setzero_si256();

            return _mm256_cvttps_epi32(
                _mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(b))
            );
        }

        // Modulo operators
        Int256 operator%(const Int256 &bV) const {
            __m256i sub_zero = _mm256_cmpgt_epi32(_mm256_setzero_si256(), v);

            __m256i absV = _mm256_abs_epi32(v);
            __m256i absVB = _mm256_abs_epi32(bV.v);

            __m256i divided = _mm256_cvttps_epi32(
                _mm256_div_ps(
                    _mm256_cvtepi32_ps(absV),
                    _mm256_cvtepi32_ps(absVB)
                )
            );

            absV = _mm256_sub_epi32(absV, _mm256_mullo_epi32(absVB, divided));

            // Correction if division returns too high number
            __m256i mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), absV);
            while(!_mm256_testz_si256(mask, mask)){
                absV = _mm256_add_epi32(absV, _mm256_and_si256(absVB, mask));
                mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), absV);
            }
            
            // Correction if division returns too small number
            mask = _mm256_cmpgt_epi32(absV, absVB);
            while(!_mm256_testz_si256(mask, mask)){
                absV = _mm256_sub_epi32(absV, _mm256_and_si256(absVB, mask));
                mask = _mm256_cmpgt_epi32(absV, absVB);
            }

            absV = _mm256_xor_si256(absV, sub_zero);
            return _mm256_add_epi32(absV, _mm256_and_si256(sub_zero, constants::EPI32_ONE));
        }


        // TODO: Fix float value limit resulting in wrong mod values.        
        Int256 operator%(const int &b) const {
            if(b) {
                __m256i sub_zero = _mm256_cmpgt_epi32(_mm256_setzero_si256(), v);

                __m256i absV = _mm256_abs_epi32(v);
                __m256i absVB = _mm256_abs_epi32(_mm256_set1_epi32(b));
    
                __m256i divided = _mm256_cvttps_epi32(
                    _mm256_div_ps(
                        _mm256_cvtepi32_ps(absV),
                        _mm256_cvtepi32_ps(absVB)
                    )
                );
    
                absV = _mm256_sub_epi32(absV, _mm256_mullo_epi32(absVB, divided));
                
                // Correction if division returns too high number
                __m256i mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), absV);
                while(!_mm256_testz_si256(mask, mask)){
                    absV = _mm256_add_epi32(absV, _mm256_and_si256(absVB, mask));
                    mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), absV);
                }
                
                // Correction if division returns too small number
                mask = _mm256_cmpgt_epi32(absV, absVB);
                while(!_mm256_testz_si256(mask, mask)){
                    absV = _mm256_sub_epi32(absV, _mm256_and_si256(absVB, mask));
                    mask = _mm256_cmpgt_epi32(absV, absVB);
                }

                absV = _mm256_xor_si256(absV, sub_zero);
                return _mm256_add_epi32(absV, _mm256_and_si256(sub_zero, constants::EPI32_ONE));
            }
            else 
                return _mm256_setzero_si256();
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
            __m256i vGLimit = _mm256_cmpgt_epi32(_mm256_abs_epi32(v), constants::FLOAT_LIMIT);
            __m256i bGLimit = _mm256_cmpgt_epi32(_mm256_abs_epi32(bV.v), constants::FLOAT_LIMIT);
            bGLimit = _mm256_or_si256(bGLimit, vGLimit); // Optimized for speed (see: _mm256_testz_si256)

            if(!_mm256_testz_si256(bGLimit, bGLimit)){
                #ifdef _MSC_VER
                    v = _mm256_div_epi32(v, bV.v);
                #elif defined(__GNUC__)
                    v = _mm256_cvttps_epi32(
                        _mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(bV.v))
                    );
                #else
                    static_assert(false, "Functionality for this compiler is not implemented! Verify if GNU solution works correctly!");
                #endif
            }
            else  
                v = _mm256_cvttps_epi32(
                    _mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(bV.v))
                );
            return *this;
        }
        Int256 &operator/=(const int &b) {
            v = _mm256_cvttps_epi32(
                _mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(b))
            );
        return *this;
        }


        /* OLD IMPLEMENTATION of % - AS WELL BROKEN FOR FLOAT CASTING AND SLOWER

        __m256i sub_zero = _mm256_and_si256(v, bV.v);
            sub_zero = _mm256_and_si256(sub_zero, constants::EPI32_SIGN);
            __m256i one = _mm256_srli_epi32(sub_zero, 31);
            sub_zero = _mm256_srai_epi32(sub_zero, 31);

            __m256i safeb = _mm256_add_epi32(_mm256_xor_si256(bV.v, sub_zero), one);
            __m256i safev = _mm256_add_epi32(_mm256_xor_si256(v, sub_zero), one);

            __m256i divided = _mm256_cvttps_epi32(
                _mm256_div_ps(_mm256_cvtepi32_ps(safev), _mm256_cvtepi32_ps(safeb))
            );

            __m256i multiplied = _mm256_mullo_epi32(safeb, divided);

            // Creating a fix for float casting rounding problem
            __m256i mask = _mm256_cmpgt_epi32(safev, _mm256_setzero_si256());
            mask = _mm256_and_si256(mask, _mm256_cmpgt_epi32(safeb, _mm256_setzero_si256()));

            __m256i gt = _mm256_cmpgt_epi32(multiplied, safev);
            gt = _mm256_xor_si256(multiplied, gt);
            gt = _mm256_srai_epi32(gt, 31);

            __m256i f_fix_mask = _mm256_and_si256(gt, safeb);
            f_fix_mask = _mm256_and_si256(f_fix_mask, mask);

            multiplied = _mm256_sub_epi32(multiplied, f_fix_mask);

            __m256i result = _mm256_sub_epi32(safev, multiplied);

            v = _mm256_add_epi32(_mm256_xor_si256(result, sub_zero), one);
        
        */

        /**
         * Performs integer division. IMPORTANT: Does not work for 0x8000'0000 aka -2 147 483 648
         * @param bV Second modulo operand (divisor)
         * @return Result of modulo operation.
         */
        Int256 &operator%=(const Int256 &bV) {
            __m256i sub_zero = _mm256_cmpgt_epi32(_mm256_setzero_si256(), v);

            __m256i absV = _mm256_abs_epi32(v);
            __m256i absVB = _mm256_abs_epi32(bV.v);

            __m256i divided = _mm256_cvttps_epi32(
                _mm256_div_ps(
                    _mm256_cvtepi32_ps(absV),
                    _mm256_cvtepi32_ps(absVB)
                )
            );

            absV = _mm256_sub_epi32(absV, _mm256_mullo_epi32(absVB, divided));
            
            // Correction if division returns too high number
            __m256i mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), absV);
            while(!_mm256_testz_si256(mask, mask)){
                absV = _mm256_add_epi32(absV, _mm256_and_si256(absVB, mask));
                mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), absV);
            }
            
            // Correction if division returns too small number
            mask = _mm256_cmpgt_epi32(absV, absVB);
            while(!_mm256_testz_si256(mask, mask)){
                absV = _mm256_sub_epi32(absV, _mm256_and_si256(absVB, mask));
                mask = _mm256_cmpgt_epi32(absV, absVB);
            }

            absV = _mm256_xor_si256(absV, sub_zero);
            v = _mm256_add_epi32(absV, _mm256_and_si256(sub_zero, constants::EPI32_ONE));
            
            return *this;
        }

        Int256 &operator%=(const int &b){
            if(b) {
                __m256i sub_zero = _mm256_cmpgt_epi32(_mm256_setzero_si256(), v);

                __m256i absV = _mm256_abs_epi32(v);
                __m256i absVB = _mm256_abs_epi32(_mm256_set1_epi32(b));
    
                __m256i divided = _mm256_cvttps_epi32(
                    _mm256_div_ps(
                        _mm256_cvtepi32_ps(absV),
                        _mm256_cvtepi32_ps(absVB)
                    )
                );
    
                absV = _mm256_sub_epi32(absV, _mm256_mullo_epi32(absVB, divided));
                
                // Correction if division returns too high number
                __m256i mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), absV);
                while(!_mm256_testz_si256(mask, mask)){
                    absV = _mm256_add_epi32(absV, _mm256_and_si256(absVB, mask));
                    mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), absV);
                }
                
                // Correction if division returns too small number
                mask = _mm256_cmpgt_epi32(absV, absVB);
                while(!_mm256_testz_si256(mask, mask)){
                    absV = _mm256_sub_epi32(absV, _mm256_and_si256(absVB, mask));
                    mask = _mm256_cmpgt_epi32(absV, absVB);
                }

                absV = _mm256_xor_si256(absV, sub_zero);
                v = _mm256_add_epi32(absV, _mm256_and_si256(sub_zero, constants::EPI32_ONE));
            }
            else 
                v = _mm256_setzero_si256();
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