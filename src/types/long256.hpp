#pragma once
#ifndef LONG256_HPP__
#define LONG256_HPP__
#include <set>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include "constants.hpp"
#include "../misc/simd_ext_gcc.h"

namespace avx {
    /**
     * Class providing vectorized version of `long long` aka `int64_t`.
     * Can hold 4 individual `long long` values.
     * Provides arithmetic and bitwise operators.
     * Provides comparison operators == !=.
     */
    class Long256 {
        private:
            __m256i v;
        
        public:

            /**
             * Number of individual values stored by object. This value can be used to iterate over elements.
            */
            static constexpr int size = 4;
            
            /**
             * Type that is stored inside vector.
             */
            using storedType = long long;

        /**
         * Default constructor. Initializes vector with zeros.
         */
        Long256() noexcept : v(_mm256_setzero_si256()) {}

        /**
         * Initializes vector by loading data from memory (via `_mm256_lddqu_si256`).
         * @param init Valid memory address of minimal size of 256-bits (32 bytes).
         */
        Long256(const long long* init) : v(_mm256_lddqu_si256((const __m256i*)init)) {}

        /**
         * Initializes vector with const value. Each cell will be set with value of `init`.
         * @param init Value to be set.
         */
        Long256(const long long& init) noexcept : v(_mm256_set1_epi64x(init)) {}

        /**
         * Initializes vector from __m256i value.
         * @param init Value of type __m256i to initialize the vector.
         */
        Long256(__m256i init) noexcept : v(init) {}

        /**
         * Copy constructor.
         * Initializes vector from another Long256 vector.
         * @param init Another Long256 vector to copy from.
         */
        Long256(Long256& init) noexcept : v(init.v) {}

        /**
         * Copy constructor (const).
         * Initializes vector from another Long256 vector.
         * @param init Another Long256 vector to copy from.
         */
        Long256(const Long256& init) : v(init.v) {}

        /**
         * Initializes vector from std::array of 4 long long values.
         * @param init Array of 4 long long values to initialize the vector.
         */
        Long256(const std::array<long long, 4>& init) noexcept : v(_mm256_lddqu_si256((const __m256i*)init.data())) {}

        /**
         * Initializes vector from std::array of 4 int values. Each int value is promoted to long long.
         * @param init Array of 4 int values to initialize the vector.
         */
        Long256(const std::array<int, 4>& init) noexcept : v(_mm256_set_epi64x(init[0], init[1], init[2], init[3])) {}

        /**
         * Initializes vector from std::array of 4 short values. Each short value is promoted to long long.
         * @param init Array of 4 short values to initialize the vector.
         */
        Long256(const std::array<short, 4>& init) noexcept : v(_mm256_set_epi64x(init[0], init[1], init[2], init[3])) {}

        /**
         * Initializes vector from std::array of 4 char values. Each char value is promoted to long long.
         * @param init Array of 4 char values to initialize the vector.
         */
        Long256(const std::array<char, 4>& init) noexcept : v(_mm256_set_epi64x(init[0], init[1], init[2], init[3])) {}

        /**
         * Initializes vector from initializer_list of long long values.
         * If the list contains fewer than 4 elements, remaining elements are set to zero.
         * If the list contains more than 4 elements, only the first 4 are used.
         * @param init Initializer list of long long values.
         */
        Long256(std::initializer_list<long long> init) noexcept {
            alignas(32) long long init_v[4];
            memset(init_v, 0, 32);
            if(init.size() < 4){
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

        /**
         * Returns the internal __m256i value stored by the object.
         * @return The __m256i value.
         */
        __m256i get() const noexcept { return v; }

        /**
         * Sets the internal __m256i value stored by the object.
         * @param val New value of type __m256i.
         */
        void set(__m256i val) noexcept { v = val; }

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param pSrc Pointer to memory from which to load data.
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void load(const long long *pSrc) N_THROW_REL {
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
            void save(std::array<long long, 4>& dest) const noexcept {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (4x `long long`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void save(long long *pDest) const N_THROW_REL {
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
             * @param pDest A valid pointer to a memory of at least 32 bytes (4x `long long`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release builds this method never throws (for `nullptr` method will have no effect).
             */
            void saveAligned(long long *pDest) const N_THROW_REL {
                if(pDest)
                    _mm256_store_si256((__m256i*)pDest, v);
            #ifndef NDEBUG
                else
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
            #endif
            }

            long long operator[](unsigned int index) const
            #ifndef NDEBUG
                {
                    if(index > 4)
                        throw std::out_of_range("Range be within range 0-3! Got: " + std::to_string(index));

                    return ((long long*)&v)[index];
                }
            #else
                noexcept { return ((long long*)&v)[index & 3]; }
            #endif

            /**
             * Compares with second vector for equality.
             * @param bV Object to compare.
             * @returns `true` if all elements are equal or `false` if not.
             */
            bool operator==(const Long256 &bV) const noexcept {
                __m256i eq = _mm256_xor_si256(v, bV.v); // Doing XOR. If all bits are the same then resulting vector should be all 0s.
                return _mm256_testz_si256(eq, eq) != 0; // Returns 1 if AND of eq and eq yields 0 (equal vectors).
            }


            /**
             * Compares with value for equality.
             * @param b Value to compare.
             * @returns `true` if all elements are equal to passed value `false` if not.
             */
            bool operator==(const long long b) const noexcept{
                __m256i bV = _mm256_set1_epi64x(b);
                __m256i eq = _mm256_xor_si256(v, bV); 
                return _mm256_testz_si256(eq, eq) != 0;
            }


            /**
             * Compares with second vector for inequality.
             * @param bV Object to compare.
             * @returns `true` if any alement is not equal to corresponding element in `bV` otherwise `false`.
             */
            bool operator!=(const Long256 &bV) const noexcept{
                __m256i eq = _mm256_xor_si256(v, bV.v);
                return _mm256_testz_si256(eq, eq) == 0;
            }


            /**
             * Compares with value for inequality.
             * @param b Value
             * @returns `true` if any alement is not equal to corresponding element in `bV` otherwise `false`.
             */
            bool operator!=(const long long b) const noexcept{
                __m256i bV = _mm256_set1_epi64x(b);
                __m256i eq = _mm256_xor_si256(v, bV);
                return _mm256_testz_si256(eq, eq) == 0;
            }           


            Long256 operator+(const Long256& bV) const noexcept {
                return _mm256_add_epi64(v, bV.v);
            }

            Long256 operator+(const long long& b) const noexcept {
                return _mm256_add_epi64(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            Long256& operator+=(const Long256& bV) noexcept {
                v = _mm256_add_epi64(v, bV.v);
                return *this;
            }

            Long256& operator+=(const long long& b) noexcept {
                v = _mm256_add_epi64(
                    v, 
                    _mm256_set1_epi64x(b)
                );
                return *this;
            }

            Long256 operator-(const Long256& bV) const noexcept {
                return _mm256_sub_epi64(
                   v, 
                    bV.v
                );
            }

            Long256 operator-(const long long& b) const noexcept {
                return _mm256_sub_epi64(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            Long256& operator-=(const Long256& bV) noexcept {
                v = _mm256_sub_epi64(
                   v, 
                    bV.v
                );

                return *this;
            }

            Long256& operator-=(const long long& b) noexcept {
                v = _mm256_sub_epi64(
                    v, 
                    _mm256_set1_epi64x(b)
                );

                return *this;
            }

            Long256 operator*(const Long256& bV) const noexcept {
                #if (defined __AVX512DQ__ && defined __AVX512VL__)
                    return _mm256_mullo_epi64(v, bV.v);
                #else
                    long long* av = (long long*)&v, *bv = (long long*)&bV.v;
                    return _mm256_set_epi64x(
                        av[3] * bv[3],
                        av[2] * bv[2],
                        av[1] * bv[1],
                        av[0] * bv[0]
                    );
                #endif
            }

            Long256 operator*(const long long& b) const noexcept {
                #if (defined __AVX512DQ__ && defined __AVX512VL__)
                    return _mm256_mullo_epi64(v, _mm256_set1_epi64x(b));
                #else
                    long long* aP = (long long*)&v;
                    return _mm256_set_epi64x(
                        aP[3] * b,
                        aP[2] * b,
                        aP[1] * b,
                        aP[0] * b
                    );
                #endif
            }

            Long256& operator*=(const Long256& bV) noexcept {
                #if (defined __AVX512DQ__ && defined __AVX512VL__)
                    v = _mm256_mullo_epi64(v, bV.v);
                #else
                    long long* av = (long long*)&v, *bv = (long long*)&bV.v;
                    v = _mm256_set_epi64x(
                        av[3] * bv[3],
                        av[2] * bv[2],
                        av[1] * bv[1],
                        av[0] * bv[0]
                    );
                #endif

                return *this;
            }

            Long256& operator*=(const long long& b) noexcept {
                #if (defined __AVX512DQ__ && defined __AVX512VL__)
                    v = _mm256_mullo_epi64(v, _mm256_set1_epi64x(b));
                #else
                    long long* aP = (long long*)&v;
                    v = _mm256_set_epi64x(
                        aP[3] * b,
                        aP[2] * b,
                        aP[1] * b,
                        aP[0] * b
                    );
                #endif
                return *this;
            }

            Long256 operator/(const Long256& bV) const noexcept {    
                #ifdef __FORCE_AVX2_           
                    return _mm256_div_epi64(v, bV.v);
                #else
                    long long* aP = (long long*)&v;
                    long long* bP = (long long*)&bV.v;
                    return _mm256_set_epi64x(
                            aP[3] / bP[3],
                            aP[2] / bP[2],
                            aP[1] / bP[1],
                            aP[0] / bP[0]
                        );
                #endif
            }

            Long256 operator/(const long long& b) const noexcept {
                #ifdef __FORCE_AVX2_
                    return _mm256_div_epi64(v, _mm256_set1_epi64x(b));
                #else 
                    long long* aP = (long long*)&v;
                    return _mm256_set_epi64x(
                            aP[3] / b,
                            aP[2] / b,
                            aP[1] / b,
                            aP[0] / b
                        );
                #endif
            }

            Long256& operator/=(const Long256& bV) noexcept {
                #ifdef __FORCE_AVX2_
                    v = _mm256_div_epi64(v, bV.v);            
                #else
                    long long* aP = (long long*)&v;
                    long long* bP = (long long*)&bV.v;
                    v = _mm256_set_epi64x(
                            aP[3] / bP[3],
                            aP[2] / bP[2],
                            aP[1] / bP[1],
                            aP[0] / bP[0]
                        );
                #endif
                return *this;
            }

            Long256& operator/=(const long long& b) noexcept {
                #ifdef __FORCE_AVX2_
                    v = _mm256_div_epi64(v, _mm256_set1_epi64x(b));
                #else
                    long long* aP = (long long*)&v;
                    v = _mm256_set_epi64x(
                            aP[3] / b,
                            aP[2] / b,
                            aP[1] / b,
                            aP[0] / b
                        );
                #endif
                return *this;
            }

            Long256 operator%(const Long256& bV) const noexcept {
                long long* a = (long long*)&v;
                long long* bv = (long long*)&bV.v;

                return _mm256_set_epi64x(
                    a[3] % bv[3],
                    a[2] % bv[2],
                    a[1] % bv[1],
                    a[0] % bv[0]
                );
            }

            Long256 operator%(const long long& b) const noexcept {
                long long* a = (long long*)&v;

                return _mm256_set_epi64x(
                    a[3] % b,
                    a[2] % b,
                    a[1] % b,
                    a[0] % b
                );
            }

            Long256& operator%=(const Long256& bV) noexcept {
                long long* a = (long long*)&v;
                long long* bv = (long long*)&bV.v;

                v = _mm256_set_epi64x(
                    a[3] % bv[3],
                    a[2] % bv[2],
                    a[1] % bv[1],
                    a[0] % bv[0]
                );

                return *this;
            }

            Long256& operator%=(const long long& b) noexcept {
                long long* a = (long long*)&v;

                v = _mm256_set_epi64x(
                    a[3] % b,
                    a[2] % b,
                    a[1] % b,
                    a[0] % b
                );

                return *this;
            }

            Long256 operator|(const Long256& bV) const noexcept {
                return _mm256_or_si256(
                    v, 
                    bV.v
                );
            }

            Long256 operator|(const long long& b) const noexcept {
                return _mm256_or_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            Long256& operator|=(const Long256& bV) noexcept {
                v = _mm256_or_si256(
                    v, 
                    bV.v
                );

                return *this;
            }

            Long256& operator|=(const long long& b) noexcept {
                v = _mm256_or_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );

                return *this;
            }

            Long256 operator^(const Long256& bV) const noexcept {
                return _mm256_xor_si256(
                    v, 
                    bV.v
                );
            }

            Long256 operator^(const long long& b) const noexcept {
                return  _mm256_xor_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            Long256& operator^=(const Long256& bV) noexcept {
                v = _mm256_xor_si256(
                    v, 
                    bV.v
                );

                return *this;
            }

            Long256& operator^=(const long long& b) noexcept {
                v = _mm256_xor_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );

                return *this;
            }

            Long256 operator&(const Long256& bV) const noexcept {
                return _mm256_and_si256(
                    v, 
                    bV.v
                );
            }

            Long256 operator&(const long long& b) const noexcept {
                return _mm256_and_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            Long256& operator&=(const Long256& bV) noexcept {
                v = _mm256_and_si256(
                    v, 
                    bV.v
                );

                return *this;
            }

            Long256& operator&=(const long long& b) noexcept {
                v = _mm256_and_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );

                return *this;
            }

            Long256 operator~() const noexcept { return _mm256_xor_si256(v, constants::ONES); }

            Long256 operator<<(const Long256& bV) const noexcept {
                return _mm256_sllv_epi64(v, bV.v);
            }

            Long256 operator<<(const unsigned int& b) const noexcept {
                return _mm256_slli_epi64(v, b);
            }

            Long256& operator<<=(const Long256& bV) noexcept {
                v = _mm256_sllv_epi64(v, bV.v);
                return *this;
            }

            Long256& operator<<=(const unsigned int& b) noexcept {
                v = _mm256_slli_epi64(v, b);
                return *this;
            }

            Long256 operator>>(const Long256& bV) const noexcept {
            #if defined(__AVX512F__) && defined(__AVX512VL__)
                return _mm256_srav_epi64(
                    v,
                    bV.v
                );
            #else
                long long* a = (long long*)&v;
                long long* bv = (long long*)&bV.v;

                return _mm256_set_epi64x(
                    a[3] >> bv[3],
                    a[2] >> bv[2],
                    a[1] >> bv[1],
                    a[0] >> bv[0]
                );
            #endif
            }

            Long256 operator>>(const unsigned int& b) const noexcept {
            #if defined(__AVX512F__) && defined(__AVX512VL__)
                return _mm256_srai_epi64(v, b);
            #else
                long long* a = (long long*)&v;

                return _mm256_set_epi64x(
                    a[3] >> b,
                    a[2] >> b,
                    a[1] >> b,
                    a[0] >> b
                );
            #endif
            }

            Long256& operator>>=(const Long256& bV) noexcept {
            #if defined(__AVX512F__) && defined(__AVX512VL__)
                v = _mm256_srav_epi64(v, bV.v);
            #else
                long long* a = (long long*)&v;
                long long* bv = (long long*)&bV.v;

                v = _mm256_set_epi64x(
                    a[3] >> bv[3],
                    a[2] >> bv[2],
                    a[1] >> bv[1],
                    a[0] >> bv[0]
                );
            #endif
                return *this;
            }

            Long256& operator>>=(const unsigned int& b) noexcept {
            #if defined(__AVX512F__) && defined(__AVX512VL__)
                v = _mm256_srai_epi64(v, b);
            #else
                long long* a = (long long*)&v;

                v = _mm256_set_epi64x(
                    a[3] >> b,
                    a[2] >> b,
                    a[1] >> b,
                    a[0] >> b
                );
            #endif
                return *this;
            }

            std::string str() const noexcept {
                std::string result = "Long256(";
                long long* iv = (long long*)&v; 
                for(unsigned i{0}; i < 3; ++i)
                    result += std::to_string(iv[i]) + ", ";
                
                result += std::to_string(iv[3]);
                result += ")";
                return result;
            }
    };

};

#endif