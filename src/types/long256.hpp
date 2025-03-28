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
            static constexpr int size = 4;
            using storedType = long long;

            Long256() noexcept :v(_mm256_setzero_si256()){};

            Long256(const long long* init) :
                v(_mm256_lddqu_si256((const __m256i*)init))
            {}

            Long256(const long long& init) noexcept :
                v(_mm256_set1_epi64x(init))
            {}

            Long256(__m256i init) noexcept : v(init){}

            Long256(Long256& init) noexcept : v(init.v){}

            Long256(const Long256& init):v(init.v){};

            Long256(const std::array<long long, 4>& init) noexcept :
                v(_mm256_lddqu_si256((const __m256i*)init.data()))
            {}

            Long256(const std::array<int, 4>& init) noexcept :
                    v(_mm256_set_epi64x(
                    init[0], 
                    init[1], 
                    init[2], 
                    init[3]
                    )
                )
            {}

            Long256(const std::array<short, 4>& init) noexcept :
                    v(_mm256_set_epi64x(
                    init[0], 
                    init[1], 
                    init[2], 
                    init[3]
                    )
                )
            {}

            Long256(const std::array<char, 4>& init) noexcept :
                    v(_mm256_set_epi64x(
                    init[0], 
                    init[1], 
                    init[2], 
                    init[3]
                    )
                )
            {}

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

            __m256i get() const noexcept {return v;}

            void set(__m256i val) noexcept {v = val;}

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param sP Pointer to memory from which to load data.
             */
            void load(const long long *sP) {
                if(sP != nullptr)
                    v = _mm256_lddqu_si256((const __m256i*)sP);
            }

            /**
             * Saves vector data into an array.
             * @param dest Destination array.
             */
            void save(std::array<long long, 4>& dest) const noexcept {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            
            /**
             * Saves data into given memory address. Memory doesn't need to be aligned to any specific boundary.
             * @param dest A valid (non-nullptr) memory address with size of at least 32 bytes.
             */
            void save(const long long* dest) const {
                _mm256_storeu_si256((__m256i*)dest, v);
            }

            /**
             * Saves data from vector into given memory address. Memory needs to be aligned on 32 byte boundary.
             * See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html for more details.
             * @param dest A valid (non-NULL) memory address aligned to 32-byte boundary.
             */
            void saveAligned(const long long* dest) const {
                _mm256_store_si256((__m256i*)dest, v);
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
                #ifdef _MSC_VER
                    return _mm256_div_epi64(v, bV.v);
                #else
                    long long* a = (long long*)&v;
                    long long* bv = (long long*)&bV.v;

                    return _mm256_set_epi64x(
                        a[3] / bv[3],
                        a[2] / bv[2],
                        a[1] / bv[1],
                        a[0] / bv[0]
                    );
                #endif
            }

            Long256 operator/(const long long& b) const noexcept {
                 #ifdef _MSC_VER
                    return _mm256_div_epi64(v, _mm256_set1_epi64x(b));
                #else
                    long long* a = (long long*)&v;

                    return _mm256_set_epi64x(
                        a[3] / b,
                        a[2] / b,
                        a[1] / b,
                        a[0] / b
                    );
                #endif
            }

            Long256& operator/=(const Long256& bV) noexcept {
                #ifdef _MSC_VER
                    v = _mm256_div_epi64(v, bV.v);
                #else
                    long long* a = (long long*)&v;
                    long long* bv = (long long*)&bV.v;

                    v = _mm256_set_epi64x(
                        a[3] / bv[3],
                        a[2] / bv[2],
                        a[1] / bv[1],
                        a[0] / bv[0]
                    );
                #endif
                return *this;
            }

            Long256& operator/=(const long long& b) noexcept {
                #ifdef _MSC_VER
                    v = _mm256_div_epi64(v, _mm256_set1_epi64x(b));
                #else
                    long long* a = (long long*)&v;

                    v = _mm256_set_epi64x(
                        a[3] / b,
                        a[2] / b,
                        a[1] / b,
                        a[0] / b
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
                return _mm256_srlv_epi64(
                    v,
                    bV.v
                );
            }

            Long256 operator>>(const unsigned int& b) const noexcept {
                return _mm256_srli_epi64(v, b);
            }

            Long256& operator>>=(const Long256& bV) noexcept {
                v = _mm256_srlv_epi64(v, bV.v);
                return *this;
            }

            Long256& operator>>=(const unsigned int& b) noexcept {
                v = _mm256_srli_epi64(v, b);
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

            

            friend Long256 sum(std::vector<Long256>&);
            friend Long256 sum(std::set<Long256>&);

    };

};

#endif