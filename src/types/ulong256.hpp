#pragma once
#ifndef ULONG256_HPP__
#define ULONG256_HPP__

#include <set>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include "constants.hpp"
namespace avx {
    class ULong256 {
        private:
            __m256i v;

        public:
            static constexpr int size = 4;

            ULong256() noexcept :v(_mm256_setzero_si256()){}

            ULong256(__m256i init):v(init){};
            
            ULong256(const unsigned long long& init):
                v(_mm256_set1_epi64x(init))
            {}

            ULong256(const ULong256& init):v(init.v){};
            
            ULong256(const unsigned long long* init): 
                v(_mm256_lddqu_si256((const __m256i*)init))
            {}
            
            ULong256(const std::array<unsigned long long, 4>& init):
                v(_mm256_lddqu_si256((const __m256i*)init.data()))
            {}
            
            ULong256(const std::array<unsigned int, 4>& init) noexcept :
                v(_mm256_set_epi64x(
                    init[0],
                    init[1],
                    init[2],
                    init[3]
                    )
                )
            {}
            
            ULong256(const std::array<unsigned short, 4>& init) noexcept :
                v(_mm256_set_epi64x(
                    init[0],
                    init[1],
                    init[2],
                    init[3]
                    )
                )
            {}
            
            ULong256(const std::array<unsigned char, 4>& init) noexcept :
                v(_mm256_set_epi64x(
                    init[0],
                    init[1],
                    init[2],
                    init[3]
                    )
                )
            {}
            
            ULong256(std::initializer_list<unsigned long long> init) noexcept {
                alignas(32) unsigned long long init_v[4];
                memset(init_v, 0, 32);
                if(init.size() < 4){
                    auto begin = init.begin();
                    for(int i{0}; i < init.size(); ++i)
                        init_v[i] = *begin++;
                }
                else {
                    auto begin = init.begin();
                    for(int i{0}; i < size; ++i)
                        init_v[i] = *begin++;
                }
                v = _mm256_load_si256((const __m256i*)init_v);
            }

            __m256i get() const noexcept {return v;}
            void set(__m256i val) noexcept {v = val;}

            /**
             * Saves vector data into an array.
             * @param dest Destination array.
             */
            void save(std::array<unsigned long long, 4>& dest) const noexcept {
                _mm256_storeu_si256((__m256i *)dest.data(), v);
            }

            
            /**
             * Saves data into given memory address. Memory doesn't need to be aligned to any specific boundary.
             * @param dest A valid (non-nullptr) memory address with size of at least 32 bytes.
             */
            void save(unsigned long long* dest) const 
            #ifndef NDEBUG
                {
                    if(dest == nullptr)
                        throw std::invalid_argument("Passed address is nullptr!");
                    
                    _mm256_storeu_si256((__m256i *)dest, v);
                }
            #else
                noexcept {
                    if(dest != nullptr)
                        _mm256_storeu_si256((__m256i *)dest, v);
                }
            #endif

            /**
             * Saves data from vector into given memory address. Memory needs to be aligned on 32 byte boundary.
             * See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html for more details.
             * @param dest A valid (non-NULL) memory address aligned to 32-byte boundary.
             */
            void saveAligned(unsigned long long* dest) const
            #ifndef NDEBUG
                {
                    if(dest == nullptr)
                        throw std::invalid_argument("Passed address is nullptr!");
                    
                    _mm256_store_si256((__m256i *)dest, v);
                }
            #else
                noexcept {
                    if(dest != nullptr)
                        _mm256_store_si256((__m256i *)dest, v);
                }
            #endif

            bool operator==(const ULong256& bV) const noexcept  {
                unsigned long long* v1,* v2;
                v1 = (unsigned long long*)&v;
                v2 = (unsigned long long*)&bV.v;

                for(unsigned short i{0}; i < 4; ++i)
                    if(v1[i] != v2[i])
                        return false;

                return true;
            }

            bool operator==(const unsigned long long& b) const noexcept {
                unsigned long long* v1 = (unsigned long long*)&v;

                for(unsigned short i{0}; i < 4; ++i)
                    if(v1[i] != b)
                        return false;

                return true;
            }

            bool operator!=(const ULong256& bV) const noexcept {
                unsigned long long* v1,* v2;
                v1 = (unsigned long long*)&v;
                v2 = (unsigned long long*)&bV.v;

                for(unsigned short i{0}; i < 4; ++i)
                    if(v1[i] != v2[i])
                        return true;

                return false;
            }

            bool operator!=(const unsigned long long& b) const noexcept {
                unsigned long long* v1 = (unsigned long long*)&v;

                for(unsigned short i{0}; i < 4; ++i)
                    if(v1[i] != b)
                        return true;

                return false;
            }

            unsigned long long operator[](unsigned int index) const
            #ifndef NDEBUG
                {
                    if(index > 4)
                        throw std::out_of_range("Range be within range 0-3! Got: " + std::to_string(index));
                        
                    return ((unsigned long long*)&v)[index];
                }
            #else 
             noexcept {
                return ((unsigned long long*)&v)[index & 3];
            }
            #endif


            /**
             * Adds values from other vector and returns new vector.
             * @return New vector being a sum of this vector and `bv`.
             */
            ULong256 operator+(const ULong256& bV) const noexcept {
                return _mm256_add_epi64(v, bV.v);
            }


            /**
             * Adds single value across all vector fields.
             * @return New vector being a sum of this vector and `b`.
             */
            ULong256 operator+(const unsigned long long& b) const noexcept {
                return _mm256_add_epi64(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            ULong256& operator+=(const ULong256& bV) noexcept {
                v = _mm256_add_epi64(
                    v,
                    bV.v
                );
                return *this;
            }

            ULong256& operator+=(const unsigned long long& b) noexcept {
                v = _mm256_add_epi64(
                    v,
                    _mm256_set1_epi64x(b)
                );
                return *this;
            }


            ULong256 operator-(const ULong256& bV) const noexcept {
                return _mm256_sub_epi64(v, bV.v);
            }

            ULong256 operator-(const unsigned long long& b) const noexcept {
                return _mm256_sub_epi64(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            ULong256& operator-=(const ULong256& bV) noexcept {
                v = _mm256_sub_epi64(v, bV.v);
                return *this;
            }

            ULong256& operator-=(const unsigned long long& b) noexcept {
                v = _mm256_sub_epi64(
                    v,
                    _mm256_set1_epi64x(b)
                );
                return *this;
            }

            ULong256 operator*(const ULong256& bV) const noexcept{
                #if defined(__AVX512DQ__) && defined(__AVX512VL__)
                    return _mm256_mullo_epi64(v, b.v);
                #else
                    unsigned long long* aP = (unsigned long long*)&v;
                    unsigned long long* bP = (unsigned long long*)&bV.v;
                    unsigned long long result[] = {
                        aP[0] * bP[0],
                        aP[1] * bP[1],
                        aP[2] * bP[2],
                        aP[3] * bP[3]
                    };

                    return _mm256_lddqu_si256((const __m256i*)result);
                #endif
            }

            ULong256 operator*(const unsigned long long& b) const noexcept{
                #if defined(__AVX512DQ__) && defined(__AVX512VL__)
                    return _mm256_mullo_epi64(v, _mm256_set1_epi64x(b));
                #else
                    unsigned long long* aP = (unsigned long long*)&v;
                    alignas(32) unsigned long long result[] = {
                        aP[0] * b,
                        aP[1] * b,
                        aP[2] * b,
                        aP[3] * b
                    };
                    
                    return _mm256_load_si256((const __m256i*)result);
                #endif
            }

            ULong256& operator*=(const ULong256& bV) noexcept {
                #if defined(__AVX512DQ__) && defined(__AVX512VL__)
                    v = _mm256_mullo_epi64(v, bV.v);
                #else
                    unsigned long long* aP = (unsigned long long*)&v;
                    unsigned long long* bP = (unsigned long long*)&bV.v;
                    alignas(32) unsigned long long result[] = {
                        aP[0] * bP[0],
                        aP[1] * bP[1],
                        aP[2] * bP[2],
                        aP[3] * bP[3]
                    };

                    v = _mm256_load_si256((const __m256i*)result);
                #endif

                return *this;
            }

            ULong256& operator*=(const unsigned long long& b) noexcept {
                #if defined(__AVX512DQ__) && defined(__AVX512VL__)
                    v = _mm256_mullo_epi64(v, _mm256_set1_epi64x(b));
                #else
                    unsigned long long* aP = (unsigned long long*)&v;
                    alignas(32) unsigned long long result[] = {
                        aP[0] * b,
                        aP[1] * b,
                        aP[2] * b,
                        aP[3] * b
                    };
                    
                    v = _mm256_load_si256((const __m256i*)result);
                #endif

                return *this;
            }

            ULong256 operator/(const ULong256& bV) const noexcept {
                #ifdef _MSC_VER
                    return _mm256_div_epi64(v, bV.v);
                #else
                    unsigned long long* aP = (unsigned long long*)&v;
                    unsigned long long* bP = (unsigned long long*)&bV.v;
                    unsigned long long result[] = {
                        aP[0] / bP[0],
                        aP[1] / bP[1],
                        aP[2] / bP[2],
                        aP[3] / bP[3]
                    };

                    return result;
                #endif
            }

            ULong256 operator/(const unsigned long long& b) const noexcept {
                unsigned long long* aP = (unsigned long long*)&v;
                unsigned long long result[] = {
                    aP[0] / b,
                    aP[1] / b,
                    aP[2] / b,
                    aP[3] / b
                };
                
                return result;
            }

            ULong256& operator/=(const ULong256& bV) noexcept {
                #ifdef _MSC_VER
                    v = _mm256_div_epi64(v, bV.v);
                #else
                    unsigned long long* aP = (unsigned long long*)&v;
                    unsigned long long* bP = (unsigned long long*)&bV.v;
                    alignas(32) unsigned long long result[] = {
                        aP[0] / bP[0],
                        aP[1] / bP[1],
                        aP[2] / bP[2],
                        aP[3] / bP[3]
                    };

                    v = _mm256_load_si256((__m256i*)result);
                #endif
                return *this;
            }

            ULong256& operator/=(const unsigned long long& b) noexcept {
                unsigned long long* aP = (unsigned long long*)&v;
                alignas(32) unsigned long long result[] = {
                    aP[0] / b,
                    aP[1] / b,
                    aP[2] / b,
                    aP[3] / b
                };
                v = _mm256_load_si256((const __m256i*)result);

                return *this;
            }

            ULong256 operator%(const ULong256& bV) const noexcept {
                unsigned long long* aP = (unsigned long long*)&v;
                unsigned long long* bP = (unsigned long long*)&bV.v;
                unsigned long long result[] = {
                    aP[0] % bP[0],
                    aP[1] % bP[1],
                    aP[2] % bP[2],
                    aP[3] % bP[3]
                };

                return result;
            }

            ULong256 operator%(const unsigned long long& b) const noexcept {
                unsigned long long* aP = (unsigned long long*)&v;
                unsigned long long result[] = {
                    aP[0] % b,
                    aP[1] % b,
                    aP[2] % b,
                    aP[3] % b
                };
        
                return result;
            }

            ULong256& operator%=(const ULong256& bV) noexcept {
                unsigned long long* aP = (unsigned long long*)&v;
                unsigned long long* bP = (unsigned long long*)&bV.v;
                alignas(32) unsigned long long result[] = {
                    aP[0] % bP[0],
                    aP[1] % bP[1],
                    aP[2] % bP[2],
                    aP[3] % bP[3]
                };
                v = _mm256_load_si256((const __m256i*)result);

                return *this;
            }

            ULong256& operator%=(const unsigned long long& b) noexcept {
                unsigned long long* aP = (unsigned long long*)&v;
                alignas(32) unsigned long long result[] = {
                    aP[0] % b,
                    aP[1] % b,
                    aP[2] % b,
                    aP[3] % b
                };
                v = _mm256_load_si256((const __m256i*)result);

                return *this;
            }

            ULong256 operator&(const ULong256& bV) const noexcept {
                return _mm256_and_si256(v, bV.v);
            }

            ULong256 operator&(const unsigned long long& b) const noexcept {
                return _mm256_and_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            ULong256& operator&=(const ULong256& bV) noexcept {
                v = _mm256_and_si256(v, bV.v);
                return *this;
            }

            ULong256& operator&=(const unsigned long long& b) noexcept {
                v = _mm256_and_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
                return *this;
            }

            ULong256 operator|(const ULong256& bV) const noexcept {
                return _mm256_or_si256(v, bV.v);
            }

            ULong256 operator|(const unsigned long long& b) const noexcept {
                return _mm256_or_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            ULong256& operator|=(const ULong256& bV) noexcept {
                v = _mm256_or_si256(v, bV.v);
                return *this;
            }

            ULong256& operator|=(const unsigned long long& b) noexcept {
                v = _mm256_or_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
                return *this;
            }

            ULong256 operator^(const ULong256& bV) const noexcept {
                return _mm256_xor_si256(v, bV.v);
            }

            ULong256 operator^(const unsigned long long& b) const noexcept {
                return _mm256_xor_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
            }

            ULong256& operator^=(const ULong256& bV) noexcept {
                v = _mm256_xor_si256(v, bV.v);
                return *this;
            }

            ULong256& operator^=(const unsigned long long& b) noexcept {
                 v = _mm256_xor_si256(
                    v, 
                    _mm256_set1_epi64x(b)
                );
                return *this;
            }            

            ULong256 operator~() const noexcept { return _mm256_xor_si256(v, constants::ONES); }

            ULong256 operator<<(const ULong256& bV) const noexcept {
                return _mm256_sllv_epi64(v, bV.v);
            }

            ULong256 operator<<(const unsigned long long& b) const noexcept {
                return _mm256_slli_epi64(v, b);
            }

            ULong256& operator<<=(const ULong256& bV) noexcept {
                v = _mm256_sllv_epi64(v, bV.v);
                return *this;
            }

            ULong256& operator<<=(const unsigned long long& b) noexcept {
                v = _mm256_slli_epi64(v, b);
                return *this;
            }

            ULong256 operator>>(const ULong256& bV) const noexcept {
                return _mm256_srlv_epi64(v, bV.v);
            }

            ULong256 operator>>(const unsigned long long& b) const noexcept {
                return _mm256_srli_epi64(v, b);
            }

            ULong256& operator>>=(const ULong256& bV) noexcept {
                v = _mm256_srlv_epi64(v, bV.v);
                return *this;
            }

            ULong256& operator>>=(const unsigned char& b) noexcept {
                v = _mm256_srli_epi64(v, b);
                return *this;
            }

            std::string str() const noexcept { 
                std::string result = "Long256(";
                unsigned long long* iv = (unsigned long long*)&v; 
                for(unsigned i{0}; i < 3; ++i)
                    result += std::to_string(iv[i]) + ", ";
                
                result += std::to_string(iv[3]);
                result += ")";
                return result;
            }

            friend ULong256 sum(const std::vector<ULong256>& ulongs) noexcept;

            friend ULong256 sum(const std::set<ULong256>& ulongs) noexcept;

    };
}

#endif