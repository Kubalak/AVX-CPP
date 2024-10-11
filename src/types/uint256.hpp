#pragma once
#ifndef UINT256_HPP__
#define UINT256_HPP__

#include <set>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>

#define UINT256_SIZE 8
namespace avx {
    class UInt256 {
        private:
            __m256i v;
            const static __m256i ones; 
            const static __m256i crate;
        
        public:

            static constexpr int size = 8;

            /**
             * Default constructor.
             * Initializes vector with zeros using `_mm256_setzero_si256()`.
             */
            UInt256():v(_mm256_setzero_si256()){}

            /** Initializes vector by loading data from memory (via `_mm256_lddq_si256`). 
             * @param init Valid memory addres of minimal size of 256-bits (32 bytes).
            */
            UInt256(const unsigned int* init) 
            #ifndef NDEBUG
                {
                    if(init == nullptr) throw std::invalid_argument("Passed address is nullptr!");
                    v = _mm256_lddqu_si256((const __m256i*)init);
                }
            #else 
            noexcept :
                v(_mm256_lddqu_si256((const __m256i*)init))
            {}
            #endif

            /**
             * Fills vector with passed value using `_mm256_set1_epi32()`.
             * @param init Value to be set.
             */
            UInt256(const unsigned int& init) noexcept : v(_mm256_set1_epi32(init))
            {}

            /**
             * Just sets vector value using passed `__m256i`.
             * @param init Vector value to be set.
             */
            UInt256(__m256i init) noexcept : v(init)
            {}

            /**
             * Set value using reference.
             * @param init Reference to object which value will be copied.
             */
            UInt256(UInt256& init) noexcept : v(init.v)
            {}

            /**
             * Set value using const reference.
             * @param init Const reference to object which value will be copied.
             */
            UInt256(const UInt256& init) noexcept : v(init.v){};

            /** Sets vector values using array of unsigned integers.
             * 
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned integers which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned int, 8> init) noexcept :
                v(_mm256_lddqu_si256((const __m256i*)init.data()))
            {}

            /** Sets vector values using array of unsigned shorts.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned shorts which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned short, 8> init) noexcept :
                v(_mm256_set_epi32(
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

            /** Sets vector values using array of unsigned chars.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Array of unsigned chars which values will be assigned to vector fields.
             */
            UInt256(std::array<unsigned char, 8> init) noexcept :
                v(_mm256_set_epi32(
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

            /** Sets vector values using array of unsigned integers.
             * If list is longer than 8 other values will be ignored.
             * When accessing vector fields using `[]` order of values will be inversed.
             * @param init Initlizer list containing unsigned integers which values will be assigned to vector fields.
             * @throws `std::invalid_argument` When initializer list length is lower than 8.
             */
            UInt256(std::initializer_list<unsigned int> init) noexcept {
                alignas(32) unsigned int init_v[size];
                std::memset(init_v, 0, 32);
                if(init.size() < size){
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

            /**
             * Get underlying `__m256i` vector.
             * @returns Copy of internal `__m256i` vector.
             */
            __m256i get() const noexcept {return v;}

            /**
             * Set underlying `__m256i` vector value.
             * @param val Vector which value will be copied.
             */
            void set(__m256i val) noexcept {v = val;}

            /**
             * Saves vector data into an array.
             * @param dest Destination array.
             */
            void save(std::array<unsigned int, 8>& dest) const noexcept {
                _mm256_storeu_si256((__m256i*)dest.data(), v);
            }

            
            /**
             * Saves data into given memory address. Memory doesn't need to be aligned to any specific boundary.
             * @param dest A valid (non-nullptr) memory address with size of at least 32 bytes.
             */
            void save(unsigned int* dest) const {
            #ifndef NDEBUG
                if(dest == nullptr) throw std::invalid_argument("Passed address is nullptr!");
            #endif 
                _mm256_storeu_si256((__m256i*)dest, v);
            }

            /**
             * Saves data from vector into given memory address. Memory needs to be aligned on 32 byte boundary.
             * See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html for more details.
             * @param dest A valid (non-NULL) memory address aligned to 32-byte boundary.
             */
            void saveAligned(unsigned int* dest) const {
            #ifndef NDEBUG
                if(dest == nullptr) throw std::invalid_argument("Passed address is nullptr!");
            #endif
                _mm256_store_si256((__m256i*)dest, v);
            }

            bool operator==(const UInt256& b) const noexcept{
                unsigned int* v1,* v2;
                v1 = (unsigned int*)&v;
                v2 = (unsigned int*)&b.v;

                for(unsigned short i{0}; i < 8; ++i)
                    if(v1[i] != v2[i])
                        return false;

                return true;
            }

            bool operator==(const unsigned int& b) const noexcept{
                unsigned int* v1 = (unsigned int*)&v;

                for(unsigned short i{0}; i < 8; ++i)
                    if(v1[i] != b)
                        return false;

                return true;
            }

            bool operator!=(const UInt256& b) const {
                unsigned int* v1,* v2;
                v1 = (unsigned int*)&v;
                v2 = (unsigned int*)&b.v;

                for(unsigned short i{0}; i < 8; ++i)
                    if(v1[i] != v2[i])
                        return true;

                return false;
            }

            bool operator!=(const unsigned int&b) const {
                unsigned int* v1 = (unsigned int*)&v;

                for(unsigned short i{0}; i < 8; ++i)
                    if(v1[i] != b)
                        return true;

                return false;
            }

            unsigned int operator[](unsigned int index) const {
                if(index > 7) {
                    std::string error_text = "Invalid index! Valid range is [0-7] (was ";
                    error_text += std::to_string(index);
                    error_text += ").";
                    throw std::out_of_range(error_text);
                }
                unsigned int* tmp = (unsigned int*)&v;
                return tmp[index];
            }



            UInt256 operator+(const UInt256& b) const noexcept {return _mm256_add_epi32(v, b.v);}

            UInt256 operator+(const unsigned int& b) const noexcept {return _mm256_add_epi32(v, _mm256_set1_epi32(b));}

            UInt256& operator+=(const UInt256& b) noexcept {
                v = _mm256_add_epi32(v, b.v);
                return *this;
            }

            UInt256& operator+=(const unsigned int& b) noexcept {
                v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
                return *this;
            }

            UInt256 operator-(const UInt256& b) const noexcept {return _mm256_sub_epi32(v, b.v);}
            
            UInt256 operator-(const unsigned int& b) const noexcept {return _mm256_sub_epi32(v, _mm256_set1_epi32(b));}

            UInt256& operator-=(const UInt256& b) noexcept {
                v = _mm256_sub_epi32(v,b.v);
                return *this;
            }
            UInt256& operator-=(const unsigned int& b) noexcept {
                v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
                return *this;
            }

            UInt256 operator*(const UInt256& b) const noexcept {
                __m256i first = _mm256_mul_epu32(v, b.v);
                __m256i av = _mm256_srli_si256(v, sizeof(unsigned int));
                __m256i bv = _mm256_srli_si256(b.v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(av, bv);

                // Full AVX version
                second = _mm256_and_si256(second, crate);
                first = _mm256_and_si256(first, crate);
                second = _mm256_slli_si256(second, sizeof(unsigned int));

                return _mm256_or_si256(first, second);
            }

            UInt256 operator*(const unsigned int& b) const noexcept {
                __m256i bv = _mm256_set1_epi32(b);
                __m256i first = _mm256_mul_epu32(v, bv);
                __m256i av = _mm256_srli_si256(v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(av, bv);

                second = _mm256_and_si256(second, crate);
                first = _mm256_and_si256(first, crate);
                second = _mm256_slli_si256(second, sizeof(unsigned int));


                return _mm256_or_si256(first, second);
            }

            UInt256& operator*=(const UInt256& b) noexcept {
                __m256i first = _mm256_mul_epu32(v, b.v);
                v = _mm256_srli_si256(v, sizeof(unsigned int));
                
                __m256i bv = _mm256_srli_si256(b.v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(v, bv);

                second = _mm256_and_si256(second, crate);
                first = _mm256_and_si256(first, crate);
                second = _mm256_slli_si256(second, sizeof(unsigned int));

                v =_mm256_or_si256(first, second);
                return *this;
            }

            UInt256& operator*=(const unsigned int& b) noexcept {
                __m256i bv = _mm256_set1_epi32(b);
                __m256i first = _mm256_mul_epu32(v, bv);

                v = _mm256_srli_si256(v, sizeof(unsigned int));
                __m256i second = _mm256_mul_epu32(v, bv);

                second = _mm256_and_si256(second, crate);
                first = _mm256_and_si256(first, crate);
                second = _mm256_slli_si256(second, sizeof(unsigned int));
                
                v =_mm256_or_si256(first, second);

                return *this;
            }


            UInt256 operator/(const UInt256& b) const noexcept {
                unsigned int* a = (unsigned int*)&v;
                unsigned int* bv = (unsigned int*)&b.v;

                return UInt256(
                    _mm256_set_epi32(
                        a[7] / bv[7],
                        a[6] / bv[6],
                        a[5] / bv[5],
                        a[4] / bv[4],
                        a[3] / bv[3],
                        a[2] / bv[2],
                        a[1] / bv[1],
                        a[0] / bv[0]
                    )
                );
            }

            UInt256 operator/(const unsigned int& b) const noexcept {
                unsigned int* a = (unsigned int*)&v;

                return UInt256(
                    _mm256_set_epi32(
                        a[7] / b,
                        a[6] / b,
                        a[5] / b,
                        a[4] / b,
                        a[3] / b,
                        a[2] / b,
                        a[1] / b,
                        a[0] / b
                    )
                );
            }

            UInt256& operator/=(const UInt256& b) noexcept {
                unsigned int* a = (unsigned int*)&v;
                unsigned int* bv = (unsigned int*)&b.v;

                v = _mm256_set_epi32(
                    a[7] / bv[7],
                    a[6] / bv[6],
                    a[5] / bv[5],
                    a[4] / bv[4],
                    a[3] / bv[3],
                    a[2] / bv[2],
                    a[1] / bv[1],
                    a[0] / bv[0]
                );
                return *this;
            }

            UInt256& operator/=(const unsigned int& b) noexcept {
                unsigned int* a = (unsigned int*)&v;
                v = _mm256_set_epi32(
                    a[7] / b,
                    a[6] / b,
                    a[5] / b,
                    a[4] / b,
                    a[3] / b,
                    a[2] / b,
                    a[1] / b,
                    a[0] / b
                );
                return *this;
            }


            UInt256 operator%(const UInt256& b) const noexcept {
                unsigned int* a = (unsigned int*)&v;
                unsigned int* bv = (unsigned int*)&b.v;

                return UInt256(
                    _mm256_set_epi32(
                        a[7] % bv[7],
                        a[6] % bv[6],
                        a[5] % bv[5],
                        a[4] % bv[4],
                        a[3] % bv[3],
                        a[2] % bv[2],
                        a[1] % bv[1],
                        a[0] % bv[0]
                    )
                );
            }

            UInt256 operator%(const unsigned int& b) const noexcept {
                unsigned int* a = (unsigned int*)&v;

                return UInt256(
                    _mm256_set_epi32(
                        a[7] % b,
                        a[6] % b,
                        a[5] % b,
                        a[4] % b,
                        a[3] % b,
                        a[2] % b,
                        a[1] % b,
                        a[0] % b
                    )
                );

            }


            UInt256& operator%=(const UInt256& b) noexcept {
                unsigned int* a = (unsigned int*)&v;
                unsigned int* bv = (unsigned int*)&b.v;

                v = _mm256_set_epi32(
                    a[7] % bv[7],
                    a[6] % bv[6],
                    a[5] % bv[5],
                    a[4] % bv[4],
                    a[3] % bv[3],
                    a[2] % bv[2],
                    a[1] % bv[1],
                    a[0] % bv[0]
                );
                return *this;
            }

            UInt256& operator%=(const unsigned int& b) noexcept {
                unsigned int* a = (unsigned int*)&v;

                v = _mm256_set_epi32(
                        a[7] % b,
                        a[6] % b,
                        a[5] % b,
                        a[4] % b,
                        a[3] % b,
                        a[2] % b,
                        a[1] % b,
                        a[0] % b
                    );
                return *this;
            }

            UInt256 operator^(const UInt256& b) const noexcept {
                return _mm256_xor_si256(v, b.v);
            }

            UInt256 operator^(const unsigned int& b) const noexcept {
                return _mm256_xor_si256(v, _mm256_set1_epi32(b));
            }

            UInt256& operator^=(const UInt256& b) noexcept {
                v = _mm256_xor_si256(v, b.v);
                return *this;
            }

            UInt256& operator^=(const unsigned int& b) noexcept {
                v = _mm256_xor_si256(v, _mm256_set1_epi32(b));
                return *this;
            }

            UInt256 operator|(const UInt256& b) const noexcept {
                return _mm256_or_si256(v, b.v);
            }

            UInt256 operator|(const unsigned int& b) const noexcept {
                return _mm256_or_si256(v, _mm256_set1_epi32(b));
            }

            UInt256& operator|=(const UInt256& b) noexcept {
                v = _mm256_or_si256(v, b.v);
                return *this;
            }

            UInt256& operator|=(const unsigned int& b) noexcept {
                v = _mm256_or_si256(v, _mm256_set1_epi32(b));
                return *this;
            }

            UInt256 operator&(const UInt256& b) const noexcept {
                return _mm256_and_si256(v, b.v);
            }

            UInt256 operator&(const unsigned int& b) const noexcept {
                return _mm256_and_si256(v, _mm256_set1_epi32(b));
            }

            UInt256& operator&=(const UInt256& b) noexcept {
                v = _mm256_and_si256(v, b.v);
                return *this;
            }

            UInt256& operator&=(const unsigned int& b) noexcept {
                v = _mm256_and_si256(v, _mm256_set1_epi32(b));
                return *this;
            }

            UInt256 operator<<(const UInt256& b) const noexcept {
                return _mm256_sllv_epi32(v, b.v);
            }

            UInt256 operator<<(const unsigned int& b) const noexcept {
                return _mm256_slli_epi32(v, b);
            }

            UInt256& operator<<=(const UInt256& b) noexcept {
                v = _mm256_sllv_epi32(v, b.v);
                return *this;
            }

            UInt256& operator<<=(const unsigned int& b) noexcept {
                v = _mm256_slli_epi32(v,b);
                return *this;
            }

            UInt256 operator>>(const UInt256& b) const noexcept {
                return _mm256_srlv_epi32(v, b.v);
            }

            UInt256 operator>>(const unsigned int& b) const noexcept {
                return _mm256_srli_epi32(v, b);
            }

            UInt256& operator>>=(const UInt256& b) noexcept {
                v = _mm256_srlv_epi32(v, b.v);
                return *this;
            }
            UInt256& operator>>=(const unsigned int& b) noexcept {
                v = _mm256_srli_epi32(v, b);
                return *this;
            }
            
            UInt256 operator~() const noexcept { return _mm256_xor_si256(v, ones);}


            /**
             * Return string representation of vector.
             * 
             * @returns String representation of vector.
             */
            std::string str() const noexcept {
                std::string result = "UInt256(";
                unsigned int* iv = (unsigned int*)&v; 
                for(unsigned i{0}; i < 7; ++i)
                    result += std::to_string(iv[i]) + ", ";
                
                result += std::to_string(iv[7]);
                result += ")";
                return result;
            }


            
            /**
             * Sums all elements in vector.
             * @param items Vector containing `UInt256` values.
             * @return Sum of all elements in vector.
             */
            friend UInt256 sum(std::vector<UInt256>& b);

            
            /**
             * Sums all elements in set.
             * @param items Set containing `UInt256` values.
             * @return Sum of all elements in set.
             */
            friend UInt256 sum(std::set<UInt256>& b);

    };
};


#endif