#pragma once
#ifndef FLOAT256_HPP__
#define FLOAT256_HPP__

#include <array>
#include <string>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>
#include <types/constants.hpp>

namespace avx {
    /**
     * Class providing vectorized version of `float`.
     * Can hold 8 individual `float` values.
     * Provides arithmetic operators.
     * Provides comparison operators == !=.
     */
    class Float256 {
        private:
            __m256 v;

        public:

            /**
             * Number of individual values stored by object. This value can be used to iterate over elements.
            */
            static constexpr int size = 8;

            /**
             * Type that is stored inside vector.
             */
            using storedType = float;

            /**
             * Default constructor. Initializes vector with zeros.
             */
            Float256() noexcept : v(_mm256_setzero_ps()) {}

            /**
             * Copy constructor.
             * Initializes vector from another Float256 vector.
             * @param init Another Float256 vector to copy from.
             */
            Float256(const Float256 &init) noexcept : v(init.v) {}

            /**
             * Initializes vector with const value. Each cell will be set with value of `value`.
             * @param value Value to be set.
             */
            Float256(const float value) noexcept : v(_mm256_set1_ps(value)) {}

            /**
             * Initializes vector from __m256 value.
             * @param init Value of type __m256 to initialize the vector.
             */
            Float256(const __m256 init) noexcept : v(init) {}

            /**
             * Initializes vector from std::array of 8 float values.
             * @param init Array of 8 float values to initialize the vector.
             */
            Float256(const std::array<float, 8> &init) noexcept : v(_mm256_loadu_ps(init.data())) {}

            /**
             * Initializes vector from initializer_list of float values.
             * If the list contains fewer than 8 elements, remaining elements are set to zero.
             * If the list contains more than 8 elements, only the first 8 are used.
             * @param init Initializer list of float values.
             */
            Float256(std::initializer_list<float> init) noexcept {
                alignas(32) float init_v[size];
                memset((char*)init_v, 0, 32);
                if(init.size() < size){
                    auto begin = init.begin();
                    for(char i{0}; i < init.size(); ++i){
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                else {
                    auto begin = init.begin();
                    for(char i{0}; i < size; ++i){
                        init_v[i] = *begin;
                        begin++;
                    }
                }
                v = _mm256_load_ps((const float*)init_v);
            }

            /**
             * Initializes vector by loading data from memory (via `_mm256_loadu_ps`).
             * @param pSrc Pointer to memory of at least 32 bytes (8 floats).
             * @throws std::invalid_argument If in Debug mode and passed `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            Float256(const float* pSrc) {
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                v = _mm256_loadu_ps(pSrc);
            }

            /**
             * Returns the internal __m256 value stored by the object.
             * @return The __m256 value.
             */
            const __m256 get() const noexcept { return v; }

            /**
             * Sets the internal __m256 value stored by the object.
             * @param val New value of type __m256.
             */
            void set(__m256 val) noexcept { v = val; }

            /**
             * Loads data from memory into vector (memory should be of size of at least 32 bytes). Memory doesn't need to be aligned to any specific boundary. If `sP` is `nullptr` this method has no effect.
             * @param pSrc Pointer to memory from which to load data.
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void load(const float *pSrc) {
            #ifndef NDEBUG
                if(!pSrc)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                v = _mm256_loadu_ps(pSrc);
            }

            /**
             * Saves data to destination in memory.
             * @param dest Reference to the list to which vector will be saved. Array doesn't need to be aligned to any specific boundary.
             */
            void save(std::array<float, 8>& dest) const noexcept {
                _mm256_storeu_ps(dest.data(), v);
            }

            /**
             * Saves data to destination in memory. The memory doesn't have to be aligned to any specific boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (8x `float`).
             * @throws std::invalid_argument If in Debug mode and `pDest` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void save(float *pDest) const {
            #ifndef NDEBUG
                if(!pDest)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                _mm256_storeu_ps(pDest, v);
            }

            /**
             * Saves data to destination in memory. The memory must be aligned at 32-byte boundary.
             * 
             * See https://en.cppreference.com/w/cpp/memory/c/aligned_alloc for more details.
             * @param pDest A valid pointer to a memory of at least 32 bytes (8x `float`).
             * @throws std::invalid_argument If in Debug mode and `pSrc` is `nullptr`. In Release mode no checks are performed to improve efficiency.
             */
            void saveAligned(float *pDest) const {
            #ifndef NDEBUG
                if(!pDest)
                    throw std::invalid_argument(__AVX_LOCALIZED_NULL_STR);
                else
            #endif
                _mm256_store_ps(pDest, v);
            }

            /**
             * Compares with second vector for equality. This method is secured to return true when comparing 0.0f with -0.0f.
             * @param bV Second vector to compare.
             * @returns `true` if all fields in both vectors have the same value, `false` otherwise.
             */
            bool operator==(const Float256& bV) {
                __m256 eq = _mm256_xor_ps(v, bV.v); // Bitwise XOR - equal values return field with 0.

                /*
                    Explanation - compute bitwise AND with value and sign bit set to 0.
                    Next is comparing result to 0 (remove sign bit from equation).
                    Compute AND of NOT v and NOT bV.v - if they are equal (0) then corresponding field will be set to 0.
                */
                __m256 zerofx = _mm256_castsi256_ps(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(bV.v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256())
                ));

                // Fixes 0.0f == -0.0f mismatch by zeroing corresponding fields
                eq = _mm256_and_ps(eq, zerofx);

                return _mm256_testz_si256(_mm256_castps_si256(eq), _mm256_castps_si256(eq)) != 0;
            }


            /**
             * Compares with value for equality. This method is secured to return true when comparing 0.0f with -0.0f.
             * @param b Value to compare with.
             * @returns `true` if all fields in vectors have the same value as b, `false` otherwise.
             */
            bool operator==(const float b) {
                __m256 bV = _mm256_set1_ps(b);
                __m256 eq = _mm256_xor_ps(v, bV);

                __m256 zerofx = _mm256_castsi256_ps(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(bV, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_ps(eq, zerofx);

                return _mm256_testz_si256(_mm256_castps_si256(eq), _mm256_castps_si256(eq)) != 0;
            }

            /**
             * Compares with second vector for equality. This method is secured to return true when comparing 0.0f with -0.0f.
             * @param bV Second vector to compare.
             * @returns `true` if ANY field in one vector has different value than one in scond vector, `false` if vector are equal.
             */
            bool operator!=(const Float256& bV) {
                __m256 eq = _mm256_xor_ps(v, bV.v);

                __m256 zerofx = _mm256_castsi256_ps(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(bV.v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_ps(eq, zerofx);

                return _mm256_testz_si256(_mm256_castps_si256(eq), _mm256_castps_si256(eq)) == 0;
            }


            /**
             * Compares with second vector for equality. This method is secured to return true when comparing 0.0f with -0.0f.
             * @param b Value to compare with.
             * @returns `true` if ANY field in vector has different value than passed value, `false` if vector are equal.
             */
            bool operator!=(const float b) {
                __m256 bV = _mm256_set1_ps(b);
                __m256 eq = _mm256_xor_ps(v, bV);

                __m256 zerofx = _mm256_castsi256_ps(_mm256_andnot_si256(
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(v, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256()),
                    _mm256_cmpeq_epi32(_mm256_castps_si256(_mm256_and_ps(bV, constants::FLOAT_NO_SIGN)), _mm256_setzero_si256())
                ));

                eq = _mm256_and_ps(eq, zerofx);

                return _mm256_testz_si256(_mm256_castps_si256(eq), _mm256_castps_si256(eq)) == 0;
            }

            /**
             * Adds two vectors together.
             * @param bV Second vector.
             * @returns New vector being result of adding `bV` to vector.
             */
            Float256 operator+(const Float256& bV) const noexcept {
                return _mm256_add_ps(v, bV.v);
            }

            /**
             * Adds scalar to all vector fields.
             * @param b Scalar value to be added.
             * @returns New vector being result of adding `b` to vector.
             */
            Float256 operator+(const float b) const noexcept {
                return _mm256_add_ps(v, _mm256_set1_ps(b));
            }

            /**
             * Adds two vectors together and stores result inside original vector.
             * @param bV Second vector.
             * @returns Reference to same vector after adding `bV` to vector.
             */
            Float256& operator+=(const Float256& bV) noexcept {
                v = _mm256_add_ps(v, bV.v);
                return *this;
            }

            /**
             * Adds scalar to vector and stores result inside original vector.
             * @param b Scalar to be added.
             * @returns Reference to same vector after adding `b` to vector.
             */
            Float256& operator+=(const float b) noexcept {
                v = _mm256_add_ps(v, _mm256_set1_ps(b));
                return *this;
            }

            /**
             * Subtracts two vectors.
             * @param bV Second vector.
             * @returns New vector being result of subtracting `bV` from vector.
             */
            Float256 operator-(const Float256& bV) const noexcept {
                return _mm256_sub_ps(v, bV.v);
            }

            /**
             * Subtracts scalar from all vector fields.
             * @param b Scalar value to be subtracted.
             * @returns New vector being result of subtracting `b` from vector.
             */
            Float256 operator-(const float b) const noexcept {
                return _mm256_sub_ps(v, _mm256_set1_ps(b));
            }

            /**
             * Subtracts two vectors and stores result inside original vector.
             * @param bV Second vector.
             * @returns Reference to same vector after subtracting `bV` from vector.
             */
            Float256& operator-=(const Float256& bV) noexcept {
                v = _mm256_sub_ps(v, bV.v);
                return *this;
            }

            /**
             * Subtracts scalar from vector and stores result inside original vector.
             * @param b Scalar to be subtracted.
             * @returns Reference to same vector after subtracting `b` from vector.
             */
            Float256& operator-=(const float b) noexcept {
                v = _mm256_sub_ps(v, _mm256_set1_ps(b));
                return *this;
            }

            /**
             * Multiplies two vectors.
             * @param bV Second vector.
             * @returns New vector being result of multiplying vector by `bV`.
             */
            Float256 operator*(const Float256& bV) const noexcept {
                return _mm256_mul_ps(v, bV.v);
            }

            /**
             * Multiplies all vector fields by scalar.
             * @param b Scalar value to multiply by.
             * @returns New vector being result of multiplying vector by `b`.
             */
            Float256 operator*(const float b) const noexcept {
                return _mm256_mul_ps(v, _mm256_set1_ps(b));
            }

            /**
             * Multiplies two vectors and stores result inside original vector.
             * @param bV Second vector.
             * @returns Reference to same vector after multiplying by `bV`.
             */
            Float256& operator*=(const Float256& bV) noexcept {
                v = _mm256_mul_ps(v, bV.v);
                return *this;
            }

            /**
             * Multiplies vector by scalar and stores result inside original vector.
             * @param b Scalar to multiply by.
             * @returns Reference to same vector after multiplying by `b`.
             */
            Float256& operator*=(const float b) noexcept {
                v = _mm256_mul_ps(v, _mm256_set1_ps(b));
                return *this;
            }

            /**
             * Divides two vectors.
             * @param bV Second vector (divisor).
             * @returns New vector being result of dividing vector by `bV`.
             */
            Float256 operator/(const Float256& bV) const noexcept {
                return _mm256_div_ps(v, bV.v);
            }

            /**
             * Divides all vector fields by scalar.
             * @param b Scalar value (divisor).
             * @returns New vector being result of dividing vector by `b`.
             */
            Float256 operator/(const float b) const noexcept {
                return _mm256_div_ps(v, _mm256_set1_ps(b));
            }

            /**
             * Divides two vectors and stores result inside original vector.
             * @param bV Second vector (divisor).
             * @returns Reference to same vector after dividing by `bV`.
             */
            Float256& operator/=(const Float256& bV) noexcept {
                v = _mm256_div_ps(v, bV.v);
                return *this;
            }

            /**
             * Divides vector by scalar and stores result inside original vector.
             * @param b Scalar value (divisor).
             * @returns Reference to same vector after dividing by `b`.
             */
            Float256& operator/=(const float b) noexcept {
                v = _mm256_div_ps(v, _mm256_set1_ps(b));
                return *this;
            }

            /**
            * Indexing operator.
            * Does not support value assignment through this method (e.g. aV[0] = 1 won't work).
            * @param index Position of desired element between 0 and 7.
            * @return Value of underlying element.
            * @throws std::out_of_range If index is not within the correct range and build type is debug will be thrown. Otherwise bitwise AND will prevent index to be out of range. Side effect is that only 3 LSBs are used from `index`.
            */
            float operator[](const unsigned int index) const 
            #ifndef NDEBUG 
                {
                    if(index > 7)
                        throw std::invalid_argument("Invalid index! Index should be within 0-7, passed: " + std::to_string(index));
                    return ((float*)&v)[index];
                }
            #else 
                noexcept { return ((float*)&v)[index & 7]; }
            #endif
            
            /**
             * Returns string representation of vector.
             * Printing will result in Float256(<vector_values>) eg. Float256(1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000)
             * @returns String representation of underlying vector.
             */
            std::string str() const noexcept {
                std::string result = "Float256(";
                float* iv = (float*)&v; 
                for(unsigned i{0}; i < 7; ++i)
                    result += std::to_string(iv[i]) + ", ";

                result += std::to_string(iv[7]);
                result += ")";
                return result;
            }

            
            /**
             * Provides support for `float` + Float256 operation.
             * @param a Scalar to which `bV` should be added.
             * @param bV Vector which will be added.
             * @returns Float256 Vector being a result of `a` + `bV`
             */
            friend Float256 operator+(float a, const Float256 &bV) {
                return _mm256_add_ps(bV.v, _mm256_set1_ps(a));
            }

            /**
             * Provides support for `float` - Float256 operation.
             * @param a Scalar from which `bV` should be subtracted.
             * @param bV Vector which will be subtracted.
             * @returns Float256 Vector being a result of `a` - `bV`
             */
            friend Float256 operator-(float a, const Float256 &bV) {
                return _mm256_sub_ps(_mm256_set1_ps(a), bV.v);
            }

            /**
             * Provides support for `float` * Float256 operation.
             * @param a Scalar which should be multiplied by `bV`.
             * @param bV Vector which will be multiplier.
             * @returns Float256 Vector being a result of `a` * `bV`
             */
            friend Float256 operator*(float a, const Float256 &bV) {
                return _mm256_mul_ps(_mm256_set1_ps(a), bV.v);
            }

            /**
             * Provides support for `float` / Float256 operation.
             * @param a Scalar which should be divided by `bV`.
             * @param bV Vector which will be divisor.
             * @returns Float256 Vector being a result of `a` / `bV`
             */
            friend Float256 operator/(float a, const Float256 &bV) {
                return _mm256_div_ps(_mm256_set1_ps(a), bV.v);
            }

    };
} 


#endif