#include "int256.hpp"
#include <string>
#include <stdexcept>

namespace avx {
    static_assert(sizeof(int) == 4, "You are compiling to 32-bit. Please switch to x64 to avoid undefined behaviour.");
    const __m256i Int256::ones = _mm256_set1_epi8(0xFF);


    Int256::Int256(std::initializer_list<int> init) {
        if(init.size() < 8)
            throw  std::invalid_argument("Initial list size must be at least 8");
        
        auto start = init.begin();
        int a0, a1, a2, a3, a4, a5, a6, a7;
        a0 = *start;
        start++;
        a1 = *start;
        start++;
        a2 = *start;
        start++;
        a3 = *start;
        start++;
        a4 = *start;
        start++;
        a5 = *start;
        start++;
        a6 = *start;
        start++;
        a7 = *start;
        start++;
        v = _mm256_set_epi32(
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7
        );
    }


    bool Int256::operator==(const Int256& b) const {
        int* v1,* v2;
        v1 = (int*)&v;
        v2 = (int*)&b.v;

        for(unsigned short i{0}; i < 8; ++i)
            if(v1[i] != v2[i])
                return false;

        return true;
    }


    bool Int256::operator==(const int& b) const {
        int* v1 = (int*)&v;

        for(unsigned short i{0}; i < 8; ++i)
            if(v1[i] != b)
                return false;

        return true;
    }


    bool Int256::operator!=(const Int256& b) const {
        int* v1,* v2;
        v1 = (int*)&v;
        v2 = (int*)&b.v;

        for(unsigned short i{0}; i < 8; ++i)
            if(v1[i] != v2[i])
                return true;

        return false;
    }


    bool Int256::operator!=(const int& b) const {
        int* v1 = (int*)&v;

        for(unsigned short i{0}; i < 8; ++i)
            if(v1[i] != b)
                return true;

        return false;
    }


    Int256 Int256::operator/(const Int256& b) const {

        /*int* a = (int*)&v;
        int* bv = (int*)&b.v;

        
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
        );*/ 
        
        return _mm256_cvtps_epi32(
            _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(b.v)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
        );
    }


    Int256 Int256::operator/(const int& b) const {

        /*int* a = (int*)&v;

        
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
        );*/

        return _mm256_cvtps_epi32(
            _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(b)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
        );
    }


    Int256& Int256::operator/=(const Int256& b){
        /*int* a = (int*)&v;
        int* bv = (int*)&b.v;

        v = _mm256_set_epi32(
            a[7] / bv[7],
            a[6] / bv[6],
            a[5] / bv[5],
            a[4] / bv[4],
            a[3] / bv[3],
            a[2] / bv[2],
            a[1] / bv[1],
            a[0] / bv[0]
        );*/
        v = _mm256_cvtps_epi32(
            _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_cvtepi32_ps(b.v)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
        );
        return *this;
    }


    Int256& Int256::operator/=(const int& b){
        /*int* a = (int*)&v;
        v = _mm256_set_epi32(
            a[7] / b,
            a[6] / b,
            a[5] / b,
            a[4] / b,
            a[3] / b,
            a[2] / b,
            a[1] / b,
            a[0] / b
        );*/

        v = _mm256_cvtps_epi32(
            _mm256_round_ps(_mm256_div_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(b)), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
        );
        return *this;
    }


    std::string Int256::str() const{
        std::string result = "Int256(";
        int* iv = (int*)&v; 
        for(unsigned i{0}; i < 7; ++i)
            result += std::to_string(iv[i]) + ", ";
        
        result += std::to_string(iv[7]);
        result += ")";
        return result;
    }

    
    Int256 sum(std::vector<Int256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Int256& item : a)
            result = _mm256_add_epi32(result, item.v);
        
        return result;
    }


    Int256 sum(std::set<Int256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Int256& item : a)
            result = _mm256_add_epi32(result, item.v);
        
        return result;
    }
};