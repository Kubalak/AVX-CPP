#include "int256.hpp"
#include <string>
#include <stdexcept>

namespace avx {
    const __m256i Int256::ones = _mm256_set1_epi8(0xFF);

    Int256::Int256(__m256i init):
        v(init)
    {}


    Int256::Int256(Int256& init):
        v(init.v)
    {}


    Int256::Int256(std::array<int, 8> init):
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


    Int256::Int256(std::array<short, 8> init):
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


    Int256::Int256(std::array<char, 8> init):
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


    int Int256::operator[](unsigned int index) const {
        if(index > 7) {
            std::string error_text = "Invalid index! Valid range is [0-7] (was ";
            error_text += std::to_string(index);
            error_text += ").";
            throw std::out_of_range::out_of_range(error_text);
        }
        int* tmp = (int*)&v;
        return tmp[index];
    }


    Int256 Int256::operator+(Int256& b) const {
        return Int256(_mm256_add_epi32(v, b.v));
    }


    Int256 Int256::operator+(const int& b) const{
        return Int256(
            _mm256_add_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator+(const short& b) const{
        return Int256(
            _mm256_add_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator+(const char& b) const{
        return Int256(
            _mm256_add_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator-(const Int256& b) const{
        return Int256(
            _mm256_sub_epi32(
                v, 
                b.v
            )
        );
    }


    Int256 Int256::operator-(const int& b) const{
        return Int256(
            _mm256_sub_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator-(const short& b) const{
        return Int256(
            _mm256_sub_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator-(const char& b) const{
        return Int256(
            _mm256_sub_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator*(const Int256& b) const{
        return Int256(
            _mm256_mul_epi32(
                v, 
                b.v
            )
        );
    }


    Int256 Int256::operator*(const int& b) const{
        return Int256(
            _mm256_mul_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator*(const short& b) const{
        return Int256(
            _mm256_mul_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator*(const char& b) const{
        return Int256(
            _mm256_mul_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }

    // Int256 Int256::operator/(const int& b) const{
    //     return Int256(
    //         _mm256_mullo_epi32(
    //             v, 
    //             _mm256_set1_epi32(2147483647 / b)
    //         )
    //     );
    // }


    // Int256 Int256::operator/(const Int256& b) const{
    //     return Int256(
    //         _mm256_div_epi32(
    //             v, 
    //             b.v
    //         )
    //     );
    // }


    // Int256 Int256::operator/(const int& b) const{
    //     return Int256(
    //         _mm256_div_epi32(
    //             v, 
    //             _mm256_set1_epi32(b)
    //         )
    //     );
    // }


    // Int256 Int256::operator/(const short& b) const{
    //     return Int256(
    //         _mm256_div_epi32(
    //             v, 
    //             _mm256_set1_epi32(b)
    //         )
    //     );
    // }


    // Int256 Int256::operator/(const char& b) const{
    //     return Int256(
    //         _mm256_div_epi32(
    //             v, 
    //             _mm256_set1_epi32(b)
    //         )
    //     );
    // }

    /*
    Int256 Int256::operator%(const Int256& b) const{
    }


    Int256 Int256::operator%(const int& b) const{

    }


    Int256 Int256::operator%(const short& b) const{

    }


    Int256 Int256::operator%(const char& b) const{

    }*/


    Int256 Int256::operator^(const Int256& b) const{
        return Int256(
            _mm256_xor_si256(
                v, 
                b.v
            )
        );
    }


    Int256 Int256::operator^(const int& b) const{
        return Int256(
            _mm256_xor_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator^(const short& b) const{
        return Int256(
            _mm256_xor_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator^(const char& b) const{
        return Int256(
            _mm256_xor_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }



    Int256 Int256::operator|(const Int256& b) const{
        return Int256(
            _mm256_or_si256(
                v, 
                b.v
            )
        );
    }


    Int256 Int256::operator|(const int& b) const{
        return Int256(
            _mm256_or_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator|(const short& b) const{
        return Int256(
            _mm256_or_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator|(const char& b) const{
        return Int256(
            _mm256_or_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }



    Int256 Int256::operator&(const Int256& b) const{
        return Int256(
            _mm256_or_si256(
                v, 
                b.v
            )
        );
    }


    Int256 Int256::operator&(const int& b) const{
        return Int256(
            _mm256_and_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator&(const short& b) const{
        return Int256(
            _mm256_and_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator&(const char& b) const{
        return Int256(
            _mm256_and_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    Int256 Int256::operator~() const {
        return Int256(
            _mm256_xor_si256(v, ones)
        );
    }


    Int256& Int256::operator+=(const Int256& b) {
        v = _mm256_add_epi32(v,b.v);
        return *this;
    }


    Int256& Int256::operator+=(const int& b) {
        v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator+=(const short& b) {
        v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator+=(const char& b) {
        v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator-=(const Int256& b) {
        v = _mm256_sub_epi32(v,b.v);
        return *this;
    }


    Int256& Int256::operator-=(const int& b) {
        v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator-=(const short& b) {
        v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator-=(const char& b) {
        v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }

    
    Int256 sum(std::vector<Int256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Int256& item : a)
            result = _mm256_add_epi32(result, item.v);
        
        return Int256(result);
    }

    Int256 sum(std::set<Int256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Int256& item : a)
            result = _mm256_add_epi32(result, item.v);
        
        return Int256(result);
    }
};