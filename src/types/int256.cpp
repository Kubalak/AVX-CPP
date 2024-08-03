#include "int256.hpp"
#include <string>
#include <stdexcept>

namespace avx {
    const __m256i Int256::ones = _mm256_set1_epi8(0xFF);

    Int256::Int256(const int* init):
        v(_mm256_lddqu_si256((const __m256i*)init))
    {}


    Int256::Int256(const int& init):
        v(_mm256_set1_epi32(init))
    {}


    Int256::Int256(__m256i init):
        v(init)
    {}


    Int256::Int256(Int256& init):
        v(init.v)
    {}


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
            throw std::out_of_range(error_text);
        }
        int* tmp = (int*)&v;
        return tmp[index];
    }


    Int256 Int256::operator+(const Int256& b) const {
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
            _mm256_mullo_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }

    // TODO: Optimize division and modulo.
    Int256 Int256::operator/(const Int256& b) const {

        int* a = (int*)&v;
        int* bv = (int*)&b.v;

        return Int256(
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


    Int256 Int256::operator/(const int& b) const {

        int* a = (int*)&v;

        return Int256(
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


    Int256 Int256::operator%(const Int256& b) const{
        int* a = (int*)&v;
        int* bv = (int*)&b.v;

        return Int256(
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


    Int256 Int256::operator%(const int& b) const{
        int* a = (int*)&v;

        return Int256(
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


    Int256 Int256::operator~() const {
        return Int256(
            _mm256_xor_si256(v, ones)
        );
    }


    Int256 Int256::operator<<(const Int256& b) const {
        return Int256(
            _mm256_srlv_epi32(
                v,
                b.v
            )
        );
    }
    
    
    Int256 Int256::operator<<(const int& b) const {
        return Int256(
            _mm256_srli_epi32(
                v,
                b
            )
        );
    }

    
    Int256 Int256::operator>>(const Int256& b) const {
        return Int256(
            _mm256_srav_epi32(
                v,
                b.v
            )
        );
    }
    
    
    Int256 Int256::operator>>(const int& b) const {
        return Int256(
            _mm256_srai_epi32(
                v,
                b
            )
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


    Int256& Int256::operator-=(const Int256& b) {
        v = _mm256_sub_epi32(v,b.v);
        return *this;
    }


    Int256& Int256::operator-=(const int& b) {
        v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator*=(const Int256& b){
        v = _mm256_mullo_epi32(v, b.v);
        return *this;
    }


    Int256& Int256::operator*=(const int& b){
        v = _mm256_mullo_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator/=(const Int256& b){
        int* a = (int*)&v;
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
        );
        return *this;
    }


    Int256& Int256::operator/=(const int& b){
        int* a = (int*)&v;
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


    Int256& Int256::operator%=(const Int256& b) {
        int* a = (int*)&v;
        int* bv = (int*)&b.v;

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


    Int256& Int256::operator%=(const int& b){
        int* a = (int*)&v;

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


    Int256& Int256::operator|=(const Int256& b){
        v = _mm256_or_si256(v, b.v);
        return *this;
    }


    Int256& Int256::operator|=(const int& b){
        v = _mm256_or_si256(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator&=(const Int256& b){
        v = _mm256_and_si256(v, b.v);
        return *this;
    }


    Int256& Int256::operator&=(const int& b){
        v = _mm256_and_si256(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator^=(const Int256& b){
        v = _mm256_xor_si256(v, b.v);
        return *this;
    }


    Int256& Int256::operator^=(const int& b){
        v = _mm256_xor_si256(v, _mm256_set1_epi32(b));
        return *this;
    }


    Int256& Int256::operator<<=(const Int256& b) {
        v = _mm256_srlv_epi32(
            v,
            b.v
        );
        return *this;
    }
    
    
    Int256& Int256::operator<<=(const int& b) {
        v = _mm256_srli_epi32(
            v,
            b
        );
        return *this;
    }

    
    Int256& Int256::operator>>=(const Int256& b) {
        v = _mm256_srav_epi32(
            v,
            b.v
        );
        return *this;
    }
    
    
    Int256& Int256::operator>>=(const int& b) {
        v = _mm256_srai_epi32(
            v,
            b
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
        
        return Int256(result);
    }


    Int256 sum(std::set<Int256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Int256& item : a)
            result = _mm256_add_epi32(result, item.v);
        
        return Int256(result);
    }
};