#include "uint256.hpp"
#include <stdexcept>

namespace avx {
    const __m256i UInt256::ones = _mm256_set1_epi8(0xFF);
    const __m256i UInt256::crate = _mm256_set_epi32(0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF);

    UInt256::UInt256(const unsigned int* init):
        v(_mm256_lddqu_si256((const __m256i*)init))
    {}

    UInt256::UInt256(const unsigned int& init):
        v(_mm256_set1_epi32(init))
    {}


    UInt256::UInt256(__m256i init):
        v(init)
    {}


    UInt256::UInt256(UInt256& init):
        v(init.v)
    {}


    UInt256::UInt256(std::initializer_list<unsigned int> init) {
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

    UInt256::UInt256(std::array<unsigned int, 8> init):
        v(_mm256_lddqu_si256((const __m256i*)init.data()))
    {}


    UInt256::UInt256(std::array<unsigned short, 8> init):
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


    UInt256::UInt256(std::array<unsigned char, 8> init):
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


    bool UInt256::operator==(const UInt256& b) const {
        unsigned int* v1,* v2;
        v1 = (unsigned int*)&v;
        v2 = (unsigned int*)&b.v;

        for(unsigned short i{0}; i < 8; ++i)
            if(v1[i] != v2[i])
                return false;

        return true;
    }


    bool UInt256::operator==(const unsigned int& b) const {
        unsigned int* v1 = (unsigned int*)&v;

        for(unsigned short i{0}; i < 8; ++i)
            if(v1[i] != b)
                return false;

        return true;
    }


    bool UInt256::operator!=(const UInt256& b) const {
        unsigned int* v1,* v2;
        v1 = (unsigned int*)&v;
        v2 = (unsigned int*)&b.v;

        for(unsigned short i{0}; i < 8; ++i)
            if(v1[i] != v2[i])
                return true;

        return false;
    }


    bool UInt256::operator!=(const unsigned int& b) const {
        unsigned int* v1 = (unsigned int*)&v;

        for(unsigned short i{0}; i < 8; ++i)
            if(v1[i] != b)
                return true;

        return false;
    }


    const unsigned int UInt256::operator[](unsigned int index) const {
        if(index > 7) {
            std::string error_text = "Invalid index! Valid range is [0-7] (was ";
            error_text += std::to_string(index);
            error_text += ").";
            throw std::out_of_range(error_text);
        }
        unsigned int* tmp = (unsigned int*)&v;
        return tmp[index];
    }


    UInt256 UInt256::operator+(const UInt256& b) const {
        return UInt256(_mm256_add_epi32(v, b.v));
    }


    UInt256 UInt256::operator+(const unsigned int& b) const{
        return UInt256(
            _mm256_add_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    UInt256 UInt256::operator-(const UInt256& b) const{
        return UInt256(
            _mm256_sub_epi32(
                v, 
                b.v
            )
        );
    }


    UInt256 UInt256::operator-(const unsigned int& b) const{
        return UInt256(
            _mm256_sub_epi32(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    UInt256 UInt256::operator*(const UInt256& b) const{
        __m256i first = _mm256_mul_epu32(v, b.v);
        __m256i av = _mm256_srli_si256(v, sizeof(unsigned int));
        __m256i bv = _mm256_srli_si256(b.v, sizeof(unsigned int));
        __m256i second = _mm256_mul_epu32(av, bv);

        // Full AVX version
        second = _mm256_and_si256(second, crate);
        first = _mm256_and_si256(first, crate);
        second = _mm256_slli_si256(second, sizeof(unsigned int));

        return UInt256(_mm256_or_si256(first, second));

        // int* fp = (int*) &first;
        // int* sp = (int*) &second;

        // return UInt256(_mm256_set_epi32(
        //     sp[6],
        //     fp[6],
        //     sp[4],
        //     fp[4],
        //     sp[2],
        //     fp[2],
        //     sp[0],
        //     fp[0]
        // ));
    }


    UInt256 UInt256::operator*(const unsigned int& b) const{
        __m256i bv = _mm256_set1_epi32(b);
        __m256i first = _mm256_mul_epu32(v, bv);
        __m256i av = _mm256_srli_si256(v, sizeof(unsigned int));
        __m256i second = _mm256_mul_epu32(av, bv);

        second = _mm256_and_si256(second, crate);
        first = _mm256_and_si256(first, crate);
        second = _mm256_slli_si256(second, sizeof(unsigned int));


        return UInt256(
            _mm256_or_si256(first, second)
        );
    }

    // TODO: Optimize division and modulo.
    UInt256 UInt256::operator/(const UInt256& b) const {

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


    UInt256 UInt256::operator/(const unsigned int& b) const {

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


    UInt256 UInt256::operator%(const UInt256& b) const{
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


    UInt256 UInt256::operator%(const unsigned int& b) const{
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


    UInt256 UInt256::operator^(const UInt256& b) const{
        return UInt256(
            _mm256_xor_si256(
                v, 
                b.v
            )
        );
    }


    UInt256 UInt256::operator^(const unsigned int& b) const{
        return UInt256(
            _mm256_xor_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    UInt256 UInt256::operator|(const UInt256& b) const{
        return UInt256(
            _mm256_or_si256(
                v, 
                b.v
            )
        );
    }


    UInt256 UInt256::operator|(const unsigned int& b) const{
        return UInt256(
            _mm256_or_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    UInt256 UInt256::operator&(const UInt256& b) const{
        return UInt256(
            _mm256_and_si256(
                v, 
                b.v
            )
        );
    }


    UInt256 UInt256::operator&(const unsigned int& b) const{
        return UInt256(
            _mm256_and_si256(
                v, 
                _mm256_set1_epi32(b)
            )
        );
    }


    UInt256 UInt256::operator~() const {
        return UInt256(
            _mm256_xor_si256(v, ones)
        );
    }


    UInt256 UInt256::operator<<(const UInt256& b) const {
        return UInt256(
            _mm256_sllv_epi32(
                v,
                b.v
            )
        );
    }
    
    
    UInt256 UInt256::operator<<(const unsigned int& b) const {
        return UInt256(
            _mm256_slli_epi32(
                v,
                b
            )
        );
    }

    
    UInt256 UInt256::operator>>(const UInt256& b) const {
        return UInt256(
            _mm256_srlv_epi32(
                v,
                b.v
            )
        );
    }
    
    
    UInt256 UInt256::operator>>(const unsigned int& b) const {
        return UInt256(
            _mm256_srli_epi32(
                v,
                b
            )
        );
    }


    UInt256& UInt256::operator+=(const UInt256& b) {
        v = _mm256_add_epi32(v,b.v);
        return *this;
    }


    UInt256& UInt256::operator+=(const unsigned int& b) {
        v = _mm256_add_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    UInt256& UInt256::operator-=(const UInt256& b) {
        v = _mm256_sub_epi32(v,b.v);
        return *this;
    }


    UInt256& UInt256::operator-=(const unsigned int& b) {
        v = _mm256_sub_epi32(v, _mm256_set1_epi32(b));
        return *this;
    }


    UInt256& UInt256::operator*=(const UInt256& b){
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


    UInt256& UInt256::operator*=(const unsigned int& b){
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


    UInt256& UInt256::operator/=(const UInt256& b){
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


    UInt256& UInt256::operator/=(const unsigned int& b){
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


    UInt256& UInt256::operator%=(const UInt256& b) {
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


    UInt256& UInt256::operator%=(const unsigned int& b){
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


    UInt256& UInt256::operator|=(const UInt256& b){
        v = _mm256_or_si256(v, b.v);
        return *this;
    }


    UInt256& UInt256::operator|=(const unsigned int& b){
        v = _mm256_or_si256(v, _mm256_set1_epi32(b));
        return *this;
    }


    UInt256& UInt256::operator&=(const UInt256& b){
        v = _mm256_and_si256(v, b.v);
        return *this;
    }


    UInt256& UInt256::operator&=(const unsigned int& b){
        v = _mm256_and_si256(v, _mm256_set1_epi32(b));
        return *this;
    }


    UInt256& UInt256::operator^=(const UInt256& b){
        v = _mm256_xor_si256(v, b.v);
        return *this;
    }


    UInt256& UInt256::operator^=(const unsigned int& b){
        v = _mm256_xor_si256(v, _mm256_set1_epi32(b));
        return *this;
    }


    UInt256& UInt256::operator<<=(const UInt256& b) {
        v = _mm256_sllv_epi32(
            v,
            b.v
        );
        return *this;
    }
    
    
    UInt256& UInt256::operator<<=(const unsigned int& b) {
        v = _mm256_slli_epi32(
            v,
            b
        );
        return *this;
    }

    
    UInt256& UInt256::operator>>=(const UInt256& b) {
        v = _mm256_srlv_epi32(
            v,
            b.v
        );
        return *this;
    }
    
    
    UInt256& UInt256::operator>>=(const unsigned int& b) {
        v = _mm256_srli_epi32(
            v,
            b
        );
        return *this;
    }


    std::string UInt256::str() const{
        std::string result = "UInt256(";
        unsigned int* iv = (unsigned int*)&v; 
        for(unsigned i{0}; i < 7; ++i)
            result += std::to_string(iv[i]) + ", ";
        
        result += std::to_string(iv[7]);
        result += ")";
        return result;
    }


    void UInt256::save(std::array<unsigned int, 8>& dest) {
        _mm256_storeu_si256((__m256i*)dest.data(), v);
    }

            

    void UInt256::save(unsigned int* dest) {
        _mm256_storeu_si256((__m256i*)dest, v);
    }


    void UInt256::saveAligned(unsigned int* dest){
        _mm256_store_si256((__m256i*)dest, v);
    }

    
    UInt256 sum(std::vector<UInt256>& items){
        __m256i result = _mm256_setzero_si256();
        for(const UInt256& item : items)
            result = _mm256_add_epi32(result, item.v);
        
        return UInt256(result);
    }


    UInt256 sum(std::set<UInt256>& items){
        __m256i result = _mm256_setzero_si256();
        for(const UInt256& item : items)
            result = _mm256_add_epi32(result, item.v);
        
        return UInt256(result);
    }
}