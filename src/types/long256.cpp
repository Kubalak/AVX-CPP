#include "long256.hpp"
#include <string>
#include <stdexcept>

namespace avx {
    const __m256i Long256::ones = _mm256_set1_epi8(0xFF);

    Long256::Long256(const long long* init):
        v(_mm256_lddqu_si256((const __m256i*)init))
    {}


    Long256::Long256(const long long& init):
        v(_mm256_set1_epi64x(init))
    {}


    Long256::Long256(__m256i init):
        v(init)
    {}


    Long256::Long256(Long256& init):
        v(init.v)
    {}


    Long256::Long256(std::initializer_list<long long> init) {
        if(init.size() < 4)
            throw  std::invalid_argument("Initial list size must be at least 8");
        
        auto start = init.begin();
        long long a0, a1, a2, a3;
        a0 = *start;
        start++;
        a1 = *start;
        start++;
        a2 = *start;
        start++;
        a3 = *start;
        v = _mm256_set_epi64x(
            a0,
            a1,
            a2,
            a3
        );
    }

    Long256::Long256(std::array<long long, 4> init):
        v(_mm256_lddqu_si256((const __m256i*)init.data()))
    {}


    Long256::Long256(std::array<short, 4> init):
        v(_mm256_set_epi64x(
            init[0], 
            init[1], 
            init[2], 
            init[3]
            )
        )
    {}


    Long256::Long256(std::array<char, 4> init):
        v(_mm256_set_epi64x(
            init[0], 
            init[1], 
            init[2], 
            init[3]
            )
        )
    {}


    bool Long256::operator==(const Long256& b) const {
        long long* v1,* v2;
        v1 = (long long*)&v;
        v2 = (long long*)&b.v;

        for(unsigned short i{0}; i < 4; ++i)
            if(v1[i] != v2[i])
                return false;

        return true;
    }


    bool Long256::operator==(const long long& b) const {
        long long* v1 = (long long*)&v;

        for(unsigned short i{0}; i < 4; ++i)
            if(v1[i] != b)
                return false;

        return true;
    }


    bool Long256::operator!=(const Long256& b) const {
        long long* v1,* v2;
        v1 = (long long*)&v;
        v2 = (long long*)&b.v;

        for(unsigned short i{0}; i < 4; ++i)
            if(v1[i] != v2[i])
                return true;

        return false;
    }


    bool Long256::operator!=(const long long& b) const {
        long long* v1 = (long long*)&v;

        for(unsigned short i{0}; i < 4; ++i)
            if(v1[i] != b)
                return true;

        return false;
    }


    long long Long256::operator[](unsigned long long index) const {
        if(index > 4) {
            std::string error_text = "Invalid index! Valid range is [0-7] (was ";
            error_text += std::to_string(index);
            error_text += ").";
            throw std::out_of_range(error_text);
        }
        long long* tmp = (long long*)&v;
        return tmp[index];
    }


    Long256 Long256::operator+(const Long256& b) const {
        return Long256(_mm256_add_epi64(v, b.v));
    }


    Long256 Long256::operator+(const long long& b) const{
        return Long256(
            _mm256_add_epi64(
                v, 
                _mm256_set1_epi64x(b)
            )
        );
    }


    Long256 Long256::operator-(const Long256& b) const{
        return Long256(
            _mm256_sub_epi64(
                v, 
                b.v
            )
        );
    }


    Long256 Long256::operator-(const long long& b) const{
        return Long256(
            _mm256_sub_epi64(
                v, 
                _mm256_set1_epi64x(b)
            )
        );
    }


    Long256 Long256::operator*(const Long256& b) const{
        long long* av = (long long*)&v, *bv = (long long*)&b.v;
       return Long256(
        _mm256_set_epi64x(
            av[3] * bv[3],
            av[2] * bv[2],
            av[1] * bv[1],
            av[0] * bv[0]
        )
       );
        // return Long256(
        //     _mm256_mullo_epi64(
        //         v, 
        //         b.v
        //     )
        // );
    }


    Long256 Long256::operator*(const long long& b) const{

        __m256d tmpa = _mm256_castsi256_pd(v);
        __m256d c = _mm256_mul_pd(tmpa,_mm256_set1_pd(b));
        return Long256(
            _mm256_castpd_si256(c)
        );
        // return Long256(
        //     _mm256_mullo_epi64(
        //         v, 
        //         _mm256_set1_epi64x(b)
        //     )
        // );
    }

    // TODO: Optimize division and modulo.
    Long256 Long256::operator/(const Long256& b) const {

        long long* a = (long long*)&v;
        long long* bv = (long long*)&b.v;

        return Long256(
            _mm256_set_epi64x(
                a[3] / bv[3],
                a[2] / bv[2],
                a[1] / bv[1],
                a[0] / bv[0]
            )
        );
    }


    Long256 Long256::operator/(const long long& b) const {

        long long* a = (long long*)&v;

        return Long256(
            _mm256_set_epi64x(
                a[3] / b,
                a[2] / b,
                a[1] / b,
                a[0] / b
            )
        );
    }


    Long256 Long256::operator%(const Long256& b) const{
        long long* a = (long long*)&v;
        long long* bv = (long long*)&b.v;

        return Long256(
            _mm256_set_epi64x(
                a[3] % bv[3],
                a[2] % bv[2],
                a[1] % bv[1],
                a[0] % bv[0]
            )
        );
    }


    Long256 Long256::operator%(const long long& b) const{
        long long* a = (long long*)&v;

        return Long256(
            _mm256_set_epi64x(
                a[3] % b,
                a[2] % b,
                a[1] % b,
                a[0] % b
            )
        );

    }


    Long256 Long256::operator^(const Long256& b) const{
        return Long256(
            _mm256_xor_si256(
                v, 
                b.v
            )
        );
    }


    Long256 Long256::operator^(const long long& b) const{
        return Long256(
            _mm256_xor_si256(
                v, 
                _mm256_set1_epi64x(b)
            )
        );
    }


    Long256 Long256::operator|(const Long256& b) const{
        return Long256(
            _mm256_or_si256(
                v, 
                b.v
            )
        );
    }


    Long256 Long256::operator|(const long long& b) const{
        return Long256(
            _mm256_or_si256(
                v, 
                _mm256_set1_epi64x(b)
            )
        );
    }


    Long256 Long256::operator&(const Long256& b) const{
        return Long256(
            _mm256_and_si256(
                v, 
                b.v
            )
        );
    }


    Long256 Long256::operator&(const long long& b) const{
        return Long256(
            _mm256_and_si256(
                v, 
                _mm256_set1_epi64x(b)
            )
        );
    }


    Long256 Long256::operator~() const {
        return Long256(
            _mm256_xor_si256(v, ones)
        );
    }


    Long256 Long256::operator<<(const Long256& b) const {
        return Long256(
            _mm256_sllv_epi64(
                v,
                b.v
            )
        );
    }
    
    
    Long256 Long256::operator<<(const unsigned int& b) const {
        return Long256(
            _mm256_slli_epi64(
                v,
                b
            )
        );
    }

    
    Long256 Long256::operator>>(const Long256& b) const {
        return Long256(
            _mm256_srlv_epi64(
                v,
                b.v
            )
        );
    }
    
    
    Long256 Long256::operator>>(const unsigned int& b) const {
        return Long256(
            _mm256_srli_epi64(
                v,
                b
            )
        );
    }


    Long256& Long256::operator+=(const Long256& b) {
        v = _mm256_add_epi64(v,b.v);
        return *this;
    }


    Long256& Long256::operator+=(const long long& b) {
        v = _mm256_add_epi64(v, _mm256_set1_epi64x(b));
        return *this;
    }


    Long256& Long256::operator-=(const Long256& b) {
        v = _mm256_sub_epi64(v,b.v);
        return *this;
    }


    Long256& Long256::operator-=(const long long& b) {
        v = _mm256_sub_epi64(v, _mm256_set1_epi64x(b));
        return *this;
    }


    Long256& Long256::operator*=(const Long256& b){
        // TODO: Fix multiplication when AVX-512 is not available
        // v = _mm256_mullo_epi64(v, b.v);
        long long* av = (long long*)&v, *bv = (long long*)&b.v;
        v = _mm256_set_epi64x(
            av[3] * bv[3],
            av[2] * bv[2],
            av[1] * bv[1],
            av[0] * bv[0]
        );

        return *this;
    }


    Long256& Long256::operator*=(const long long& b){
        // TODO: Fix multiplication w/o AVX-512
        // v = _mm256_mullo_epi64(v, _mm256_set1_epi64x(b));
        long long* av = (long long*)&v;
        // Check if works.
        v = _mm256_insert_epi64(v, av[3] * b, 3);
        v = _mm256_insert_epi64(v, av[2] * b, 2);
        v = _mm256_insert_epi64(v, av[1] * b, 1);
        v = _mm256_insert_epi64(v, av[0] * b, 0);
        return *this;
    }


    Long256& Long256::operator/=(const Long256& b){
        long long* a = (long long*)&v;
        long long* bv = (long long*)&b.v;

        v = _mm256_set_epi64x(
            a[3] / bv[3],
            a[2] / bv[2],
            a[1] / bv[1],
            a[0] / bv[0]
        );
        return *this;
    }


    Long256& Long256::operator/=(const long long& b){
        long long* a = (long long*)&v;
        v = _mm256_set_epi64x(
            a[3] / b,
            a[2] / b,
            a[1] / b,
            a[0] / b
        );
        return *this;
    }


    Long256& Long256::operator%=(const Long256& b) {
        long long* a = (long long*)&v;
        long long* bv = (long long*)&b.v;

        v = _mm256_set_epi64x(
            a[3] % bv[3],
            a[2] % bv[2],
            a[1] % bv[1],
            a[0] % bv[0]
        );
        return *this;
    }


    Long256& Long256::operator%=(const long long& b){
        long long* a = (long long*)&v;

        v = _mm256_set_epi64x(
                a[3] % b,
                a[2] % b,
                a[1] % b,
                a[0] % b
            );
        return *this;
    }


    Long256& Long256::operator|=(const Long256& b){
        v = _mm256_or_si256(v, b.v);
        return *this;
    }


    Long256& Long256::operator|=(const long long& b){
        v = _mm256_or_si256(v, _mm256_set1_epi64x(b));
        return *this;
    }


    Long256& Long256::operator&=(const Long256& b){
        v = _mm256_and_si256(v, b.v);
        return *this;
    }


    Long256& Long256::operator&=(const long long& b){
        v = _mm256_and_si256(v, _mm256_set1_epi64x(b));
        return *this;
    }


    Long256& Long256::operator^=(const Long256& b){
        v = _mm256_xor_si256(v, b.v);
        return *this;
    }


    Long256& Long256::operator^=(const long long& b){
        v = _mm256_xor_si256(v, _mm256_set1_epi64x(b));
        return *this;
    }


    Long256& Long256::operator<<=(const Long256& b) {
        v = _mm256_sllv_epi64(
            v,
            b.v
        );
        return *this;
    }
    
    
    Long256& Long256::operator<<=(const unsigned int& b) {
        v = _mm256_slli_epi64(
            v,
            b
        );
        return *this;
    }

    
    Long256& Long256::operator>>=(const Long256& b) {
        v = _mm256_srlv_epi64(
            v,
            b.v
        );
        return *this;
    }
    
    
    Long256& Long256::operator>>=(const unsigned int& b) {
        v = _mm256_srli_epi64(
            v,
            b
        );
        return *this;
    }


    std::string Long256::str() const{
        std::string result = "Long256(";
        long long* iv = (long long*)&v; 
        for(unsigned i{0}; i < 3; ++i)
            result += std::to_string(iv[i]) + ", ";
        
        result += std::to_string(iv[3]);
        result += ")";
        return result;
    }

    
    void Long256::save(std::array<long long, 4>& dest) const{
        _mm256_storeu_si256((__m256i*)dest.data(), v);
    }

            

    void Long256::save(const long long* dest) const{
        _mm256_storeu_si256((__m256i*)dest, v);
    }


    void Long256::saveAligned(const long long* dest) const{
        _mm256_store_si256((__m256i*)dest, v);
    }

    Long256 sum(std::vector<Long256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Long256& item : a)
            result = _mm256_add_epi64(result, item.v);
        
        return Long256(result);
    }


    Long256 sum(std::set<Long256>& a){
        __m256i result = _mm256_setzero_si256();
        for(const Long256& item : a)
            result = _mm256_add_epi64(result, item.v);
        
        return Long256(result);
    }
};