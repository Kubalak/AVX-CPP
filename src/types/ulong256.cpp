#include "ulong256.hpp"
#include <stdexcept>
namespace avx {

    ULong256::ULong256(const unsigned long long& init):
        v(_mm256_set1_epi64x(init))
    {}


    ULong256::ULong256(const unsigned long long* init):
        v(_mm256_lddqu_si256((const __m256i*)init))
    {}


    ULong256::ULong256(const std::array<unsigned long long, 4>& init):
        v(_mm256_lddqu_si256((const __m256i*)init.data()))
    {}


    ULong256::ULong256(const std::array<unsigned int, 4>& init):
        v(_mm256_set_epi64x(
            init[0],
            init[1],
            init[2],
            init[3]
            )
        )
    {}


    ULong256::ULong256(const std::array<unsigned short, 4>& init):
        v(_mm256_set_epi64x(
            init[0],
            init[1],
            init[2],
            init[3]
            )
        )
    {}


    ULong256::ULong256(const std::array<unsigned char, 4>& init):
        v(_mm256_set_epi64x(
            init[0],
            init[1],
            init[2],
            init[3]
            )
        )
    {}


    ULong256::ULong256(std::initializer_list<unsigned long long> init) {
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


    void ULong256::save(std::array<unsigned long long, 4>& dest) const {
        _mm256_storeu_si256((__m256i *)dest.data(), v);
    }


    void ULong256::save(unsigned long long* dest) const {
        _mm256_storeu_si256((__m256i *)dest, v);   
    }


    void ULong256::saveAligned(unsigned long long* dest) const {
        _mm256_store_si256((__m256i *)dest, v);   
    }


    bool ULong256::operator==(const ULong256& b) const {
        unsigned long long* v1,* v2;
        v1 = (unsigned long long*)&v;
        v2 = (unsigned long long*)&b.v;

        for(unsigned short i{0}; i < 4; ++i)
            if(v1[i] != v2[i])
                return false;

        return true;
    }


    bool ULong256::operator==(const unsigned long long& b) const {
        unsigned long long* v1 = (unsigned long long*)&v;

        for(unsigned short i{0}; i < 4; ++i)
            if(v1[i] != b)
                return false;

        return true;
    }


    bool ULong256::operator!=(const ULong256& b) const {
        unsigned long long* v1,* v2;
        v1 = (unsigned long long*)&v;
        v2 = (unsigned long long*)&b.v;

        for(unsigned short i{0}; i < 4; ++i)
            if(v1[i] != v2[i])
                return true;

        return false;
    }


    bool ULong256::operator!=(const unsigned long long& b) const {
        unsigned long long* v1 = (unsigned long long*)&v;

        for(unsigned short i{0}; i < 4; ++i)
            if(v1[i] != b)
                return true;

        return false;
    }


    unsigned long long ULong256::operator[](unsigned int index) const {
        if(index > 4)
            throw std::out_of_range("Range be within range 0-3! Got: " + std::to_string(index));
            
        return ((unsigned long long*)&v)[index];
    }


    ULong256 ULong256::operator+(const ULong256& b) const {
        return _mm256_add_epi64(v, b.v);
    }


    ULong256 ULong256::operator+(const unsigned long long& b) const{
        return _mm256_add_epi64(
            v, 
            _mm256_set1_epi64x(b)
        );
    }


    ULong256 ULong256::operator-(const ULong256& b) const{
        return _mm256_sub_epi64(v, b.v);
    }


    ULong256 ULong256::operator-(const unsigned long long& b) const{
        return _mm256_sub_epi64(
            v, 
            _mm256_set1_epi64x(b)
        );
    }


    ULong256 ULong256::operator*(const ULong256& b) const {
        #ifdef USE_AVX_512
            return _mm256_mullo_epi64(v, b.v);
        #else
            unsigned long long* aP = (unsigned long long*)&v;
            unsigned long long* bP = (unsigned long long*)&b.v;
            unsigned long long result[] = {
                aP[0] * bP[0],
                aP[1] * bP[1],
                aP[2] * bP[2],
                aP[3] * bP[3]
            };

            return _mm256_lddqu_si256((const __m256i*)result);
        #endif
    }


    ULong256 ULong256::operator*(const unsigned long long& b) const {
        #ifdef USE_AVX_512
            return _mm256_mullo_epi64(v, _mm256_set1_epi64x(b));
        #else
            unsigned long long* aP = (unsigned long long*)&v;
            unsigned long long result[] = {
                aP[0] * b,
                aP[1] * b,
                aP[2] * b,
                aP[3] * b
            };
            
            return _mm256_lddqu_si256((const __m256i*)result);
        #endif
    }


    ULong256 ULong256::operator/(const ULong256& b) const {
        unsigned long long* aP = (unsigned long long*)&v;
        unsigned long long* bP = (unsigned long long*)&b.v;
        unsigned long long result[] = {
            aP[0] / bP[0],
            aP[1] / bP[1],
            aP[2] / bP[2],
            aP[3] / bP[3]
        };

        return result;
    }


    ULong256 ULong256::operator/(const unsigned long long& b) const {
        unsigned long long* aP = (unsigned long long*)&v;
        unsigned long long result[] = {
            aP[0] / b,
            aP[1] / b,
            aP[2] / b,
            aP[3] / b
        };
        
        return result;
    }


    ULong256 ULong256::operator%(const ULong256& b) const {
        unsigned long long* aP = (unsigned long long*)&v;
        unsigned long long* bP = (unsigned long long*)&b.v;
        unsigned long long result[] = {
            aP[0] % bP[0],
            aP[1] % bP[1],
            aP[2] % bP[2],
            aP[3] % bP[3]
        };

        return result;
    }


    ULong256 ULong256::operator%(const unsigned long long& b) const {
        unsigned long long* aP = (unsigned long long*)&v;
        unsigned long long result[] = {
            aP[0] % b,
            aP[1] % b,
            aP[2] % b,
            aP[3] % b
        };
        
        return result;
    }


    ULong256 ULong256::operator&(const ULong256& b) const {
        return _mm256_and_si256(v, b.v);
    }


    ULong256 ULong256::operator&(const unsigned long long& b) const {
        return _mm256_and_si256(
            v, 
            _mm256_set1_epi64x(b)
        );
    }

    
    ULong256 ULong256::operator|(const ULong256& b) const {
        return _mm256_or_si256(v, b.v);
    }


    ULong256 ULong256::operator|(const unsigned long long& b) const {
        return _mm256_or_si256(
            v, 
            _mm256_set1_epi64x(b)
        );
    }

    
    ULong256 ULong256::operator^(const ULong256& b) const {
        return _mm256_xor_si256(v, b.v);
    }


    ULong256 ULong256::operator^(const unsigned long long& b) const {
        return _mm256_xor_si256(
            v, 
            _mm256_set1_epi64x(b)
        );
    }


    ULong256 ULong256::operator~() const {
        return _mm256_xor_si256(v, constants::ONES);
    }


    ULong256 ULong256::operator<<(const ULong256& b) const {
        return _mm256_sllv_epi64(v, b.v);
    }


    ULong256  ULong256::operator<<(const unsigned long long& b) const {
        return _mm256_sllv_epi64(v, _mm256_set1_epi64x(b));
    }


    ULong256  ULong256::operator>>(const ULong256& b) const {
        return _mm256_srlv_epi64(v, b.v);
    }


    ULong256  ULong256::operator>>(const unsigned long long& b) const {
        return _mm256_srlv_epi64(v, _mm256_set1_epi64x(b));
    }


    ULong256& ULong256::operator+=(const ULong256& b) {
        v = _mm256_add_epi64(v, b.v);
        return *this;
    }


    ULong256& ULong256::operator+=(const unsigned long long& b) {
        v = _mm256_add_epi64(
            v,
            _mm256_set1_epi64x(b)
        );
        return *this;
    }


    ULong256& ULong256::operator-=(const ULong256& b) {
        v = _mm256_sub_epi64(v, b.v);
        return *this;
    }


    ULong256& ULong256::operator-=(const unsigned long long& b) {
        v = _mm256_sub_epi64(
            v,
            _mm256_set1_epi64x(b)
        );
        return *this;
    }


    ULong256& ULong256::operator*=(const ULong256& b) {
        #ifdef USE_AVX_512
            v = _mm256_mullo_epi64(v, b.v);
        #else
            unsigned long long* aP = (unsigned long long*)&v;
            unsigned long long* bP = (unsigned long long*)&b.v;
            unsigned long long result[] = {
                aP[0] * bP[0],
                aP[1] * bP[1],
                aP[2] * bP[2],
                aP[3] * bP[3]
            };

            v = _mm256_lddqu_si256((const __m256i*)result);
        #endif

        return *this;
    }


    ULong256& ULong256::operator*=(const unsigned long long& b) {
        #ifdef USE_AVX_512
            v = _mm256_mullo_epi64(v, _mm256_set1_epi64x(b));
        #else
            unsigned long long* aP = (unsigned long long*)&v;
            unsigned long long result[] = {
                aP[0] * b,
                aP[1] * b,
                aP[2] * b,
                aP[3] * b
            };
            
            v = _mm256_lddqu_si256((const __m256i*)result);
        #endif

        return *this;
    }


    ULong256& ULong256::operator/=(const ULong256& b) {
        unsigned long long* aP = (unsigned long long*)&v;
        unsigned long long* bP = (unsigned long long*)&b.v;
        unsigned long long result[] = {
            aP[0] / bP[0],
            aP[1] / bP[1],
            aP[2] / bP[2],
            aP[3] / bP[3]
        };
        v = _mm256_lddqu_si256((const __m256i*)result);

        return *this;
    }


    ULong256& ULong256::operator/=(const unsigned long long& b) {
        unsigned long long* aP = (unsigned long long*)&v;
        unsigned long long result[] = {
            aP[0] / b,
            aP[1] / b,
            aP[2] / b,
            aP[3] / b
        };
        v = _mm256_lddqu_si256((const __m256i*)result);

        return *this;
    }


    ULong256& ULong256::operator%=(const ULong256& b) {
        unsigned long long* aP = (unsigned long long*)&v;
        unsigned long long* bP = (unsigned long long*)&b.v;
        unsigned long long result[] = {
            aP[0] % bP[0],
            aP[1] % bP[1],
            aP[2] % bP[2],
            aP[3] % bP[3]
        };
        v = _mm256_lddqu_si256((const __m256i*)result);

        return *this;
    }


    ULong256& ULong256::operator%=(const unsigned long long& b) {
        unsigned long long* aP = (unsigned long long*)&v;
        unsigned long long result[] = {
            aP[0] % b,
            aP[1] % b,
            aP[2] % b,
            aP[3] % b
        };
        v = _mm256_lddqu_si256((const __m256i*)result);

        return *this;
    }


    ULong256& ULong256::operator|=(const ULong256& b) {
        v = _mm256_or_si256(v, b.v);
        return *this;
    }


    ULong256& ULong256::operator|=(const unsigned long long& b) {
        v = _mm256_or_si256(
            v, 
            _mm256_set1_epi64x(b)
        );
        return *this;
    }


    ULong256& ULong256::operator&=(const ULong256& b) {
        v = _mm256_and_si256(v, b.v);
        return *this;
    }


    ULong256& ULong256::operator&=(const unsigned long long& b) {
         v = _mm256_and_si256(
            v, 
            _mm256_set1_epi64x(b)
        );
        return *this;
    }


    ULong256& ULong256::operator^=(const ULong256& b) {
        v = _mm256_xor_si256(v, b.v);
        return *this;
    }


    ULong256& ULong256::operator^=(const unsigned long long& b) {
         v = _mm256_xor_si256(
            v, 
            _mm256_set1_epi64x(b)
        );
        return *this;
    }


    ULong256& ULong256::operator<<=(const ULong256& b) {
        v = _mm256_sllv_epi64(v, b.v);
        return *this;
    }


    ULong256& ULong256::operator<<=(const unsigned long long& b) {
        v = _mm256_sllv_epi64(v, _mm256_set1_epi64x(b));
        return *this;
    }


    ULong256& ULong256::operator>>=(const ULong256& b) {
        v = _mm256_srlv_epi64(v, b.v);
        return *this;
    }


    ULong256& ULong256::operator>>=(const unsigned long long& b) {
        v = _mm256_srlv_epi64(v, _mm256_set1_epi64x(b));
        return *this;
    }



    std::string ULong256::str() const{
        std::string result = "Long256(";
        unsigned long long* iv = (unsigned long long*)&v; 
        for(unsigned i{0}; i < 3; ++i)
            result += std::to_string(iv[i]) + ", ";
        
        result += std::to_string(iv[3]);
        result += ")";
        return result;
    }
}