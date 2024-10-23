#include "uchar256.hpp"


namespace avx {

    std::ostream& operator<<(std::ostream& os, const UChar256& a) {
        alignas(32) unsigned char tmp[33];
        tmp[32] = '\0';

        _mm256_store_si256((__m256i*)tmp, a.v);
        
        os << tmp;
        return os;
    }

}