#include "ulong256.hpp"
#include <stdexcept>

namespace avx {
    ULong256 sum(const std::vector<ULong256>& ulongs) noexcept {
        ULong256 a;
        for(const ULong256& ulong : ulongs)
            a.v = _mm256_add_epi64(a.v, ulong.v);
        
        return a;
    }

    ULong256 sum(const std::set<ULong256>& ulongs) noexcept {
        ULong256 a;
        for(const ULong256& ulong : ulongs)
            a.v = _mm256_add_epi64(a.v, ulong.v);
        
        return a;
    }
}