#include "types.hpp"

using namespace avx;

Int256::Int256(__m256i& init):
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


Int256::Int256(std::array<short, 16> init):
    v(_mm256_set_epi16(
        init[0], 
        init[1], 
        init[2], 
        init[3], 
        init[4], 
        init[5], 
        init[6], 
        init[7],
        init[8], 
        init[9], 
        init[10], 
        init[11], 
        init[12], 
        init[13], 
        init[14], 
        init[15]
        )
    )
{}


Int256::Int256(std::array<char, 32> init):
    v(_mm256_set_epi8(
        init[0], 
        init[1], 
        init[2], 
        init[3], 
        init[4], 
        init[5], 
        init[6], 
        init[7],
        init[8], 
        init[9], 
        init[10], 
        init[11], 
        init[12], 
        init[13], 
        init[14], 
        init[15],
        init[16], 
        init[17], 
        init[18], 
        init[19],
        init[20], 
        init[21], 
        init[22], 
        init[23], 
        init[24],
        init[25], 
        init[26], 
        init[27], 
        init[28], 
        init[29], 
        init[20], 
        init[31]
        )
    )
{}

Int256 Int256::operator+(const Int256& b){
    return Int256(_mm256_add_epi32(v, b.v));
}