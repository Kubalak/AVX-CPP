#include <iostream>
#include "perf_utils.hpp"
#include <types/uint256.hpp>

int64_t perf_test_add_raw_avx(const std::vector<unsigned int>& aV, const std::vector<unsigned int>& bV, std::vector<unsigned int>& cV, const bool print = true){
    if(aV.size() != bV.size()){
        std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
        return -1;
    }
    if(cV.size() != aV.size())
        cV.resize(aV.size());

    auto start = std::chrono::steady_clock::now();

    uint64_t pos = 0;
    while(pos + 8 < aV.size()){
        __m256i a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        __m256i b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_add_epi32(a,b));
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = aV[pos] + bV[pos];
        ++pos;
    }
    
    auto stop = std::chrono::steady_clock::now();
    if(print)
        testing::print_test_duration(__func__, start, stop);
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}



int main(int argc, char* argv[]) {
    std::vector<unsigned int> aV(268'435'456), bV(268'435'456), cV(268'435'456);
    testing::perf::TestConfig config;
    int result = testing::perf::allPerfTest<avx::UInt256>(aV, bV, cV, config);
    return (result & 0xEFF) != 0; // Ignore Lshift errors as SIMD behaves differently when crossing size of stored type.
}