#include <iostream>
#include "perf_utils.hpp"
#include <types/int256.hpp>

int64_t perf_test_add_raw_avx(const std::vector<int>& aV, const std::vector<int>& bV, std::vector<int>& cV, const bool print = true){
    if(aV.size() != bV.size()){
        std::cerr << "Sizes don't match (" << aV.size() << " != " << bV.size() << ")!\n";
        return -1;
    }
    if(cV.size() != aV.size())
        cV.resize(aV.size());

    auto start = std::chrono::steady_clock::now();

    uint64_t pos = 0;
    int cLit = bV[bV.size() / 2];
    __m256i a, b, c;
    __m256i d = _mm256_set1_epi32(cLit);
    
    while(pos + 8 < aV.size()){
        a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        c = _mm256_add_epi32(a, b);
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_add_epi32(c, d));
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = *(aV.data()+ pos) + *(bV.data() + pos);
        cV[pos] += cLit;
        ++pos;
    }
    
    auto stop = std::chrono::steady_clock::now();
    if(print)
        testing::printTestDuration(__func__, start, stop);
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}



int main(int argc, char* argv[]) {
    std::vector<int> aV(268'435'456), bV(268'435'456), cV(268'435'456);
    testing::perf::TestConfig<int> config;
    config.avxFuncs.addRaw = perf_test_add_raw_avx;
    config.doWarmup = true;
    config.warmupDuration = 20;
    config.printWarmupInfo = true;
    int result = testing::perf::allPerfTest<avx::Int256>(aV, bV, cV, config);
    return (result & _AVX_IGNORE_LSH) != 0; // Ignore Lshift errors as SIMD behaves differently when crossing size of stored type.
}
