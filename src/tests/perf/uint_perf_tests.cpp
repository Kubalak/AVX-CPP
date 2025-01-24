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
    unsigned int cLit = bV[bV.size() / 2];
    __m256i c = _mm256_set1_epi32(cLit);
    while(pos + 8 < aV.size()){
        __m256i a = _mm256_lddqu_si256((const __m256i*)(aV.data() + pos));
        __m256i b = _mm256_lddqu_si256((const __m256i*)(bV.data() + pos));
        _mm256_storeu_si256((__m256i*)(cV.data() + pos), _mm256_add_epi32(_mm256_add_epi32(a,b), c));
        pos += 8;
    }

    while(pos < aV.size()){
        cV[pos] = aV[pos] + bV[pos];
        cV[pos] += cLit;
        ++pos;
    }
    
    auto stop = std::chrono::steady_clock::now();
    if(print)
        testing::printTestDuration(__func__, start, stop);
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
}



int main(int argc, char* argv[]) {
    std::vector<unsigned int> aV(268'435'456), bV(268'435'456), cV(268'435'456);
    testing::perf::TestConfig<unsigned int> config;
    config.avxFuncs.addRaw = perf_test_add_raw_avx;
    /*std::srand(config.randomSeed);
    for(size_t i = 0; i < aV.size(); ++i){               
        aV[i] = std::rand();
        bV[i] = std::rand() | 1;
    }
    std::cout << "Items count: " << aV.size() << '\n';
    std::cout << avx::UInt256(aV.data() + 5544480).str() << " / " << avx::UInt256(bV.data() + 5544480).str() << " = \033[31m" << (avx::UInt256(aV.data() + 5544480) / avx::UInt256(bV.data() + 5544480)).str() << "\033[0m\n";
    printf("As unsigned: %u / %u = \033[0;32m%u\033[0m\n", 1472234252u, 5057u, 1472234252u /5057u);
    printf("As float:    %f / %f = %f\n", 1472234252.f, 5057.f, 1472234252.f / 5057.f);
    printf("As double:   %lf / %lf = %lf\n", 1472234252., 5057., 1472234252. / 5057.);
    testing::perf::testModAVX<avx::UInt256>(aV, bV, cV);
    printf("%u %% %u = %u (valid = %u)\n", aV[5544480], bV[5544480], cV[5544480], aV[5544480] % bV[5544480]);
    return 0;8*/
    int result = testing::perf::allPerfTest<avx::UInt256>(aV, bV, cV, config);
    return (result & 0xEFF) != 0; // Ignore Lshift errors as SIMD behaves differently when crossing size of stored type.
}