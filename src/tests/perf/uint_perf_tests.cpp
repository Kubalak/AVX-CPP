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

std::string validationToStr(int64_t retcode){
    if(retcode == -2) return "E_INVAL_SIZE";
    if(retcode == -1) return "OK";
    return "[" + std::to_string(retcode) + "]";
}


int main(int argc, char* argv[]) {
    for(int i = 0; i < argc; ++i)
        std::cout << argv[i] << ' ';
    std::cout << '\n';

    auto start = std::chrono::steady_clock::now();
    std::vector<unsigned int> aV(268'435'456), bV(268'435'456), cV(268'435'456);
    
    srand(42); // Get repeated values.

    for(unsigned int i{0}; i < aV.size(); ++i){
        aV[i] = rand();
        bV[i] = rand() | 1;
    }
    auto stop = std::chrono::steady_clock::now();

    auto[value, unit] = testing::universal_duration(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

    std::cout << "Preparation took: " << value << ' ' << unit << '\n';
    std::cout << "Testing performance of: " << testing::demangle(typeid(avx::UInt256).name()) <<'\n';

    uint64_t times[11];
    int64_t validations[10];
    testing::perf::doCPUWarmup(100, true);

    times[0] = perf_test_add_raw_avx(aV, bV, cV, false);
    times[1] = testing::perf::testAddAVX<avx::UInt256>(aV, bV, cV, false);
    validations[0] = testing::perf::verifyAdd(aV, bV, cV, false);
    times[2] = testing::perf::testAddSeq<unsigned int>(aV, bV, cV, false);
    validations[1] = testing::perf::verifyAdd(aV, bV, cV, false);
    times[3] = testing::perf::testMulAVX<avx::UInt256>(aV, bV, cV, false);
    validations[2] = testing::perf::verifyMul(aV, bV, cV, false);
    times[4] = testing::perf::testMulSeq<unsigned int>(aV, bV, cV, false);
    validations[3] = testing::perf::verifyMul(aV, bV, cV, false);
    times[5] = testing::perf::testDivAVX<avx::UInt256>(aV, bV, cV, false);
    validations[4] = testing::perf::verifyDiv(aV, bV, cV, false);
    times[6] = testing::perf::testDivSeq<unsigned int>(aV, bV, cV, false);
    validations[5] = testing::perf::verifyDiv(aV, bV, cV, false);
    times[7] = testing::perf::testModAVX<avx::UInt256>(aV, bV, cV, false);
    validations[6] = testing::perf::verifyMod(aV, bV, cV, false);
    times[8] = testing::perf::testModSeq<unsigned int>(aV, bV, cV, false);
    validations[7] = testing::perf::verifyMod(aV, bV, cV, false);
    for(size_t i{0}; i< bV.size(); ++i)
        bV[i] &= (sizeof(unsigned int) * 8 - 1);
    times[9] = testing::perf::testLshiftAVX<avx::UInt256>(aV, bV, cV, false);
    validations[8] = testing::perf::verifyLshift(aV, bV, cV, false);
    times[10] = testing::perf::testLshiftSeq<unsigned int>(aV, bV, cV, false);
    validations[9] = testing::perf::verifyLshift(aV, bV, cV, false);

    auto duration = testing::universal_duration(times[0]);
    
    printf("%-20s %8.4lf %-3s\n", "Test add raw AVX2:", duration.first, duration.second.c_str());
    
    duration = testing::universal_duration(times[1]);
    
    printf("%-20s %8.4lf %-3s%s\n", "Test add AVX2:", duration.first, duration.second.c_str(), validationToStr(validations[0]).c_str());
    duration = testing::universal_duration(times[2]);
    printf("%-20s %8.4lf %-3s%s\n", "Test add seq:", duration.first, duration.second.c_str(), validationToStr(validations[1]).c_str());

    duration = testing::universal_duration(times[3]);
    printf("%-20s %8.4lf %-3s%s\n", "Test mul AVX2:", duration.first, duration.second.c_str(), validationToStr(validations[2]).c_str());
    duration = testing::universal_duration(times[4]);
    printf("%-20s %8.4lf %-3s%s\n", "Test mul seq:", duration.first, duration.second.c_str(), validationToStr(validations[3]).c_str());

    duration = testing::universal_duration(times[5]);
    printf("%-20s %8.4lf %-3s%s\n", "Test div AVX2:", duration.first, duration.second.c_str(), validationToStr(validations[4]).c_str());
    duration = testing::universal_duration(times[6]);
    printf("%-20s %8.4lf %-3s%s\n", "Test div seq:", duration.first, duration.second.c_str(), validationToStr(validations[5]).c_str());

    duration = testing::universal_duration(times[7]);
    printf("%-20s %8.4lf %-3s%s\n", "Test mod AVX2:", duration.first, duration.second.c_str(), validationToStr(validations[6]).c_str());
    duration = testing::universal_duration(times[8]);
    printf("%-20s %8.4lf %-3s%s\n", "Test mod seq:", duration.first, duration.second.c_str(), validationToStr(validations[7]).c_str());

    duration = testing::universal_duration(times[9]);
    printf("%-20s %8.4lf %-3s%s\n", "Test lshift AVX2:", duration.first, duration.second.c_str(), validationToStr(validations[8]).c_str());
    duration = testing::universal_duration(times[10]);
    printf("%-20s %8.4lf %-3s%s\n", "Test lshift seq:", duration.first, duration.second.c_str(), validationToStr(validations[9]).c_str());

    return 0;
}