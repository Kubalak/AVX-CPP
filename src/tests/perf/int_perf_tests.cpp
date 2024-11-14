#include <iostream>
#include "perf_utils.hpp"
#include <types/int256.hpp>


int main(int argc, char* argv[]) {

    std::cout << "Started performance test for " << testing::demangle(typeid(avx::Int256).name()) << " {" << testing::demangle(typeid(int).name()) << "}\n";
    auto start = std::chrono::steady_clock::now();
    std::vector<int> aV(268'435'456), bV(268'435'456), cV(268'435'456);
    
    srand(42); // Get repeated values.

    for(unsigned int i{0}; i < aV.size(); ++i){
        aV[i] = rand();
        bV[i] = rand() | 1;
    }
    auto stop = std::chrono::steady_clock::now();

    auto[value, unit] = testing::universal_duration(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

    std::cout << "Preparation took: " << value << ' ' << unit << '\n';

    uint64_t times[10];
    std::tuple<int64_t, int, int> validations[10];
    testing::perf::doCPUWarmup(100, true);

    times[0] = testing::perf::testAddAVX<avx::Int256>(aV, bV, cV, false);
    validations[0] = testing::perf::verifyAdd(aV, bV, cV, false);
    times[1] = testing::perf::testAddSeq<int>(aV, bV, cV, false);
    validations[1] = testing::perf::verifyAdd(aV, bV, cV, false);
    times[2] = testing::perf::testMulAVX<avx::Int256>(aV, bV, cV, false);
    validations[2] = testing::perf::verifyMul(aV, bV, cV, false);
    times[3] = testing::perf::testMulSeq<int>(aV, bV, cV, false);
    validations[3] = testing::perf::verifyMul(aV, bV, cV, false);
    times[4] = testing::perf::testDivAVX<avx::Int256>(aV, bV, cV, false);
    validations[4] = testing::perf::verifyDiv(aV, bV, cV, false);
    times[5] = testing::perf::testDivSeq<int>(aV, bV, cV, false);
    validations[5] = testing::perf::verifyDiv(aV, bV, cV, false);
    times[6] = testing::perf::testModAVX<avx::Int256>(aV, bV, cV, false);
    validations[6] = testing::perf::verifyMod(aV, bV, cV, false);
    times[7] = testing::perf::testModSeq<int>(aV, bV, cV, false);
    validations[7] = testing::perf::verifyMod(aV, bV, cV, false);
    
    for(size_t i{0}; i< bV.size(); ++i)
        bV[i] &= (sizeof(int) * 8 - 1);
    times[8] = testing::perf::testLshiftAVX<avx::Int256>(aV, bV, cV, false);
    validations[8] = testing::perf::verifyLshift(aV, bV, cV, false);
    times[9] = testing::perf::testLshiftSeq<int>(aV, bV, cV, false);
    validations[9] = testing::perf::verifyLshift(aV, bV, cV, false);

    auto duration = testing::universal_duration(times[0]);
    
    printf("%-20s %8.4lf %-3s%s\n", "Test add AVX2:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[0]).c_str());
    duration = testing::universal_duration(times[1]);
    printf("%-20s %8.4lf %-3s%s\n", "Test add seq:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[1]).c_str());

    duration = testing::universal_duration(times[2]);
    printf("%-20s %8.4lf %-3s%s\n", "Test mul AVX2:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[2]).c_str());
    duration = testing::universal_duration(times[3]);
    printf("%-20s %8.4lf %-3s%s\n", "Test mul seq:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[3]).c_str());

    duration = testing::universal_duration(times[4]);
    printf("%-20s %8.4lf %-3s%s\n", "Test div AVX2:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[4]).c_str());
    duration = testing::universal_duration(times[5]);
    printf("%-20s %8.4lf %-3s%s\n", "Test div seq:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[5]).c_str());

    duration = testing::universal_duration(times[6]);
    printf("%-20s %8.4lf %-3s%s\n", "Test mod AVX2:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[6]).c_str());
    duration = testing::universal_duration(times[7]);
    printf("%-20s %8.4lf %-3s%s\n", "Test mod seq:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[7]).c_str());

    duration = testing::universal_duration(times[8]);
    printf("%-20s %8.4lf %-3s%s\n", "Test lshift AVX2:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[8]).c_str());
    duration = testing::universal_duration(times[9]);
    printf("%-20s %8.4lf %-3s%s\n", "Test lshift seq:", duration.first, duration.second.c_str(), testing::perf::validationToStr(validations[9]).c_str());   

    return 0;
}
