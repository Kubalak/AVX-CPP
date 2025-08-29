#include <iostream>
#include "perf_utils.hpp"
#include <types/ulong256.hpp>


int main(int argc, char* argv[]) {
    std::vector<unsigned long long> aV(134'217'728), bV(134'217'728), cV(134'217'728);
    testing::perf::TestConfig<unsigned long long> config;
    config.printVerificationFailed = true;
    int result = testing::perf::allPerfTest<avx::ULong256>(aV, bV, cV, config);
    // std::cout << aV[16] << ' ' << bV[16] << ' ' << ((aV[16] >> bV[16]) >> bV[bV.size() / 2]) << '\n';
    // avx::ULong256 a(aV.data() + 16), b(bV.data() + 16);
    // std::cout << (a >> b).str() << '\n';
    return (result & (_AVX_IGNORE_LSH & _AVX_IGNORE_RSH)) != 0;
}