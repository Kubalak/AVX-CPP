#include <iostream>
#include "perf_utils.hpp"
#include <types/long256.hpp>


int main(int argc, char* argv[]) {
    std::vector<long long> aV(134'217'728), bV(134'217'728), cV(134'217'728);
    testing::perf::TestConfig<long long> config;
    return testing::perf::allPerfTest<avx::Long256>(aV, bV, cV, config) & (_AVX_IGNORE_LSH & _AVX_IGNORE_RSH);
}