#include <iostream>
#include "perf_utils.hpp"
#include <types/ulong256.hpp>


int main(int argc, char* argv[]) {
    std::vector<unsigned long long> aV(134'217'728), bV(134'217'728), cV(134'217'728);
    testing::perf::TestConfig<unsigned long long> config;
    config.printVerificationFailed = true;
    return testing::perf::allPerfTest<avx::ULong256>(aV, bV, cV, config);
}