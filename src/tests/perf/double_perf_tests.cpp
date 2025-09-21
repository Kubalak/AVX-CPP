#include <perf_utils.hpp>
#include <types/double256.hpp>
#include <iostream>

int main(int argc, char **argv) {
    std::vector<double> aV(134'217'728), bV(134'217'728), cV(134'217'728);
    testing::perf::TestConfig<double> config;
    config.doWarmup = true;
    config.warmupDuration = 20;
    config.printWarmupInfo = true;
    int result = testing::perf::allPerfTest<avx::Double256>(aV, bV, cV, config);
    return result;
}