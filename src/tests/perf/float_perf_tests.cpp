#include <perf_utils.hpp>
#include <types/float256.hpp>
#include <iostream>

int main(int argc, char **argv) {
    std::vector<float> aV(268'435'456), bV(268'435'456), cV(268'435'456);
    testing::perf::TestConfig<float> config;
    config.doWarmup = true;
    config.warmupDuration = 20;
    config.printWarmupInfo = true;
    int result = testing::perf::allPerfTest<avx::Float256>(aV, bV, cV, config);
    return result;
}