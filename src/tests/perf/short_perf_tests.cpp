#include <iostream>
#include "perf_utils.hpp"
#include <types/short256.hpp>

int main(int argc, char *argv[]) {
    std::vector<short> aV(536'870'912), bV(536'870'912), cV(536'870'912);
    testing::perf::TestConfig config;
    return testing::perf::allPerfTest<avx::Short256>(aV, bV, cV, config);
}