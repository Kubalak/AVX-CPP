#include <iostream>
#include "perf_utils.hpp"
#include <types/ushort256.hpp>

int main(int argc, char *argv[]) {
    std::vector<unsigned short> aV(536'870'912), bV(536'870'912), cV(536'870'912);
    testing::perf::TestConfig<unsigned short> config;
    return testing::perf::allPerfTest<avx::UShort256>(aV, bV, cV, config);
}