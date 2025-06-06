#include <iostream>
#include "perf_utils.hpp"
#include <types/uchar256.hpp>

int main(int argc, char *argv[]) {
    std::vector<unsigned char> aV(1'073'741'824), bV(1'073'741'824), cV(1'073'741'824);
    testing::perf::TestConfig<unsigned char> config;
    return testing::perf::allPerfTest<avx::UChar256>(aV, bV, cV, config);
}