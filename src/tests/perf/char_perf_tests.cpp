#include <iostream>
#include "perf_utils.hpp"
#include <types/char256.hpp>

int main(int argc, char *argv[]) {
    std::vector<char> aV(1'073'741'824), bV(1'073'741'824), cV(1'073'741'824);
    testing::perf::TestConfig config;
    return testing::perf::allPerfTest<avx::Char256>(aV, bV, cV, config);
}