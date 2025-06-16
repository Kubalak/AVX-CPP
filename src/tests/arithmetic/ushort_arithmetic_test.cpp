#include <iostream>
#include <types/ushort256.hpp>
#include <test_utils.hpp>
#include <cpuinfo.hpp>

int main(int argc, char* argv[]) {

    int result = 0;

    std::cout << "CPU Model: " << cpuinfo::getCPUName() << '\n';

    std::cout << "Manufacturer ID: " << cpuinfo::getManufactID() << '\n';

    //cpuinfo::experimental();
    cpuinfo::CPUInfoFeatures features = cpuinfo::CPUInfoFeatures::buildCPUInfo();

    cpuinfo::CPUDetails details;

    std::cout << details.toJSON() << '\n';

    result |= testing::universalTestAdd<avx::UShort256>();
    result |= testing::universalTestSub<avx::UShort256>();
    result |= testing::universalTestMul<avx::UShort256>();
    result |= testing::universalTestDiv<avx::UShort256>();
    result |= testing::universalTestMod<avx::UShort256>();
    result |= testing::universalTestAND<avx::UShort256>();
    result |= testing::universalTestOR<avx::UShort256>();
    result |= testing::universalTestXOR<avx::UShort256>();
    result |= testing::universalTestNOT<avx::UShort256>();
    result |= testing::universalTestLshift<avx::UShort256>();
    result |= testing::universalTestRshift<avx::UShort256>();
    result |= testing::universalTestIndexing<avx::UShort256>();
    result |= testing::universalTestCompare<avx::UShort256>();

    return result;
}