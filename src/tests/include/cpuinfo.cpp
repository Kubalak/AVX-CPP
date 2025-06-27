#include "cpuinfo.hpp"
#include <sstream>
#include <iostream>

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <cpuid.h>
    #include <cstring>
    #include <x86intrin.h>
#endif

#define __BIT_0  0x1
#define __BIT_1  0x2
#define __BIT_2  0x4
#define __BIT_3  0x8
#define __BIT_4  0x10
#define __BIT_5  0x20
#define __BIT_6  0x40
#define __BIT_7  0x80
#define __BIT_8  0x100
#define __BIT_9  0x200
#define __BIT_10 0x400
#define __BIT_11 0x800
#define __BIT_12 0x1000
#define __BIT_13 0x2000
#define __BIT_14 0x4000
#define __BIT_15 0x8000
#define __BIT_16 0x10000
#define __BIT_17 0x20000
#define __BIT_18 0x40000
#define __BIT_19 0x80000
#define __BIT_20 0x100000
#define __BIT_21 0x200000
#define __BIT_22 0x400000
#define __BIT_23 0x800000
#define __BIT_24 0x1000000
#define __BIT_25 0x2000000
#define __BIT_26 0x4000000
#define __BIT_27 0x8000000
#define __BIT_28 0x10000000
#define __BIT_29 0x20000000
#define __BIT_30 0x40000000
#define __BIT_31 0x80000000

namespace cpuinfo {

    struct _regs {
        int eax, ebx, ecx, edx;
    };

    union regs{
        int data[4];
        _regs reg; 
    };

    CPUInfoFeatures CPUInfoFeatures::buildCPUInfo() {

        regs info, extInfo, amdInfo;
        #ifdef _MSC_VER
            __cpuid(info.data, 0x1);
            __cpuidex(extInfo.data, 0x7, 0x0);
            __cpuid(amdInfo.data, 0x80000001);
        #else
            __cpuid(0x1, info.reg.eax, info.reg.ebx, info.reg.ecx, info.reg.edx);
            __cpuid_count(0x7, 0x0, extInfo.reg.eax, extInfo.reg.ebx, extInfo.reg.ecx, extInfo.reg.edx);
            __cpuid(0x80000001, amdInfo.reg.eax, amdInfo.reg.ebx, amdInfo.reg.ecx, amdInfo.reg.edx);
        #endif

        CPUInfoFeatures features {
            (info.reg.edx & __BIT_0) != 0,
            (info.reg.edx & __BIT_1) != 0,
            (info.reg.edx & __BIT_2) != 0,
            (info.reg.edx & __BIT_3) != 0,
            (info.reg.edx & __BIT_4) != 0,
            (info.reg.edx & __BIT_5) != 0,
            (info.reg.edx & __BIT_6) != 0,
            (info.reg.edx & __BIT_7) != 0,
            (info.reg.edx & __BIT_8) != 0,
            (info.reg.edx & __BIT_9) != 0,
            (info.reg.edx & __BIT_11) != 0,
            (info.reg.edx & __BIT_12) != 0,
            (info.reg.edx & __BIT_13) != 0,
            (info.reg.edx & __BIT_14) != 0,
            (info.reg.edx & __BIT_15) != 0,
            (info.reg.edx & __BIT_16) != 0,
            (info.reg.edx & __BIT_17) != 0,
            (info.reg.edx & __BIT_18) != 0,
            (info.reg.edx & __BIT_19) != 0,
            (info.reg.edx & __BIT_21) != 0,
            (info.reg.edx & __BIT_22) != 0,
            (info.reg.edx & __BIT_23) != 0,
            (info.reg.edx & __BIT_24) != 0,
            (info.reg.edx & __BIT_25) != 0,
            (info.reg.edx & __BIT_26) != 0,
            (info.reg.edx & __BIT_27) != 0,
            (info.reg.edx & __BIT_28) != 0,
            (info.reg.edx & __BIT_29) != 0,
            (info.reg.edx & __BIT_31) != 0,
            (info.reg.ecx & __BIT_0) != 0,
            (info.reg.ecx & __BIT_1) != 0,
            (info.reg.ecx & __BIT_2) != 0,
            (info.reg.ecx & __BIT_3) != 0,
            (info.reg.ecx & __BIT_4) != 0,
            (info.reg.ecx & __BIT_5) != 0,
            (info.reg.ecx & __BIT_6) != 0,
            (info.reg.ecx & __BIT_7) != 0,
            (info.reg.ecx & __BIT_8) != 0,
            (info.reg.ecx & __BIT_9) != 0,
            (info.reg.ecx & __BIT_10) != 0,
            (info.reg.ecx & __BIT_11) != 0,
            (info.reg.ecx & __BIT_12) != 0,
            (info.reg.ecx & __BIT_13) != 0,
            (info.reg.ecx & __BIT_14) != 0,
            (info.reg.ecx & __BIT_15) != 0,
            (info.reg.ecx & __BIT_17) != 0,
            (info.reg.ecx & __BIT_18) != 0,
            (info.reg.ecx & __BIT_19) != 0,
            (info.reg.ecx & __BIT_20) != 0,
            (info.reg.ecx & __BIT_21) != 0,
            (info.reg.ecx & __BIT_22) != 0,
            (info.reg.ecx & __BIT_23) != 0,
            (info.reg.ecx & __BIT_24) != 0,
            (info.reg.ecx & __BIT_25) != 0,
            (info.reg.ecx & __BIT_26) != 0,
            (info.reg.ecx & __BIT_27) != 0,
            (info.reg.ecx & __BIT_28) != 0,
            (info.reg.ecx & __BIT_29) != 0,
            (info.reg.ecx & __BIT_30) != 0,
            (info.reg.ecx & __BIT_31) != 0,
            (extInfo.reg.ebx & __BIT_0) != 0,
            (extInfo.reg.ebx & __BIT_1) != 0,
            (extInfo.reg.ebx & __BIT_2) != 0,
            (extInfo.reg.ebx & __BIT_3) != 0,
            (extInfo.reg.ebx & __BIT_4) != 0,
            (extInfo.reg.ebx & __BIT_5) != 0,
            (extInfo.reg.ebx & __BIT_7) != 0,
            (extInfo.reg.ebx & __BIT_8) != 0,
            (extInfo.reg.ebx & __BIT_9) != 0,
            (extInfo.reg.ebx & __BIT_10) != 0,
            (extInfo.reg.ebx & __BIT_11) != 0,
            (extInfo.reg.ebx & __BIT_12) != 0,
            (extInfo.reg.ebx & __BIT_14) != 0,
            (extInfo.reg.ebx & __BIT_16) != 0,
            (extInfo.reg.ebx & __BIT_17) != 0,
            (extInfo.reg.ebx & __BIT_18) != 0,
            (extInfo.reg.ebx & __BIT_19) != 0,
            (extInfo.reg.ebx & __BIT_20) != 0,
            (extInfo.reg.ebx & __BIT_21) != 0,
            (extInfo.reg.ebx & __BIT_23) != 0,
            (extInfo.reg.ebx & __BIT_24) != 0,
            (extInfo.reg.ebx & __BIT_25) != 0,
            (extInfo.reg.ebx & __BIT_26) != 0,
            (extInfo.reg.ebx & __BIT_27) != 0,
            (extInfo.reg.ebx & __BIT_28) != 0,
            (extInfo.reg.ebx & __BIT_29) != 0,
            (extInfo.reg.ebx & __BIT_30) != 0,
            (extInfo.reg.ebx & __BIT_31) != 0,
            (extInfo.reg.ecx & __BIT_0) != 0,
            (extInfo.reg.ecx & __BIT_1) != 0,
            (extInfo.reg.ecx & __BIT_2) != 0,
            (extInfo.reg.ecx & __BIT_3) != 0,
            (extInfo.reg.ecx & __BIT_4) != 0,
            (extInfo.reg.ecx & __BIT_5) != 0,
            (extInfo.reg.ecx & __BIT_6) != 0,
            (extInfo.reg.ecx & __BIT_7) != 0,
            (extInfo.reg.ecx & __BIT_8) != 0,
            (extInfo.reg.ecx & __BIT_9) != 0,
            (extInfo.reg.ecx & __BIT_10) != 0,
            (extInfo.reg.ecx & __BIT_11) != 0,
            (extInfo.reg.ecx & __BIT_12) != 0,
            (extInfo.reg.ecx & __BIT_13) != 0,
            (extInfo.reg.ecx & __BIT_14) != 0,
            (extInfo.reg.edx & __BIT_16) != 0,
            (extInfo.reg.ecx & __BIT_22) != 0,
            (extInfo.reg.edx & __BIT_25) != 0,
            (extInfo.reg.ecx & __BIT_27) != 0,
            (extInfo.reg.ecx & __BIT_28) != 0,
            (extInfo.reg.ecx & __BIT_29) != 0,
            (extInfo.reg.ecx & __BIT_30) != 0,
            (amdInfo.reg.ecx & __BIT_0) != 0,
            (amdInfo.reg.ecx & __BIT_1) != 0,
            (amdInfo.reg.ecx & __BIT_2) != 0,
            (amdInfo.reg.ecx & __BIT_3) != 0,
            (amdInfo.reg.ecx & __BIT_4) != 0,
            (amdInfo.reg.ecx & __BIT_5) != 0,
            (amdInfo.reg.ecx & __BIT_6) != 0,
            (amdInfo.reg.ecx & __BIT_7) != 0,
            (amdInfo.reg.ecx & __BIT_8) != 0,
            (amdInfo.reg.ecx & __BIT_9) != 0,
            (amdInfo.reg.ecx & __BIT_10) != 0,
            (amdInfo.reg.ecx & __BIT_11) != 0,
            (amdInfo.reg.ecx & __BIT_12) != 0,
            (amdInfo.reg.ecx & __BIT_13) != 0,
            (amdInfo.reg.ecx & __BIT_15) != 0,
            (amdInfo.reg.ecx & __BIT_16) != 0,
            (amdInfo.reg.ecx & __BIT_17) != 0,
            (amdInfo.reg.ecx & __BIT_19) != 0,
            (amdInfo.reg.ecx & __BIT_21) != 0,
            (amdInfo.reg.ecx & __BIT_22) != 0,
            (amdInfo.reg.ecx & __BIT_23) != 0,
            (amdInfo.reg.ecx & __BIT_24) != 0,
            (amdInfo.reg.ecx & __BIT_26) != 0,
            (amdInfo.reg.ecx & __BIT_27) != 0,
            (amdInfo.reg.ecx & __BIT_28) != 0,
            (amdInfo.reg.ecx & __BIT_29) != 0,
            (amdInfo.reg.edx & __BIT_11) != 0,
            (amdInfo.reg.edx & __BIT_20) != 0,
            (amdInfo.reg.edx & __BIT_22) != 0,
            (amdInfo.reg.edx & __BIT_25) != 0,
            (amdInfo.reg.edx & __BIT_26) != 0,
            (amdInfo.reg.edx & __BIT_27) != 0,
            (amdInfo.reg.edx & __BIT_29) != 0,
            (amdInfo.reg.edx & __BIT_30) != 0,
            (amdInfo.reg.edx & __BIT_31) != 0
        };

        return features;
    }

    CPUDetails::CPUDetails() {
        
        regs info, extInfo, amdInfo;

        #ifdef _MSC_VER
            __cpuid(info.data, 0x1);
            __cpuidex(extInfo.data, 0x7, 0x0);
            __cpuid(amdInfo.data, 0x80000001);
        #else
            __cpuid(0x1, info.reg.eax, info.reg.ebx, info.reg.ecx, info.reg.edx);
            __cpuid_count(0x7, 0x0, extInfo.reg.eax, extInfo.reg.ebx, extInfo.reg.ecx, extInfo.reg.edx);
            __cpuid(0x80000001, amdInfo.reg.eax, amdInfo.reg.ebx, amdInfo.reg.ecx, amdInfo.reg.edx);
        #endif

        mExtensions["FPU"] = info.reg.edx & __BIT_0;
        mExtensions["VME"] = info.reg.edx & __BIT_1;
        mExtensions["DE"] = info.reg.edx & __BIT_2;
        mExtensions["PSE"] = info.reg.edx & __BIT_3;
        mExtensions["TSC"] = info.reg.edx & __BIT_4;
        mExtensions["MSR"] = info.reg.edx & __BIT_5;
        mExtensions["PAE"] = info.reg.edx & __BIT_6;
        mExtensions["MCE"] = info.reg.edx & __BIT_7;
        mExtensions["CX8"] = info.reg.edx & __BIT_8;
        mExtensions["APIC"] = info.reg.edx & __BIT_9;
        mExtensions["SEP"] = info.reg.edx & __BIT_11;
        mExtensions["MTRR"] = info.reg.edx & __BIT_12;
        mExtensions["PGE"] = info.reg.edx & __BIT_13;
        mExtensions["MCA"] = info.reg.edx & __BIT_14;
        mExtensions["CMOV"] = info.reg.edx & __BIT_15;
        mExtensions["PAT"] = info.reg.edx & __BIT_16;
        mExtensions["PSE-36"] = info.reg.edx & __BIT_17;
        mExtensions["PSN"] = info.reg.edx & __BIT_18;
        mExtensions["CLFSH"] = info.reg.edx & __BIT_19;
        mExtensions["DS"] = info.reg.edx & __BIT_21;
        mExtensions["ACPI"] = info.reg.edx & __BIT_22;
        mExtensions["MMX"] = info.reg.edx & __BIT_23;
        mExtensions["FXSR"] = info.reg.edx & __BIT_24;
        mExtensions["SSE"] = info.reg.edx & __BIT_25;
        mExtensions["SSE2"] = info.reg.edx & __BIT_26;
        mExtensions["SS"] = info.reg.edx & __BIT_27;
        mExtensions["HTT"] = info.reg.edx & __BIT_28;
        mExtensions["TM"] = info.reg.edx & __BIT_29;
        mExtensions["PBE"] = info.reg.edx & __BIT_31;
        mExtensions["SSE3"] = info.reg.ecx & __BIT_0;
        mExtensions["PCLMULQDQ"] = info.reg.ecx & __BIT_1;
        mExtensions["DTES64"] = info.reg.ecx & __BIT_2;
        mExtensions["MONITOR"] = info.reg.ecx & __BIT_3;
        mExtensions["DS-CPL"] = info.reg.ecx & __BIT_4;
        mExtensions["VMX"] = info.reg.ecx & __BIT_5;
        mExtensions["SMX"] = info.reg.ecx & __BIT_6;
        mExtensions["EIST"] = info.reg.ecx & __BIT_7;
        mExtensions["TM2"] = info.reg.ecx & __BIT_8;
        mExtensions["SSSE3"] = info.reg.ecx & __BIT_9;
        mExtensions["CNXT-ID"] = info.reg.ecx & __BIT_10;
        mExtensions["SDBG"] = info.reg.ecx & __BIT_11;
        mExtensions["FMA"] = info.reg.ecx & __BIT_12;
        mExtensions["CX16"] = info.reg.ecx & __BIT_13;
        mExtensions["xTPR Update Control"] = info.reg.ecx & __BIT_14;
        mExtensions["PDCM"] = info.reg.ecx & __BIT_15;
        mExtensions["PCID"] = info.reg.ecx & __BIT_17;
        mExtensions["DCA"] = info.reg.ecx & __BIT_18;
        mExtensions["SSE4.1"] = info.reg.ecx & __BIT_19;
        mExtensions["SSE4.2"] = info.reg.ecx & __BIT_20;
        mExtensions["x2APIC"] = info.reg.ecx & __BIT_21;
        mExtensions["MOVBE"] = info.reg.ecx & __BIT_22;
        mExtensions["POPCNT"] = info.reg.ecx & __BIT_23;
        mExtensions["TSC-Deadline"] = info.reg.ecx & __BIT_24;
        mExtensions["AESNI"] = info.reg.ecx & __BIT_25;
        mExtensions["XSAVE"] = info.reg.ecx & __BIT_26;
        mExtensions["OSXSAVE"] = info.reg.ecx & __BIT_27;
        mExtensions["AVX"] = info.reg.ecx & __BIT_28;
        mExtensions["F16C"] = info.reg.ecx & __BIT_29;
        mExtensions["RDRAND"] = info.reg.ecx & __BIT_30;
        mExtensions["HYPERVISOR"] = info.reg.ecx & __BIT_31;

        mExtensions["FSGSBASE"] = extInfo.reg.ebx & __BIT_0;
        mExtensions["IA32_TSC_ADJUST"] = extInfo.reg.ebx & __BIT_1;
        mExtensions["SGX"] = extInfo.reg.ebx & __BIT_2;
        mExtensions["BMI1"] = extInfo.reg.ebx & __BIT_3;
        mExtensions["HLE"] = extInfo.reg.ebx & __BIT_4;
        mExtensions["AVX2"] = extInfo.reg.ebx & __BIT_5;
        mExtensions["SMEP"] = extInfo.reg.ebx & __BIT_7;
        mExtensions["BMI2"] = extInfo.reg.ebx & __BIT_8;
        mExtensions["ERMS"] = extInfo.reg.ebx & __BIT_9;
        mExtensions["INVPCID"] = extInfo.reg.ebx & __BIT_10;
        mExtensions["RTM"] = extInfo.reg.ebx & __BIT_11;
        mExtensions["PQE"] = extInfo.reg.ebx & __BIT_12;
        mExtensions["MPX"] = extInfo.reg.ebx & __BIT_14;
        mExtensions["AVX512F"] = extInfo.reg.ebx & __BIT_16;
        mExtensions["AVX512DQ"] = extInfo.reg.ebx & __BIT_17;
        mExtensions["RDSEED"] = extInfo.reg.ebx & __BIT_18;
        mExtensions["ADX"] = extInfo.reg.ebx & __BIT_19;
        mExtensions["SMAP"] = extInfo.reg.ebx & __BIT_20;
        mExtensions["AVX512IFMA"] = extInfo.reg.ebx & __BIT_21;
        mExtensions["CLFLUSHOPT"] = extInfo.reg.ebx & __BIT_23;
        mExtensions["CLWB"] = extInfo.reg.ebx & __BIT_24;
        mExtensions["INTEL_PT"] = extInfo.reg.ebx & __BIT_25;
        mExtensions["AVX512PF"] = extInfo.reg.ebx & __BIT_26;
        mExtensions["AVX512ER"] = extInfo.reg.ebx & __BIT_27;
        mExtensions["AVX512CD"] = extInfo.reg.ebx & __BIT_28;
        mExtensions["SHA"] = extInfo.reg.ebx & __BIT_29;
        mExtensions["AVX512BW"] = extInfo.reg.ebx & __BIT_30;
        mExtensions["AVX512VL"] = extInfo.reg.ebx & __BIT_31;

        mExtensions["PREFETCHWT1"] = extInfo.reg.ecx & __BIT_0;
        mExtensions["AVX512_VBMI"] = extInfo.reg.ecx & __BIT_1;
        mExtensions["UMIP"] = extInfo.reg.ecx & __BIT_2;
        mExtensions["PKU"] = extInfo.reg.ecx & __BIT_3;
        mExtensions["OSPKE"] = extInfo.reg.ecx & __BIT_4;
        mExtensions["WAITPKG"] = extInfo.reg.ecx & __BIT_5;
        mExtensions["AVX512_VBMI2"] = extInfo.reg.ecx & __BIT_6;
        mExtensions["CET_SS"] = extInfo.reg.ecx & __BIT_7;
        mExtensions["GFNI"] = extInfo.reg.ecx & __BIT_8;
        mExtensions["VAES"] = extInfo.reg.ecx & __BIT_9;
        mExtensions["VPCLMULQDQ"] = extInfo.reg.ecx & __BIT_10;
        mExtensions["AVX512_VNNI"] = extInfo.reg.ecx & __BIT_11;
        mExtensions["AVX512_BITALG"] = extInfo.reg.ecx & __BIT_12;
        mExtensions["TME_EN"] = extInfo.reg.ecx & __BIT_13;
        mExtensions["AVX512_VPOPCNTDQ"] = extInfo.reg.ecx & __BIT_14;
        mExtensions["LA57"] = extInfo.reg.edx & __BIT_16;
        mExtensions["RDPID"] = extInfo.reg.ecx & __BIT_22;
        mExtensions["CLDEMOTE"] = extInfo.reg.edx & __BIT_25;
        mExtensions["MOVDIRI"] = extInfo.reg.ecx & __BIT_27;
        mExtensions["MOVDIR64B"] = extInfo.reg.ecx & __BIT_28;
        mExtensions["ENQCMD"] = extInfo.reg.ecx & __BIT_29;
        mExtensions["SGX_LC"] = extInfo.reg.ecx & __BIT_30;

        mExtensions["LAHF_LM"] = amdInfo.reg.ecx & __BIT_0;
        mExtensions["CMP_LEGACY"] = amdInfo.reg.ecx & __BIT_1;
        mExtensions["SVM"] = amdInfo.reg.ecx & __BIT_2;
        mExtensions["EXTAPIC"] = amdInfo.reg.ecx & __BIT_3;
        mExtensions["CR8_LEGACY"] = amdInfo.reg.ecx & __BIT_4;
        mExtensions["ABM"] = amdInfo.reg.ecx & __BIT_5;
        mExtensions["SSE4A"] = amdInfo.reg.ecx & __BIT_6;
        mExtensions["MISALIGNSSE"] = amdInfo.reg.ecx & __BIT_7;
        mExtensions["PREFETCHW"] = amdInfo.reg.ecx & __BIT_8;
        mExtensions["OSVW"] = amdInfo.reg.ecx & __BIT_9;
        mExtensions["IBS"] = amdInfo.reg.ecx & __BIT_10;
        mExtensions["XOP"] = amdInfo.reg.ecx & __BIT_11;
        mExtensions["SKINIT"] = amdInfo.reg.ecx & __BIT_12;
        mExtensions["WDT"] = amdInfo.reg.ecx & __BIT_13;
        mExtensions["LWP"] = amdInfo.reg.ecx & __BIT_15;
        mExtensions["FMA4"] = amdInfo.reg.ecx & __BIT_16;
        mExtensions["TCE"] = amdInfo.reg.ecx & __BIT_17;
        mExtensions["NODEID_MSR"] = amdInfo.reg.ecx & __BIT_19;
        mExtensions["TBM"] = amdInfo.reg.ecx & __BIT_21;
        mExtensions["TOPOEXT"] = amdInfo.reg.ecx & __BIT_22;
        mExtensions["PERFCTR_CORE"] = amdInfo.reg.ecx & __BIT_23;
        mExtensions["PERFCTR_NB"] = amdInfo.reg.ecx & __BIT_24;
        mExtensions["BPEXT"] = amdInfo.reg.ecx & __BIT_26;
        mExtensions["PTSC"] = amdInfo.reg.ecx & __BIT_27;
        mExtensions["PERFCTR_L2"] = amdInfo.reg.ecx & __BIT_28;
        mExtensions["MONITORX"] = amdInfo.reg.ecx & __BIT_29;
        mExtensions["SYSCALL"] = amdInfo.reg.edx & __BIT_11;
        mExtensions["NX"] = amdInfo.reg.edx & __BIT_20;
        mExtensions["MMXEXT"] = amdInfo.reg.edx & __BIT_22;
        mExtensions["FXSR_OPT"] = amdInfo.reg.edx & __BIT_25;
        mExtensions["PDPE1GB"] = amdInfo.reg.edx & __BIT_26;
        mExtensions["RDTSCP"] = amdInfo.reg.edx & __BIT_27;
        mExtensions["LM"] = amdInfo.reg.edx & __BIT_29;
        mExtensions["3DNOWEXT"] = amdInfo.reg.edx & __BIT_30;
        mExtensions["3DNOW"] = amdInfo.reg.edx & __BIT_31;

        mCpuName = cpuinfo::getCPUName();
        mManufactID = cpuinfo::getManufactID();
    }

    bool CPUDetails::supportsFeature(std::string sFeatureName) {
        if(mExtensions.find(sFeatureName) != mExtensions.end()) return mExtensions[sFeatureName];

        return false;
    }

    std::string CPUDetails::toJSON() {
        std::stringstream results;
        results << "{\n  \"name\": \"" <<  mCpuName << "\",\n";
        results << "  \"manufactId\":\"" << mManufactID << "\",\n";
        results << "  \"features\":{\n";

        auto pre_last = mExtensions.end();
        pre_last--;
        auto it = mExtensions.begin();
        for(; it != pre_last; it++)
            results << "    \"" + it->first + "\": " << it->second << ",\n";
        
        results << "    \"" + it->first + "\": " << it->second << "\n  }\n}";

        return results.str();
    }

    std::string getCPUName() {
        unsigned int regs[12];
        char nameStr[sizeof(regs)+1];
        #ifdef _MSC_VER
            __cpuid((int*)regs, 0x80000000);
        #else
            __cpuid( 0x80000000, regs[0], regs[1], regs[2], regs[3]);
        #endif

        if (regs[0] < 0x80000004)
            return "N/A";

        #ifdef _MSC_VER
            __cpuid((int*)regs, 0x80000002);
            __cpuid((int*)regs + 4, 0x80000003);
            __cpuid((int*)regs + 8, 0x80000004);
            memcpy_s(nameStr, sizeof(nameStr), regs, sizeof(regs));
        #else
            __cpuid(0x80000002, regs[0], regs[1], regs[2], regs[3]);
            __cpuid(0x80000003, regs[4], regs[5], regs[6], regs[7]);
            __cpuid(0x80000004, regs[8], regs[9], regs[10], regs[11]);
            memcpy(nameStr, regs, sizeof(regs));
        #endif

        
        nameStr[sizeof(regs)] = 0;
        return nameStr;
    }

    std::string getManufactID(int* eaxVal) {
        regs info;
        char manID [13];

        #ifdef _MSC_VER
            __cpuid(info.data, 0);
        #else
            __cpuid(0x0, info.reg.eax, info.reg.ebx, info.reg.ecx, info.reg.edx);
        #endif
        
        if(eaxVal != nullptr)
            *eaxVal = info.data[0];        

        ((int*)manID)[0] = info.reg.ebx; // Easier to treat it as int lol.
        ((int*)manID)[1] = info.reg.edx;
        ((int*)manID)[2] = info.reg.ecx;

        
        manID[12] = 0;
        return manID;
    }
};