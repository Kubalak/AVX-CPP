#pragma once
#ifndef CPUINFO_HPP
#define CPUINFO_HPP
#include <string>
#include <bitset>
#include <map>

namespace cpuinfo {
    struct _regs;
    union regs;

    struct CPUInfoFeatures {
        const bool has_FPU;
        const bool has_VME;
        const bool has_DE;
        const bool has_PSE;
        const bool has_TSC;
        const bool has_MSR;
        const bool has_PAE;
        const bool has_MCE;
        const bool has_CX8;
        const bool has_APIC;
        const bool has_SEP;
        const bool has_MTRR;
        const bool has_PGE;
        const bool has_MCA;
        const bool has_CMOV;
        const bool has_PAT;
        const bool has_PSE_36;
        const bool has_PSN;
        const bool has_CLFSH;
        const bool has_DS;
        const bool has_ACPI;
        const bool has_MMX;
        const bool has_FXSR;
        const bool has_SSE;
        const bool has_SSE2;
        const bool has_SS;
        const bool has_HTT;
        const bool has_TM;
        const bool has_PBE;
        const bool has_SSE3;
        const bool has_PCLMULQDQ;
        const bool has_DTES64;
        const bool has_MONITOR;
        const bool has_DS_CPL;
        const bool has_VMX;
        const bool has_SMX;
        const bool has_EIST;
        const bool has_TM2;
        const bool has_SSSE3;
        const bool has_CNXT_ID;
        const bool has_SDBG;
        const bool has_FMA;
        const bool has_CX16;
        const bool has_xTPR_UPDT_CTRL;
        const bool has_PDCM;
        const bool has_PCID;
        const bool has_DCA;
        const bool has_SSE4_1;
        const bool has_SSE4_2;
        const bool has_x2APIC;
        const bool has_MOVBE;
        const bool has_POPCNT;
        const bool has_TSC_Deadline;
        const bool has_AESNI;
        const bool has_XSAVE;
        const bool has_OSXSAVE;
        const bool has_AVX;
        const bool has_F16C;
        const bool has_RDRAND;
        const bool has_HYPERVISOR;
        const bool has_FSGSBASE;
        const bool has_IA32_TSC_ADJUST;
        const bool has_SGX;
        const bool has_BMI1;
        const bool has_HLE;
        const bool has_AVX2;
        const bool has_SMEP;
        const bool has_BMI2;
        const bool has_ERMS;
        const bool has_INVPCID;
        const bool has_RTM;
        const bool has_PQE;
        const bool has_MPX;
        const bool has_AVX512F;
        const bool has_AVX512DQ;
        const bool has_RDSEED;
        const bool has_ADX;
        const bool has_SMAP;
        const bool has_AVX512IFMA;
        const bool has_CLFLUSHOPT;
        const bool has_CLWB;
        const bool has_INTEL_PT;
        const bool has_AVX512PF;
        const bool has_AVX512ER;
        const bool has_AVX512CD;
        const bool has_SHA;
        const bool has_AVX512BW;
        const bool has_AVX512VL;
        const bool has_PREFETCHWT1;
        const bool has_AVX512_VBMI;
        const bool has_UMIP;
        const bool has_PKU;
        const bool has_OSPKE;
        const bool has_WAITPKG;
        const bool has_AVX512_VBMI2;
        const bool has_CET_SS;
        const bool has_GFNI;
        const bool has_VAES;
        const bool has_VPCLMULQDQ;
        const bool has_AVX512_VNNI;
        const bool has_AVX512_BITALG;
        const bool has_TME_EN;
        const bool has_AVX512_VPOPCNTDQ;
        const bool has_LA57;
        const bool has_RDPID;
        const bool has_CLDEMOTE;
        const bool has_MOVDIRI;
        const bool has_MOVDIR64B;
        const bool has_ENQCMD;
        const bool has_SGX_LC;
        const bool has_LAHF_LM;
        const bool has_CMP_LEGACY;
        const bool has_SVM;
        const bool has_EXTAPIC;
        const bool has_CR8_LEGACY;
        const bool has_ABM;
        const bool has_SSE4A;
        const bool has_MISALIGNSSE;
        const bool has_PREFETCHW;
        const bool has_OSVW;
        const bool has_IBS;
        const bool has_XOP;
        const bool has_SKINIT;
        const bool has_WDT;
        const bool has_LWP;
        const bool has_FMA4;
        const bool has_TCE;
        const bool has_NODEID_MSR;
        const bool has_TBM;
        const bool has_TOPOEXT;
        const bool has_PERFCTR_CORE;
        const bool has_PERFCTR_NB;
        const bool has_BPEXT;
        const bool has_PTSC;
        const bool has_PERFCTR_L2;
        const bool has_MONITORX;
        const bool has_SYSCALL;
        const bool has_NX;
        const bool has_MMXEXT;
        const bool has_FXSR_OPT;
        const bool has_PDPE1GB;
        const bool has_RDTSCP;
        const bool has_LM;
        const bool has_3DNOWEXT;
        const bool has_3DNOW;

        static CPUInfoFeatures buildCPUInfo();
    };

    class CPUDetails {
        std::map<std::string, bool> mExtensions;
        std::string mCpuName, mManufactID;

        public:
        CPUDetails();
        bool supportsFeature(std::string sFeatureName);
        std::string toJSON();
        std::string getCPUName() { return mCpuName; }
        std::string getManufactID() { return mManufactID; }
    };

    extern std::string getCPUName();
    extern std::string getManufactID(int* eaxVal = nullptr);
    //extern void experimental();
};

#endif