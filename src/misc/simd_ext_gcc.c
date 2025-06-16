#include "simd_ext_gcc.h"
#ifndef _MSC_VER

/**
 * Reimplements missing division for two vectors containing `int` type.
 * Please note that this function is only intended to be used with GCC or Clang compilers.
 * MSVC provides implementation of this function by default.
 * 
 * @param a Vector with 8 `int` values to be divided.
 * @param b Vector containing 8 divisors.
 * @return Division results.
 */
__m256i _mm256_div_epi32(__m256i a, __m256i b) {

    __asm__ (
        "vmovdqa %[a], %%ymm0\n\t"
        "vmovdqa %[b], %%ymm6\n\t"
        "vcvtdq2ps %%ymm6, %%ymm2\n\t"
        "vrcpps %%ymm2, %%ymm7\n\t"
        "vmovupd %[data_0], %%ymm2\n\t"
        "vextractf128 $1, %%ymm7, %%xmm3\n\t"
        "vcvtps2pd %%xmm3, %%ymm1\n\t"
        "vmovdqa %%ymm2, %%ymm3\n\t"
        "vextracti128 $1, %%ymm6, %%xmm5\n\t"
        "vcvtdq2pd %%xmm6, %%ymm4\n\t"
        "vcvtdq2pd %%xmm5, %%ymm5\n\t"
        "vfnmadd231pd %%ymm1, %%ymm5, %%ymm2\n\t"
        "vmulpd %%ymm2, %%ymm1, %%ymm2\n\t"
        "vcvtps2pd %%xmm7, %%ymm7\n\t"
        "vfnmadd231pd %%ymm7, %%ymm4, %%ymm3\n\t"
        "vmulpd %%ymm3, %%ymm7, %%ymm3\n\t"
        "vcvtdq2pd %%xmm0, %%ymm1\n\t"
        "vextracti128 $1, %%ymm0, %%xmm7\n\t"
        "vmovupd %[data_1], %%ymm0\n\t"
        "vmulpd %%ymm3, %%ymm1, %%ymm1\n\t"
        "vcvtdq2pd %%xmm7, %%ymm7\n\t"
        "vfnmadd213pd %%ymm0, %%ymm3, %%ymm4\n\t"
        "vfnmadd213pd %%ymm0, %%ymm2, %%ymm5\n\t"
        "vmulpd %%ymm2, %%ymm7, %%ymm2\n\t"
        "vmulpd %%ymm1, %%ymm4, %%ymm0\n\t"
        "vmulpd %%ymm2, %%ymm5, %%ymm1\n\t"
        // Check for errors
        "vpxor %%ymm3, %%ymm3, %%ymm3\n\t"
        "vpcmpeqd %%ymm3, %%ymm6, %%ymm4\n\t"
        "vpmovmskb %%ymm4, %%eax\n\t"
        "test %%eax, %%eax\n\t"
        "je continue\n\t"
        // If error then crash
            "mov $0, %%eax\n\t"
            "div %%al\n\t"

        // If not then continue
        "continue:\n\t"
        "vcvttpd2dq %%ymm0, %%xmm0\n\t"
        "vcvttpd2dq %%ymm1, %%xmm1\n\t"
        "vinserti128 $1, %%xmm1, %%ymm0, %%ymm0\n\t"
        "vmovdqa %%ymm0, %[a]\n\t"

        : [a] "+x" (a), 
          [b] "+x" (b)
        : [data_0] "x" (__DATA_0),
          [data_1] "x" (__DATA_1)
        : "al",
          "eax",
          "ymm0",
          "ymm1",
          "ymm2", 
          "ymm3",
          "ymm4",           
          "ymm5",
          "ymm6",
          "ymm7",
          "xmm0",
          "xmm1",
          "xmm3",
          "xmm5",
          "xmm6",
          "xmm7"
    );

    return a;
}

#endif // _MSC_VER