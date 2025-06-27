#include "simd_ext_gcc.h"
#ifndef _MSC_VER


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
        "je div_epi_cont\n\t"
        // If error then crash
            "mov $0, %%eax\n\t"
            "div %%al\n\t"

        // If not then continue
        "div_epi_cont:\n\t"
        "vcvttpd2dq %%ymm0, %%xmm0\n\t"
        "vcvttpd2dq %%ymm1, %%xmm1\n\t"
        "vinserti128 $1, %%xmm1, %%ymm0, %%ymm0\n\t"
        "vmovdqa %%ymm0, %[a]\n\t"

        : [a] "+x" (a), 
          [b] "+x" (b)
        : [data_0] "x" (__DATA_0),
          [data_1] "x" (__DATA_1)
        : "al", "eax", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",  "ymm5", "ymm6", "ymm7", "xmm0", "xmm1", "xmm3", "xmm5", "xmm6", "xmm7"
    );

    return a;
}

__m256i _mm256_div_epu32(__m256i a, __m256i b) {

    __m256i tmp_1 = _mm256_setzero_si256();
    __m256d tmp_2 = _mm256_setzero_pd();

    __asm__ (
        "vmovdqu      %[a], %%ymm0\n\t"
        "vmovdqu      %[b], %%ymm1\n\t"
        "vmovdqu      %[data2], %%ymm4\n\t"
        "vmovups      %[data3], %%ymm2\n\t"
        "vmovdqu      %%ymm1, %%ymm6\n\t"
        "vpand        %%ymm4, %%ymm6, %%ymm5\n\t"
        "vmovdqu      %%ymm6, %[tmp_1]\n\t"
        "vorps        %%ymm2, %%ymm5, %%ymm7\n\t"
        "vsubpd       %%ymm2, %%ymm7, %%ymm1\n\t"
        "vpsrlq       $0x20, %%ymm6, %%ymm5\n\t"
        "vcvtpd2ps    %%ymm1, %%xmm7\n\t"
        "vmovupd      %%ymm1, %[tmp_2]\n\t"
        "vmovdqa      %%ymm0, %%ymm3\n\t"
        "vorps        %%ymm2, %%ymm5, %%ymm0\n\t"
        "vsubpd       %%ymm2, %%ymm0, %%ymm5\n\t"
        "vrcpps       %%xmm7, %%xmm0\n\t"
        "vcvtpd2ps    %%ymm5, %%xmm7\n\t"
        "vcvtps2pd    %%xmm0, %%ymm6\n\t"
        "vpand        %%ymm4, %%ymm3, %%ymm4\n\t"
        "vpsrlq       $0x20, %%ymm3, %%ymm3\n\t"
        "vrcpps       %%xmm7, %%xmm0\n\t"
        "vmovupd      %[data0], %%ymm7\n\t"
        "vcvtps2pd    %%xmm0, %%ymm0\n\t"
        "vfnmadd231pd %%ymm6, %%ymm1, %%ymm7\n\t"
        "vmovupd      %[data0], %%ymm1\n\t"
        "vmulpd       %%ymm7, %%ymm6, %%ymm6\n\t"
        "vfnmadd231pd %%ymm0, %%ymm5, %%ymm1\n\t"
        "vmulpd       %%ymm1, %%ymm0, %%ymm0\n\t"
        "vorps        %%ymm2, %%ymm4, %%ymm1\n\t"
        "vsubpd       %%ymm2, %%ymm1, %%ymm4\n\t"
        "vorps        %%ymm2, %%ymm3, %%ymm1\n\t"
        "vmovupd      %[data1], %%ymm3\n\t"
        "vmulpd       %%ymm4, %%ymm6, %%ymm4\n\t"
        "vsubpd       %%ymm2, %%ymm1, %%ymm7\n\t"
        "vmovupd      %[tmp_2], %%ymm2\n\t"
        "vfnmadd213pd %%ymm3, %%ymm0, %%ymm5\n\t"
        "vmulpd       %%ymm7, %%ymm0, %%ymm0\n\t"
        "vfnmadd213pd %%ymm3, %%ymm6, %%ymm2\n\t"
        "vmulpd       %%ymm0, %%ymm5, %%ymm0\n\t"
        "vmulpd       %%ymm4, %%ymm2, %%ymm1\n\t"
        "vmovdqu      %[tmp_1], %%ymm2\n\t"
        "vpxor        %%ymm7, %%ymm7, %%ymm7\n\t"
        "vpcmpeqd     %%ymm7, %%ymm2, %%ymm5\n\t"
        "vpmovmskb    %%ymm5, %%eax\n\t"
        "test         %%eax, %%eax\n\t"
        "je           div_epu_cont\n\t"

        "mov     $0, %%eax\n\t"
        "div     %%al\n\t"

        "div_epu_cont:\n\t"
        "vmovupd     %[data3], %%ymm2\n\t"
        "vroundpd    $3, %%ymm1, %%ymm1\n\t"
        "vroundpd    $3, %%ymm0, %%ymm0\n\t"
        "vaddpd      %%ymm2, %%ymm1, %%ymm4\n\t"
        "vaddpd      %%ymm2, %%ymm0, %%ymm3\n\t"
        "vpand       %[data2], %%ymm4, %%ymm6\n\t"
        "vpsllq      $0x20, %%ymm3, %%ymm5\n\t"
        "vpor        %%ymm6, %%ymm5, %[a]\n\t"

        // Output
        : [a] "+x" (a),
          [tmp_1] "+x" (tmp_1),
          [tmp_2] "+x" (tmp_2)
        // Inputs
        : [b] "x" (b),
          [data0] "x" (__DATA_0),
          [data1] "x" (__DATA_1),
          [data2] "x" (__DATA_2),
          [data3] "x" (__DATA_3)
        // Clobbers
        : "al", "eax", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "xmm0", "xmm7"
    );
    return a;
}

#endif // _MSC_VER