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

__m256i _mm256_div_epi64(__m256i a, __m256i b) {

  __m256i tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8;
  tmp_1 = tmp_2 = tmp_3 = tmp_4 = tmp_5 = tmp_6 = tmp_7 = tmp_8 = _mm256_setzero_si256();

  __asm__(
        "nop\n\t"
        "vmovdqu        %[a], %%ymm0\n\t"
        "vmovdqu        %[b], %%ymm1\n\t"
        "vpxor          %%ymm3, %%ymm3, %%ymm3\n\t"
        "vmovupd        %[const_1], %%ymm7\n\t"
        "vpcmpgtq       %%ymm1, %%ymm3, %%ymm4\n\t"
        "vpxor          %%ymm4, %%ymm1, %%ymm2\n\t"
        "vmovdqu        %%ymm4, %[var_1]\n\t"
        "vmovdqa        %%ymm0, %%ymm5\n\t"
        "vpcmpgtq       %%ymm5, %%ymm3, %%ymm1\n\t"
        "vpsubq         %%ymm4, %%ymm2, %%ymm0\n\t"
        "vpxor          %%ymm1, %%ymm5, %%ymm2\n\t"
        "vmovupd        %[const_2], %%ymm4\n\t"
        "vmovdqu        %%ymm1, %[var_2]\n\t"
        "vmovdqu        %%ymm0, %[var_3]\n\t"
        "vpsubq         %%ymm1, %%ymm2, %%ymm5\n\t"
        "vmovupd        %[const_3], %%ymm2\n\t"
        "vmovdqu        %%ymm5, %[var_4]\n\t"
        "vandpd         %%ymm7, %%ymm0, %%ymm6\n\t"
        "vpsrlq         $0x20,  %%ymm0, %%ymm0\n\t"
        "vorpd          %%ymm2, %%ymm6, %%ymm1\n\t"
        "vorpd          %%ymm4, %%ymm0, %%ymm3\n\t"
        "vsubpd         %%ymm2, %%ymm1, %%ymm1\n\t"
        "vsubpd         %%ymm4, %%ymm3, %%ymm6\n\t"
        "vmovdqu        %%ymm0, %[var_5]\n\t"
        "vaddpd         %%ymm6, %%ymm1, %%ymm3\n\t"
        "vcvtpd2ps      %%ymm3, %%xmm4\n\t"
        "vrcpps         %%xmm4, %%xmm4\n\t"
        "vcvtps2pd      %%xmm4, %%ymm0\n\t"
        "vmovupd        %[const_4], %%ymm4\n\t"
        "vfnmadd231pd   %%ymm0, %%ymm3, %%ymm4\n\t"
        "vandpd         %[const_5], %%ymm3, %%ymm3\n\t"
        "vsubpd         %%ymm3, %%ymm6, %%ymm6\n\t"
        "vfmadd213pd    %%ymm0, %%ymm4, %%ymm0\n\t"
        "vaddpd         %%ymm6, %%ymm1, %%ymm1\n\t"
        "vmovupd        %%ymm0, %[var_6]\n\t"
        "vandpd         %%ymm7, %%ymm5, %%ymm7\n\t"
        "vorpd          %%ymm2, %%ymm7, %%ymm4\n\t"
        "vsubpd         %%ymm2, %%ymm4, %%ymm6\n\t"
        "vmovupd        %[const_2], %%ymm4\n\t"
        "vpsrlq         $0x20,  %%ymm5, %%ymm5\n\t"
        "vorpd          %%ymm4, %%ymm5, %%ymm7\n\t"
        "vsubpd         %%ymm4, %%ymm7, %%ymm5\n\t"
        "vaddpd         %%ymm5, %%ymm6, %%ymm7\n\t"
        "vmulpd         %%ymm7, %%ymm0, %%ymm0\n\t"
        "vroundpd       $3, %%ymm0, %%ymm7\n\t"
        "vmovupd        %[const_6], %%ymm0\n\t"
        "vandpd         %%ymm0, %%ymm7, %%ymm7\n\t"
        "vfnmadd231pd   %%ymm3, %%ymm7, %%ymm5\n\t"
        "vfnmadd231pd   %%ymm1, %%ymm7, %%ymm6\n\t"
        "vaddpd         %%ymm6, %%ymm5, %%ymm5\n\t"
        "vaddpd         %%ymm4, %%ymm7, %%ymm6\n\t"
        "vsubpd         %%ymm4, %%ymm6, %%ymm4\n\t"
        "vmovupd        %%ymm6, %[var_7]\n\t"
        "vsubpd         %%ymm4, %%ymm7, %%ymm7\n\t"
        "vaddpd         %%ymm2, %%ymm7, %%ymm6\n\t"
        "vmovupd        %[var_6], %%ymm7\n\t"
        "vmovupd        %%ymm6, %[var_8]\n\t"
        "vmulpd         %%ymm5, %%ymm7, %%ymm4\n\t"
        "vroundpd       $3,     %%ymm4, %%ymm6\n\t"
        "vandpd         %%ymm0, %%ymm6, %%ymm4\n\t"
        "vfnmadd231pd   %%ymm3, %%ymm4, %%ymm5\n\t"
        "vfnmadd231pd   %%ymm1, %%ymm4, %%ymm5\n\t"
        "vaddpd         %%ymm2, %%ymm4, %%ymm4\n\t"
        "vmulpd         %%ymm5, %%ymm7, %%ymm6\n\t"
        "vroundpd       $3,     %%ymm6, %%ymm6\n\t"
        "vandpd         %%ymm0, %%ymm6, %%ymm0\n\t"
        "vfnmadd213pd   %%ymm5, %%ymm0, %%ymm3\n\t"
        "vfnmadd213pd   %%ymm3, %%ymm0, %%ymm1\n\t"
        "vaddpd         %%ymm2, %%ymm0, %%ymm0\n\t"
        "vmulpd         %%ymm1, %%ymm7, %%ymm2\n\t"
        "vmovdqu        %[var_3], %%ymm1\n\t"
        "vmovdqu        %[var_4], %%ymm7\n\t"
        "vpxor          %%ymm3, %%ymm3, %%ymm3\n\t"
        "vpcmpeqq       %%ymm3, %%ymm1, %%ymm3\n\t"
        "vpmovmskb      %%ymm3, %%eax\n\t"
        "test           %%eax, %%eax\n\t"
        "je             div_epi64_cont\n\t"

        "mov            $0, %%eax\n\t"
        "div            %%al\n\t"

        "div_epi64_cont:\n\t"
        "vmovupd        %[var_7], %%ymm3\n\t"
        "vcvttpd2dq     %%ymm2, %%xmm2\n\t"
        "vpsllq         $0x20, 	%%ymm3, %%ymm5\n\t"
        "vpshuflw       $0xa4, %[var_8], %%ymm3\n\t"
        "vpshufhw       $0xa4, 	%%ymm3, %%ymm6\n\t"
        "vpaddd         %%ymm6, %%ymm5, %%ymm3\n\t"
        "vmovdqu        %[const_7], %%ymm6\n\t"
        "vpand          %%ymm6, %%ymm4, %%ymm4\n\t"
        "vpand          %%ymm6, %%ymm0, %%ymm0\n\t"
        "vpaddq         %%ymm4, %%ymm3, %%ymm5\n\t"
        "vpermq         $0xd8,  %%ymm2, %%ymm3\n\t"
        "vpshufd        $0xd8,  %%ymm3, %%ymm4\n\t"
        "vpaddd         %%ymm0, %%ymm4, %%ymm0\n\t"
        "vpaddq         %%ymm0, %%ymm5, %%ymm3\n\t"
        "vpmuludq       %%ymm3, %%ymm1, %%ymm2\n\t"
        "vpsubq         %%ymm2, %%ymm7, %%ymm5\n\t"
        "vpsrlq         $0x20,  %%ymm3, %%ymm7\n\t"
        "vpmuludq       %%ymm7, %%ymm1, %%ymm0\n\t"
        "vpmuludq       %[var_5], %%ymm3, %%ymm2\n\t"
        "vpaddq         %%ymm2, %%ymm0, %%ymm4\n\t"
        "vpsrlq         $0x3f, %%ymm1, %%ymm2\n\t"
        "vmovdqu        %[var_2], %%ymm7\n\t"
        "vpsllq         $0x20, %%ymm4, %%ymm6\n\t"
        "vpxor          %[var_1], %%ymm7, %%ymm0\n\t"
        "vpsubq         %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpsrlq         $0x3f,	%%ymm5, %%ymm4\n\t"
        "vpcmpeqq       %%ymm4, %%ymm2, %%ymm6\n\t"
        "vpcmpgtq       %%ymm5, %%ymm1, %%ymm1\n\t"
        "vpand          %%ymm1, %%ymm6, %%ymm1\n\t"
        "vpcmpgtq       %%ymm4, %%ymm2, %%ymm2\n\t"
        "vpor           %%ymm2, %%ymm1, %%ymm4\n\t"
        "vpandn         %[const_8], %%ymm4, %%ymm5\n\t"
        "vpaddq         %%ymm5, %%ymm3, %%ymm3\n\t"
        "vpxor          %%ymm0, %%ymm3, %%ymm6\n\t"
        "vpsubq         %%ymm0, %%ymm6, %%ymm0\n\t"
        "vmovdqu        %%ymm0, %[a]\n\t"
        "nop\n\t"

        : [a] "+m" (a),
          [var_1] "+x" (tmp_1),
          [var_2] "+x" (tmp_2),
          [var_3] "+x" (tmp_3),
          [var_4] "+x" (tmp_4),
          [var_5] "+x" (tmp_5),
          [var_6] "+x" (tmp_6),
          [var_7] "+x" (tmp_7),
          [var_8] "+x" (tmp_8)
        : 
          [b] "m" (b),
          [const_1] "m" (__DATA_2),
          [const_2] "m" (__DATA_4),
          [const_3] "m" (__DATA_5),
          [const_4] "m" (__DATA_6),
          [const_5] "m" (__DATA_7),
          [const_6] "m" (__DATA_8),
          [const_7] "m" (__DATA_9),
          [const_8] "m" (__DATA_10)
          
        : "al", "eax", "xmm2", "xmm4", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "memory"
    );

    return a;
}

__m256 _mm256_sin_ps(__m256 a) {
    // __asm__(
    //   "vxorps %[a], %[a], %[a]"
    //   :
    //     [a] "+x" (a)
    // );

    const __m256 inv_two_pi = _mm256_set1_ps(1.0f / M_2PI_F);
    const __m256 two_pi = _mm256_set1_ps(M_2PI_F);
    const __m256 pi = _mm256_set1_ps(M_PI_F);
    const __m256 minus_pi = _mm256_set1_ps(-M_PI_F);

    // Redukcja argumentu do [-π, π]
    a = _mm256_sub_ps(a, _mm256_mul_ps(two_pi, _mm256_floor_ps(_mm256_mul_ps(a, inv_two_pi))));
    a = _mm256_add_ps(a, _mm256_and_ps(_mm256_cmp_ps(a, minus_pi, _CMP_LT_OS), two_pi)); // a += 2π if a < -π
    a = _mm256_sub_ps(a, _mm256_and_ps(_mm256_cmp_ps(a, pi, _CMP_GT_OS), two_pi));       // x -= 2π if x > π

    // x, x^3, x^5, x^7 (potęgi obliczamy iteracyjnie)
    __m256 x2 = _mm256_mul_ps(a, a);
    __m256 x3 = _mm256_mul_ps(x2, a);
    __m256 x5 = _mm256_mul_ps(x3, x2);
    __m256 x7 = _mm256_mul_ps(x5, x2);

    // Współczynniki szeregu Maclaurina (float, 5 wyrazów)
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c3 = _mm256_set1_ps(-1.0f / 6.0f);       // -x^3 / 3!
    const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);      // x^5 / 5!
    const __m256 c7 = _mm256_set1_ps(-1.0f / 5040.0f);    // -x^7 / 7!

    // Wielomian: sin(x) ≈ x + c3*x^3 + c5*x^5 + c7*x^7
    __m256 res = _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(c7, x7), _mm256_mul_ps(c5, x5)),
                        _mm256_mul_ps(c3, x3)),
                    a);

    return res;

    //return a;
}

__m256d _mm256_sin_pd(__m256d a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256 _mm256_cos_ps(__m256 a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256d _mm256_cos_pd(__m256d a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256 _mm256_tan_ps(__m256 a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256d _mm256_tan_pd(__m256d a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256 _mm256_asin_ps(__m256 a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256d _mm256_asin_pd(__m256d a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256 _mm256_acos_ps(__m256 a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256d _mm256_acos_pd(__m256d a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256 _mm256_atan_ps(__m256 a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}

__m256d _mm256_atan_pd(__m256d a) {
  __asm__(
    "nop\n\t"
    "vxorps %[a], %[a], %[a]\n\t"
    "nop\n\t"
    :
      [a] "+x" (a)
  );

  return a;
}


#endif // _MSC_VER