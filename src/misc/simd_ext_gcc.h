#pragma once
#ifndef __SVML_GCC_H
#define __SVML_GCC_H

#include <immintrin.h>

#ifndef _MSC_VER // If not MSVC provide function definitions for GCC/Clang

static const __m256d __DATA_0 = {2.0, 2.0, 2.0, 2.0};
static const __m256d __DATA_1 = {2.0000000000002274, 2.0000000000002274, 2.0000000000002274, 2.0000000000002274};

#ifdef __cplusplus // Use for linking C lib inside C++
extern "C" { 
#endif
    __m256i _mm256_div_epi32(__m256i a, __m256i b);

#ifdef __cplusplus // End of extern C block
}
#endif

#endif // _MSC_VER
#endif // __SVML_GCC_H