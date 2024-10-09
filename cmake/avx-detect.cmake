include(CheckCSourceRuns)

set(AVX2_CODE "
    #include <immintrin.h>

    int main(int argc, char*argv[]) {
        __m256i a = _mm256_set1_epi32(2);
        __m256i b = _mm256_mullo_epi32(a, _mm256_set1_epi32(3));
        return 0;
    }
")

set(AVX512_CODE "
    #include <immintrin.h>

    int main(int argc, char*argv[]) {
        __m512i a = _mm512_set_epi64(1,2,3,4,5,6,7,8);
        __m512i b = _mm512_mullo_epi64(a, _mm512_set1_epi64(3));
        return 0;
    }
")

if(CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
    set(AVX2_FLAGS -mavx2)
    set(AVX512_FLAGS -mavx512f -mavx512dq -mavx512vl)
elseif(MSVC)
    set(AVX2_FLAGS /arch:AVX2)
    set(AVX512_FLAGS /arch:AVX512)
endif()

set(CMAKE_REQUIRED_FLAGS_PRIOR ${CMAKE_REQUIRED_FLAGS})
set(CMAKE_REQUIRED_FLAGS ${AVX2_FLAGS})
check_c_source_runs("${AVX2_CODE}" AVX2_SUPPORTED)
if(NOT AVX2_SUPPORTED)
    message(FATAL_ERROR "AVX2 not supported! This library requires AVX2 to work!")
endif()
set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PRIOR})
set(CMAKE_REQUIRED_FLAGS ${AVX512_FLAGS})
check_c_source_runs("${AVX512_CODE}" AVX512_SUPPORTED)
if(AVX512_SUPPORTED)
    message(STATUS "AVX512 supported")
else()
    message(STATUS "AVX512 not supported")
endif()
set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PRIOR})
