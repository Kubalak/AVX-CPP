# AVX-CPP - AVX2 made easy in C++

![Github Actions Status](https://github.com/Kubalak/AVX-CPP/workflows/CMake%20multiplatform/badge.svg) [![LICENSE](https://img.shields.io/badge/LICENSE-MIT-royalblue?logo=github&logoColor=lightgray)](LICENSE) [![VERSION](https://img.shields.io/badge/Version-v0.9.9-blue)](CMakeLists.txt) [![C++ version](https://img.shields.io/badge/version-20-royalblue?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20.html) [![C++ version](https://img.shields.io/badge/version-17-royalblue?logo=c)](https://www.geeksforgeeks.org/c/c-17-standard/)

AVX-CPP aims to provide efficient and easy way of using AVX2 in C++. It provides basic numeric types and math operations.

## Types and functions

Library provides both integer and floating-point types:

- Integer types:
  - `Int256` - Vector containing 8 signed 32-bit integers (`__m256i`) &#x2705;
  - `Uint256` - Vector containing 8 unsigned 32-bit integers (`__m256i`) &#x2705;
  - `Short256` - Vector containing 16 signed 16-bit integers (`__m256i`) &#x2705;
  - `Ushort256` - Vector containing 16 unsigned 16-bit integers (`__m256i`) &#x2705;
  - `Long256` - Vector containing 4 signed 64-bit integers (`__m256i`) [**](#known-issues)
  - `ULong256` - Vector containing 4 unsigned 64-bit integers (`__m256i`) [**](#known-issues)
  - `Char256` - Vector containing 32 signed 8-bit integers (`__m256i`) &#x2705;
  - `Uchar256` - Vector containing 32 unsigned 8-bit integers (`__m256i`) &#x2705;
- Floating-point types:
  - `Float256` - Vector containing 8 floats (`__m256`) &#x2705;
  - `Double256` - Vector containing 4 doubles (`__m256d`) &#x2705;

Supported math functions:

- Trigonometric functions `Double256` and `Float256`. On MSVC implemented using included SVML library, on GCC and Clang uses [Sleef](https://github.com/shibatch/sleef) library to provide same functionality.
- Inverse trigonometric functions `Double256` and `Float256` - same as trigonometric functions.

<!-- Other supported functions: 
- `sum` - supports all types
- `avg` - supports all types
- `stddev` - supports all types
- -->

Supported operators:

- `==` `!=` - all types + scalars [*](#details)
- `+` `+=` - all types + scalars
- `-` `-=` - all types + scalars
- `*` `*=` - all types + scalars
- `/` `/=` - all types + scalars
- `%` `%=` - integer types + integer scalars
- `|` `|=` - integer types + integer scalars
- `&` `&=` - integer types + integer scalars
- `^` `^=` - integer types + integer scalars
- `<<` `<<=` - integer types + integer scalars
- `>>` `>>=` - integer types + integer scalars
- `[]` - all types, **<u>read only</u>**
- `~` - integer types

Other than that all types support initialization using:

- Literal value e.g. `Int256` can be initialized using single `int` value.
- `std::array` with corresponding type and valid size e.g. `Int256` requires `std::array<int, 8>` as initializer.
- Pointer to memory address of at least 32 bytes.
- Initializer lists.

Elements from vectors can be extracted using following methods:

- `[]` returns a value from selected index. In debug builds providing inaccurate index raises an `std::invalid_argument` exception. For release builds invalid index is suppresed by using `index & (size - 1)` formula.
- `load(*pSrc)` loads data from memory using `_mm256_lddqu_si256`/`_mm256_loadu_pd`/`_mm256_loadu_ps` according to stored type.
- `save(std::array&)` saves data to array using `_mm256_storeu_si256` function.
- `save(*pDest)` saves data to memory pointed by `pDest` using `_mm256_storeu_si256` or `_mm256_storeu_ps/pd` function.
- `saveAligned(*pDest)` saves data to memory pointed by `pDest` for memory that is aligned on 32 byte boundary using `_mm256_store_si256` or `_mm256_store_ps/pd`.

<span id="details">* For `Float256` and `Double256` `0` and `-0` are considered equal.</span>

## Almost zero cost abstraction

Here is the table comparing runtime between non-AVX2 algorithm, raw AVX2 and the one using AVX-CPP library. To see how performance is tested go [here](src/tests/perf). Table below shows operations time for 1GB of data.

Benchmark details (this is to show best-case scenario as MSVC does not optimize for SIMD by default as GCC does):

- CPU: AMD Ryzen 9950X3D
- OS: Win 11 Pro
- Compiler: MSVC v19.44

With AVX2 only [`Int256`](src/types/int256.hpp) 10 runs avg:

| Type/operator | +, +=  | -, -=  | \*, \*= | /, /=  | %, %=  |
|---------------|--------|--------|---------|--------|--------|
| SEQ           | 135 ms | 135 ms | 143 ms  | 569 ms | 568 ms |
| AVX2 raw      | 72 ms  | 72 ms  | 72 ms   | 149 ms | 168 ms |
| AVX-CPP       | 72 ms  | 72 ms  | 72 ms   | 149 ms | 168 ms |

With AVX512 enabled `Int256` 10 runs avg:

| Type/operator | +, +=  | -, -=  | \*, \*= | /, /=  | %, %=  |
|---------------|--------|--------|---------|--------|--------|
| SEQ           | 135 ms | 148 ms | 147 ms  | 569 ms | 568 ms |
| AVX2 raw      | 72 ms  | 72 ms  | 72 ms   | 73 ms  | 73 ms  |
| AVX-CPP       | 72 ms  | 72 ms  | 72 ms   | 73 ms  | 73 ms  |

The difference is even more visible for [`Char256`](src/types/char256.hpp):

| Type/operator | +, +=  | -, -=  | \*, \*= | /, /=   | %, %=   | Comment |
|---------------|--------|--------|---------|---------|---------|---------|
| SEQ           | 530 ms | 531 ms | 782 ms  | 2318 ms | 2319 ms |         |
| AVX2 raw      | 72 ms  | 72 ms  | 72 ms   | 138 ms  | 325 ms  | Optimized for this specific case |
| AVX-CPP       | 72 ms  | 72 ms  | 72 ms   | 258 ms  | 327 ms  | |

With AVX512 enabled `Char256` 10 runs avg:

| Type/operator    | +, +=  | -, -=  | \*, \*= | /, /=   | %, %=  | Comment |
|------------------|--------|--------|---------|---------|--------|---------|
| SEQ              | 532 ms | 527 ms | 610 ms  | 2316 ms | 2321 ms | |
| AVX2 (512) raw   | 72 ms  | 72 ms  | 72 ms   | **82 ms**   | **132**  ms | Optimized for this specific case |
| AVX-CPP (AVX512) | 72 ms  | 72 ms  | 72 ms   | **125 ms**  | **143** ms | |

## Sample usage

```cpp
#include "types/int256.hpp"
#include <iostream>

int main(int argc, char* argv[]) {

    avx::Int256 a({1, 2, 3, 4, 5, 6, 7, 8}); // Initialize vector values
    avx::Int256 b(std::array<int,8>{0, 43, 5, 8, 7, 9, 2, 12});
    avx::Int256 c(5); // Will initialize vector with 5 (5, 5, 5..5).

    // The code below will print "Int256(13, 12, 11, 10, 9, 8, 7, 6)"
    std::cout << (a + 5).str() << '\n'; 

    // The code below will print "Int256(20, 9, 15, 12, 12, 8, 45, 1)"
    std::cout << (a + b).str() << '\n';

    if(c == 5) // Compare with scalar. Returns true only if all fields are equal.
      std::cout << "All c vector fields are equal to 5.\n";

    if(a != b) // Compare with other vector. Returns true if ANY field is different between vectors.
      std::cout << "Not all values in a and b vectors are equal\n";
    
    a *= 2; 
    std::cout << a.str() << '\n'; // Will print "Int256(2, 4, 6, 8, 10, 12, 14, 16)"

    return 0;
}
```

## Compilation

The project uses CMake tool for compilation.

Available building options are

| Option | Type | Defualt | Description |
| --- | --- | --- | ---|
| BUILD_SHARED_LIBS | BOOLEAN| ON | Build shared libraries (\*.dll or \*.so) |
| BUILD_TESTING | BOOLEAN | OFF | Build tests when building library |
| BUILD_PERFORMANCE_TESTS | BOOLEAN | OFF | Build performance tests |
| BUILD_DEEP_TESTS | BOOLEAN | OFF | Build types deep testing (brute force testing for all possible values, doesn't work on Windows) |
| BUILD_USE_AVX512 | BOOLEAN | [CPU Support](cmake/avx-detect.cmake) | Use AVX512 when generatring binaries |

---

Building has been tested on following compilers (build and run):

- GCC 11.4.0-1ubuntu1~22.04
- Clang 14.0.0-1ubuntu1.1
- MSVC 19.44.35209 (VS 2022)

## Documentation

Documentation is available here: [https://kubalak.github.io/AVX-CPP/](https://kubalak.github.io/AVX-CPP/)

If you want to read documentation offline go to [docs/sphinx](docs/sphinx).

## Known issues

- `Long256` and `ULong256` don't use SIMD for `*`, `*=`, `/`, `/=`, `%` and `%=` due to the lack of useful functions in AVX2.
- In very specific scenario in edge cases with AVX512 enabled == operator fails for ^= checks. Debugging this indicates that compiler is using [`vpternlogq`](https://www.felixcloutier.com/x86/vpternlogd:vpternlogq) however by doing so is causing UB which is mitigated by using `vzeroupper` (which forces compiler to use `vpxor` as intended). Please use `__FIX_CMP` macro to enable this in AVX512 builds. Likelihood for this situation to happen in real life scenario has not been tested.
