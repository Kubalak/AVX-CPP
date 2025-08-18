# AVX-CPP - AVX2 made easy in C++

![Github Actions Status](https://github.com/Kubalak/AVX-CPP/workflows/CMake%20multiplatform/badge.svg) [![LICENSE](https://img.shields.io/badge/LICENSE-MIT-royalblue?logo=github&logoColor=lightgray)](LICENSE)

AVX-CPP aims to provide efficient and easy way of using AVX2 in C++. It provides basic numeric types and math operations.

**NOTE:** This is still in development state. Some features are not yet available!

## Types and functions

Library provides both integer and floating-point types:

- Integer types:
  - `Int256` - Vector containing 8 signed 32-bit integers (`__m256i`) &#x2705;
  - `Uint256` - Vector containing 8 unsigned 32-bit integers (`__m256i`) &#x2705;
  - `Short256` - Vector containing 16 signed 16-bit integers (`__m256i`) &#x2705;
  - `Ushort256` - Vector containing 16 unsigned 16-bit integers (`__m256i`) &#x2705;
  - `Long256` - Vector containing 4 signed 64-bit integers (`__m256i`) [&#9888;&#65039; &#x1F6A9;](#known-issues)
  - `ULong256` - Vector containing 4 unsigned 64-bit integers (`__m256i`) [&#9888;&#65039; &#x1F6A9;](#known-issues)
  - `Char256` - Vector containing 32 signed 8-bit integers (`__m256i`) &#x2705;
  - `Uchar256` - Vector containing 32 unsigned 8-bit integers (`__m256i`) &#x2705;
- Floating-point types:
  - `Float256` - Vector containing 8 floats (`__m256`) &#x2705;
  - `Double256` - Vector containing 4 doubles (`__m256d`) &#x2705;

Supported math functions:

- Trigonometric functions `Double256` and `Float256` (on MSVC) &#x1F6A7; (on other)
- Inverse trigonometric functions `Double256` and `Float256` &#x1F6A7;
- Hyperbolic functions `Double256` and `Float256` &#x1F6A7;
- Inverse hyperbolic functions `Double256` and `Float256` &#x1F6A7;

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
- `/` `/=` - all types + scalars [&#9888;&#65039;](#known-issues)
- `%` `%=` - integer types + integer scalars [&#9888;&#65039;](#known-issues)
- `|` `|=` - integer types + integer scalars
- `&` `&=` - integer types + integer scalars
- `^` `^=` - integer types + integer scalars
- `<<` `<<=` - integer types + integer scalars
- `>>` `>>=` - integer types + integer scalars
- `[]` - all types, read only
- `~` - integer types

Other than that all types support initialization using:

- Literal value e.g. `Int256` can be initialized using single `int` value.
- `std::array` with corresponding type and valid size e.g. `Int256` requires `std::array<int, 8>` as initializer.
- Pointer to memory address of at least 32 bytes.
- Initializer list (still in development).

Elements from vectors can be extracted using following methods:

- `[]` returns a value from selected index. In debug builds providing inaccurate index raises an `std::invalid_argument` exception. For release builds invalid index is suppresed by using `index & (size - 1)` formula.
- `load(*pSrc)` loads data from memory using `_mm256_lddqu_si256`/`_mm256_loadu_pd`/`_mm256_loadu_ps` according to stored type.
- `save(std::array&)` saves data to array using `_mm256_storeu_si256` function.
- `save(*pDest)` saves data to memory pointed by `pDest` using `_mm256_storeu_si256` or `_mm256_storeu_ps/pd` function.
- `saveAligned(*pDest)` saves data to memory pointed by `pDest` for memory that is aligned on 32 byte boundary using `_mm256_store_si256` or `_mm256_store_ps/pd`.

<span id="details">* For `Float256` and `Double256` `0` and `-0` are considered equal.</span>

<!--
# AVX-CPP is fast!

Here is the table comparing runtime between non-AVX2 algorithm, raw AVX2 and the one using AVX-CPP library. To see how performance is tested go [here](src/tests/perf). Table below shows operations time for 1GB of data.

Benchmark details (this is to show best-case scenario as MSVC does not optimize for SIMD by default as GCC does):
- CPU: AMD Ryzen 9950X3D
- OS: Win 11 Pro
- Compiler: MSVC v19.43.34810
- Flags: /arch:AVX /arch:AVX2

| Tested type | Operator `+`, `+=` (SEQ/AVXCPP/AVX) | `-`, `-=` | `*`, `*=` | `/`, `/=` | `%`, `%=` |
| --- | ---------- | ---------- | ---------- | ---------- | ---------- |
| [Char256](src/types/char256.hpp) | 1.61 s / 287.76 ms / - | 1.62 s / 286.86 ms / - | 1.85 s / 305.68 ms / - | 5.73 s / 1.57 s / - | 5.14 s / 1.65 s / - |
| [UChar256](src/types/uchar256.hpp) | 1.64 s / 271.11 ms / - | 1.64 s / 271.44 ms / - | 1.86 s / 298.45 ms / - | 6.04 s / 1.55 s / - | 6.03 s / 1.63 s / - |
| [Int256](src/types/int256.hpp) | 558.8 / 262 / 266 ms | - | - | - | - |
| [UInt256](src/types/uint256.hpp) | - | - | - | - | - |
| [Short256](src/types/short256.hpp) | - | - | - | - | - |
| [UShort256](src/types/ushort256.hpp) | - | - | - | - | - |-->

<!--| [Long256](src/types/long256.hpp) | - | - | - | - | - |
| [ULong256](src/types/ulong256.hpp) | - | - | - | - | - |
| [Float256](src/types/float256.hpp) | - | - | - | - | - |
| [Double256](src/types/double256.hpp) | - | - | - | - | - |-->

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

Building has been tested on following compilers (build and run):

- GCC 11.4.0-1ubuntu1~22.04
- Clang 14.0.0-1ubuntu1.1
- MSVC 19.44.35209 (VS 2022)

## Documentation

Documentation is available here: [https://kubalak.github.io/AVX-CPP/](https://kubalak.github.io/AVX-CPP/)

If you want to read documentation offline go to [docs/sphinx](docs/sphinx).

## Known issues

- &#9888;&#65039; - `/` and `%` might not always use SIMD instructions to calculate results due to instruction set restrictions. Some types use casting to `float` to perform those operations.
- &#x1F6A9; - `*` and `*=` don't use AVX2. If AVX512 is available (AVX512DQ and AVX512VL) AVX512 instructions are used
- `Long256` and `ULong256` don't use AVX/AVX2 for `*`, `/` and `%` due to lack available SIMD instructions. In AVX512 mode they might use some SIMD instructions (which will be checked once getting access to CPU supporting AVX512).
