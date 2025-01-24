# AVX-CPP - AVX2 made easy in C++

![Github Actions Status](https://github.com/Kubalak/AVX-CPP/workflows/CMake%20multiplatform/badge.svg) [![LICENSE](https://img.shields.io/badge/LICENSE-MIT-royalblue?logo=github&logoColor=lightgray)](LICENSE)

AVX-CPP aims to provide efficient and easy way of using AVX2 in C++. It provides basic numeric types and math operations.

**NOTE:** This is still in development state. Some features are not yet available!

## Types and functions

Library provides both integer and floating-point types:

- Integer types:
  - `Int256` - Vector containing 8 signed 32-bit integers (`__m256i`) &#x2705; [&#x2757;](#known-issues)
  - `Uint256` - Vector containing 8 unsigned 32-bit integers (`__m256i`) &#x2705;[&#x2757;](#known-issues)
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

- `==` `!=` - all types + scalars
- `+` `+=` - all types + scalars
- `-` `-=` - all types + scalars
- `*` `*=` - all types + scalars
- `/` `/=` - all types + scalars [&#9888;&#65039;](#known-issues)
- `%` `%=` - integer types + integer scalars [&#9888;&#65039;](#known-issues)
- `|` `|=` - integer types + integer scalars
- `&` `&=` - integer types + integer scalars
- `^` `^=` - integer types + integer scalars
- `<<` `<<=` - integer types + integer scala
- `>>` `>>=` - integer types + integer scala
- `[]` - all types, read only
- `~` - integer types

Other than that all types support initialization using:

- Literal value e.g. `Int256` can be initialized using single `int` value.
- `std::array` with corresponding type and valid size e.g. `Int256` requires `std::array<int, 8>` as initializer.
- Pointer to memory address of at least 32 bytes.
- Initializer list (still in development).

Elements from vectors can be extracted using following methods:

- `[]` returns a value from selected index.
- `load(*sP)` loads data from memory using `_mm256_lddqu_si256`/`_mm256_loadu_pd`/`_mm256_loadu_ps` according to stored type.
- `save(std::array&)` saves data to array using `_mm256_storeu_si256` function.
- `save(*addr)` saves data to memory pointed by `addr` using `_mm256_storeu_si256` function.
- `saveAligned(*addr)` saves data to memory pointed by `addr` for memory that is aligned on 32 byte boundary using `_mm256_store_si256`.

<!--
# AVX-CPP is fast!

Here is the table comparing runtime between non-AVX2 algorithm, raw AVX2 and the one using AVX-CPP library. To see how performance is tested go [here](src/tests/perf). Table below shows testing results on Windows 10 using MSVC. Selected vectors of size 1GiB for all tests.-->

| Tested type | Operator `+`, `+=` (SEQ/AVXCPP/AVX) | `-`, `-=` | `*`, `*=` | `/`, `/=` | `%`, `%=` |
| --- | ---------- | ---------- | ---------- | ---------- | ---------- |
| [Char256](src/types/char256.hpp) | 1.61 s / 287.76 ms / - | 1.62 s / 286.86 ms / - | 1.85 s / 305.68 ms / - | 5.73 s / 1.57 s / - | 5.14 s / 1.65 s / - |
| [UChar256](src/types/uchar256.hpp) | 1.64 s / 271.11 ms / - | 1.64 s / 271.44 ms / - | 1.86 s / 298.45 ms / - | 6.04 s / 1.55 s / - | 6.03 s / 1.63 s / - |
<!--| [Int256](src/types/int256.hpp) | 558.8 / 262 / 266 ms | - | - | - | - |
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
    avx::Int256 b(std::array<int,8>{0, 43, 5, 8, 7, 9, 2, 12})

    // The code below will print "Int256(13, 12, 11, 10, 9, 8, 7, 6)"
    std::cout << (a + 5).str() << '\n'; 

    // The code below will print "Int256(20, 9, 15, 12, 12, 8, 45, 1)"
    std::cout << (a + b).str() << '\n'; 

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
| BUILD_DEEP_TESTS | BOOLEAN | OFF | Build types deep testing (brute force testing for all possible values) |

Building has been tested on following compilers (build and run):

- GCC 11.4.0-1ubuntu1~22.04
- Clang 14.0.0-1ubuntu1.1
- MSVC 19.29.30154

## &#x1F6A7; Documentation

Under construction...

## Known issues

- &#9888;&#65039; `/` and `%` might not always use SIMD instructions to calculate results due to instruction set restrictions. Some types use casting to `float` to perform those operations.
- &#x2757; For `Int256` and `UInt256` the underlying type used in division and modulo is `float` which might cause rounding to occur which will yield inaccurate results. Please refer to [this](https://en.wikipedia.org/wiki/IEEE_754) document about `float` type structure.
- &#x1F6A9; `*` and `*=` don't use AVX2. If AVX512 is available (AVX512DQ and AVX512VL) AVX512 instructions are used
- Performance benefits need further testing (I will provide perf test table here).
- `Int256` division and modulo might produce inaccurate results for larger values.
- `Long256` and `ULong256` don't use AVX/AVX2 for `*`, `/` and `%` due to lack available SIMD instructions. In AVX512 mode they might use some SIMD instructions (which will be checked once getting access to CPU supporting AVX512).
