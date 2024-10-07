# AVX-CPP - AVX2 made easy in C++
![Github Actions Status](https://github.com/Kubalak/AVX-CPP/workflows/CMake%20multiplatform/badge.svg)
AVX-CPP aims to provide efficient and easy way of using AVX2 in C++. It provides basic numeric types and math operations.

**NOTE:** This is still in development state. Some features are not yet available!

# Types and functions

Library provides both integer and floating-point types (&#x2705; shows currently working):<br/>
- Integer types:<br/>
  - `Long256` - Vector containing 4 signed 64-bit integers (`__m256i`) &#x2705;<br/>
  - `ULong256` - Vector containing 4 unsigned 64-bit integers (`__m256i`) &#x2705;<br/>
  - `Int256` - Vector containing 8 signed 32-bit integers (`__m256i`) &#x2705;<br/>
  - `Uint256` - Vector containing 8 unsigned 32-bit integers (`__m256i`) &#x2705;<br/>
  - `Short256` - Vector containing 16 signed 16-bit integers (`__m256i`) &#x2705;<br/>
  - `Ushort256` - Vector containing 16 unsigned 16-bit integers (`__m256i`) &#x2705;<br/>
  - `Char256` - Vector containing 32 signed 8-bit integers (`__m256i`) [&#9888;&#65039;](#known-issues)<br/>
  - `Uchar256` - Vector containing 32 unsigned 8-bit integers (`__m256i`)<br/>
- Floating-point types:<br/>
  - `Float256` - Vector containing 8 floats (`__m256`) &#x2705;<br/>
  - `Double256` - Vector containing 4 doubles (`__m256d`) &#x2705;<br/>

Supported math functions:<br/>
- Trigonometric functions `Double256` and `Float256`<br/>
- Inverse trigonometric functions `Double256` and `Float256`<br/>
- Hyperbolic functions `Double256` and `Float256` <br/>
- Inverse hyperbolic functions `Double256` and `Float256` <br/>

<!-- Other supported functions: 
- `sum` - supports all types
- `avg` - supports all types
- `stddev` - supports all types
- -->
Supported operators:<br/>
- `==` `!=` - all types + scalars <br/>
- `+` `+=` - all types + scalars <br/>
- `-` `-=` - all types + scalars <br/>
- `*` `*=` - all types + scalars <br/>
- `/` `/=` - all types (need to optimize) + scalars [&#9888;&#65039;](#known-issues) <br/>
- `%` `%=` - integer types (need to optimize) + scalars [&#9888;&#65039;](#known-issues) <br/>
- `|` `|=` - integer types + integer scalars <br/>
- `&` `&=` - integer types + integer scalars <br/>
- `^` `^=` - integer types + integer scalars <br/>
- `<<` `<<=` - integer types + integer scala <br/>
- `>>` `>>=` - integer types + integer scala <br/>
- `[]` - all types (gets underlying value from vector) <br/>
- `~` - integer types <br/>

Other than that all types support initialization using:<br/>
- Literal value e.g. `Int256` can be initialized using single `int` value. <br/>
- `std::array` with corresponding type and valid size e.g. `Int256` requires `std::array<int, 8>` as initializer. <br/>
- Pointer to memory address of at least 32 bytes. <br/>
- Initializer list (still in development). <br/>

Elements from vectors can be extracted using following methods:
- `[]` returns a value from selected index. <br/>
- `save(std::array&)` saves data to array using `_mm256_storeu_*` function. <br/>
- `save(*addr)` saves data to memory pointed by `addr` using `_mm256_storeu_*` function. <br/>
- `saveAligned(*addr)` saves data to memory pointed by `addr` for memory that is aligned on 32 byte boundary.<br/>


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
| BUILD_SHARED_LIBS | BOOLEAN| ON | Build shared libraries (*.dll or *.so) |
| BUILD_TESTING | BOOLEAN | OFF | Build tests when building library |

Building has been tested on compilers:
- GCC 11.4.0-1ubuntu1~22.04
- Clang 14.0.0-1ubuntu1.1
- MSVC 19.29.30154

## Known issues
- Building binaries with MinGW on Windows may result in segfault (`ERROR_ACCESS_VIOLATION`) when running program for yet unknown reasons<br/>
- &#9888;&#65039; `/` and `%` on integer types don't use AVX2 or use it scarcely due to lack of available AVX2 functions. *SOME* of them are using `float` conversion vectors and then converting them back to integer types using truncation.<br/>
- Performance benefits need further testing. 
- `Char256` may trigger numerical exception in some situations (will be fixed soon).


