# AVX-CPP - AVX2 made easy in C++
AVX-CPP aims to provide efficient and easy way of using AVX2 in C++. It provides basic numeric types and math operations.

**NOTE:** This is still in development state. Some features are not yet available!

# Types and functions

Library provides both integer and floating-point types (&#x2705; shows currently working):<br/>
- Integer types:<br/>
  - `Long256` - Vector containing 4 signed 64-bit integers (`__m256i`) &#x2705;<br/>
  - `ULong256` - Vector containing 4 unsigned 64-bit integers (`__m256i`) &#x2705;<br/>
  - `Int256` - Vector containing 8 signed 32-bit integers (`__m256i`) &#x2705;<br/>
  - `Uint256` - Vector containing 8 unsigned 32-bit integers (`__m256i`) &#x2705;<br/>
  - `Short256` - Vector containing 16 signed 16-bit integers (`__m256i`)<br/>
  - `Ushort256` - Vector containing 16 unsigned 16-bit integers (`__m256i`)<br/>
  - `Char256` - Vector containing 32 signed 8-bit integers (`__m256i`)<br/>
  - `Uchar256` - Vector containing 32 unsigned 8-bit integers (`__m256i`)<br/>
- Floating-point types:<br/>
  - `Float256` - Vector containing 8 floats (`__m256`)<br/>
  - `Double256` - Vector containing 4 doubles (`__m256d`)<br/>

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
- `==` `!=` - all types + scalars
- `+` `+=` - all types + scalars
- `-` `-=` - all types + scalars
- `*` `*=` - all types + scalars
- `/` `/=` - all types (need to optimize) + scalars [&#9888;&#65039;](#known-issues)
- `%` `%=` - integer types (need to optimize) + scalars [&#9888;&#65039;](#known-issues)
- `|` `|=` - integer types + integer scalars
- `&` `&=` - integer types + integer scalars
- `^` `^=` - integer types + integer scalars
- `<<` `<<=` - integer types + integer scalars
- `>>` `>>=` - integer types + integer scalars
- `[]` - all types (gets underlying value from vector)
- `~` - integer types

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

## Known issues
- Building binaries with MinGW on Windows may result in segfault (`ERROR_ACCESS_VIOLATION`) when running program for yet unknown reasons<br/>
- &#9888;&#65039; `/` and `%` on integer types don't use AVX2 or use it scarcely due to lack of available AVX2 functions (type conversion may introduce more bugs).
- Performance benefits need further testing
- ~~`CMakeLists.txt` doesn't currently support building tests when `BUILD_SHARED_LIBS` is `ON`.~~

