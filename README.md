# AVX-CPP - AVX2 made easy in C++
AVX-CPP aims to provide efficient and easy way of using AVX2 in C++. It provides basic numeric types and math operations.

# Types and functions

Library provides both integer and floating-point types:<br/>
- Integer types:<br/>
  - `Long256` - Vector containing 4 signed 64-bit integers (`__m256i`)<br/>
  - `Long256` - Vector containing 4 unsigned 64-bit integers (`__m256i`)<br/>
  - `Int256` - Vector containing 8 signed 32-bit integers (`__m256i`)<br/>
  - `Uint256` - Vector containing 8 unsigned 32-bit integers (`__m256i`)<br/>
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
- `/` `/=` - all types (need to optimize) + scalars
- `%` `%=` - integer types (need to optimize) + scalars
- `|` `|=` - integer types + integer scalars
- `&` `&=` - integer types + integer scalars
- `^` `^=` - integer types + integer scalars
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
| AVX_BUILD_SHARED_LIBS | BOOLEAN| ON | Build shared libraries (*.dll or *.so) |
| AVX_ENABLE_TESTS | BOOLEAN | OFF | Build tests when building library |

