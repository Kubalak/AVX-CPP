#include "short256.hpp"

namespace avx {
    static_assert(sizeof(short) == 2, "You are compiling to 32-bit. Please switch to x64 to avoid undefined behaviour.");
};