cmake_minimum_required(VERSION 3.25)
project(AVX-CPP VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)

set(SOURCE_DIR 
    "${CMAKE_SOURCE_DIR}/src"
)

file(
    GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.hpp"
)

add_library(AVX-CPP STATIC ${SOURCES})

target_include_directories(AVX-CPP PUBLIC ${SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GCC|Clang")
    target_compile_options(AVX-CPP PRIVATE -mavx -mavx2)
elseif(MSVC)
    target_compile_options(AVX-CPP PRIVATE /arch:AVX /arch:AVX2)
endif()