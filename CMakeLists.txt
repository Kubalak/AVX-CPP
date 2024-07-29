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


if(BUILD_SHARED_LIBS)
    add_library(avxcpp SHARED ${SOURCES})
else()
    add_library(avxcpp STATIC ${SOURCES})
endif()

target_include_directories(avxcpp PUBLIC ${SOURCE_DIR})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
    target_compile_options(avxcpp PRIVATE -mavx2)
elseif(MSVC)
    target_compile_options(avxcpp PRIVATE /arch:AVX /arch:AVX2)
endif()