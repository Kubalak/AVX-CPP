cmake_minimum_required(VERSION 3.25)
project(AVX-CPP VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)

set(SOURCE_DIR 
    "${CMAKE_SOURCE_DIR}/src"
)

file(
    GLOB_RECURSE SOURCES "${SOURCE_DIR}/types/*.cpp" "${SOURCE_DIR}/types/*.hpp" "${SOURCE_DIR}/ops/*.cpp" "${SOURCE_DIR}/ops/*.hpp"
)

option(AVX_BUILD_SHARED_LIBS "Build shared libraries" ON)
option(AVX_ENABLE_TESTS "Build tests" OFF)


if(AVX_BUILD_SHARED_LIBS)
    add_library(avxcpp SHARED ${SOURCES})
else()
    add_library(avxcpp STATIC ${SOURCES})
endif()

set_target_properties(avxcpp PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
target_include_directories(avxcpp PUBLIC ${SOURCE_DIR})

if(AVX_ENABLE_TESTS)
    add_executable(int_arithmetic_test "${SOURCE_DIR}/tests/int_arithmetic_test.cpp")

    target_link_libraries(int_arithmetic_test PRIVATE avxcpp)
    enable_testing()

    add_test(
        NAME int_arithmetic_test
        COMMAND $<TARGET_FILE:int_arithmetic_test>
    )
endif()



if(CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
    target_compile_options(avxcpp PRIVATE -mavx -mavx2)
    if(AVX_ENABLE_TESTS)
        target_compile_options(int_arithmetic_test PRIVATE -mavx -mavx2)
    endif()
elseif(MSVC)
    target_compile_options(avxcpp PRIVATE /arch:AVX /arch:AVX2)
    if(AVX_ENABLE_TESTS)
        target_compile_options(int_arithmetic_test PRIVATE /arch:AVX /arch:AVX2)
    endif()
endif()

