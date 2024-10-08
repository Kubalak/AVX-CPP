cmake_minimum_required(VERSION 3.20)
project(AVX-CPP VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)

set(SOURCE_DIR 
    "${CMAKE_SOURCE_DIR}/src"
)

set(UTILS_INCLUDE_DIR "${SOURCE_DIR}/tests/include")

# TODO: Fix selecting build type.

file(
    GLOB_RECURSE SOURCES "${SOURCE_DIR}/types/*.cpp" "${SOURCE_DIR}/types/*.hpp" "${SOURCE_DIR}/ops/*.cpp" "${SOURCE_DIR}/ops/*.hpp"
)

include(${CMAKE_SOURCE_DIR}/cmake/avx-detect.cmake)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
    set(OPTIMIZATION_OPTIONS -mavx -mavx2)
elseif(MSVC)
    set(OPTIMIZATION_OPTIONS /arch:AVX /arch:AVX2)
endif()

if(AVX512_SUPPORTED)
    set(OPTIMIZATION_OPTIONS ${OPTIMIZATION_OPTIONS} ${AVX512_FLAGS})
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_TESTING "Build tests" OFF)

add_library(avxcpp_s STATIC ${SOURCES})

if(BUILD_SHARED_LIBS)
    add_library(avxcpp SHARED ${SOURCES})
    set_target_properties(avxcpp PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
    target_include_directories(avxcpp PUBLIC ${SOURCE_DIR})
endif()

set_target_properties(avxcpp_s PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
target_include_directories(avxcpp_s PUBLIC ${SOURCE_DIR})

if(BUILD_TESTING)
    add_executable(int_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/int_arithmetic_test.cpp")
    add_executable(uint_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/uint_arithmetic_test.cpp")
    add_executable(long_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/long_arithmetic_test.cpp")
    add_executable(ulong_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/ulong_arithmetic_test.cpp")
    add_executable(short_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/short_arithmetic_test.cpp")
    add_executable(ushort_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/ushort_arithmetic_test.cpp")
    add_executable(char_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/char_arithmetic_test.cpp")
    add_executable(int_perf_tests "${SOURCE_DIR}/tests/perf/int_perf_tests.cpp")
    add_executable(perf_tests "${SOURCE_DIR}/tests/perf/new_perf_tests.cpp")

    target_include_directories(int_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(uint_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(long_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(ulong_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(short_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(ushort_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(char_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})

    target_link_libraries(int_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(uint_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(long_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(ulong_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(short_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(ushort_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(char_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(int_perf_tests PRIVATE avxcpp_s)
    target_link_libraries(perf_tests PRIVATE avxcpp_s)
    include(CTest)

    add_test(
        NAME int_arithmetic_test
        COMMAND $<TARGET_FILE:int_arithmetic_test>
    )
    add_test(
        NAME uint_arithmetic_test
        COMMAND $<TARGET_FILE:uint_arithmetic_test>
    )

    add_test(
        NAME long_arithmetic_test
        COMMAND $<TARGET_FILE:long_arithmetic_test>
    )

    add_test(
        NAME ulong_arithmetic_test
        COMMAND $<TARGET_FILE:ulong_arithmetic_test>
    )

    add_test(
        NAME short_arithmetic_test
        COMMAND $<TARGET_FILE:short_arithmetic_test>
    )

    add_test(
        NAME ushort_arithmetic_test
        COMMAND $<TARGET_FILE:ushort_arithmetic_test>
    )

    add_test(
        NAME char_arithmetic_test
        COMMAND $<TARGET_FILE:char_arithmetic_test>
    )

    add_test(
        NAME int_perf_tests
        COMMAND $<TARGET_FILE:int_perf_tests> 1000000
    )

    add_test(
        NAME perf_tests
        COMMAND $<TARGET_FILE:perf_tests>
    )
endif()

target_compile_options(avxcpp_s PRIVATE ${OPTIMIZATION_OPTIONS})

if(BUILD_SHARED_LIBS)
    target_compile_options(avxcpp PRIVATE ${OPTIMIZATION_OPTIONS})
endif()

if(BUILD_TESTING)
    target_compile_options(int_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
    target_compile_options(uint_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
    target_compile_options(long_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
    target_compile_options(ulong_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
    target_compile_options(short_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
    target_compile_options(ushort_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
    target_compile_options(char_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
    target_compile_options(int_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
    if(CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
        target_compile_options(perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
    elseif(MSVC)
        target_compile_options(perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
    endif()
endif()

