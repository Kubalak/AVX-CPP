cmake_minimum_required(VERSION 3.20)
project(AVX-CPP VERSION 0.3.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)

set(SOURCE_DIR 
    "${CMAKE_SOURCE_DIR}/src"
)

set(UTILS_INCLUDE_DIR "${SOURCE_DIR}/tests/include")

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

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_PERFORMANCE_TESTS "Build performance tests" OFF)

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
    add_executable(uchar_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/uchar_arithmetic_test.cpp")

    if(BUILD_PERFORMANCE_TESTS)
        add_executable(long_perf_tests "${SOURCE_DIR}/tests/perf/long_perf_tests.cpp")
        add_executable(ulong_perf_tests "${SOURCE_DIR}/tests/perf/ulong_perf_tests.cpp")
        add_executable(int_perf_tests "${SOURCE_DIR}/tests/perf/int_perf_tests.cpp")
        add_executable(uint_perf_tests "${SOURCE_DIR}/tests/perf/uint_perf_tests.cpp")
        add_executable(short_perf_tests "${SOURCE_DIR}/tests/perf/short_perf_tests.cpp")
        add_executable(ushort_perf_tests "${SOURCE_DIR}/tests/perf/ushort_perf_tests.cpp")
        add_executable(char_perf_tests "${SOURCE_DIR}/tests/perf/char_perf_tests.cpp")
        add_executable(uchar_perf_tests "${SOURCE_DIR}/tests/perf/uchar_perf_tests.cpp")

        target_include_directories(long_perf_tests PRIVATE ${UTILS_INCLUDE_DIR})
        target_include_directories(ulong_perf_tests PRIVATE ${UTILS_INCLUDE_DIR})
        target_include_directories(int_perf_tests PRIVATE ${UTILS_INCLUDE_DIR})
        target_include_directories(uint_perf_tests PRIVATE ${UTILS_INCLUDE_DIR})
        target_include_directories(short_perf_tests PRIVATE ${UTILS_INCLUDE_DIR})
        target_include_directories(ushort_perf_tests PRIVATE ${UTILS_INCLUDE_DIR})
        target_include_directories(char_perf_tests PRIVATE ${UTILS_INCLUDE_DIR})
        target_include_directories(uchar_perf_tests PRIVATE ${UTILS_INCLUDE_DIR})

        target_link_libraries(long_perf_tests PRIVATE avxcpp_s)
        target_link_libraries(ulong_perf_tests PRIVATE avxcpp_s)
        target_link_libraries(int_perf_tests PRIVATE avxcpp_s)
        target_link_libraries(uint_perf_tests PRIVATE avxcpp_s)
        target_link_libraries(short_perf_tests PRIVATE avxcpp_s)
        target_link_libraries(ushort_perf_tests PRIVATE avxcpp_s)
        target_link_libraries(char_perf_tests PRIVATE avxcpp_s)
        target_link_libraries(uchar_perf_tests PRIVATE avxcpp_s)

        add_test(
            NAME long_perf_tests
            COMMAND $<TARGET_FILE:long_perf_tests>
        )

        add_test(
            NAME ulong_perf_tests
            COMMAND $<TARGET_FILE:ulong_perf_tests>
        )
        
        add_test(
            NAME int_perf_tests
            COMMAND $<TARGET_FILE:int_perf_tests>
        )

        add_test(
            NAME uint_perf_tests
            COMMAND $<TARGET_FILE:uint_perf_tests>
        )

        add_test(
            NAME short_perf_tests
            COMMAND $<TARGET_FILE:short_perf_tests>
        )

        add_test(
            NAME ushort_perf_tests
            COMMAND $<TARGET_FILE:ushort_perf_tests>
        )

        add_test(
            NAME char_perf_tests
            COMMAND $<TARGET_FILE:char_perf_tests>
        )

        add_test(
            NAME uchar_perf_tests
            COMMAND $<TARGET_FILE:uchar_perf_tests>
        )
    endif()

    target_include_directories(int_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(uint_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(long_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(ulong_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(short_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(ushort_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(char_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
    target_include_directories(uchar_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})


    target_link_libraries(int_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(uint_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(long_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(ulong_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(short_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(ushort_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(char_arithmetic_test PRIVATE avxcpp_s)
    target_link_libraries(uchar_arithmetic_test PRIVATE avxcpp_s)

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
        NAME uchar_arithmetic_test
        COMMAND $<TARGET_FILE:uchar_arithmetic_test>
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
    target_compile_options(uchar_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})

    if(BUILD_PERFORMANCE_TESTS)
        if(CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
            target_compile_options(long_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
            target_compile_options(ulong_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
            target_compile_options(int_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
            target_compile_options(uint_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
            target_compile_options(short_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
            target_compile_options(ushort_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
            target_compile_options(char_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
            target_compile_options(uchar_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize)
        elseif(MSVC)
            target_compile_options(ulong_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
            target_compile_options(long_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
            target_compile_options(int_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
            target_compile_options(uint_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
            target_compile_options(short_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
            target_compile_options(ushort_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
            target_compile_options(char_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
            target_compile_options(uchar_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS})
        endif()
    endif()
endif()

