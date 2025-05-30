cmake_minimum_required(VERSION 3.20)
project(AVX-CPP VERSION 0.7.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)

set(SOURCE_DIR 
    "${CMAKE_SOURCE_DIR}/src"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(UTILS_INCLUDE_DIR "${SOURCE_DIR}/tests/include")

file(
    GLOB_RECURSE SOURCES "${SOURCE_DIR}/types/*.cpp" "${SOURCE_DIR}/types/*.hpp" "${SOURCE_DIR}/ops/*.cpp" "${SOURCE_DIR}/ops/*.hpp" "${SOURCE_DIR}/tests/include/*.cpp"
)

include(${CMAKE_SOURCE_DIR}/cmake/avx-detect.cmake)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
    set(OPTIMIZATION_OPTIONS -mavx -mavx2)
elseif(MSVC)
    set(OPTIMIZATION_OPTIONS /arch:AVX /arch:AVX2)
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_PERFORMANCE_TESTS "Build performance tests" OFF)
option(BUILD_USE_AVX512 "Use AVX512 when building binaries" ${AVX512_SUPPORTED})
option(BUILD_DEEP_TESTS "Build deep types tests (not recommended)" OFF)

if(BUILD_USE_AVX512)
    if(not AVX512_SUPPORTED)
        message(WARNING "AVX512 is not supported on this system. Some tests may fail on runtime.")
    endif()
    set(OPTIMIZATION_OPTIONS ${OPTIMIZATION_OPTIONS} ${AVX512_FLAGS})
endif()

add_library(avxcpp-s STATIC ${SOURCES})

if(BUILD_SHARED_LIBS)
    add_library(avxcpp SHARED ${SOURCES})
    set_target_properties(avxcpp PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
    target_include_directories(avxcpp PUBLIC ${SOURCE_DIR})
    target_compile_options(avxcpp PRIVATE ${OPTIMIZATION_OPTIONS})
endif()

set_target_properties(avxcpp-s PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
target_include_directories(avxcpp-s PUBLIC ${SOURCE_DIR})

target_compile_options(avxcpp-s PRIVATE ${OPTIMIZATION_OPTIONS})

if(BUILD_PERFORMANCE_TESTS OR BUILD_TESTING)
    include(CTest)
endif()

if(BUILD_PERFORMANCE_TESTS)
    include(${CMAKE_SOURCE_DIR}/cmake/perf-tests.cmake)
endif()

if(BUILD_TESTING)
    include(${CMAKE_SOURCE_DIR}/cmake/arithmetic-tests.cmake)
endif()

if(BUILD_DEEP_TESTS)
    include(${CMAKE_SOURCE_DIR}/cmake/deep-arithmetic-tests.cmake)
endif()


set_target_properties(avxcpp-s
    PROPERTIES 
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
)

if(BUILD_SHARED_LIBS)
    set_target_properties(avxcpp
        PROPERTIES 
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
    )
endif()

