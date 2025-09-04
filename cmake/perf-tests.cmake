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

target_link_libraries(long_perf_tests PRIVATE avxcpp-s)
target_link_libraries(ulong_perf_tests PRIVATE avxcpp-s)
target_link_libraries(int_perf_tests PRIVATE avxcpp-s)
target_link_libraries(uint_perf_tests PRIVATE avxcpp-s)
target_link_libraries(short_perf_tests PRIVATE avxcpp-s)
target_link_libraries(ushort_perf_tests PRIVATE avxcpp-s)
target_link_libraries(char_perf_tests PRIVATE avxcpp-s)
target_link_libraries(uchar_perf_tests PRIVATE avxcpp-s)

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

if(CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
    target_compile_options(long_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize -fno-tree-slp-vectorize)
    target_compile_options(ulong_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize -fno-tree-slp-vectorize)
    target_compile_options(int_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize -fno-tree-slp-vectorize)
    target_compile_options(uint_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize -fno-tree-slp-vectorize)
    target_compile_options(short_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize -fno-tree-slp-vectorize)
    target_compile_options(ushort_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize -fno-tree-slp-vectorize)
    target_compile_options(char_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize -fno-tree-slp-vectorize)
    target_compile_options(uchar_perf_tests PRIVATE ${OPTIMIZATION_OPTIONS} -fno-tree-vectorize -fno-tree-slp-vectorize)
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