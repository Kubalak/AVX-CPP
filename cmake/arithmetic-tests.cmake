add_executable(int_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/int_arithmetic_test.cpp")
add_executable(uint_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/uint_arithmetic_test.cpp")
add_executable(long_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/long_arithmetic_test.cpp")
add_executable(ulong_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/ulong_arithmetic_test.cpp")
add_executable(short_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/short_arithmetic_test.cpp")
add_executable(ushort_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/ushort_arithmetic_test.cpp")
add_executable(char_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/char_arithmetic_test.cpp")
add_executable(uchar_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/uchar_arithmetic_test.cpp")
add_executable(fp_arithmetic_test "${SOURCE_DIR}/tests/arithmetic/fp_arithmetic_test.cpp")

target_include_directories(int_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
target_include_directories(uint_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
target_include_directories(long_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
target_include_directories(ulong_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
target_include_directories(short_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
target_include_directories(ushort_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
target_include_directories(char_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
target_include_directories(uchar_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})
target_include_directories(fp_arithmetic_test PRIVATE ${UTILS_INCLUDE_DIR})

target_link_libraries(int_arithmetic_test PRIVATE avxcpp-s)
target_link_libraries(uint_arithmetic_test PRIVATE avxcpp-s)
target_link_libraries(long_arithmetic_test PRIVATE avxcpp-s)
target_link_libraries(ulong_arithmetic_test PRIVATE avxcpp-s)
target_link_libraries(short_arithmetic_test PRIVATE avxcpp-s)
target_link_libraries(ushort_arithmetic_test PRIVATE avxcpp-s)
target_link_libraries(char_arithmetic_test PRIVATE avxcpp-s)
target_link_libraries(uchar_arithmetic_test PRIVATE avxcpp-s)
target_link_libraries(fp_arithmetic_test PRIVATE avxcpp-s)

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

add_test(
    NAME fp_arithmetic_test
    COMMAND $<TARGET_FILE:fp_arithmetic_test>
)

target_compile_options(int_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(uint_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(long_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(ulong_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(short_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(ushort_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(char_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(uchar_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})
target_compile_options(fp_arithmetic_test PRIVATE ${OPTIMIZATION_OPTIONS})