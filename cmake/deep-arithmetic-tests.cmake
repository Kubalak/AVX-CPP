set(ZIPLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external-libs/ziplib/include")
set(ZIPLIB_LIB_DIR "${CMAKE_SOURCE_DIR}/external-libs/ziplib/lib")

add_executable(deep_long_test "${SOURCE_DIR}/tests/deep/long_test.cpp")
add_executable(deep_ulong_test "${SOURCE_DIR}/tests/deep/ulong_test.cpp")
add_executable(deep_int_test "${SOURCE_DIR}/tests/deep/int_test.cpp")
add_executable(deep_uint_test "${SOURCE_DIR}/tests/deep/uint_test.cpp")
add_executable(deep_short_test "${SOURCE_DIR}/tests/deep/short_test.cpp")
add_executable(deep_ushort_test "${SOURCE_DIR}/tests/deep/ushort_test.cpp")
add_executable(deep_char_test "${SOURCE_DIR}/tests/deep/char_test.cpp")
add_executable(deep_uchar_test "${SOURCE_DIR}/tests/deep/uchar_test.cpp")

target_include_directories(deep_long_test PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_long_test PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

target_include_directories(deep_ulong_test PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_ulong_test PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

target_include_directories(deep_int_test PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_int_test PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

target_include_directories(deep_uint_test PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_uint_test PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

target_include_directories(deep_short_test PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_short_test PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

target_include_directories(deep_ushort_test PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_ushort_test PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

target_include_directories(deep_char_test PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_char_test PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

target_include_directories(deep_uchar_test PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_uchar_test PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

add_test(
    NAME deep_long_test
    COMMAND $<TARGET_FILE:deep_long_test>
)

add_test(
    NAME deep_ulong_test
    COMMAND $<TARGET_FILE:deep_ulong_test>
)

add_test(
    NAME deep_int_test
    COMMAND $<TARGET_FILE:deep_int_test>
)

add_test(
    NAME deep_uint_test
    COMMAND $<TARGET_FILE:deep_uint_test>
)

add_test(
    NAME deep_short_test
    COMMAND $<TARGET_FILE:deep_short_test>
)

add_test(
    NAME deep_ushort_test
    COMMAND $<TARGET_FILE:deep_ushort_test>
)

add_test(
    NAME deep_char_test
    COMMAND $<TARGET_FILE:deep_char_test>
)

add_test(
    NAME deep_uchar_test
    COMMAND $<TARGET_FILE:deep_uchar_test>
)

target_compile_options(deep_long_test PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
target_compile_options(deep_ulong_test PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
target_compile_options(deep_int_test PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
target_compile_options(deep_uint_test PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
target_compile_options(deep_short_test PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
target_compile_options(deep_ushort_test PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
target_compile_options(deep_char_test PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
target_compile_options(deep_uchar_test PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
