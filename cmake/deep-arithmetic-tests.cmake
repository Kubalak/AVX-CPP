set(ZIPLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external-libs/ziplib/include")
set(ZIPLIB_LIB_DIR "${CMAKE_SOURCE_DIR}/external-libs/ziplib/lib")

add_executable(deep_test_int "${SOURCE_DIR}/tests/deep/int_test.cpp")

target_include_directories(deep_test_int PRIVATE ${UTILS_INCLUDE_DIR} ${ZIPLIB_INCLUDE_DIR})
target_link_libraries(deep_test_int PRIVATE avxcpp-s ${ZIPLIB_LIB_DIR}/libzip.a)

add_test(
    NAME deep_test_int
    COMMAND $<TARGET_FILE:deep_test_int>
)

target_compile_options(deep_test_int PRIVATE ${OPTIMIZATION_OPTIONS} -lnsl)
