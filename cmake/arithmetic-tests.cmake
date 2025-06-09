file(GLOB ARITHMETIC_TEST_SOURCES 
    "${SOURCE_DIR}/tests/arithmetic/*.cpp"
)

foreach(ARITHMETIC_TEST IN LISTS ARITHMETIC_TEST_SOURCES)
 
    string(REGEX REPLACE "^.+/" ""  TEST_NAME ${ARITHMETIC_TEST})
    string(REGEX REPLACE ".cpp$" ""  TEST_NAME ${TEST_NAME})

    add_executable(${TEST_NAME} ${ARITHMETIC_TEST})
    target_include_directories(${TEST_NAME} PRIVATE ${UTILS_INCLUDE_DIR})
    target_compile_options(${TEST_NAME} PRIVATE ${OPTIMIZATION_OPTIONS})
    target_link_libraries(${TEST_NAME} PRIVATE ${TESTS_LINKED_LIBS})
    
    add_test(
        NAME ${TEST_NAME}
        COMMAND $<TARGET_FILE:${TEST_NAME}>
    )

endforeach()