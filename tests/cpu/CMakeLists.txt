# =============================================================================
# CPU Tests using GoogleTest
# =============================================================================

add_executable(linalg_test linalg_test.cpp)
target_link_libraries(linalg_test 
    gtest
    gtest_main
    linalg
    Eigen3::Eigen
)
gtest_discover_tests(linalg_test
    TEST_PREFIX "CPU."
    PROPERTIES TIMEOUT 300
)

add_executable(linalgx_test linalgx_test.cpp)
target_link_libraries(linalgx_test 
    gtest
    gtest_main
    linalg
    Eigen3::Eigen
)
gtest_discover_tests(linalgx_test
    TEST_PREFIX "CPU."
    PROPERTIES TIMEOUT 300
)

add_executable(ldlt_solver_test ldlt_solver_test.cpp)
target_link_libraries(ldlt_solver_test 
    gtest
    gtest_main
    linalg
    Eigen3::Eigen
)
gtest_discover_tests(ldlt_solver_test
    TEST_PREFIX "CPU."
    PROPERTIES TIMEOUT 300
)

add_executable(ldlt_solverx_test ldlt_solverx_test.cpp)
target_link_libraries(ldlt_solverx_test 
    gtest
    gtest_main
    linalg
    Eigen3::Eigen
)
gtest_discover_tests(ldlt_solverx_test
    TEST_PREFIX "CPU."
    PROPERTIES TIMEOUT 300
)

add_executable(schurcomplement_solver_test schurcomplement_solver_test.cpp)
target_include_directories(schurcomplement_solver_test PRIVATE ${SUITESPARSE_INCLUDE_DIR})
target_link_libraries(schurcomplement_solver_test 
    gtest
    gtest_main
    linalg
    Eigen3::Eigen
    LAPACK::LAPACK 
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIB}
    ${SUITESPARSE_LIBS}
)
gtest_discover_tests(schurcomplement_solver_test
    TEST_PREFIX "CPU."
    PROPERTIES TIMEOUT 300
)
