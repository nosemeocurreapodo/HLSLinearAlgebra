cmake_minimum_required(VERSION 3.14)

project(HLSLinearAlgebra 
    VERSION 1.0.0
    DESCRIPTION "HLS Linear Algebra Library with Posit and FloatX number formats"
    LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build CPU tests using GoogleTest" ON)
option(BUILD_HLS_TESTS "Build HLS/FPGA tests using Vitis" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)

include(FetchContent)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Found Eigen3: ${Eigen3_DIR}")

# Create header-only library for unified linear algebra (CPU + FPGA compatible)
add_library(linalg INTERFACE)
target_include_directories(linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(linalg INTERFACE cxx_std_17)

# For HLS builds, add synthesis-specific definitions
#if(BUILD_HLS_TESTS)
#    target_compile_definitions(linalg INTERFACE __SYNTHESIS__)
#endif()

# Create header-only library for HLS numerical formats (FloatX, Posit)
add_library(hls_numerics INTERFACE)
target_include_directories(hls_numerics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(hls_numerics INTERFACE cxx_std_17)
target_compile_definitions(hls_numerics INTERFACE 
    $<$<NOT:$<BOOL:${BUILD_HLS_TESTS}>>:HLS_NO_XIL_FPO_LIB>
)

# GoogleTest setup for CPU tests
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1)
    
    if(WIN32)
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(googletest)
    
    include(GoogleTest)
    enable_testing()

    # Add subdirectories
    add_subdirectory(tests)
endif()


# =============================================================================
# Installation (Optional)
# =============================================================================
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    install(TARGETS linalg hls_numerics
        EXPORT linalgTargets
    )
    
    install(FILES
        include/linalg/linalg.h
        include/linalg/common.h
        include/linalg/shift_registers.h
        DESTINATION include/linalg
    )
    
    install(FILES
        include/hls_numerics/FloatX.h
        include/hls_numerics/Posit.h
        include/hls_numerics/common.h
        include/hls_numerics/hls_compat_ultra_simple.h
        DESTINATION include/hls_numerics
    )
endif()

