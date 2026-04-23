cmake_minimum_required(VERSION 3.14)
project(faster_lio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(WITH_IVOX_NODE_TYPE_PHC "Use PHC instead of default ivox node" OFF)
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_TESTS "Build unit tests" ON)

# Per-frame diagnostics CSV. Off in Release builds — runtime cost is small
# but unnecessary in production. Enable explicitly for algorithm debugging.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(_DIAG_DEFAULT OFF)
else()
    set(_DIAG_DEFAULT ON)
endif()
option(FASTER_LIO_ENABLE_DIAGNOSTICS
       "Enable per-frame diagnostic CSV writer (overhead; debug-only)"
       ${_DIAG_DEFAULT})


# Dependencies
include(FetchContent)

# Use system spdlog if available, otherwise fetch from source
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
  FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG v1.15.1
  )
  FetchContent_MakeAvailable(spdlog)
endif()

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters)
find_package(yaml-cpp REQUIRED)

# yaml-cpp: older packages export 'yaml-cpp' target, newer export 'yaml-cpp::yaml-cpp'
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp found but no known target (yaml-cpp or yaml-cpp::yaml-cpp)")
endif()

# TBB: required for iVox data structure, also enables C++17 parallel execution on Linux/GCC
find_package(TBB REQUIRED)

# Detect C++17 parallel execution support.
#   - GCC libstdc++: works out of the box with -ltbb
#   - LLVM libc++ (since 17): needs -fexperimental-library to enable parallel STL
#   - Apple Clang: no <execution> support, falls back to sequential
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES TBB::tbb)

# First try plain (works for GCC/libstdc++)
set(CMAKE_REQUIRED_FLAGS "-std=c++17")
check_cxx_source_compiles("
    #include <execution>
    #include <algorithm>
    #include <vector>
    int main() {
        std::vector<int> v{1,2,3};
        std::for_each(std::execution::par_unseq, v.begin(), v.end(), [](int&){});
        return 0;
    }
" HAS_STD_EXECUTION_PLAIN)

set(_PAR_EXTRA_FLAGS "")
if(HAS_STD_EXECUTION_PLAIN)
    set(HAS_STD_EXECUTION TRUE)
else()
    # Retry with libc++ experimental library flag
    set(CMAKE_REQUIRED_FLAGS "-std=c++17 -fexperimental-library")
    check_cxx_source_compiles("
        #include <execution>
        #include <algorithm>
        #include <vector>
        int main() {
            std::vector<int> v{1,2,3};
            std::for_each(std::execution::par_unseq, v.begin(), v.end(), [](int&){});
            return 0;
        }
    " HAS_STD_EXECUTION_EXPERIMENTAL)
    if(HAS_STD_EXECUTION_EXPERIMENTAL)
        set(HAS_STD_EXECUTION TRUE)
        set(_PAR_EXTRA_FLAGS "-fexperimental-library")
    endif()
endif()
unset(CMAKE_REQUIRED_LIBRARIES)
unset(CMAKE_REQUIRED_FLAGS)

if(HAS_STD_EXECUTION)
    if(_PAR_EXTRA_FLAGS)
        message(STATUS "C++17 parallel execution policies: ENABLED (libc++ experimental, ${_PAR_EXTRA_FLAGS})")
    else()
        message(STATUS "C++17 parallel execution policies: ENABLED (libstdc++ + TBB)")
    endif()
else()
    message(STATUS "C++17 parallel execution policies: DISABLED (sequential fallback)")
endif()

# g2o: pose graph optimization backend.
# When built via butterfli_find_faster_lio(), g2o targets already exist.
# For standalone builds, try find_package as a fallback.
if(NOT TARGET core)
    find_package(g2o REQUIRED)
endif()
message(STATUS "Pose graph backend (g2o): ENABLED")

# For ubuntu 18.04 with custom TBB path
if(CUSTOM_TBB_DIR)
    set(TBB2018_INCLUDE_DIR "${CUSTOM_TBB_DIR}/include")
    set(TBB2018_LIBRARY_DIR "${CUSTOM_TBB_DIR}/lib/intel64/gcc4.7")
endif()

# Library
add_library(${PROJECT_NAME}
    src/laser_mapping.cc
    src/pointcloud_preprocess.cc
    src/imu_processing.cc
    src/options.cc
    src/utils.cc
    src/pose_graph.cc
    src/loop_closer.cc
    src/scan_context.cc
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    # Rigtorp SPSCQueue — header-only, MIT-licensed, vendored under thirdparty/
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${PCL_LIBRARIES}
    TBB::tbb
    spdlog::spdlog
    ${YAML_CPP_TARGET}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
    ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/"
)

if(HAS_STD_EXECUTION)
    target_compile_definitions(${PROJECT_NAME} PUBLIC FASTER_LIO_HAS_STD_EXECUTION)
    if(_PAR_EXTRA_FLAGS)
        target_compile_options(${PROJECT_NAME} PUBLIC ${_PAR_EXTRA_FLAGS})
        target_link_options(${PROJECT_NAME} PUBLIC ${_PAR_EXTRA_FLAGS})
    endif()
endif()

if(FASTER_LIO_ENABLE_DIAGNOSTICS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC FASTER_LIO_ENABLE_DIAGNOSTICS)
    message(STATUS "Diagnostics CSV writer: ENABLED")
else()
    message(STATUS "Diagnostics CSV writer: DISABLED (Release build or explicitly off)")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC
    core stuff types_slam3d solver_eigen)

if(WITH_IVOX_NODE_TYPE_PHC)
    message(STATUS "Using IVOX_NODE_TYPE_PHC")
    target_compile_definitions(${PROJECT_NAME} PUBLIC IVOX_NODE_TYPE_PHC)
else()
    message(STATUS "Using IVOX_NODE_TYPE_DEFAULT")
endif()

if(CUSTOM_TBB_DIR)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${TBB2018_INCLUDE_DIR})
    target_link_directories(${PROJECT_NAME} PUBLIC ${TBB2018_LIBRARY_DIR})
endif()

target_compile_options(${PROJECT_NAME} PRIVATE
    -fexceptions -g -ggdb
)

# Examples
if(BUILD_EXAMPLES)
    find_package(gflags REQUIRED)

    add_executable(run_mapping examples/run_mapping.cc)
    target_link_libraries(run_mapping PRIVATE ${PROJECT_NAME} gflags)

    add_executable(evaluate_lio examples/evaluate_lio.cc)
    target_link_libraries(evaluate_lio PRIVATE ${PROJECT_NAME} gflags)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    set(ABSL_PROPAGATE_CXX_STD ON)
    FetchContent_MakeAvailable(googletest)

    function(add_faster_lio_test TEST_NAME TEST_SRC)
        add_executable(${TEST_NAME} tests/${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE ${PROJECT_NAME} GTest::gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    add_faster_lio_test(test_types test_types.cc)
    add_faster_lio_test(test_ivox test_ivox.cc)
    add_faster_lio_test(test_imu_processing test_imu_processing.cc)
    add_faster_lio_test(test_pointcloud_preprocess test_pointcloud_preprocess.cc)
    add_faster_lio_test(test_laser_mapping test_laser_mapping.cc)

    # Edge case tests for production readiness
    add_faster_lio_test(test_common_lib_edge_cases test_common_lib_edge_cases.cc)
    add_faster_lio_test(test_ivox_edge_cases test_ivox_edge_cases.cc)
    add_faster_lio_test(test_imu_processing_edge_cases test_imu_processing_edge_cases.cc)
    add_faster_lio_test(test_pointcloud_preprocess_edge_cases test_pointcloud_preprocess_edge_cases.cc)
    add_faster_lio_test(test_laser_mapping_edge_cases test_laser_mapping_edge_cases.cc)
    add_faster_lio_test(test_wheel_odom test_wheel_odom.cc)
    add_faster_lio_test(test_imu_mount test_imu_mount.cc)
    add_faster_lio_test(test_lio_drift_attribution test_lio_drift_attribution.cc)
    add_faster_lio_test(test_pose_graph test_pose_graph.cc)
    add_faster_lio_test(test_scan_context test_scan_context.cc)
endif()
