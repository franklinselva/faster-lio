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

# Detect C++17 parallel execution support (requires <execution> header + TBB backend on GCC)
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES TBB::tbb)
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
" HAS_STD_EXECUTION)
unset(CMAKE_REQUIRED_LIBRARIES)
unset(CMAKE_REQUIRED_FLAGS)

if(HAS_STD_EXECUTION)
    message(STATUS "C++17 parallel execution policies: ENABLED (via TBB)")
else()
    message(STATUS "C++17 parallel execution policies: DISABLED (sequential fallback)")
endif()

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
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
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
endif()

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
endif()
