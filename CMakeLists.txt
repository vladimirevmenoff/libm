cmake_minimum_required(VERSION 3.12)
project(libm_long_fixed_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Paths
set(PROJECT_ROOT        ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_INCLUDE_DIR ${PROJECT_ROOT}/include)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MPFR REQUIRED mpfr)
pkg_check_modules(GMP  REQUIRED gmp)

add_executable(test_long_fixed
    tests/test_long_fixed.cpp
)

target_include_directories(test_long_fixed PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_ROOT}
    ${MPFR_INCLUDE_DIRS}
    ${GMP_INCLUDE_DIRS}
)

target_link_directories(test_long_fixed PRIVATE
    ${MPFR_LIBRARY_DIRS}
    ${GMP_LIBRARY_DIRS}
)

target_link_libraries(test_long_fixed PRIVATE
    ${MPFR_LIBRARIES}
    ${GMP_LIBRARIES}
)


add_executable(test_long_fixed_quad
    tests/test_long_fixed_quad.cpp
)

target_include_directories(test_long_fixed_quad PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_ROOT}
)

target_link_libraries(test_long_fixed_quad PRIVATE quadmath)

