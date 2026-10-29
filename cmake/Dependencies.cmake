# Third-party code, downloaded at configure time into build/_deps.
include(FetchContent)

# Eigen 3.4.0 (header only). SOURCE_SUBDIR points at a directory that does not exist, so
# MakeAvailable only downloads it and does not run Eigen's own CMake (its tests, docs, install rules).
FetchContent_Declare(eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  SOURCE_SUBDIR no-cmake
)
FetchContent_MakeAvailable(eigen)
add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
add_library(Eigen3::Eigen ALIAS eigen)

# GoogleTest 1.15.2.
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  URL_HASH SHA256=f179ec217f9b3b3f3c6e8b02d3e7eda997b49e4ce26d6b235c9053bec9c0bf9f
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # MSVC: same runtime library as our targets
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
