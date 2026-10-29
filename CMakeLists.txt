cmake_minimum_required(VERSION 3.24)
project(rcipc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # /utf-8: sources are UTF-8 (the default code page here would warn on any non-ASCII byte).
  add_compile_options(/utf-8 /W4 /permissive-)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)  # fopen and friends are fine here
else()
  add_compile_options(-Wall -Wextra)
endif()

include(cmake/Dependencies.cmake)

# The library: everything later stages reuse.
add_library(rcipc
  src/barrier.cc
)
target_include_directories(rcipc PUBLIC include)
target_link_libraries(rcipc PUBLIC Eigen3::Eigen)

# Stage 1: one-dimensional IPC (a ball dropped on the ground). A toy, not part of the library.
add_library(ipc1d toys/ipc1d/ipc1d.cc)
target_include_directories(ipc1d PUBLIC toys)
target_link_libraries(ipc1d PUBLIC rcipc)
add_executable(ipc1d_demo toys/ipc1d/main.cc)
target_link_libraries(ipc1d_demo PRIVATE ipc1d)

enable_testing()
add_subdirectory(tests)
