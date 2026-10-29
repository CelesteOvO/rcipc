# 第三方依赖。配置（cmake -S . -B build）时自动下载到 build/_deps/，不需要事先安装。
# URL_HASH 是下载包的 SHA256，用来确认下载的文件没有被改过；换版本时要一起更新。
include(FetchContent)

# Eigen 3.4.0：线性代数库（矩阵、向量、分解），只有头文件，不需要编译。
# SOURCE_SUBDIR 故意指向一个不存在的目录，这样只下载、不运行 Eigen 自己的 CMake
# （否则会带进它的测试、文档和安装规则）。然后我们自己定义一个只含头文件路径的目标 Eigen3::Eigen。
FetchContent_Declare(eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  SOURCE_SUBDIR no-cmake
)
FetchContent_MakeAvailable(eigen)
add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})  # SYSTEM：不报 Eigen 内部的警告
add_library(Eigen3::Eigen ALIAS eigen)

# GoogleTest 1.15.2：单元测试框架。提供 TEST、EXPECT_NEAR 等宏，以及目标 GTest::gtest_main。
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  URL_HASH SHA256=f179ec217f9b3b3f3c6e8b02d3e7eda997b49e4ce26d6b235c9053bec9c0bf9f
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # MSVC：和我们的代码用同一种 C 运行库，否则链接报错
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)          # 不需要安装
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)            # 不需要 gmock
FetchContent_MakeAvailable(googletest)
