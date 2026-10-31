# 第三方依赖：Eigen、GoogleTest、MuJoCo。配置（cmake -S . -B build）时自动下载到 build/_deps/，不需要事先安装。
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

# MuJoCo 3.14.0（阶段 2 起）：只用来读 MJCF 文件，并作为正运动学、Jacobian 和质量矩阵的参照答案。
# 下载的是官方编译好的 Windows 版本（include/、lib/mujoco.lib、bin/mujoco.dll），不用自己编译。
# 包里的 bin/simulate.exe 可以直接打开 models/ 下的模型看一看。
# 同样用 SOURCE_SUBDIR 跳过包里的 CMake，然后自己定义导入目标 mujoco::mujoco。
if(NOT WIN32)
  message(FATAL_ERROR "cmake/Dependencies.cmake 现在只下载 Windows 版的 MuJoCo；其他平台请改 URL")
endif()
FetchContent_Declare(mujoco
  URL https://github.com/google-deepmind/mujoco/releases/download/3.14.0/mujoco-3.14.0-windows-x86_64.zip
  URL_HASH SHA256=d2a786e0ea48667cee60b7eb3c50cc3addd7ae0f22137d71d66ca52888e4aaf0
  SOURCE_SUBDIR no-cmake
)
FetchContent_MakeAvailable(mujoco)
add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
set_target_properties(mujoco::mujoco PROPERTIES
  IMPORTED_LOCATION ${mujoco_SOURCE_DIR}/bin/mujoco.dll   # 运行时加载的 DLL
  IMPORTED_IMPLIB ${mujoco_SOURCE_DIR}/lib/mujoco.lib     # 链接时用的导入库
  INTERFACE_INCLUDE_DIRECTORIES ${mujoco_SOURCE_DIR}/include
)

# rcipc_copy_dlls(目标)：编译完以后把目标用到的 DLL（现在只有 mujoco.dll）复制到 .exe 旁边。
# Windows 运行 .exe 时在它自己的目录里找 DLL；不复制的话一运行就报“找不到 mujoco.dll”。
function(rcipc_copy_dlls target)
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
    COMMAND_EXPAND_LISTS
  )
endfunction()
