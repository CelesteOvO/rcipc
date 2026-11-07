# 顶层构建文件。用法见 README.md：
#   cmake -S . -B build                      配置（第一次会下载依赖）
#   cmake --build build --config Debug       编译
#   ctest --test-dir build -C Debug          跑测试
cmake_minimum_required(VERSION 3.24)
project(rcipc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # /utf-8：源文件按 UTF-8 读（注释里有中文；不加的话 MSVC 按 GBK 读，会报警告甚至读错）。
  # /W4：较高的警告级别。/permissive-：按标准 C++ 严格检查。
  # 只给 C++ 源文件（阶段 11 的 .cu 文件由 nvcc 编译，它不认这些选项，见下面的 CUDA 部分）。
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
  # /bigobj：contact.cc 里固定大小的 Eigen 模板（12 x 12 的旋量块、特征分解）在 Debug 下节数超过 .obj 的默认上限。
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)  # 允许用 fopen 等函数，不必换成 fopen_s
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

# 第三方依赖：Eigen、GoogleTest、MuJoCo。
include(cmake/Dependencies.cmake)

# 库 rcipc：后面各阶段都会复用的代码。头文件在 include/rcipc/，实现在 src/。
# 每个阶段往这里加源文件（阶段 4 会加接触，……）。
add_library(rcipc
  src/barrier.cc     # 阶段 1：障碍函数
  src/quat.cc        # 阶段 2：四元数的 exp 和 log
  src/model.cc       # 阶段 2：从 MuJoCo 读出树结构
  src/kinematics.cc  # 阶段 2：正运动学、q 的加减、点的 Jacobian
  src/inertia.cc     # 阶段 3：body 形式的惯性项
  src/step.cc        # 阶段 3：关节坐标里的时间步（阶段 4 加上接触和 CCD）
  src/contact.cc     # 阶段 4：地面接触：距离、接触行、能量、梯度、Hessian、kappa（阶段 6 加曲率上界）
  src/ccd.cc         # 阶段 4：直弦 CCD；阶段 5：弦加矢高
  src/chain.cc       # 阶段 5：链上的关节和静态力臂
  src/distance.cc    # 阶段 7：点–三角形、线段–线段的最近点
  src/friction.cc    # 阶段 8：滞后的光滑化 Coulomb 摩擦
  src/solver.cc      # 阶段 9：PCG 和树块预条件器
  src/limits.cc      # 阶段 10：关节限位
  src/drive.cc       # 阶段 12：驱动、关节阻尼、armature
)
target_include_directories(rcipc PUBLIC include)
target_link_libraries(rcipc PUBLIC Eigen3::Eigen mujoco::mujoco)
# 模型文件（MJCF）所在的目录，测试和演示程序用 RCIPC_MODEL_DIR "/zoo.xml" 这样的写法找到它们。
target_compile_definitions(rcipc PUBLIC RCIPC_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")

# 阶段 1：一维 IPC（小球落地）。它是一个练习用的小模型，不属于库。
#   库 ipc1d：被测试和演示程序共用；
#   可执行文件 ipc1d_demo：演示程序。
add_library(ipc1d toys/ipc1d/ipc1d.cc)
target_include_directories(ipc1d PUBLIC toys)
target_link_libraries(ipc1d PUBLIC rcipc)
add_executable(ipc1d_demo toys/ipc1d/main.cc)
target_link_libraries(ipc1d_demo PRIVATE ipc1d)
rcipc_copy_dlls(ipc1d_demo)

# 阶段 3：关节坐标里的时间步（双摆、自由旋转的盒子）。只用库 rcipc，所以没有自己的库。
add_executable(dynamics_demo toys/dynamics/main.cc)
target_link_libraries(dynamics_demo PRIVATE rcipc)
rcipc_copy_dlls(dynamics_demo)

# 阶段 4：地面接触（落地的盒子、甩过地面的杆）。
add_executable(contact_demo toys/contact/main.cc)
target_link_libraries(contact_demo PRIVATE rcipc)
rcipc_copy_dlls(contact_demo)

# 阶段 6：曲率上界（逐次迭代看间隙）。
add_executable(curvature_demo toys/curvature/main.cc)
target_link_libraries(curvature_demo PRIVATE rcipc)
rcipc_copy_dlls(curvature_demo)

# 阶段 10：跑任意场景、写出轨迹（tools/replay.py 回放）。
add_executable(scene_demo toys/scenes/main.cc)
target_link_libraries(scene_demo PRIVATE rcipc)
rcipc_copy_dlls(scene_demo)

# 阶段 11：GPU 版（CUDA）。机器上没有 CUDA 编译器时跳过，CPU 版照常编译；也可以 -DRCIPC_CUDA=OFF 关掉。
#   CMAKE_CUDA_ARCHITECTURES 默认 native：为这台机器的显卡编译（RTX 4060 Ti 是 sm_89）。
#   设备代码只在 src/gpu/*.cu 里，主机代码（stepper.cc）是普通 C++，和 Eigen、库 rcipc 一起编译。
option(RCIPC_CUDA "Build the GPU version (stage 11)" ON)
include(CheckLanguage)
check_language(CUDA)
if(RCIPC_CUDA AND CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  add_library(rcipc_gpu
    src/gpu/stepper.cc   # 主机：上传模型、Newton 循环
    src/gpu/kin.cu       # 正运动学、惯性项、关节限位
    src/gpu/contact.cu   # 接触检测、接触行、摩擦
    src/gpu/contact_full.cu  # 阶段 14：完整的接触 Hessian（12 x 12 的旋量块、特征分解成行）
    src/gpu/narrow.cu    # 检测和 CCD 共用的窄相剔除（刚体坐标系里的分组）
    src/gpu/ccd.cu       # 曲线 CCD
    src/gpu/solver.cu    # PCG 和树块预条件器
    src/gpu/pcg_coop.cu  # 协作内核版的 PCG（整个循环一个内核）
    src/gpu/iter.cu      # CUDA graph 版的 Newton 迭代里，设备上做的判断
  )
  target_link_libraries(rcipc_gpu PUBLIC rcipc CUDA::cudart)
  target_compile_definitions(rcipc_gpu PUBLIC RCIPC_HAVE_CUDA)
  # nvcc 把主机部分交给 MSVC 编译：一样要 /utf-8。
  target_compile_options(rcipc_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  # 每个线程自己的默认流（不是旧的全局默认流）：不带流参数的发射和异步拷贝都在它上面排队，
  # 这样整个 Newton 迭代可以用流捕获录成 CUDA graph（旧的默认流不能捕获）。.cu 用 nvcc 的选项，.cc 用宏。
  target_compile_options(rcipc_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
  target_compile_definitions(rcipc_gpu PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
  target_link_libraries(scene_demo PRIVATE rcipc_gpu)
endif()

# 测试，定义在 tests/CMakeLists.txt。
enable_testing()
add_subdirectory(tests)
