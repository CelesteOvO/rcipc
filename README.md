# rcipc

在关节体的关节坐标里做 IPC（RC-IPC），从头一个阶段一个阶段地重写。
路线和约定见 [notes/00_roadmap.md](notes/00_roadmap.md)，每个阶段有自己的笔记。

## 编译和测试

需要 CMake 3.24 以上和支持 C++17 的编译器（在 MSVC 2022 上测过）。Eigen、GoogleTest 和 MuJoCo 在配置时自动下载（MuJoCo 用的是官方编译好的 Windows 版）。

```
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

场景程序用 Release 构建会快很多（同一个 build 目录）：

```
cmake --build build --config Release --target scene_demo
build\Release\scene_demo.exe model=ragdoll_pile.xml T=2 check=1 out=pile.csv
python tools/replay.py models/ragdoll_pile.xml pile.csv
```

## 目录

| 目录 | 内容 |
|---|---|
| `include/rcipc/`、`src/` | 库 `rcipc`：后面各阶段都会复用的代码（障碍函数、四元数、模型、运动学、惯性项、接触、CCD、时间步） |
| `toys/` | 演示程序和练习用的小模型（阶段 1：`toys/ipc1d/`，小球落地；阶段 3：`toys/dynamics/`，双摆和旋转的盒子；阶段 4、5、7、8：`toys/contact/`，落地的盒子、甩过地面的杆、堆叠、台阶、斜面；阶段 6：`toys/curvature/`，逐次迭代看间隙；阶段 10：`toys/scenes/`，跑任意场景并写出轨迹） |
| `tests/` | GoogleTest 测试，一个主题一个文件 |
| `models/` | MJCF 模型（阶段 2 起），可以用 MuJoCo 的 `simulate.exe` 打开看 |
| `notes/` | 推导和练习 |
| `tools/` | Python 脚本：画图、生成布娃娃堆、在 MuJoCo 窗口里回放轨迹（只用来看，不参与计算） |
| `cmake/` | 依赖的下载（`Dependencies.cmake`） |
| `build/` | 编译输出，不进 git |
