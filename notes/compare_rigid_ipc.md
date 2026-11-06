# 和 Rigid-IPC 对比

**目的：** Rigid-IPC（Ferguson 等，SIGGRAPH 2021，`ipc-sim/rigid-ipc`）是"刚体 + IPC + 保证不穿透"的 CPU 参考实现。
它和我们回答同一个问题（刚体接触，每一步都不穿透），差别只在未知量：
- Rigid-IPC：每个刚体 6 个自由度（质心位置 + 转动向量），**没有关节**。机构靠接触传力（齿轮靠齿、锁链靠环环相扣），
  能表达的约束只有 `is_dof_fixed`（把 6 个分量里的几个钉在世界坐标轴上）和运动学体（给定角速度，用增广拉格朗日压上去）；
- 我们：关节坐标 $q$，关节是坐标本身，刚体由正运动学生成。

所以这一篇比的是：**同样的接触几何、同样的 $\hat d$、同样的步长，两边各要多少次 Newton、每步多长时间**；以及用接触当关节的代价。
布娃娃、G1 这类带刚体间关节的场景 Rigid-IPC 搭不出来，不比。

脚本和原始结果都在 `compare/rigid_ipc/`。Rigid-IPC 的源码和网格在 `D:\Project\rigid-ipc`（不进仓库），
按它的 README 用 CMake 构建，MSVC 2022，关掉 GUI、测试和 tools，只编 `rigid_ipc_sim`。Windows 上要多给两样：
`/utf-8`（它的源码里有 UTF-8 注释，中文代码页下 MSVC 会解析错）和 `-p:VcpkgEnabled=false`（否则 VS 全局的
vcpkg 会塞进另一个版本的 fmt）；依赖走 CPM，`ipc-toolkit` 钉在 d41d6ca，全历史 clone 太慢，浅取那一个提交
放进 CPM 缓存就行。

---

## 1. 两边的参数怎么对应

| | Rigid-IPC（`sim.json` 的 `args`） | 我们（`scene_demo`） |
|---|---|---|
| 时间积分 | `time_stepper: default` = 隐式 Euler | 隐式 Euler |
| 障碍距离 | `initial_barrier_activation_distance`，场景里 1e-3 | `dhat=1e-3` |
| 障碍刚度 | IPC 原版的 $\kappa$ 初值和自适应（`min_barrier_stiffness_scale` 1e11） | 阶段 9 的 $\kappa$ 初值和调整（`adapt=1`） |
| CCD | `trajectory_type: piecewise_linear`（默认）：刚体螺旋轨迹分段线性化，Tight-Inclusion 做线性 CCD | 弦加矢高的曲线 CCD，每个刚体的加速度界 |
| Newton 停止 | `velocity`：这次 Newton 修正让顶点移动的最大速度 ≤ `velocity_conv_tol` × 场景包围盒对角线，默认 1e-2（**相对量**；`newton_solver.cpp` 第 105 行） | 同一个量（这次修正的最大顶点位移 / h）≤ `tol`，默认 1e-6 m/s（绝对量；`step.cc` 第 124 行） |
| 线性求解 | Eigen SimplicialLDLT（直接法） | 树块预条件的 PCG（CPU）或 GPU 上的协作 PCG |
| 摩擦 | `coefficient_friction`，滞后 1 次（`friction_constraints.iterations: 1`） | `mu=`，滞后的光滑摩擦 |
| 运动学体 | 增广拉格朗日，罚函数 1e3 翻倍到 1e8 | 那个铰链上的 velocity 驱动，kv = 1e4 |

Newton 的停止条件差得最多：齿轮场景的包围盒对角线约 4 m，Rigid-IPC 默认的容差就是 0.04 m/s。
对比时两边都报两组：各自的默认，和 Rigid-IPC 改成绝对的 1e-6 m/s（`--patch`）。

**Newton 的计数也不一样。** Rigid-IPC 的 `total_iter` 是真的走了几步：第 k 次迭代算出的方向够小就停，
这次不算（`newton_solver.cpp` 第 215 行；日志里 `count_grad` 恰好比 `total_newton_steps` 多一倍于步数）。
我们的 `newton` 把只检查、不移动的最后一次也算上（`step.cc` 第 124 行）。下面的表把我们的减 1，统一成"走了几步"。

**Rigid-IPC 的 `stats.step_minimum_distances` 全是 −1**（没写进去），最小距离取 `animation.state_sequence`
里每一步的 `min_distance`（没有激活的接触对时是 null）。$\kappa$ 两边的定义和归一化不同（我们的障碍项在
`barrier.cc`，Rigid-IPC 用 ipc-toolkit 的），数值不直接比，比的是静止时的间隙和 Newton 步数。

## 2. 场景从哪里来

**先用 Rigid-IPC 自己的场景**（`fixtures/3D/`），由 `compare/rigid_ipc/import_rigid_ipc.py` 转成 MJCF：
- 自由刚体 → `freejoint`；`is_dof_fixed` 只留一个转动分量 → 过质心、沿世界坐标轴的 `hinge`；全钉死 → worldbody 上的几何体；
- 运动学体 → 铰链加 velocity 驱动，目标角速度写进关键帧；
- 网格直接引用它的 OBJ（MuJoCo 读 OBJ），密度照抄：两边都从网格算质量，`gear.obj` 密度 1 时两边都是 1.266；
- 体积为 0 的固定网格（地面那一片四边形）换成 MuJoCo 的无限平面。

| 场景 | Rigid-IPC 的文件 | 内容 | 我们的文件 |
|---|---|---|---|
| 一对齿轮 | `mechanisms/gears/single-kinematic.json` | 主动轮运动学体 −100°/s，从动轮绕 z 的铰链，齿 304 顶点，μ = 0，dt 0.01，T 10 s | `compare/rigid_ipc/gears_single.xml` |
| 圆环链 | `chain/10-links.json` | 10 个圆环（200 顶点），第 0 个固定，其余自由落下挂住；每个 142 kg | `compare/rigid_ipc/chain_10.xml` |

（待补：我们的锁链、搅拌机导出给 Rigid-IPC；见 00_roadmap.md。）

## 3. 结果（2026-11-06，第一轮）

机器：i9-14900KF（32 线程；Rigid-IPC 的 TBB 默认用满），RTX 4060 Ti。两边都是 h = 0.01、T = 10 s（1000 步）、
$\hat d$ = 1 mm、μ = 0。时间都是各自程序自己量的每步均值，不含写文件；只跑了一遍，±5% 以内的差别别当真。
原始数据在 `compare/rigid_ipc/results/{gears_single,chain_10}/`，表由 `compare/rigid_ipc/compare.py` 打印。

### 一对齿轮（`gears_single`）

| | Newton 容差 (m/s) | ms/步 均值 | 中位 | 最大 | Newton 步/步 | 最大 | 步末最小距离 |
|---|---|---|---|---|---|---|---|
| Rigid-IPC，默认 | 0.01 × 包围盒 ≈ 0.04 | 4.5 | 3.0 | 116 | 1.05 | 4 | 0.32 mm |
| Rigid-IPC，绝对容差 | 1e-6 | 4.3 | 3.0 | 110 | 1.04 | 3 | 0.36 mm |
| 我们，CPU | 1e-6 | 72.7 | | | 2.21 | 12 | 9.2 µm |
| 我们，GPU | 1e-6 | 2.2 | | | 2.24 | 12 | 9.2 µm |

从动轮转角（弧度；主动轮 −1.7453 rad/s，齿数相同，理想是 +1.7453 rad/s）：

| t | 1 s | 2 s | 5 s | 10 s |
|---|---|---|---|---|
| Rigid-IPC，默认 | 1.737 | 3.480 | 8.730 | 17.463 |
| Rigid-IPC，绝对容差 | 1.755 | 3.506 | 8.730 | 17.462 |
| 我们，GPU | 1.731 | 3.482 | 8.715 | 17.460 |

- 传动比两边都是 1.000（10 s 转 17.46 rad 对 17.45 rad），1 s 内的差别在 1% 以内：从动轮被齿推一下就以主动轮的
  速度空转，大部分步里没有活跃接触（Rigid-IPC 1000 步里只有 74 步有激活的接触对；我们的日志里 gap 大多是 inf）。
- Rigid-IPC 每步差不多只走 1 步 Newton，把容差从 0.04 m/s 收到 1e-6 m/s 也不变：空转的齿轮，惯性预测就是答案。
  我们走 2.2 步（含检查是 3.2），多出的那一步来自主动轮的 velocity 驱动——它是外力项，不在惯性预测里。
- 最小距离：Rigid-IPC 齿与齿最近到 0.3 mm；我们到 9 µm（第一步，齿初始就在接触范围里，$\kappa$ 刚初始化）。
  两边都没穿透。
- 时间：Rigid-IPC 4.5 ms/步（32 线程 CPU；最大的 116 ms 是第一步），我们 GPU 2.2 ms，CPU 72.7 ms。我们的 CPU 路径
  是单线程的参考实现，这里只能说明"同一台机器上两个 CPU 程序差 16 倍"，不是本文要比的东西。

### 圆环链（`chain_10`）

| | Newton 容差 (m/s) | ms/步 均值 | 中位 | 最大 | Newton 步/步 | 最大 | 步末最小距离 | 接触对/步 |
|---|---|---|---|---|---|---|---|---|
| Rigid-IPC，默认 | 0.01 × 包围盒 | 4.1 | 4.0 | 30 | 1.04 | 7 | 0.93 mm | 314 |
| Rigid-IPC，绝对容差 | 1e-6 | 4.2 | 4.0 | 29 | 1.04 | 7 | 0.86 mm | 314 |
| 我们，CPU | 1e-6 | 3561 | | | 6.12 | 49 | 1.5 µm | 423（每次 Newton） |
| 我们，GPU | 1e-6 | 5.3 | | | 6.12 | 49 | 1.5 µm | 423（每次 Newton） |

每个圆环最后一步的 y（m；第 0 个固定在 1.5，初始每个间隔 0.55，重力 −y）：

| 环 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Rigid-IPC | 0.901 | 0.302 | −0.297 | −0.896 | −1.495 | −2.094 | −2.693 | −3.292 | −3.891 |
| 我们 | 0.900 | 0.300 | −0.300 | −0.899 | −1.499 | −2.099 | −2.699 | −3.298 | −3.898 |

- 两边的链都挂住了，没有环掉出去；每个环落下 4.9 cm 挂在上一个环上，终态逐环差 0.8 mm，到第 9 个环累计 6.5 mm，
  我们略低。Rigid-IPC 在 10 步（0.1 s）内就静止了，我们 0.4 s。
- **Newton 步数差 6 倍，原因是静止时的间隙。** Rigid-IPC 的环之间停在 0.93 mm（$\hat d$ 的 93%），障碍项还在
  它平缓的那一段，一步 Newton 就到 1e-6 m/s；我们的 $\kappa$ 从 0 起步、第一次接触时按梯度平衡初始化（3.2e9），
  142 kg 的环压着它，前 0.4 s 里 $\kappa$ 升到 5.2e10 又回到 2.6e10，间隙被压到 6 µm，障碍项在接近奇异的那一段，
  每步要 6 步 Newton、86 次 PCG，前 0.4 s 里有 3 步撞到 50 步的上限。Rigid-IPC 的 $\kappa_0$ 是 3.8e9、$\kappa_{max}$
  3.8e11，一步都没调过（`num_kappa_updates=0`）。**要查：我们对重物的 $\kappa$ 初值是不是太小**，或者它的调整
  规则（阶段 9）在这种"接触当关节、长期受载"的场景下是不是该往大调而不是往回调。
- Rigid-IPC 的 314 个接触对是它的 `num_contacts`（步末的活跃对）；我们的 423 是每次 Newton 迭代的平均行数，
  口径不同，只说明数量级一样。
- 时间：Rigid-IPC 4.1 ms/步，我们 GPU 5.3 ms（Newton 多 6 倍、每步还要 86 次 PCG，GPU 上每步 4.9 ms 都在那张图里），
  CPU 3.56 s/步——10 个 200 顶点的圆环，CPU 的接触查找和 PCG 都没做过优化，这一列今后不再跑。

### 这一轮说明了什么

1. 在 Rigid-IPC 自己的场景上，两边给出同样的运动（传动比、链的挂点都对得上），都不穿透。
2. 齿轮这种"接触只偶尔激活"的场景两边都是每步 1–2 步 Newton，GPU 上我们 2 倍快于它 32 线程的 CPU。
3. 链这种"接触当关节、长期受载"的场景是我们的弱项：Newton 多 6 倍，GPU 上还比它慢 30%。
   症结在 $\kappa$ 和静止间隙，见上。这正是 00_roadmap.md 里"用接触当关节的代价"那一条。
4. Rigid-IPC 默认的相对容差在这两个场景里没起作用（1 步 Newton 就满足 1e-6 m/s），所以表里两行一样；
   到带摩擦、受载变化的场景再比才有区别。

### 待办

- 把 $\kappa$ 的问题查清楚之后再跑一遍链，并加上 μ = 0.3 的两组（Rigid-IPC 的 `coefficient_friction`）。
- 第二步：我们的锁链、搅拌机导出给 Rigid-IPC（见 00_roadmap.md）。

## 4. 怎么复现

```
# 场景转换（已转好的 xml 在仓库里）
python compare/rigid_ipc/import_rigid_ipc.py D:/Project/rigid-ipc/fixtures/3D/mechanisms/gears/single-kinematic.json compare/rigid_ipc/gears_single.xml --meshdir D:/Project/rigid-ipc/meshes
# 两个场景、两边、几种设置依次跑（Rigid-IPC 默认 -> sim.json，绝对容差 -> sim_abs.json，我们 GPU、CPU）
sh compare/rigid_ipc/run_all.sh all
# 单跑
python compare/rigid_ipc/run_rigid_ipc.py D:/Project/rigid-ipc/fixtures/3D/mechanisms/gears/single-kinematic.json --out compare/rigid_ipc/results/gears_single
python compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/gears_single.xml --out compare/rigid_ipc/results/gears_single --T 10 --h 0.01 --dhat 0.001 --mu 0 --gpu 1
# 表
python compare/rigid_ipc/compare.py compare/rigid_ipc/results/gears_single --gear-angles
python compare/rigid_ipc/compare.py compare/rigid_ipc/results/chain_10 --final-y
```

结果目录里 Rigid-IPC 还会写 `sim-chkpt*.json`（每 100 步的检查点）和 `sim.glb`、`log-*`；`compare.py` 跳过检查点。
