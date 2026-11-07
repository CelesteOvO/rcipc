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
| 障碍刚度 | IPC 原版的 $\kappa$ 初值和自适应（`min_barrier_stiffness_scale` 1e11）。目标函数是 $E/\bar m + \kappa B$，它的 $\kappa\,\bar m$ 才等于我们的 $\kappa$（3.3 节） | 阶段 9 的 $\kappa$ 初值和调整（`adapt=1`；实验开关 `relax=`、`kappa=`） |
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
里每一步的 `min_distance`（没有激活的接触对时是 null）。障碍函数两边一样（$b(d^2, \hat d^2)$，我们在
`barrier.cc`，Rigid-IPC 用 ipc-toolkit 的），$\kappa$ 差一个平均质量的归一化，换算见 3.3 节。

## 2. 场景从哪里来

**先用 Rigid-IPC 自己的场景**（`fixtures/3D/`），由 `compare/rigid_ipc/import_rigid_ipc.py` 转成 MJCF：
- 自由刚体 → `freejoint`；`is_dof_fixed` 只留一个转动分量 → 过质心、沿世界坐标轴的 `hinge`；全钉死 → worldbody 上的几何体；
- 运动学体 → 铰链加 velocity 驱动，目标角速度写进关键帧；
- 网格直接引用它的 OBJ（MuJoCo 读 OBJ），密度照抄（它不写时默认 1000）：两边都从网格算质量，但 MuJoCo 的 mesh
  默认按凸包算，要写 `inertia="exact"` 才和 Rigid-IPC 一致（`gear.obj` 密度 1 时都是 1.199；圆环差一倍，3.3 节）；
- 体积为 0 的固定网格（地面那一片四边形）换成 MuJoCo 的无限平面。

| 场景 | Rigid-IPC 的文件 | 内容 | 我们的文件 |
|---|---|---|---|
| 一对齿轮 | `mechanisms/gears/single-kinematic.json` | 主动轮运动学体 −100°/s，从动轮绕 z 的铰链，齿 304 顶点，μ = 0，dt 0.01，T 10 s | `compare/rigid_ipc/gears_single.xml` |
| 圆环链 | `chain/10-links.json` | 10 个圆环（200 顶点），第 0 个固定，其余自由落下挂住；每个 72.7 kg | `compare/rigid_ipc/chain_10.xml` |

（待补：我们的锁链、搅拌机导出给 Rigid-IPC；见 00_roadmap.md。）

## 3. 结果（2026-11-06）

机器：i9-14900KF（32 线程；Rigid-IPC 的 TBB 默认用满），RTX 4060 Ti。两边都是 h = 0.01、T = 10 s（1000 步）、
$\hat d$ = 1 mm、μ = 0。时间都是各自程序自己量的每步均值，不含写文件；只跑了一遍，±5% 以内的差别别当真。
原始数据在 `compare/rigid_ipc/results/{gears_single,chain_10}/`，表由 `compare/rigid_ipc/compare.py` 打印。

第一轮跑完发现我们这边的质量错了一倍（第 3.3 节），下面是修正以后重跑的数字；第一轮的原始文件留在
`results/*/hull_mass/`。

### 3.1 一对齿轮（`gears_single`）

| | Newton 容差 (m/s) | ms/步 均值 | 中位 | 最大 | Newton 步/步 | 最大 | 步末最小距离 |
|---|---|---|---|---|---|---|---|
| Rigid-IPC，默认 | 0.01 × 包围盒 ≈ 0.04 | 4.5 | 3.0 | 116 | 1.05 | 4 | 0.32 mm |
| Rigid-IPC，绝对容差 | 1e-6 | 4.3 | 3.0 | 110 | 1.04 | 3 | 0.36 mm |
| 我们，CPU | 1e-6 | 78.8 | | | 2.23 | 12 | 8.8 µm |
| 我们，GPU | 1e-6 | 2.0 | | | 2.22 | 12 | 8.8 µm |

从动轮转角（弧度；主动轮 −1.7453 rad/s，齿数相同，理想是 +1.7453 rad/s）：

| t | 1 s | 2 s | 5 s | 10 s |
|---|---|---|---|---|
| Rigid-IPC，默认 | 1.737 | 3.480 | 8.730 | 17.463 |
| Rigid-IPC，绝对容差 | 1.755 | 3.506 | 8.730 | 17.462 |
| 我们，GPU | 1.731 | 3.479 | 8.715 | 17.458 |

- 传动比两边都是 1.000（10 s 转 17.46 rad 对 17.45 rad），1 s 内的差别在 1% 以内：从动轮被齿推一下就以主动轮的
  速度空转，大部分步里没有活跃接触（Rigid-IPC 1000 步里只有 74 步有激活的接触对；我们的日志里 gap 大多是 inf）。
- Rigid-IPC 每步差不多只走 1 步 Newton，把容差从 0.04 m/s 收到 1e-6 m/s 也不变：空转的齿轮，惯性预测就是答案。
  我们走 2.2 步（含检查是 3.2），多出的那一步来自主动轮的 velocity 驱动——它是外力项，不在惯性预测里。
- 最小距离：Rigid-IPC 齿与齿最近到 0.3 mm；我们到 9 µm（第一步，齿初始就在接触范围里，$\kappa$ 刚初始化）。
  两边都没穿透。
- 时间：Rigid-IPC 4.5 ms/步（32 线程 CPU；最大的 116 ms 是第一步），我们 GPU 2.0 ms。我们的 CPU 路径
  是单线程的参考实现，这一列只说明"同一台机器上两个 CPU 程序差多少倍"，不是本文要比的东西。

### 3.2 圆环链（`chain_10`）

| | Newton 容差 (m/s) | ms/步 均值 | 中位 | 最大 | Newton 步/步 | 最大 | 步末最小距离 | 接触对/步 |
|---|---|---|---|---|---|---|---|---|
| Rigid-IPC，默认 | 0.01 × 包围盒 | 4.1 | 4.0 | 30 | 1.04 | 7 | 0.93 mm | 314 |
| Rigid-IPC，绝对容差 | 1e-6 | 4.2 | 4.0 | 29 | 1.04 | 7 | 0.86 mm | 314 |
| 我们，GPU | 1e-6 | 4.7 | | | 5.53 | 49 | 1.9 µm（静止时 6.3 µm） | 423（每次 Newton） |
| 我们，CPU（第一轮，质量 142 kg） | 1e-6 | 3561 | | | 6.12 | 49 | 1.5 µm | |

每个圆环最后一步的 y（m；第 0 个固定在 1.5，初始每个间隔 0.55，重力 −y）：

| 环 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Rigid-IPC，$\hat d$ = 1 mm | 0.901 | 0.302 | −0.297 | −0.896 | −1.495 | −2.094 | −2.693 | −3.292 | −3.891 |
| Rigid-IPC，$\hat d$ = 0.1 mm | 0.9001 | 0.3002 | −0.2997 | −0.8996 | −1.4995 | −2.0994 | −2.6993 | −3.2992 | −3.8991 |
| 我们 | 0.9002 | 0.3003 | −0.2995 | −0.8993 | −1.4990 | −2.0988 | −2.6985 | −3.2982 | −3.8978 |

- 两边的链都挂住了，没有环掉出去；每个环落下 4.9 cm 挂在上一个环上。终态逐环差 0.8 mm 是静止间隙的差
  （Rigid-IPC 停在 0.93 mm，我们 6 µm）：把 Rigid-IPC 的 $\hat d$ 改成 0.1 mm，它的间隙变成 0.096 mm，终态和我们
  对到 0.1 mm 以内。Rigid-IPC 在 10 步（0.1 s）内就静止了，我们 0.4 s。
- **Newton 步数差 5 倍**：Rigid-IPC 每步 1 步，我们静止时每步 5 步（含检查 6 次）、83 次 PCG。原因见 3.3 节，
  不是 $\kappa$。
- Rigid-IPC 的 314 个接触对是它的 `num_contacts`（步末的活跃对）；我们的 423 是每次 Newton 迭代的平均行数，
  口径不同，只说明数量级一样。
- 时间：Rigid-IPC 4.1 ms/步，我们 GPU 4.7 ms（Newton 多 5 倍，每步 4.9 ms 都在那张图里）。CPU 这一列没有重跑
  （第一轮 3.56 s/步——10 个 200 顶点的圆环，CPU 的接触查找和 PCG 都没做过优化），今后不再跑。

### 3.3 查 $\kappa$：查到的是三件别的事

链上 Newton 多 5 倍，第一轮怀疑是 $\kappa$ 让受载的环压到 6 µm。沿着 $\kappa$ 查下去，依次发现：

**（1）我们的质量错了一倍。** MuJoCo 的 mesh 默认按**凸包**算质量和惯量（`inertia="convex"`），圆环的凸包把洞填上了：
torus.obj 密度 1000 时 MuJoCo 给 142 kg，精确体积是 72.7 kg（Rigid-IPC 用精确体积；它的 JSON 没写密度，默认也是
1000；用它输出的势能差核对：9 个环各落 0.049k m，ΔPE = 1571 J ⇒ m = 72.8 kg）。齿轮 1.266 对 1.199。
`import_rigid_ipc.py` 现在给动的刚体写 `inertia="exact"`，两边质量一致。上面 3.1、3.2 的表是修正以后的。
修正以后链上 Newton 步数不变（5.5 步对第一轮的 6.1 步）。

**（2）Rigid-IPC 的 $\kappa$ 和我们的差一个平均质量。** 它的目标函数是 $E/\bar m + \kappa B$
（`distance_barrier_rb_problem.cpp` 的 `compute_objective`：`Ex /= average_mass()`），所以它的 $\kappa\,\bar m$
才是我们的 $\kappa$：

| 场景 | Rigid-IPC 的 $\kappa_0$ | × $\bar m$ ⇒ 我们的单位 | 我们实际的 $\kappa$ |
|---|---|---|---|
| 齿轮 | 3.69e7 | 4.4e7 | 2.9e7 |
| 链 | 3.77e9 | 2.7e11 | 1.3e10（第一轮 2.6e10） |

链上它比我们硬 20 倍。它的 $\kappa_0$ 就是 IPC 的下限 $\kappa_{\min} = 10^{11}\bar m / (4 d_0^2 b''(d_0^2))$，
$d_0 = 10^{-8}\times$包围盒对角线（ipc-toolkit `adaptive_stiffness.cpp`）；梯度平衡给的值比它小，被夹到下限，
之后一步都没调过（`num_kappa_updates=0`；它只在 $d_{\min}$ 小于 $10^{-9}\times$包围盒时翻倍，从不减半）。
我们：第一次接触时梯度平衡给 3.3e9，书 §10.7 的规则（间隙 $<10^{-3}\hat d$ 翻倍、$>10^{-2}\hat d$ 减半）把它调到
1.3e10，静止间隙按设计落在 $[10^{-3}, 10^{-2}]\hat d$ 里（6.3 µm）。
用它的 $\kappa$ 算一下受力也对得上：$\kappa$ = 2.7e11、间隙 0.93 mm 时每个接触对 30 N，130–200 个受载对撑起
6.4 kN；把它的 `min_barrier_stiffness_scale` 改小 100 倍（$\kappa_0$ = 3.77e7），间隙变成 0.12 mm、每对 49 N，
一致。

**（3）$\kappa$ 不是 Newton 多的原因；是线性收敛。** 在链上（GPU，修正质量后）换规则：

| 设置 | Newton 步/步（含检查） | 静止间隙 | PCG/步 | ms/步 |
|---|---|---|---|---|
| 默认（adapt=1，减半开） | 5.53（6.53） | 6.3 µm | 83 | 4.7 |
| relax=0（只翻倍不减半） | 5.46（6.46） | 2.8 µm 以上 | 52 | 4.2 |
| adapt=0（钉在初值 4.6e9） | 6.24（7.24） | 1.9 µm | 135 | 5.9 |
| kappa=2.7e11（Rigid-IPC 的值）relax=0 | 5.35（6.35） | 0.31 mm | 64 | 4.2 |
| kappa=2.7e11，减半开 | 5.53（6.53） | 6.3 µm | 83 | 4.7 |
| pcg_tol=1e-6 / 1e-8 | 5.36 / 5.40 | 同默认 | 110 / 150 | 4.9 / 5.4 |
| icurv=0 | 5.53 | 同默认 | 83 | 4.6 |
| curv=0（纯 Gauss-Newton，不加曲率上界） | 4.67（5.67） | 同默认 | 76 | 4.1 |
| tol=1e-3 / 1e-4 / 1e-5 / 1e-6 | 1.43 / 3.17 / 4.30 / 5.53 | 同默认 | 34 / 56 / 71 / 83 | 1.8 / 3.1 / 3.9 / 4.7 |

$\kappa$ 换 20 倍、PCG 精度换 $10^4$ 倍、惯性曲率开关，静止时都是 5 步。容差每放宽一个数量级少一步：
**Newton 在这里是线性收敛，每步缩一个数量级。** 静止的环，每步惯性预测把它往下拉 $h^2 g$ = 1 mm，起点的
"速度" $hg$ = 0.1 m/s，到 1e-6 m/s 要 5 个数量级，正好 5 步。$\kappa$ 预设成 Rigid-IPC 的值时间隙 0.31 mm
（它是 0.93 mm，受载对数不同），步数照旧。
把阶段 6 的曲率上界去掉（curv=0）静止时少 1 步（4 步 → 3 步），说明上界让模型偏硬，但剩下的 3 步仍是线性的：
接触 Hessian 只有 Gauss-Newton 项 $\kappa b''\,r^Tr$，丢了 $\kappa b'\,\partial^2(d^2)/\partial\delta^2$
（`contact.cc` 的 `contact_hessian`；阶段 9 第 7 节只补回了法向），圆环之间的接触法向随转动变化快，这一项不小。
Rigid-IPC 用完整的二阶导数（自动微分）再逐块投影到半正定（`project_to_psd`），所以静止时 1 步就到。
Rigid-IPC 自己把 $\hat d$ 改成 0.1 mm 时也不好过：每步 23 ms、Newton 方向的线搜索失败后沿负梯度退出
（`total_iter=0`，"converged without taking a step"），说明它的 1 步也靠间隙停在 $0.9\hat d$ 这个平缓段。

**补全以后（阶段 14，[17_contact_hessian.md](17_contact_hessian.md)）。** 完整的接触 Hessian（顶点空间按包络定理精确算，
每个接触 12 x 12 的旋量块投影半正定，链上的剩余项进树块）做出来以后，静止链上每次迭代的收缩比从 1/9 变成约 1/30，
每步少 1 次 Newton（6.53 → 5.50，齿轮不变），容差扫描整体下移一步（tol 1e-3…1e-6：1.16 / 2.21 / 3.32 / 4.50 步）。
所以上一段的判断只对了一部分：Hessian 缺项解释了 5 步里的 1 步，剩下的线性收敛和 PCG 容差、惯性曲率都无关，
还没查清（怀疑半正定投影和最近点子类型的切换）。现在的实现每步慢 4 倍，默认不开（`fullh=1`）。

**结论。** 链上我们比 Rigid-IPC 多的 Newton 步数来自线性收敛，不是 $\kappa$；接触 Hessian 不精确是其中一部分原因。
$\kappa$ 规则该不该改是另一个问题：减半规则把静止间隙压到 $10^{-2}\hat d$，Rigid-IPC 让它停在 $0.9\hat d$；
在这两个场景里"不减半"（relax=0）各快 10%（PCG 少三分之一，齿轮最大 Newton 13 → 8），默认值暂时没动，
`scene_demo` 多了 `relax=`、`kappa=`、`curv=` 三个开关做实验。阶段 9 的问题 5（切向项多大）在这里有了答案：
受载的曲面接触上它决定收敛阶。

### 待办

- ~~接触 Hessian 补全~~（阶段 14 做了：少 1 步，没到 1–2 步，见上）。剩下的线性收敛从哪来：
  半正定投影？最近点子类型在迭代之间切换？再回头决定 $\kappa$ 的减半规则留不留。
- 加 μ = 0.3 的两组（Rigid-IPC 的 `coefficient_friction`）。
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
