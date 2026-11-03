# 和 libuipc 对比（不看墙钟时间的部分）

**目的：** 在同一个场景、同一组参数下，比较 RC-IPC 的重写版（阶段 1–10，CPU）和 libuipc（GPU）。墙钟时间要等 GPU 版（阶段 11）才公平，这里只比和硬件无关的量：
- 关节、刚体是不是精确的；
- 离地间隙；
- 每步的 Newton 次数；
- 是否穿模。

脚本和原始结果都在 `compare/libuipc/`。旧分支的 G0 报告（`G0_REPORT_zh.md`，60 个布娃娃，Linux + RTX 3090）是同一套场景定义，这里用 8 个布娃娃，CPU 版跑得动。

---

## 1. 场景和设置

**场景：** 旧分支 `scenes/ragdoll.py` 的布娃娃（复制成 `compare/libuipc/ragdoll_scene.py`，没有改）。
- 每个布娃娃 11 个胶囊，10 个关节：颈、腰、肩、髋是球关节，肘、膝是铰链，没有限位。
- 8 个布娃娃脸朝下躺着、随机朝向，分 2 × 2 列、每列 2 层落下。
- dt = 0.01 s，密度 1000，μ = 0.2，y 轴朝上，模拟 3 s。

两边用的几何、关节锚点、铰链轴、初始姿态完全一样。具体做法：
- `make_mjcf.py` 从同一个定义生成我们的 MJCF（`ragdolls_8.xml`）；
- `run_uipc.py` 用同一个定义搭 libuipc 的场景。

| | libuipc 0.0.28 | RC-IPC 重写版 |
|---|---|---|
| 未知量 | 每个刚体 12 个仿射坐标（88 × 12 = 1056） | 关节坐标（8 × 28 = 224） |
| 刚体 | 仿射体，ABD 刚度 κ = 1e8 | 严格刚体（FK） |
| 关节 | 罚函数，强度 γ = 100 | 关节就是坐标 |
| 几何 | 胶囊的三角网格（每个 26–34 个顶点） | 精确胶囊（两个球心 + 一条边） |
| 质量 | 网格体积 × 密度 | 同左（`make_mjcf.py` 把网格体积写进 MJCF） |
| 接触 | 所有刚体对（包括相邻） | 同左（MJCF 里 `filterparent="disable"`，这次新加的开关） |
| d̂ | 1 cm | 1 cm |
| 摩擦 ε_v | 0.01 m/s | 0.01 m/s |
| 障碍刚度 κ | libuipc 的自适应规则（1e9，被夹到 5.3e8） | 阶段 4、9 的规则（梯度平衡 + 按间隙调整） |
| 线性求解 | fused PCG，tol_rate 1e-3 | 树块预条件 PCG，rz 准则 1e-4 |

**两组停止规则：**
- **默认（libuipc 的默认值）**：两边都开半隐式停止（K_min = 6，β_tol = 1e-3），收敛容差 0.05 m/s。
- **完全收敛**：两边都关掉半隐式停止，收敛容差 1e-3 m/s。

两边的容差都是“一个整步里顶点的最大速度变化”，定义相近，但不完全相同。libuipc 对仿射部分还有一个 transrate_tol。

---

## 2. 结果

`python compare/libuipc/compare.py results/*.json`，3 s，300 步：

| 设置 | 引擎 | Newton/步：均值 / 中位 / p95 / 最大 | 关节漂移最大 | 铰链轴偏差最大 | 非刚性 max‖AᵀA−I‖ | 最后离地 | 步末最小距离 |
|---|---|---|---|---|---|---|---|
| 默认 | libuipc | 3.56 / 2 / 7 / 7 | 0.029 mm | 0.013° | 5.5e-4 | 8.84 mm | 未测 |
| 默认 | RC-IPC | 1.84 / 1 / 6 / 15 | 5e-13 mm | 1.2e-6°（舍入） | 5e-15 | 0.063 mm | 0.8 μm |
| 完全收敛 | libuipc | 7.11 / 5 / 15 / 65 | 0.066 mm | 0.033° | 1.1e-4 | 8.74 mm | 未测 |
| 完全收敛 | RC-IPC | 7.74 / 6 / 17 / 50（3 步撞上限） | 5e-13 mm | 1.2e-6°（舍入） | 6e-15 | 0.078 mm | 4.9 μm |

**指标说明：**
- 关节漂移、铰链轴偏差、非刚性都每 10 步采样一次，取所有布娃娃的最大值，两边用同一个函数算（`metrics.py`）。
- 离地是几何体最低点的 y：libuipc 量网格顶点，RC-IPC 量精确胶囊。
- RC-IPC 另外打开了逐次迭代的轨迹检查（`check=1`）：每次接受的迭代轨迹上，最小距离是 0.25 μm（默认）和 0.7 μm（完全收敛）。
- 铰链轴的 1.2e-6° 是 arccos 在 1 附近的舍入：$\arccos(1 - 10^{-16})\approx 1.4\times10^{-8}$ rad，不是误差。

**分时段的 Newton 次数**（前 1 s 是砸下来、堆起来；最后 1 s 已经静止）：

| 设置 | 引擎 | 前 1 s 均值 | 最后 1 s 均值 | 3 s 总数 |
|---|---|---|---|---|
| 默认 | libuipc | 6.02 | 1.49 | 1069 |
| 默认 | RC-IPC | 3.53 | 1.00 | 553 |
| 完全收敛 | libuipc | 12.19 | 3.88 | 2133 |
| 完全收敛 | RC-IPC | 12.42 | 3.00 | 2321 |

---

## 3. 怎么读这些数字

**1. 精确性：这是两者最大的差别，而且是结构性的。**
- libuipc 的关节是罚函数，刚体是仿射体，所以有关节漂移（0.03–0.07 mm）、铰链轴偏差（0.01–0.03°）和 0.01–0.05% 的拉伸或剪切。
- RC-IPC 这三项都是舍入误差：关节就是坐标，顶点由 FK 得到。
- 8 个布娃娃载荷轻，所以 libuipc 的误差不大。G0 报告里 60 个布娃娃堆在一起时，漂移是 0.5–0.7 mm，轴偏差 0.43°，非刚性 1.1–1.5%。罚函数关节的误差随载荷增长；把 γ 提到 1e4 可以压到 0，但当时每步慢了 2.3 倍。

**2. 离地间隙：差别来自 κ 的规则，不是方法。**
- libuipc 静止时离地 8.8 mm：d̂ 的 1 cm 里，大部分留成了看得见的间隙。
- 我们 0.06–0.08 mm：阶段 9 的 κ 调整在间隙大于 $10^{-2}\hat d$ 时把障碍放软，一路压到 0.1 mm 以下。
- 阶段 4 没有这个调整时，落地盒子也停在接近 d̂ 的地方（0.98 mm，d̂ = 1 mm）。

**3. Newton 次数：完全收敛时两边持平，默认设置下我们少一半。**
- 完全收敛时总数 2321 对 2133，冲击段 12.4 对 12.2。在这个场景上，约化坐标并没有让 Newton 少迭代。它的优势在每次迭代的规模：224 个未知量，而 libuipc 是 1056 个；而且没有关节和仿射变形这两组刚度要解。
- 默认设置下我们是 553 对 1069。但两边的停止规则只是“同名同参数”：
  - libuipc 的半隐式最多停在第 7 次（最大值正好是 7）；
  - 我们的 β 在 CCD 截断时降得慢，最多到 15 次；
  - 静止段我们每步 1 次（第一次迭代就满足 0.05 m/s 的容差），libuipc 1.5 次。
  所以这一组的差别主要来自容差的具体定义，不能当成算法上的优势。
- 我们的完全收敛有 3 步撞了 50 次的上限，都在冲击段。这是阶段 10 笔记里说的“CCD 限制下的爬行”（书 §10.6.4）；libuipc 最多用了 65 次，它的上限是 1024。

**4. 不穿模：两边都是 IPC，都有保证。** 我们这边做了独立检查：步末和逐次迭代的轨迹，最小距离都大于 0。libuipc 这边只量了离地，刚体之间的距离没有独立检查。

**墙钟时间，只作参照：**
- libuipc 在 RTX 4060 Ti 上每步中位数 24.7 ms（默认）和 43.5 ms（完全收敛），而且打开了计时器，有同步开销。
- RC-IPC 是 CPU 单线程 Release，不开逐次迭代检查，每步平均 23.1 ms 和 69.8 ms。

两边的硬件、计时口径都不同，不能直接比。而且重写版还缺原版的宽相、接触岛、紧加速度界这些优化（见 `notes/10_scenes.md`）。

---

## 4. 复现

```
python compare/libuipc/make_mjcf.py 8
cmake --build build --config Release --target scene_demo
build\Release\scene_demo.exe model=<仓库>/compare/libuipc/ragdolls_8.xml T=3 dhat=0.01 mu=0.2 eps_v=0.01 tol=0.05 semi=1 check=1 out=rcipc_default.csv
build\Release\scene_demo.exe model=<仓库>/compare/libuipc/ragdolls_8.xml T=3 dhat=0.01 mu=0.2 eps_v=0.01 tol=1e-3 semi=0 check=1 out=rcipc_full.csv
python compare/libuipc/measure_rcipc.py compare/libuipc/ragdolls_8.xml rcipc_default.csv compare/libuipc/results/rcipc_default.json
python compare/libuipc/measure_rcipc.py compare/libuipc/ragdolls_8.xml rcipc_full.csv compare/libuipc/results/rcipc_full.json
D:/Project/uipc-venv/Scripts/python.exe compare/libuipc/run_uipc.py 8 300 compare/libuipc/results/uipc_default.json --workspace <临时目录>
D:/Project/uipc-venv/Scripts/python.exe compare/libuipc/run_uipc.py 8 300 compare/libuipc/results/uipc_full.json --full --vel-tol 1e-3 --workspace <临时目录>
python compare/libuipc/compare.py compare/libuipc/results/uipc_default.json compare/libuipc/results/rcipc_default.json compare/libuipc/results/uipc_full.json compare/libuipc/results/rcipc_full.json
```

pyuipc 0.0.28 装在单独的 venv（`D:\Project\uipc-venv`），用 CUDA 后端。libuipc 不是确定性的，重跑时它的数字会有几个百分点的波动（G0 报告：3 次之间漂移相差 0.50–0.70 mm）。这里每组只跑了一次。

---

## 5. 思考题

1. 把布娃娃加到 60 个，libuipc 的关节漂移会涨到 0.5 mm 以上（G0 报告）。为什么罚函数关节的误差随载荷增长？γ 提高 100 倍，误差和 Newton 次数各怎么变？
2. 我们的 κ 调整把静止间隙压到 0.1 mm 以下，libuipc 留着 8.8 mm。两种选择各自的代价是什么（提示：Hessian 的条件数、看得见的悬空、冲击时的最小间隙）？
3. 完全收敛时两边的 Newton 次数差不多，可我们只有 224 个未知量，libuipc 有 1056 个。每次 Newton 迭代的代价比是多少？PCG 次数（我们每步 38）该怎么和 libuipc 比？
4. 把这次的对比扩展到 60 个布娃娃，CPU 版哪一部分先撑不住（提示：`find_contacts` 的暴力循环、稠密的 $H_k$）？
