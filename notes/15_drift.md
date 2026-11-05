# 看得见的关节漂移和刚体变形：摆锤越过矮墙

**问题：** 和 libuipc 比的时候（[compare_libuipc.md](compare_libuipc.md)），它的关节漂移只有 0.03–0.07 mm，刚体变形 0.01–0.05%，画面上看不出来。原因是布娃娃堆的载荷轻。
- libuipc 的刚体是**仿射体**（ABD）：每个刚体 12 个未知量，靠一个“正交性能量”把 $A$ 拉回转动矩阵，所以会被拉长、剪切。
- 它的关节是**罚函数**：两个刚体上的锚点之间一根很硬的弹簧，所以会被拉开。
- 我们这两项都是舍入误差：关节就是坐标，刚体由正运动学生成。

这一篇和 [14_visible.md](14_visible.md) 的思路一样：设计一个场景，让这两项误差**改变结局**，回放时一眼能看出来，也能用程序独立地查。

| 场景 | 文件 | 生成 | 测试 |
|---|---|---|---|
| 摆锤：30 kg 的钢块挂在 10 节 5 mm 粗的钢条连成的 1 m 长的链上，从水平位置释放，摆过一堵矮墙 | `models/wrecking_ball.xml` | `tools/make_wrecking_ball.py` | `test_gpu.cc` 的 `Drift.WreckingBallClearsWall` |

视频在 `media/`（不进仓库）：
- `wrecking_ball_vs_libuipc.mp4`：左边我们，右边 libuipc，都是 libuipc 默认的 10 ms 步长，原速的一半。
- `wrecking_ball_close.mp4`：摆到最低点那 0.5 s 的特写，慢放 5 倍。
- `wrecking_ball_hinge.png`：同一时刻、同一个铰链的特写。

---

## 1. 两项误差从哪里来，怎样让它们变大

**罚函数关节。** libuipc 的说明：关节刚度 = strength_ratio $\gamma$ × 两个刚体的质量之和（默认 $\gamma$ = 100）。从测到的数字看，这个刚度是按 $1/h^2$ 放大的（漂移和 $h^2$ 成正比，下面第 3 节），所以一个铰链传力 $F$ 时，两个轴点各分一半，被拉开

$$
\delta \approx \frac{F\,h^2}{2\gamma\,(m_A + m_B)} .
$$

- 刚度只看两个刚体自己的质量，**和它要传的力无关**。轻的连杆传大的力，就被拉开得多。
- 和 $h^2$ 成正比：步长减半，漂移变成四分之一。
- 核对：下面的场景里，摆到最低点时链上的拉力约 $3Mg$ = 880 N，相邻两节钢条各 23.4 g，$h$ = 10 ms，公式给出 9.4 mm；测到的最大是 8.4 mm。

**仿射体。** ABD 的正交性能量是 $\kappa V \|A^\top A - I\|_F^2$。单向拉伸 $A = \mathrm{diag}(1+\varepsilon, 1, 1)$ 时约 $4\kappa V\varepsilon^2$，相当于杨氏模量约 $8\kappa$。
- libuipc 的例子用 $\kappa$ = 100 MPa，相当于杨氏模量约 0.7 GPa，比钢（200 GPa）软约 300 倍。
- 所以细杆里的应力 $\sigma$ 让它伸长约 $\sigma / (7\kappa)$。和步长无关。
- 核对：5 mm × 5 mm 的钢条拉 880 N，应力 35 MPa，伸长约 5%；测到 3.3%（10 ms）到 7.0%（1 ms）。

**所以场景是“轻链吊重物”：**
- 链：10 节 5 mm × 5 mm 的钢条，节距 10 cm，每节 23.4 g。
- 摆锤：16 cm 的立方体，30 kg，是一节钢条的 1282 倍。
- 这在机械里很平常（链条吊重物、细连杆带大负载），钢条里的应力 35 MPa 也远低于钢的屈服强度。真实的钢条在这个拉力下只伸长 0.02%。

## 2. 场景

- **链：** 像自行车链，相邻两节在 y 方向错开 5 mm（一节在 +y，一节在 −y），两端各伸过铰链 1 cm，看起来是连着的，但互相碰不到。铰链轴沿 y。
- **摆锤：** 用最后一个铰链挂在链的末端，铰链离立方体的近端面 1.5 cm。摆锤可以绕这个铰链转。
- **转轴：** 铰链 0 穿过一个固定的支架（在链的 −y 一侧）。libuipc 里关节的另一方要是一个刚体，所以支架是一个固定的仿射体；它的质量也进了这个关节的刚度。
- **矮墙：** 30 cm 高、6 cm 厚，在转轴的正下方。转轴的高度取得正好让**链竖直时摆锤的底面比墙顶高 2 cm**。
- **过程：** 链从水平伸直的姿态静止释放，摆下去，从墙顶上越过，摆到另一边，再摆回来。
  - 链只要被拉长 2 cm 以上，摆锤就会撞上墙，过不去。
  - 摆锤会绕自己的铰链转一点，最低的那个角会比底面中心低。我们这边实际的最近距离是 9–15 mm。
- **显示用的东西（contype = 0，两边都不参与计算）：**
  - 秋千一样的架子；
  - 每个铰链一个深色的“孔”（在父刚体上）和一根橙色的销子（在子刚体上）。关节没被拉开时销子在孔的正中；拉开 8 mm 时销子整个跑出孔外。

**同一个 MJCF，两边都跑：**
- `compare/libuipc/export_mjcf.py`（系统的 Python，有 MuJoCo）把刚体、盒子、铰链、世界上的几何导出成 JSON；
- `compare/libuipc/run_uipc_scene.py`（libuipc 的 venv）照着 JSON 搭 libuipc 的场景：每个刚体一个仿射体（盒子合成的三角网格，密度 = 质量 / 体积），每个铰链一个 `AffineBodyRevoluteJoint`，世界上的盒子是固定的仿射体，平面是地面。
- 两边的 d̂ 都是 1 mm，μ = 0.2，ε_v = 0.01 m/s；都检测所有刚体对（MJCF 里关掉了 `filterparent`）。
- 世界上的东西之间不检测（墙的底面贴着地面，libuipc 的起始检查会拒绝）：libuipc 里把固定的盒子和地面放进同一个接触元素，元素自己和自己的接触关掉。

**检查**（`tools/joint_check.py`，两边用同一个函数）：
- 我们的轨迹：用 MuJoCo 的正运动学得到每个刚体的位姿。
- libuipc 的轨迹：每个刚体的仿射变换 $[A \mid p]$（`run_uipc_scene.py` 写的 npz）。
- 每帧量：
  - 关节漂移：同一个铰链在父、子刚体上的锚点之间的距离；
  - 主应变：$A$ 的奇异值离 1 最远的那个；
  - 链长：沿铰链走一遍的折线长度，分成“关节拉开的”和“钢条（铰链到铰链）的”两部分；
  - 摆锤和墙之间距离的下界（分离轴），小于 2 mm 算碰到；
  - 摆锤的质心从墙的一边到另一边一次，算越过一次。

---

## 3. 结果（4 s）

**我们**（GPU，`mu=0.2 eps_v=0.01 tol=0.01 semi=1`）：

| $h$ | 关节漂移 | 主应变 | 最长的链 | 摆锤 | 墙钟（4 s） | Newton/步 |
|---|---|---|---|---|---|---|
| 10 ms | 0 | 0（$\|A^\top A - I\|$ 4e-15） | 1000.0 mm | 越过 3 次，离墙最近 14.7 mm | 0.4 s（1.0 ms/步） | 2.24，不撞上限 |
| 5 ms | 0 | 0 | 1000.0 mm | 越过 3 次，最近 11.7 mm | 0.8 s（1.0 ms/步） | 2.27 |
| 2 ms | 0 | 0 | 1000.0 mm | 越过 3 次，最近 9.1 mm | 1.9 s（1.0 ms/步） | 2.24 |

这是阶段 13 以后的数字（[16_inertia_curvature.md](16_inertia_curvature.md)）。这一篇最初提交时，惯性 Hessian 只有 $M/h^2$，10 ms 时每步 73 ms、72% 的步撞上 Newton 上限，见第 5 节。

**libuipc**（pyuipc 0.0.28，GPU；默认是半隐式停止、velocity_tol 0.05 m/s；墙钟开着计时器，有同步开销）：

| 设置 | 关节拉开（总和，最长时） | 钢条伸长（最长时） | 最长的链 | 摆锤 | 墙钟（4 s） |
|---|---|---|---|---|---|
| **默认**：dt 10 ms，γ 100，κ 100 MPa | 74 mm（单个最大 8.4 mm） | +30 mm（主应变最大 3.3%） | **1104 mm** | **0.65 s 撞墙**，最后搁在墙顶上 | 3.8 s |
| dt 5 ms | 27 mm | +43 mm（4.7%） | 1070 mm | 0.62 s 撞墙 | 10.4 s |
| dt 2 ms | 6 mm | +60 mm（6.6%） | 1067 mm | 0.62 s 撞墙 | 28.6 s |
| dt 1 ms | 1.6 mm | +64 mm（7.0%） | 1065 mm | 0.61 s 撞墙 | 51.8 s |
| γ = 1e4 | 1.1 mm | +43 mm（4.8%） | 1044 mm | 0.82 s 撞墙，之后停在墙的正上方 | 4.0 s |
| κ = 1 GPa | 80 mm | +3 mm（0.38%） | 1084 mm | 0.64 s 撞墙 | 3.2 s |
| κ = 10 GPa | 81 mm | +0.3 mm | 1081 mm | 0.63 s 撞墙 | 3.2 s |
| κ = 100 GPa | 86 mm | 0 | 1086 mm | 0.63 s 撞墙 | 5.0 s |
| dt 2 ms，κ = 10 GPa | 11 mm | +1.2 mm | 1013 mm | 0.64 s 擦到墙（最近 0.8 mm），之后照样摆 | 27.3 s |
| 完全收敛（关掉半隐式停止，1e-3 m/s） | 82 mm | +33 mm | 1115 mm | 0.64 s 撞墙 | 5.7 s |

（γ 和 κ 都调大的几组见第 4 节。）

**录像里：**
- 我们：摆锤三次从墙顶上越过，左下角一直是“链长 1000 mm（关节 +0，钢条 +0）”。
- libuipc：摆到最低点前链已经长了 10 cm，摆锤撞上墙，最后搁在墙顶上（离墙 0.8 mm，就是障碍留出的间隙），再也没摆过去。特写里链中间的销子都离开了孔（第 9 个铰链拉开 8.4 mm）。

## 4. 公平地说：libuipc 能不能调好

**两项误差要分开调，各管一项：**
- **减小步长**只管关节：漂移随 $h^2$ 下降（10 ms 的 74 mm 到 1 ms 的 1.6 mm），但钢条的伸长反而更大（3.3% 到 7.0%）。可能是步长大时隐式欧拉的数值阻尼削掉了一部分拉力的峰值（没有细查）。
- **加大 γ** 只管关节：γ = 1e4 时漂移 1 mm，钢条照样伸长 4.8%。
- **加大 κ** 只管钢条：κ = 10 GPa 时钢条不伸长了，关节照样拉开 81 mm。
- 只调一项，**每一组都撞墙**。“完全收敛”那一组也撞，所以这不是 Newton 没收敛的问题，是罚函数关节和仿射体本身的平衡位置。

**两项一起调大：**

| 设置 | 停止规则 | 最长的链 | 摆锤 | 墙钟（4 s） | Newton/步 |
|---|---|---|---|---|---|
| dt 10 ms，γ 1e6 | 默认（半隐式） | 1009 mm | **几乎不往下摆**：4 s 只下降了 27 cm | 2.8 s | 2.5 |
| dt 10 ms，γ 1e4，κ 10 GPa | 默认 | 1001 mm | **慢慢往下爬**，4 s 还没摆到最低点 | 4.8 s | 4.7 |
| dt 10 ms，γ 1e6，κ 100 GPa | 默认 | 1000 mm | 几乎不往下摆：4 s 只下降了 25 cm | 2.7 s | 2.8 |
| dt 2 ms，γ 1e4，κ 10 GPa | 默认 | 1003 mm | 慢慢往下爬 | 18.5 s | 2.9 |
| dt 10 ms，γ 1e6 | 完全收敛 | 1043 mm（钢条 +43） | 0.62 s 撞墙 | 21.8 s | 11.8 |
| dt 10 ms，γ 1e4，κ 10 GPa | 完全收敛 | 1001.7 mm | **越过**（最近 9.4 mm），但摆一次就停在底下 | 13.9 s（35 ms/步） | 12.8，最多 61 |
| dt 2 ms，γ 1e4，κ 10 GPa | 完全收敛 | 1000.4 mm | **越过 3 次**（最近 6.4 mm） | 68.7 s（34 ms/步） | 11.0 |

- **关节和刚体都调硬、再关掉半隐式停止，libuipc 能做对**：链长误差 1 mm 以内，摆锤越过墙。
- **只调硬、不关半隐式停止，结果是另一种错**：摆锤几乎停在空中（4 s 只下降二十几厘米），或者 4 s 才爬下来一半。半隐式停止在 Newton 还远没收敛时就停了，罚函数太硬时，每一步只走了一点点。
- **代价**：
  - dt 10 ms 时每步 35 ms，是 libuipc 默认设置的 3.7 倍，我们同样步长 1.0 ms。而且数值耗散大，摆一次就停了，我们 10 ms 还能越过 3 次。
  - dt 2 ms 时每步 34 ms，我们同样步长 1.0 ms。
- 所以和 MuJoCo 那一篇的结论一样：调参以后能做对，但要同时调三样东西（γ、κ、停止规则），每一样调错都是一种看得见的错；我们没有要调的。

**我们这边没有要调的：** 关节和刚性是坐标本身，换步长也只改变动力学的精度，不改变链长。

**我们这边没有要调的：** 关节和刚性是坐标本身，换步长也只改变动力学的精度，不改变链长。

## 5. 这个场景暴露了我们的一个问题（已修，阶段 13）

最初跑这个场景时，$h$ = 10 ms 每步平均 42 次 Newton，72% 的步撞上 50 次的上限，每步 73 ms，比 libuipc 默认设置（9.5 ms）慢 8 倍。撞上限的步照样不穿透、关节照样精确，但没有收敛。

**不是 CCD 的问题。** `scene_demo debug=2` 打印撞上限的步：每次迭代 CCD 允许的步长都是 1，线搜索每次只接受 1/16。所以是 Newton 方向不好：Hessian 对实际的能量估得太软。

**原因：Gauss-Newton 丢掉的那一项。** 阶段 3 的惯性 Hessian 是 $M(q)/h^2$，丢掉了“残差 × 正运动学的二阶导数”（[03_inertia.md](03_inertia.md) 第 4 节）。
- 平时这一项很小：刚体接近按惯性运动时残差接近 0。
- 这里残差就是链上 880 N 的拉力，乘上二阶导数，就是**被拉紧的绳子的几何刚度**。
- 看“之字形”模式（铰链 $k, k+1, k+2$ 转 $(\delta, -2\delta, \delta)$）：摆锤一阶不动，只有两节 23 g 的钢条在动。$M/h^2$ 给这个模式的曲率约 $m p^2/h^2$，拉力给的约 $Tp$ 量级，大几十倍。Newton 在这些方向上走得太远。
- $h$ 越小，$M/h^2$ 越大，这一项相对越小，所以 2 ms 时还算正常（6.8 次）。

**修法**见 [16_inertia_curvature.md](16_inertia_curvature.md)：把这一项（质心那一部分）加进每棵树的 Hessian 块，不正定的树退回 $M/h^2$。修好以后三个步长都是每步 2–3 次 Newton，$h$ = 10 ms 时每步 1.0 ms，比 libuipc 默认设置快 10 倍，比 libuipc 调对了的那一组（35 ms）快 35 倍。录像和特写用的是修好以后的轨迹。

## 6. 复现

```
python tools/make_wrecking_ball.py
python compare/libuipc/export_mjcf.py models/wrecking_ball.xml media/drift/wb.json
build\Release\scene_demo.exe model=wrecking_ball.xml T=4 h=0.01 gpu=1 mu=0.2 eps_v=0.01 tol=0.01 semi=1 out=media/drift/wb_ours_h10.csv
D:/Project/uipc-venv/Scripts/python.exe compare/libuipc/run_uipc_scene.py media/drift/wb.json media/drift/wb_uipc_dt10 --T 4 --dt 0.01 --newton
python tools/joint_check.py models/wrecking_ball.xml media/drift/wb_ours_h10.csv --status media/drift/wb_ours_h10.status
python tools/joint_check.py models/wrecking_ball.xml media/drift/wb_uipc_dt10.csv --affine media/drift/wb_uipc_dt10.npz --status media/drift/wb_uipc_dt10.status
python tools/render.py models/wrecking_ball.xml media/drift/wb_ours_h10.csv media/wrecking_ball_vs_libuipc.mp4 --pair models/wrecking_ball.xml media/drift/wb_uipc_dt10.csv --affine2 media/drift/wb_uipc_dt10.npz --keep_rgba pin,eye --status media/drift/wb_ours_h10.status --status2 media/drift/wb_uipc_dt10.status --w 1280 --h 544 --speed 0.5 --T 4 --az -18 --el 10 --range 3.0 --target 0,0,0.8 --title "RC-IPC (h = 10 ms)" --title2 "libuipc (dt = 10 ms, defaults)"
```

libuipc 的其他设置：`run_uipc_scene.py` 加 `--dt`、`--gamma`、`--kappa`、`--full --vel_tol 1e-3`。

`tools/render.py` 这次加了两样：
- `--affine`/`--affine2`：libuipc 的轨迹按每个刚体的仿射变换画。几何体的位置 = $A p + t$，朝向取 $A R_g$ 极分解里的转动，盒子三个方向的半长乘上 $A R_g$ 各列的长度。所以钢条的伸长、关节的拉开都画出来了（剪切画不出来，这里很小）。
- `--keep_rgba`：保留 MJCF 里写的颜色（孔和销子）。

## 7. 思考题

1. 罚函数关节的刚度是 $\gamma(m_A+m_B)$，为什么 libuipc 要按质量来定，而不是一个固定的数？在“轻链吊重物”里，这个选择为什么正好最吃亏？
2. 推导 $\delta \approx F h^2 / (2\gamma(m_A+m_B))$：罚函数能量进增量势时乘了什么？两个轴点为什么各分一半？用它估计 $h$ = 2 ms 时的漂移，和第 3 节的表比一比。
3. 仿射体的正交性能量 $\kappa V\|A^\top A - I\|_F^2$ 在单向拉伸下等效的杨氏模量是多少？为什么 κ = 100 MPa 的仿射体比钢软几百倍，在布娃娃堆里却看不出来？
4. libuipc 减小步长时，关节漂移变小，钢条的伸长反而变大。为什么？
5. 第 5 节的“之字形”模式：写出铰链 $k, k+1, k+2$ 转 $(\delta, -2\delta, \delta)$ 时链的末端移动了多少（到二阶），以及拉力 $T$ 做的功。由此算出这个模式上被丢掉的曲率。
6. 把丢掉的那一项加回 Hessian，压缩的链（比如一根竖直顶着重物的链）会出什么问题？为什么可以“试着分解，失败就退回 $M/h^2$”？
