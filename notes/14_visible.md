# 看得见的穿透：锁链和搅拌机

**问题：** 前面的场景（布娃娃堆、齿轮、Go2、G1 多米诺）就算穿透了，画面上也看不出来。
- **物体都是厚的。** 穿进去几毫米，藏在几厘米厚的躯干、脚底里面。G1 那一节里 MuJoCo 穿透了 40 mm，录像里还是看不出来，只能看右下角的数字。
- **穿透以后会被推回来。** 软接触把物体推出去，结局和不穿透差不多：机器人照样倒，方块照样堆着。
- 所以“不穿透”只能靠 `scene_demo` 打印的最小间隙（几百纳米）来证明，而这个数字是求解器自己算的。

**这一篇反过来设计场景：任何一处穿透都改变结局，而且改变的是拓扑。**
- **几何体比“允许穿透的方法”会穿进去的深度还薄**：锁链的线 6 mm 粗，搅拌机的墙 4 mm、叶片 3 mm 厚。
- **穿过去以后回不来。** 两个环脱开了就是脱开了；球到了罐子外面就留在外面。接触消失以后，结局还留着。
- **能用程序独立地查。** `tools/topology.py` 只拿记录下来的 `qpos`，用 MuJoCo 算正运动学：
  - 锁链：算每一对环的环绕数；
  - 搅拌机：看每个球的球心在不在罐子里。
  - 它不用求解器的距离计算，所以也是对“最小间隙 > 0”的一个独立检查。

两个场景都只用 MuJoCo 原生的几何（胶囊、球、盒子、圆柱），MuJoCo 和我们碰撞的是完全相同的几何体，同一个文件直接拿来对比。

| 场景 | 文件 | 生成方式 | 测试 |
|---|---|---|---|
| 锁链：24 个环套在横杆上，甩下去、堆在地上，再拎起来 | `models/ring_chain.xml` | `tools/make_stress.py chain` | `test_gpu.cc` 的 `Topology.RingChainStaysLinked` |
| 搅拌机：封闭的罐子里，100 rad/s 的转子打 20 个球 | `models/blender.xml` | `tools/make_stress.py blender` | `test_gpu.cc` 的 `Topology.BlenderKeepsBallsInside` |

视频在 `media/`（不进仓库）：`chain_vs_mujoco.mp4`、`blender_vs_mujoco.mp4`，左边是我们，右边是 MuJoCo。
- 左下角是 `topology.py` 逐帧写的状态：绿色是拓扑没变，红色是变了（几对环互相穿过、几个球出了罐子）。
- 右下角和以前一样：我们的最小间隙，或者 MuJoCo 这一刻最深的穿透。

---

## 1. 锁链

**几何。**
- 每个环是一个自由刚体：中心线是半径 $R$ = 20 mm 的正 12 边形，每条边一段胶囊，线的半径 $r$ = 3 mm（6 mm 粗）。钢的密度，每个环 38 g。
- 相邻的环互相垂直（一个在 xz 平面，一个在 xy 平面），节距 $p$ = 30 mm。拉直时，相邻两环的线在“扣”的地方相距 $2R - p$，减去两根线的半径，起始间隙是 $2R - p - 2r$ = 4 mm。
- 第 0 个环套在一根横杆上。横杆是 4 mm 半径的胶囊，两端各一个 20 mm 半径的挡球（比环的孔大，环滑不下去）。
- 横杆挂在一个竖直的滑动关节上，position 驱动 $k_p$ = 20000、`dampratio="1"`。

**过程。**
- 整串从水平伸直的姿态静止释放（横杆高 0.5 m，链长约 0.73 m）：甩下去，末端砸到地面，后半截堆在地上。
- 1.6–2.6 s，横杆的目标位置在 1 s 里升高 0.8 m（关键帧的 ctrl 时间表），把整串从地面上拎起来。
  - 挂着时链被拉开，节距接近 $2R - 2r$ = 34 mm，整串长约 0.8 m。
  - 拎到 2.6 s，最下面的环离地约 0.5 m。
- 只要有一处两根线互相穿过，两个环就脱开了，拎起来时整串断成两截，剩下的留在地上。

**环绕数。** 两条闭合曲线的环绕数（linking number）$\mathrm{Lk}$ 是整数，只有当两条曲线相交时才会变。对两个胶囊环来说，就是两根线的轴碰到一起，穿透深度至少是线的直径 $2r$。
- 所以 $\mathrm{Lk}$ 变了 $\iff$ 有一对环“穿过去了”。
- 起点：相邻的两个是 $\pm 1$，其余是 0。整段都不能变。另外查第 0 个环还套在横杆上（横杆的轴线穿过它围成的多边形）。
- **怎么算：** 沿一个一般的方向投影，数两条曲线在投影里的交叉。在每个交叉处，上面那段的方向叉乘下面那段的方向，和视线同向记 +1，反向记 −1，总和除以 2。
- 投影方向退化（正好有两段在投影里重合）的概率是 0；算出来的不是整数也能马上发现。

**结果**（GPU，`mu=0.3 eps_v=0.01 semi=1`；MuJoCo 3.6，implicitfast 加 CG）：

| | 步长 | 拓扑 | 最深穿透（刚体–刚体 / 刚体–世界） | 墙钟（3.5 s） |
|---|---|---|---|---|
| 我们 | $h$ = 2 ms，`tol=0.002` | 不变 | 无，最小间隙 84 nm | 4.7–4.8 s（2.7 ms/步，平均 4.3 次 Newton） |
| 我们 | $h$ = 5 ms，`tol=0.005` | 不变 | 无，最小间隙 65 nm | 2.9 s（4.1 ms/步，平均 4.4 次 Newton） |
| MuJoCo，默认接触 | dt = 1 ms | **0.245 s 第一对环脱开**，最后只有 11 个环还连在横杆上 | 6.0 / 22.1 mm | 0.90 s |
| MuJoCo，默认接触 | 0.5、0.25、0.1 ms | **0.235–0.245 s 脱开**，最后 9–10 个还连着 | 6.0 / 22–23 mm | |
| MuJoCo，最硬的接触 | dt = 1 ms | 不变 | 1.82 / 2.77 mm | 1.44 s |
| MuJoCo，最硬的接触 | 0.5 ms | 不变 | 0.70 / 1.25 mm | 3.19 s |
| MuJoCo，最硬的接触 | 0.25、0.1 ms | 不变 | 0.57 / 0.73，0.12 / 0.16 mm | |

- **MuJoCo 的墙钟**只量 `mj_step` 本身（单线程 CPU，`mujoco_traj.py --bench`）。我们的是 GPU（RTX 4060 Ti）。
- **“刚体–刚体”最深 6.0 mm**，正好是线的直径。穿过一半以后，最近点的法向翻过来，接触反而把线推过去；穿过以后接触就没了，所以记录到的最深穿透不会超过线的直径太多。
- **录像里看得很清楚：**
  - MuJoCo：0.245 s 链子绷紧的那一下，第 17、18 个环之间先断开，后来又断了几处，断下来的几截甩出去散在地上。拎起来时只带走了前 11 个环。
  - 我们：24 个环全部被拎起来。

**为什么 MuJoCo 默认的接触会断，换小步长也没用。**
- MuJoCo 的接触是软的：`solref` 的时间常数默认 0.02 s，接触像一个这个时间尺度的弹簧阻尼。0.245 s 甩下去的链子突然绷紧，扣着的两根线之间的力很大，线被压进去的深度超过了 6 mm，就穿过去了。
- **这和步长无关：** 弹簧的硬度由时间常数决定，dt 从 1 ms 减到 0.1 ms，断开的时刻和断开的环几乎不变（上表）。
- 把时间常数减到 MuJoCo 允许的最小值 $2\,dt$（`--timeconst 2dt`，所有 geom 的 `solref` 一起改，阻尼比 1），dt = 1 ms 就保住了拓扑，穿透 1.8 mm。dt 再减小，穿透跟着减小。
- 这时 MuJoCo 比我们快：1.44 s 对 2.9–4.8 s。

**公平地说：** MuJoCo 调了接触参数以后能跑这个场景，而且更快。我们这边：
- 没有接触参数可调（只有 $\hat d$，它管的是“多近算接触”，不是“能穿多深”）；
- 换步长也不会改变拓扑；
- 间隙始终是正的。

---

## 2. 搅拌机

**几何。**
- 封闭的罐子：12 块 4 mm 厚的盒子围成正 12 边形（内表面半径 0.30 m、高 0.35 m），顶上一块 4 mm 厚的盖子，底是地面。
  - 相邻两块墙在外面的角上重叠 2 mm，不留缝。世界上的几何体之间不检测，重叠没关系。
- 底部一个转子：竖直的铰链，中间一个圆柱，4 片叶片。
  - 叶片是 3 mm 厚、6 cm 高的钢板，从半径 0.03 m 伸到 0.26 m。
  - velocity 驱动 $k_v$ = 20，0.2 s 里目标从 0 加到 100 rad/s，叶尖速度 26 m/s。
- 20 个半径 12 mm、8 g 的球（密度 1100）随机摆在转子上方 0.12–0.32 m 处，静止释放，约 0.2 s 落到叶片上。

**过程。**
- 叶片把球打出去：0.21 s 时最快的球 29.7 m/s，0.24 s 时已经有 16 个球超过 5 m/s。
- 球撞墙、撞盖子、互相撞，再被叶片打。
- 只要有一次穿透（叶片穿过球，或者球穿过墙、盖子），球就会到罐子外面去。
- **检查：** 每个球的球心离转轴是否超过墙的外表面（按它所在的那块墙的法向算），是否高于盖子的上表面。

**结果**（1.5 s）：

| | 步长 | 球出了罐子 | 最深穿透（刚体–刚体 / 刚体–世界） | 墙钟（1.5 s） |
|---|---|---|---|---|
| 我们 | $h$ = 1 ms | 0 | 无，最小间隙 53 nm | 12.3 s（平均 5.6 次 Newton，3 步撞上限） |
| 我们 | $h$ = 2 ms | 0 | 无，最小间隙 117 nm | 15.8–18.7 s（平均 10.7 次 Newton，约 18 步撞上限） |
| 我们 | $h$ = 5 ms | 0 | 无，最小间隙 124 nm | 27.7 s（平均 37 次 Newton，300 步里 206 步撞上限） |
| MuJoCo，默认接触 | dt = 1 ms | **18 个**（第一个在 0.222 s） | 13.5 / 15.4 mm | 0.04 s |
| MuJoCo，默认接触 | 0.5–0.05 ms | **18–19 个** | 13.5 / 14.0 mm | |
| MuJoCo，最硬的接触 | dt = 1 ms | **3 个**（0.230–0.244 s） | 13.4 / 13.0 mm | 0.03 s |
| MuJoCo，最硬的接触 | 0.5 ms | 0 | 10.2 / 11.5 mm | 0.05 s |
| MuJoCo，最硬的接触 | 0.25 ms | 0 | 5.6 / 3.3 mm | 0.09 s |
| MuJoCo，最硬的接触 | 0.1、0.05 ms | 0 | 1.8 / 2.4，1.2 / 0.9 mm | |

- **穿出去的条件**：MuJoCo 那几行最深的穿透都在 13–15 mm 附近。球心越过墙的中面，要穿进 $r + t/2$ = 12 + 2 = 14 mm；越过叶片的中面，要 12 + 1.5 = 13.5 mm。过了这个深度，接触的法向翻过来，把球推到另一边去。
- **默认接触：** 和锁链一样，减小 dt 没用。时间常数 0.02 s 的弹簧挡不住 25 m/s 的球：要在 14 mm 里停下来，得在约 1 ms 里把速度减到 0，比 0.02 s 短得多。
- **最硬的接触：**
  - dt = 1 ms（时间常数 2 ms）还挡不住，3 个球出去了。
  - 0.5 ms 时都留在罐子里，但最深 11.5 mm，离 14 mm 只差 2.5 mm。
  - 要有余量，得 0.25 ms 以下。
- **我们慢得多：**
  - MuJoCo 0.05–0.09 s，我们 12–28 s，差 100–300 倍。
  - 这个场景几何很少（39 个 geom），MuJoCo 每步只要十几微秒。
  - 我们的时间都花在 0.2–0.5 s 的高速撞击上：每步几十次 Newton，$\kappa$ 涨到 $3\times10^{10}$。0.6 s 以后，球都停在叶尖和墙之间的那一圈里，每步约 2 ms。

**我们这边要说清楚的几件事。**
- **撞上限的步。**
  - Newton 的上限是 50 次，撞上限时这一步就用最后一个迭代点。
  - 每个迭代点都经过 CCD，所以撞上限的步照样不穿透（上表的最小间隙），但没有收敛，那一步的物理不准。
  - $h$ = 5 ms 时三分之二的步撞上限：转子每步转 0.5 rad，Newton 走得很慢。这个还没有细查（第 4 节）。
  - $h$ = 1 ms 反而最快：每步的 Newton 次数少得多，总时间最短。
- **转子转得比 100 rad/s 慢。**
  - $h$ = 2 ms 时稳态是 97 rad/s，正好是阶段 12 笔记第 4 节的惯性项阻力：
    $$\frac{2I}{h^2}(1-\cos h\omega)\sin h\omega = k_v(u-\omega)$$
    转子的 $I \approx 0.033$ kg m²，$h\omega \approx 0.19$，左边约 60 N m，右边 $20\times 3$ = 60 N m。
  - 正在打球的那一段（0.2–0.5 s）是 80–88 rad/s。
  - MuJoCo 是 99.7–100 rad/s。
- **我们的球很快就“安静”了。**
  - 0.6 s 以后，我们的 20 个球都停在叶尖（0.26 m）和墙（0.30 m）之间的那一圈地面上。叶片够不着它们：缝 40 mm，比球的直径 24 mm 大。
  - MuJoCo（最硬的接触、0.5 ms）的球到 1.5 s 还在罐子里弹。
  - 我们的耗散大得多：隐式欧拉的数值阻尼，加上障碍函数的接触本身没有回弹。这一点两边的动力学不一样，只有“出没出罐子”可以比。

---

## 3. 复现

```
python tools/make_stress.py chain
python tools/make_stress.py blender
build\Release\scene_demo.exe model=ring_chain.xml T=3.5 h=0.002 gpu=1 mu=0.3 eps_v=0.01 tol=0.002 semi=1 out=media/visible/chain_ours.csv
build\Release\scene_demo.exe model=blender.xml T=1.5 h=0.002 gpu=1 mu=0.3 eps_v=0.01 tol=0.002 semi=1 out=media/visible/blender_ours.csv
python tools/mujoco_traj.py models/ring_chain.xml "media/visible/chain_mj_{dt}ms.csv" --T 3.5 --dt 0.001,0.0005,0.00025,0.0001 --every 0.005
python tools/mujoco_traj.py models/ring_chain.xml "media/visible/chain_mj_{dt}ms_{tc}.csv" --T 3.5 --dt 0.001,0.0005 --every 0.005 --timeconst 2dt --bench
python tools/mujoco_traj.py models/blender.xml "media/visible/blender_mj_{dt}ms.csv" --T 1.5 --dt 0.001,0.0005,0.00025,0.0001,0.00005 --every 0.002
python tools/mujoco_traj.py models/blender.xml "media/visible/blender_mj_{dt}ms_{tc}.csv" --T 1.5 --dt 0.001,0.0005,0.00025 --every 0.002 --timeconst 2dt --bench
python tools/topology.py chain models/ring_chain.xml media/visible/chain_ours.csv --status media/visible/chain_ours.status
python tools/topology.py chain models/ring_chain.xml media/visible/chain_mj_1ms.csv --status media/visible/chain_mj_1ms.status
python tools/topology.py blender models/blender.xml media/visible/blender_ours.csv --status media/visible/blender_ours.status
python tools/topology.py blender models/blender.xml media/visible/blender_mj_1ms.csv --status media/visible/blender_mj_1ms.status
python tools/render.py models/ring_chain.xml media/visible/chain_ours.csv media/chain_vs_mujoco.mp4 --pair models/ring_chain.xml media/visible/chain_mj_1ms.csv --status media/visible/chain_ours.status --status2 media/visible/chain_mj_1ms.status --w 1280 --h 544 --speed 0.5 --az 20 --el 10 --range 2.2 --target 0.22,0,0.66 --title "RC-IPC (h = 2 ms)" --title2 "MuJoCo (dt = 1 ms, default contacts)"
python tools/render.py models/blender.xml media/visible/blender_ours.csv media/blender_vs_mujoco.mp4 --pair models/blender.xml media/visible/blender_mj_1ms.csv --status media/visible/blender_ours.status --status2 media/visible/blender_mj_1ms.status --glass wall,lid --w 1280 --h 544 --speed 0.1 --T 0.8 --az 35 --el 38 --range 1.5 --target 0,0,0.12 --title "RC-IPC (h = 2 ms)" --title2 "MuJoCo (dt = 1 ms, default contacts)"
```

`tools/mujoco_traj.py` 这次加了三样：
- 控制量按关键帧的时刻插值（和 `ctrl_at` 一样，锁链的横杆要用）；
- `--timeconst`（接触的时间常数）；
- `--bench`（只量 `mj_step` 的墙钟）。

`--dt` 可以给几个值，输出文件名里的 `{dt}`、`{tc}` 会换成步长和时间常数的设置。

---

## 4. 还差的

- **搅拌机里 Newton 为什么慢。** $h$ = 5 ms 时每步平均 37 次 Newton，三分之二撞上限。猜测是转子每步转 0.5 rad，CCD 每次只放它走到下一次接触，加上惯性项对大转角的非线性。要用 `debug=1` 看每次迭代的步长再说。
- **更公平的速度对比。** MuJoCo 的墙钟是单线程 CPU，我们是 GPU。这两个场景都很小（25 个刚体以下），GPU 吃不满。几十条锁链、几十个罐子一起跑时才能比吞吐量。
- **回弹。** 我们的接触没有回弹（隐式欧拉加障碍函数），球很快就停了。要比动力学（不只是拓扑），得先把这一点对齐，或者换一个不靠回弹的场景。

---

## 5. 思考题

1. 两个胶囊环的环绕数要变，两根线的轴必须相交。这对应多深的穿透？MuJoCo 的锁链断了，记录到的最深穿透却只有 6.0 mm，为什么不是更深？
2. 用一个临界阻尼的弹簧（时间常数 $\tau$）挡一个速度 $v$ 的球：估计最深能压进去多少。代入 $v$ = 25 m/s，$\tau$ = 0.02 s 和 2 ms，和表里 MuJoCo 的数字比一比。为什么减小 dt 没用，减小 $\tau$ 才有用？
3. 我们的 CCD 保证每个迭代点都不穿透。撞上 Newton 上限的步，结果还是不穿透的吗？它“错”在哪里？
4. 环绕数用投影来算。如果投影方向正好让两段线在投影里重合，会出什么问题？为什么实际上不用担心？
5. $h$ = 2 ms 时转子的稳态是 97 rad/s。用阶段 12 第 4 节的公式，算一下 $h$ = 1 ms 时是多少；$k_v$ 要多大才能让它到 99 rad/s？
6. 搅拌机里我们的球很快停在外圈，MuJoCo 的还在弹。真实的橡胶球更像哪一边？我们的能量是在哪几个地方耗掉的？
