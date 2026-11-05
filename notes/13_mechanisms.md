# 机构和机器人场景（阶段 12 之后）

**目标：** 用阶段 12 的驱动、阻尼，跑几个前面没有的场景：被驱动的齿轮组、夹爪抓取，真实的机器人模型（Unitree Go2）堆在一起，Allegro 手抓住物体再翻过来，以及 Unitree G1 人形机器人的多米诺。

为此读模型时加了几样东西：
- 圆柱（cylinder）geom（为了 Go2）；
- 碰撞过滤改成 MuJoCo 的写法，支持 MJCF 的 `<exclude>`（为了 Allegro 手）；
- 关节的干摩擦 `frictionloss` 和关节上的驱动力矩限制 `actuatorfrcrange`（为了 G1，阶段 12 笔记第 5 节）；
- 场景程序从第一个关键帧的姿态开始。

| 场景 | 文件 | 生成方式 | 测试 |
|---|---|---|---|
| 齿轮组 | `models/gears.xml` | `tools/make_gears.py` | `test_gpu.cc` 的 `Mechanism.GearsTransmit` |
| 夹爪抓方块 | `models/gripper.xml` | 手写 | `test_mechanisms.cc` 的 `Gripper.LiftsCube`，`test_gpu.cc` 的 `Mechanism.GripperMatchesCpu` |
| 圆柱 | `models/cylinders.xml` | 手写 | `Cylinder.Geometry`、`Cylinder.RestsOnGroundAndBox` |
| 24 只 Go2 | `models/local/go2_pile.xml`（不进仓库） | `tools/make_robot_pile.py <go2.xml>` | 只有演示 |
| 3 只 Allegro 手抓住再翻过来 | `models/local/allegro_grasp.xml`（不进仓库） | `tools/make_hand_grasp.py <right_hand.xml>` | 只有演示；过滤的测试见第 6 节 |
| 碰撞过滤 | `models/filters.xml` | 手写 | `Filter.WeldsAndExcludes`，`test_gpu.cc` 的 `Terms.FilteredPairs` |
| 6 台 G1 多米诺 | `models/local/g1_domino.xml`（不进仓库） | `tools/make_humanoids.py <g1.xml>` | 只有演示；干摩擦和力矩限制的测试见阶段 12 笔记第 5 节 |

视频用 `tools/render.py` 渲染到 `media/`（不进仓库），命令见第 8 节。

---

## 1. 齿轮组

**模型。** 一排 4 个一样的直齿轮，16 个齿，模数 1 cm（节圆半径 8 cm），厚 2 cm，钢的密度（每个 2.9 kg）。齿形是渐开线，压力角 20°。
- 每个齿轮装在世界上的铰链上（绕 z 轴），中心距等于两倍节圆半径。相邻的齿轮错开半个齿，节圆上留 3 mm 的侧隙，开始时互不接触（最近 2.5 mm）。
- gear0 是主动轮：velocity 驱动（$k_v$ = 5 N m s/rad），目标转速在 0.2 s 里从 0 加到 −100°/s。
- 最后一个齿轮的铰链阻尼 0.02 N m s/rad 当负载，所以转矩要经过每一对啮合的齿。重力沿转轴，不产生力矩。

**网格**（`make_gears.py`）：齿廓一圈逆时针拉伸成柱体，两个端面从中心扇形三角化（齿廓对中心是星形的）；每个齿轮 418 个顶点、832 个三角形。三角形都要朝外，MuJoCo 才能算出正的体积和惯量。

**结果**（GPU，`mu=0`，3 s）：

| t | gear0 | gear1 | gear2 | gear3 | 比值 1/0、2/0、3/0 |
|---|---|---|---|---|---|
| 0.5 s | −39.7° | 38.1° | −36.5° | 35.0° | −0.960、0.922、−0.882 |
| 1.0 s | −89.4° | 87.9° | −86.3° | 84.7° | −0.982、0.965、−0.947 |
| 2.0 s | −189.0° | 187.5° | −185.9° | 184.5° | −0.992、0.984、−0.976 |

- 一样的齿轮相邻啮合，第 k 个转了主动轮的 $(-1)^k$ 倍；每往后一个落后约 1.5°，这是侧隙。转角越来越大，比值就越来越接近 ±1。
- 主动轮 0.5 s 以后的转速是 −99.6°/s，比目标慢一点：velocity 驱动要输出转矩，才能带动负载。
- 不穿透，最近 0.48 mm。每步 3.6 ms，平均 4.3 次 Newton。
- CPU 版一步约 0.25 s（网格大，刚体之间的检测是暴力的），所以只在 GPU 上跑。

---

## 2. 夹爪抓方块

**模型。** 树是 world → lift（沿 z 滑动）→ 左右两个手指（沿 x 滑动），三个都是 position 驱动。方块 5 cm、0.1 kg，放在地面上方 0.5 mm。控制量的时间表写在关键帧里：
- 0–0.4 s 往下 5.5 cm；
- 0.6–0.9 s 两个手指各往里 3.5 cm，比方块的侧面多 5 mm，夹紧力约 $k_p \times$ 5 mm = 2 N；
- 1.1–1.8 s 往上提 15.5 cm，之后不动。

方块重 0.98 N，摩擦系数 0.5 时两侧最多能给 2 N，提得起来。

**结果**（CPU，`mu=0.5`，2.5 s）：lift 升了 15.5 cm，方块跟着升了 15.4 cm，在手指里只滑了约 0.4 mm（光滑化摩擦在“静止”时也允许 $\epsilon_v$ 以下的蠕动）。手指被方块挡在离目标 5 mm 的地方，正好是 2 N 的夹紧力。最近 14 μm，每步 0.2 ms，平均 3.1 次 Newton。

GPU 的两种 Newton 迭代都和 CPU 逐步比了 250 步（`Mechanism.GripperMatchesCpu`），图版确实用上了。这也是驱动在 CUDA graph 里的测试：`drives.xml` 没有顶点，用不上图版。

---

## 3. 圆柱

圆柱做成正 12 边形的棱柱：两圈顶点、两个端面的中心，侧面每格两个三角形，端面扇形三角化，边包括两圈、竖边、侧面的对角线和端面的辐条。每个圆柱 26 个顶点、72 条边、48 个三角形。多边形内接于圆，所以比真圆柱略小，半径上差 $1 - \cos 15° = 3.4\%$。

`Cylinder.RestsOnGroundAndBox`：
- 竖着落下的圆柱停在半长的高度；
- 横着落下的停在 $r\cos 15°$：一个侧面着地；
- 第三个落到盒子上。

第一版测试要求横着的那个停在 $r\cos15°$，结果是 $r$：它开始时正好一条棱朝下，左右对称，就一直立在这条棱上。这是不稳定的平衡，没有扰动就不会倒。后来让它绕轴转 15°，一个侧面朝下。

---

## 4. 24 只 Go2

**模型**：`tools/make_robot_pile.py` 把 MuJoCo Menagerie 的 Unitree Go2 复制 24 份。
- Go2 本身：13 个刚体，12 个限位铰链加一个浮动基座；碰撞体是 5 个盒子、12 个圆柱、4 个球。
- 摆法：排成两层的方阵，从 0.5–1.1 m 高摔到三级台阶和几块石头上。每只随机一个朝向和一点侧倾，腿摆成站立的姿势再加一点随机（关键帧 0 的 `qpos`）。
- 合起来 312 个刚体、24 棵树、432 个自由度、8616 个顶点。

Go2 的模型文件不在仓库里，生成的 MJCF 写到 `models/local/`（被忽略）。

**第一版的起始姿态就是穿透的**：方阵间距 0.7 m，而一只 Go2 连腿长约 0.7 m，随机转向以后相邻的两只叠在一起（MuJoCo 自己的检测也报了 17 个接触，最深 5 cm）。现在间距 0.9 m，生成器用 MuJoCo 的检测（1 cm 的余量）检查起始姿态，有接触就重新抽。

**忽略的设置**（读模型时记进 `Model::ignored`，`scene_demo` 开头打印）：第一版忽略了关节的 `frictionloss`（干摩擦）和 `actuatorfrcrange`。为 G1 补上以后（阶段 12 笔记第 5 节），只剩浮动基座上的干摩擦、阻尼不支持。Go2 的这个文件里没有驱动，所以这是一堆“没有电机的”机器人，腿是软的，只有关节的干摩擦。

**结果**（GPU，`dhat=0.005 eps_v=0.01 tol=0.05 semi=1`，3 s）：
- 每步平均 4.0 ms；冲击时最慢约 12 ms，静止以后约 1.1 ms。
- 平均 3.4 次 Newton（最多 36 次，没有撞上限），每步 10.9 次 PCG。
- 不穿透，最近 0.45 μm；关节都在限位里，最近 0.3 mrad。
- 补上关节的干摩擦以后重跑：每步 4.6 ms，平均 3.8 次 Newton，每步 11.9 次 PCG，最近 0.43 μm，关节最近 0.3 mrad。

旧分支在 RTX 3090 上报的同类场景（24 只四足、432 自由度）是 5.6 ms，但场景不完全一样（台阶、石块、碰撞体、参数），只能作参考。

---

## 5. Allegro 手：抓住再翻过来

**模型**：`tools/make_hand_grasp.py` 用 MuJoCo Menagerie 的 `wonik_allegro/right_hand.xml`（BSD 许可）。
- 每只手：手掌加 4 个手指，每个手指 4 个铰链，共 16 个，都有限位、关节阻尼 0.1 和 position 驱动；碰撞体是盒子和胶囊。
- 摆法：3 只手排成一排，手心朝上，手心里各放一个物体（半径 3 cm 的球、圆柱、方块），离手心 2 mm。
- 每只手挂在一个 mount 刚体上。mount 绕世界 x 轴（手指的方向）转，转轴穿过物体的中心，也是 position 驱动（$k_p$ = 300、$k_v$ = 5）。
- 合起来 69 个刚体、69 个自由度、51 个驱动、15 对 `<exclude>`。

时间表写在关键帧里：0–0.2 s 手张开、物体落到手心；0.2–0.8 s 手指的目标从张开线性变到握拳；0.8–1.8 s mount 转 180°，手心朝下，物体只靠手指和摩擦挂住；之后不动。
- 手指的目标超过了碰到物体的位置，握力就是 $k_p(\text{目标} - q)$。
- Menagerie 的手指驱动 $k_p$ = 1，这里加到 3（旧分支也是）。用 `mu=0.8` 跑。

**结果**（GPU，3.5 s）：三个物体都挂住了，3.5 s 时都还在转轴的高度（z ≈ 0.30 m）。
- 每步平均 11 ms，平均 11.7 次 Newton（最多 50 次，2 步撞上限），每步 160 次 PCG。
- 不穿透，最近 2.6 μm；关节都在限位里。
- CPU 上一步约 0.12 s。

**做出来之前改过的几处：**
- **起始姿态。** 圆柱和方块比球宽，放在旧分支的位置会压到拇指根部（拇指根部比手心高约 2 mm）。生成器用 MuJoCo 的检测检查起始姿态，有 1 mm 以内的接触就报错。MuJoCo 把两个 geom 的 margin 相加，所以每个 geom 的 margin 要设成 0.5 mm。圆柱和方块往远离拇指、靠近手指的方向挪了 1.3 cm 和 0.7 cm。
- **指尖和中节重叠。** 第一次在我们这边跑，第一步就是 −3 mm 的“穿透”，是同一根手指的指尖和中节。指尖没有关节，焊在末节上；MuJoCo 把焊在一起的刚体当成一个整体，指尖和中节因此算父子，不检测。我们原来只看刚体自己的父刚体，两者隔了一层，于是去检测了。见第 6 节。
- **球掉了。** 第一版球放在手心中间（手掌坐标系 z = −0.035），翻过来以后在 1.8–2.5 s 之间从手腕那一侧滚出去，落到地上。往手指的方向挪 1 cm（z = −0.025）以后就挂住了，z = −0.015 也行。手指弯过来包住的是靠指根的那一半，物体离手腕越近，翻过来以后越容易从手腕那边滚出去。

---

## 6. 碰撞过滤

`Model::collide(a, b)` 决定两个刚体要不要检测（CPU 的宽相、GPU 的 `k_body_pairs` 都用它）。现在和 MuJoCo 的规则一样：
- 比较“焊在一起的整体”：没有关节的刚体和它的父刚体是一个整体，即 `body_weldid`，存在 `Model::weld` 里。同一个整体里的两个刚体不检测；
- 两个整体是父子时不检测：一个整体的父刚体所在的整体，正好是另一个整体。父子过滤可以用 `filterparent` 关掉；世界照旧除外；
- MJCF 的 `<contact><exclude body1 body2/>` 列出的对不检测：`Model::exclude`，编码成 min·nbody + max 排好序，CPU 用二分查找，GPU 也是。

每个刚体都有关节时，和原来的规则完全一样。不一样的模型目前只有 `zoo.xml`：它的 hand 焊在 forearm 上，hand 和 arm、finger 和 forearm 现在不再检测。`Energy.GradientMatchesFiniteDifference` 在这个模型上，有一个自由度的梯度因此变小，相对容差跟着变紧。原来 1e-6 的差分步长，舍入误差超过了容差，换成 1e-5 以后就过了。

测试 `filters.xml`：a → b → c（焊在 b 上）→ d，另一个自由刚体 e，`<exclude>` 去掉 a–e；胶囊挤在一起，被过滤的几对互相穿透 2 cm。
- `Filter.WeldsAndExcludes` 逐对检查 `collide()`，再检查检测到的接触里没有被过滤的对。
- `Terms.FilteredPairs`（GPU）在 $\hat d$ = 0.2 m 时比较 GPU 和 CPU 的接触个数、能量、梯度、Hessian。GPU 要是没过滤，穿透的那几对会让障碍能量变成无穷，对不上。

---

## 7. G1 人形：多米诺

**模型**：`tools/make_humanoids.py` 用 MuJoCo Menagerie 的 `unitree_g1/g1.xml`（Unitree 的 BSD 许可）。
- 每台 30 个刚体：浮动基座加 29 个限位铰链，35 个自由度。
- 每个铰链一个 position 驱动，$k_p$ = 500，$k_v$ 由 `dampratio="1"` 在 MuJoCo 编译时算出（每个关节不一样，4.6–43）。
- 关节 armature 0.01、干摩擦 0.3 N m，`actuatorfrcrange` 限制关节力矩：腿 88–139 N m，腰和踝 50，手臂 25，手腕俯仰和偏航 5。后两样是为它补的（阶段 12 笔记第 5 节），补之前读模型时只记进 `ignored`。

**碰撞体换成凸包。** 24 个碰撞网格一共 21.9 万个三角形，最大的躯干 5.1 万个。
- MuJoCo 自己碰撞时用的是每个网格的凸包（qhull），但曲面细分得很密，凸包上的顶点也多，24 个凸包还有 5.6 万个三角形。
- 生成器让 MuJoCo 算凸包时最多取 64 个顶点（MJCF 的 `maxhullvert`），把凸包从网格坐标系换到刚体坐标系，作为新的网格直接写进 MJCF（`vertex`、`face` 属性），碰撞 geom 改用它。每台 24 个凸包一共 2976 个三角形，少了 74 倍。
- 坐标系要换，是因为 MuJoCo 编译时把网格挪到它的质心、转到主轴上（`mesh_vert` 在这个坐标系里），这个变换并进了 `geom_pos`、`geom_quat`。新网格的顶点直接写成刚体坐标系里的位置，geom 的位置和姿态清零，MuJoCo 再编译一遍也还是同样的位置。
- 三角形的朝向统一朝外（凸包的每个面都背着中心），否则 MuJoCo 算网格体积时会出问题。
- 这样生成的文件里，MuJoCo 和我们用的是同一套碰撞体，下面和 MuJoCo 的对比只差在接触和积分的做法上。
- 我们的读模型本身不取凸包，直接用网格的三角形：IPC 能处理非凸的网格（第 1 节的齿轮就是），这是比 MuJoCo 多的能力。这里取凸包只是为了少三角形。

**场景。** 6 台排成一列，面朝 +x，间距 0.7 m，都摆成模型自带的 stand 姿势，驱动的目标就是这个姿势（关键帧里的 ctrl）。第一台绕前脚尖往前倾 0.3 rad，倒下撞第二台，依次倒下。
- stand 关键帧里脚底比地面低 1.9 mm（MuJoCo 的软接触允许），每台抬到最低的脚底球离地 1 mm。
- 生成器检查起始姿态：除了脚底和地面，1 mm 以内不能有接触。
- `attach_body` 会把每台自带的 stand 关键帧也搬过来：那一帧里只有这一台的姿态，其余几台在 `qpos0`，全叠在原点。第一次检查起始姿态就报了 718 个接触。现在搬完以后删掉这些帧，换成整个场景的一帧。

**结果**（GPU，$h$ = 5 ms，`dhat=0.002 mu=0.6 eps_v=0.01 tol=0.01 semi=1`，5 s）：
- 每步 6.4–6.5 ms（两次运行），平均 3.7 次 Newton（最多 17–26 次，没有撞上限），每步约 30 次 PCG，每个 Newton 迭代约 120 个接触。
- 不穿透，最近 0.37–0.54 μm；关节都在限位里，最近 0.6 mrad。
- 180 个刚体、210 个自由度、174 个驱动、9888 个顶点、19008 个三角形。

**和 MuJoCo 比**（`tools/mujoco_traj.py`：同一个文件，MuJoCo 3.6，dt = 1 ms，implicitfast 加 CG，接触参数全用默认值）。下表是每台的骨盆降到 0.5 m 以下的时刻（s）：

| | 第 1 台 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| MuJoCo | 1.14 | 1.41 | 1.77 | 2.08 | 2.42 | 2.60 |
| 我们，$h$ = 5 ms（5 次运行的范围） | 1.15–1.16 | 1.43–1.46 | 1.76–1.85 | 2.08–2.25 | 2.42–2.60 | 2.63–2.81 |
| 我们，$h$ = 10 ms | 1.22 | 1.69 | 2.25 | 2.86 | 3.35 | 3.65 |

- **第一台倒下之前不碰别的东西**，只有驱动、关节摩擦、惯性和脚底的接触。$h$ = 5 ms 时和 MuJoCo 差 0.01–0.02 s。
- **往后越差越多。** GPU 上原子加的顺序每次不一样，两次运行从舍入开始就不同。每撞一次，差别就放大一次，到第 6 台，5 次运行之间差 0.18 s。MuJoCo 落在这个范围的早的那一头，或者比最早的再早一点（第 6 台 2.60 对 2.63）。
- **$h$ = 10 ms 明显慢。** 隐式欧拉的数值耗散让倒下的转动变慢（阶段 12 笔记第 4 节那种阻力，加上隐式欧拉本身的阻尼），每倒一台都慢一点，累积起来最后一台晚了 1 s。所以这个场景用 5 ms。
- **穿透。** MuJoCo 这一段里最深的穿透：刚体和地面 40.4 mm（最后一台的躯干在 2.74 s 脸朝下砸到地面），刚体和刚体 20.1 mm（两台的躯干撞在一起）。站着的时候脚底一直陷进地面约 0.3 mm。这是 MuJoCo 默认的软接触（`solref` 的时间常数 0.02 s），调硬接触参数、减小步长都能减少，这里没有调。我们这边整段没有穿透。
- 并排的录像 `media/g1_vs_mujoco.mp4`，左边是我们，右边是 MuJoCo，右下角写着这一刻的最小间隙或最深穿透（`tools/render.py` 看到负的间隙就写成“最深穿透”）。

**`tol` 的一个坑。** 第一次用 $h$ = 5 ms 跑，所有机器人一动不动，每步 1 次 Newton。收敛判据是“顶点位移 / $h$ < `tol`”（IPC 的写法）。从静止开始，一步里重力带来的速度变化是 $gh$ = 0.049 m/s，比 `tol` = 0.05 m/s 小，所以第一次迭代、还没动就判为收敛了，每一步都是这样。`tol` 改成 0.01 就好了。$h$ = 10 ms 时 $gh$ = 0.098，才勉强躲过去。`tol` 必须明显小于 $gh$。

**时间花在哪里**（$h$ = 10 ms 时逐个发射的计时）：每步 10 ms 里 CCD 占 4 ms。接触不多（每个迭代约 100 个），但有一些真正截断的对，ACCD 只能一步一步走（GPU 笔记 `11_gpu.md` 第 14.1 节最后一段）。

---

## 8. 复现

```
python tools/make_gears.py models/gears.xml
python tools/make_robot_pile.py <menagerie>/unitree_go2/go2.xml models/local/go2_pile.xml --n 24
build\Release\scene_demo.exe model=gears.xml T=3 mu=0 gpu=1 out=media/gears.csv
build\Release\scene_demo.exe model=gripper.xml T=2.5 mu=0.5 out=media/gripper.csv
build\Release\scene_demo.exe model=<仓库>/models/local/go2_pile.xml T=3 gpu=1 dhat=0.005 eps_v=0.01 tol=0.05 semi=1 out=media/go2_pile.csv
python tools/render.py models/gears.xml media/gears.csv media/gears.mp4 --speed 1 --el 75 --az 90 --range 1.0 --target 0.24,0,0
python tools/render.py models/gripper.xml media/gripper.csv media/gripper.mp4 --speed 0.5 --el 12 --az 70 --range 0.8 --target 0,0,0.17
python tools/render.py models/local/go2_pile.xml media/go2_pile.csv media/go2_pile.mp4 --speed 1 --el 25 --az 35 --dist 0.9
python tools/make_hand_grasp.py <menagerie>/wonik_allegro/right_hand.xml models/local/allegro_grasp.xml --n 3
build\Release\scene_demo.exe model=<仓库>/models/local/allegro_grasp.xml T=3.5 mu=0.8 gpu=1 out=media/allegro.csv
python tools/render.py models/local/allegro_grasp.xml media/allegro.csv media/allegro.mp4 --speed 0.5 --el 20 --az 150 --range 0.9 --target 0,0.3,0.3
python tools/make_humanoids.py <menagerie>/unitree_g1/g1.xml models/local/g1_domino.xml --n 6
build\Release\scene_demo.exe model=<仓库>/models/local/g1_domino.xml T=5 gpu=1 mu=0.6 dhat=0.002 eps_v=0.01 tol=0.01 semi=1 out=media/g1_domino.csv
python tools/mujoco_traj.py models/local/g1_domino.xml media/g1_domino_mujoco.csv --T 5 --dt 0.001
python tools/render.py models/local/g1_domino.xml media/g1_domino.csv media/g1_domino.mp4 --speed 1 --az 125 --el 12 --range 4.6 --target 1.9,0,0.55
python tools/render.py models/local/g1_domino.xml media/g1_domino.csv media/g1_vs_mujoco.mp4 --pair models/local/g1_domino.xml media/g1_domino_mujoco.csv --w 1280 --h 544 --speed 1 --az 140 --el 16 --range 4.4 --target 2.0,0,0.45 --title "RC-IPC (h = 5 ms)" --title2 "MuJoCo (dt = 1 ms, default contacts)"
```

---

## 9. 还差的

- **G1 能站着动起来**：现在的 G1 只是“驱动锁住姿势的雕像”。站稳、走路要一个平衡控制器（或者回放一段 MuJoCo 里录好的控制量），那是控制的问题，不是接触的问题。带灵巧手的 `g1_with_hands.xml` 也可以照同样的办法生成。
- **更多的手**：旧分支跑到过 64 只手。这里 3 只时每步 11 ms，大部分是 PCG（每步 160 次）：手指的驱动很软，接触很硬，系统的条件数大。
- **闭环**（自行车链、活塞、履带车）：要等式约束，和“关节精确”的说法冲突（路线图第 3 类）。
- **和 MuJoCo 的对比**只用了默认的接触参数。公平一点，应该把 MuJoCo 的接触调硬、步长调小，看穿透和速度怎么变。

---

## 10. 思考题

1. 齿轮的转角比值为什么随时间越来越接近 ±1？用侧隙算一下 2 s 时 gear3 应该落后多少度，和表里的比一比。
2. velocity 驱动的主动轮最后是 −99.6°/s。负载阻尼 0.02 N m s/rad 时，算一下稳态的转速差（别忘了阶段 12 第 4 节的惯性阻力）。
3. 夹爪提起方块时，方块在手指里滑了约 0.4 mm。光滑化摩擦在静止时允许多大的滑动速度？把 `eps_v` 减小 10 倍会怎样，代价是什么？
4. 圆柱横着时停在一条棱上。在真实的物理里它会倒吗？我们的模拟为什么不倒？
5. Go2 的腿上没有阻尼也没有驱动，落地以后很快就不动了（每步 1 次 Newton）。能量去哪了？
6. Allegro 手的握力是 $k_p(\text{目标} - q)$。翻过来以后，物体的重量由哪几个力平衡？把 $k_p$ 从 3 降回 Menagerie 的 1，估计哪个物体先掉？
7. 第 6 节的过滤里，为什么“同一个整体里的两个刚体”根本不用检测？如果检测了，会出什么问题（提示：它们之间的距离永远不变）？
8. G1 的碰撞体取了最多 64 个顶点的凸包。顶点少了，凸包比原来的凸包小还是大？对“不穿透”的保证有什么影响（保证的是哪个几何体之间不穿透）？
9. 第 7 节的 `tol` 坑：收敛判据为什么要和 $gh$ 比？如果场景里没有重力，只有驱动，同样的坑会以什么样子出现？
10. 第一台 G1 倒下的时刻，$h$ = 5 ms 时和 MuJoCo 差 0.01 s，$h$ = 10 ms 时差 0.08 s。估计一下 $h$ = 2.5 ms 时差多少，这说明误差是几阶的？
