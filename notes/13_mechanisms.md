# 机构和机器人场景（阶段 12 之后）

**目标：** 用阶段 12 的驱动、阻尼，跑几个前面没有的场景：被驱动的齿轮组、夹爪抓取，以及真实的机器人模型（Unitree Go2）堆在一起。为了 Go2，读模型时加了圆柱（cylinder）geom，场景程序从第一个关键帧的姿态开始。

| 场景 | 文件 | 生成方式 | 测试 |
|---|---|---|---|
| 齿轮组 | `models/gears.xml` | `tools/make_gears.py` | `test_gpu.cc` 的 `Mechanism.GearsTransmit` |
| 夹爪抓方块 | `models/gripper.xml` | 手写 | `test_mechanisms.cc` 的 `Gripper.LiftsCube`，`test_gpu.cc` 的 `Mechanism.GripperMatchesCpu` |
| 圆柱 | `models/cylinders.xml` | 手写 | `Cylinder.Geometry`、`Cylinder.RestsOnGroundAndBox` |
| 24 只 Go2 | `models/local/go2_pile.xml`（不进仓库） | `tools/make_robot_pile.py <go2.xml>` | 只有演示 |

视频用 `tools/render.py` 渲染到 `media/`（不进仓库），命令见第 5 节。

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

**忽略的设置**（读模型时记进 `Model::ignored`，`scene_demo` 开头打印）：关节的 `frictionloss`（干摩擦）、`actuatorfrcrange`。Go2 的这个文件里没有驱动，所以这是一堆“没有电机的”机器人，腿是软的。

**结果**（GPU，`dhat=0.005 eps_v=0.01 tol=0.05 semi=1`，3 s）：
- 每步平均 4.0 ms；冲击时最慢约 12 ms，静止以后约 1.1 ms。
- 平均 3.4 次 Newton（最多 36 次，没有撞上限），每步 10.9 次 PCG。
- 不穿透，最近 0.45 μm；关节都在限位里，最近 0.3 mrad。

旧分支在 RTX 3090 上报的同类场景（24 只四足、432 自由度）是 5.6 ms，但场景不完全一样（台阶、石块、碰撞体、参数），只能作参考。

---

## 5. 复现

```
python tools/make_gears.py models/gears.xml
python tools/make_robot_pile.py <menagerie>/unitree_go2/go2.xml models/local/go2_pile.xml --n 24
build\Release\scene_demo.exe model=gears.xml T=3 mu=0 gpu=1 out=media/gears.csv
build\Release\scene_demo.exe model=gripper.xml T=2.5 mu=0.5 out=media/gripper.csv
build\Release\scene_demo.exe model=<仓库>/models/local/go2_pile.xml T=3 gpu=1 dhat=0.005 eps_v=0.01 tol=0.05 semi=1 out=media/go2_pile.csv
python tools/render.py models/gears.xml media/gears.csv media/gears.mp4 --speed 1 --el 75 --az 90 --range 1.0 --target 0.24,0,0
python tools/render.py models/gripper.xml media/gripper.csv media/gripper.mp4 --speed 0.5 --el 12 --az 70 --range 0.8 --target 0,0,0.17
python tools/render.py models/local/go2_pile.xml media/go2_pile.csv media/go2_pile.mp4 --speed 1 --el 25 --az 35 --dist 0.9
```

---

## 6. 还差的

- **G1 人形**：24 个碰撞网格一共 21.9 万个三角形，直接用太重。要先换成凸包或者减面（旧分支是把每个连杆的碰撞体合成一个凸包）。
- **Allegro 手**：碰撞体是盒子和胶囊，驱动是 position，都支持了；还缺一个抓取的时间表和一个物体。
- **闭环**（自行车链、活塞、履带车）：要等式约束，和“关节精确”的说法冲突（路线图第 3 类）。
- **关节干摩擦**（frictionloss）：可以做成光滑化的 Coulomb 摩擦，和接触摩擦同一个写法。

---

## 7. 思考题

1. 齿轮的转角比值为什么随时间越来越接近 ±1？用侧隙算一下 2 s 时 gear3 应该落后多少度，和表里的比一比。
2. velocity 驱动的主动轮最后是 −99.6°/s。负载阻尼 0.02 N m s/rad 时，算一下稳态的转速差（别忘了阶段 12 第 4 节的惯性阻力）。
3. 夹爪提起方块时，方块在手指里滑了约 0.4 mm。光滑化摩擦在静止时允许多大的滑动速度？把 `eps_v` 减小 10 倍会怎样，代价是什么？
4. 圆柱横着时停在一条棱上。在真实的物理里它会倒吗？我们的模拟为什么不倒？
5. Go2 的腿上没有阻尼也没有驱动，落地以后很快就不动了（每步 1 次 Newton）。能量去哪了？
