# 阶段 4：地面接触

**目标：** 第一次在关节坐标里处理接触。障碍物只有地面（平面），接触的一方是刚体上的顶点。
这个阶段把阶段 1 的障碍、CCD 和阶段 3 的时间步接起来，并且故意保留一个漏洞：CCD 只看直弦，阶段 5 再修。

对应书上：§8.1 接触行（第 169–171 页），§8.2 障碍能量（第 172 页），§8.3 $\kappa$ 的初值（第 174 页），
§9.5 地面的解析处理（第 204 页）。全流程的位置见 [00_pipeline.md](00_pipeline.md) 第 4、10 节。

---

## 1. 几何：顶点和平面

接触需要两样东西，都从 MJCF 的 geom 读出来（`Model::vertices`、`Model::planes`，`src/model.cc`）：

| geom | 变成什么 |
|---|---|
| 刚体上的 box | 8 个角，半径 0 |
| sphere | 球心，半径 $r$ |
| capsule | 两个端点（球心），半径 $r$ |
| 世界上的 plane | 障碍：法向 $n$（geom 的 $z$ 轴）和偏移 $c = n\cdot\text{pos}$，自由的一侧是 $n\cdot x > c$ |

顶点带半径，是为了让球和胶囊的圆弧面也能用“点”来处理：

$$
d = n\cdot x_v - c - r_v
$$

$x_v$ 是顶点（球心）的世界坐标，由阶段 2 的 `world_point` 算出。$d$ 是球面到平面的真实距离，带符号，CCD 保证它 $> 0$。

**为什么只有顶点：** 刚体是凸的（盒子、球、胶囊），它离一个平面最近的点一定在角上、球面上或胶囊的端部，这些都被上面的顶点覆盖了。阶段 7 的刚体间接触才需要边和面。

---

## 2. 接触行

阶段 1 的障碍写在距离 $d$ 上；从这里起写在**平方距离** $x = d^2$ 上，$\hat x = \hat d^2$。原因是到阶段 7，点–三角形、边–边的平方距离是顶点坐标的多项式，处处可导，不用开方（书 §3.3）。障碍函数本身不变：`barrier(d * d, dhat * dhat)`。

Newton 需要 $x$ 对关节增量 $\delta$ 的导数。链式法则分三层：

$$
r = \frac{\partial x}{\partial\delta} = \underbrace{2d}_{\partial x/\partial d}\;\underbrace{n^\top}_{\partial d/\partial x_v}\;\underbrace{J(x_v)}_{\partial x_v/\partial\delta}
$$

- $\partial x/\partial d = 2d$：平方的导数；
- $\partial d/\partial x_v = n$：平面距离对点的导数是法向（半径是常数）；
- $J(x_v)$：阶段 2 的 `point_jacobian`，转动自由度那一列是 $a_d\times(x_v - o_d)$，平移自由度是 $a_d$，不在链上的是 0。

$r$ 是 $1\times n_v$ 的行向量，叫**接触行**（`contact_row`）。含义：关节沿 $\delta$ 动时，平方距离的变化率是 $r\delta$。

按列展开就是书第 171 页的公式：转动自由度 $d$ 的分量是 $2d\, n\cdot\big(a_d\times(x_v - o_d)\big)$。

---

## 3. 能量、梯度、Hessian

$$
E_{\text{contact}}(q) = \kappa\sum_{\text{活跃的 } (v,\,\text{平面})} b\big(d_v^2\big), \qquad \text{活跃} \iff d_v < \hat d
$$

每次 FK 之后都要重新找活跃的接触（`ground_contacts`）：换了位置，活跃的接触就可能变了。$d \ge \hat d$ 时 $b = 0$，所以“不活跃”和“贡献为 0”是一回事。

**梯度：**

$$
\nabla_\delta\big(\kappa\, b(x)\big) = \kappa\, b'(x)\, r^\top
$$

物理含义：$-\kappa b'(x)\cdot 2d$ 是法向力的大小（$b' < 0$，所以是斥力），$n^\top J$ 把它投影到每个关节上。这和阶段 3 的“力经 Jacobian 投影到关节”是同一个套路。

**Hessian：** 再求一次导，乘积法则给出两项

$$
\nabla^2_\delta\big(\kappa\, b(x)\big) = \underbrace{\kappa\, b''(x)\, r^\top r}_{\text{保留}} + \underbrace{\kappa\, b'(x)\,\frac{\partial^2 x}{\partial\delta^2}}_{\text{丢掉}}
$$

- 第一项是“行的外积”，半正定；代码里取 $\kappa\max(b'', 0)$ 保险（在 $(0,\hat x)$ 上 $b''$ 本来就是正的）。
- 第二项含 $b' < 0$ 和顶点轨迹的曲率，可能不定。Gauss-Newton 丢掉它，这正是阶段 3 对惯性项做的事。

对一个只做平移的盒子，$\partial^2 x/\partial\delta^2 = 2\,|n\cdot a|^2 \ge 0$，第二项 $\le 0$：丢掉它只会让模型更硬，是安全的。
对转动的关节，顶点走弧线，第二项可以是正的，丢掉它会让 Newton 低估能量的增长，这是阶段 6 要处理的。

**整个 Newton 系统：**

$$
H = \frac{M(q)}{h^2} + \sum\kappa\, b''\, r^\top r, \qquad g = g_{\text{inert}} + \sum\kappa\, b'\, r^\top
$$

和 MuJoCo 的 Newton 求解器的 $M + J^\top DJ$ 是同一种形状：$J^\top DJ = \sum_i D_i J_i^\top J_i$，一行一个接触。
不同的是 $D$ 的来源：MuJoCo 是软约束的刚度，这里是障碍的二阶导数，离得越近越硬。

---

## 4. $\kappa$ 的初值

阶段 1 说过：$\kappa$ 决定停在多远，不决定会不会穿透。但它太小时物体会压得很深、障碍被迫变得极陡，太大时 Hessian 条件数很差，Newton 都会变慢。书 §8.3 的做法是在第一次出现接触时让它和惯性“势均力敌”：

$$
\min_\kappa\,\lvert g_e + \kappa\, g_b\rvert^2 \quad\Longrightarrow\quad \kappa = -\frac{g_b\cdot g_e}{\lvert g_b\rvert^2}
$$

$g_e$ 是惯性梯度（含重力），$g_b$ 是 $\kappa = 1$ 时的接触梯度。对 $\kappa$ 求导令其为 0 就得到这个式子。

**数值例子（测试 `Kappa.InitialValueBalancesGravityOnTheBox`）：** 盒子静止悬在 $d = \hat d/2$ 处。惯性目标是“再往下掉 $gh^2$”，所以 $g_e = (0, 0, -mg, 0, 0, 0)$；4 个角各有一行 $r = 2d\,e_z^\top$。解出来的 $\kappa$ 正好让 4 个障碍力之和等于 $mg$。

**范围：** 梯度平衡可能给出不合理的值（比如惯性正在把物体拉离障碍时 $g_b\cdot g_e > 0$，$\kappa$ 为负）。所以先定一个参考量级，再把结果夹在参考值的 $[10^{-2}, 10^2]$ 倍之间（上下限相差 $10^4$，同书第 174 页）：

$$
\kappa_{\text{ref}}\;\lvert b'(\hat d^2/4)\rvert\;\hat d = \bar m\,\lvert g\rvert
$$

即“一个障碍在 $\hat d/2$ 处的斥力等于一个平均刚体的重量”。$b'(\hat x/4) = -4.33\,\hat x$，所以 $\kappa_{\text{ref}} \approx \bar m g/(4.33\,\hat d^3)$：$\bar m = 1$ kg、$\hat d = 1$ mm 时约 $2.3\times10^9$ J/m⁴。

> 书上和 ipctk 用的下限是 $10^{11}\bar m / b''(d_0)$，$d_0 = 10^{-8}\times$ 包围盒对角线。它在平方距离的写法下给出的量级和本仓库的单位对不上，所以这里改用上面这个有物理含义的参考值。它只是夹的范围，真正的初值由梯度平衡决定。

**在哪个点上算：** $g_b$ 在接触第一次出现的那个 Newton 迭代点上算（那之前 $g_b = 0$，没法平衡）；$g_e$ 用**步首** $q_n$ 处的惯性梯度 $-m(v/h + g)$，也就是这一步要“刹住”的动量加重力。不能用当前迭代点的惯性梯度：接触常常在一个已经收敛的迭代点上才第一次出现，那里它正好是 0。

落地的盒子里，接触第一次出现时 $d$ 接近 $\hat d$、$b'$ 很小，平衡出来的 $\kappa$ 约 $10^{13}$，被夹到上限 $2.3\times10^{11}$。`contact_demo drop kappa=2.27e9` 可以试试参考值：着地时压得更深（0.01 mm 对 0.7 mm），静止间隙更小（0.77 mm 对 0.98 mm），都不穿透。

$\kappa$ 存在 `SolverState` 里，跨时间步沿用；按间隙调整它是阶段 9 的内容。

---

## 5. 直弦 CCD

线搜索只在离散的点上检查能量。障碍在穿透处是无穷大，但一步可能直接跨过它落到另一侧，那里能量又是有限的（阶段 1 思考题 6）。所以走之前要先问：沿 $q\oplus s\,p$（$s\in[0,1]$）走，最多走到哪里不穿透？

**直弦版本（这个阶段）：** 假装每个顶点从 $s = 0$ 的位置沿直线走到 $s = 1$ 的位置。平面距离沿直线是线性的：

$$
y(s) = y_a + s\,(y_b - y_a)
$$

按阶段 1 的保守推进规则，每个顶点这一次最多闭合当前距离的 $c = 0.9$ 倍：$y(s)\ge(1-c)\,y_a$，所以

$$
s \le \frac{c\,y_a}{y_a - y_b} \quad (y_b < y_a)
$$

远离平面的顶点（$y_b \ge y_a$）不限制。所有 (顶点, 平面) 对取最小，再和 1 取较小，就是 $\alpha_{\max}$（`chord_ccd_ground`）。
它正是阶段 1 的 `ccd_max_step`，一维里是精确的。

**它会漏什么：** 关节转动时顶点走的是弧线，弧和弦之间有一段“缝”（矢高，书 §9.1 图 44）：

$$
\text{矢高} = R\Big(1 - \cos\frac{\theta}{2}\Big)\approx\frac{R\theta^2}{8}
$$

障碍如果正好在弧和弦之间，直弦 CCD 看不见它。

**反例（`models/flail.xml`，测试 `Flail.ChordCcdMissesTheArcThroughTheFloor`）：** 一根 0.5 m 的杆绕离地 0.5075 m 的铰链转，杆端（球心加半径）扫出的圆最低点在地面以下 2.5 mm，只有杆指向正下方 $\pm 0.1$ rad 的一小段弧在地面以下。以 40 rad/s、$h = 0.01$ 一步转约 0.34 rad，可以从地面上方 3 mm 直接跨到另一侧上方 2 mm：两端都在 $\hat d$ 之外，Newton 一次收敛，CCD 和线搜索都放行；真实弧线在中间低到地面以下 2.5 mm。

$R\theta^2/8 = 0.51\times0.34^2/8 \approx 7$ mm，远大于两端的 2–3 mm 间隙，所以漏掉是必然的。阶段 5 用这个矢高做余量：要求弦上的距离始终大于矢高。

**独立检查：** `trajectory_min_distance` 沿真实轨迹（每个 $s$ 都做 FK）采样最小距离。它不参与求解，只用来抓 CCD 漏掉的穿透。每个测试和演示都用它检查。

---

## 6. 时间步

在阶段 3 的循环里加三处（`src/step.cc`，对照 `toys/ipc1d/ipc1d.cc`）：

| | 阶段 3 | 阶段 4 |
|---|---|---|
| 第 0 步 | FK | FK，找活跃接触；第一次有接触时初始化 $\kappa$ |
| 第 1 步 | $g = g_{\text{inert}}$，$H = M/h^2$ | 加上 $\kappa b' r^\top$ 和 $\kappa b'' r^\top r$ |
| 第 3 步 | $\alpha_{\max} = 1$ | $\alpha_{\max}$ = `chord_ccd_ground(FK(q), FK(q ⊕ p))` |
| 第 4 步 | 线搜索从 1 开始 | 从 $\alpha_{\max}$ 开始；每个试探点重新做 FK、找接触 |

能量是惯性项加接触项。线搜索保证能量不升，CCD 保证路径上（按弦）不穿透，两者缺一不可。

---

## 7. 结果

**落地的盒子（`drop.xml`，从 0.5 m 静止落下 1.5 s，$h = 2$ ms，$\hat d = 1$ mm）：**

| | $\kappa$ 由梯度平衡（夹到上限 $2.27\times10^{11}$） | `kappa=2.27e9`（参考值） |
|---|---|---|
| 轨迹上的最小间隙（着地那一步，t = 0.32 s） | 0.71 mm | 0.014 mm |
| 静止间隙 | 0.98 mm | 0.77 mm |
| 静止时 4 个角的障碍力之和 | 9.8105 N（重力 9.81 N，差的是收敛容差） | 同左 |
| 每步 Newton 次数 | 最多 10，平均 1.28 | 最多 20，平均 1.54 |

两种 $\kappa$ 都不穿透，差别只在停在多远：阶段 1 的结论在三维里仍然成立。

**甩杆（`flail.xml`，`contact_demo flail`）：** 一步转 0.346 rad，两端离地 +3.1 mm 和 +7.1 mm，弧线最低点 −2.5 mm；每次迭代 CCD 的 $\alpha_{\max}$ 都是 1。这是阶段 5 要修的。
`flail h=0.005` 是另一种失败：一步转 0.25 rad，Newton 想走到的终点在地面以下，直弦 CCD 按弦截到 0.556，但弧线上这个点比弦上低，结束位置刚好嵌进地面（−0.0002 mm）。

---

## 8. 代码对照

| 顺序 | 函数 | 文件 | 笔记 | 测试 |
|---|---|---|---|---|
| 1 | `from_mujoco` 里的 geom 部分 | `src/model.cc` | 第 1 节 | `test_contact --gtest_filter=Geometry.*` |
| 2 | `ground_distance`、`ground_contacts` | `src/contact.cc` | 第 1 节 | 同上 |
| 3 | `contact_row` | `src/contact.cc` | 第 2 节 | `Row.*` |
| 4 | `contact_energy`、`contact_gradient`、`contact_hessian` | `src/contact.cc` | 第 3 节 | `Energy.*` |
| 5 | `kappa_range`、`initial_kappa` | `src/contact.cc` | 第 4 节 | `Kappa.*` |
| 6 | `chord_ccd_ground`、`trajectory_min_distance` | `src/ccd.cc` | 第 5 节 | `test_ccd` |
| 7 | `step` 里新加的三处 | `src/step.cc` | 第 6 节 | `Drop.*`、`Flail.*` |

演示程序：

```
build/Debug/contact_demo.exe drop
build/Debug/contact_demo.exe drop h=0.01 dhat=5e-3
build/Debug/contact_demo.exe flail
build/Debug/contact_demo.exe flail w=20
```

---

## 9. 思考题

1. 接触行 $r = 2d\,n^\top J$ 在 $d\to 0$ 时趋于 0，而 $b'(d^2)$ 趋于 $-\infty$。两者的乘积（法向力）是什么量级？为什么用 $d^2$ 做自变量不会让力在 $d\to 0$ 时消失？
2. 落地的盒子静止间隙约 0.4 mm，和阶段 1 的 0.54 mm 不同。哪些因素变了？（提示：$\kappa$ 的单位、接触点的个数、障碍写在 $d^2$ 上。）
3. 如果 $\kappa$ 初始化时盒子正在上升（$g_b\cdot g_e > 0$），梯度平衡给出负值。代码里退到了参考值。有没有更好的选择？
4. `Drop.NeverPenetratesAndRestsInForceBalance` 里着地那一步的最小间隙远小于 $\hat d$。是什么决定了它？把 $c$ 改成 0.5 或 0.99 会怎样？（`contact_demo` 没有这个参数，可以在测试里改。）
5. 甩杆反例里，如果把 $h$ 减半（`flail h=0.005`），一步只转 0.2 rad，矢高降到 1/4。还会漏吗？从矢高公式估计，什么样的 $(\omega, h, R)$ 组合是安全的？
6. 直弦 CCD 对自由关节的平移是精确的，对转动不是。阶段 5 要加的余量 $\xi = A\,\Delta s^2/8$ 需要每个刚体的加速度上界 $A$。对单个铰链，$A$ 是什么？对两节的链条呢？
