# 阶段 7：刚体之间的接触

**目标：** 从“顶点对地面”推广到“刚体对刚体”。两个凸多面体之间的最近点对只有两种：一个顶点对一个三角形，或一条边对另一条边。只要每一对都保持正距离，两个网格就不相交。能量、梯度、Hessian、曲率上界、CCD 的形式都不变，换的只是距离怎么算。

对应书上：§3.3 平方距离（第 50 页），§8.1 模板、权重与平方距离（第 169–170 页），§8.2 障碍行（第 171 页），§9.2 弦上的 ACCD（第 197–198 页），§9.7 宽相（第 207–210 页）。全流程见 [00_pipeline.md](00_pipeline.md) 第 4、10 节。

---

## 1. 几何：顶点、边、面

从 MJCF 的 geom 读出（`src/model.cc`）：

| geom | 顶点 | 边 | 三角形 |
|---|---|---|---|
| box | 8 个角（半径 0） | 12 条 | 12 个（每个面两个） |
| capsule | 2 个端点（半径 $r$） | 1 条 | — |
| sphere | 1 个球心（半径 $r$） | 1 条退化的边（两端是同一个顶点） | — |

边和面的半径就是它们顶点的半径。球给一条退化的边，是为了让球和球、球和边都走边–边的最近点，不用单独写点–点、点–边。

世界（0 号刚体）上也可以放 box 等，当作静止的障碍物：它们的顶点参与接触，Jacobian 为 0。世界上的静态几何不和平面比（它贴着地面放时距离是 0，障碍会变成无穷大，书 §8.1）。

哪些刚体对要检测（`Model::collide`）：不同的刚体，而且不是父子（MuJoCo 的默认规则；世界和它的子刚体要检测）。

**最近点（`src/distance.cc`）：**

- 点–三角形：按 7 个 Voronoi 区域分情况（3 个角、3 条边、面内），返回最近点的重心坐标 $\lambda$（Ericson《Real-Time Collision Detection》5.1.5）。
- 线段–线段：先解两条直线的最近点，再把参数夹回 $[0, 1]$ 并重解另一个（Ericson 5.1.9）。线段可以退化成点；平行时取其中一对最近点，距离仍然正确。

测试 `test_distance` 拿它们和密集采样比。

---

## 2. 接触：模板、权重、距离

每个接触是两个最近点之间的向量 $\Delta = \sum_v w_v X_v$（书 §8.1）：

| 类型 | 模板 | 权重 |
|---|---|---|
| 顶点–平面（阶段 4） | $X_v$ | $(1)$，$\Delta = n\,(n\cdot X_v - c)$ |
| 点–三角形 | $(X_p, X_a, X_b, X_c)$ | $(1, -\lambda_0, -\lambda_1, -\lambda_2)$ |
| 边–边 | $(X_{p0}, X_{p1}, X_{q0}, X_{q1})$ | $(1-s,\ s,\ -(1-t),\ -t)$ |

A 侧的权重之和是 $+1$，B 侧是 $-1$。距离

$$
d = \lvert\Delta\rvert - r_A - r_B, \qquad n = \Delta/\lvert\Delta\rvert
$$

$n$ 从 B 侧的最近点指向 A 侧的最近点。活跃的接触是 $d < \hat d$（`find_contacts`）。

### 2.1 最近点的参数当常数

$\lambda$、$s$、$t$ 随 $q$ 变化，但它们是“让距离最小”的参数，在最小点处 $\partial\lvert\Delta\rvert/\partial\lambda = 0$（包络定理，书第 170 页）。所以求距离对顶点的导数时，它们可以当常数：

$$
\frac{\partial d}{\partial X_v} = w_v\, n
$$

在区域的边界上（最近点从面内移到边上）这个导数是连续的，距离函数是 $C^1$ 的。

### 2.2 接触行

和阶段 4 完全一样的三层链式法则，只是模板从 1 个顶点变成最多 4 个：

$$
r = \frac{\partial(d^2)}{\partial\delta} = 2d\;n^\top\sum_v w_v\,J(X_v)
$$

$J(X_v)$ 是顶点所在刚体的 `point_jacobian`。两侧的刚体各占一条链；公共祖先上的自由度两侧都有贡献，相互抵消一部分（“两个刚体一起平移，距离不变”）。世界上的顶点 $J = 0$。

能量、梯度、GN Hessian、曲率上界的形式不变（阶段 4 第 3 节、阶段 6 第 3 节）；曲率上界加在**两侧**刚体的链上。测试 `Row.MatchesFiniteDifferenceForEveryContactType` 在 zoo.xml 的随机姿态上对三种接触做有限差分。

### 2.3 边–边的 mollifier

两条边接近平行时最近点不唯一：转一个很小的角度，最近点对就从一头跳到另一头。距离本身连续，但它的导数有“折角”，Newton 会在折角两侧来回跳。IPC 原文（§4.2）的办法是把边–边的能量乘上一个光滑的系数

$$
c = \lvert e_a\times e_b\rvert^2, \qquad \varepsilon = 10^{-3}\lvert e_a\rvert^2\lvert e_b\rvert^2, \qquad
m(c) = \begin{cases}\big(2 - \tfrac{c}{\varepsilon}\big)\tfrac{c}{\varepsilon}, & c < \varepsilon \\ 1, & c\ge\varepsilon\end{cases}
$$

平行时 $m = 0$：这时点–三角形接触会接管（顶点对对方的面）。$m$ 在 $c = \varepsilon$ 处值和一阶导数都连续。

能量 $\kappa\,m\,b(d^2)$。梯度是乘积的导数：$\kappa\big(m\,b'\,r + b\,m'\,\rho\big)$，其中 $\rho = \partial c/\partial\delta$：记 $u = e_a$、$v = e_b$、$w = u\times v$，$\partial c/\partial u = 2\,v\times w$，$\partial c/\partial v = 2\,w\times u$，再经 Jacobian 投影。Hessian 只保留 $\kappa\,m\,b''\,r^\top r$（mollifier 自己的二阶项 $m'' < 0$，交叉项不定，都丢掉，模型仍然半正定）。

书上的第一版实现没有 mollifier（第 170 页的注）。本仓库加上它，是因为台阶场景里盒子的底边和台阶的顶边正好平行，没有它 Newton 每步都跑满上限。测试 `Row.MollifiedEdgeEdgeGradientMatchesFiniteDifference` 检查带 mollifier 的梯度。

---

## 3. 宽相

对所有点–三角形、边–边组合算距离是 $O(V F + E^2)$。宽相的任务是用便宜的测试排除“绝不可能接近”的对，而且**可以多留，绝不能漏**（书 §9.7）。

这个阶段只做第一层：每个刚体的包围盒（顶点位置按半径放大），按 $\hat d$（接触检测）或矢高（CCD）放大后不重叠的刚体对直接跳过。两个盒子的场景里这已经够了；书上的第二层（体坐标系里的 BVH、fp32 块表）留到场景大了再说。

---

## 4. 刚体之间的 CCD：弦上的 ACCD

阶段 5 的框架不变：切段、两端 FK、弦加矢高。一对接触的矢高是两侧刚体的矢高之和 $\xi = (A_a + A_b)\Delta s^2/8$。区别在推进：平面距离沿弦是线性的，有闭式解；点–三角形和边–边的距离沿弦不是线性的。书 §9.2 用 C-IPC 的加性 CCD（ACCD）：

1. 4 个顶点沿弦匀速直线运动，位移 $\Delta X_k$。先减去平均位移 $\bar m$（整体平移不改变距离）。
2. 相对运动的上界 $\ell = \max_{k\in A}\lvert\Delta X_k - \bar m\rvert + \max_{k\in B}\lvert\Delta X_k - \bar m\rvert$。距离对顶点位置的变化率不超过 1，所以沿弦走 $\Delta t$，距离最多减少 $\ell\,\Delta t$。
3. 反复推进 $t\leftarrow t + c\,(d(t) - \xi)/\ell$，直到 $t\ge 1$（整段安全）或 $d(t) - \xi\le(1-c)\big(d(0) - \xi\big)$（已经很近，停在这里）。每推进一次重新算一次最近点。

大多数对第一次推进就到 $t\ge 1$（书上统计每对平均 1.2 次）。宽相：两侧刚体在段两端的包围盒按 $\xi$ 放大后不重叠的对跳过。

---

## 5. 结果

**堆叠（`contact_demo stack`）：** 转了 45° 的盒子从 0.5 m 落到另一个盒子上。着地时 8 个边–边接触（两组底边交叉）承重；轨迹上的最小距离 0.0076 mm（着地那一步，t = 0.204 s），之后静止在下面的盒子上：$z$ = 0.3009 对 0.1003。每步 Newton 最多 17、平均 2.2。

**台阶（`contact_demo ledge`）：** 质心在台阶边缘外 3 cm 的盒子落下，先压在台阶的边上（盒子的底边和台阶的顶边平行：边–边的 mollifier 为 0，点–三角形接管），绕边翻下去，掉到地面上，再滑出去（没有摩擦）。轨迹上的最小距离 0.26 mm；每步 Newton 最多 24、平均 2.4。

两个场景、`Ccd.FastBoxOnBoxIsStoppedBeforeImpact`（20 m/s 砸下去并带着转动）都是 0 穿透。

---

## 6. 代码对照

| 顺序 | 函数 | 文件 | 笔记 | 测试 |
|---|---|---|---|---|
| 1 | `point_triangle_closest`、`segment_segment_closest` | `src/distance.cc` | 第 1 节 | `test_distance` |
| 2 | `from_mujoco` 的边和面、`Model::collide` | `src/model.cc`、`model.h` | 第 1 节 | `Mesh.*` |
| 3 | `find_contacts`（`for_each_candidate`、`make_pt`、`make_ee`） | `src/contact.cc` | 第 2、3 节 | `Detect.*` |
| 4 | `contact_row`、`mollifier_row`、能量/梯度/Hessian 里的 `moll` | `src/contact.cc` | 第 2.2、2.3 节 | `Row.*` |
| 5 | `curved_ccd` 里的刚体对部分、`accd_pair` | `src/ccd.cc` | 第 4 节 | `Ccd.*`、`Scene.*` |

---

## 7. 思考题

1. 包络定理说最近点参数可以当常数。在点–三角形的最近点从面内移到边上的那一刻，$\partial d/\partial X_v$ 为什么还是连续的？边–边平行时为什么不连续？
2. 一个顶点正好在对方的角上时，它和那个角周围的 6 个三角形距离相同，于是有 6 个接触，能量是单个的 6 倍。这会改变静止间隙吗？会改变“不穿透”吗？IPC 为什么能容忍这种重复？
3. mollifier 的阈值 $\varepsilon = 10^{-3}\lvert e_a\rvert^2\lvert e_b\rvert^2$ 对应多大的夹角？台阶场景里盒子翻倒时，底边和台阶顶边的夹角从 0 变到多少时 mollifier 才回到 1？
4. ACCD 的“距离变化率不超过 $\ell$”用了“距离对顶点位置的变化率不超过 1”。对带半径的边–边（两根胶囊）这还对吗？
5. 两个盒子的场景里 `find_contacts` 每次要算 $8\times24\times2 + 12\times12$ 个距离。60 个布娃娃有 2 万多个顶点，书 §9.7 的两层宽相各砍掉了什么？
6. 堆叠场景着地那一步的最小距离只有 0.0076 mm，比落地盒子（0.7 mm）小得多。看看那一步的 trace，是 $\kappa$、CCD 还是线搜索决定的？
