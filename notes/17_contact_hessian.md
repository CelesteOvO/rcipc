# 阶段 14：完整的接触 Hessian

**问题**（[compare_rigid_ipc.md](compare_rigid_ipc.md) 第 3.3 节）：Rigid-IPC 的 10 环圆环链静止时每步 1 次 Newton，我们每步 5 次。查下来和 $\kappa$、PCG 容差、惯性曲率都无关：每次迭代把收敛量（整步的顶点位移 / $h$）降一个数量级，典型的线性收敛，收敛判据每放宽一个数量级就少一步。怀疑是接触的 Hessian 不精确：

- 阶段 4 的 Gauss-Newton 行 $\kappa b''\,r^\top r$ 丢掉了 $\kappa b'\,\partial^2(d^2)/\partial\delta^2$；
- 阶段 9 第 7 节补回了它沿法向的那一部分（行的权重 $b'' + b'/(2d^2)$），切向的部分和轨迹曲率仍然丢掉；
- 阶段 6 用一个对角上界盖住轨迹曲率项，它把两侧刚体链上所有转动自由度都加硬，方向不对（第 3.3 节里 `curv=0` 少一步）。

Rigid-IPC 用完整的二阶导数（autodiff）做半正定投影。这一阶段把完整的接触 Hessian 做出来（CPU 和 GPU），看链上能不能也到 1–2 步。

**结论先说**（第 3 节）：精确的 Hessian 把静止链上每次迭代的收缩比从 1/9 改到约 1/30，每步少 1 次 Newton（6.5 → 5.5），齿轮和锁链上不变，布娃娃堆少 9%；但现在的实现每步慢 4–6 倍（组装内核和 12 倍的行数）。剩下的线性收敛不是 Hessian 缺项造成的。**默认不开**（`StepParams::full_contact_hessian = false`，`scene_demo fullh=1` 打开），代码和测试留着。

| 文件 | 内容 |
|---|---|
| `include/rcipc/contact.h`、`src/contact.cc` | `squared_distance_hessian`（平方距离对模板顶点的精确二阶导数）、`contact_wrench_hessian`（一个接触的 12 x 12 旋量块）、`contact_rows_full`（投影成行）、`contact_chain_curvature`（链上的剩余项）、`contact_twist_jacobian`、`contact_hessian_exact`（测试用） |
| `include/rcipc/step.h`、`src/step.cc` | `StepParams::full_contact_hessian`：行用 `contact_rows_full`，剩余项和惯性的曲率项一起经正定检查进树块，阶段 6 的上界不加 |
| `src/gpu/contact_full.cu` | GPU 版：一个线程一个接触，12 x 12 的循环 Jacobi，每个接触 12 行 |
| `src/gpu/kin.cu` | `k_inertia_curvature` 多收一个每刚体的 (G, F, P)（`CB`），把接触的链上剩余项和惯性的 K 一起算、一起检查 |
| `src/gpu/stepper.cc`、`iter.cu`、`device_model.h` | 行数 12 nc + 2 nf（`DIter::rpc`），摩擦行跟在后面；`GraphKey::fullh` |
| `toys/scenes/main.cc` | `scene_demo fullh=1`；`debug=3` 每一步都打印每次迭代 |
| `tests/test_contact_hessian.cc`、`tests/test_gpu.cc` | 第 5 节 |
| `compare/rigid_ipc/chain_2.xml` | 两环的链（一个固定、一个挂着），CPU 上几秒就能看静止时的迭代 |

---

## 1. 精确的二阶导数是什么

一个接触的能量是 $E = \kappa\, m(c)\, b(x)$，$x = d^2$，$d = \ell - R$，$\ell = \lvert\Delta\rvert$，$\Delta = \sum_v w_v X_v$（两个最近点之差，$X_v$ 是模板顶点的世界坐标，$R$ 是两侧半径之和，$m$ 是边–边的 mollifier，其余接触 $m = 1$）。顶点通过正运动学依赖关节增量 $\delta$。

对 $\delta$ 的二阶导数按链式法则分两层：

$$
\frac{\partial^2 E}{\partial\delta^2} = J_X^\top\, \frac{\partial^2 E}{\partial X^2}\, J_X \;+\; \sum_v \frac{\partial E}{\partial X_v}\cdot\frac{\partial^2 X_v}{\partial\delta^2}
$$

第一项是顶点空间的 Hessian 经顶点 Jacobian（阶段 2 的 `point_jacobian`）投影到关节上，第二项是“顶点上的力 × 顶点轨迹的曲率”，和阶段 13 惯性项的 $K$ 同一个形状。

### 1.1 顶点空间：平方距离的 Hessian

$y = \lvert\Delta\rvert^2$ 对顶点的二阶导数。固定权重时 $\partial^2 y/\partial X_v\partial X_u = 2 w_v w_u I$。但最近点的参数（点–三角形的重心坐标、边–边的 $s$、$t$）随顶点动。写成 $y(X) = \min_\lambda F(X, \lambda)$，$F = \lvert\sum_v w_v(\lambda) X_v\rvert^2$，$w$ 对 $\lambda$ 线性，包络定理给

$$
\nabla y = F_X,\qquad \nabla^2 y = F_{XX} - F_{X\lambda} F_{\lambda\lambda}^{-1} F_{\lambda X},
$$

其中 $F_{\lambda_k} = 2\Delta\cdot E_k$（$E_k = \sum_v \tfrac{\partial w_v}{\partial\lambda_k} X_v$ 是一条边），$F_{\lambda_k\lambda_l} = 2E_k\cdot E_l$，$F_{X_v\lambda_k} = 2 w_v E_k + 2\tfrac{\partial w_v}{\partial\lambda_k}\Delta$。减掉的那一项半正定，所以精确的 Hessian 比固定权重的软。

哪些参数在动，由子类型决定（和距离函数的分段结构一致）：点–三角形里重心坐标大于 0 的有三个（面内）时两个参数、两个（边上）时一个、一个（顶点）时没有；边–边里 $s$、$t$ 各自在 $(0, 1)$ 内部时才动。被夹在边界上的参数当常数。$F_{\lambda\lambda}$ 退化（两条边平行、球的边长为 0）时用伪逆。地面是 $y = (n\cdot X - o)^2$，$\nabla^2 y = 2nn^\top$。

一个检验：面内的点–三角形，顶点沿三角形平面平移时距离不变，所以切向的二阶导数应该是 0；固定权重给 2，包络项正好减掉 2（测试 `Distance.FaceContactHasNoTangentialCurvature`）。这就是阶段 6 第 2 节说的“切向的 $2\kappa b'\lvert\Delta'\rvert^2$”：它和最近点随顶点变化的项在平面上**恰好抵消**，所以当年丢掉它（$\le 0$，安全）并没有丢掉多少——真正的切向刚度来自曲面的曲率，网格上集中在边和顶点的接触里。

### 1.2 顶点空间：障碍和 mollifier

$x = (\sqrt y - R)^2$：$dx/dy = d/\ell =: \alpha$，$d^2x/dy^2 = R/(2\ell^3)$。所以

$$
\nabla b = b'\alpha\,\nabla y,\qquad
\nabla^2 b = \Big(b''\alpha^2 + \frac{b' R}{2\ell^3}\Big)\nabla y\nabla y^\top + b'\alpha\,\nabla^2 y .
$$

沿法向 $\nabla^2 y$ 给 $2nn^\top$，合起来法向的系数正是阶段 9 的行权重 $b'' + b'/(2\ell^2)$（$R = 0$ 时 $\ell = d$）——核对一致。

边–边乘 mollifier $m(c)$，$c = \lvert u\times v\rvert^2 = \lvert u\rvert^2\lvert v\rvert^2 - (u\cdot v)^2$，$u = X_1 - X_0$，$v = X_3 - X_2$：

$$
\nabla^2(m b) = m\nabla^2 b + b\,(m''\nabla c\nabla c^\top + m'\nabla^2 c) + m'(\nabla c\nabla b^\top + \nabla b\nabla c^\top),
$$

$\partial^2 c/\partial u^2 = 2\lvert v\rvert^2 I - 2vv^\top$，$\partial^2 c/\partial u\partial v = 4uv^\top - 2vu^\top - 2(u\cdot v) I$，$\partial^2 c/\partial v^2$ 对称；$c < \varepsilon$ 时 $m'' = -2/\varepsilon^2$。阶段 7 只保留了 $m\,b''\,r^\top r$，这里全部都在。

### 1.3 几何项

$\partial^2 X_v/\partial\delta^2$ 就是阶段 13 第 2 节的正运动学二阶导数：同一个关节的两个转动 $\tfrac12(a_i\times(a_j\times\rho) + a_j\times(a_i\times\rho))$，上游转动对下游转动 $a_i\times(a_j\times(X - o_j))$，上游转动对下游滑动 $a_i\times a_j$。乘上顶点上的力 $f_v = \partial E/\partial X_v$ 求和，$f\cdot(a\times(b\times\rho)) = a^\top(\rho f^\top) b - (a\cdot b)\operatorname{tr}(\rho f^\top)$。

这一项就是阶段 6 用上界盖住的轨迹曲率。它的大小是力 × 力臂（$\sim 700\ \text{N} \times 0.3\ \text{m}$），比法向刚度（$\sim \kappa\hat x^2/d^2$ 乘力臂的平方）小 $d/\ell$ 倍，在法向上可以忽略；但在接触的切向（滚动、摆动）法向刚度是 0，它和重力的几何刚度（阶段 13 的 $K$）就是切向刚度的全部。阶段 6 的上界把它按绝对值加到每个转动自由度的对角上，切向就估硬了。

---

## 2. 不定号：在旋量空间里投影

精确的 Hessian 不定号（$b' < 0$，几何项的符号看力和力臂的方向），不能直接用。IPC 的做法是每个接触的 Hessian 单独做半正定投影（特征分解，负的特征值置 0），Rigid-IPC 也是（`project_to_psd`）。

在哪个空间里投影，决定了结果能不能放进现有的求解器。Newton 系统是“树块 + 秩 1 的行”（阶段 9），GPU 上每行按两侧刚体的力旋量存（[11_gpu.md](11_gpu.md) 第 8 节）：行在自由度 $d$ 上的值是 $a_d\cdot(\tau - o_d\times f)$（转动）或 $a_d\cdot f$（滑动）。所以把一个接触的 Hessian 写在**两侧刚体的旋量空间**里：每侧 6 维 $\xi = (v, \omega)$，以刚体原点 $x_b$ 为参考点，顶点速度 $X' = v + \omega\times(X - x_b)$。

$$
H_{12} = J_\xi^\top\,\frac{\partial^2 E}{\partial X^2}\,J_\xi + \begin{pmatrix} 0 & \\ & G_A \end{pmatrix}\oplus\begin{pmatrix} 0 & \\ & G_B\end{pmatrix},
\qquad J_\xi = [\,I,\ -[X - x_b]_\times\,],\quad G = \tfrac12(Q + Q^\top) - \operatorname{tr}(Q)\,I,\quad Q = \sum_{v\in b}(X_v - x_b) f_v^\top .
$$

$G$ 是顶点随 $\omega$ 走弧线的几何项（$\exp([\omega])\rho$ 的二阶项是 $\tfrac12\omega\times(\omega\times\rho)$）。$H_{12}$ 做特征分解，特征值 $\lambda_k > 0$ 的每一对 $(\lambda_k, u_k)$ 变成一行：$u_k$ 的 12 个分量就是两侧的 $(f, \tau)$（换成关于世界原点的力矩 $\tau + x_b\times f$ 以后就是 GPU 行的存法），$D = \lambda_k$。一个接触最多 12 行（`contact_rows_full`）；实际上几乎每个接触都是 12 行（$H_{12}$ 一般满秩）。

**关节坐标对旋量不是线性的那一截。** 旋量 $\xi_b(\delta)$ 对 $\delta$ 的二阶导数乘上 $\partial E/\partial\xi$ 还有一项。自由刚体上它恰好是 0：`integrate` 对自由关节就是“原点平移 + 绕原点转 $\exp$”，正是这个参数化。铰链锚点不在刚体原点、或链上有上游关节时不是 0。写出来（`contact_chain_curvature`，$F = \sum f_v$、$P = Q$）：

- 同一个关节的两个转动，锚点 $o$：$\operatorname{sym}\,\mathrm{rr}(a_i, (x_b - o)F^\top, a_j)$；
- 上游转动 $a_u$、下游转动 $a_d$（锚点 $o_d$）：$\mathrm{rr}(a_u, P + (x_b - o_d)F^\top, a_d) - \operatorname{sym}\,\mathrm{rr}(a_u, P, a_d)$；
- 上游转动、下游滑动：$(a_u\times a_d)\cdot F$。

（$\mathrm{rr}(a, Q, b) = a^\top Q b - (a\cdot b)\operatorname{tr}Q$。）它只在同一棵树的自由度之间非零，所以进树块，和阶段 13 的 $K$ 加在一起做同一个正定检查（`add_inertia_curvature`）。

测试 `Exact.MatchesFiniteDifferenceOnTheZoo` 核对 $\sum_c J_c^\top H_{12,c} J_c$ 加这一项等于能量对 $\delta$ 的二阶差分（四种关节、锚点不在原点、焊死的刚体、地面），`Exact.ChainTermVanishesForFreeBodies` 核对自由刚体上剩余项是 0。

投影在旋量坐标里做，不是坐标不变的（$v$ 的单位是 m，$\omega$ 是 rad）；Rigid-IPC 在它的（位置、转动向量）坐标里投影，一样。

---

## 3. 结果

都是 GPU，`compare/rigid_ipc/chain_10.xml`（10 环圆环链，$h$ = 10 ms，$\hat d$ = 1 mm，μ = 0，tol = 1e-6 m/s）和 `gears_single.xml`，T = 10 s。Newton 次数含只检查不移动的那一次。

| | Newton / 步 | 最大 | PCG / 步 | ms / 步 |
|---|---|---|---|---|
| 链，Gauss-Newton（默认） | 6.53 | 50（1 步撞上限） | 82.7 | 5.1 |
| 链，完整 Hessian | 5.50 | 50（1 步） | 76.0 | 21.9 |
| 齿轮，Gauss-Newton | 3.22 | 13 | 3.8 | 2.2 |
| 齿轮，完整 Hessian | 3.20 | 13 | 3.7 | 3.0 |

### 3.1 静止时每次迭代

`debug=3` 打印每步的迭代。t = 1 s 的链（环还在以约 3 cm/s 的速度缓缓摆动——μ = 0，没有东西阻尼它）：

| 迭代 | GN：$\lvert dx\rvert/h$（m/s） | 完整 Hessian |
|---|---|---|
| 1 | 3.0e-2 | 3.3e-2 |
| 2 | 3.3e-3 | 9.8e-4 |
| 3 | 3.4e-4 | 2.3e-5 |
| 4 | 3.6e-5 | 1.5e-6 |
| 5 | 3.7e-6 | （收敛） |
| 6 | （收敛） | |

CCD 和线搜索每次都取满步长。GN 每次迭代收缩 **1/9**，稳定得像一个常数——就是“模型比真实硬一个固定比例”的线性收敛；完整 Hessian 收缩 1/33、1/43、1/16，快了 3 倍多，但仍不是二次收敛。少的那一步就是表里 6.53 → 5.50 的来源。

容差扫描（`tol=`，Newton 次数去掉检查的那一次）：

| tol（m/s） | 1e-3 | 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|
| GN | 1.43 | 3.17 | 4.30 | 5.53 |
| 完整 Hessian | 1.16 | 2.21 | 3.32 | 4.50 |

### 3.2 剩下的线性收敛不是哪些东西

同一张迭代表，换这些设置收缩比都不变：

- `pcg_tol=1e-8`（原 1e-4）：3.3e-2 → 8.4e-4 → 2.2e-5 → 收敛，比例一样，不是线性解不准；
- `icurv=2`（惯性曲率连转动那一部分也加）：逐位不变，不是惯性 Hessian；
- 两环的链（`chain_2.xml`，CPU）：GN 和完整 Hessian 静止时都是 3 次（2 次移动 + 检查），第二次迭代的残差 1.1e-5 对 1.6e-6 m/s，完整 Hessian 更准但没跨过 1e-6；
- CPU 上 10 环 T = 1.5 s（LDLT 没跑完；PCG）：GN 8.68、完整 8.27，和 GPU 一致。

还没排除的两个：半正定投影本身（负曲率的方向被置 0，模型在那些方向上比真实硬，这是投影 Newton 固有的线性收敛），和环滚动时最近点的子类型（面内、边上、顶点）在迭代之间切换——Hessian 在子类型边界上不连续，每次迭代落在不同的分段上。Rigid-IPC 的链在 0.93 mm 的间隙上，障碍平缓，这两样都不明显。

### 3.3 别的场景

| 场景 | GN Newton / 步 | 完整 Hessian | ms / 步 |
|---|---|---|---|
| `ragdoll_pile.xml`，T = 1 s，μ = 0.2 | 12.39（12 步撞上限） | 11.24（4 步） | 9.8 → 46.3 |
| `ring_chain.xml`，T = 1 s，μ = 0.3，tol = 2e-3，半隐式停止 | 5.81 | 5.88 | 4.0 → 24.1 |

布娃娃堆少 9%，撞上限的步少了三分之二；锁链不变。

### 3.4 时间都花在哪

链上不用 CUDA graph 分段计时（`graph=0 timing=1`，T = 2 s）：

| ms / 步 | 组装 | 解 | 收敛判断 | CCD | 线搜索 | 合计 |
|---|---|---|---|---|---|---|
| GN | 0.58 | 1.52 | 0.69 | 2.72 | 1.93 | 8.0 |
| 完整 Hessian | 23.0 | 2.86 | 0.76 | 2.68 | 1.87 | 31.8 |

组装内核 `k_contact_full` 一个线程一个接触，本地数组十来个 12 x 12，三个 12³ 的矩阵乘和一个循环 Jacobi（几十次 sweep），390 个接触只占 7 个块：延迟和本地内存把它拖到每次 3 ms。行数 392 → 4668（每个接触 12 行），PCG 的矢量乘和预条件器的行那部分跟着慢一倍。

要让它划得来，要做两件事：组装改成一个 warp 一个接触（矩阵乘和 Jacobi 按 lane 分），行改成每个接触一个 12 x 12 的块（矢量乘 144 次乘加加 12 次原子加，预条件器直接拿两个 6 x 6 块加到 `Wb`）。链上最多省 15% 的 Newton，这两件现在不值得做。

---

## 4. GPU

`contact_full.cu`：一个线程一个接触，算顶点空间的 $f$、$H_X$，投到 12 维旋量，`sym_eig`（循环 Jacobi，相对阈值 1e-30）分解，正的特征值写到行 $12i + k$（用不到的行两侧刚体是 0、$D$ = 0，矢量乘跳过）；每侧的 $(G = P + x_b F^\top, F, P)$ 原子加到 `CB`（nbody x 21）。`k_inertia_curvature` 的每刚体量从 12 个扩成 21 个（多了 $P$），$G$、$F$ 直接加惯性的那一份（$x_b F^\top$ 和惯性的 $c f^\top$ 同一个形式，所以子树求和、锚点那一项的减法都不用改），转动对转动的元素再减去旋量块里已经有的 $\operatorname{sym}\,\mathrm{rr}(a_i, \bar P, a_j)$。`inertia_curvature = false` 时 K 里只有接触的那一份。

行数：`DIter::rpc`（每接触几行）由主机写，`k_it_begin` 算 `nrow = rpc nc + 2 nf` 和摩擦行的起点 `row0f`；逐个发射的路径同样。`GraphKey` 多一个 `fullh`，切换时重录图。

`Terms.FullContactHessian`（`tests/test_gpu.cc`）：两个盒子、台阶、布娃娃（含 `inertia_curvature = false`），GPU 和 CPU 的能量、梯度、Hessian 一致到 1e-9，并且和 Gauss-Newton 的确实不同。

---

## 5. 测试

`tests/test_contact_hessian.cc`：

- `Distance.HessianMatchesFiniteDifferenceForEverySubtype`：zoo 的随机姿态（所有候选对）加两个盒子交叉着叠的姿态，平方距离对顶点的二阶差分，点–三角形的三种子类型和边–边的内部、端点都出现。球的边退化成一个点（两个端点是同一个顶点），差分把它们分开扰动会乱跳，跳过。
- `Distance.FaceContactHasNoTangentialCurvature`：面内接触切向二阶导数为 0，整体平移是零空间；整体转动不是（$H r = [e\times\nabla_v y]_v$），按这个核对。
- `Exact.*`：zoo、两个自由盒子、台阶上的 mollifier，精确 Hessian 和能量的二阶差分一致；Gauss-Newton 的差得多。
- `Rows.AreTheProjectedWrenchBlocks`：行加起来就是 $J^\top[H_{12}]_+J$；`Rows.TwistJacobianReproducesPointJacobians`。
- `Step.StackRestsWithTheFullHessian`：开着完整 Hessian 的时间步不穿透、静止。

---

## 6. 思考题

1. 面内的点–三角形接触，固定权重的切向二阶导数 2 和包络项正好抵消。边–边的内部接触呢？两条边都沿自己平移时距离不变，但一条边绕另一条转时距离变——写出 $F_{X\lambda}F_{\lambda\lambda}^{-1}F_{\lambda X}$ 在这个方向上的值。
2. 投影 Newton 的线性收敛比例是“被置 0 的负曲率”和“真实曲率”之比。在静止的链上，哪些方向有负曲率？（提示：几何项 $G$ 的特征值是 $-\lvert f\rvert\lvert\rho\rvert$ 的那个方向。）
3. 把 12 行换成一个 12 x 12 的块，预条件器的 `Wb`（每个刚体一个 6 x 6）正好能装下两侧的对角块；两侧在同一棵树里时的交叉块怎么办？
