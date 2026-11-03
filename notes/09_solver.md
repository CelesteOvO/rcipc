# 阶段 9：求解器

**目标：** 把 Newton 的几件“工程”做对：解线性方程（PCG 加树块预条件器）、什么时候停（世界坐标的收敛判据、半隐式提前停止）、$\kappa$ 怎么调。顺带修一个让 Newton 变慢的模型问题（障碍行的权重）。

对应书上：§10.2 骨架（第 218–219 页），§10.3 PCG（第 219–221 页），§10.4 预条件器（第 221–224 页），§10.6 停止规则（第 226–229 页），§10.7 $\kappa$（第 229–230 页）。全流程见 [00_pipeline.md](00_pipeline.md) 第 8、9、11 节。

---

## 1. 树

挂在世界上的每个刚体连同它的子孙是一棵运动学树（`Model::ntree`、`body_tree`、`dof_tree`）。树内部的自由度通过质量矩阵紧密耦合（转动根部会带动整棵树），树和树之间只通过接触耦合。zoo.xml 有两棵（躯干和摆），stack.xml 两棵（两个盒子）。

---

## 2. Newton 系统的行形式

到阶段 8 为止，代码把 $H$ 加成一个稠密矩阵再做 LDLT，$O(n_v^3)$。$H$ 其实是

$$
H = \underbrace{\frac{M(q)}{h^2} + \text{曲率上界的对角}}_{H_k\text{：树块}} + \sum_r D_r\, r_r^\top r_r
$$

每一行 $r_r$ 只在两侧刚体的链上非零。障碍行：$D = \kappa\,m\,w(d)$（第 7 节）；摩擦行：切平面里的 $2\times2$ 刚度 $K$ 特征分解成两个秩 1 项，每个给一行（`friction_rows`）。于是 $Hp = H_kp + \sum_r D_r\,r_r(r_r\cdot p)$：每行一次点积、一次 axpy，不用组装（`solver.h` 的 `NewtonSystem`、`apply`）。测试 `Rows.RowFormEqualsDenseHessian` 核对它和稠密 Hessian 一致。

这正是 MuJoCo 的 $M + J^\top DJ$ 的结构（阶段 4 第 3 节）。

---

## 3. PCG 和树块预条件器

**PCG（书第 220 页）。** $x_0 = 0$，$r_0 = -g$，$z_0 = P^{-1}r_0$，$p_0 = z_0$；

$$
\alpha_k = \frac{r_k^\top z_k}{p_k^\top Hp_k},\quad x_{k+1} = x_k + \alpha_kp_k,\quad r_{k+1} = r_k - \alpha_kHp_k,\quad z_{k+1} = P^{-1}r_{k+1},\quad \beta_k = \frac{r_{k+1}^\top z_{k+1}}{r_k^\top z_k},\quad p_{k+1} = z_{k+1} + \beta_kp_k
$$

**停止准则用 rz：** $\lvert r_k^\top z_k\rvert\le\text{tol}\,\lvert r_0^\top z_0\rvert$（libuipc 的默认）。rr 准则（$\lVert r\rVert^2\le\text{tol}^2\lVert r_0\rVert^2$）在硬接触主导残差时会过早停：一行接触很硬，它就主导残差，PCG 只解掉那一行就停了。$r^\top P^{-1}r$ 里各自由度的权重更均衡。书上的消融：布娃娃堆换成 rr，PCG 次数是 3.2 倍。$p^\top Hp\le0$ 或出现 NaN 时立即停，保留当前的 $x$（breakdown 保护）。

**树块预条件器（书 §10.4.1）。** 每棵树一块：

$$
P^{(t)} = H_k^{(t)} + \sum_r D_r\,[r_r^\top r_r]_{tt}
$$

只保留两个自由度都在树 $t$ 里的项，跨树的项扔掉（块 Jacobi）。每块先做 Jacobi 缩放 $SPS$（$S = \mathrm{diag}(1/\sqrt{P_{ii})}$，让 rad 和 m 的自由度量级一致），加 $10^{-10}$ 的对角岭，LDLT。应用时逐块解。

树内部的耦合全在块里，所以：**单棵树、没有跨树接触时 $P = H$，PCG 一次就收敛**（测试 `Pcg.ExactInOneIterationForASingleTree`）。两个盒子互相接触时需要几次（堆叠场景每次 Newton 平均 2.2 次 PCG；tol $10^{-12}$ 时解和 LDLT 一致到 $10^{-6}$，测试 `Pcg.MatchesDenseSolveWithCrossTreeRows`）。

书上还有两样这里没做：接触岛（被很硬的接触连在一起的几棵树合成一块，手抓物体时 PCG 16.5 → 1.0 次）和粗网格校正（默认关）。

---

## 4. $\kappa$ 的调整

阶段 4 只在第一次接触时初始化 $\kappa$（梯度平衡，夹到范围里）。书 §10.7 在每次线搜索之后再调：

- 接受点的最小间隙 $d_{\min} < 10^{-3}\hat d$ 并且比上一次还小：障碍太软，$\kappa\leftarrow\min(2\kappa, \kappa_{\max})$；
- $d_{\min} > 10^{-2}\hat d$：可以放软，$\kappa\leftarrow\max(\kappa/2, \kappa_{\min})$。

效果：静止的接触被压到 $10^{-2}\hat d$ 的量级（落地的盒子静止间隙从 0.98 mm 变成 0.019 mm；$\kappa$ 从上限一路减半到下限），障碍尽量软、Newton 尽量好算，同时间隙不低于 $10^{-3}\hat d$。书上不用 IPC 原来的规则（只在 $d_{\min}$ 小于包围盒对角线的 $10^{-9}$ 时翻倍）是因为它太晚：被挤压的接触对在几次 CCD 限制的迭代里就塌到 $10^{-10}$。

`StepParams::adapt_kappa` 默认开。阶段 6 的两个顶杆测试关掉它：两次运行的迭代次数不同，$\kappa$ 的历史就不同，极小点也不同，比较就没有意义。

**一个后果（测试 `Iterates.EveryAcceptedIterateStaysSeparated`）：** 间隙只有 10 μm 时，“从步首到步末画一条直线再采样”可能钻到地面以下 7 μm，因为那条直线不是真实轨迹（中间的迭代可能先抬起再落下）。CCD 保证的是**每一次接受的迭代**的轨迹不相交：打开 `check_iterates`，逐次迭代沿真实轨迹采样，台阶和翻滚的盒子都 $> 0$。阶段 5、7 的场景测试保留了直线采样的检查，所以关掉了 $\kappa$ 调整。

---

## 5. 停止规则

**完全收敛（书 §10.6.1）：** 整步（$\alpha = 1$）里顶点的最大位移除以 $h$ 小于 `tol`：

$$
\frac{\max_i\lvert X_i(q\oplus p) - X_i(q)\rvert}{h} < \text{tol}
$$

阶段 3 用的是关节空间的 $\max\lvert p_i\rvert/h$，单位混合了 rad/s 和 m/s；现在在世界坐标里量。整步终点的 FK 直弦 CCD 也要用，顺便算。书上的教训：tol 必须远小于这一步里真正要发生的速度变化（自行车链用 0.05 m/s 时 Newton 第 2 次就停，链轮越来越滞后；盒子被摩擦“粘住”）。默认 $10^{-6}$ m/s。

**半隐式提前停止（书 §10.6.2，Stiff-GIPC / libuipc 的默认）：** 从第 6 次迭代起 $\beta\leftarrow\beta(1-\alpha_{\max})$；$\beta\le10^{-3}$ 并且这次取满了 $\alpha_{\max}$ 时停。意思是“过了 6 次以后，第一次能放心走满一整步，就走完这一步并停”。它不检查这一步是不是已经收敛——它是一种不同的积分器。

| 场景（750 步） | 完全收敛：Newton/步 | 半隐式：Newton/步 | 提前停的步 |
|---|---|---|---|
| 堆叠 | 1.25（最多 11） | 1.36（最多 6） | 22 |
| 台阶 | 2.67（最多 22） | 2.27（最多 7） | 134 |
| 斜面下滑（500 步） | 8.5 | 6.0 | 497 |

斜面是反例：每步 Newton 本来要 8–9 次（第 7 节），半隐式在第 6 次就停，盒子 1 s 后只滑到 0.09 m/s（完全收敛 0.94 m/s，解析解 0.94）。这正是书上的话：半隐式一步和收敛一步相差约一步的运动量；什么时候能用它要看场景。`StepParams::semi_implicit` 默认关。

---

## 6. 结果

| 场景 | 阶段 8 | 阶段 9 |
|---|---|---|
| 堆叠：Newton/步 | 2.23 | 1.25 |
| 台阶：Newton/步 | 2.16 | 2.67（间隙从 0.5 mm 压到 10 μm，接触更硬） |
| 斜面下滑：Newton/步 | 10.7 | 8.5 |
| 落地的盒子：静止间隙 | 0.98 mm | 0.019 mm |

PCG 和 LDLT 跑同一个堆叠场景 300 步，轨迹一致到 $10^{-4}$（测试 `Simulation.PcgAndLdltAgree`）。小场景里 PCG 并不比 LDLT 快（两个盒子 12 个自由度），它的意义在刚体多了以后。

---

## 7. 障碍行的权重：补回法向的那一项

阶段 4 第 3 节丢掉了 $\kappa b'\,\partial^2x/\partial\delta^2$；阶段 6 第 2 节把它拆成 $2\kappa b'\lvert\Delta'\rvert^2$（$\le0$，丢掉安全）和轨迹曲率项（用上界处理）。安全归安全，第一部分丢掉有代价：$d\ll\hat d$ 时 $b''\approx\hat x^2/d^4$，$b'\approx-\hat x^2/d^2$，沿法向

$$
\frac{d^2}{dd^2}b(d^2) = b''(2d)^2 + 2b' \approx \frac{4\hat x^2}{d^2} - \frac{2\hat x^2}{d^2} = \frac{2\hat x^2}{d^2}
$$

而 GN 只有第一项 $4\hat x^2/d^2$：**模型比真实硬一倍**，Newton 在法向上每次只走一半，线性收敛、比例 $1/2$（阶段 6 思考题 1 的现象，顶杆和斜面上的“每步十几次”都是它）。

补回法向的那一部分很便宜：行的外积 $r^\top r = 4d^2\,(nJ)^\top(nJ)$，所以权重从 $b''$ 改成

$$
w(d) = b'' + \frac{b'}{2d^2}
$$

它在 $(0, \hat d)$ 上处处为正（测试 `Weight.*`：逐点核对，并和 $b(d^2)$ 对 $d$ 的有限差分二阶导数一致），所以模型仍然半正定，而法向上变成二次收敛。切向的那部分 $2\kappa b'\lvert\Delta'_t\rvert^2$ 仍然丢掉（$\le0$，安全）；阶段 6 的上界不等式仍然成立（只是把一个负项的一部分补了回来）。

效果：斜面下滑每步 Newton 18 → 8.5，堆叠 1.46 → 1.25，台阶 4.2 → 2.7。这是书 §8.6 “秩 1 曲率行”同一个思路里最便宜的一步。

---

## 8. 代码对照

| 顺序 | 函数 | 文件 | 笔记 | 测试 |
|---|---|---|---|---|
| 1 | `from_mujoco` 的树编号 | `src/model.cc` | 第 1 节 | `Tree.*` |
| 2 | `contact_rows`、`friction_rows`、`NewtonSystem`、`apply`、`dense` | `src/contact.cc`、`src/friction.cc`、`src/solver.cc` | 第 2 节 | `Rows.*` |
| 3 | `build_preconditioner`、`pcg` | `src/solver.cc` | 第 3 节 | `Pcg.*` |
| 4 | `barrier_row_weight` | `src/contact.cc` | 第 7 节 | `Weight.*` |
| 5 | `step` 的第 1、2、6、7 步 | `src/step.cc` | 第 4、5 节 | `Kappa.*`、`Stopping.*`、`Iterates.*`、`Simulation.*` |

演示程序 `contact_demo stack|ledge|slope` 的开关：`ldlt=1`、`semi=1`、`adapt=0`，输出里有每步的 PCG 次数。

---

## 9. 思考题

1. rz 准则比较的是 $r^\top P^{-1}r$。对块 Jacobi 预条件器，它和 rr 准则各自“看到”什么？为什么 rr 在硬接触下会过早停？
2. 两个盒子互相接触时 PCG 要 2–4 次。接触岛会把两棵树合成一块，PCG 变成 1 次，但块更大。什么时候划算？（书上的上限：$\max(32, \text{最大的树})$。）
3. $\kappa$ 减半在 $d_{\min} > 10^{-2}\hat d$ 时触发，所以静止的接触最后都停在 $10^{-2}\hat d$ 左右。这对 CCD 的保守系数 $c = 0.9$ 和矢高余量意味着什么？$\hat d$ 取 1 cm 和 1 mm 时，静止间隙差多少？
4. 半隐式停止在斜面上得到 0.09 m/s 而不是 0.94。看 trace：第 6 次迭代时盒子到了哪里？为什么 CCD 从不截断（$\alpha_{\max} = 1$）却仍然没收敛？书上说它是“另一种积分器”，你同意吗？
5. 第 7 节只补回了法向的 $\lvert\Delta'_n\rvert^2$。切向的部分 $2\kappa b'\lvert\Delta'_t\rvert^2$ 多大？在什么运动里它主导（提示：沿表面滑动）？补回它会不会破坏半正定？
6. 书 §8.6 的秩 1 曲率行把 $\Delta''$ 沿行方向的分量也补回来。写出它的权重，说明它和第 7 节的权重分别对应哪一项。
