# 阶段 3：body 形式的惯性项

**目标：** 第一次在关节坐标里跑真正的时间步。能量里只有惯性和重力，还没有接触，所以也还没有 CCD。
检验的场景是一个双摆和一个在空中自由旋转的盒子。

这个阶段把前两个阶段接起来：

- **阶段 1 给了算法：** 增量势、Newton、线搜索、`step()` 的循环；
- **阶段 2 给了翻译器：** FK、⊕/⊖、Jacobian；
- **新内容只有一个：** 惯性项怎么写成 $q$ 的函数，以及它的梯度和 Hessian。

对应书上：§7.4 增量势（第 150–152 页），§7.5 惯性项：body 形式（第 152–157 页）。

---

## 1. 质量分布

每个刚体需要三样东西（`Body` 里的新字段，`src/model.cc` 从 MuJoCo 读，已经写好）：

| 字段 | MuJoCo | 含义 |
|---|---|---|
| `mass` | `body_mass` | 质量 $m_b$ |
| `ipos` | `body_ipos` | 质心在刚体坐标系里的位置 |
| `inertia` | `body_inertia`、`body_iquat` | 绕质心的惯性张量 $I_b$（刚体坐标系） |
| `S` | 由 $I_b$ 换算 | 绕质心的二阶矩 $S_b = \sum_p m_p X_p X_p^\top$ |

$X_p$ 是质点相对质心的坐标（刚体坐标系）。惯性张量和二阶矩是同一个信息的两种写法：

$$
I = \operatorname{tr}(S)\,\mathbf 1 - S, \qquad S = \tfrac12 \operatorname{tr}(I)\,\mathbf 1 - I .
$$

（第一个式子是 $I$ 的定义 $\sum_p m_p(|X_p|^2 \mathbf 1 - X_p X_p^\top)$；两边取迹得 $\operatorname{tr} I = 2\operatorname{tr} S$，代回去就是第二个。）

MuJoCo 给的是主惯量（3 个数）和主轴相对刚体坐标系的姿态，`from_mujoco` 先在主轴里算对角阵，再转到刚体坐标系。

---

## 2. 惯性项

### 2.1 每个质点的目标

想象刚体由很多小质点组成。如果每个质点各自做隐式 Euler，它下一步“想去”的位置是

$$
\tilde x_p = x_p^n + (x_p^n - x_p^{n-1}) + g h^2 = 2x_p^n - x_p^{n-1} + g h^2 .
$$

（按上一步的速度再走一步，再加上重力在这一步造成的位移；$g$ 是重力加速度向量，指向下方。）
这就是阶段 1 的 $\tilde x = x_0 + h v_0 - h^2 g$，只是把速度写成了两个位置的差。

质点被刚性地绑在一起，不能各走各的，于是找一个刚体运动，让所有质点离各自的目标**总体最近**：

$$
E_b = \frac{1}{2h^2} \sum_p m_p\, |x_p - \tilde x_p|^2 .
$$

### 2.2 化成每个刚体两个量

刚体 $b$ 的质心是 $c_b$、转动矩阵是 $R_b$，质点的世界位置是 $x_p = c_b + R_b X_p$。代入目标：

$$
\tilde x_p = \underbrace{(2c_b^n - c_b^{n-1} + g h^2)}_{\tilde c_b} + \underbrace{(2R_b^n - R_b^{n-1})}_{\tilde A_b} X_p .
$$

所以 $x_p - \tilde x_p = (c_b - \tilde c_b) + (R_b - \tilde A_b) X_p$。平方求和，交叉项含 $\sum_p m_p X_p = 0$（质心在原点），消掉；剩下

$$
E_{\text{inert}}(q) = \frac{1}{2h^2} \sum_b \Big[\, m_b\, |c_b(q) - \tilde c_b|^2 + \operatorname{tr}\!\big( (R_b(q) - \tilde A_b)\, S_b\, (R_b(q) - \tilde A_b)^\top \big) \Big].
$$

- 第一项就是阶段 1 的 $\frac{m}{2h^2}(x - \tilde x)^2$ 搬到三维；
- 第二项管转动：$S_b$ 越“大”的方向，转错了代价越高；
- $c_b(q)$、$R_b(q)$ 由 FK 算出来，所以这是 $q$ 的函数。

**几点说明：**

1. **重力在目标里。** 展开 $\frac{m}{2h^2}|c - \hat c - gh^2|^2 = \frac{m}{2h^2}|c - \hat c|^2 - m\,g\cdot c + \text{常数}$，其中 $-m\,g\cdot c$ 正是重力势能，所以不用另写重力项。
2. **$\tilde A_b$ 不是转动矩阵。** 两个转动矩阵的线性组合一般不在 SO(3) 里。没关系：最小化会选一个离它最近的转动（按 $S_b$ 加权），这一步“投影”会带来一点数值耗散（思考题 1）。
3. **上一步的位姿从哪来。** 我们只存了 $(q_n, v_n)$，所以用 $q_{n-1} = q_n \oplus (-h\, v_n)$ 反推，它正好是 $v_n = (q_n \ominus q_{n-1})/h$ 的逆运算。

### 2.3 为什么不用关节空间的写法

最“自然”的写法是直接在关节空间里套 IPC 的惯性项：

$$
E^{\text{joint}}(q) = \tfrac12 (v(q) - \tilde v)^\top M(q_n) (v(q) - \tilde v), \qquad \tilde v = v_n + h\, \ddot q_{\text{smooth}} .
$$

问题出在 $\tilde v$：科氏力和离心力藏在 $\ddot q_{\text{smooth}}$ 里，只在步首算一次，是**显式**的。对快速旋转的链条，显式处理速度平方项会凭空注入能量。
书 §7.5 图 35 的实验：一个布娃娃没有重力、没有接触，以 5 rad/s 起转，3 秒内关节空间写法的动能涨到初始的上万倍，body 形式始终不超过 1。

body 形式里没有科氏力：每个刚体只是“按惯性走”，FK 把它们连起来，速度平方项自然包含在 $c_b(q)$、$R_b(q)$ 的非线性里，被隐式地处理了。

---

## 3. 梯度

Newton 要的是 $\partial E / \partial \delta$（$\delta$ 是阶段 2 的切空间增量）。分两步：先求能量对**刚体运动**的导数，再用 Jacobian 投影到关节上。

### 3.1 平移

$\frac{m}{2h^2}|c - \tilde c|^2$ 对 $c$ 的导数是

$$
f_b = \frac{m_b (c_b - \tilde c_b)}{h^2} .
$$

它的量纲是力：质心离目标越远，“拉回去的力”越大。

### 3.2 转动

刚体以角速度 $\omega$（世界系）转一点时，$\delta R = [\omega]_\times R$（$[\omega]_\times$ 是 $\omega \times$ 的矩阵）。记 $D = R - \tilde A$：

$$
\delta\Big(\tfrac12 \operatorname{tr}(D S D^\top)\Big) = \operatorname{tr}(\delta R\, S D^\top) = \operatorname{tr}\big([\omega]_\times R S D^\top\big) = \omega \cdot \operatorname{vex}(R S D^\top),
$$

最后一步用了恒等式 $\operatorname{tr}([\omega]_\times A) = \omega\cdot\operatorname{vex}(A)$，其中 $\operatorname{vex}(A) = (A_{23} - A_{32},\ A_{31} - A_{13},\ A_{12} - A_{21})$（直接展开 $3\times 3$ 的迹就能验证）。所以

$$
\tau_b = \frac{1}{h^2}\, \operatorname{vex}\big(R_b S_b (R_b - \tilde A_b)^\top\big),
$$

量纲是力矩。

### 3.3 投影到关节

关节沿 $\delta$ 动时，质心的速度是 $V_b \delta$、角速度是 $W_b \delta$：

- $V_b$ = 质心处的 `point_jacobian`（阶段 2）；
- $W_b$ = `angular_jacobian`（这个阶段新加的，链上转动自由度那一列是 $a_d$，其余为 0）。

代入 $\delta E_b = f_b\cdot \delta c + \tau_b\cdot\omega$：

$$
\frac{\partial E}{\partial \delta} = \sum_b \big( V_b^\top f_b + W_b^\top \tau_b \big).
$$

按列展开，转动自由度 $d$ 得到 $a_d \cdot (\tau_b + (c_b - o_d) \times f_b)$，平移自由度得到 $a_d \cdot f_b$。这就是书上说的**“把每个刚体受的力和力矩，沿链投影到每个关节轴上”**。

阶段 4 的接触梯度是同一个套路：顶点上的“力”（障碍的梯度）经 `point_jacobian` 投影到关节上。

---

## 4. Hessian：就是质量矩阵

把每个刚体的能量在当前点按 $\delta$ 展开到二阶，丢掉“残差乘二阶导数”的项（Gauss-Newton 近似）：

$$
H = \frac{1}{h^2} \sum_b \Big( m_b\, V_b^\top V_b + W_b^\top I_b^{\text{world}} W_b \Big), \qquad I_b^{\text{world}} = R_b I_b R_b^\top .
$$

括号里正是关节空间质量矩阵的定义：动能 $\tfrac12 v^\top M v = \sum_b \big(\tfrac12 m_b |\dot c_b|^2 + \tfrac12 \omega_b^\top I_b \omega_b\big)$，而 $\dot c_b = V_b v$、$\omega_b = W_b v$。所以

$$
H = \frac{M(q)}{h^2} .
$$

测试 `Hessian.IsTheMassMatrixOverHSquared` 拿它乘 $h^2$ 和 MuJoCo 的 `mj_fullM` 比，一致到 $10^{-12}$。质量矩阵不用单独学，它是从惯性项里自然长出来的。

**为什么用 Gauss-Newton：**

- $M(q)$ 总是对称正定，所以 Newton 方向总是下降方向，`ldlt()` 总能解；
- 被丢掉的项在刚体正好按惯性运动（$c = \tilde c$、$R = \tilde A$）时为 0（测试 `Hessian.IsExactWhenEveryBodyIsOnItsTarget`），只影响收敛速度，不影响收敛到哪里。
- **后来发现它不总是小的**：轻的连杆吊着重物时，残差就是链上的拉力，丢掉的这一项是绳子的几何刚度，比 $M/h^2$ 大几十倍，Newton 每步撞上限。阶段 13 把它加回来了（[16_inertia_curvature.md](16_inertia_curvature.md)，思考题 3 也在那里接着讨论）。

---

## 5. 时间步

和阶段 1 的 `ipc1d::step` 逐行对应：

| | 阶段 1 | 阶段 3 |
|---|---|---|
| 准备 | `xt = predicted(P, x0, v0)` | `q_prev = q0 ⊕ (-h v0)`，`T = inertia_targets(FK(q0), FK(q_prev))` |
| 方向 | `p = -E'/E''` | 解 $H p = -g$（`H.ldlt().solve(-g)`） |
| 收敛 | `|p| / h < tol` | `max_i |p_i| / h < tol` |
| CCD | `ccd_max_step` | 还没有接触，上限是 1 |
| 线搜索 | `energy(x + alpha p)` | `inertia_energy(FK(integrate(q, p, alpha)))` |
| 更新 | `x = x + alpha p` | `q = integrate(q, p, alpha)` |
| 速度 | `(x - x0) / h` | `differentiate(q0, q) / h` |

`include/rcipc/step.h` 里还有 `kinetic_energy` 和 `potential_energy`（已经写好），用来检查能量守恒。动能 $\tfrac12 v^\top M v$ 直接借用 `inertia_hessian(..., h = 1)` 当 $M$。

---

## 6. 结果

参考实现的数字（你的实现应该一样）：

**自由落体（`box.xml`，从静止开始一步）：** 位移正好是 $g h^2$，速度 $g h$，Newton 次数 2（求解一次、确认一次），和阶段 1 一样。平移部分的能量是二次函数，一步就到。

**双摆（`double_pendulum.xml`，两根杆都水平、静止释放）：**

| 步长 $h$ | 0.5 s 时摆角和精确解的差 | 0.5 s 内损失的能量 | Newton 次数（最多 / 平均） |
|---|---|---|---|
| 0.002 | 0.043 rad | 0.48 J（总能量 19.62 J 的 2.4%） | 4 / 3.76 |
| 0.001 | 0.022 rad | 0.25 J | 4 / 3.51 |
| 0.0005 | 0.011 rad | 0.13 J | 4 / 3.12 |

步长减半，误差和能量损失都减半：隐式 Euler 是一阶方法。总能量从不高于初始值。
（“精确解”是 MuJoCo 的 RK4，步长 $h/10$。双摆是混沌的，比较时间再长，摆角的差就没有意义了。）

**旋转的盒子（`box.xml`，没有重力，绕中间那根主轴以 5 rad/s 转，5 秒）：**

| 步长 $h$ | 动能最大 / 初始 | 5 秒后动能 / 初始 |
|---|---|---|
| 0.01 | 1.000000 | 0.29 |
| 0.005 | 1.000000 | 0.45 |
| 0.002 | 1.000000 | 0.67 |

动能**从不增加**，这正是 body 形式要保证的。但它会减少，而且步长大时减少得很多：这是隐式 Euler 的数值耗散（思考题 1）。

---

## 7. 代码对照

建议按这个顺序读，每个函数的注释都标了对应的小节：

| 顺序 | 函数 | 文件 | 笔记 | 测试 |
|---|---|---|---|---|
| 1 | `from_mujoco` 里的质量部分 | `src/model.cc` | 第 1 节 | `test_inertia --gtest_filter=Mass.*` |
| 2 | `inertia_targets` | `src/inertia.cc` | 第 2.2 节 | `test_inertia --gtest_filter=Targets.*` |
| 3 | `inertia_energy` | `src/inertia.cc` | 第 2.2 节 | `test_inertia --gtest_filter=Gradient.*`（用能量的有限差分检查梯度） |
| 4 | `inertia_gradient` | `src/inertia.cc` | 第 3 节 | 同上 |
| 5 | `inertia_hessian` | `src/inertia.cc` | 第 4 节 | `test_inertia --gtest_filter=Hessian.*` |
| 6 | `step` | `src/step.cc` | 第 5 节 | `test_step`（和 `toys/ipc1d/ipc1d.cc` 逐行对照） |

演示程序：

```
build/Debug/dynamics_demo.exe pendulum
build/Debug/dynamics_demo.exe pendulum h=0.0005 T=0.5
build/Debug/dynamics_demo.exe spin
build/Debug/dynamics_demo.exe spin h=0.002 w=10
```

`pendulum` 和 `spin` 各写一个 CSV（`pendulum.csv`、`spin.csv`）。双摆也可以用 MuJoCo 的 `simulate.exe` 打开 `models/double_pendulum.xml` 看它摆。

---

## 8. 思考题

1. 旋转的盒子在 $h = 0.01$ 时 5 秒内损失了 71% 的动能。能量去哪了？设刚体绕一根固定轴匀速转动，每步转角 $\theta = \omega h$。算一算 $\tilde A = 2R^n - R^{n-1}$ 离它最近的转动转了多少角度（提示：只看垂直于转轴的 $2\times2$ 块），和 $\theta$ 比是大还是小？这对接触仿真是不是问题？
2. 为什么 $\tilde A$ 不是转动矩阵也没关系？如果先把 $\tilde A$ 投影成最近的转动矩阵再用，能量和梯度要改吗？
3. Gauss-Newton 丢掉了一项。双摆每步要 3–4 次 Newton 迭代，而自由落体只要 2 次，为什么？如果用精确的 Hessian，迭代次数会变少吗？代价是什么？
4. 把 $\tilde c_b = \hat c_b + g h^2$ 代进去展开，验证重力势能 $-m\,g\cdot c$ 确实藏在惯性项里。如果某个刚体不受重力（比如在水里有浮力抵消），代码里要怎么改？
5. 关节空间写法的 $\tilde v = v_n + h\,\ddot q_{\text{smooth}}$ 里，哪一部分是显式的？为什么显式处理速度平方项（科氏力、离心力）会让能量增加，而隐式的 body 形式只会让它减少？
6. 线搜索每试一个 $\alpha$ 就要做一次 FK。一个时间步里大概做了多少次 FK？`inertia_hessian` 的代价和刚体数、自由度数是什么关系？对一个 20 节的链条，它会是瓶颈吗？
