# 阶段 2：关节坐标

**目标：** 把阶段 1 的未知量 $x$（一个数）换成关节坐标 $q$（MuJoCo 的 `qpos`），并弄清楚三件事：

1. $q$ 怎么“加减”：四元数让 $q$ 的长度 $n_q$ 和自由度个数 $n_v$ 不相等，Newton 的方向 $\delta$ 活在 $n_v$ 维的切空间里；
2. 正运动学（FK）：给定 $q$，每个刚体在世界里的什么位置；
3. 点的 Jacobian：$q$ 沿 $\delta$ 变一点，刚体上的一个点怎么动。

这个阶段没有时间步，也没有接触，只有运动学。它是后面所有阶段的地基：
阶段 3 的惯性项、阶段 4 的接触距离、阶段 5 的 CCD 都要先跑一遍 FK，接触的梯度要用 Jacobian 把顶点空间的梯度变成 $q$ 空间的梯度。

参照答案是 MuJoCo：同一个模型、同一个 $q$，我们的结果要和 `mj_kinematics`、`mj_integratePos`、`mj_differentiatePos`、`mj_jac` 一致到舍入误差。

对应书上：§1.3 关节与广义坐标，§1.4 正运动学，§7.2 状态与坐标（第 144–146 页），§7.3 正运动学（第 147–149 页）。

---

## 1. 树结构

一个关节体是一棵树（或几棵树）：

- 0 号刚体是世界，不动；
- 每个刚体有一个父刚体，以及在父坐标系里的一个固定偏移 $(\bar p_b, \bar{\mathbf q}_b)$（MJCF 里的 `pos`、`quat`/`euler`）；
- 每个刚体至多带一个关节，关节决定它相对“固定偏移之后的位置”还能怎么动。没有关节的刚体焊死在父刚体上。

MuJoCo 编译模型时把刚体按深度优先排好，保证**父刚体的编号总比自己小**。所以 FK 按编号从小到大算一遍就行，算到 $b$ 时它的父刚体已经算好了。

四种关节在 $q$ 和 $v$ 里各占几个数：

| 关节 | $q$ 里 | $v$（和 $\delta$）里 | 含义 |
|---|---|---|---|
| 铰链 hinge | 1 | 1 | 转角（rad） |
| 滑动 slide | 1 | 1 | 位移（m） |
| 球 ball | 4 | 3 | 四元数；角速度写在**刚体自己的**坐标系里 |
| 自由 free | 7 | 6 | 位置 3 + 四元数 4；线速度在世界系，角速度在刚体系 |

所以 $n_q \ne n_v$。`models/zoo.xml` 里是 $n_q = 19$、$n_v = 16$。

`include/rcipc/model.h` 里的 `Model` 就是从 `mjModel` 抄出来的这些东西（`src/model.cc`，已经写好）：

| `Model` | MuJoCo | 含义 |
|---|---|---|
| `bodies[b].parent` | `body_parentid` | 父刚体 |
| `bodies[b].pos`、`.quat` | `body_pos`、`body_quat` | 固定偏移 |
| `bodies[b].joint` | `body_jntadr`（`body_jntnum` 为 0 时记 -1） | 关节编号 |
| `joints[j].qadr`、`.dadr` | `jnt_qposadr`、`jnt_dofadr` | 在 $q$、$v$ 里的起始下标 |
| `joints[j].pos`、`.axis` | `jnt_pos`、`jnt_axis` | 锚点和轴（刚体坐标系） |
| `joints[j].ref` | `qpos0[qadr]` | 铰链和滑动的参考值 |
| `dof_rot[d]` | 由关节类型推出 | 第 $d$ 个自由度是转动还是平移 |

**关于 `ref`：** MuJoCo 规定 $q = $ `ref` 时刚体正好在 MJCF 里写的姿态上，所以铰链实际转过的角度是 $q - \text{ref}$。`ref` 默认是 0，大多数模型里可以忽略，但 `zoo.xml` 故意给了非零值。

---

## 2. $q$ 的加法和减法

### 2.1 四元数的 exp 和 log

转动有 3 个自由度，四元数却有 4 个数（多出来的约束是长度为 1）。把“3 个数的转动增量”和“4 个数的四元数”联系起来的是指数映射。

**exp：** 旋转向量 $w$ 的方向 $n = w/|w|$ 是转轴，长度 $t = |w|$ 是转角：

$$
\exp(w) = \Big(\cos\tfrac t2,\ \sin\tfrac t2\, n\Big) = \Big(\cos\tfrac t2,\ \tfrac{\sin(t/2)}{t}\, w\Big).
$$

后一种写法在 $t=0$ 时是 $0/0$。它的极限是 $\tfrac12$，$t$ 很小时用展开 $\tfrac{\sin(t/2)}{t} = \tfrac12 - \tfrac{t^2}{48} + O(t^4)$。

**log：** 反过来，从 $\mathbf q = (c, v)$ 求 $w$。因为 $c = \cos\tfrac t2$、$|v| = \sin\tfrac t2$：

$$
t = 2\,\mathrm{atan2}(|v|, c), \qquad \log \mathbf q = t\,\frac{v}{|v|}.
$$

**双重覆盖：** $\mathbf q$ 和 $-\mathbf q$ 表示同一个转动（绕 $n$ 转 $t$ 等于绕 $-n$ 转 $2\pi - t$）。log 要返回短的那个（$t \le \pi$），也就是先把 $c < 0$ 的四元数整个取反。

**为什么用 atan2：** 它只看 $|v|$ 和 $c$ 的比例，所以输入不是单位四元数也能算对；而 $\arccos c$ 在 $c \approx 1$（小转角）时导数无穷大，误差会被放大。

### 2.2 ⊕ 和 ⊖

$q$ 不能直接加一个 $n_v$ 维的增量（长度都不一样）。书 §7.2 定义了两个运算：

$$
q \oplus \delta:\quad \text{铰链/滑动 } q_a + \delta_d;\qquad \text{球/自由的转动 } \mathbf q \otimes \exp(\delta_{\text{rot}});\qquad \text{自由的平移 } p + \delta_{\text{pos}},
$$

$$
q_1 \ominus q_0:\quad \text{铰链/滑动 } q_{1,a} - q_{0,a};\qquad \text{转动 } \log(\mathbf q_0^{-1} \otimes \mathbf q_1).
$$

它们和 MuJoCo 的 `mj_integratePos`、`mj_differentiatePos` 完全一样。

**右乘：** $\mathbf q \otimes \exp(w)$ 是“先按 $\mathbf q$ 摆好，再在**刚体自己的**坐标系里转 $w$”；左乘 $\exp(w) \otimes \mathbf q$ 是在世界坐标系里转。MuJoCo 的角速度写在刚体系里，所以是右乘。

**这对 Newton 意味着什么：** 从阶段 3 起，一次 Newton 迭代在当前点 $q_i$ 求一个 $n_v$ 维的方向 $\delta$，然后沿 $q_i \oplus s\delta$（$s \in [0,1]$）做 CCD 和线搜索。梯度和 Hessian 都是对 $\delta$ 在 $\delta = 0$ 处求的。
这样不用处理“四元数长度为 1”这个约束，$\delta$ 的每个分量都是真正的自由度。

一个时间步结束时，速度用 $v = (q_{n+1} \ominus q_n)/h$ 回算，这和阶段 1 的 $v = (x - x_0)/h$ 是一回事。

---

## 3. 正运动学

对每个刚体 $b$（父刚体 $p$，世界位姿 $(x_p, \mathbf q_p)$ 已知）：

**第 1 步，固定偏移。** 先假设关节不动：

$$
x = x_p + \mathbf q_p\, \bar p_b, \qquad \mathbf q = \mathbf q_p \otimes \bar{\mathbf q}_b .
$$

（$\mathbf q\, v$ 表示用 $\mathbf q$ 转动向量 $v$。偏移是在父坐标系里写的，所以要先用 $\mathbf q_p$ 转到世界。）

**第 2 步，关节。** 记锚点 $\bar o$（`jnt_pos`）、轴 $\bar a$（`jnt_axis`），它们都在刚体坐标系里。

- **铰链：** 锚点的世界坐标 $o = x + \mathbf q\, \bar o$。绕 $\bar a$ 转 $\theta = q_a - \text{ref}$：$\mathbf q \leftarrow \mathbf q \otimes (\cos\tfrac\theta2, \sin\tfrac\theta2\, \bar a)$。转动时锚点不动，所以原点要重新从锚点倒推：$x \leftarrow o - \mathbf q\, \bar o$。
- **球关节：** 同铰链，只是转动换成 $q$ 里的四元数（先归一化）。
- **滑动：** 不转，沿轴平移：$x \leftarrow x + (\mathbf q\, \bar a)(q_a - \text{ref})$。
- **自由关节：** 不管父刚体和固定偏移，直接取 $q$ 里的位置和四元数（MuJoCo 只允许自由关节挂在世界上）。

**第 3 步，自由度的轴和锚点。** 第 4 节的 Jacobian 要用到每个自由度 $d$ 在世界里的轴 $a_d$ 和锚点 $o_d$。顺手在 FK 里记下来：

| 关节 | 轴 $a_d$ | 锚点 $o_d$ |
|---|---|---|
| 铰链 | $\mathbf q\, \bar a$（转动前后一样，因为绕自己转） | $o$ |
| 滑动 | $\mathbf q\, \bar a$ | 用不到 |
| 球 | 转动**之后**的刚体坐标轴 $\mathbf q\, e_k$ | $o$ |
| 自由，平移 | 世界坐标轴 $e_k$ | 用不到 |
| 自由，转动 | 刚体坐标轴 $\mathbf q\, e_k$ | $x$ |

球关节和自由关节的转动轴是**刚体**坐标轴，这正是第 2.2 节“右乘、角速度写在刚体系里”的结果。

刚体上一个点在刚体坐标系里的坐标是 $\bar X$，它的世界坐标就是 $X = x_b + R_b \bar X$（`world_point`，已经写好）。从阶段 4 起，顶点都按这个方式由 $q$ 算出来，所以刚体严格保持刚性。

---

## 4. 点的 Jacobian

问题：刚体 $b$ 上一点 $X$，当 $q$ 沿 $q \oplus s\delta$ 动时，$\frac{dX}{ds}\big|_{s=0}$ 是多少？它对 $\delta$ 是线性的，写成 $J\delta$，$J$ 是 $3 \times n_v$ 的矩阵。

只有 $b$ 到根的链上的关节能带动 $X$。链上每个自由度 $d$ 单独以速度 1 运动时：

- 转动自由度：整个子树绕过 $o_d$、方向 $a_d$ 的轴以角速度 1 转，$X$ 的速度是 $a_d \times (X - o_d)$；
- 平移自由度：整个子树沿 $a_d$ 以速度 1 平移，$X$ 的速度是 $a_d$。

所有自由度一起动时速度叠加，所以

$$
J_{:,d} = \begin{cases} a_d \times (X - o_d), & d \text{ 是链上的转动自由度} \\ a_d, & d \text{ 是链上的平移自由度} \\ 0, & d \text{ 不在链上.} \end{cases}
$$

实现上，从 $b$ 沿 `parent` 一路走到世界，把经过的关节的列填上就行。

**检查它的两种办法：**

1. 和 MuJoCo 的 `mj_jac` 比（测试 `Jacobian.MatchesMujoco`）；
2. 有限差分：$\big[X(q \oplus \varepsilon e_d) - X(q \oplus -\varepsilon e_d)\big] / 2\varepsilon$（测试 `Jacobian.MatchesFiniteDifference`）。
   这个检查不依赖 MuJoCo，它说明 FK、⊕ 和 $J$ 三者用的是同一个切空间。注意扰动要用 ⊕，不能直接给 $q$ 的某个分量加 $\varepsilon$：$q$ 有 19 个数，$\delta$ 只有 16 个。

**以后怎么用：** 接触距离 $d(X)$ 对 $\delta$ 的梯度是 $J^\top \nabla_X d$（链式法则）。每个接触只涉及一两条链，所以这一行只有链上的几列非零，这是阶段 9 的稀疏结构的来源。

**一个预告：** $q \oplus s\delta$ 在切空间里是一条直线，但顶点 $X(q \oplus s\delta)$ 走的是曲线（铰链转动时是圆弧）。
阶段 1 的 CCD 默认“点沿直线走”，阶段 4 会照搬这个做法然后看着它穿模，阶段 5 再用“弦加矢高”修好（书图 34）。

---

## 5. 你要做的事

文件：`src/quat.cc`（2 个函数）和 `src/kinematics.cc`（4 个函数），每个函数里的 `TODO(stage 2)` 写了逐步的做法。
`src/model.cc` 和 `world_point` 已经写好。

建议的顺序，每做完一组就跑对应的测试：

| 顺序 | 函数 | 测试 |
|---|---|---|
| 1 | `quat_exp`、`quat_log` | `test_quat`（8 个） |
| 2 | `integrate`、`differentiate` | `test_kinematics --gtest_filter=Integrate.*:Differentiate.*` |
| 3 | `forward_kinematics` | `test_kinematics --gtest_filter=Fk.*` |
| 4 | `point_jacobian` | `test_kinematics --gtest_filter=Jacobian.*` |

`Model.ReadsTheZoo` 只测 `src/model.cc`，现在就应该通过。

```
cmake -S . -B build
cmake --build build --config Debug
build/tests/Debug/test_quat.exe
build/tests/Debug/test_kinematics.exe --gtest_filter=Fk.*
ctest --test-dir build -C Debug --output-on-failure
```

第一次配置会多下载一个 MuJoCo（约 22 MB）。编译时会把 `mujoco.dll` 复制到每个程序旁边。
想看看 `zoo.xml` 长什么样，可以运行 `build/_deps/mujoco-src/bin/simulate.exe models/zoo.xml`（CLion 的构建目录则是 `cmake-build-debug/_deps/...`）。

所有和 MuJoCo 的比较都要求误差在 $10^{-13}$ 左右。差一点点（比如 $10^{-3}$）通常是公式里少了一项（`ref`、锚点、归一化）；差得很多通常是左乘右乘弄反了。

---

## 6. 思考题

1. $n_q \ne n_v$。如果 Newton 直接对 $q$ 的 19 个分量求梯度和 Hessian，会遇到什么问题？（提示：四元数沿自己方向的伸缩改变 $q$，却不改变任何刚体的位置。Hessian 在这个方向上是什么样的？）
2. 把 `integrate` 里球关节的右乘改成左乘 $\exp(w) \otimes \mathbf q$，哪些测试会失败？为了让 Jacobian 和左乘的约定一致，球关节的 $a_d$ 和 $o_d$ 应该改成什么？
3. 如果 `quat_log` 不处理双重覆盖（不把 $c < 0$ 的四元数取反），哪个测试会失败？放到时间步里，用 $v = (q_{n+1} \ominus q_n)/h$ 回算速度时会发生什么？
4. 铰链转角 $\theta$、点到转轴的距离是 $R$ 时，点走的圆弧和它的弦（两个端点的连线）之间最大差多少？取 $R = 0.3$ m、$\theta = 1$ rad，和阶段 1 的 $\hat d = 1$ mm 比一比。这说明“顶点沿直线走”的 CCD 有什么问题？
5. 自由关节的平移自由度轴在世界系，转动自由度轴在刚体系。为什么平移不也用刚体系？（提示：看 `integrate` 里自由关节的平移部分是怎么加的。）
6. `point_jacobian` 沿父刚体往上走，花的时间正比于链的长度。一个接触行（两个刚体之间）最多有几列非零？对一个 20 节的链条，这意味着什么？
