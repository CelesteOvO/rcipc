# 阶段 8：滞后的光滑化 Coulomb 摩擦

**目标：** 给接触加上切向的摩擦力。难点在于 Coulomb 摩擦不是任何势能的梯度，而阶段 1 的框架要求每个力都来自势能。IPC 用两个近似把它放进增量势：滞后和光滑化。

对应书上：§3.6 摩擦（第 57–59 页），§8.7 滞后摩擦（第 186–189 页）。全流程见 [00_pipeline.md](00_pipeline.md) 第 6 节。

---

## 1. 滞后：步首把什么冻结

Coulomb 摩擦说：静止时摩擦力可以是 0 到 $\mu\lambda$ 之间的任何值，滑动时等于 $\mu\lambda$、方向和滑动相反。它依赖法向力 $\lambda$ 和切平面，而这两者又依赖未知的 $q$。

**滞后（lagged）：** 在步首（$q_n$ 处）把下面这些量算好，这一步里当常数（`lag_friction`）：

| 量 | 取值 |
|---|---|
| 接触集合 | 步首活跃的接触（模板、权重固定） |
| 法向力 $\lambda_k$ | $-\kappa\,m\,b'(x)\,2d$（阶段 4 的力，边–边的 mollifier $m$ 也在里面） |
| 切向基 $t_1, t_2$ | 垂直于步首法向 $n$ 的两个单位向量 |
| 顶点位置 $X_v^n$ | 步首的世界坐标 |

两个直接后果（书第 188 页）：这一步里新出现的接触没有摩擦，要到下一步才有；法向力用的是步首的值，冲击那一步的摩擦力不准确。第一次出现接触的那一步 $\kappa$ 还没初始化，$\lambda = 0$，也没有摩擦。

---

## 2. 光滑化：$f_0$ 和 $f_1$

**相对切向速度。** 两侧最近点（按步首的权重）的相对速度 $V = \sum_v w_v\,(X_v(q) - X_v^n)/h$（隐式 Euler：位移除以 $h$），在切平面里的分量 $u = (t_1\cdot V,\ t_2\cdot V)$。

**耗散功当势能。** 滑动摩擦力 $\mu\lambda$，一步里滑过 $h\lvert u\rvert$，耗散的功 $h\mu\lambda\lvert u\rvert$。对 $u$ 求导得 $h\mu\lambda\,\hat u$，再经 $\partial u/\partial X = 1/h$，得到力 $\mu\lambda\,\hat u$，正是滑动摩擦力。

**光滑化。** $\lvert u\rvert$ 在 0 处不可导（静摩擦的“任何值”就藏在这个角里）。在 $\varepsilon_v$ 以内把它换成 $f_0$，要求 $f_1 = f_0'$（摩擦力的系数）满足 $f_1(0) = 0$、$f_1(\varepsilon_v) = 1$、$f_1'(\varepsilon_v) = 0$。最简单的是二次多项式：

$$
f_1(y) = \frac{2y}{\varepsilon_v} - \frac{y^2}{\varepsilon_v^2}, \qquad
f_0(y) = -\frac{y^3}{3\varepsilon_v^2} + \frac{y^2}{\varepsilon_v} + \frac{\varepsilon_v}{3} \quad (y < \varepsilon_v)
$$

$y \ge \varepsilon_v$ 时 $f_0 = y$、$f_1 = 1$。$f_0$ 的常数项让它在 $\varepsilon_v$ 处接上 $y$。于是

$$
E_{\text{fric}} = \sum_k h\,\mu\,\lambda_k\, f_0(\lvert u_k\rvert)
$$

单位：s · N · m/s = J。静摩擦被近似成“在 $\varepsilon_v$ 以内随速度线性增长的摩擦力”：静止的物体会以远小于 $\varepsilon_v$ 的速度蠕动（第 5 节）。

---

## 3. 梯度和 Hessian

记切向行 $\rho_j = \partial(t_j\cdot P)/\partial\delta$：接触行的算法，把 $n$ 换成 $t_j$，不乘 $2d$（`tangent_row`）。由 $\partial u_j/\partial\delta = \rho_j/h$ 和链式法则：

$$
\nabla E_{\text{fric}} = \sum_k\sum_j c_k\,u_{k,j}\,\rho_{k,j}^\top, \qquad c_k = \frac{\mu\lambda_k\,f_1(\lvert u_k\rvert)}{\lvert u_k\rvert}
$$

$f_1(y)/y$ 在 $y\to0$ 时趋于 $2/\varepsilon_v$，没有除零（`friction_f1_over_y`）。每个接触两条行，梯度系数 $c\,u_j$。

**Hessian。** 切平面里 $f_0(\lvert u\rvert)$ 的精确二阶导数是

$$
\frac{f_1}{\lvert u\rvert}\big(\mathbf 1 - \hat u\hat u^\top\big) + f_1'\,\hat u\hat u^\top
$$

垂直于 $u$ 的方向刚度 $f_1/\lvert u\rvert$，沿 $u$ 的方向刚度 $f_1'$（滑动时为 0：滑动摩擦力的大小不随速度变）。两个系数都 $\ge 0$，所以它半正定，可以直接用：

$$
H_{\text{fric}} = \sum_k\frac{\mu\lambda_k}{h}\,R_k^\top\Big[\frac{f_1}{\lvert u\rvert}(\mathbf 1 - \hat u\hat u^\top) + f_1'\,\hat u\hat u^\top\Big]R_k, \qquad R_k = \begin{pmatrix}\rho_{k,1}\\ \rho_{k,2}\end{pmatrix}
$$

书上的代码用 $\tfrac{f_1}{\lvert u\rvert}\mathbf 1$ 代替方括号（径向不小于精确值，一个上界，第 186–187 页）。本仓库先照做，台阶场景里撞上台阶边的那一步 Newton 跑满了 50 次：撞击时 $\lambda$ 很大而 $\lvert u\rvert$ 刚过 $\varepsilon_v$，上界在滑动方向上比真实值硬几百倍，Newton 在那个方向上爬行。换成精确值以后那一步 24 次。这和阶段 6 思考题 1 是同一类现象：模型比真实能量硬，Newton 就慢。

---

## 4. 时间步里的改动

`step()` 加三处（`src/step.cc`）：

1. 步首：`lag_friction(F0, find_contacts(F0), state->kappa)`；
2. 梯度和 Hessian 加上摩擦项；
3. 能量（线搜索用）加上摩擦项。

`StepParams::mu`（默认 0.2）和 `eps_v`（默认 1 mm/s）。$\mu = 0$ 关掉摩擦。所有接触用同一个 $\mu$；每个几何体自己的摩擦系数（书第 188 页的 `scale_friction`）留到场景阶段。

---

## 5. 结果

斜面用“转重力”实现（`models/slope.xml`：重力设成 $g(\sin\theta, 0, -\cos\theta)$，盒子沿 $+x$ 下滑），$\tan\theta = 0.3$：

| | $\mu$ | 预期 | 测试 |
|---|---|---|---|
| 静止 | 0.5 | 不动。光滑化让它以 $f_1(u)\mu = \tan\theta$ 对应的速度蠕动：$f_1 = 0.6$，$u = 0.37\varepsilon_v$ | 1 s 后 $v_x$ 在 $0.37\varepsilon_v$ 的 20% 内，位移小于 $\varepsilon_v$ |
| 下滑 | 0.2 | 匀加速 $a = g(\sin\theta - \mu\cos\theta) = 0.938$ m/s² | 1 s 后 $v_x$ 在 $a$ 的 5% 内（实测 0.943） |
| 平地上 1 m/s 滑动 | 0.2 | 以 $\mu g$ 减速，$1/(0.2\times9.81) = 0.51$ s 停下 | 停下的时刻在 0.51 ± 0.1 s |
| 无摩擦 | 0 | $a = g\sin\theta$ | 3% 内 |

**代价：** 滑动的盒子每步 Newton 约 11 次（静止时 1–2 次）。原因在步首：$u = 0$ 时静摩擦的刚度 $2\mu\lambda/\varepsilon_v$ 很大，Newton 的第一步把盒子当成粘在地上，惯性的推力和底面的“弹簧”一起产生一个俯仰力矩，前面的角被压下去；后面的迭代里 $u$ 变大、刚度变小，再慢慢回来。书上用 $\varepsilon_v = 0.01$ m/s 的场景居多，这个效应小 10 倍。

台阶和堆叠场景加上摩擦后仍然 0 穿透；台阶上的盒子翻倒后不再滑远（没有摩擦时滑了 1.2 m，现在停在 0.49 m）。

---

## 6. 代码对照

| 顺序 | 函数 | 文件 | 笔记 | 测试 |
|---|---|---|---|---|
| 1 | `friction_f0`、`friction_f1`、`friction_f1_over_y` | `src/friction.cc` | 第 2 节 | `Smoothing.*` |
| 2 | `lag_friction` | `src/friction.cc` | 第 1 节 | `Lag.*` |
| 3 | `friction_velocity`、`friction_energy`、`friction_gradient`、`friction_hessian` | `src/friction.cc` | 第 2、3 节 | `Gradient.*` |
| 4 | `step` 里的三处 | `src/step.cc` | 第 4 节 | `Slope.*` |

演示程序：`contact_demo slope mu=0.5`、`contact_demo slope mu=0.2`、`contact_demo slope theta=0.1 mu=0.2`、`contact_demo ledge mu=0`。

---

## 7. 思考题

1. 静摩擦被近似成“随速度线性增长”，静止的物体会蠕动。蠕动速度和 $\tan\theta/\mu$、$\varepsilon_v$ 的关系是什么？$\varepsilon_v$ 取 0.01 m/s 时 1 s 蠕动多少？这对机器人抓取（书上的 Allegro 手，$\varepsilon_v = 10^{-3}$）意味着什么？
2. 法向力滞后一步。盒子落地那一步，$\lambda$ 是步首的值（还没接触时是 0），所以落地的瞬间没有摩擦。这会让落地的盒子多滑多远？
3. 为什么摩擦能量不能像障碍那样直接用当前的 $\lambda(q)$？试着写出 $h\mu\lambda(q)\lvert u(q)\rvert$ 的梯度，看看哪一项不是一个力该有的样子。
4. 第 5 节的“每步 11 次”：把 $\varepsilon_v$ 改成 0.01 m/s 再跑 `contact_demo slope mu=0.2 eps_v=0.01`，Newton 次数怎么变？蠕动速度呢？
5. 滑动方向上精确 Hessian 是 0，书上用的上界是 $f_1/\lvert u\rvert$。用阶段 6 的“$H$ 偏大安全但慢”解释台阶场景那 50 次迭代；为什么这里可以放心用精确值，而障碍的二阶项不行？
6. 两个盒子堆叠、上面的盒子转 45°：接触是边–边的。边–边的切向基怎么取？两条边各自滑动时 $u$ 是什么？
