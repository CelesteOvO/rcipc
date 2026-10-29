# 阶段 1：一维 IPC

**目标：** 在最简单的情形下把 IPC 的四个部件弄清楚：增量势、障碍函数、Newton 法加线搜索、CCD。
一个 1 kg 的球竖直落向地面，状态是球底到地面的间隙 $x$（m）和竖直速度 $v$（m/s）。

对应书上第 3 章：§3.1 增量势（第 46–47 页），§3.2 障碍函数（第 48–50 页），§3.4 CCD 与保守推进（第 52–53 页），
§3.5 Newton 法与线搜索（第 54–56 页），§3.8 一维例子（第 61–63 页）。

后面的阶段会把这里的每一个量换成“大的”：$x$ 换成关节坐标 $q$，$m$ 换成质量矩阵，距离换成点-三角形和边-边距离，
CCD 换成沿弧线的保守推进。算法的骨架不变。

---

## 1. 增量势

隐式 Euler 用的是新时刻的力：

$$
m\,\frac{x_1 - x_0 - h v_0}{h^2} = F(x_1), \qquad v_1 = \frac{x_1 - x_0}{h}.
$$

这里 $F$ 包括重力 $-mg$（指向地面，$x$ 是间隙）和障碍力 $-\kappa b'(x)$。把重力并进**预测位置**

$$
\tilde x = x_0 + h v_0 - h^2 g ,
$$

它就是“没有地面时这一步会走到的位置”。然后定义**增量势**

$$
E(x) = \frac{m}{2h^2}\,(x - \tilde x)^2 + \kappa\, b(x).
$$

求导得到

$$
E'(x) = \frac{m}{h^2}(x - \tilde x) + \kappa\, b'(x), \qquad
E''(x) = \frac{m}{h^2} + \kappa\, b''(x).
$$

令 $E'(x) = 0$，得到的正好是上面的隐式 Euler 方程。所以一步时间积分就是一次最小化：$x_1 = \arg\min_x E(x)$。

**单位：** $m/h^2$ 是 kg/s²，乘上 m² 得到焦耳。$b$ 的单位是 m²，$\kappa$ 的单位是 N/m，所以 $\kappa b$ 也是焦耳。

> 记号说明：IPC 原论文写的是 $\tfrac12 (x-\tilde x)^\top M (x-\tilde x) + h^2 \kappa B(x)$，比这里多乘了一个 $h^2$，
> 最小点相同。本仓库沿用书上的约定（能量单位是焦耳，$\kappa$ 的单位是 N/m）。到阶段 9 用 IPC 的公式初始化 $\kappa$ 时，要换算这个因子。

**要实现：** `predicted`、`energy`、`gradient`、`hessian`（在 [toys/ipc1d/ipc1d.cc](../toys/ipc1d/ipc1d.cc) 里）。

## 2. 障碍函数

$$
b(d) = \begin{cases} -(d - \hat d)^2 \ln\dfrac{d}{\hat d}, & 0 < d < \hat d, \\[4pt] 0, & d \ge \hat d. \end{cases}
$$

记 $r = d - \hat d$，$L = \ln(d/\hat d)$，则 $b = -r^2 L$。由乘积法则（$r' = 1$，$L' = 1/d$，$L'' = -1/d^2$）：

$$
b'(d) = -2 r L - \frac{r^2}{d}, \qquad
b''(d) = -2L - \frac{4r}{d} + \frac{r^2}{d^2}.
$$

几个性质（测试会逐条检查）：
- 在 $d = \hat d$ 处 $r = 0$、$L = 0$，所以 $b$、$b'$、$b''$ 都等于 0，和外侧的 0 函数 $C^2$ 光滑地接上。Newton 法不会在接触刚激活时被不连续绊倒。
- $d \to 0^+$ 时 $b \to +\infty$，$-b'(d) \approx \hat d^2/d \to +\infty$：斥力没有上界，有限的载荷压不到 $d = 0$。
- 在 $(0, \hat d)$ 上 $b > 0$，$b' < 0$（推开），$b'' > 0$（凸）。

**$\kappa$ 决定停在多远，不决定会不会穿透。** 静止时障碍力和重力平衡：$-\kappa b'(d) = mg$。
$\kappa$ 越大，平衡点越靠近 $\hat d$；但无论 $\kappa$ 取多少，平衡点都在 $d > 0$。

**接口为什么写成 `barrier(x, xhat)`：** 从阶段 4 开始，自变量换成平方距离，调用 `barrier(d*d, dhat*dhat)`，
旧实现也是这样做的。函数本身不变。原因是平方距离对顶点的导数在任何距离区域（点-面、点-边、点-点）都是同一个形式，
不需要开方（书 §3.3）。

**要实现：** `barrier`（在 [src/barrier.cc](../src/barrier.cc) 里）。$x \ge \hat x$ 时返回 `{}`，即三项全为 0。

## 3. 一步怎么解

### 3.1 Newton 方向

在当前点 $x$ 用二次函数近似 $E$，跳到这个二次函数的最低点：

$$
p = -\frac{E'(x)}{E''(x)}.
$$

一维里 $E'' = m/h^2 + \kappa b'' > 0$，所以 $p$ 一定指向能量下降的方向。

### 3.2 CCD：保守推进

沿 $x + s p$（$s \in [0, 1]$）走，什么时候碰到地面？$p \ge 0$ 时永远不会碰到。$p < 0$ 时碰撞时刻是
$t^* = x / (-p)$，一维里这是精确的。

我们不走到 $t^*$，而是只允许这一次迭代**最多闭合当前间隙的 $c$ 倍**（$c = 0.9$），也就是保证 $x + s p \ge (1 - c)\,x$：

$$
\alpha_{\max} = \begin{cases} 1, & p \ge 0, \\ \min\!\left(1,\ \dfrac{c\,x}{-p}\right), & p < 0. \end{cases}
$$

这就是保守推进（conservative advancement）的一维版本：留出余量，停下时间隙仍然是正的。
后面的阶段用的是同一个规则，只不过是对每一个接触对分别做（“每对最多闭合初始间隙的 $c$ 倍”，旧代码的 `curved_ca`）。

> 和书上代码的区别：书 §3.8 写的是 `amax = 1 if x + p > 0 else 0.9 * x / (-p)`，只有整步会穿过地面时才截断。
> 整步落在 $(0, (1-c)x)$ 之间时两者不同：书上的版本会走满整步，我们的版本会截断。书上所有例子里两者给出的数字完全相同。

**要实现：** `ccd_max_step`。

### 3.3 回溯线搜索

Newton 的二次模型会低估障碍有多陡，所以 CCD 允许的步长不一定让能量下降。
从 $\alpha = \alpha_{\max}$ 开始，只要 $E(x + \alpha p) > E(x)$ 就把 $\alpha$ 减半（$\alpha$ 小于 $10^{-12}$ 时停止）。

为什么 CCD 和线搜索两个都要：CCD 保证路径上不相交，线搜索保证能量下降。只靠能量是不够的，见思考题 6。

### 3.4 整个循环

```
x~ = predicted(x0, v0);  x = x0
for it = 1 .. max_newton:
    p = -E'(x) / E''(x)
    if |p| / h < tol: break                    # 收敛；这一次也计入 newton
    amax = ccd_max_step(x, p, c)
    a = amax
    while E(x + a p) > E(x) and a > 1e-12: a /= 2
    trace 追加 {x, p, amax, a}                  # trace 不为空指针时
    x = x + a p
return {x, (x - x0) / h, it}                    # 跑满 max_newton 时 it 是 max_newton
```

收敛判据用“位移除以 $h$”，因为它的单位是速度，换步长以后仍然可以比较。

**要实现：** `step`。

### 3.5 落地仿真

球底从 0.45 m 静止落下，$h$ = 2 ms，$\kappa = 10^4$ N/m，$\hat d$ = 1 mm，模拟 1.5 s。书上的结果是：

| | 书 §3.8 |
|---|---|
| 最小间隙 | 0.0078 mm（t = 0.304 s） |
| 静止间隙 | 0.5359 mm |
| 静止速度 | 0 |
| 每步 Newton 次数 | 最多 9，平均 1.44 |

参数扫描（书 §3.8 末尾的表）：

| κ（N/m） | h（ms） | 初速度（m/s） | 最小间隙（mm） | 静止间隙（mm） |
|---|---|---|---|---|
| 1e3 | 2 | 0 | 0.0007 | 0.1244 |
| 1e4 | 2 | 0 | 0.0078 | 0.5359 |
| 1e5 | 2 | 0 | 0.0997 | 0.8301 |
| 1e4 | 10 | 0 | 0.0496 | 0.5359 |
| 1e4 | 10 | -40 | 0.0029 | 0.5359 |

最后一行是以 40 m/s 撞向地面、每步走 40 cm，间隙仍然从未为负。测试 `Drop.*` 和 `Book/DropSweep.*` 会检查这些数字。

---

## 4. 你要做的事

1. 实现 7 个函数，每个都只有几行：
   - [src/barrier.cc](../src/barrier.cc)：`barrier`；
   - [toys/ipc1d/ipc1d.cc](../toys/ipc1d/ipc1d.cc)：`predicted`、`energy`、`gradient`、`hessian`、`ccd_max_step`、`step`。
2. 建议的顺序，以及每一步对应的测试：

   | 函数 | 测试 |
   |---|---|
   | `barrier` | `test_barrier`（全部） |
   | `predicted`、`energy`、`gradient`、`hessian` | `test_ipc1d --gtest_filter=Energy.*` |
   | `ccd_max_step` | `test_ipc1d --gtest_filter=Ccd.*` |
   | `step` | `test_ipc1d --gtest_filter=Step.*:Drop.*:Book/*` |

3. 编译和测试（在仓库根目录）：

   ```
   cmake -S . -B build
   cmake --build build --config Debug
   ctest --test-dir build -C Debug --output-on-failure
   ```

   只跑一组：`build\tests\Debug\test_ipc1d.exe --gtest_filter=Ccd.*`

4. 全部通过以后跑演示：

   ```
   build\Debug\ipc1d_demo.exe impact
   build\Debug\ipc1d_demo.exe drop
   python tools/plot_ipc1d.py ipc1d.csv
   ```

   `impact` 会打印书 §3.5.3 那张逐次迭代的表，`drop` 会打印书 §3.8 的结果，并写出 `ipc1d.csv`。
   演示的参数都可以改，例如 `ipc1d_demo drop kappa=1e5 h=0.01 v0=-40`，或者 `ipc1d_demo impact c=1`。
5. 做完告诉我。我会 review 代码，然后我们一起过一遍下面的思考题，最后提交阶段 1。

## 5. 思考题

用演示程序可以直接验证的题，后面给了命令。

1. 为什么 $E'(x) = 0$ 就是隐式 Euler？如果不把重力并进 $\tilde x$，而是把它写成势能 $\Psi(x) = m g x$ 加进 $E$，最小点会变吗？
2. 用 $-\kappa b'(d) = mg$ 解释：为什么扫描表里静止间隙只和 $\kappa$ 有关，和 $h$、初速度都无关？
3. 把 $c$ 改成 1（`ipc1d_demo impact c=1`）。为什么还是不穿透，但迭代次数从 7 变成了 13？$c < 1$ 带来了什么？
4. 一维里 $E''$ 永远是正的，所以 $p$ 一定是下降方向。到了阶段 4，障碍的 Hessian 是
   $\kappa\,(b''\,\nabla d\,\nabla d^\top + b'\,\nabla^2 d)$，其中 $b' < 0$，后一项可能不定。为什么一维里没有这一项？
   旧实现的做法是丢掉它（Gauss-Newton）。在关节坐标里丢掉它会出什么问题，是阶段 6 的内容。
5. 测试 `Step.ImpactMatchesTheBook` 里，我们比书上的表多一次迭代：第 6 次的修正只有 7.5 nm，但 $|p|/h$ = 3.8e-6 m/s，
   仍然大于 `tol` = 1e-6 m/s。试试 `ipc1d_demo impact tol=1e-5`。容差应该怎么选？
   （旧实现在自行车链上吃过这个亏：布娃娃用的 0.05 m/s 相当于每步 0.5 mm，而链条每步一共只动 0.17 mm。）
6. 去掉 CCD（`ipc1d_demo impact c=1e9`，这时 $\alpha_{\max}$ 总是 1）。障碍在 $d \to 0$ 时是无穷大，为什么线搜索没有挡住穿透？
   提示：看第 1 次迭代之后的间隙，再想一想 $\ln$ 在负数上返回什么，以及 `NaN > e0` 的值是什么。
   在三维里情况更糟：跳到墙另一侧以后，距离又是正的，能量是有限值。
7. 把 $\hat d$ 加大到 5 mm（`ipc1d_demo impact dhat=5e-3`），撞击那一步只要 5 次迭代。为什么？代价是什么？
