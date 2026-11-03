# 阶段 6：曲率上界（curvature majoriser）

**目标：** 处理阶段 4 第 3 节丢掉的那一项。Gauss-Newton 只保留了障碍 Hessian 里“行的外积”$\kappa b''\,r^\top r$，
丢掉了 $\kappa b'\,\partial^2x/\partial\delta^2$。在笛卡尔 IPC 里这一项要么不存在、要么是安全的；在关节坐标里它含顶点轨迹的曲率，可能是正的，丢掉会让 Newton 低估能量的增长。

对应书上：§8.4 轨迹曲率与曲率 majoriser（第 175–180 页），§8.5 缩放 curv_scale（第 181–182 页）。全流程见 [00_pipeline.md](00_pipeline.md) 第 5 节。

---

## 1. 问题：Newton 把转动当成沿表面滑动

GN 用线性模型预测距离：$x(\delta)\approx x_0 + r\delta$。关节转动时顶点走弧线，距离还有二阶变化：

$$
x(\delta) = x_0 + r\delta + \tfrac12\,\delta^\top\frac{\partial^2x}{\partial\delta^2}\,\delta + \dots
$$

最极端的情形是顶点的速度正好平行于障碍表面：$r = 0$，GN 完全看不到障碍，以为这个方向可以随便走。弧线却朝障碍弯过去。于是 Newton 给出一个很大的步子，CCD 把它截短，下一次迭代再犯同样的错。书上在自行车链条上观察到间隙被这样一步步“挖小”（$8\times10^{-8}\to7\times10^{-10}$，第 175 页）。

**最小复现（`models/lever.xml`）：** 一根杆从地板下面的铰链伸上来，顶端一个球，球在它的圆弧的**最高点**，离地 0.5 mm。往哪边转球都会沿弧线压进地面，而在最高点处球的速度是水平的，接触行 $r = 2d\,n\cdot(a\times(x - o)) = 0$。

`curvature_demo lever w=1 h=0.1`（惯性弱，极小点由障碍决定，间隙约 0.19 mm）：

| 迭代 | GN（scale 0）的间隙 | CCD $\alpha_{\max}$ | 上界（scale 1）的间隙 | CCD $\alpha_{\max}$ |
|---|---|---|---|---|
| 1 | 0.500 mm | 0.58 | 0.500 mm | 0.70 |
| 2 | **0.128 mm** | 1 | 0.268 mm | 1 |
| 3 | 0.152 mm | 1 | 0.210 mm | 1 |
| 4 | 0.171 mm | 1 | 0.199 mm | 1 |
| … | 爬回 0.191 mm | | 单调落到 0.191 mm | |

GN 的第一步被 CCD 截断，把间隙挖到极小点以下，再慢慢爬回来；加了上界的模型知道“转动会压进地面”，间隙单调地落到极小点，从不低于它（测试 `Lever.GaussNewtonOvershootsTheGapAndTheMajoriserDoesNot`）。两者收敛到同一个点：上界只改 Hessian，不改梯度。

---

## 2. 推导：丢掉的项有多大

沿方向 $\delta$ 走 $s$，记 $x(s)$ 为平方距离，$\Delta(s)$ 为顶点到平面最近点的向量（$x = \Delta\cdot\Delta$）。障碍能量的二阶导数是

$$
\frac{d^2}{ds^2}\kappa b(x) = \kappa b''(x)\,(x')^2 + \kappa b'(x)\,x'', \qquad x'' = 2\lvert\Delta'\rvert^2 + 2\,\Delta\cdot\Delta''
$$

第一项就是 GN 行 $\kappa b''(r\cdot\delta)^2$。被丢掉的 $\kappa b' x''$ 分两部分：

1. **$2\kappa b'\lvert\Delta'\rvert^2$：** $b' < 0$，这一项 $\le 0$。丢掉它只会让模型比真实能量“更弯”，不危险（只是慢：思考题 1）。
2. **$2\kappa b'\,\Delta\cdot\Delta''$：** 轨迹曲率项。$\Delta''$ 是顶点沿 $q$ 空间直线运动时的加速度。笛卡尔 IPC 和 ABD 里顶点对未知量是线性的，$\Delta'' = 0$；关节坐标里它不为零，而且不定号。当弧线朝障碍弯（$\Delta\cdot\Delta'' < 0$）时它是正的，丢掉就低估了能量的增长。

对平面，$\Delta = d\,n$，$\Delta'' = (n\cdot X'')\,n$，$X$ 是顶点。

---

## 3. 推导：给它一个上界

**第 1 步：Cauchy-Schwarz。** 由 $\lvert\Delta\rvert = d$：

$$
\lvert 2\kappa b'\,\Delta\cdot\Delta''\rvert \le 2\kappa\lvert b'\rvert\,d\,\lvert\Delta''\rvert = f\,\lvert\Delta''\rvert, \qquad f = 2\kappa\lvert b'(x)\rvert\,d
$$

$f$ 恰好是法向力的大小（阶段 4 第 3 节）。

**第 2 步：界定 $\lvert X''\rvert$。** 顶点的二阶导数是

$$
X'' = \sum_{j,k}\delta_j\delta_k\,\frac{\partial^2X}{\partial q_j\,\partial q_k}
$$

把链上的转动自由度按关节分组（铰链一组 1 个，球关节和自由关节的转动一组 3 个），记第 $j$ 组的增量大小 $w_j = \lvert\delta_{\text{组 }j}\rvert$。两个转动的二阶导数形如 $a_j\times\big(a_k\times(X - o_k)\big)$（$k$ 是较深的那个），大小不超过顶点到较深那个锚点的距离。用静态力臂 $R_{\max}(j,k) = R_{\max(j,k)}$（阶段 5 的 `chain_joints`）界它：

$$
\lvert X''\rvert \le \sum_{j,k} w_jw_k\,R_{\max}(j,k)
$$

**第 3 步：摊到对角上。** 用 $w_jw_k\le\tfrac12(w_j^2 + w_k^2)$ 和 $R_{\max}$ 的对称性，交叉项变成平方项：

$$
\lvert X''\rvert \le \sum_j C_j\,w_j^2, \qquad C_j = \sum_k R_{\max}(j,k)
$$

$C_j$ 只依赖模型的尺寸。对最深的关节 $j$，$C_j = n\,R_j$（$n$ 是链上转动组的个数）；对根，$C_{\text{root}} = \sum_k R_k$。

**合起来：** 对任意 $\delta$，

$$
\delta^\top\big(\text{真实 Hessian}\big)\,\delta \le \delta^\top\Big(\kappa b''\,r^\top r + f\sum_{j\in\text{链}}C_j\,P_j\Big)\,\delta
$$

$P_j$ 是第 $j$ 组自由度上的单位对角块。也就是说：**在 GN 行之外，往顶点所在链上每个转动组的对角线加 $f\,C_j$**，得到的模型在每个方向上都不比真实 Hessian 弯得少，而且是半正定的（测试 `Majoriser.BoundsTheTrueHessianAlongAnyDirection` 用接触梯度的中心差分验证这个不等式）。

“上界”指的是当前迭代点处的二次型，不是整个能量的全局上界。CCD 保证每个迭代点都不相交，线搜索保证能量下降；上界只决定 Newton 模型有多“硬”。

**物理含义：** $f\,C_j\,w_j^2$ 约等于“接触力 × 力臂 × 角速度²”，即沿弧线运动的向心效应在接触方向上最多能造成多大的能量变化。

---

## 4. 缩放 curv_scale

上界是按最坏情况的力臂算的，常比真实曲率大很多。既然它不影响安全性（CCD 管不穿透，线搜索管能量下降），就可以乘一个小于 1 的系数换更快的收敛。书 §8.5 在 10 个场景上扫了 1.0 / 0.3 / 0.1：驱动的链条和机构（自行车链、手、齿轮、履带）Newton 更少，自由堆叠在噪声内，Go2 四足堆和锁链略差。结论写进了书上的调参表：驱动机构 0.1–0.3，其余 1.0；0 不是选项（链条会卡死）。

`StepParams::curv_scale` 默认 1.0（可证明的上界）。

---

## 5. 结果

- **顶杆（第 1 节）：** GN 第一步被截断、挖过极小点；上界单调。Newton 次数 14 对 15：在单个铰链上两者的收敛速度都受另一件事限制（思考题 1）。
- **顶杆以 20 rad/s 撞向地面（`curvature_demo lever w=20`）：** 惯性强，球被压到 1 μm；GN 18 次迭代（CCD 每次截到 15%、9%、6%、4%、3%……），上界 13 次；之后几步两者相同。任何 scale 都不穿透、都收敛到同一个点（测试 `Lever.EveryScaleReachesTheSamePoint`）。
- **甩杆、落地盒子：** 上界几乎不起作用（$f\,C_j$ 远小于 $M/h^2$）。它真正发挥作用的地方是书上的多节链条，本仓库到阶段 10 的场景再比。

---

## 6. 代码对照

| 顺序 | 函数 | 文件 | 笔记 | 测试 |
|---|---|---|---|---|
| 1 | `chain_joints`（阶段 5 的静态力臂，现在也给曲率用） | `src/chain.cc` | 第 3 节第 2 步 | `Chain.*` |
| 2 | `contact_curvature` | `src/contact.cc` | 第 3 节 | `Majoriser.*` |
| 3 | `step` 第 1 步里加上它 | `src/step.cc` | — | `Lever.*` |

演示程序 `curvature_demo lever w=1 h=0.1`、`curvature_demo lever w=20`、`curvature_demo flail`，每次迭代打印间隙、CCD 上限和线搜索步长，scale 0 和 1 各跑一遍。

---

## 7. 思考题

1. 顶杆的两种 scale 都要十几次迭代，每次 $\lvert p\rvert/h$ 大约减半。算一算第 2 节的第 1 部分 $2\kappa b'\lvert\Delta'\rvert^2$ 在收敛点附近有多大（和 GN 行 $\kappa b''(x')^2$ 比）。丢掉一个负的项，模型比真实能量硬多少？这解释了“减半”吗？书 §8.6 的秩 1 曲率行是怎样补这一项的？
2. 上界加在对角线上，丢掉了自由度之间的耦合（书第 180 页：问题出在这里，不在上界的大小）。构造一个两节链的例子，说明只靠对角项为什么会让 Newton 变慢。
3. $f = 2\kappa\lvert b'\rvert d$ 在接触刚激活时（$d\approx\hat d$）几乎是 0，上界也几乎是 0。这是合理的吗？什么时候上界最重要？
4. 书上说“任何缩放下，安全性都不受影响”。安全性由哪两件事保证？scale 取 0 时链条为什么会卡死而不是穿透？
5. 静态力臂 $R_{\max}(j,k)$ 对深链很松。把它换成当前姿态下的真实距离加刚体半径，上界还成立吗？（提示：上界是当前点处的二次型，不是整段的。）书上试过，中位数只小到 0.956 倍，为什么？
6. 滑动关节的“转动 × 滑动”混合二阶导数不在这个上界里。写出它的形式和大小，想想该加到哪里。
