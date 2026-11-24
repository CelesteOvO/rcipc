# 22. 原始-对偶障碍、ACCD 的分离平面界、两阶段曲率：默认的求解器（2026-11-24）

阶段 14 以后排查“撞上 Newton 上限的步”（分支 `cap-hits-primal-dual` 上的 `notes/19_cap_hits.md`、`notes/20_long_chain_newton.md`），
试了二十多个实验开关。留下来、变成默认的是三样：

| | 修什么 | 第几节 |
|---|---|---|
| 原始-对偶障碍 | 被压扁的间隙：原始 Newton 每次迭代最多把间隙翻一倍 | 1、2 |
| ACCD 的分离平面界 | 贴着表面滑动：ACCD 每步只敢走一点点，1000 步走不完一段 | 3 |
| 两阶段曲率 | 深链上阶段 6 的曲率上界太硬：整步、但线性收敛 | 4 |

它们都没有开关（`StepParams::primal_dual` 只留着做对比，`scene_demo pd=0`）。第 5 节是同一段时间里修在默认路径上的东西，
第 6 节是量到的效果和还没解决的，第 7 节列出没有进 main 的实验（都在分支上）。

## 1. 问题：被压扁的间隙每次迭代只能翻一倍

一个接触，间隙 $d$ 远小于 $\hat d$ 时障碍是 $b \approx -2\hat d^4\ln(d/\hat d)$。法向载荷 $F_{\rm ext}$ 压着它，一维的能量
$E(d) = -2\kappa\hat d^4\ln d + F_{\rm ext} d$，平衡间隙 $d^* = 2\kappa\hat d^4/F_{\rm ext}$。Newton 一步：

$$
E' = -\frac{2\kappa\hat d^4}{d} + F_{\rm ext},\quad E'' = \frac{2\kappa\hat d^4}{d^2},\qquad
d_+ = d - \frac{E'}{E''} = 2d - \frac{d^2}{d^*}.
$$

$d \ll d^*$ 时 $d_+ \approx 2d$：间隙从 $10^{-8}$ m 回到 $10^{-4}$ m 要十几次迭代。冲击、转动把间隙压扁以后（阶段 6 讲的那种），
这一步就撞上限，或者在上限之前停在一个被压扁的点上，下一步再被压一次（笔记 19 第 10、14 节）。

## 2. 原始-对偶障碍

### 2.1 对偶变量

把每个接触的法向力 $\lambda$ 当作这一步里的独立变量，互补条件写成 $\lambda\,(\kappa/F(d)) = \kappa$（$F(d)$ 是原始障碍的法向力，
$F = -2\kappa\,m\,b'(d^2)\,d$，$m$ 是 mollifier）。对 $(q, \lambda)$ 一起做 Newton，消掉 $\Delta\lambda$ 以后：

- 方程的右端就是原始的梯度（解不变：收敛的点上 $\lambda = F(d)$）；
- 接触的 Hessian（Gauss-Newton 行、阶段 6 的上界、旋量块、链上项）整个乘上 $\lambda / F(d)$。

一维的例子里 Hessian 变成 $\lambda/d$。第一次迭代 $\lambda$ 还不知道，取 $F(d)$，就是原始的一步；之后 $\lambda \approx F_{\rm ext}$：

$$
\Delta d = \frac{F(d) - F_{\rm ext}}{\lambda / d} \approx \frac{2\kappa\hat d^4/d - F_{\rm ext}}{F_{\rm ext}/d} = d^* - d ,
$$

一步回到平衡间隙（沿割线走，而不是沿切线）。测试 `PrimalDual.SqueezedGapSameSolutionFewerIterations`：盒子 4 个角压在
$0.1\,d^*$，两种方法走到同一个点（差 $< 10^{-9}$ m），原始-对偶的迭代次数少。

对偶的更新（互补条件的线性化），$\Delta d = r\cdot p / (2d)$ 取整步的 Newton 方向 $p$（$r$ 是 `contact_row`）：

$$
\lambda_+ = \max\big(\lambda + (F - \lambda) + \lambda\,\tfrac{F'}{F}\,\Delta d,\ 0.01\,\lambda\big),
$$

下限 $0.01\lambda$ 保正（fraction-to-boundary）。这一步里第一次出现的接触从 $\lambda = F$ 开始；$\kappa$ 翻倍、减半时表里的
$\lambda$ 按同样的比例缩放，否则 $\lambda/F$ 会跟着 $\kappa$ 跳。

### 2.2 截断区：按局部幂次混合

截断障碍在 $d \to \hat d$ 时 $F \to 0$、$F'/F \to \infty$，比值 $\lambda/F$ 乱跳。所以组装用

$$
\lambda_{\rm eff} = w\,\lambda + (1 - w)\,F,\qquad w = \min(1, 1/g),\qquad g = -d\,\frac{F'}{F} = -1 - \frac{2x\,b''(x)}{b'(x)},\ x = d^2 .
$$

### 2.3 局部幂次 $g$ 的表

纯对数障碍 $F \propto 1/d$ 时 $g = 1$。测试 `PrimalDual.LocalPowerMatchesTheTable`：

| $d/\hat d$ | 0.01 | 0.1 | 0.3 | 0.5 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| $g$ | 0.997 | 0.905 | 0.954 | 1.745 | 7.45 | 17.4 | 37.4 |

$g = 1$ 在 $0.3226\,\hat d$：比它近用完整的对偶（$w = 1$），比它远逐渐退回原始（$d \to \hat d$ 时 $g \approx 2d/(\hat d - d) - 1$）。

### 2.4 实现

- CPU（`step.cc`）：$\lambda$ 按接触的图元（类型、前 `nstencil` 个模板顶点、平面）记在一张表里，只在这一步之内有效。
  组装时在接触的副本上把 `Contact::kw` 乘上 $\lambda_{\rm eff}/F$；梯度、能量、线搜索仍用原始的接触。
- GPU（`src/gpu/contact.cu` 的 `launch_pd_*`、`device_model.h` 的 `DPd`）：$\lambda$ 记在显存的开放寻址哈希表里，在 CUDA graph 里更新。
  组装前 `launch_pd_scale` 按键查表写每个接触的系数，PCG 之后 `launch_pd_prepare` 记下 $F$、$F'/F$、$\Delta d$，
  线搜索之后 `launch_pd_update` 写回。GPU 版不和完整的接触 Hessian 一起用（`full_contact_hessian` 要配 `primal_dual = false`）。
- 2026-11-24 修了 CPU 版的键：以前四个模板顶点全进键，地面接触后三个槽位是没有初始化的内存，同一个接触在相邻两次迭代里
  可能对不上，查不到上一次的 $\lambda$（换一种编译出来的栈布局就会出现）。现在 `Contact` 的 `verts`、`w`、`plane` 有默认值，
  键只取前 `nstencil` 个顶点（和 GPU 的 `pd_key_of` 一样）。

## 3. ACCD 的分离平面界

### 3.1 问题

ACCD（书 §9.2）每步走 $\Delta t = c\,(d - \xi)/\ell$，$\ell$ 是两侧顶点相对位移之和。贴着表面滑动时 $\ell$ 几乎全是切向位移，
距离却不变，于是要走 $\ell/(c(d - \xi))$ 步：搅拌机第 70 步要 2700 步，上限是 1000，返回的位置保守但很短。

### 3.2 界

对任何单位向量 $n$ 和两侧图元上的点 $x$、$y$，$\lvert x - y\rvert \ge n\cdot(x - y)$；图元是顶点的凸包，所以

$$
d(t_k + \tau) + r \;\ge\; \min_{j\in A,\ i\in B}\ n(\tau)\cdot\big(P_j(\tau) - P_i(\tau)\big),\qquad P(\tau) = P + \tau D .
$$

取 $n(\tau) = (n + \tau\nu)/\lvert n + \tau\nu\rvert$（$\nu \perp n$，平面跟着图元转），用 $\lvert n + \tau\nu\rvert \le 1 + \tfrac12\tau^2\lvert\nu\rvert^2$，
每对顶点给一个二次式

$$
\underbrace{n\cdot m - L}_{A} + \tau\,\underbrace{(n\cdot s + \nu\cdot m)}_{B} + \tau^2\,\underbrace{\big(\nu\cdot s - \tfrac12 L\lvert\nu\rvert^2\big)}_{C} \;\ge\; 0
\quad\Longrightarrow\quad d + r \ge L ,
$$

$m = P_j - P_i$，$s = D_j - D_i$，走到所有二次式的第一个正根。$\nu$ 取什么界都成立：取成让每对顶点的一次项相等
（$\nu\cdot(P_w - P_u) = -n\cdot(D_w - D_u)$，每条边一个方程，最小二乘），$B$ 就是接触点的法向相对速度，转着滑动时是 0。
$\nu = 0$ 是固定的平面，平移滑动时整段安全。

分离方向试两个：最近点的方向，和图元自己的法向（三角形的法向、两条边的叉积）。间隙只有 $10^{-9}$ m、没有半径时，
最近点的方向算不准（舍入 $10^{-17}$ m 就让它偏 $10^{-8}$ rad），图元的法向和间隙无关。两个方向各算固定和转动的平面，
再加上原来的 $\ell$，几个界都成立，每步取走得最远的。

每步要保住的下界 $L = r + \xi + \max\big((1-c)(d_0-\xi),\ (1-c)(d-\xi)\big)$，不低于停下的门槛，离门槛 10% 以内就停，
所以一次最多闭合 $c = 90\%$。（原来的规则一步跨过门槛时会闭合 99%。）

### 3.3 例子和测试

- `Accd.RotatingPlaneFollowsABladeSweepingABall`：球贴着转动的叶片（间隙 $1.8\times10^{-7}$ m），只用 $\ell$ 时 1000 步走到 0.087，
  这里 2 步走到 0.449，停下时正好闭合 90%。
- `CurvedCcd.SlidingAtATinyGapPassesTheWholeStep`：两个盒子间隙 1 μm，上面的沿接触面滑 5 mm，整步放行（只用 $\ell$ 时停在 0.18）。
- `Accd.PlaneBoundsAreSafeOnRandomRigidMotions`：3000 个随机刚体运动，返回的 $t$ 都安全、都闭合不超过 90%。
- GPU 版同一个界（`src/gpu/ccd.cu`），`Ccd.MatchesCpu` 比 $\alpha_{\max}$。

ACCD 的总步数：搅拌机 250 步 980 万 → 30 万，布娃娃堆 24 个 $\mu$ 5450 万 → 20 万（笔记 19 第 13 节）。
有了它，GPU 上 ACCD 很少走到交给 warp 的“尾巴”（`notes/11_gpu.md` 第 14 节），尾巴的代码留着，测试只确认交不交结果逐位相同。

## 4. 两阶段曲率：秩 1 曲率行

被 Gauss-Newton 丢掉、阶段 6 用对角上界盖住的轨迹曲率项是 $S = \sum_v f_v\cdot\partial^2 X_v/\partial\delta^2$。
上界只用转角的大小，丢了方向，深链上大多数方向都太硬。秩 1 曲率行只取它沿这个接触自己那一行 $r$ 的方向：

$$
r^\top S r = \kappa\,m\,b'(x)\,2d\;n\cdot\Delta''[r,r],\qquad D \mathrel{+}= \frac{\max(0,\ r^\top S r)}{\lvert r\rvert^4}
$$

（`contact_curvature_row_weight`；$X''[v,v]$ 沿链走一遍，`point_acceleration_along`）。行数不变，PCG、预条件器、GPU 的行存法都不用动。

什么时候用（书 §8.6，旧实现的 `CURVPHASE`）：

- CCD 还在截断步长时用上界：截断说明某一对在逼近，间隙被压的时候要各方向都保守。
- 这一步里某次迭代走了整步（CCD 没截断、线搜索也取了整步）以后，Gauss-Newton 行的接触改用秩 1 行，用到这一步结束。
  撞上限的步正是“整步、线性收敛”，问题全在上界太硬（笔记 19 第 1 节）。

`FullStepTrack`（`step.h`）记这件事，CPU、GPU 共用；GPU 每次迭代把开关写进 `DIter::crow`。完整的接触 Hessian 没有这种行。

单开它时 12 节链条 Newton 20.4 → 8.1 次/步、撞上限 26 → 3 步，但最小间隙从 $9.6\times10^{-7}$ 掉到 $3.5\times10^{-8}$ m：
小间隙的接触在整步阶段也需要二阶项。原始-对偶管住了这一点（第 6 节）。

## 5. 同一段时间修在默认路径上的

这些不是开关，阶段 14 的默认参数下结果也会变：

- **CCD 的最小分离** `StepParams::min_sep = 1e-12`：CCD 把 $d - \text{sep}$ 当距离用，起点已经 $\le$ sep 的对返回 0（卡住），
  不再放行。原来 $d \le 0$ 时返回 1：浮点把 $10^{-16}$ m 的间隙算成负数时整步就穿过去了。限位的 CCD 同样。
- **线搜索的第二道网**：图元重叠（$d \le 0$）时障碍的能量是 $+\infty$（`barrier` 对 $x \le 0$ 返回无穷，原来对 $d$ 是偶函数）；
  减半到底还是无穷，这次迭代一步也不走。
- **胶囊的端点边**：每个胶囊再加两条退化的边（端点），两根平行的胶囊之间边–边的 mollifier 趋于 0 时，“端点–边”的对接上障碍。
- **网格的重合顶点合并**（MuJoCo 按法向、纹理坐标把一个位置拆成几个顶点，障碍会被算几遍）。
- **滑动关节限位的激活距离**用接触的 $\hat d$（米），铰链用 `limit_dhat`（弧度）（`LimitBarrier`）。
- **焊在世界上的刚体**不和平面比。
- **没有顶点的模型**的收敛判据用刚体质量的均方根位移（原来是关节空间的 $\max\lvert p\rvert$，单位不对，长链上空转到上限）。
- **CCD 的紧界**：三阶导数的界逐层递推，系数从 7、5 变成 3、3（`third_derivative_bound`）；没有顶点的刚体不缩短段长。
- **惯性的 Hessian 和曲率**按子树求和（组合刚体），$O(n)$；GPU 的完整接触 Hessian 一个 warp 一个接触。

## 6. 效果和还没解决的

### 6.1 和阶段 14 的默认配置比（GPU，2026-11-20 的对比快照，提交 1a6db23）

每个场景跑 3 次取最快。“默认”是阶段 14 的默认参数，“现在”是这一篇的三样一起开；Newton 是每步平均，撞上限是步数，
间隙是步末最小间隙（m）。

| 场景 | ms/步 默认 → 现在 | Newton | 撞上限 | 最小间隙 |
|---|---|---|---|---|
| 圆锥堆（Rigid-IPC） | 28.7 → 7.6 | 19.5 → 13.8 | 38 → 20 | 3.1e-7 → 6.2e-7 |
| 单摆（Rigid-IPC） | 12.6 → 4.3 | 13.4 → 6.9 | 41 → 7 | 4.4e-8 → 1.5e-8 |
| 门板（libuipc） | 11.5 → 3.7 | 5.9 → 3.8 | 0 → 0 | 2.5e-6 → 2.5e-6 |
| 8 个网格布娃娃 | 16.0 → 6.7 | 22.9 → 7.9 | 34 → 4 | 9.9e-6 → 2e-5 |
| 30 个网格布娃娃 | 52.3 → 33.9 | 46.5 → 24.1 | 269 → 34 | 9.9e-6 → 1.5e-5 |
| 60 个网格布娃娃 | 77.3 → 69.0 | 46.5 → 33.0 | 274 → 113 | 9.9e-6 → 2e-5 |
| 碗里的关节对（M-ABD 论文） | 431 → 383 | 43.1 → 29.6 | 214 → 108 | 4.9e-9 → 2.5e-7 |
| 布料下垂（libuipc） | 53.0 → 76.2 | 48.1 → 17.0 | 384 → 42 | 3.4e-7 → **1.3e-11** |
| 摆锤（Rigid-IPC） | 97.2 → 120.0 | 39.1 → 37.4 | 99 → 99 | 2.6e-8 → 8.8e-8 |
| 搅拌机（Rigid-IPC） | 38.9 → 37.4 | 16.7 → 17.8 | 34 → 34 | 1.7e-7 → 2.3e-6 |

其余场景（齿轮、悬挂树、投石机、纸牌屋、螺栓、M-ABD 的链、Franka 抓取）两边相差在 10% 以内。

### 6.2 还没解决的

- 布料下垂 Newton 少了 2/3，每步却慢 44%，最小间隙掉到 $10^{-11}$ m；摆锤每步慢 23%、撞上限不变。没有逐步查。
- 大步长的链条和布娃娃堆上仍有间隙塌缩：h × 0.96–1.04 各 5 次，12 节链条 2、5、20 ms 和布娃娃堆 20 ms 一共 20 次运行里有 6 次
  （提交 1a6db23 的记录；分支上的笔记 19 第 23 节）。后来的排查：塌缩发生在单个撞上限的步里，一对链上的接触被压扁的同时
  $\kappa$ 在步内翻倍。
- 撞上限的步数在搅拌机、摆锤上不变。搅拌机剩下的主要是 PCG 的精度（`pcg_tol` 收紧到 $10^{-8}$ 时撞上限少一个数量级，
  但别的场景变慢）；摆锤没有查。

### 6.3 这一次提交怎么验证的

- 三样都做成默认、删掉其余开关以后，和 49c9931（参数 `accdp=2 pd=5 cphase=1`）跑同一组场景比轨迹：CPU 上逐位相同
  （每步的 $q$ 按 `%.17g` 比），GPU 上 Newton 次数相同、差异和参照自己两次运行之间的差异同一量级（GPU 的原子加让求和顺序不固定）。
  唯一的例外是上面 2.4 节的键：修之前 CPU 版在布娃娃上从第 138 步起和参照分开（参照碰巧查到了 $\lambda$），修好以后逐位相同。
- 测试：`tests/test_primal_dual.cc`（局部幂次、压扁的间隙、落地、截断区），`tests/test_ccd.cc` 的 `Accd.*`、最小分离，
  `tests/test_curvature.cc` 的 `RankOneRow.*`，`tests/test_gpu.cc` 的 `Step.DefaultSolverMatchesCpu`（布娃娃、叠盒子、12 节链条，
  GPU 和 CPU 每步的 Newton 次数一样、轨迹一致）、`PrimalDual.SqueezedGapMatchesCpu`。

## 7. 没有进 main 的

都在提交 49c9931（分支 `cap-hits-primal-dual` 之后的那条线，笔记 19、20 和对比快照也在那里）及其后的分支上，要复现那些结果用它们：

- 求解器的实验开关：BDF2 和混合积分器、热启动、信赖域（`tr=`、`trind=`）、特征值过滤、按间隙或按物理判据给旋量块
  （`fhg=`、`csel=`）、`chain_fallback`、惯量度量的投影、按岛取步长（`ialpha=`）、$\kappa$ 的其它调法（按步、锁存、按目标间隙取初值）、
  线搜索往外扩、按段的加速度界、按方向的矢高、直线 CCD、可调的 ACCD 上限；
- pd 的其它档位（沿实际步长更新、跨步保留 $\lambda$、只在 $0.1\hat d$ 以内用对偶），两阶段曲率的逐次切换（`cphase=2`），
  只用固定平面的分离平面界（`accdp=1`）；
- `scene_demo` 的诊断模式（`inside=`、`scan=`、`start=` 等）和各家方法的对比脚本、结果。
