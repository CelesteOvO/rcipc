# 阶段 12：驱动、关节阻尼和 armature

**目标：** 有了这三样，才能跑被驱动的机构（齿轮、夹爪、手、机械臂）和真实的机器人模型。三者都只作用在单个自由度上，所以和阶段 10 的关节限位一样：能量是这个自由度的一元凸函数，梯度和 Hessian 只落在对角上，进树块 $H_k$。

代码：`include/rcipc/drive.h`、`src/drive.cc`（CPU），`src/gpu/kin.cu` 的 `k_drives`（GPU）；模型 `models/drives.xml`；测试 `tests/test_drives.cc`、`tests/test_gpu.cc` 的 `Step.DrivesMatchCpu`。

---

## 1. 阻尼和 armature

一个铰链或滑动自由度，阻尼系数 $b$、armature $a$（MJCF 的 `damping`、`armature`）。隐式欧拉里速度取这一步的 $(q - q_0)/h$：

- **阻尼**的力 $-b\dot q$ 是瑞利耗散函数 $\tfrac12 b\dot q^2$ 的导数，换成 $q$：$E_b = \dfrac{b}{2h}(q - q_0)^2$，它的梯度 $\dfrac{b}{h}(q - q_0) = b\,\dot q$。
- **armature** 是加在质量矩阵对角上的转子惯量，和惯性项同一个形状：$E_a = \dfrac{a}{2h^2}(q - q_0 - h v_0)^2$（没有重力那一项，转子的质心不动）。

两项加起来还是一个二次函数，写成

$$
E = \tfrac12 K (q - r)^2 + \text{常数},\qquad K = \frac bh + \frac a{h^2},\qquad r = \frac{(b/h)\,q_0 + (a/h^2)(q_0 + h v_0)}{K}
$$

$K$、$r$ 在步首算一次（`drive_terms`）。

**只用在铰链和滑动上。** 球关节、自由关节的阻尼要在切空间里做（$q \ominus q_0$ 对增量的导数不是单位阵），这里不支持；模型里有的话记进 `Model::ignored`，`scene_demo` 开头会打印。MuJoCo 模型里自由关节的阻尼几乎总是 0。

---

## 2. 驱动

只支持 MuJoCo 的 general 模型里的一部分：传动是关节（铰链或滑动），没有激活动力学，增益固定，偏置是 none 或 affine。`position`、`velocity`、`motor` 这几种快捷写法都在里面。其余的读模型时直接报错（悄悄忽略一个驱动，整个场景就错了）。

驱动的“长度”$L = s q$（$s$ 是传动比 `gear`），驱动力和关节力矩是

$$
f = g\,u + b_0 + b_1 L + b_2 \dot L,\qquad \tau = s f
$$

$u$ 是控制量。position 驱动是 $g = k_p$，$b = (0, -k_p, -k_v)$；velocity 是 $g = k_v$，$b = (0, 0, -k_v)$；motor 是 $g = 1$，没有偏置。

**隐式化。** 把 $\dot L$ 换成 $s (q - q_0)/h$，$f$ 就是 $q$ 的一次函数：

$$
f = A + C q,\qquad A = g u + b_0 - \frac{b_2\, s\, q_0}{h},\qquad C = s\,w,\quad w = b_1 + \frac{b_2}{h}
$$

能量取 $E = -\int \tau\,dq$，梯度就是 $-\tau$，Hessian 是 $-s C = -s^2 w$。$b_1, b_2 \le 0$（正常的刚度和阻尼）时 $w \le 0$，$E$ 是凸的。偏置让驱动变成负刚度或负阻尼的模型，读的时候报错。

所以 position 驱动就是一个隐式的 PD 控制：$k_p$ 那一项是弹簧（位置隐式），$k_v$ 那一项和关节阻尼一样。MuJoCo 自己的 `implicitfast` 积分器只把速度那一项隐式处理，所以结果和它差一个 $O(h)$。

**力的上下限**（`forcelimited`、`forcerange`）：$\tau = s\,\mathrm{clamp}(f, lo, hi)$。令 $y = A + Cq$、$c = \mathrm{clamp}(y)$，

$$
E = -\frac{1}{w}\Big(\tfrac12 c^2 + c\,(y - c)\Big)
$$

在范围里 $c = y$，$E$ 是二次的；在范围外 $c$ 是常数，$E$ 是一次的。这正是 Huber 函数的形状：仍然凸、一阶导数连续，Hessian 在范围外是 0。验证：$\dfrac{d}{dy}\big(\tfrac12c^2 + c(y-c)\big) = \mathrm{clamp}(y)$，所以 $E' = -\frac1w \cdot \mathrm{clamp}(y)\cdot C = -s\,\mathrm{clamp}(y) = -\tau$。

$C = 0$ 时（motor，或者偏置里没有 $L$、$\dot L$ 的项）力是常数，$E = -s\,\mathrm{clamp}(A)\,q$。

控制量先按 `ctrlrange` 夹（`ctrllimited` 时）。关节上的 `actuatorfrcrange` 不支持，记进 `ignored`。

---

## 3. 控制量的时间表

`StepParams::ctrl` 是这一步的控制量（长度 nu）。场景文件自己带时间表：MJCF 的关键帧 `<key time="..." ctrl="..."/>`，`ctrl_at(model, t)` 在相邻两帧之间线性插值，第一帧之前、最后一帧之后取端点的值。`scene_demo` 每一步用这一步结束时刻的值（隐式欧拉在步末求力）。

---

## 4. 验证（`models/drives.xml`、`tests/test_drives.cc`）

模型里是五个互不相干的刚体，各自检验一项；没有碰撞几何。

| 测试 | 检验什么 | 和什么比 |
|---|---|---|
| `Drive.Derivatives` | 能量、梯度、对角 Hessian | 有限差分，含力限制的两侧 |
| `Drive.DampingArmature` | 只有阻尼和 armature 的滑块，100 步 | 每步 $v \leftarrow v\,(m+a)/(m+a+bh)$，1e-10 |
| `Drive.PositionServo` | PD 伺服带力限制，800 步 | 隐式欧拉的闭式递推（先按不夹解，越界再按常力解），逐步 1e-10；最后停在目标上 |
| `Drive.ForceLimit` | motor 控制量 10、力夹在 ±2 N | 匀加速 $a = 2/m$ |
| `Drive.VelocityServo` | 速度伺服的稳态 | 下面这个方程的解，1e-9 |
| `Drive.GravityBalance` | 摆在重力下被 position 驱动托住 | 力矩平衡 $k_p(u - \theta) = mgl\sin\theta$，1e-6 |
| `Step.DrivesMatchCpu`（GPU） | 300 步 | CPU 版，逐步 1e-9、Newton 次数相同 |

**速度伺服停不在目标上。** 第一版测试要求稳态角速度等于目标 $u$，结果差了 $4\times10^{-4}$。原因不在驱动，在阶段 3 的惯性项：它把每个质点按直线外推，$\tilde x = 2x_0 - x_{-1}$。匀速转动时，圆周上的点按直线外推，落在圆外、而且转角比真实的少一点。以 $\theta = h\omega$ 为每步的转角，算一下惯性项在稳态的梯度（只看垂直于转轴的平面，二阶矩 $S_{xx} = S_{yy} = I_{zz}/2$）：

$$
\frac{\partial E_{\text{inertia}}}{\partial\phi} = \frac{2I}{h^2}(1 - \cos\theta)\sin\theta \approx \frac{I\,\theta^3}{h^2}
$$

它像一个很小的阻力，所以稳态满足 $\frac{2I}{h^2}(1-\cos h\omega)\sin h\omega = k_v(u - \omega)$。这里 $h\omega = 0.02$，角速度比目标小约 $(h\omega)^2/2$ 倍，正好是看到的差。这是阶段 3 那种惯性项的数值耗散，转得越快（每步转角越大）越明显；齿轮 100°/s、$h$ = 0.01 s 时每步 0.017 rad，阻力约为 $5\times10^{-6}\,I/h^2$。

---

## 5. 思考题

1. 阻尼的能量为什么是 $\frac{b}{2h}(q-q_0)^2$，而不是 $\frac b2 \dot q^2$？两者差在哪里，哪一个的梯度才是“力乘 $h$”的那个量纲？
2. position 驱动在 MuJoCo 的 `implicitfast` 里，$k_p$ 那一项是显式的。$k_p$ 很大、$h$ 不变时，两种处理各会怎样？
3. 力限制的能量是 Huber 形状，Hessian 在范围外是 0。如果一个自由度上只有一个被夹住的驱动、没有别的项（比如惯性），Newton 法会出什么问题？实际场景里为什么不会？
4. 第 4 节的阻力 $\approx I\theta^3/h^2$。固定转速 $\omega$，把 $h$ 减半，阻力怎么变？这说明惯性项的这个误差是几阶的？
5. 球关节的阻尼要在切空间里做。写出 $E = \frac{b}{2h}\lVert\log(R_0^{\mathsf T}R)\rVert^2$ 对增量的梯度，和铰链的情况比，多了什么？
