#include "rcipc/quat.h"

#include <cmath>

// 四元数的 exp 和 log。推导见 notes/02_kinematics.md 第 2 节，测试见 tests/test_quat.cc。
namespace rcipc {

Eigen::Quaterniond quat_exp(const Eigen::Vector3d& w) {
  // 指数映射：旋转向量 w -> 单位四元数（笔记第 2.1 节）。
  //
  // w 的方向 n = w / |w| 是转轴，长度 t = |w| 是转角。绕 n 转 t 的四元数是
  //
  //     exp(w) = ( cos(t/2),  sin(t/2) * n )  =  ( cos(t/2),  [sin(t/2) / t] * w )
  //
  // 后一种写法不用先算 n，只要算一个系数 k = sin(t/2) / t，向量部分就是 k * w。
  //
  // 检查：exp(0) = (1, 0, 0, 0)；exp(pi * (0, 0, 1)) = (0, 0, 0, 1)，绕 z 转半圈。

  // 第 1 步：转角 t。
  const double t = w.norm();

  // 第 2 步：系数 k = sin(t/2) / t。
  //   t = 0 时这是 0/0，会得到 NaN。它的极限是 1/2（sin(t/2) ≈ t/2）。
  //   t 很小时用 Taylor 展开 k = 1/2 - t^2/48，误差是 t^4 / 3840 量级，t < 1e-4 时远小于舍入误差。
  //   注意写 0.5 而不是 1/2：两个整数相除是整数除法，1/2 等于 0。
  double k;  // 声明在 if 外面，否则出了 if 的花括号就没有 k 了
  if (t < 1e-4) {
    k = 0.5 - t * t / 48;
  } else {
    k = std::sin(0.5 * t) / t;
  }

  // 第 3 步：拼成四元数。构造函数的参数顺序是 (w, x, y, z)，实部在前。
  return Eigen::Quaterniond(std::cos(0.5 * t), k * w.x(), k * w.y(), k * w.z());
}

Eigen::Vector3d quat_log(const Eigen::Quaterniond& q) {
  // 对数映射：exp 的逆（笔记第 2.1 节）。
  //
  // 把 q 写成 (c, v)：c = q.w() 是实部，v = q.vec() 是向量部分。如果 q = exp(t n)，
  // 那么 c = cos(t/2)、v = sin(t/2) n，所以
  //
  //     t = 2 * atan2(|v|, c)，  log(q) = t * v / |v|
  //
  // 为什么用 atan2 而不是 acos(c)：
  //   - atan2(s, c) 只看 s 和 c 的比例，q 不是单位四元数（比如被乘了 3）也能算对；
  //   - acos 在 c ≈ 1（转角很小）时很不准：acos 的导数在 1 处是无穷大。
  //
  // 检查：log(exp(w)) = w（|w| < pi 时）；log(-q) = log(q)。

  // 第 1 步：取出实部和向量部分（复制一份，因为下面可能要取反）。
  double c = q.w();
  Eigen::Vector3d v = q.vec();

  // 第 2 步：处理“双重覆盖”。q 和 -q 是同一个转动（绕 n 转 t 和绕 -n 转 2pi - t 是一回事）。
  //   我们要短的那个，也就是 t <= pi。t/2 <= pi/2 等价于 c = cos(t/2) >= 0，
  //   所以 c < 0 时把整个四元数取反。
  if (c < 0) {
    c = -c;
    v = -v;
  }

  // 第 3 步：s = |v| = sin(t/2)（差一个正的倍数）。s == 0 说明没有转动，返回零向量（否则下面是 0/0）。
  //   只有 s 恰好为 0 才需要特殊处理：s 很小但不为 0 时，atan2(s, c) / s 仍然算得很准。
  const double s = v.norm();
  if (s == 0) return Eigen::Vector3d::Zero();

  // 第 4 步：转角 t = 2 atan2(s, c)，方向 v / s，相乘就是旋转向量。
  const double angle = 2 * std::atan2(s, c);
  return v * (angle / s);
}

}  // namespace rcipc
