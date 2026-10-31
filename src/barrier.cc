#include "rcipc/barrier.h"

#include <cmath>

namespace rcipc {

BarrierValue barrier(double x, double xhat) {
  // 激活区之外（x >= xhat），b、b'、b'' 都是 0。
  if (x >= xhat) return {};

  // 激活区之内，先算两个中间量（它们都是负数），这样 b = -r^2 * L：
  const double r = x - xhat;
  const double L = std::log(x / xhat);

  // 对 b = -r^2 * L 用乘积法则求导，其中 r' = 1，L' = 1/x，L'' = -1/x^2（推导见笔记第 2 节）：
  //   b   = -r^2 * L
  //   b'  = -2 r L - r^2 / x
  //   b'' = -2 L - 4 r / x + r^2 / x^2
  //
  // 性质：x 稍小于 xhat 时三项都接近 0（和外侧的 0 函数光滑地接上）；
  //       在 (0, xhat) 上 b > 0，b' < 0（斥力），b'' > 0（凸）；x -> 0 时 b -> +无穷。
  BarrierValue v;
  v.b = -r * r * L;
  v.db = -2 * r * L - r * r / x;
  v.ddb = -2 * L - 4 * r / x + r * r / (x * x);
  return v;
}

}  // namespace rcipc
