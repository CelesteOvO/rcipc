#pragma once

namespace rcipc {

// 障碍函数在某一点的值，以及它的一阶、二阶导数。
struct BarrierValue {
  double b = 0;    // b(x)
  double db = 0;   // b'(x)
  double ddb = 0;  // b''(x)
};

// IPC 的截断 log 障碍（推导见 notes/01_ipc1d.md 第 2 节，书 §3.2 第 48 页）：
//
//   b(x) = -(x - xhat)^2 * ln(x / xhat)    当 0 < x < xhat
//   b(x) = 0                               当 x >= xhat
//
// 直观理解：x 是间隙，xhat 是激活距离。
//   - 间隙大于 xhat 时，障碍完全不存在；
//   - 间隙小于 xhat 时，障碍开始把间隙往外推，越近推得越狠；
//   - x -> 0 时 b -> +无穷，所以“让能量下降”的算法永远走不到 x = 0。
//
// 参数怎么传：
//   - 阶段 1 传距离本身：barrier(d, dhat)；
//   - 从阶段 4 起传平方距离：barrier(d * d, dhat * dhat)，旧实现就是这样做的。函数本身不用改。
//
// 前提：x > 0。调用者（CCD）保证每个迭代点都不相交，所以这里不需要处理 x <= 0 的情况。
BarrierValue barrier(double x, double xhat);

}  // namespace rcipc
