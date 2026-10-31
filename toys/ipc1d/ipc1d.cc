#include "ipc1d/ipc1d.h"

#include <algorithm>
#include <cmath>

#include "rcipc/barrier.h"

// 一维 IPC 的实现。推导见 notes/01_ipc1d.md，测试见 tests/test_ipc1d.cc。
namespace rcipc::ipc1d {

double predicted(const Params& P, double x0, double v0) {
  // 预测位置 x~：只有惯性和重力、没有地面时，隐式 Euler 一步会走到哪里。
  //
  //     x~ = x0 + h * v0 - h^2 * g
  //
  // 三项分别是：当前位置；按当前速度走一步的位移；重力在这一步里造成的位移。
  // 重力使间隙减小，所以是减号。
  //
  // 为什么是 h^2 * g 而不是匀加速公式里的 h^2 * g / 2：隐式 Euler 先更新速度 v1 = v0 - h g，
  // 再用新速度更新位置 x1 = x0 + h v1，展开就是上式。它是一种离散方式，不是精确解。
  return x0 + P.h * v0 - P.h * P.h * P.g;
}

double energy(const Params& P, double x, double xt) {
  // 增量势 E(x)：一步时间积分要最小化的函数（笔记第 1 节）。
  //
  //     E(x) = m / (2 h^2) * (x - xt)^2  +  kappa * b(x)
  //            \_______ 惯性项 ________/     \_ 障碍项 _/
  //
  // 惯性项：x 离预测位置 xt 越远，能量越高。它表达的是“球想按惯性走到 xt”。
  // 障碍项：b(x) 用 barrier(x, P.dhat).b。x >= dhat 时它是 0，这时 E 只剩惯性项。
  // 两项的单位都是焦耳。
  const double inertia = 0.5 * P.m / (P.h * P.h) * (x - xt) * (x - xt);
  const double contact = P.kappa * barrier(x, P.dhat).b;
  return inertia + contact;
}

double gradient(const Params& P, double x, double xt) {
  // E 对 x 的一阶导数：
  //
  //     E'(x) = m / h^2 * (x - xt)  +  kappa * b'(x)
  //
  // b'(x) 用 barrier(x, P.dhat).db。
  //
  // 物理意义：E'(x) = 0 正是隐式 Euler 方程 m (x - xt) / h^2 = -kappa b'(x)。
  // 左边是质量乘加速度（重力已经并进 xt），右边是障碍力（b' < 0，所以力 -kappa b' > 0，向上推）。
  return P.m / (P.h * P.h) * (x - xt) + P.kappa * barrier(x, P.dhat).db;
}

double hessian(const Params& P, double x, double xt) {
  // E 对 x 的二阶导数：
  //
  //     E''(x) = m / h^2  +  kappa * b''(x)
  //
  // b''(x) 用 barrier(x, P.dhat).ddb。
  // xt 在这里用不到：惯性项是二次函数，二阶导数是常数。(void)xt; 用来消掉“参数未使用”的警告。
  //
  // 一维里 m / h^2 > 0、b'' >= 0，所以 E'' 总是正的，Newton 方向一定是下降方向（思考题 4）。
  (void)xt;
  return P.m / (P.h * P.h) + P.kappa * barrier(x, P.dhat).ddb;
}

double ccd_max_step(double x, double p, double c) {
  // 一维 CCD（保守推进，笔记第 3.2 节）：从间隙 x 沿方向 p 走，s 最大能取到多少。
  //
  // 输入：x 当前间隙（> 0）；p Newton 方向（这次迭代想走的位移）；c 每次最多闭合间隙的比例（0.9）。
  // 输出：alpha_max ∈ (0, 1]。要求对所有 s ∈ [0, alpha_max]，间隙 x + s * p 都 >= (1 - c) * x，
  //       也就是这次迭代最多闭合当前间隙的 c 倍。
  //
  // 分两种情况：
  //   情况 1：p >= 0。球在远离地面（或不动），走整步也安全，返回 1。
  //   情况 2：p < 0。球在靠近地面，走 s 时间隙减小了 s * (-p)。
  //           要求减小量 <= c * x，所以 s <= c * x / (-p)。
  //           这个值可能大于 1（整步都闭合不到 c * x），而一次迭代最多走整步，所以再和 1 取较小值。
  //
  // 例子：x = 3 mm，p = -6 mm，c = 0.9：s <= 0.9 * 3 / 6 = 0.45，停在 3 - 0.45 * 6 = 0.3 mm。
  //
  // 为什么不直接走到地面（c = 1）：停在离地还有 (1 - c) * x 的地方，障碍仍是有限值，
  // 下一次迭代还有余地。思考题 3 会让你试试 c = 1。
  if (p >= 0) return 1.0;            // 情况 1：远离地面（或不动）
  return std::min(1.0, c * x / -p);  // 情况 2：靠近地面
}

StepResult step(const Params& P, double x0, double v0, std::vector<Iterate>* trace) {
  // 一个时间步：用 Newton 法最小化 E(x)（笔记第 3.4 节）。
  //
  // 准备：
  //   xt = predicted(P, x0, v0)       预测位置，整个时间步里不变
  //   x  = x0                         从上一步结束的位置出发，它是不相交的
  const double xt = predicted(P, x0, v0);
  double x = x0;

  // newton 记录做了几次迭代。它声明在循环外面，因为循环结束以后还要把它返回。
  // 从 1 开始数：第 1 次迭代 newton = 1。
  int newton = 0;
  for (int it = 1; it <= P.max_newton; ++it) {
    newton = it;

    // 第 1 步：Newton 方向。
    //   在当前点用二次函数近似 E，p 是跳到这个二次函数最低点所需的位移：p = -E'(x) / E''(x)。
    //   一维里 E'' > 0，所以 p 总是指向能量下降的方向。
    const double p = -gradient(P, x, xt) / hessian(P, x, xt);

    // 第 2 步：收敛判断。
    //   |p| / h 是“这次修正相当于多大的速度”。它小于 tol，说明已经在最小点上了，不再移动。
    //   这次只检查、不移动的迭代也算一次（newton 已经等于 it），所以自由落体是 2 次，静止时是 1 次。
    if (std::abs(p) / P.h < P.tol) break;

    // 第 3 步：CCD 定上限。
    //   alpha_max 保证从 x 走到 x + alpha * p（任何 alpha <= alpha_max）的整段路径都不碰地面，
    //   并且这次最多吃掉当前间隙的 c 倍。
    const double alpha_max = ccd_max_step(x, p, P.c);

    // 第 4 步：回溯线搜索。
    //   从 CCD 允许的最大步长开始试。如果那里的能量比现在高，就把步长减半再试，直到能量不升。
    //   为什么需要它：Newton 的二次模型会低估障碍有多陡，CCD 允许的步长不一定让能量下降。
    //   alpha > 1e-12 是保险：万一能量一直降不下来，也不会无限循环。
    const double e0 = energy(P, x, xt);
    double alpha = alpha_max;
    while (energy(P, x + alpha * p, xt) > e0 && alpha > 1e-12) {
      alpha *= 0.5;
    }

    // 第 5 步：记录，然后更新。
    //   先记录，因为记录里的 x 要是这次迭代开始时的间隙；然后才移动 x。
    if (trace != nullptr) trace->push_back({x, p, alpha_max, alpha});
    x = x + alpha * p;
  }

  // 返回这一步的结果：
  //   - 速度用隐式 Euler 的更新 v = (x - x0) / h；
  //   - newton 包括最后那次收敛检查；如果跑满 max_newton 也没收敛，它就是 max_newton。
  return {x, (x - x0) / P.h, newton};
}

}  // namespace rcipc::ipc1d
