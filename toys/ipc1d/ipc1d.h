#pragma once

#include <vector>

// 阶段 1：一维 IPC（笔记 notes/01_ipc1d.md）。
//
// 模型：一个质量为 m 的球竖直落向地面。状态只有两个数：
//   x：球底到地面的间隙（m），始终 > 0；
//   v：竖直速度（m/s），向上为正，所以下落时 v < 0。
//
// 一个时间步做的事：给定上一步结束时的 (x0, v0)，求这一步结束时的 (x, v)。
// 做法是最小化增量势 E(x)（第 1 节），最小点就是新的 x，再由 v = (x - x0) / h 得到新的速度。
namespace rcipc::ipc1d {

// 所有参数。默认值就是书 §3.8 的例子。
struct Params {
  double m = 1.0;      // 质量，kg
  double g = 9.81;     // 重力加速度，m/s^2，指向地面（使 x 减小）
  double h = 0.002;    // 时间步长，s
  double dhat = 1e-3;  // 障碍激活距离，m：间隙小于它时障碍才起作用
  double kappa = 1e4;  // 障碍刚度，N/m：决定球静止时停在离地多远
  double c = 0.9;      // CCD 系数：一次迭代最多闭合当前间隙的这个比例
  double tol = 1e-6;   // 收敛容差，m/s：Newton 修正量 |p| 除以 h 小于它时停止
  int max_newton = 50;  // 每步最多 Newton 迭代次数
};

// 预测位置 x~：没有地面时，隐式 Euler 一步会走到的位置。
double predicted(const Params& P, double x0, double v0);

// 增量势 E(x) = m / (2 h^2) * (x - xt)^2 + kappa * b(x)，以及它对 x 的一阶、二阶导数。
// xt 是预测位置 x~（由 predicted 算出）。前提：x > 0。
double energy(const Params& P, double x, double xt);
double gradient(const Params& P, double x, double xt);
double hessian(const Params& P, double x, double xt);

// 一维 CCD（保守推进）：从间隙 x 沿方向 p 走，最多能走多远。
// 返回 alpha_max ∈ (0, 1]，保证对所有 s ∈ [0, alpha_max]，间隙 x + s * p 都 >= (1 - c) * x。
// p >= 0（远离地面），或者整步都不会闭合超过 c * x 时，返回 1。
double ccd_max_step(double x, double p, double c);

// 一次“移动了”的 Newton 迭代的记录，测试和演示程序用它来打印、检查每次迭代。
struct Iterate {
  double x;          // 这次迭代开始时的间隙
  double p;          // Newton 方向
  double alpha_max;  // CCD 给的步长上限
  double alpha;      // 线搜索最终接受的步长；这次迭代之后间隙变成 x + alpha * p
};

// 一个时间步的结果。
struct StepResult {
  double x;    // 这一步结束时的间隙
  double v;    // 这一步结束时的速度，(x - x0) / h
  int newton;  // 做了几次 Newton 迭代，包括最后那次“发现已经收敛、不再移动”的检查
};

// 一个时间步：用 Newton 法最小化 E(x)。每次迭代先用 CCD 限制最大步长，再用回溯线搜索保证能量不升
// （笔记第 3 节）。如果 trace 不是 nullptr，每次移动了的迭代都会追加一条记录。
StepResult step(const Params& P, double x0, double v0, std::vector<Iterate>* trace = nullptr);

}  // namespace rcipc::ipc1d
