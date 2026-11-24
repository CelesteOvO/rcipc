#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/limits.h"
#include "rcipc/model.h"

// 关节坐标里的一个时间步（阶段 3，笔记 notes/03_inertia.md 第 5 节；阶段 4 加上接触，notes/04_ground.md 第 6 节；
// 阶段 9 换成 PCG、世界坐标的收敛判据、kappa 调整和半隐式停止，notes/09_solver.md）。
//
// 和阶段 1 的 ipc1d::step 是同一个算法，只是未知量从一个数 x 换成了关节坐标 q：
//   x + alpha * p   ->  integrate(q, p, alpha)        （q ⊕ alpha p）
//   p = -E' / E''   ->  解 H p = -g                    （阶段 9 起用 PCG；稠密 LDLT 留着做参照）
//   (x - x0) / h    ->  differentiate(q0, q) / h      （(q ⊖ q0) / h）
//   ccd_max_step    ->  curved_ccd                     （阶段 5；阶段 4 是 chord_ccd_ground；没有障碍时上限是 1）
// 能量 E = 惯性项（含重力） + kappa * 障碍项 + 滞后摩擦（阶段 8）。
namespace rcipc {

enum class CcdMode {
  kChord,   // 阶段 4：只看弦。会漏掉弧线中间的穿透，留着做对比
  kCurved,  // 阶段 5：弦加矢高（默认）
};

enum class LinearSolver {
  kLdlt,  // 稠密 LDLT：组装 H，O(nv^3)。阶段 3–8 的做法，留着做参照
  kPcg,   // 阶段 9：树块预条件的共轭梯度（默认）
};

struct StepParams {
  double h = 0.002;     // 时间步长（s）
  double tol = 1e-6;    // 收敛判断：整步里顶点的最大位移除以 h 小于它（m/s；没有顶点时用刚体质量的均方根位移）
  int max_newton = 50;  // 最多迭代次数
  double dhat = 1e-3;   // 阶段 4：障碍激活距离（m）
  double c = 0.9;       // 阶段 4：CCD 的保守系数，一次迭代最多闭合当前距离的这个比例
  // 最小分离（m；限位里是 rad 或 m）。CCD 把 d - min_sep 当距离用，所以每个接受的迭代点上真实距离 > min_sep，
  // 只要 FK 和距离的舍入误差（约 1e-16 |x|）比它小，算出来的距离就不会变号。一对的起点距离已经 <= min_sep 时
  // CCD 对它返回 0（卡住，不再放行）。笔记 notes/22_primal_dual.md 第 5 节。
  double min_sep = 1e-12;
  CcdMode ccd = CcdMode::kCurved;  // 阶段 5：用哪种 CCD
  double curv_scale = 1.0;         // 阶段 6：曲率上界的缩放。1 是可证明的上界，0 是纯 Gauss-Newton（复现间隙塌缩）
  bool tight_bound = true;         // 阶段 5（后加）：CCD 的加速度界用静态界和紧界的较小值；false 只用静态界（笔记 05 第 6 节）
  double mu = 0.2;                 // 阶段 8：摩擦系数（所有接触同一个值；0 关掉摩擦）
  double eps_v = 1e-3;             // 阶段 8：摩擦光滑化的速度（m/s），低于它的相对滑动按“静止”处理
  // 阶段 9。
  LinearSolver solver = LinearSolver::kPcg;
  double pcg_tol = 1e-4;           // PCG 的 rz 准则：|r^T z| <= pcg_tol |r0^T z0|
  int pcg_max = 200;               // PCG 最多迭代次数
  bool adapt_kappa = true;         // 每次线搜索之后按最小间隙调整 kappa（笔记 09 第 4 节）
  bool kappa_relax = true;         // adapt_kappa 开着时，间隙 > 1e-2 dhat 要不要把 kappa 减半（false：只加硬不放软，IPC 原来的做法）
  bool semi_implicit = false;      // 半隐式提前停止（Stiff-GIPC / libuipc 的规则，笔记 09 第 5 节）
  int semi_kmin = 6;               // 半隐式：从第几次迭代起累积 beta
  double semi_beta_tol = 1e-3;     // 半隐式：beta <= 它且这次取满 alpha_max 时停
  bool check_iterates = false;     // 调试：每次接受的迭代沿真实轨迹采样最小距离，写进 StepResult::iterate_min_distance
  // 阶段 10：关节限位的障碍（limits.h）。limit_k 是刚度（J/rad^2 或 J/m^2）。
  //   limit_dhat 是铰链的激活距离（rad）；滑动关节用接触的 dhat（m），见 limits.h 和 limit_barrier()。
  double limit_k = 10.0;
  double limit_dhat = 0.1;
  LimitBarrier limit_barrier() const { return {limit_k, limit_dhat, dhat}; }
  // 阶段 12：这一步的控制量（长度 nu；空表示全 0）。时间表通常来自 MJCF 的关键帧（drive.h 的 ctrl_at）。
  Eigen::VectorXd ctrl;
  // 阶段 13：惯性 Hessian 加上 Gauss-Newton 丢掉的那一项（inertia_curvature；每棵树正定时才加）。
  // false 退回阶段 3 的 M / h^2（笔记 notes/16_inertia_curvature.md）。
  bool inertia_curvature = true;
  // 其中转动那一部分（R S (R - A~)^T / h^2，刚体绕自己的质心转得快时它让 Newton 走过头）要不要（笔记 16 第 5 节）。
  bool inertia_curvature_rot = false;
  // 树块上 floor * M / h^2 + K 正定才加 K（add_inertia_curvature）。0.99：K 可以抵掉质量矩阵 99% 的曲率；
  // 小的 floor 只在 K 接近半正定（链被拉紧）时才加。笔记 16 第 5 节比较了两者。
  double inertia_curvature_floor = 0.99;
  // 阶段 14：接触的 Hessian 用完整的二阶导数（每个接触 12 x 12 的旋量块做半正定投影，contact_rows_full），
  // 链上的剩余项和惯性的曲率项一起经正定检查进树块；这时阶段 6 的曲率上界不再需要。
  // false 退回 Gauss-Newton 行加曲率上界（笔记 notes/17_contact_hessian.md）。GPU 版要和 primal_dual = false 一起用。
  bool full_contact_hessian = false;
  // 原始-对偶障碍（笔记 notes/22_primal_dual.md 第 2 节）。每个接触的法向力 lambda 是这一步里的独立变量（按接触的图元记住），
  //   对互补条件 lambda (kappa / F(d)) = kappa 和 q 一起做 Newton；消掉 Delta lambda 以后方程和原始障碍的一样
  //   （右端就是原始的梯度），只是接触的 Hessian（行、阶段 6 的上界、旋量块、链上项）乘上 lambda_eff / F(d)，
  //   lambda_eff = w lambda + (1 - w) F，w = min(1, 1 / (-d F'/F))。
  //   为什么：原始的 Newton 在被压扁的间隙上每次迭代最多把间隙翻一倍，而对偶变量沿割线一步回到平衡间隙；
  //   间隙不再一次次被压塌（笔记 22 第 1 节）。false 退回原始障碍，只用来对比。
  bool primal_dual = true;
};

// 跨时间步保留的求解器状态。kappa 在第一次出现接触时初始化，之后沿用并按间隙调整。
struct SolverState {
  double kappa = 0;      // 障碍刚度（J/m^4）；0 表示还没初始化
  double dmin_prev = 0;  // 上一次线搜索之后的最小间隙（kappa 调整用：间隙还在缩小才加硬）
};

// 一次 Newton 迭代的记录，用来看收敛过程。
struct NewtonIterate {
  double energy;        // 迭代开始时的能量 E(q)
  double step;          // 收敛判据用的量：整步里顶点的最大位移 / h
  double alpha_max;     // CCD 给的步长上限（没有接触时是 1）
  double alpha;         // 线搜索接受的步长
  double min_distance;  // 迭代开始时的最小距离（阶段 6 用它看间隙有没有被一次次挖小）
  int pcg;              // 阶段 9：这次迭代 PCG 的次数（LDLT 时是 0）
};

struct StepResult {
  Eigen::VectorXd q;        // 新位置 q_{n+1}
  Eigen::VectorXd v;        // 新速度 v_{n+1} = (q_{n+1} ⊖ q_n) / h
  int newton = 0;           // 迭代次数，含最后那次只检查不移动的迭代（同阶段 1）
  int pcg = 0;              // 阶段 9：这一步 PCG 的总次数
  bool semi_stopped = false;  // 阶段 9：是不是被半隐式规则提前停下的
  bool capped = false;        // 撞了 max_newton 的上限：没收敛，也不是半隐式停下的
  double kappa = 0;         // 这一步结束时的 kappa（没有接触时是 0）
  double min_distance = 0;  // 结束位置上所有候选对的最小距离；没有障碍时是 +无穷
  double iterate_min_distance = 0;  // check_iterates 时：所有接受的迭代轨迹（采样）上的最小距离，否则 +无穷
  double min_limit_distance = 0;    // 阶段 10：结束位置上到最近的关节限位的距离；没有限位时 +无穷
  int curvature_dropped = 0;        // 阶段 13：这一步里有几次（迭代 x 树）因为不正定没加惯性的曲率项
};

// 两阶段曲率（笔记 notes/22_primal_dual.md 第 4 节，书 §8.6）：这一步里某次迭代走了整步（CCD 没截断、
//   线搜索也取了整步）以后，Gauss-Newton 行的接触不再加阶段 6 的对角上界，改成沿每个接触自己那一行 r 的秩 1 曲率
//   （contact_curvature_row_weight），一直用到这一步结束。CCD 截断的时候（间隙可能被压塌）仍用上界。
//   CPU 版和 GPU 版共用：每次要移动的迭代做完线搜索以后调一次 track_full_step。
struct FullStepTrack {
  bool full_step_seen = false;
};
inline void track_full_step(double alpha_max, double alpha, FullStepTrack* track) {
  if (alpha_max >= 1 && alpha == alpha_max) track->full_step_seen = true;
}

// kappa 的调整（书 §10.7，笔记 notes/09_solver.md 第 4 节；step.cc 和 GPU 版共用）。每接受一个迭代点调用一次，
//   dmin 是这个点上的最小间隙：小于 1e-3 dhat 并且比上一次还小，翻倍；大于 1e-2 dhat，减半。都夹在 [kmin, kmax] 里。
//   adapt_kappa 关着或 kappa 还没初始化时什么也不做。
void kappa_after_iterate(const StepParams& P, double kmin, double kmax, double dmin, SolverState* state);

// 从 (q0, v0) 走一个时间步。state 为空时用一个临时状态（kappa 每步重新初始化）。
// trace 不为空时，把每次移动的迭代记进去。
StepResult step(const Model& model, const StepParams& params, const Eigen::VectorXd& q0,
                const Eigen::VectorXd& v0, SolverState* state = nullptr,
                std::vector<NewtonIterate>* trace = nullptr);

// 动能 1/2 v^T M(q) v 和重力势能 -sum_b m_b g . c_b（单位 J），用来检查能量守恒。
double kinetic_energy(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& v);
double potential_energy(const Model& model, const Eigen::VectorXd& q);

}  // namespace rcipc
