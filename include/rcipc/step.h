#pragma once

#include <vector>

#include <Eigen/Core>

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
  double tol = 1e-6;    // 收敛判断：整步里顶点的最大位移除以 h 小于它（m/s；没有顶点时退回 max_i |p_i| / h）
  int max_newton = 50;  // 最多迭代次数
  double dhat = 1e-3;   // 阶段 4：障碍激活距离（m）
  double c = 0.9;       // 阶段 4：CCD 的保守系数，一次迭代最多闭合当前距离的这个比例
  CcdMode ccd = CcdMode::kCurved;  // 阶段 5：用哪种 CCD
  double curv_scale = 1.0;         // 阶段 6：曲率上界的缩放。1 是可证明的上界，0 是纯 Gauss-Newton（复现间隙塌缩）
  double mu = 0.2;                 // 阶段 8：摩擦系数（所有接触同一个值；0 关掉摩擦）
  double eps_v = 1e-3;             // 阶段 8：摩擦光滑化的速度（m/s），低于它的相对滑动按“静止”处理
  // 阶段 9。
  LinearSolver solver = LinearSolver::kPcg;
  double pcg_tol = 1e-4;           // PCG 的 rz 准则：|r^T z| <= pcg_tol |r0^T z0|
  int pcg_max = 200;               // PCG 最多迭代次数
  bool adapt_kappa = true;         // 每次线搜索之后按最小间隙调整 kappa（笔记第 4 节）
  bool semi_implicit = false;      // 半隐式提前停止（Stiff-GIPC / libuipc 的规则，笔记第 5 节）
  int semi_kmin = 6;               // 半隐式：从第几次迭代起累积 beta
  double semi_beta_tol = 1e-3;     // 半隐式：beta <= 它且这次取满 alpha_max 时停
  bool check_iterates = false;     // 调试：每次接受的迭代沿真实轨迹采样最小距离，写进 StepResult::iterate_min_distance
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
  double kappa = 0;         // 这一步结束时的 kappa（没有接触时是 0）
  double min_distance = 0;  // 结束位置上所有候选对的最小距离；没有障碍时是 +无穷
  double iterate_min_distance = 0;  // check_iterates 时：所有接受的迭代轨迹（采样）上的最小距离，否则 +无穷
};

// 从 (q0, v0) 走一个时间步。state 为空时用一个临时状态（kappa 每步重新初始化）。
// trace 不为空时，把每次移动的迭代记进去。
StepResult step(const Model& model, const StepParams& params, const Eigen::VectorXd& q0,
                const Eigen::VectorXd& v0, SolverState* state = nullptr,
                std::vector<NewtonIterate>* trace = nullptr);

// 动能 1/2 v^T M(q) v 和重力势能 -sum_b m_b g . c_b（单位 J），用来检查能量守恒。
double kinetic_energy(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& v);
double potential_energy(const Model& model, const Eigen::VectorXd& q);

}  // namespace rcipc
