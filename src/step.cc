#include "rcipc/step.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Cholesky>

#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/drive.h"
#include "rcipc/friction.h"
#include "rcipc/inertia.h"
#include "rcipc/kinematics.h"
#include "rcipc/limits.h"
#include "rcipc/solver.h"

// 关节坐标里的时间步。推导见 notes/03_inertia.md 第 5 节（惯性）、notes/04_ground.md 第 6 节（接触）、
// notes/08_friction.md 第 4 节（摩擦）、notes/09_solver.md（求解器和停止规则）；测试见 tests/test_step.cc 等。
namespace rcipc {

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// 整步里顶点的最大位移（世界坐标的收敛判据，书 §10.6.1）：max_i |X_i(F1) - X_i(F0)|。没有顶点时返回 -1。
double max_vertex_displacement(const Model& model, const Frames& F0, const Frames& F1) {
  if (model.nvertex() == 0) return -1;
  double dx = 0;
  for (const Vertex& v : model.vertices) {
    dx = std::max(dx, (world_point(F1, v.body, v.local) - world_point(F0, v.body, v.local)).norm());
  }
  return dx;
}

}  // namespace

StepResult step(const Model& model, const StepParams& P, const VectorXd& q0, const VectorXd& v0, SolverState* state,
                std::vector<NewtonIterate>* trace) {
  // 一个时间步：用 Newton 法最小化 E(q) = 惯性项 + kappa * 障碍项 + 摩擦项。
  // 结构和阶段 1 的 ipc1d::step 完全一样（可以打开 toys/ipc1d/ipc1d.cc 对照），区别都标在下面。
  const double h = P.h;
  SolverState local;
  if (state == nullptr) state = &local;
  const KappaRange krange = kappa_range(model, P.dhat);

  // 准备（阶段 1 里是 xt = predicted(P, x0, v0)）。
  //   上一步的位置 q_{n-1} = q_n ⊕ (-h v_n)：我们只存了 (q, v)，而惯性目标要用两个时刻的位姿，所以倒推一下。
  //   目标在整个时间步里不变，只算一次。
  const VectorXd q_prev = integrate(model, q0, v0, -h);
  const Frames F0 = forward_kinematics(model, q0);
  const InertiaTargets T = inertia_targets(model, F0, forward_kinematics(model, q_prev), h);
  // 步首的惯性梯度 m (c0 - c~) / h^2 = -m (v / h + g)：这一步要“刹住”的动量加重力。kappa 的初值用它来平衡
  // （阶段 4 笔记第 4 节）。不能用当前迭代点的惯性梯度：接触常常在一个已经收敛的迭代点上才第一次出现，那里它是 0。
  const VectorXd g_e0 = inertia_gradient(model, F0, T, h);

  // 阶段 8：步首把活跃的接触冻结成摩擦接触（法向力、切向基、模板都取步首的值，这一步里当常数）。
  //   kappa 还没初始化（第一次接触的那一步）时没有摩擦：这一步里新出现的接触要到下一步才有摩擦。
  const std::vector<FrictionContact> FC =
      P.mu > 0 ? lag_friction(model, F0, find_contacts(model, F0, P.dhat), state->kappa, P.dhat)
               : std::vector<FrictionContact>();
  // 阶段 12：驱动、阻尼、armature 的系数（步首的 q0、v0 和这一步的控制量定下来，这一步里不变）。
  const DriveTerms D = drive_terms(model, q0, v0, P.ctrl, h, P.eps_v);

  // 能量作为 q 的函数：先做 FK，重新找活跃的接触（换了位置，活跃的接触可能变了），再把各项加起来。线搜索里反复用。
  const auto energy_at = [&](const VectorXd& q, double kappa) {
    const Frames F = forward_kinematics(model, q);
    return inertia_energy(model, F, T, h) + contact_energy(model, find_contacts(model, F, P.dhat), kappa, P.dhat) +
           friction_energy(model, F, FC, h, P.mu, P.eps_v) + limit_energy(model, q, P.limit_k, P.limit_dhat) +
           drive_energy(model, D, q);
  };

  // 从上一步结束的位置出发（它是不相交的）。newton 记录做了几次迭代，循环结束以后还要返回它。
  VectorXd q = q0;
  int newton = 0, pcg_total = 0, curv_dropped = 0;
  bool semi_stopped = false;
  double beta = 1.0;  // 半隐式停止的累积量
  double iterate_min = std::numeric_limits<double>::infinity();
  for (int it = 1; it <= P.max_newton; ++it) {
    newton = it;

    // 第 0 步：当前点的 FK 和活跃接触。第一次出现接触时初始化 kappa（梯度平衡，阶段 4 笔记第 4 节）。
    const Frames F = forward_kinematics(model, q);
    const std::vector<Contact> C = find_contacts(model, F, P.dhat);
    if (!C.empty() && state->kappa == 0) {
      state->kappa = initial_kappa(model, F, C, g_e0, P.dhat);
      state->dmin_prev = min_distance(model, F);
    }
    const double kappa = state->kappa;

    // 第 1 步：Newton 方向。阶段 1 是 p = -E' / E''，现在是解线性方程组 H p = -g。
    //   g = 惯性梯度 + 接触梯度 + 摩擦梯度；
    //   H = H_k + sum_r D_r r_r^T r_r：树块 H_k = M(q) / h^2 + 曲率上界的对角（阶段 3、6），
    //       行 = 障碍行（D = kappa b''，阶段 4）和摩擦行（阶段 8）。
    //   阶段 9 起用树块预条件的 PCG 解它（不组装 H）；LDLT 留着做参照（solver 参数）。
    //   阶段 10：关节限位的梯度和对角 Hessian（进树块）。阶段 12：驱动、阻尼、armature 也是（drive.h）。
    const VectorXd g = inertia_gradient(model, F, T, h) + contact_gradient(model, F, C, kappa, P.dhat) +
                       friction_gradient(model, F, FC, h, P.mu, P.eps_v) + limit_gradient(model, q, P.limit_k, P.limit_dhat) +
                       drive_gradient(model, D, q);
    NewtonSystem sys;
    const MatrixXd Mh = inertia_hessian(model, F, h);
    sys.Hk = Mh + contact_curvature(model, q, C, kappa, P.dhat, P.curv_scale);
    sys.Hk.diagonal() += limit_hessian_diagonal(model, q, P.limit_k, P.limit_dhat) + drive_hessian_diagonal(model, D, q);
    //   阶段 13：再加上 Gauss-Newton 丢掉的惯性项 K（轻链吊重物时它是链上拉力的几何刚度，比 M / h^2 大几十倍），
    //   每棵树单独判断，不正定的树不加。
    if (P.inertia_curvature) curv_dropped += add_inertia_curvature(model, Mh, inertia_curvature(model, F, T, h, P.inertia_curvature_rot), &sys.Hk,
                                                     P.inertia_curvature_floor);
    sys.rows = contact_rows(model, F, C, kappa, P.dhat);
    const std::vector<Row> frows = friction_rows(model, F, FC, h, P.mu, P.eps_v);
    sys.rows.insert(sys.rows.end(), frows.begin(), frows.end());
    VectorXd p;
    int pcg_iters = 0;
    if (P.solver == LinearSolver::kPcg) {
      const PcgResult sol = pcg(sys, build_preconditioner(model, sys), -g, P.pcg_tol, P.pcg_max);
      p = sol.x;
      pcg_iters = sol.iterations;
      pcg_total += pcg_iters;
    } else {
      p = dense(sys).ldlt().solve(-g);
    }

    // 第 2 步：收敛判断（阶段 9 起在世界坐标里做，书 §10.6.1）：再走一个整步，顶点最多动多少。
    //   除以 h 相当于“这次修正的最大速度”，小于 tol 就不再移动。这次只检查、不移动的迭代也算一次。
    //   整步终点的 FK 直弦 CCD 也要用，所以在这里算一次。
    const VectorXd q_full = integrate(model, q, p, 1.0);
    const Frames F1 = forward_kinematics(model, q_full);
    const double dx = max_vertex_displacement(model, F, F1);
    const double size = (dx >= 0 ? dx : p.lpNorm<Eigen::Infinity>()) / h;
    if (size < P.tol) break;

    // 第 3 步：CCD 定上限。
    //   阶段 4 的直弦 CCD 只看 s = 0 和 s = 1 两端的顶点位置，保证沿弦走到 alpha_max 之前每个顶点离平面
    //   都还有 (1 - c) 倍的距离；阶段 5 的曲线 CCD 把弧线偏离弦的余量（矢高）也算进去。没有障碍时都返回 1。
    //   阶段 10：关节限位也要 CCD（q 沿 p 是直线，闭式解），取两者的较小值。
    const double alpha_max =
        std::min(P.ccd == CcdMode::kCurved ? curved_ccd(model, q, p, P.dhat, P.c, P.tight_bound) : chord_ccd_ground(model, F, F1, P.c),
                 limit_ccd(model, q, p, P.c));

    // 第 4 步：回溯线搜索。从 CCD 允许的最大步长开始试；能量比现在高就减半，直到能量不升。
    //   阶段 1 是 energy(x + alpha * p)；现在“加法”换成 integrate，算能量前先做 FK、找接触。
    //   alpha > 1e-12 是保险，防止无限循环。
    const double e0 = inertia_energy(model, F, T, h) + contact_energy(model, C, kappa, P.dhat) +
                      friction_energy(model, F, FC, h, P.mu, P.eps_v) + limit_energy(model, q, P.limit_k, P.limit_dhat) +
                      drive_energy(model, D, q);
    double alpha = alpha_max;
    while (energy_at(integrate(model, q, p, alpha), kappa) > e0 && alpha > 1e-12) {
      alpha *= 0.5;
    }

    // 第 5 步：记录，然后更新。阶段 1 是 x = x + alpha * p。
    if (trace != nullptr) trace->push_back({e0, size, alpha_max, alpha, min_distance(model, F), pcg_iters});
    // 调试：CCD 保证的是每次迭代的轨迹不相交。沿这一次迭代的真实轨迹采样，确认这一点（测试用）。
    if (P.check_iterates) iterate_min = std::min(iterate_min, trajectory_min_distance(model, q, p, alpha, 16));
    q = integrate(model, q, p, alpha);

    // 第 6 步（阶段 9）：kappa 调整（书 §10.7）。接受点的最小间隙小于 1e-3 dhat 并且比上一次还小：障碍太软，翻倍；
    //   大于 1e-2 dhat：可以放软，减半。都夹在范围里。
    if (P.adapt_kappa && state->kappa > 0) {
      const double dmin = min_distance(model, forward_kinematics(model, q));
      if (dmin < 1e-3 * P.dhat && dmin < state->dmin_prev) {
        state->kappa = std::min(2 * state->kappa, krange.max);
      } else if (dmin > 1e-2 * P.dhat) {
        state->kappa = std::max(0.5 * state->kappa, krange.min);
      }
      state->dmin_prev = dmin;
    }

    // 第 7 步（阶段 9）：半隐式提前停止（书 §10.6.2）。从第 semi_kmin 次迭代起累积 beta <- beta (1 - alpha_max)；
    //   CCD 连续几次都放行整步时 beta 很快降到 0，说明离障碍已经够远，这次又取满了 alpha_max，就停。
    if (P.semi_implicit && it >= P.semi_kmin) {
      beta *= (1 - alpha_max);
      if (beta <= P.semi_beta_tol && alpha == alpha_max) {
        semi_stopped = true;
        break;
      }
    }
  }

  // 速度 v = (q ⊖ q0) / h，阶段 1 是 (x - x0) / h。顺便记下结束位置上的最小距离，给调用者检查不穿透。
  StepResult r;
  r.q = q;
  r.v = differentiate(model, q0, q) / h;
  r.newton = newton;
  r.pcg = pcg_total;
  r.semi_stopped = semi_stopped;
  r.kappa = state->kappa;
  r.min_distance = min_distance(model, forward_kinematics(model, q));
  r.iterate_min_distance = iterate_min;
  r.min_limit_distance = min_limit_distance(model, q);
  r.curvature_dropped = curv_dropped;
  return r;
}

double kinetic_energy(const Model& model, const VectorXd& q, const VectorXd& v) {
  // 动能 1/2 v^T M(q) v。质量矩阵借用 inertia_hessian：取 h = 1 时它就是 M(q)。
  const MatrixXd M = inertia_hessian(model, forward_kinematics(model, q), 1.0);
  return 0.5 * v.dot(M * v);
}

double potential_energy(const Model& model, const VectorXd& q) {
  // 重力势能 -sum_b m_b g . c_b（g 指向下方，所以质心越高势能越大）。
  const Frames F = forward_kinematics(model, q);
  double e = 0;
  for (int b = 1; b < model.nbody(); ++b) e -= model.bodies[b].mass * model.gravity.dot(body_com(model, F, b));
  return e;
}

}  // namespace rcipc
