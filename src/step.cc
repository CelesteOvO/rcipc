#include "rcipc/step.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

#include <Eigen/Cholesky>

#include "rcipc/barrier.h"
#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/drive.h"
#include "rcipc/friction.h"
#include "rcipc/inertia.h"
#include "rcipc/kinematics.h"
#include "rcipc/limits.h"
#include "rcipc/solver.h"

// 关节坐标里的时间步。推导见 notes/03_inertia.md 第 5 节（惯性）、notes/04_ground.md 第 6 节（接触）、
// notes/08_friction.md 第 4 节（摩擦）、notes/09_solver.md（求解器和停止规则）、notes/22_primal_dual.md（原始-对偶障碍、
// 两阶段曲率）；测试见 tests/test_step.cc 等。
namespace rcipc {

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// 整步里顶点的最大位移（世界坐标的收敛判据，书 §10.6.1）：max_i |X_i(F1) - X_i(F0)|。
//
// 模型没有顶点时（几何都不参与碰撞的长链）用刚体质量的均方根位移代替，单位同样是米：
//   d_b^2 = |dc|^2 + dth^T I_c dth / m，  dc 是质心的位移，dth 是转动向量（世界坐标），I_c 是绕质心的惯性张量。
//   它是刚体上各质点位移平方按质量的平均（sum_p m_p |dc + dth x r_p|^2 / m），也就是惯性能量用的那个度量。
//   以前这里用关节空间的 max|p|，单位是弧度：它在能量的舍入误差以下才降得到容差，线搜索看不见那么小的下降，
//   于是空转到上限（笔记 22 第 5 节）。
// 没有顶点、也没有带质量的刚体时返回 -1，调用者退回 max|p|。
double max_vertex_displacement(const Model& model, const Frames& F0, const Frames& F1) {
  double dx = 0;
  if (model.nvertex() > 0) {
    for (const Vertex& v : model.vertices) {
      dx = std::max(dx, (world_point(F1, v.body, v.local) - world_point(F0, v.body, v.local)).norm());
    }
    return dx;
  }
  bool any = false;
  for (int b = 1; b < model.nbody(); ++b) {
    const Body& B = model.bodies[b];
    if (B.mass <= 0) continue;
    any = true;
    const Eigen::AngleAxisd turn(F1.xquat[b] * F0.xquat[b].conjugate());
    const Eigen::Vector3d th = turn.angle() * turn.axis();
    const Eigen::Matrix3d R = F0.xquat[b].toRotationMatrix();
    const double d2 = (body_com(model, F1, b) - body_com(model, F0, b)).squaredNorm() +
                      th.dot(R * B.inertia * R.transpose() * th) / B.mass;
    dx = std::max(dx, std::sqrt(d2));
  }
  return any ? dx : -1;
}

// 原始-对偶障碍的对偶变量按接触的图元记：类型、模板顶点（只取前 nstencil 个，其余记 -1）、平面。GPU 版的键同样
//   只散列前 nstencil 个顶点（src/gpu/contact.cu 的 pd_key_of）。地面接触只有一个模板顶点：以前四个槽位全进键，
//   后三个是没有初始化的内存，同一个接触在相邻两次迭代里可能对不上，查不到上一次的 lambda（CPU 版的 pd 时有时无）。
using PdKey = std::array<int, 6>;
PdKey pd_key(const Contact& c) {
  PdKey k = {static_cast<int>(c.type), -1, -1, -1, -1, c.plane};
  for (int i = 0; i < c.nstencil; ++i) k[1 + i] = c.verts[i];
  return k;
}

// 原始障碍的法向力 F = -2 kappa kw m b'(d^2) d，和 dlog = (dF/dd) / F。
double pd_force(const Contact& c, double kappa, double dhat, double* dlog) {
  const BarrierValue bv = barrier(c.d * c.d, dhat * dhat);
  *dlog = bv.db != 0 ? (bv.db + 2 * c.d * c.d * bv.ddb) / (c.d * bv.db) : 0;
  return -2 * kappa * c.kw * c.moll * bv.db * c.d;
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
  const LimitBarrier L = P.limit_barrier();

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
           friction_energy(model, F, FC, h, P.mu, P.eps_v) + limit_energy(model, q, L) + drive_energy(model, D, q);
  };

  // 从上一步结束的位置出发（它是不相交的）。newton 记录做了几次迭代，循环结束以后还要返回它。
  VectorXd q = q0;
  int newton = 0, pcg_total = 0, curv_dropped = 0;
  bool semi_stopped = false, converged = false;
  double beta = 1.0;  // 半隐式停止的累积量
  double iterate_min = std::numeric_limits<double>::infinity();
  // 两阶段曲率：这一步里有没有哪次迭代走了整步（笔记 22 第 4 节）。
  FullStepTrack fstep;
  // 原始-对偶障碍：这一步里每个接触（按图元）的对偶变量 lambda（法向力，N），以及它是按哪个 kappa 记的
  //   （kappa 翻倍、减半时按比例缩放，否则 lambda / F 会跟着 kappa 跳）。
  std::map<PdKey, double> pd_lambda;
  double pd_kappa = 0;
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
                       friction_gradient(model, F, FC, h, P.mu, P.eps_v) + limit_gradient(model, q, L) +
                       drive_gradient(model, D, q);

    // 原始-对偶障碍（笔记 22 第 2 节）：组装用的接触副本，kw 乘上 lambda_eff / F（梯度、能量仍用原始的 C）。
    //   lambda_eff = w lambda + (1 - w) F，w = 1 / (-d F'/F)（夹在 <= 1）：纯对数障碍 F ∝ 1/d 时 -d F'/F = 1，w = 1；
    //   截断障碍越接近 dhat，F 掉得比 1/d 快，w -> 0，组装退回原始的 Hessian。这一步里第一次出现的接触没有 lambda，用原始的。
    std::vector<Contact> Ch = C;
    if (P.primal_dual) {
      if (pd_kappa > 0 && kappa != pd_kappa)
        for (auto& kv : pd_lambda) kv.second *= kappa / pd_kappa;
      pd_kappa = kappa;
      for (Contact& c : Ch) {
        double dlog = 0;
        const double Fc = pd_force(c, kappa, P.dhat, &dlog);
        if (!(Fc > 0)) continue;
        const auto found = pd_lambda.find(pd_key(c));
        if (found == pd_lambda.end()) continue;
        const double gl = -c.d * dlog;
        const double w = gl > 1 ? 1 / gl : 1.0;
        c.kw *= w * found->second / Fc + (1 - w);
      }
    }

    NewtonSystem sys;
    const MatrixXd Mh = inertia_hessian(model, F, h);
    //   阶段 14：完整的接触 Hessian 时所有接触都用旋量块（Cf），不再加阶段 6 的上界；否则都是 Gauss-Newton 行（Cb）。
    const bool full = P.full_contact_hessian;
    const std::vector<Contact> none;
    const std::vector<Contact>& Cf = full ? Ch : none;
    const std::vector<Contact>& Cb = full ? none : Ch;
    //   两阶段曲率（笔记 22 第 4 节）：这一步里走过整步以后，Cb 的上界换成秩 1 曲率行（下面加到行的权重上）。
    const bool curv_rows = fstep.full_step_seen;
    sys.Hk = Mh + contact_curvature(model, q, curv_rows ? none : Cb, kappa, P.dhat, P.curv_scale);
    sys.Hk.diagonal() += limit_hessian_diagonal(model, q, L) + drive_hessian_diagonal(model, D, q);
    //   阶段 13：再加上 Gauss-Newton 丢掉的惯性项 K（轻链吊重物时它是链上拉力的几何刚度，比 M / h^2 大几十倍），
    //   每棵树单独判断，不正定的树不加。阶段 14 的链上剩余项也走这一道检查。
    if (P.inertia_curvature || full) {
      MatrixXd K = MatrixXd::Zero(model.nv, model.nv);
      if (P.inertia_curvature) K += inertia_curvature(model, F, T, h, P.inertia_curvature_rot);
      if (!Cf.empty()) K += contact_chain_curvature(model, F, Cf, kappa, P.dhat);
      curv_dropped += add_inertia_curvature(model, Mh, K, &sys.Hk, P.inertia_curvature_floor);
    }
    sys.rows = contact_rows(model, F, Cb, kappa, P.dhat);
    if (curv_rows)
      for (size_t i = 0; i < Cb.size(); ++i)
        sys.rows[i].D += contact_curvature_row_weight(model, F, Cb[i], sys.rows[i].r, kappa, P.dhat);
    if (!Cf.empty()) contact_rows_full(model, F, Cf, kappa, P.dhat, &sys.rows);
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

    // 第 2 步：收敛判断（阶段 9 起在世界坐标里做，书 §10.6.1）：再走一个整步，顶点最多动多少
    //   （没有顶点的模型看刚体质量的均方根位移，见 max_vertex_displacement）。
    //   除以 h 相当于“这次修正的最大速度”，小于 tol 就不再移动。这次只检查、不移动的迭代也算一次。
    //   整步终点的 FK 直弦 CCD 也要用，所以在这里算一次。
    const VectorXd q_full = integrate(model, q, p, 1.0);
    const Frames F1 = forward_kinematics(model, q_full);
    const double dx = max_vertex_displacement(model, F, F1);
    const double size = (dx >= 0 ? dx : p.lpNorm<Eigen::Infinity>()) / h;
    if (size < P.tol) {
      converged = true;
      break;
    }

    // 第 3 步：CCD 定上限。
    //   阶段 4 的直弦 CCD 只看 s = 0 和 s = 1 两端的顶点位置，保证沿弦走到 alpha_max 之前每个顶点离平面
    //   都还有 (1 - c) 倍的距离；阶段 5 的曲线 CCD 把弧线偏离弦的余量（矢高）也算进去。没有障碍时都返回 1。
    //   阶段 10：关节限位也要 CCD（q 沿 p 是直线，闭式解），取两者的较小值。都按最小分离 min_sep 留余量。
    const double alpha_max =
        std::min(P.ccd == CcdMode::kCurved ? curved_ccd(model, q, p, P.dhat, P.c, P.tight_bound, P.min_sep)
                                           : chord_ccd_ground(model, F, F1, P.c),
                 limit_ccd(model, q, p, P.c, P.min_sep));

    // 第 4 步：回溯线搜索。从 CCD 允许的最大步长开始试；能量比现在高就减半，直到能量不升。
    //   阶段 1 是 energy(x + alpha * p)；现在“加法”换成 integrate，算能量前先做 FK、找接触。
    //   alpha > 1e-12 是保险，防止无限循环。
    //   第二道网：有接触距离 <= 0 或越过限位时能量是 +无穷（contact_energy、barrier）。减半到底还是无穷，
    //   说明试探点穿进去了（CCD 漏了，比如浮点），这一次一步也不走，而不是接受一个 1e-12 的步长。
    const double e0 = inertia_energy(model, F, T, h) + contact_energy(model, C, kappa, P.dhat) +
                      friction_energy(model, F, FC, h, P.mu, P.eps_v) + limit_energy(model, q, L) +
                      drive_energy(model, D, q);
    double alpha = alpha_max;
    double e_trial = energy_at(integrate(model, q, p, alpha), kappa);
    while (e_trial > e0 && alpha > 1e-12) {
      alpha *= 0.5;
      e_trial = energy_at(integrate(model, q, p, alpha), kappa);
    }
    if (!std::isfinite(e_trial)) alpha = 0;
    track_full_step(alpha_max, alpha, &fstep);

    // 原始-对偶障碍：对偶变量的 Newton 更新（互补条件线性化，Delta d = r . p 取整步的 Newton 方向），
    //   夹在 >= 0.01 lambda（fraction-to-boundary）；这一步里没见过的接触从 F 开始。
    if (P.primal_dual) {
      std::map<PdKey, double> next;
      for (const Contact& c : C) {
        double dlog = 0;
        const double Fc = pd_force(c, kappa, P.dhat, &dlog);
        if (!(Fc > 0)) continue;
        const auto found = pd_lambda.find(pd_key(c));
        const double lam = found != pd_lambda.end() ? found->second : Fc;
        const double dd = contact_row(model, F, c).dot(p) / (2 * c.d);
        next[pd_key(c)] = std::max(lam + (Fc - lam) + lam * dlog * dd, 0.01 * lam);
      }
      pd_lambda.swap(next);
    }

    // 第 5 步：记录，然后更新。阶段 1 是 x = x + alpha * p。
    if (trace != nullptr) trace->push_back({e0, size, alpha_max, alpha, min_distance(model, F), pcg_iters});
    // 调试：CCD 保证的是每次迭代的轨迹不相交。沿这一次迭代的真实轨迹采样，确认这一点（测试用）。
    if (P.check_iterates) iterate_min = std::min(iterate_min, trajectory_min_distance(model, q, p, alpha, 16));
    q = integrate(model, q, p, alpha);

    // 第 6 步（阶段 9）：kappa 调整（书 §10.7，kappa_after_iterate）。
    if (P.adapt_kappa && state->kappa > 0) {
      kappa_after_iterate(P, krange.min, krange.max, min_distance(model, forward_kinematics(model, q)), state);
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
  r.capped = !converged && !semi_stopped;
  r.min_distance = min_distance(model, forward_kinematics(model, q));
  r.kappa = state->kappa;
  r.iterate_min_distance = iterate_min;
  r.min_limit_distance = min_limit_distance(model, q);
  r.curvature_dropped = curv_dropped;
  return r;
}

void kappa_after_iterate(const StepParams& P, double kmin, double kmax, double dmin, SolverState* state) {
  if (!P.adapt_kappa || state->kappa <= 0) return;
  // 冲击的那一步里 kappa 通常是一路减半：第一次接触时梯度平衡给的值偏大（夹在上限），间隙停在接近 dhat 的地方，
  //   那里障碍像一根软弹簧；减半把间隙压进障碍硬的那一段，隐式 Euler 才把回弹吃掉（落地的盒子回弹系数 0.002）。
  if (dmin < 1e-3 * P.dhat && dmin < state->dmin_prev) {
    state->kappa = std::min(2 * state->kappa, kmax);
  } else if (P.kappa_relax && dmin > 1e-2 * P.dhat) {
    state->kappa = std::max(0.5 * state->kappa, kmin);
  }
  state->dmin_prev = dmin;
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
