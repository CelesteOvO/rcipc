#include "rcipc/step.h"

#include <Eigen/Cholesky>

#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/friction.h"
#include "rcipc/inertia.h"
#include "rcipc/kinematics.h"

// 关节坐标里的时间步。推导见 notes/03_inertia.md 第 5 节（惯性）和 notes/04_ground.md 第 6 节（接触），
// 测试见 tests/test_step.cc 和 tests/test_contact.cc。
namespace rcipc {

using Eigen::MatrixXd;
using Eigen::VectorXd;

StepResult step(const Model& model, const StepParams& P, const VectorXd& q0, const VectorXd& v0, SolverState* state,
                std::vector<NewtonIterate>* trace) {
  // 一个时间步：用 Newton 法最小化 E(q) = 惯性项 + kappa * 障碍项。
  // 结构和阶段 1 的 ipc1d::step 完全一样（可以打开 toys/ipc1d/ipc1d.cc 对照），区别都标在下面。
  const double h = P.h;
  SolverState local;
  if (state == nullptr) state = &local;

  // 准备（阶段 1 里是 xt = predicted(P, x0, v0)）。
  //   上一步的位置 q_{n-1} = q_n ⊕ (-h v_n)：我们只存了 (q, v)，而惯性目标要用两个时刻的位姿，所以倒推一下。
  //   目标在整个时间步里不变，只算一次。
  const VectorXd q_prev = integrate(model, q0, v0, -h);
  const Frames F0 = forward_kinematics(model, q0);
  const InertiaTargets T = inertia_targets(model, F0, forward_kinematics(model, q_prev), h);
  // 步首的惯性梯度 m (c0 - c~) / h^2 = -m (v / h + g)：这一步要“刹住”的动量加重力。kappa 的初值用它来平衡
  // （笔记第 4 节）。不能用当前迭代点的惯性梯度：接触常常在一个已经收敛的迭代点上才第一次出现，那里它是 0。
  const VectorXd g_e0 = inertia_gradient(model, F0, T, h);

  // 阶段 8：步首把活跃的接触冻结成摩擦接触（法向力、切向基、模板都取步首的值，这一步里当常数）。
  //   kappa 还没初始化（第一次接触的那一步）时没有摩擦：这一步里新出现的接触要到下一步才有摩擦。
  const std::vector<FrictionContact> FC =
      P.mu > 0 ? lag_friction(model, F0, find_contacts(model, F0, P.dhat), state->kappa, P.dhat)
               : std::vector<FrictionContact>();

  // 能量作为 q 的函数：先做 FK，重新找活跃的接触（换了位置，活跃的接触可能变了），再把两项加起来。
  // 线搜索里反复用。
  const auto energy_at = [&](const VectorXd& q, double kappa) {
    const Frames F = forward_kinematics(model, q);
    return inertia_energy(model, F, T, h) + contact_energy(model, find_contacts(model, F, P.dhat), kappa, P.dhat) +
           friction_energy(model, F, FC, h, P.mu, P.eps_v);
  };

  // 从上一步结束的位置出发（它是不相交的）。newton 记录做了几次迭代，循环结束以后还要返回它。
  VectorXd q = q0;
  int newton = 0;
  for (int it = 1; it <= P.max_newton; ++it) {
    newton = it;

    // 第 0 步：当前点的 FK 和活跃接触。第一次出现接触时初始化 kappa（梯度平衡，笔记第 4 节）。
    const Frames F = forward_kinematics(model, q);
    const std::vector<Contact> C = find_contacts(model, F, P.dhat);
    if (!C.empty() && state->kappa == 0) state->kappa = initial_kappa(model, F, C, g_e0, P.dhat);
    const double kappa = state->kappa;
    const VectorXd g_e = inertia_gradient(model, F, T, h);

    // 第 1 步：Newton 方向。阶段 1 是 p = -E' / E''，现在是解线性方程组 H p = -g。
    //   g = 惯性梯度 + 接触梯度，H = M(q) / h^2 + sum kappa b'' r^T r：两项都对称正定（半正定），
    //   LDLT 分解（一种 Cholesky 分解）解得又快又稳。
    //   阶段 6 再加上曲率上界（对角）：补回 Gauss-Newton 丢掉的轨迹曲率项的上界，不改梯度。
    //   阶段 8 再加上摩擦的梯度和 Hessian（每个摩擦接触两条切向行，权重 c / h）。
    const VectorXd g = g_e + contact_gradient(model, F, C, kappa, P.dhat) + friction_gradient(model, F, FC, h, P.mu, P.eps_v);
    const MatrixXd H = inertia_hessian(model, F, h) + contact_hessian(model, F, C, kappa, P.dhat) +
                       friction_hessian(model, F, FC, h, P.mu, P.eps_v) +
                       contact_curvature(model, q, C, kappa, P.dhat, P.curv_scale);
    const VectorXd p = H.ldlt().solve(-g);

    // 第 2 步：收敛判断。max_i |p_i| / h 相当于“这次修正的最大速度”，小于 tol 就不再移动。
    //   这次只检查、不移动的迭代也算一次（newton 已经等于 it），所以自由落体是 2 次，静止时是 1 次。
    const double size = p.lpNorm<Eigen::Infinity>() / h;
    if (size < P.tol) break;

    // 第 3 步：CCD 定上限。
    //   阶段 4 的直弦 CCD 只看 s = 0 和 s = 1 两端的顶点位置，保证沿弦走到 alpha_max 之前每个顶点离平面
    //   都还有 (1 - c) 倍的距离；阶段 5 的曲线 CCD 把弧线偏离弦的余量（矢高）也算进去。没有平面时都返回 1。
    const double alpha_max =
        P.ccd == CcdMode::kCurved
            ? curved_ccd(model, q, p, P.dhat, P.c)
            : chord_ccd_ground(model, F, forward_kinematics(model, integrate(model, q, p, 1.0)), P.c);

    // 第 4 步：回溯线搜索。从 CCD 允许的最大步长开始试；能量比现在高就减半，直到能量不升。
    //   阶段 1 是 energy(x + alpha * p)；现在“加法”换成 integrate，算能量前先做 FK、找接触。
    //   alpha > 1e-12 是保险，防止无限循环。
    const double e0 = inertia_energy(model, F, T, h) + contact_energy(model, C, kappa, P.dhat) +
                      friction_energy(model, F, FC, h, P.mu, P.eps_v);
    double alpha = alpha_max;
    while (energy_at(integrate(model, q, p, alpha), kappa) > e0 && alpha > 1e-12) {
      alpha *= 0.5;
    }

    // 第 5 步：记录，然后更新。阶段 1 是 x = x + alpha * p。
    if (trace != nullptr) trace->push_back({e0, size, alpha_max, alpha, min_distance(model, F)});
    q = integrate(model, q, p, alpha);
  }

  // 速度 v = (q ⊖ q0) / h，阶段 1 是 (x - x0) / h。顺便记下结束位置上的最小距离，给调用者检查不穿透。
  StepResult r;
  r.q = q;
  r.v = differentiate(model, q0, q) / h;
  r.newton = newton;
  r.kappa = state->kappa;
  r.min_distance = min_distance(model, forward_kinematics(model, q));
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
