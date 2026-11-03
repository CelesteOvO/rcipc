#include "rcipc/step.h"

#include <Eigen/Cholesky>

#include "rcipc/inertia.h"
#include "rcipc/kinematics.h"

// 关节坐标里的时间步。推导见 notes/03_inertia.md 第 5 节，测试见 tests/test_step.cc。
namespace rcipc {

using Eigen::MatrixXd;
using Eigen::VectorXd;

StepResult step(const Model& model, const StepParams& P, const VectorXd& q0, const VectorXd& v0,
                std::vector<NewtonIterate>* trace) {
  // 一个时间步：用 Newton 法最小化惯性能量（笔记第 5 节）。
  // 结构和阶段 1 的 ipc1d::step 完全一样（可以打开 toys/ipc1d/ipc1d.cc 对照），区别都标在下面。
  const double h = P.h;

  // 准备（阶段 1 里是 xt = predicted(P, x0, v0)）。
  //   上一步的位置 q_{n-1} = q_n ⊕ (-h v_n)：我们只存了 (q, v)，而惯性目标要用两个时刻的位姿，所以倒推一下。
  //   目标在整个时间步里不变，只算一次。
  const VectorXd q_prev = integrate(model, q0, v0, -h);
  const InertiaTargets T =
      inertia_targets(model, forward_kinematics(model, q0), forward_kinematics(model, q_prev), h);

  // 能量作为 q 的函数：先做 FK，再算惯性能量。线搜索里反复用。
  const auto energy_at = [&](const VectorXd& q) { return inertia_energy(model, forward_kinematics(model, q), T, h); };

  // 从上一步结束的位置出发。newton 记录做了几次迭代，循环结束以后还要返回它。
  VectorXd q = q0;
  int newton = 0;
  for (int it = 1; it <= P.max_newton; ++it) {
    newton = it;

    // 第 1 步：Newton 方向。阶段 1 是 p = -E' / E''，现在是解线性方程组 H p = -g。
    //   H = M(q) / h^2 是对称正定矩阵，LDLT 分解（一种 Cholesky 分解）解得又快又稳。
    const Frames F = forward_kinematics(model, q);
    const VectorXd g = inertia_gradient(model, F, T, h);
    const MatrixXd H = inertia_hessian(model, F, h);
    const VectorXd p = H.ldlt().solve(-g);

    // 第 2 步：收敛判断。max_i |p_i| / h 相当于“这次修正的最大速度”，小于 tol 就不再移动。
    //   这次只检查、不移动的迭代也算一次（newton 已经等于 it），所以自由落体是 2 次，静止时是 1 次。
    const double size = p.lpNorm<Eigen::Infinity>() / h;
    if (size < P.tol) break;

    // 第 3 步：CCD。现在没有接触，步长上限就是 1。阶段 4 会在这里加上 CCD。

    // 第 4 步：回溯线搜索。阶段 1 是 energy(x + alpha * p)；现在“加法”换成 integrate，算能量前先做 FK。
    //   如果走整步能量反而变高，就把步长减半再试，直到能量不升。alpha > 1e-12 是保险，防止无限循环。
    const double e0 = inertia_energy(model, F, T, h);
    double alpha = 1.0;
    while (energy_at(integrate(model, q, p, alpha)) > e0 && alpha > 1e-12) {
      alpha *= 0.5;
    }

    // 第 5 步：记录，然后更新。阶段 1 是 x = x + alpha * p。
    if (trace != nullptr) trace->push_back({e0, size, alpha});
    q = integrate(model, q, p, alpha);
  }

  // 速度 v = (q ⊖ q0) / h，阶段 1 是 (x - x0) / h。
  return {q, differentiate(model, q0, q) / h, newton};
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
