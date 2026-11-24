// 阶段 8：滞后的光滑化 Coulomb 摩擦（笔记 notes/08_friction.md）。
//
//   Smoothing.*  第 2 节：f0、f1 的性质
//   Lag.*        第 1 节：步首冻结的法向力和切向基
//   Gradient.*   第 3 节：摩擦能量的梯度和有限差分一致（|u| 在 eps_v 之下和之上），Hessian 对称半正定
//   Slope.*      第 5 节：斜面上的盒子，tan theta < mu 静止、> mu 匀加速下滑；滑动的盒子被摩擦刹住
#include "rcipc/friction.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// 把重力转到“斜面”上：盒子沿 +x 下滑。
Model slope(double theta) {
  Model M = load("slope.xml");
  M.gravity = 9.81 * Vector3d(std::sin(theta), 0, -std::cos(theta));
  return M;
}

// 先让盒子在平地上静止几步，让 kappa 初始化、接触稳定下来，再开始计时。
struct Settled {
  VectorXd q, v;
  SolverState S;
};
Settled settle(const Model& M, const StepParams& P, int steps) {
  Settled s;
  s.q = M.qpos0;
  s.v = VectorXd::Zero(6);
  Model flat = M;
  flat.gravity = Vector3d(0, 0, -9.81);
  for (int i = 0; i < steps; ++i) {
    const StepResult r = step(flat, P, s.q, s.v, &s.S);
    s.q = r.q;
    s.v = r.v;
  }
  return s;
}

}  // namespace

// ---------- 第 2 节：光滑化 ----------

// f1 = f0'；f1(0) = 0；在 eps 处 f0、f1 连续、f1' = 0；eps 之外 f0 = y、f1 = 1；f1 / y 在 0 处是 2 / eps。
TEST(Smoothing, F0AndF1AreConsistent) {
  const double eps = 1e-3, d = 1e-7;
  for (double y : {1e-4, 5e-4, 9e-4, 1e-3, 2e-3}) {
    const double fd = (friction_f0(y + d, eps) - friction_f0(y - d, eps)) / (2 * d);
    EXPECT_NEAR(friction_f1(y, eps), fd, 1e-6) << "y " << y;
    EXPECT_NEAR(friction_f1_over_y(y, eps), friction_f1(y, eps) / y, 1e-9);
  }
  EXPECT_EQ(friction_f1(0, eps), 0.0);
  EXPECT_NEAR(friction_f1_over_y(0, eps), 2 / eps, 1e-9);
  EXPECT_NEAR(friction_f0(eps, eps), eps, 1e-15);
  EXPECT_NEAR(friction_f1(eps, eps), 1.0, 1e-15);
  EXPECT_NEAR(friction_f1(eps - 1e-9, eps), 1.0, 1e-6);  // f1'(eps) = 0
  EXPECT_EQ(friction_f0(2 * eps, eps), 2 * eps);
}

// ---------- 第 1 节：滞后量 ----------

// 静止在平地上的盒子：4 个摩擦接触，法向力之和等于重力，切向基垂直于法向且互相垂直；kappa = 0 时没有摩擦接触。
TEST(Lag, NormalForcesSumToTheWeight) {
  const Model M = load("slope.xml");
  StepParams P;
  const Settled s = settle(M, P, 400);
  const Frames F = forward_kinematics(M, s.q);
  const std::vector<Contact> C = find_contacts(M, F, P.dhat);
  const std::vector<FrictionContact> FC = lag_friction(M, F, C, s.S.kappa, P.dhat);
  ASSERT_EQ(FC.size(), 4u);
  double sum = 0;
  for (const FrictionContact& f : FC) {
    sum += f.lambda;
    EXPECT_GT(f.lambda, 0);
    EXPECT_NEAR(f.t1.dot(f.contact.n), 0.0, 1e-12);
    EXPECT_NEAR(f.t2.dot(f.contact.n), 0.0, 1e-12);
    EXPECT_NEAR(f.t1.dot(f.t2), 0.0, 1e-12);
    EXPECT_NEAR(f.t1.norm(), 1.0, 1e-12);
    EXPECT_NEAR(f.t2.norm(), 1.0, 1e-12);
    EXPECT_NEAR(friction_velocity(M, F, f, P.h).norm(), 0.0, 1e-15);  // 还没动
  }
  EXPECT_NEAR(sum, 9.81, 2e-3);
  EXPECT_TRUE(lag_friction(M, F, C, 0.0, P.dhat).empty());
}

// ---------- 第 3 节：梯度和 Hessian ----------

// 摩擦能量的梯度和有限差分一致：盒子相对步首的位置平移、转动一点，|u| 既有在 eps_v 之下的也有之上的。
TEST(Gradient, MatchesFiniteDifference) {
  const Model M = load("slope.xml");
  StepParams P;
  const Settled s = settle(M, P, 400);
  const Frames F0 = forward_kinematics(M, s.q);
  const std::vector<FrictionContact> FC = lag_friction(M, F0, find_contacts(M, F0, P.dhat), s.S.kappa, P.dhat);
  ASSERT_EQ(FC.size(), 4u);
  // 有限差分的步长要远小于位移本身（f0 在 eps_v 以内是三次的；位移 / h 才 0.1 eps_v），所以取 1e-9。
  const double eps = 1e-9;
  for (double scale : {0.2e-6, 3e-6}) {  // 位移 / h 分别约 0.1 eps_v 和 1.5 eps_v
    VectorXd d = VectorXd::Zero(6);
    d[0] = scale;
    d[1] = -0.5 * scale;
    d[5] = 2 * scale;  // 绕 z 转一点：4 个角的切向速度各不相同
    const VectorXd q = integrate(M, s.q, d);
    const Frames F = forward_kinematics(M, q);
    double umin = 1e9, umax = 0;
    for (const FrictionContact& f : FC) {
      const double u = friction_velocity(M, F, f, P.h).norm();
      umin = std::min(umin, u);
      umax = std::max(umax, u);
    }
    EXPECT_GT(umin, 0.0);
    const VectorXd g = friction_gradient(M, F, FC, P.h, P.mu, P.eps_v);
    for (int k = 0; k < 6; ++k) {
      const VectorXd e = VectorXd::Unit(6, k);
      const double fd = (friction_energy(M, forward_kinematics(M, integrate(M, q, e, eps)), FC, P.h, P.mu, P.eps_v) -
                         friction_energy(M, forward_kinematics(M, integrate(M, q, e, -eps)), FC, P.h, P.mu, P.eps_v)) /
                        (2 * eps);
      EXPECT_NEAR(g[k], fd, 1e-5 * (1 + std::abs(fd))) << "scale " << scale << ", dof " << k;
    }
    const MatrixXd H = friction_hessian(M, F, FC, P.h, P.mu, P.eps_v);
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    const VectorXd eig = Eigen::SelfAdjointEigenSolver<MatrixXd>(H).eigenvalues();
    EXPECT_GE(eig.minCoeff(), -1e-9 * (1 + eig.maxCoeff()));
  }
}

// ---------- 第 5 节：斜面 ----------

// tan theta = 0.3 < mu = 0.5：盒子静止在斜面上。光滑化让它以 f1(u) mu = tan theta 对应的速度蠕动：
// f1(u) = 0.6，u ≈ 0.37 eps_v；1 s 内位移不到 eps_v。
TEST(Slope, SticksBelowTheFrictionAngle) {
  const double theta = std::atan(0.3);
  const Model M = slope(theta);
  StepParams P;
  P.mu = 0.5;
  Settled s = settle(M, P, 400);
  const double x0 = s.q[0];
  for (int i = 0; i < 500; ++i) {
    const StepResult r = step(M, P, s.q, s.v, &s.S);
    EXPECT_GT(r.min_distance, 0.0);
    s.q = r.q;
    s.v = r.v;
  }
  const double creep = 0.367 * P.eps_v;  // 2y - y^2 = 0.6 的解
  EXPECT_NEAR(s.v[0], creep, 0.2 * creep);
  EXPECT_LT(s.q[0] - x0, 1.0 * P.eps_v);
  EXPECT_LT(std::abs(s.v[1]), 1e-6);
  EXPECT_LT(s.v.tail<3>().norm(), 1e-3);  // 没有翻倒
}

// tan theta = 0.3 > mu = 0.2：匀加速下滑，a = g (sin theta - mu cos theta) = 0.938 m/s^2。
TEST(Slope, SlidesAboveTheFrictionAngle) {
  const double theta = std::atan(0.3);
  const Model M = slope(theta);
  StepParams P;
  P.mu = 0.2;
  Settled s = settle(M, P, 400);
  const double T = 1.0;
  for (int i = 0; i < int(T / P.h + 0.5); ++i) {
    const StepResult r = step(M, P, s.q, s.v, &s.S);
    EXPECT_GT(r.min_distance, 0.0);
    s.q = r.q;
    s.v = r.v;
  }
  const double a = 9.81 * (std::sin(theta) - P.mu * std::cos(theta));
  EXPECT_NEAR(s.v[0], a * T, 0.05 * a * T);
  EXPECT_LT(s.v.tail<3>().norm(), 1e-2);
}

// 平地上以 1 m/s 滑动的盒子：摩擦让它以 mu g 减速，0.5 s 内停下（1 / (0.2 * 9.81) = 0.51 s）。
TEST(Slope, SlidingBoxIsStoppedByFriction) {
  const Model M = load("slope.xml");
  StepParams P;
  P.mu = 0.2;
  Settled s = settle(M, P, 400);
  s.v[0] = 1.0;
  double t_stop = -1;
  for (int i = 0; i < 400; ++i) {
    const StepResult r = step(M, P, s.q, s.v, &s.S);
    EXPECT_GT(r.min_distance, 0.0);
    s.q = r.q;
    s.v = r.v;
    if (t_stop < 0 && std::abs(s.v[0]) < 2 * P.eps_v) t_stop = (i + 1) * P.h;
  }
  EXPECT_GT(t_stop, 0.0);
  EXPECT_NEAR(t_stop, 1.0 / (P.mu * 9.81), 0.1);
  EXPECT_LT(std::abs(s.v[0]), 2 * P.eps_v);
}

// 关掉摩擦（mu = 0）：斜面上的盒子以 g sin theta 加速，和没有摩擦的解析解一致。
TEST(Slope, NoFrictionSlidesFreely) {
  const double theta = std::atan(0.3);
  const Model M = slope(theta);
  StepParams P;
  P.mu = 0.0;
  Settled s = settle(M, P, 400);
  for (int i = 0; i < 500; ++i) {
    const StepResult r = step(M, P, s.q, s.v, &s.S);
    s.q = r.q;
    s.v = r.v;
  }
  EXPECT_NEAR(s.v[0], 9.81 * std::sin(theta) * 1.0, 0.03 * 9.81 * std::sin(theta));
}
