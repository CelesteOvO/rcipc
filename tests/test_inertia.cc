// 阶段 3：body 形式的惯性项（笔记 notes/03_inertia.md）。
//
// 测试分四组，和笔记的小节对应：
//   Mass.*      第 1 节：从 MuJoCo 读出的质量分布
//   Targets.*   第 2 节：惯性目标
//   Gradient.*  第 3 节：梯度，用有限差分检查
//   Hessian.*   第 4 节：Gauss-Newton Hessian = M(q) / h^2，和 MuJoCo 的 mj_fullM 比较
//   Curvature.* 阶段 13（notes/16_inertia_curvature.md）：Gauss-Newton 丢掉的那一项，加上以后 Hessian 是精确的
//
// 模型主要用 models/zoo.xml（四种关节、两棵树），每个测试在几个随机姿态上比较，随机数种子固定。
#include "rcipc/inertia.h"

#include <cmath>
#include <random>

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"

using namespace rcipc;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kH = 0.002;

struct Zoo {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/zoo.xml");
  mjData* d = mj_makeData(m.get());
  Model model = from_mujoco(m.get());
  ~Zoo() { mj_deleteData(d); }
};

// 随机姿态（同 test_kinematics.cc）。
VectorXd random_q(const Model& model, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  std::normal_distribution<double> n;
  VectorXd q = model.qpos0;
  for (const Joint& j : model.joints) {
    switch (j.type) {
      case JointType::kFree:
        q.segment<3>(j.qadr) = Vector3d(u(rng), u(rng), u(rng));
        quat_store(Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized(), &q[j.qadr + 3]);
        break;
      case JointType::kBall:
        quat_store(Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized(), &q[j.qadr]);
        break;
      case JointType::kHinge:
        q[j.qadr] = j.ref + kPi * u(rng);
        break;
      case JointType::kSlide:
        q[j.qadr] = j.ref + 0.3 * u(rng);
        break;
    }
  }
  return q;
}

VectorXd random_delta(int nv, std::mt19937& rng, double scale) {
  std::uniform_real_distribution<double> u(-scale, scale);
  VectorXd delta(nv);
  for (int i = 0; i < nv; ++i) delta[i] = u(rng);
  return delta;
}

// 一组“像时间步里那样”的目标：q_n 随机，q_{n-1} = q_n ⊕ (-h v)，v 随机（几 rad/s、几 m/s）。
InertiaTargets random_targets(const Model& model, const VectorXd& qn, std::mt19937& rng) {
  const VectorXd q_prev = integrate(model, qn, random_delta(model.nv, rng, 3.0), -kH);
  return inertia_targets(model, forward_kinematics(model, qn), forward_kinematics(model, q_prev), kH);
}

// 在 q 处的能量（先做 FK）。
double energy_at(const Model& model, const VectorXd& q, const InertiaTargets& T) {
  return inertia_energy(model, forward_kinematics(model, q), T, kH);
}

}  // namespace

// ---------- 第 1 节：质量分布 ----------

// 总质量和 MuJoCo 一致；每个刚体的惯性张量对称、特征值就是 MuJoCo 的主惯量；I = tr(S) 1 - S。
TEST(Mass, ReadsMassAndInertia) {
  Zoo z;
  double total = 0;
  for (int b = 1; b < z.model.nbody(); ++b) {
    const Body& B = z.model.bodies[b];
    total += B.mass;
    EXPECT_LT((B.inertia - B.inertia.transpose()).norm(), 1e-15) << "body " << b;
    Vector3d eig = Eigen::SelfAdjointEigenSolver<Matrix3d>(B.inertia).eigenvalues();
    Vector3d expect(z.m->body_inertia + 3 * b);
    std::sort(expect.data(), expect.data() + 3);
    EXPECT_LT((eig - expect).norm(), 1e-12 * expect.norm()) << "body " << b;
    EXPECT_LT((B.inertia - (B.S.trace() * Matrix3d::Identity() - B.S)).norm(), 1e-15) << "body " << b;
  }
  EXPECT_NEAR(total, mj_getTotalmass(z.m.get()), 1e-12);
}

// ---------- 第 2 节：惯性目标 ----------

// 静止、没有重力时，目标就是现在的位姿：能量和梯度都是 0。
TEST(Targets, AtRestWithoutGravityTheTargetIsTheCurrentPose) {
  Zoo z;
  z.model.gravity.setZero();
  std::mt19937 rng(40);
  const VectorXd q = random_q(z.model, rng);
  const Frames F = forward_kinematics(z.model, q);
  const InertiaTargets T = inertia_targets(z.model, F, F, kH);
  EXPECT_LT(inertia_energy(z.model, F, T, kH), 1e-20);
  EXPECT_LT(inertia_gradient(z.model, F, T, kH).norm(), 1e-9);
}

// 质心的目标是 2 c^n - c^{n-1} + g h^2：静止时（c^{n-1} = c^n）就是 c^n + g h^2，往下 h^2 * 9.81。
// 转动的目标是 2 R^n - R^{n-1}：静止时就是 R^n。
TEST(Targets, GravityMovesTheComTargetDown) {
  Zoo z;
  std::mt19937 rng(41);
  const Frames F = forward_kinematics(z.model, random_q(z.model, rng));
  const InertiaTargets T = inertia_targets(z.model, F, F, kH);
  for (int b = 1; b < z.model.nbody(); ++b) {
    EXPECT_LT((T.c[b] - body_com(z.model, F, b) - Vector3d(0, 0, -9.81 * kH * kH)).norm(), 1e-15);
    EXPECT_LT((T.A[b] - F.xquat[b].toRotationMatrix()).norm(), 1e-15);
  }
}

// ---------- 第 3 节：梯度 ----------

// 梯度的第 d 个分量 ≈ [E(q ⊕ eps e_d) - E(q ⊕ -eps e_d)] / (2 eps)。
TEST(Gradient, MatchesFiniteDifference) {
  Zoo z;
  const Model& M = z.model;
  std::mt19937 rng(42);
  const double eps = 1e-6;
  for (int i = 0; i < 5; ++i) {
    const VectorXd qn = random_q(M, rng);
    const InertiaTargets T = random_targets(M, qn, rng);
    const VectorXd q = integrate(M, qn, random_delta(M.nv, rng, 0.01));  // 迭代点离 q_n 不远
    const VectorXd g = inertia_gradient(M, forward_kinematics(M, q), T, kH);
    for (int d = 0; d < M.nv; ++d) {
      const VectorXd e = VectorXd::Unit(M.nv, d);
      const double fd = (energy_at(M, integrate(M, q, e, eps), T) - energy_at(M, integrate(M, q, e, -eps), T)) /
                        (2 * eps);
      EXPECT_NEAR(g[d], fd, 1e-6 * (1 + std::abs(fd))) << "pose " << i << ", dof " << d;
    }
  }
}

// ---------- 第 4 节：Hessian ----------

// Gauss-Newton Hessian 乘 h^2 就是 MuJoCo 的质量矩阵 M(q)（mj_fullM）。
TEST(Hessian, IsTheMassMatrixOverHSquared) {
  Zoo z;
  std::mt19937 rng(43);
  MatrixXd expect(z.model.nv, z.model.nv);
  for (int i = 0; i < 10; ++i) {
    const VectorXd q = random_q(z.model, rng);
    for (int k = 0; k < z.m->nq; ++k) z.d->qpos[k] = q[k];
    mj_fwdPosition(z.m.get(), z.d);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M(expect.data(), z.model.nv,
                                                                                       z.model.nv);
    mj_fullM(z.m.get(), z.d, M.data());
    const MatrixXd H = inertia_hessian(z.model, forward_kinematics(z.model, q), kH);
    EXPECT_LT((H * kH * kH - M).norm(), 1e-12 * M.norm()) << "pose " << i;
  }
}

// 当刚体正好在目标上（c = c~、R = A~）时，Gauss-Newton 丢掉的那一项是 0，Hessian 就是精确的：
// 它的第 d 列 ≈ [g(q ⊕ eps e_d) - g(q ⊕ -eps e_d)] / (2 eps)。
TEST(Hessian, IsExactWhenEveryBodyIsOnItsTarget) {
  Zoo z;
  const Model& M = z.model;
  std::mt19937 rng(44);
  const double eps = 1e-6;
  const VectorXd q = random_q(M, rng);
  const Frames F = forward_kinematics(M, q);
  InertiaTargets T;  // 目标直接取当前位姿
  T.c.resize(M.nbody());
  T.A.resize(M.nbody());
  for (int b = 0; b < M.nbody(); ++b) {
    T.c[b] = body_com(M, F, b);
    T.A[b] = F.xquat[b].toRotationMatrix();
  }
  const MatrixXd H = inertia_hessian(M, F, kH);
  for (int d = 0; d < M.nv; ++d) {
    const VectorXd e = VectorXd::Unit(M.nv, d);
    const VectorXd gp = inertia_gradient(M, forward_kinematics(M, integrate(M, q, e, eps)), T, kH);
    const VectorXd gm = inertia_gradient(M, forward_kinematics(M, integrate(M, q, e, -eps)), T, kH);
    EXPECT_LT((H.col(d) - (gp - gm) / (2 * eps)).norm(), 1e-5 * H.norm()) << "dof " << d;
  }
}

// ---------- 阶段 13：Gauss-Newton 丢掉的那一项 ----------

namespace {

// phi(delta) = E(q ⊕ delta) 在 delta = 0 处的 Hessian，用能量的二阶差分（同一个 chart，球关节、自由关节也对）：
//   H_ij ≈ [phi(e e_i + e e_j) - phi(e e_i - e e_j) - phi(-e e_i + e e_j) + phi(-e e_i - e e_j)] / (4 e^2)。
MatrixXd fd_hessian(const Model& M, const VectorXd& q, const InertiaTargets& T, double eps) {
  const int n = M.nv;
  MatrixXd H(n, n);
  const auto phi = [&](const VectorXd& d) { return energy_at(M, integrate(M, q, d), T); };
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      const VectorXd ei = eps * VectorXd::Unit(n, i), ej = eps * VectorXd::Unit(n, j);
      H(i, j) = H(j, i) = (phi(ei + ej) - phi(ei - ej) - phi(-ei + ej) + phi(-ei - ej)) / (4 * eps * eps);
    }
  return H;
}

}  // namespace

// M / h^2 + K 就是精确的 Hessian（四种关节）；只有 M / h^2 时差得多（检查这个测试本身有区分度）。
TEST(Curvature, MakesTheHessianExact) {
  Zoo z;
  const Model& M = z.model;
  std::mt19937 rng(45);
  for (int i = 0; i < 4; ++i) {
    const VectorXd qn = random_q(M, rng);
    const InertiaTargets T = random_targets(M, qn, rng);
    const VectorXd q = integrate(M, qn, random_delta(M.nv, rng, 0.05));
    const Frames F = forward_kinematics(M, q);
    const MatrixXd Mh = inertia_hessian(M, F, kH);
    const MatrixXd K = inertia_curvature(M, F, T, kH);
    const MatrixXd fd = fd_hessian(M, q, T, 1e-4);
    EXPECT_LT((K - K.transpose()).norm(), 1e-12 * K.norm()) << "pose " << i;
    EXPECT_LT((Mh + K - fd).norm(), 1e-6 * fd.norm()) << "pose " << i;
    EXPECT_GT((Mh - fd).norm(), 1e-3 * fd.norm()) << "pose " << i;  // K 不是可以忽略的
  }
}

// 刚体都在目标上时，残差力是 0，K = 0（和 Hessian.IsExactWhenEveryBodyIsOnItsTarget 对应）。
TEST(Curvature, IsZeroOnTarget) {
  Zoo z;
  const Model& M = z.model;
  std::mt19937 rng(46);
  const Frames F = forward_kinematics(M, random_q(M, rng));
  InertiaTargets T;
  T.c.resize(M.nbody());
  T.A.resize(M.nbody());
  for (int b = 0; b < M.nbody(); ++b) {
    T.c[b] = body_com(M, F, b);
    T.A[b] = F.xquat[b].toRotationMatrix();
  }
  EXPECT_LT(inertia_curvature(M, F, T, kH).norm(), 1e-9);
}

// 正定的才加：K = 0 时每棵树都加（H 不变），K = -M / h^2 时每棵树都不加。
TEST(Curvature, IsAddedOnlyWhereItKeepsTheBlockPositiveDefinite) {
  Zoo z;
  const Model& M = z.model;
  std::mt19937 rng(47);
  const MatrixXd Mh = inertia_hessian(M, forward_kinematics(M, random_q(M, rng)), kH);
  MatrixXd H = Mh;
  EXPECT_EQ(add_inertia_curvature(M, Mh, MatrixXd::Zero(M.nv, M.nv), &H), 0);
  EXPECT_EQ(add_inertia_curvature(M, Mh, -Mh, &H), M.ntree);
  EXPECT_LT((H - Mh).norm(), 1e-12 * Mh.norm());
  MatrixXd K = 0.5 * Mh;  // 正定的方向：加上
  EXPECT_EQ(add_inertia_curvature(M, Mh, K, &H), 0);
  EXPECT_LT((H - 1.5 * Mh).norm(), 1e-12 * Mh.norm());
}
