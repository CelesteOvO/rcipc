// 阶段 6：曲率上界（笔记 notes/06_curvature.md）。
//
//   Chain.*      链上的关节和静态力臂
//   Majoriser.*  上界的结构；它加上 Gauss-Newton 项以后在每个方向上都不比真实 Hessian 弯得少
//   Lever.*      顶杆：Gauss-Newton 第一步就被 CCD 截断、把间隙挖到极小点以下；加上上界后间隙单调；收敛点相同
#include "rcipc/contact.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rcipc/ccd.h"
#include "rcipc/chain.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// zoo.xml 的随机姿态，躯干抬高到地面之上（同 test_contact.cc）。
VectorXd random_q(const Model& model, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  std::normal_distribution<double> n;
  VectorXd q = model.qpos0;
  for (int attempt = 0; attempt < 100; ++attempt) {  // 相交的姿态重新抽（阶段 7 起刚体之间也算）
  for (const Joint& j : model.joints) {
    switch (j.type) {
      case JointType::kFree:
        q.segment<3>(j.qadr) = Vector3d(u(rng), u(rng), 1.5 + 0.3 * u(rng));
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
  if (min_distance(model, forward_kinematics(model, q)) > 1e-3) break;
  }
  return q;
}

}  // namespace

// ---------- 链 ----------

// zoo.xml 的 finger：链是 root（自由关节的转动）、shoulder（铰链）、elbow（滑动）、knuckle（铰链），从根到叶；
// 静态力臂从根到叶递减，每一段都等于“更深一段加上中间的固定偏移”。
TEST(Chain, FingerChainRootFirst) {
  const Model M = load("zoo.xml");
  int finger = -1;
  for (int b = 1; b < M.nbody(); ++b) {
    if (M.bodies[b].joint >= 0 && M.joints[M.bodies[b].joint].type == JointType::kHinge &&
        M.joints[M.bodies[b].joint].axis.isApprox(Vector3d::UnitZ())) {
      finger = b;
    }
  }
  ASSERT_GT(finger, 0);
  const std::vector<ChainJoint> chain = chain_joints(M, M.qpos0, finger);
  ASSERT_EQ(chain.size(), 4u);
  EXPECT_TRUE(chain[0].rotation);
  EXPECT_EQ(chain[0].nv, 3);
  EXPECT_EQ(chain[0].dadr, M.joints[chain[0].joint].dadr + 3);  // 自由关节的转动是后 3 个
  EXPECT_TRUE(chain[1].rotation);
  EXPECT_FALSE(chain[2].rotation);  // elbow 滑动
  EXPECT_TRUE(chain[3].rotation);
  for (size_t i = 1; i < chain.size(); ++i) EXPECT_GT(chain[i - 1].reach, chain[i].reach);
  // 最深的关节：力臂 = |jnt_pos| + 刚体半径。
  EXPECT_NEAR(chain[3].reach, M.joints[chain[3].joint].pos.norm() + M.bodies[finger].radius, 1e-12);
  // 静态力臂确实不小于真实距离（任意姿态）。
  std::mt19937 rng(70);
  for (int i = 0; i < 10; ++i) {
    const VectorXd q = random_q(M, rng);
    const Frames F = forward_kinematics(M, q);
    const std::vector<ChainJoint> ch = chain_joints(M, q, finger);
    for (const ChainJoint& cj : ch) {
      const Vector3d anchor = F.dof_anchor[cj.dadr];
      for (const Vertex& v : M.vertices) {
        if (v.body != finger) continue;
        EXPECT_LE((world_point(F, finger, v.local) - anchor).norm() + v.radius, cj.reach * (1 + 1e-12));
      }
    }
  }
}

// ---------- 上界的结构 ----------

// 只在对角线上、只在活跃顶点所在链的转动自由度上非零；scale = 0 时为 0；和 scale 成正比。
TEST(Majoriser, DiagonalOnTheChainsRotations) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(71);
  const VectorXd q = random_q(M, rng);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, 1.0);
  ASSERT_FALSE(C.empty());
  const MatrixXd H = contact_curvature(M, q, C, 3.0, 1.0, 1.0);
  EXPECT_LT((H - MatrixXd(H.diagonal().asDiagonal())).norm(), 1e-15);
  EXPECT_EQ(contact_curvature(M, q, C, 3.0, 1.0, 0.0).norm(), 0.0);
  EXPECT_LT((contact_curvature(M, q, C, 3.0, 1.0, 0.3) - 0.3 * H).norm(), 1e-12 * H.norm());
  for (int d = 0; d < M.nv; ++d) {
    if (!M.dof_rot[d]) EXPECT_EQ(H(d, d), 0.0) << "dof " << d;  // 平移、滑动自由度不加
    else EXPECT_GE(H(d, d), 0.0);
  }
}

// 书 §8.4 第 177 页的不等式：delta^T (真实 Hessian) delta <= delta^T (kappa b'' r^T r + 上界) delta，对任意 delta。
// 真实 Hessian 用接触梯度的中心差分；delta 的滑动分量置 0（推导只覆盖转动之间的二阶导数）。
TEST(Majoriser, BoundsTheTrueHessianAlongAnyDirection) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(72);
  std::uniform_real_distribution<double> u(-1, 1);
  const double dhat = 1.0, kappa = 1.0, eps = 1e-5;
  for (int i = 0; i < 5; ++i) {
    const VectorXd q = random_q(M, rng);
    const Frames F = forward_kinematics(M, q);
    const std::vector<Contact> C = find_contacts(M, F, dhat);
    ASSERT_FALSE(C.empty());
    const MatrixXd Hgn = contact_hessian(M, F, C, kappa, dhat);
    const MatrixXd Hcurv = contact_curvature(M, q, C, kappa, dhat, 1.0);
    for (int t = 0; t < 10; ++t) {
      VectorXd delta(M.nv);
      for (int d = 0; d < M.nv; ++d) delta[d] = u(rng);
      for (const Joint& j : M.joints) {
        if (j.type == JointType::kSlide) delta[j.dadr] = 0;
      }
      // 真实的二阶方向导数：沿 delta 走 ±eps，接触梯度的差。
      const auto grad = [&](double s) {
        const VectorXd qq = integrate(M, q, delta, s);
        const Frames FF = forward_kinematics(M, qq);
        return contact_gradient(M, FF, find_contacts(M, FF, dhat), kappa, dhat);
      };
      const double exact = delta.dot(grad(eps) - grad(-eps)) / (2 * eps);
      const double model = delta.dot((Hgn + Hcurv) * delta);
      EXPECT_LE(exact, model + 1e-6 * (1 + std::abs(model))) << "pose " << i << ", direction " << t;
    }
  }
}

// ---------- 顶杆 ----------

// 顶杆（lever.xml）：球在弧线最高点、离地 0.5 mm，以 1 rad/s、h = 0.1 转（惯性弱，极小点由障碍决定，间隙约 0.19 mm）。
//   纯 Gauss-Newton（scale 0）：在最高点接触行 r = 0，Newton 看不到障碍，第一步就被 CCD 截断，
//     间隙被挖到 0.13 mm（极小点以下），再慢慢爬回来。
//   加上上界（scale 1）：模型知道转动会压进地面，间隙单调地落到极小点，从不低于它。
//   两者收敛到同一个点（上界不改梯度）。
TEST(Lever, GaussNewtonOvershootsTheGapAndTheMajoriserDoesNot) {
  const Model M = load("lever.xml");
  StepParams P;
  P.h = 0.1;
  P.adapt_kappa = false;  // 两次运行的迭代次数不同，kappa 调整会让它们的极小点不同；这里只看 Newton 模型
  VectorXd q = VectorXd::Zero(1), v = VectorXd::Zero(1);
  v[0] = 1.0;
  EXPECT_NEAR(min_distance(M, forward_kinematics(M, q)), 0.5e-3, 1e-12);

  P.curv_scale = 0.0;
  std::vector<NewtonIterate> t0;
  const StepResult r0 = step(M, P, q, v, nullptr, &t0);
  P.curv_scale = 1.0;
  std::vector<NewtonIterate> t1;
  const StepResult r1 = step(M, P, q, v, nullptr, &t1);

  // 同一个极小点（差别在收敛容差 tol * h = 1e-7 rad 的量级），间隙健康（远大于 0，远小于 dhat 的一半）。
  EXPECT_NEAR(r0.q[0], r1.q[0], 5e-6);  // PCG 的容差也在里面
  EXPECT_NEAR(r0.min_distance, r1.min_distance, 1e-7);
  EXPECT_GT(r0.min_distance, 1e-4);
  EXPECT_LT(r0.min_distance, 0.4e-3);

  // Gauss-Newton：第一步被 CCD 截断，间隙挖到极小点以下。
  ASSERT_GE(t0.size(), 2u);
  EXPECT_LT(t0[0].alpha_max, 1.0);
  EXPECT_LT(t0[1].min_distance, 0.7 * r0.min_distance);
  // 上界：每次迭代的间隙都不低于极小点间隙的 90%（阶段 9 把障碍行的权重换成精确值以后，尾部有一点 5% 的回摆；
  // 纯 Gauss-Newton 的权重下它是严格单调的）。
  for (size_t k = 0; k < t1.size(); ++k) {
    EXPECT_GE(t1[k].min_distance, 0.9 * r1.min_distance) << "iteration " << k;
  }
  EXPECT_LT(t1[1].min_distance, t1[0].min_distance);  // 第一步就朝极小点走，而不是先挖过头
}

// 任何 scale 都不穿透、都收敛到同一个点：顶杆以 20 rad/s 撞向地面（惯性强，球被压到 1 um），走 3 步。
TEST(Lever, EveryScaleReachesTheSamePointWithoutPenetration) {
  const Model M = load("lever.xml");
  VectorXd q_end[3];
  for (int i = 0; i < 3; ++i) {
    StepParams P;
    P.h = 0.01;
    P.adapt_kappa = false;
    P.curv_scale = i == 0 ? 0.0 : (i == 1 ? 0.3 : 1.0);
    SolverState S;
    VectorXd q = VectorXd::Zero(1), v = VectorXd::Zero(1);
    v[0] = 20.0;
    for (int k = 0; k < 3; ++k) {
      const StepResult r = step(M, P, q, v, &S);
      EXPECT_GT(trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 64), 0.0);
      EXPECT_LT(r.newton, P.max_newton);
      q = r.q;
      v = r.v;
    }
    q_end[i] = q;
  }
  EXPECT_NEAR(q_end[0][0], q_end[2][0], 1e-6);
  EXPECT_NEAR(q_end[1][0], q_end[2][0], 1e-6);
}
