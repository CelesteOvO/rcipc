// 阶段 4：地面接触（笔记 notes/04_ground.md）。
//
//   Geometry.*  第 1 节：从 MJCF 读出的顶点和平面
//   Row.*       第 2 节：接触行 r = d(d^2)/d(delta)，用有限差分检查
//   Energy.*    第 3 节：能量的梯度，用有限差分检查；Hessian 对称半正定
//   Kappa.*     第 4 节：kappa 的初值
//   Drop.*      第 6 节：盒子落地，永不穿透，静止时力平衡
#include "rcipc/contact.h"

#include <cmath>
#include <random>

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "rcipc/barrier.h"
#include "rcipc/ccd.h"
#include "rcipc/inertia.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// zoo.xml 的随机姿态，但躯干抬高到 1.2–1.8 m，保证所有顶点都在地面之上（能量在 d <= 0 处没有定义）；
// 阶段 7 起刚体之间也不能相交，所以相交的姿态重新抽。
VectorXd random_q_above_floor(const Model& model, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  std::normal_distribution<double> n;
  VectorXd q = model.qpos0;
  for (int attempt = 0; attempt < 100; ++attempt) {
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

// ---------- 第 1 节：几何 ----------

// 盒子有 8 个角（半径 0），地面的法向是 +z、偏移 0。
TEST(Geometry, BoxCornersAndFloor) {
  const Model M = load("drop.xml");
  ASSERT_EQ(M.nvertex(), 8);
  ASSERT_EQ(M.planes.size(), 1u);
  EXPECT_LT((M.planes[0].normal - Vector3d::UnitZ()).norm(), 1e-15);
  EXPECT_NEAR(M.planes[0].offset, 0.0, 1e-15);
  for (const Vertex& v : M.vertices) {
    EXPECT_EQ(v.body, 1);
    EXPECT_EQ(v.radius, 0.0);
    EXPECT_LT((v.local.cwiseAbs() - Vector3d(0.05, 0.1, 0.2)).norm(), 1e-15);
  }
}

// zoo.xml：盒子 8 角、球 1 点、胶囊各 2 点，半径和 MJCF 一致；胶囊 fromto 的端点正好是顶点。
TEST(Geometry, SphereAndCapsuleCarryTheirRadius) {
  const Model M = load("zoo.xml");
  // torso box 8 + head sphere 1 + arm capsule 2 + forearm capsule 2 + hand box 8 + finger capsule 2
  // + post capsule 2 + bob sphere 1 = 26
  EXPECT_EQ(M.nvertex(), 26);
  int spheres = 0, capsule_ends = 0;
  for (const Vertex& v : M.vertices) {
    if (v.radius == 0.08 || v.radius == 0.06) ++spheres;
    if (v.radius == 0.04 || v.radius == 0.03 || v.radius == 0.01 || v.radius == 0.02) ++capsule_ends;
  }
  EXPECT_EQ(spheres, 2);
  EXPECT_EQ(capsule_ends, 8);
  // post 的胶囊 fromto="0 0 0 0 0 -0.4"：端点在 (0,0,0) 和 (0,0,-0.4)（post 刚体坐标系）。
  int found = 0;
  for (const Vertex& v : M.vertices) {
    if (v.radius == 0.02 && (v.local.norm() < 1e-12 || (v.local - Vector3d(0, 0, -0.4)).norm() < 1e-12)) ++found;
  }
  EXPECT_EQ(found, 2);
}

// 距离 = n . x - offset - radius：落地盒子静止在 qpos0 时，下面 4 个角离地 0.5 m，上面 4 个角 0.9 m。
TEST(Geometry, DistanceSubtractsTheRadius) {
  const Model M = load("drop.xml");
  const Frames F = forward_kinematics(M, M.qpos0);
  int low = 0, high = 0;
  for (int v = 0; v < M.nvertex(); ++v) {
    const double d = ground_distance(M, F, v, 0);
    if (std::abs(d - 0.5) < 1e-12) ++low;
    if (std::abs(d - 0.9) < 1e-12) ++high;
  }
  EXPECT_EQ(low, 4);
  EXPECT_EQ(high, 4);
  EXPECT_NEAR(min_distance(M, F), 0.5, 1e-12);
  EXPECT_TRUE(find_contacts(M, F, 1e-3).empty());
  EXPECT_EQ(find_contacts(M, F, 0.6).size(), 4u);

  const Model Z = load("zoo.xml");
  const Frames FZ = forward_kinematics(Z, Z.qpos0);
  for (int v = 0; v < Z.nvertex(); ++v) {
    const Vertex& V = Z.vertices[v];
    const double expect = world_point(FZ, V.body, V.local).z() - V.radius;
    EXPECT_NEAR(ground_distance(Z, FZ, v, 0), expect, 1e-15);
  }
}

// ---------- 第 2 节：接触行 ----------

// r 的第 d 个分量 ≈ [d^2(q ⊕ eps e_d) - d^2(q ⊕ -eps e_d)] / (2 eps)。
TEST(Row, MatchesFiniteDifference) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(50);
  const double eps = 1e-6;
  for (int i = 0; i < 5; ++i) {
    const VectorXd q = random_q_above_floor(M, rng);
    const Frames F = forward_kinematics(M, q);
    std::vector<Contact> C;  // dhat 大到所有顶点都活跃；这里只看顶点–平面（刚体之间的在 test_pairs.cc）
    for (const Contact& c : find_contacts(M, F, 10.0)) {
      if (c.type == ContactType::kGround) C.push_back(c);
    }
    ASSERT_EQ(C.size(), static_cast<size_t>(M.nvertex()));
    for (const Contact& c : C) {
      const RowVectorXd r = contact_row(M, F, c);
      for (int d = 0; d < M.nv; ++d) {
        const VectorXd e = VectorXd::Unit(M.nv, d);
        const auto x2 = [&](double s) {
          const double dist = ground_distance(M, forward_kinematics(M, integrate(M, q, e, s)), c.verts[0], c.plane);
          return dist * dist;
        };
        const double fd = (x2(eps) - x2(-eps)) / (2 * eps);
        EXPECT_NEAR(r[d], fd, 1e-7 * (1 + std::abs(fd))) << "pose " << i << ", vertex " << c.verts[0] << ", dof " << d;
      }
    }
  }
}

// 落地盒子：r 对竖直平移自由度（自由关节的第 3 个）的分量是 2d，对水平平移是 0。
TEST(Row, VerticalTranslationGivesTwoD) {
  const Model M = load("drop.xml");
  const Frames F = forward_kinematics(M, M.qpos0);
  for (const Contact& c : find_contacts(M, F, 0.6)) {
    const RowVectorXd r = contact_row(M, F, c);
    EXPECT_NEAR(r[2], 2 * c.d, 1e-12);
    EXPECT_NEAR(r[0], 0.0, 1e-12);
    EXPECT_NEAR(r[1], 0.0, 1e-12);
  }
}

// ---------- 第 3 节：能量、梯度、Hessian ----------

// 梯度和能量的有限差分一致。dhat 取 1 m，让一部分顶点活跃、一部分不活跃。
TEST(Energy, GradientMatchesFiniteDifference) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(51);
  const double dhat = 1.0, kappa = 3.0, eps = 1e-6;
  for (int i = 0; i < 5; ++i) {
    const VectorXd q = random_q_above_floor(M, rng);
    const Frames F = forward_kinematics(M, q);
    const std::vector<Contact> C = find_contacts(M, F, dhat);
    ASSERT_FALSE(C.empty());
    const VectorXd g = contact_gradient(M, F, C, kappa, dhat);
    const auto energy = [&](const VectorXd& qq) {
      const Frames FF = forward_kinematics(M, qq);
      return contact_energy(M, find_contacts(M, FF, dhat), kappa, dhat);
    };
    for (int d = 0; d < M.nv; ++d) {
      const VectorXd e = VectorXd::Unit(M.nv, d);
      const double fd = (energy(integrate(M, q, e, eps)) - energy(integrate(M, q, e, -eps))) / (2 * eps);
      EXPECT_NEAR(g[d], fd, 1e-6 * (1 + std::abs(fd))) << "pose " << i << ", dof " << d;
    }
  }
}

// Gauss-Newton Hessian 对称、半正定，并且只在活跃顶点所在的链上非零。
TEST(Energy, HessianIsSymmetricPositiveSemidefinite) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(52);
  const VectorXd q = random_q_above_floor(M, rng);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, 1.0);
  const MatrixXd H = contact_hessian(M, F, C, 3.0, 1.0);
  EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
  const VectorXd eig = Eigen::SelfAdjointEigenSolver<MatrixXd>(H).eigenvalues();
  EXPECT_GE(eig.minCoeff(), -1e-12 * (1 + eig.maxCoeff()));
}

// 只有 d < dhat 的接触进 ground_contacts；没有接触时能量、梯度、Hessian 都是 0。
TEST(Energy, InactiveContactsContributeNothing) {
  const Model M = load("drop.xml");
  const Frames F = forward_kinematics(M, M.qpos0);  // 下面 4 个角离地 0.5
  const std::vector<Contact> none = find_contacts(M, F, 0.49);
  EXPECT_TRUE(none.empty());
  EXPECT_EQ(contact_energy(M, none, 1e9, 0.49), 0.0);
  EXPECT_EQ(contact_gradient(M, F, none, 1e9, 0.49).norm(), 0.0);
  EXPECT_EQ(contact_hessian(M, F, none, 1e9, 0.49).norm(), 0.0);
  EXPECT_EQ(find_contacts(M, F, 0.51).size(), 4u);
  EXPECT_EQ(find_contacts(M, F, 0.91).size(), 8u);
}

// ---------- 第 4 节：kappa ----------

// 参考值：一个障碍在 dhat/2 处的力等于平均刚体的重量。范围的上下限相差 1e4。
TEST(Kappa, RangeFollowsTheMassAndDhat) {
  const Model M = load("drop.xml");
  const double dhat = 1e-3;
  const KappaRange R = kappa_range(M, dhat);
  const double d = 0.5 * dhat;
  EXPECT_NEAR(R.ref * std::abs(barrier(d * d, dhat * dhat).db) * 2 * d, 1.0 * 9.81, 1e-9);
  EXPECT_NEAR(R.max / R.min, 1e4, 1e-9);
  EXPECT_GT(R.ref, 0);
}

// 梯度平衡：盒子静止悬在 d = dhat/2 处（惯性目标是“再往下掉 g h^2”），kappa 让障碍梯度正好抵消惯性梯度。
// 4 个角一起，每个角的力是重力的 1/4：kappa |b'(d^2)| 2d * 4 = m g。
TEST(Kappa, InitialValueBalancesGravityOnTheBox) {
  const Model M = load("drop.xml");
  const double dhat = 1e-3, h = 0.002;
  VectorXd q = M.qpos0;
  q[2] = 0.2 + 0.5 * dhat;  // 下面 4 个角离地 dhat/2
  const Frames F = forward_kinematics(M, q);
  const InertiaTargets T = inertia_targets(M, F, F, h);  // 静止：目标 = 当前 + g h^2（在下方）
  const VectorXd g_e = inertia_gradient(M, F, T, h);     // = (0, 0, +m g, 0, 0, 0)：往上走能量升高
  EXPECT_NEAR(g_e[2], 9.81, 1e-9);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  ASSERT_EQ(C.size(), 4u);
  const double kappa = initial_kappa(M, F, C, g_e, dhat);
  const double d = 0.5 * dhat;
  EXPECT_NEAR(4 * kappa * std::abs(barrier(d * d, dhat * dhat).db) * 2 * d, 9.81, 1e-6);
  const KappaRange R = kappa_range(M, dhat);
  EXPECT_GE(kappa, R.min);
  EXPECT_LE(kappa, R.max);
}

// ---------- 第 6 节：落地 ----------

// 盒子从 0.5 m 静止落下 1.5 s：每一步的整条轨迹都不穿透（64 个采样点），最后静止，4 个角的障碍力之和等于重力。
TEST(Drop, NeverPenetratesAndRestsInForceBalance) {
  const Model M = load("drop.xml");
  StepParams P;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(6);
  double dmin = 1e9;
  int nmax = 0;
  for (int i = 0; i < 750; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    const VectorXd p = differentiate(M, q, r.q);
    dmin = std::min(dmin, trajectory_min_distance(M, q, p, 1.0, 64));
    nmax = std::max(nmax, r.newton);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(nmax, P.max_newton);
  EXPECT_LT(v.norm(), 1e-5);  // 静止
  // 力平衡：4 个角都在同一个距离 d 上。
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, P.dhat);
  ASSERT_EQ(C.size(), 4u);
  // 收敛容差 tol = 1e-6 m/s 对应的残余力是 m tol / h = 5e-4 N，所以比到 2e-3 N。
  double force = 0;
  for (const Contact& c : C) force += S.kappa * std::abs(barrier(c.d * c.d, P.dhat * P.dhat).db) * 2 * c.d;
  EXPECT_NEAR(force, 9.81, 2e-3);
  // 这个场景里梯度平衡给出的值超过上限（接触第一次出现时 d 接近 dhat，b' 很小），被夹到上限（笔记第 4 节）。
  EXPECT_GE(S.kappa, kappa_range(M, P.dhat).min);
  EXPECT_LE(S.kappa, kappa_range(M, P.dhat).max);
  EXPECT_GT(C[0].d, 0.0);
  EXPECT_LT(C[0].d, P.dhat);
}

// 以 40 m/s 撞向地面（每步 40 cm，远大于离地高度）：一步之后间隙仍然为正（同阶段 1 的 Book 测试）。
TEST(Drop, HighSpeedImpactStopsAboveTheFloor) {
  const Model M = load("drop.xml");
  StepParams P;
  P.h = 0.01;
  VectorXd q = M.qpos0, v = VectorXd::Zero(6);
  q[2] = 0.2 + 0.3;  // 离地 0.3 m
  v[2] = -40;
  const StepResult r = step(M, P, q, v);
  EXPECT_GT(r.min_distance, 0.0);
  EXPECT_LT(r.min_distance, P.dhat);
  EXPECT_GT(trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 64), 0.0);
}
