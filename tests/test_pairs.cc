// 阶段 7：刚体之间的接触（笔记 notes/07_contact.md）。
//
//   Mesh.*      第 1 节：盒子的边和面、球和胶囊的边；世界上的静态盒子
//   Detect.*    第 2 节：两个盒子之间找到点–三角形和边–边接触，距离和重新算的一致
//   Row.*       第 3 节：点–三角形、边–边的接触行和有限差分一致；能量的梯度也是
//   Ccd.*       第 4 节：高速相撞的两个盒子，曲线 CCD 把步长截在相撞之前
//   Scene.*     第 5 节：堆叠、台阶，整条轨迹不穿透，最后静止
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rcipc/barrier.h"
#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::Quaterniond;
using Eigen::RowVectorXd;
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

// 堆叠场景里，上面的盒子放到离下面的盒子 gap 处（底面平行）。
VectorXd stacked_pose(const Model& M, double gap) {
  VectorXd q = M.qpos0;
  q[7 + 2] = 0.1005 + 0.1 + gap + 0.1;  // 上面盒子的中心：下面盒子的顶面 0.2005 再加 gap 和半边长
  return q;
}

int count(const std::vector<Contact>& C, ContactType t) {
  int n = 0;
  for (const Contact& c : C) n += c.type == t;
  return n;
}

}  // namespace

// ---------- 第 1 节：网格 ----------

// 每个盒子 8 个顶点、12 条边、12 个三角形；球 1 条退化的边；胶囊 1 条边。
TEST(Mesh, BoxesSpheresAndCapsules) {
  const Model S = load("stack.xml");
  EXPECT_EQ(S.nvertex(), 16);
  EXPECT_EQ(S.nedge(), 24);
  EXPECT_EQ(S.nface(), 24);
  // 每条边长 0.2，每个三角形的两条直角边长 0.2。
  for (const Edge& e : S.edges) EXPECT_NEAR((S.vertices[e.v0].local - S.vertices[e.v1].local).norm(), 0.2, 1e-15);
  for (const Face& f : S.faces) {
    const Vector3d a = S.vertices[f.v0].local, b = S.vertices[f.v1].local, c = S.vertices[f.v2].local;
    EXPECT_NEAR(((b - a).cross(c - a)).norm(), 0.04, 1e-15);  // 面积的 2 倍 = 0.2 * 0.2
    EXPECT_EQ(f.body, S.vertices[f.v0].body);
  }
  const Model Z = load("zoo.xml");
  // zoo：2 个盒子（24 条边）+ 4 根胶囊（4 条边）+ 2 个球（2 条退化的边）= 30 条边，24 个三角形。
  EXPECT_EQ(Z.nedge(), 30);
  EXPECT_EQ(Z.nface(), 24);
  int degenerate = 0;
  for (const Edge& e : Z.edges) degenerate += e.v0 == e.v1;
  EXPECT_EQ(degenerate, 2);
}

// 世界上的静态盒子：顶点在 0 号刚体上，接触行里它那一侧为 0；父子不检测但世界和它的子刚体要检测。
TEST(Mesh, StaticBoxOnTheWorld) {
  const Model L = load("ledge.xml");
  int world_vertices = 0;
  for (const Vertex& v : L.vertices) world_vertices += v.body == 0;
  EXPECT_EQ(world_vertices, 8);
  EXPECT_TRUE(L.collide(0, 1));
  EXPECT_FALSE(L.collide(1, 1));
  const Model Z = load("zoo.xml");
  // zoo 的 torso(1) 是 head(2) 的父刚体：不检测；head 和 arm(3) 是兄弟：检测。
  EXPECT_FALSE(Z.collide(1, 2));
  EXPECT_TRUE(Z.collide(2, 3));
}

// ---------- 第 2 节：检测 ----------

// 上面的盒子（转 45 度）悬在下面的盒子上方 0.5 mm：两个盒子的边交叉，找到 8 个边–边接触（上面的 4 条底边
// 和下面的 4 条顶边两两交叉），距离都是 0.5 mm；点–三角形都在 4 cm 之外（转了 45 度以后顶角都伸到对方的面外面）。
// 每个接触重新算的距离和记录的一致；两侧权重之和是 +1 和 -1；法向从 B 指向 A。
TEST(Detect, RotatedStackGivesEdgeEdgeContacts) {
  const Model M = load("stack.xml");
  const double dhat = 1e-3;
  const VectorXd q = stacked_pose(M, 0.5 * dhat);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  EXPECT_EQ(count(C, ContactType::kPointTriangle), 0);
  EXPECT_EQ(count(C, ContactType::kEdgeEdge), 8);
  EXPECT_EQ(count(C, ContactType::kGround), 4);  // 下面盒子的 4 个底角离地 0.5 mm
  for (const Contact& c : C) {
    EXPECT_GT(c.d, 0.0);
    EXPECT_LT(c.d, dhat);
    EXPECT_NEAR(contact_distance(M, F, c), c.d, 1e-12);
    EXPECT_NEAR(c.n.norm(), 1.0, 1e-12);
    if (c.type != ContactType::kGround) {
      EXPECT_NE(c.bodyA, c.bodyB);
      EXPECT_EQ(c.moll, 1.0);  // 交叉的边离平行很远
      double wa = 0, wb = 0;
      for (int i = 0; i < c.nstencil; ++i) (c.w[i] > 0 ? wa : wb) += c.w[i];
      EXPECT_NEAR(wa, 1.0, 1e-12);
      EXPECT_NEAR(wb, -1.0, 1e-12);
      EXPECT_NEAR(std::abs(c.n.z()), 1.0, 1e-9);
      EXPECT_GT(c.n.z() * (c.bodyA == 2 ? 1 : -1), 0);  // A 在上面时 n 朝上
    }
  }
  // 抬高到 dhat 之外：只剩地面接触，最小距离就是地面的 0.5 mm。
  const Frames F2 = forward_kinematics(M, stacked_pose(M, 2 * dhat));
  EXPECT_EQ(find_contacts(M, F2, dhat).size(), 4u);
  EXPECT_NEAR(min_distance(M, F2), 0.5e-3, 1e-12);
}

// 上面的盒子不转（底面正对顶面）：每个角正好在对方的角上。点–三角形接触很多（一个角离对方角上的 6 个三角形
// 都是 0.5 mm），平行的边–边接触也在（mollifier 把它们的能量乘到 0，但它们仍然被检测到），
// 垂直相交的边–边接触的 mollifier 是 1。所有接触的距离都是 0.5 mm。
TEST(Detect, AlignedStackGivesPointTriangleContacts) {
  const Model M = load("stack.xml");
  const double dhat = 1e-3;
  VectorXd q = stacked_pose(M, 0.5 * dhat);
  quat_store(Quaterniond::Identity(), &q[7 + 3]);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  EXPECT_GE(count(C, ContactType::kPointTriangle), 8);
  int parallel = 0, crossing = 0;
  for (const Contact& c : C) {
    if (c.type == ContactType::kGround) continue;
    EXPECT_NEAR(c.d, 0.5e-3, 1e-12);
    if (c.type == ContactType::kEdgeEdge) {
      parallel += c.moll == 0.0;
      crossing += c.moll == 1.0;
      EXPECT_TRUE(c.moll == 0.0 || c.moll == 1.0);  // 要么平行，要么垂直
    }
  }
  EXPECT_GT(parallel, 0);
  EXPECT_GT(crossing, 0);
  EXPECT_GT(contact_energy(M, C, 1.0, dhat), 0.0);
}

// ---------- 第 3 节：接触行 ----------

// zoo.xml 的随机姿态，dhat 取 10 让所有候选对都活跃：每个接触（三种）的行 r 和 d^2 的有限差分一致。
TEST(Row, MatchesFiniteDifferenceForEveryContactType) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(90);
  const double eps = 1e-6;
  int seen[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    const VectorXd q = random_q(M, rng);
    const Frames F = forward_kinematics(M, q);
    const std::vector<Contact> C = find_contacts(M, F, 10.0);
    for (const Contact& c : C) {
      ++seen[static_cast<int>(c.type)];
      const RowVectorXd r = contact_row(M, F, c);
      for (int d = 0; d < M.nv; ++d) {
        const VectorXd e = VectorXd::Unit(M.nv, d);
        const auto x2 = [&](double s) {
          const double dist = contact_distance(M, forward_kinematics(M, integrate(M, q, e, s)), c);
          return dist * dist;
        };
        const double fd = (x2(eps) - x2(-eps)) / (2 * eps);
        EXPECT_NEAR(r[d], fd, 1e-6 * (1 + std::abs(fd))) << "pose " << i << ", type " << int(c.type) << ", dof " << d;
      }
    }
  }
  EXPECT_GT(seen[0], 0);
  EXPECT_GT(seen[1], 0);
  EXPECT_GT(seen[2], 0);
}

// 两个盒子的接触能量的梯度和有限差分一致（dhat = 5 cm，上面的盒子悬在 2 cm 处并稍微歪一点）。
TEST(Row, EnergyGradientMatchesFiniteDifference) {
  const Model M = load("stack.xml");
  const double dhat = 0.05, kappa = 1e3, eps = 1e-6;
  VectorXd q = stacked_pose(M, 0.02);
  quat_store(Quaterniond(Eigen::AngleAxisd(0.6, Vector3d(0, 0, 1)) * Eigen::AngleAxisd(0.1, Vector3d(1, 0, 0))),
             &q[7 + 3]);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  ASSERT_GT(count(C, ContactType::kPointTriangle), 0);
  ASSERT_GT(count(C, ContactType::kEdgeEdge), 0);
  const VectorXd g = contact_gradient(M, F, C, kappa, dhat);
  const auto energy = [&](const VectorXd& qq) {
    const Frames FF = forward_kinematics(M, qq);
    return contact_energy(M, find_contacts(M, FF, dhat), kappa, dhat);
  };
  for (int d = 0; d < M.nv; ++d) {
    const VectorXd e = VectorXd::Unit(M.nv, d);
    const double fd = (energy(integrate(M, q, e, eps)) - energy(integrate(M, q, e, -eps))) / (2 * eps);
    EXPECT_NEAR(g[d], fd, 1e-5 * (1 + std::abs(fd))) << "dof " << d;
  }
}

// 台阶：盒子的底边和台阶的顶边几乎平行（绕 z 转 0.02 rad），边–边的 mollifier 在起作用（m < 1）；
// 能量的梯度（含 mollifier 的导数）和有限差分一致。
TEST(Row, MollifiedEdgeEdgeGradientMatchesFiniteDifference) {
  const Model M = load("ledge.xml");
  const double dhat = 0.05, kappa = 1e3, eps = 1e-6;
  VectorXd q = M.qpos0;
  q[0] = 0.4;               // 盒子的一条底边正好在台阶顶边（x = 0.3）的上方
  q[2] = 0.2 + 0.1 + 0.02;  // 底面离台阶顶面 2 cm
  quat_store(Quaterniond(Eigen::AngleAxisd(0.02, Vector3d(0, 0, 1))), &q[3]);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  int mollified = 0;
  for (const Contact& c : C) mollified += c.type == ContactType::kEdgeEdge && c.moll < 1;
  ASSERT_GT(mollified, 0);
  const VectorXd g = contact_gradient(M, F, C, kappa, dhat);
  const auto energy = [&](const VectorXd& qq) {
    const Frames FF = forward_kinematics(M, qq);
    return contact_energy(M, find_contacts(M, FF, dhat), kappa, dhat);
  };
  for (int d = 0; d < M.nv; ++d) {
    const VectorXd e = VectorXd::Unit(M.nv, d);
    const double fd = (energy(integrate(M, q, e, eps)) - energy(integrate(M, q, e, -eps))) / (2 * eps);
    EXPECT_NEAR(g[d], fd, 1e-5 * (1 + std::abs(fd))) << "dof " << d;
  }
}

// ---------- 第 4 节：CCD ----------

// 上面的盒子以 20 m/s 砸向下面的盒子（一步 20 cm，远大于 4 cm 的间隙）并带着转动：曲线 CCD 把步长截在相撞之前，
// 真实轨迹上的最小距离 > 0；步结束时在 dhat 之内。
TEST(Ccd, FastBoxOnBoxIsStoppedBeforeImpact) {
  const Model M = load("stack.xml");
  StepParams P;
  P.h = 0.01;
  VectorXd q = stacked_pose(M, 0.04), v = VectorXd::Zero(12);
  v[6 + 2] = -20.0;
  v[6 + 3] = 3.0;
  v[6 + 5] = 2.0;
  std::vector<NewtonIterate> trace;
  const StepResult r = step(M, P, q, v, nullptr, &trace);
  ASSERT_FALSE(trace.empty());
  EXPECT_LT(trace[0].alpha_max, 1.0);
  EXPECT_GT(r.min_distance, 0.0);
  EXPECT_LT(r.min_distance, P.dhat);
  EXPECT_GT(trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 128), 0.0);
}

// ---------- 第 5 节：场景 ----------

// 堆叠：上面的盒子（转 45 度）从 0.5 m 落到下面的盒子上，1.5 s。每一步的真实轨迹都不穿透；
// 最后两个盒子都静止，上面的盒子压在下面的盒子上（间隙 < dhat），地面承受两个盒子的重量。
TEST(Scene, StackNeverPenetratesAndRests) {
  const Model M = load("stack.xml");
  StepParams P;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(12);
  double dmin = 1e9;
  int nmax = 0;
  for (int i = 0; i < 750; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 16));
    nmax = std::max(nmax, r.newton);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(nmax, P.max_newton);
  EXPECT_LT(v.norm(), 1e-2);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, P.dhat);
  EXPECT_GT(count(C, ContactType::kPointTriangle) + count(C, ContactType::kEdgeEdge), 0);
  EXPECT_NEAR(q[2], 0.1 + (q[7 + 2] - 0.3), 2e-3);  // 上面的盒子中心比下面的高 0.2 左右
  EXPECT_GT(q[7 + 2], 0.3);
}

// 台阶：盒子落在世界上的静态盒子的边缘，翻下来掉到地面。整条轨迹不穿透。
TEST(Scene, LedgeNeverPenetrates) {
  const Model M = load("ledge.xml");
  StepParams P;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(6);
  double dmin = 1e9;
  for (int i = 0; i < 750; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 16));
    EXPECT_LT(r.newton, P.max_newton) << "step " << i;
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(q[2], 0.45);  // 掉下来了（或者至少不在原来的高度）
}

