// 阶段 7：刚体之间的接触（笔记 notes/07_contact.md）。
//
//   Mesh.*      第 1 节：盒子的边和面、球和胶囊的边；世界上的静态盒子
//   Detect.*    第 2 节：两个盒子之间找到点–三角形和边–边接触，距离和重新算的一致
//               平行的胶囊：边–边的 mollifier 是 0，端点–边接触（胶囊端点的退化边）仍然给出障碍
//               焊在世界上的刚体（weld 为 0）不和地面比，第 1.1 节
//   Row.*       第 3 节：点–三角形、边–边的接触行和有限差分一致；能量的梯度也是
//   Ccd.*       第 4 节：高速相撞的两个盒子，曲线 CCD 把步长截在相撞之前
//   Scene.*     第 5 节：堆叠、台阶、平行胶囊的槽，整条轨迹不穿透，最后静止
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

// 每个盒子 8 个顶点、12 条边、12 个三角形；球 1 条退化的边；胶囊 1 条边加 2 条退化的边（两个端点）。
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
  // zoo：2 个盒子（24 条边）+ 4 根胶囊（4 条边 + 8 条退化的边）+ 2 个球（2 条退化的边）= 38 条边，24 个三角形。
  EXPECT_EQ(Z.nedge(), 38);
  EXPECT_EQ(Z.nface(), 24);
  int degenerate = 0;
  for (const Edge& e : Z.edges) degenerate += e.v0 == e.v1;
  EXPECT_EQ(degenerate, 10);
}

// 三角网格（和 libuipc 同一套胶囊网格，compare/libuipc/ragdolls_1_mesh.xml）：一个布娃娃 11 个刚体，
// 10 个胶囊网格各 34 个顶点、64 个三角形，头（球）26 个顶点、48 个三角形；每个闭合网格 V - E + F = 2。
TEST(Mesh, TriangleMeshesFromMjcf) {
  const Model M = from_mujoco(load_mjcf(RCIPC_MODEL_DIR "/../compare/libuipc/ragdolls_1_mesh.xml").get());
  EXPECT_EQ(M.nvertex(), 10 * 34 + 26);
  EXPECT_EQ(M.nface(), 10 * 64 + 48);
  EXPECT_EQ(M.nedge(), 3 * M.nface() / 2);
  for (int b = 1; b < M.nbody(); ++b) {
    int nv = 0, ne = 0, nf = 0;
    for (const Vertex& v : M.vertices) nv += v.body == b;
    for (const Edge& e : M.edges) ne += e.body == b;
    for (const Face& f : M.faces) nf += f.body == b;
    EXPECT_EQ(nv - ne + nf, 2) << "body " << b;
  }
  for (const Vertex& v : M.vertices) EXPECT_EQ(v.radius, 0.0);
  // 静止姿态下不相交，最近的一对至少隔开关节处的 1.5 cm 减去网格的离散误差。
  EXPECT_GT(min_distance(M, forward_kinematics(M, M.qpos0)), 0.01);
}

// 重合的顶点合并（notes/22_primal_dual.md 第 5 节）：split_tetra.obj 每个面的角各写一遍，MuJoCo 编译后照样是
// 12 个顶点。合并后是一个闭合的四面体：4 个顶点、6 条边、4 个三角形（V - E + F = 2）。底面离地 0.5 mm，
// 底面的三个角各给出一个顶点–平面接触（不合并时每个角出现 3 次，是 9 个）。
TEST(Mesh, CoincidentVerticesMerged) {
  const MjModelPtr mj = load_mjcf(RCIPC_MODEL_DIR "/split_tetra.xml");
  ASSERT_EQ(mj->mesh_vertnum[0], 12);  // MuJoCo 不合并
  const Model M = from_mujoco(mj.get());
  EXPECT_EQ(M.nvertex(), 4);
  EXPECT_EQ(M.nedge(), 6);
  EXPECT_EQ(M.nface(), 4);
  EXPECT_EQ(find_contacts(M, forward_kinematics(M, M.qpos0), 1e-3).size(), 3u);
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

// 平行的胶囊（models/capsule_cradle.xml，笔记第 2.3 节）：自由胶囊和两根固定胶囊严格平行，轴间距 0.06 + gap。
//   它们之间的边–边接触 mollifier 是 0（能量里不起作用）；障碍来自自由胶囊两个端点对两根固定胶囊的点–边接触
//   （端点的退化边对另一条边，mollifier 是 1），一共 4 对，距离都是 gap。
//   所以能量随 gap 变小而增大，对高度 z 的梯度是负的（斥力向上），大小约和 1/gap 成正比。
//   没有端点的退化边时，这里的能量和梯度都恒为 0：没有东西挡住它。
TEST(Detect, ParallelCapsulesKeepABarrier) {
  const Model M = load("capsule_cradle.xml");
  const double dhat = 1e-3, kappa = 1e8;
  const auto at_gap = [&](double gap) {
    VectorXd q = M.qpos0;
    q[2] = 0.2 + std::sqrt((0.06 + gap) * (0.06 + gap) - 0.04 * 0.04);
    return q;
  };
  double e_prev = 0, f_prev = 0;
  for (double gap : {1e-4, 1e-5, 1e-6, 1e-7, 1e-9}) {
    const Frames F = forward_kinematics(M, at_gap(gap));
    const std::vector<Contact> C = find_contacts(M, F, dhat);
    EXPECT_NEAR(min_distance(M, F), gap, 1e-12);
    int holding = 0;
    for (const Contact& c : C) {
      ASSERT_EQ(c.type, ContactType::kEdgeEdge);
      if (c.moll == 1) {
        EXPECT_NEAR(c.d, gap, 1e-12);
        ++holding;
      } else {
        EXPECT_LT(c.moll, 1e-12);  // 胶囊的轴对轴：平行，被 mollifier 关掉
      }
    }
    EXPECT_EQ(holding, 4) << "gap " << gap;
    const double e = contact_energy(M, C, kappa, dhat);
    const double fz = -contact_gradient(M, F, C, kappa, dhat)[2];  // 对自由胶囊向上的力
    EXPECT_GT(e, e_prev) << "gap " << gap;
    EXPECT_GT(fz, 5 * f_prev) << "gap " << gap;  // gap 每小 10 倍（最后一次 100 倍），力至少大 5 倍
    e_prev = e;
    f_prev = fz;
  }
  EXPECT_GT(f_prev, 1e5);  // gap = 1e-9 m 时斥力已经远大于胶囊的重量（9.81 N）
}

// 焊在世界上的刚体（models/welded_base.xml，笔记第 1.1 节）：base 没有关节、直接挂在世界上，plate 焊在 base 上，
// 两个盒子的底面都在地面下 0.03 mm。它们和世界上的静态几何一样不动，不和地面比：否则开局的最小距离是 -0.03 mm
// （判成穿透），曲线 CCD 在这些顶点上卡住（起点已经在最小分离以内，步长 0）。
// 焊在自由盒子上的 ball 虽然也没有关节，但跟着盒子动（weld 不是 0），照样和地面比。
TEST(Detect, BodiesWeldedToTheWorldSkipThePlane) {
  const Model M = load("welded_base.xml");
  const int base = 1, plate = 2, box = 3, ball = 4;  // MJCF 里的先序编号
  ASSERT_EQ(M.nbody(), 5);
  ASSERT_EQ(M.bodies[plate].parent, base);
  ASSERT_EQ(M.bodies[ball].parent, box);
  EXPECT_EQ(M.weld[base], 0);
  EXPECT_EQ(M.weld[plate], 0);
  EXPECT_EQ(M.weld[ball], box);
  StepParams P;
  // 自由盒子放低，球离地 0.5 mm：地面接触只有球这一个，最小距离就是它。
  VectorXd q = M.qpos0;
  q[2] = 0.08 + 0.03 + 0.5e-3;
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, P.dhat);
  EXPECT_EQ(count(C, ContactType::kGround), 1);
  for (const Contact& c : C) {
    EXPECT_NE(M.weld[c.bodyA], 0);
    if (c.type == ContactType::kGround) EXPECT_EQ(c.bodyA, ball);
  }
  EXPECT_NEAR(min_distance(M, F), 0.5e-3, 1e-12);
  EXPECT_TRUE(std::isfinite(contact_energy(M, C, 1e3, P.dhat)));
  // 往下走 1 mm：曲线 CCD 截在球碰地之前（0.5 处），但不是 0。
  VectorXd p = VectorXd::Zero(M.nv);
  p[2] = -1e-3;
  const double a = curved_ccd(M, q, p, P.dhat, P.c);
  EXPECT_GT(a, 0.25);
  EXPECT_LT(a, 0.5);
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
  P.adapt_kappa = false;  // 沿步首到步末的直线采样检查需要阶段 7 的大间隙；阶段 9 的检查见 test_solver.cc
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
  P.adapt_kappa = false;
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

// 焊在世界上的底座（models/welded_base.xml）：底座的盒子在地面下 0.03 mm，自由盒子带着焊在它下面的球落下来，0.4 s。
// 每一步都不穿透、Newton 不撞上限；盒子确实落到了地面（CCD 没有被底座的顶点卡住）。
TEST(Scene, BodiesWeldedToTheWorldDoNotStallTheStep) {
  const Model M = load("welded_base.xml");
  StepParams P;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  double dmin = 1e9;
  for (int i = 0; i < 200; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, r.min_distance);
    EXPECT_LT(r.newton, P.max_newton) << "step " << i;
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(q[2], 0.08 + 0.03 + 2 * P.dhat);  // 起点 0.2；球底落到地面上
}

// 平行胶囊的槽（models/capsule_cradle.xml）：自由胶囊从 1.5 cm 高落进两根平行的固定胶囊之间，0.5 s。
//   三根严格平行，所有边–边接触的 mollifier 都是 0。胶囊端点没有退化边（点–边接触）时，这个场景 0.1 s 内间隙就变成 0。
//   检验：每一步、每次迭代的真实轨迹都不穿透，间隙没有塌下去；最后停在槽里（轴比固定胶囊高 0.0447）。
TEST(Scene, CapsuleRestsInAParallelCradle) {
  const Model M = load("capsule_cradle.xml");
  StepParams P;
  P.check_iterates = true;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  double dmin = 1e9, dmin_iter = 1e9;
  int nmax = 0;
  for (int i = 0; i < 250; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, r.min_distance);
    dmin_iter = std::min(dmin_iter, r.iterate_min_distance);
    nmax = std::max(nmax, r.newton);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 1e-3 * P.dhat);  // 间隙没有塌（塌的时候是 1e-11 m 以下）
  EXPECT_GT(dmin_iter, 0.0);
  EXPECT_LT(nmax, P.max_newton);
  EXPECT_LT(v.norm(), 1e-2);
  EXPECT_NEAR(q[2], 0.2 + std::sqrt(0.06 * 0.06 - 0.04 * 0.04), 1e-3);
  EXPECT_NEAR(q[1], 0.0, 1e-3);
}
