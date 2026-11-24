// 阶段 11：GPU 版和 CPU 版逐项对比（笔记 notes/11_gpu.md 第 7 节）。
//
// GPU 版的每一项都对应 CPU 版的一个函数，所以这里一项一项地比：
//   Fk.*      正运动学（zoo.xml，四种关节）
//   Terms.*   能量、梯度、稠密 Hessian（惯性和它的曲率项、接触三种、mollifier、摩擦、曲率上界、限位、世界上的静态几何）
//   Pcg.*     PCG 的解和稠密 LDLT 的解
//   Ccd.*     CCD 的步长上限（地面、刚体之间、网格）
//   Step.*    整步：落地盒子、堆叠、布娃娃、链条，和 CPU 版的轨迹一致；网格布娃娃堆不穿透
//   Topology.*  穿透会改变拓扑的两个场景：锁链的环绕数不变、搅拌机的球不出罐子（notes/14_visible.md）
//   Drift.*     摆锤越过矮墙：轻链吊重物，关节精确、刚体刚性，链不会被拉长（notes/15_drift.md）
//   PrimalDual.*  原始-对偶障碍（StepParams::primal_dual）：开、关都和 CPU 版一致
// GPU 的原子加让求和顺序不固定，所以都按相对误差比（1e-9 量级），不要求逐位相同。没有显卡时整组跳过。
#include "rcipc/gpu.h"

#include <algorithm>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rcipc/barrier.h"
#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/drive.h"
#include "rcipc/friction.h"
#include "rcipc/inertia.h"
#include "rcipc/kinematics.h"
#include "rcipc/limits.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/solver.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;

Model load(const std::string& file) {
  const std::string path = file.find('/') != std::string::npos ? file : std::string(RCIPC_MODEL_DIR "/") + file;
  return from_mujoco(load_mjcf(path).get());
}
const char* kMesh1 = RCIPC_MODEL_DIR "/../compare/libuipc/ragdolls_1_mesh.xml";
const char* kMesh8 = RCIPC_MODEL_DIR "/../compare/libuipc/ragdolls_8_mesh.xml";
const char* kCaps8 = RCIPC_MODEL_DIR "/../compare/libuipc/ragdolls_8.xml";

#define REQUIRE_GPU()                                   \
  if (!gpu::available()) GTEST_SKIP() << "no CUDA device"

// CPU 版在 q 处的各项（和 GPU 的 Stepper::terms 一一对应）：步首 (q0, v0) 的惯性目标和滞后摩擦。
// rank1_rows：两阶段曲率用秩 1 行的那一次迭代（同 Stepper::terms 的 rank1_rows）。
gpu::Stepper::Terms cpu_terms(const Model& M, const StepParams& P, const VectorXd& q0, const VectorXd& v0, double kappa,
                              const VectorXd& q, bool rank1_rows = false) {
  const VectorXd q_prev = integrate(M, q0, v0, -P.h);
  const Frames F0 = forward_kinematics(M, q0);
  const InertiaTargets T = inertia_targets(M, F0, forward_kinematics(M, q_prev), P.h);
  const std::vector<FrictionContact> FC =
      P.mu > 0 ? lag_friction(M, F0, find_contacts(M, F0, P.dhat), kappa, P.dhat) : std::vector<FrictionContact>();
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, P.dhat);
  gpu::Stepper::Terms out;
  out.energy = inertia_energy(M, F, T, P.h) + contact_energy(M, C, kappa, P.dhat) +
               friction_energy(M, F, FC, P.h, P.mu, P.eps_v) + limit_energy(M, q, P.limit_barrier());
  out.g = inertia_gradient(M, F, T, P.h) + contact_gradient(M, F, C, kappa, P.dhat) +
          friction_gradient(M, F, FC, P.h, P.mu, P.eps_v) + limit_gradient(M, q, P.limit_barrier());
  const MatrixXd Mh = inertia_hessian(M, F, P.h);
  if (P.full_contact_hessian) {  } else if (P.full_contact_hessian) {
    // 阶段 14：行是每个接触的旋量块的半正定部分，链上剩余项和惯性的曲率项一起检查（同 step.cc）。
    NewtonSystem sys;
    sys.Hk = MatrixXd::Zero(M.nv, M.nv);
    contact_rows_full(M, F, C, kappa, P.dhat, &sys.rows);
    out.H = Mh + dense(sys) + friction_hessian(M, F, FC, P.h, P.mu, P.eps_v);
    out.H.diagonal() += limit_hessian_diagonal(M, q, P.limit_barrier());
    MatrixXd K = contact_chain_curvature(M, F, C, kappa, P.dhat);
    if (P.inertia_curvature) K += inertia_curvature(M, F, T, P.h, P.inertia_curvature_rot);
    add_inertia_curvature(M, Mh, K, &out.H, P.inertia_curvature_floor);
  } else if (rank1_rows) {
    // 两阶段曲率用秩 1 行的那一次迭代（同 step.cc 的 curv_rows）：不加阶段 6 的上界，每一行的权重加
    // contact_curvature_row_weight。
    out.H = Mh + contact_hessian(M, F, C, kappa, P.dhat) + friction_hessian(M, F, FC, P.h, P.mu, P.eps_v);
    for (const Contact& c : C) {
      const Eigen::RowVectorXd r = contact_row(M, F, c);
      out.H += contact_curvature_row_weight(M, F, c, r, kappa, P.dhat) * (r.transpose() * r);
    }
    out.H.diagonal() += limit_hessian_diagonal(M, q, P.limit_barrier());
    if (P.inertia_curvature) add_inertia_curvature(M, Mh, inertia_curvature(M, F, T, P.h, P.inertia_curvature_rot), &out.H,
                                               P.inertia_curvature_floor);
  } else {
    out.H = Mh + contact_curvature(M, q, C, kappa, P.dhat, P.curv_scale) + contact_hessian(M, F, C, kappa, P.dhat) +
            friction_hessian(M, F, FC, P.h, P.mu, P.eps_v);
    out.H.diagonal() += limit_hessian_diagonal(M, q, P.limit_barrier());
    if (P.inertia_curvature) add_inertia_curvature(M, Mh, inertia_curvature(M, F, T, P.h, P.inertia_curvature_rot), &out.H,
                                               P.inertia_curvature_floor);  // 阶段 13
  }
  out.ncontact = static_cast<int>(C.size());
  out.nfriction = static_cast<int>(FC.size());
  return out;
}

// CPU 版跑 n 步，返回最后的状态（拿来当一个“有接触、有速度”的测试点）。
struct State {
  VectorXd q, v;
  SolverState S;
};
State cpu_run(const Model& M, const StepParams& P, int n, VectorXd q, VectorXd v) {
  State s{q, v, {}};
  for (int i = 0; i < n; ++i) {
    const StepResult r = step(M, P, s.q, s.v, &s.S);
    s.q = r.q;
    s.v = r.v;
  }
  return s;
}

double rel(const VectorXd& a, const VectorXd& b) { return (a - b).norm() / std::max(1.0, b.norm()); }
double rel(const MatrixXd& a, const MatrixXd& b) { return (a - b).norm() / std::max(1.0, b.norm()); }

void expect_terms_match(const Model& M, const StepParams& P, const State& s, const VectorXd& q, const char* what,
                        bool rank1_rows = false) {
  gpu::Stepper G(M);
  const gpu::Stepper::Terms g = G.terms(P, s.q, s.v, s.S.kappa, q, rank1_rows);
  const gpu::Stepper::Terms c = cpu_terms(M, P, s.q, s.v, s.S.kappa, q, rank1_rows);
  EXPECT_EQ(g.ncontact, c.ncontact) << what;
  EXPECT_EQ(g.nfriction, c.nfriction) << what;
  EXPECT_NEAR(g.energy, c.energy, 1e-10 * std::max(1.0, std::abs(c.energy))) << what;
  EXPECT_LT(rel(g.g, c.g), 1e-9) << what;
  EXPECT_LT(rel(g.H, c.H), 1e-9) << what;
}

}  // namespace

// ---------- 正运动学 ----------

TEST(Fk, MatchesCpuOnZoo) {
  REQUIRE_GPU();
  const Model M = load("zoo.xml");
  gpu::Stepper G(M);
  std::mt19937 rng(110);
  std::normal_distribution<double> n;
  std::uniform_real_distribution<double> u(-1, 1);
  for (int i = 0; i < 10; ++i) {
    VectorXd q = M.qpos0;
    for (const Joint& j : M.joints) {
      if (j.type == JointType::kFree) {
        q.segment<3>(j.qadr) = Vector3d(u(rng), u(rng), u(rng));
        quat_store(Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized(), &q[j.qadr + 3]);
      } else if (j.type == JointType::kBall) {
        quat_store(Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized(), &q[j.qadr]);
      } else {
        q[j.qadr] = j.ref + kPi * u(rng);
      }
    }
    const Frames a = G.fk(q), b = forward_kinematics(M, q);
    for (int k = 0; k < M.nbody(); ++k) {
      EXPECT_LT((a.xpos[k] - b.xpos[k]).norm(), 1e-14) << "body " << k;
      EXPECT_LT((a.xquat[k].coeffs() - b.xquat[k].coeffs()).norm(), 1e-14) << "body " << k;
    }
    for (int d = 0; d < M.nv; ++d) {
      EXPECT_LT((a.dof_axis[d] - b.dof_axis[d]).norm(), 1e-14) << "dof " << d;
      EXPECT_LT((a.dof_anchor[d] - b.dof_anchor[d]).norm(), 1e-14) << "dof " << d;
    }
  }
}

// ---------- 能量、梯度、Hessian ----------

// 落地的盒子（地面接触 + 摩擦）：静止在地面上的状态，和稍微挪开一点的点。
// 碰撞过滤（焊在一起的整体、<exclude>，filters.xml）：GPU 的宽相和 CPU 一样过滤，所以 dhat 很大时检测到的接触、
// 能量、梯度、Hessian 都一致。
TEST(Terms, FilteredPairs) {
  REQUIRE_GPU();
  const Model M = load("filters.xml");
  StepParams P;
  P.dhat = 0.2;
  P.mu = 0;
  const State s{M.qpos0, VectorXd::Zero(M.nv), {}};
  State sk = s;
  sk.S.kappa = 1.0;
  expect_terms_match(M, P, sk, M.qpos0, "filters");
}

TEST(Terms, BoxOnTheFloor) {
  REQUIRE_GPU();
  const Model M = load("drop.xml");
  StepParams P;
  const State s = cpu_run(M, P, 400, M.qpos0, VectorXd::Zero(6));
  ASSERT_GT(s.S.kappa, 0);
  expect_terms_match(M, P, s, s.q, "rest");
  VectorXd d = VectorXd::Zero(6);
  d << 1e-5, -2e-5, 1e-6, 1e-4, 0, -2e-4;
  expect_terms_match(M, P, s, integrate(M, s.q, d), "perturbed");
}

// 两个盒子（点–三角形、边–边，跨树的行）。
TEST(Terms, StackedBoxes) {
  REQUIRE_GPU();
  const Model M = load("stack.xml");
  StepParams P;
  const State s = cpu_run(M, P, 300, M.qpos0, VectorXd::Zero(12));
  ASSERT_GT(s.S.kappa, 0);
  expect_terms_match(M, P, s, s.q, "stack");
}

// 台阶（世界上的静态盒子，平行边的 mollifier）。
TEST(Terms, LedgeWithStaticGeometry) {
  REQUIRE_GPU();
  const Model M = load("ledge.xml");
  StepParams P;
  const State s = cpu_run(M, P, 120, M.qpos0, VectorXd::Zero(6));
  expect_terms_match(M, P, s, s.q, "ledge");
}

// 焊在世界上的刚体（models/welded_base.xml，notes/07_contact.md 第 1.1 节）：底座的盒子在地面下 0.03 mm。
// GPU 的地面检测（k_ground_detect）和地面 CCD（k_ground_ccd）和 CPU 一样跳过 weld 为 0 的刚体：球离地 0.5 mm 时
// 只有球一个地面接触，能量、梯度、Hessian 一致；往下走 1 mm 的 CCD 步长一致，而且没有被底座的顶点卡成 0。
TEST(Terms, BodiesWeldedToTheWorldSkipThePlane) {
  REQUIRE_GPU();
  const Model M = load("welded_base.xml");
  StepParams P;
  VectorXd q = M.qpos0;
  q[2] = 0.08 + 0.03 + 0.5e-3;
  State s{q, VectorXd::Zero(M.nv), {}};
  s.S.kappa = 1e3;
  EXPECT_EQ(find_contacts(M, forward_kinematics(M, q), P.dhat).size(), 1u);
  expect_terms_match(M, P, s, q, "welded base");
  VectorXd p = VectorXd::Zero(M.nv);
  p[2] = -1e-3;
  const double b = std::min(curved_ccd(M, q, p, P.dhat, P.c), limit_ccd(M, q, p, P.c));
  EXPECT_GT(b, 0.25);
  gpu::Stepper G(M);
  EXPECT_NEAR(G.ccd(P, q, p), b, 1e-12);
}

// 阶段 13：惯性的曲率项。摆锤摆到最低点附近（链上拉力最大，这一项比 M / h^2 大几十倍），以及 zoo.xml
// （四种关节）带很大的速度，转动那一部分加和不加都比。GPU 的子树求和写法和 CPU 的逐个刚体写法一致，
// 而且这一项确实改变了 Hessian。
TEST(Terms, InertiaCurvature) {
  REQUIRE_GPU();
  {
    const Model M = load("wrecking_ball.xml");
    StepParams P;
    P.h = 0.01;
    P.eps_v = 0.01;
    P.tol = 0.01;
    P.semi_implicit = true;
    const State s = cpu_run(M, P, 55, M.qpos0, VectorXd::Zero(M.nv));
    expect_terms_match(M, P, s, s.q, "wrecking ball");
    gpu::Stepper G(M);
    const MatrixXd H1 = G.terms(P, s.q, s.v, s.S.kappa, s.q).H;
    EXPECT_EQ(G.timers().curv_dropped, 0);
    P.inertia_curvature = false;
    const MatrixXd H0 = G.terms(P, s.q, s.v, s.S.kappa, s.q).H;
    // 之字形方向（铰链 4、5、6 转 (1, -2, 1)，摆锤一阶不动，只有两节钢条在动）：拉力的几何刚度比 M / h^2 大几十倍。
    VectorXd z = VectorXd::Zero(M.nv);
    z[4] = 1;
    z[5] = -2;
    z[6] = 1;
    EXPECT_GT(z.dot(H1 * z), 10 * z.dot(H0 * z));
  }
  {
    const Model M = load("zoo.xml");
    StepParams P;
    P.h = 0.01;
    P.mu = 0;
    std::mt19937 rng(120);
    std::uniform_real_distribution<double> u(-5, 5);
    VectorXd v(M.nv);
    for (int d = 0; d < M.nv; ++d) v[d] = u(rng);
    const State s{M.qpos0, v, {}};
    VectorXd dq(M.nv);
    for (int d = 0; d < M.nv; ++d) dq[d] = 0.01 * u(rng);
    expect_terms_match(M, P, s, integrate(M, M.qpos0, dq), "zoo");
    P.inertia_curvature_rot = true;  // 连转动那一部分也加（默认不加，笔记 16 第 5 节）
    expect_terms_match(M, P, s, integrate(M, M.qpos0, dq), "zoo, with the rotational part");
  }
}

// 布娃娃（球、胶囊带半径，关节限位，深的链）。
TEST(Terms, RagdollWithLimits) {
  REQUIRE_GPU();
  const Model M = load("ragdoll.xml");
  StepParams P;
  const State s = cpu_run(M, P, 300, M.qpos0, VectorXd::Zero(M.nv));
  expect_terms_match(M, P, s, s.q, "ragdoll");
}

// 阶段 14：完整的接触 Hessian。两个盒子（自由刚体：链上剩余项为 0）、台阶（mollifier）、布娃娃（铰链锚点不在原点、
// 深的链：剩余项非零，走 CB 和曲率内核的那条路）。行数变成每个接触 12 行，摩擦行跟在后面。
TEST(Terms, FullContactHessian) {
  REQUIRE_GPU();
  {
    const Model M = load("stack.xml");
    StepParams P;
    P.full_contact_hessian = true;
    const State s = cpu_run(M, P, 300, M.qpos0, VectorXd::Zero(12));
    ASSERT_GT(s.S.kappa, 0);
    expect_terms_match(M, P, s, s.q, "stack, full");
    // 和 Gauss-Newton 的确实不一样（切向的刚度不同）。
    gpu::Stepper G(M);
    const MatrixXd Hf = G.terms(P, s.q, s.v, s.S.kappa, s.q).H;
    P.full_contact_hessian = false;
    const MatrixXd Hg = G.terms(P, s.q, s.v, s.S.kappa, s.q).H;
    EXPECT_GT((Hf - Hg).norm(), 1e-6 * Hg.norm());
  }
  {
    const Model M = load("ledge.xml");
    StepParams P;
    P.full_contact_hessian = true;
    const State s = cpu_run(M, P, 120, M.qpos0, VectorXd::Zero(6));
    expect_terms_match(M, P, s, s.q, "ledge, full");
  }
  {
    const Model M = load("ragdoll.xml");
    StepParams P;
    P.full_contact_hessian = true;
    const State s = cpu_run(M, P, 300, M.qpos0, VectorXd::Zero(M.nv));
    expect_terms_match(M, P, s, s.q, "ragdoll, full");
    P.inertia_curvature = false;  // 只有接触的剩余项进曲率内核
    expect_terms_match(M, P, s, s.q, "ragdoll, full, no inertia curvature");
  }
}

// 网格布娃娃（和 libuipc 同一套网格，d̂ = 1 cm）。
TEST(Terms, MeshRagdoll) {
  REQUIRE_GPU();
  const Model M = load(kMesh1);
  StepParams P;
  P.h = M.timestep;  // 0.01，和 libuipc 一样
  P.dhat = 0.01;
  P.eps_v = 0.01;
  const State s = cpu_run(M, P, 60, M.qpos0, VectorXd::Zero(M.nv));  // 0.6 s：已经落地
  ASSERT_GT(s.S.kappa, 0);
  expect_terms_match(M, P, s, s.q, "mesh ragdoll");
}

// 两阶段曲率的秩 1 行：GPU 每一侧只沿链走一遍、就地算行的分量，CPU 对每个模板顶点各走一遍。
// 布娃娃（球关节的三个转动、自由关节的根）落地以后；布娃娃堆里手臂碰到自己躯干的时候（两侧在同一棵树里，
// 行在共同的祖先上要把两侧加起来）。逐项比稠密的 Hessian；再确认秩 1 行真的进了 Hessian（和不开时不一样）。
TEST(Terms, RankOneCurvatureRows) {
  REQUIRE_GPU();
  for (const char* file : {"ragdoll.xml", "ragdoll_pile.xml"}) {
    const Model M = load(file);
    StepParams P;
    State s{M.qpos0, VectorXd::Zero(M.nv), {}};
    int same_tree = 0;
    if (file[7] == '.') {
      s = cpu_run(M, P, 300, s.q, s.v);
    } else {
      for (int k = 0; k < 40 && same_tree == 0; ++k) {  // 每 25 步看一次，直到有自己碰自己的接触
        s = cpu_run(M, P, 25, s.q, s.v);
        for (const Contact& c : find_contacts(M, forward_kinematics(M, s.q), P.dhat))
          if (c.bodyA > 0 && c.bodyB > 0 && M.body_tree[c.bodyA] == M.body_tree[c.bodyB]) ++same_tree;
      }
      ASSERT_GT(same_tree, 0);
    }
    ASSERT_GT(s.S.kappa, 0) << file;
    expect_terms_match(M, P, s, s.q, file, true);
    const MatrixXd H1 = cpu_terms(M, P, s.q, s.v, s.S.kappa, s.q, true).H;
    const MatrixXd H0 = cpu_terms(M, P, s.q, s.v, s.S.kappa, s.q).H;
    EXPECT_GT((H1 - H0).norm(), 1e-6 * H0.norm()) << file;
  }
}

// ---------- 预条件器 ----------

// 树块预条件器的逆：每棵树一块 S (S H_tt S + 1e-10 I)^-1 S，S = diag(1 / sqrt(H_ii))（H 用 CPU 版组装）。
// 三个布娃娃的堆：落下以后手臂会碰到自己的躯干，这样就有两侧在同一棵树里的行（交叉项由 k_precond_cross 单独加）。
TEST(Precond, MatchesBlockInverse) {
  REQUIRE_GPU();
  const Model M = load("ragdoll_pile.xml");
  StepParams P;
  State s{M.qpos0, VectorXd::Zero(M.nv), {}};
  int same_tree = 0;
  for (int k = 0; k < 40 && same_tree == 0; ++k) {  // 每 25 步看一次，直到有自己碰自己的接触
    s = cpu_run(M, P, 25, s.q, s.v);
    for (const Contact& c : find_contacts(M, forward_kinematics(M, s.q), P.dhat))
      if (c.bodyA > 0 && c.bodyB > 0 && M.body_tree[c.bodyA] == M.body_tree[c.bodyB]) ++same_tree;
  }
  ASSERT_GT(same_tree, 0);
  gpu::Stepper G(M);
  G.terms(P, s.q, s.v, s.S.kappa, s.q);
  const MatrixXd Pinv = G.precond_inverse();
  const MatrixXd H = cpu_terms(M, P, s.q, s.v, s.S.kappa, s.q).H;
  MatrixXd ref = MatrixXd::Zero(M.nv, M.nv);
  for (int t = 0; t < M.ntree; ++t) {
    std::vector<int> ds;
    for (int d = 0; d < M.nv; ++d)
      if (M.dof_tree[d] == t) ds.push_back(d);
    const int n = static_cast<int>(ds.size());
    MatrixXd B(n, n);
    VectorXd sc(n);
    for (int i = 0; i < n; ++i) sc[i] = 1.0 / std::sqrt(H(ds[i], ds[i]));
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) B(i, j) = sc[i] * H(ds[i], ds[j]) * sc[j] + (i == j ? 1e-10 : 0.0);
    const MatrixXd Bi = sc.asDiagonal() * B.inverse() * sc.asDiagonal();
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) ref(ds[i], ds[j]) = Bi(i, j);
  }
  EXPECT_LT(rel(Pinv, ref), 1e-8) << same_tree << " same-tree contacts";
}

// ---------- PCG ----------

// 堆叠（两棵树、跨树的行）：PCG 收紧到 1e-12 时和稠密 LDLT 的解一致。
TEST(Pcg, MatchesDenseSolve) {
  REQUIRE_GPU();
  const Model M = load("stack.xml");
  StepParams P;
  const State s = cpu_run(M, P, 300, M.qpos0, VectorXd::Zero(12));
  gpu::Stepper G(M);
  const gpu::Stepper::Terms t = G.terms(P, s.q, s.v, s.S.kappa, s.q);
  const VectorXd ref = t.H.ldlt().solve(t.g);
  // 两种 PCG（主机循环、协作内核）都要和稠密的 LDLT 一致。
  for (bool coop : {false, true}) {
    if (G.set_cooperative_pcg(coop) != coop) continue;  // 设备不支持协作启动
    int iters = 0;
    const VectorXd x = G.solve(t.g, 1e-14, &iters);
    EXPECT_LT(rel(x, ref), 1e-6) << "cooperative " << coop;
    EXPECT_GT(iters, 0);
    EXPECT_LE(iters, M.nv + 2);
  }
}

// ---------- CCD ----------

// 两种求候选的方式（每段都求、按窗口求，set_ccd_reuse）都要和 CPU 版一样；按窗口求时确实有段用了窗口的候选。
// 单精度预筛（set_ccd_prefilter）开、关的结果逐位相同，而且确实跳过了一些对。
TEST(Ccd, MatchesCpu) {
  REQUIRE_GPU();
  struct Case {
    std::string file;
    double dhat;
  };
  long segs = 0, windows = 0;  // 按窗口求时：总段数、其中重新求候选的段数
  long prefiltered = 0;
  for (const Case& cs : {Case{"flail.xml", 1e-3}, Case{"stack.xml", 1e-3}, Case{"ragdoll.xml", 1e-3},
                         Case{kMesh1, 0.01}}) {
    const Model M = load(cs.file);
    StepParams P;
    P.dhat = cs.dhat;
    gpu::Stepper G(M);
    std::mt19937 rng(111);
    std::uniform_real_distribution<double> u(-1, 1);
    VectorXd q = M.qpos0;
    if (cs.file == "stack.xml") q[7 + 2] = 0.1005 + 0.1 + 0.02 + 0.1;  // 上面的盒子离下面 2 cm
    for (int k = 0; k < 5; ++k) {
      VectorXd p(M.nv);
      for (int i = 0; i < M.nv; ++i) p[i] = u(rng);
      p *= 0.5;
      if (cs.file == "stack.xml") p[8] -= 1.0;  // 往下砸
      const double b = std::min(curved_ccd(M, q, p, P.dhat, P.c), limit_ccd(M, q, p, P.c));
      for (bool reuse : {false, true}) {
        G.set_ccd_reuse(reuse);
        G.reset_timers();
        const double a = G.ccd(P, q, p);
        EXPECT_NEAR(a, b, 1e-12 * std::max(1.0, b)) << cs.file << " direction " << k << " reuse " << reuse;
        if (!reuse) {
          EXPECT_EQ(G.timers().ccd_windows, G.timers().ccd_segments);
          continue;
        }
        segs += G.timers().ccd_segments;
        windows += G.timers().ccd_windows;
        prefiltered += G.timers().ccd_prefiltered;
        G.set_ccd_prefilter(false);
        EXPECT_EQ(G.ccd(P, q, p), a) << cs.file << " direction " << k << " without the prefilter";
        G.set_ccd_prefilter(true);
      }
    }
  }
  EXPECT_LT(windows, segs);
  EXPECT_GT(prefiltered, 0);
}

// ACCD 的尾巴（笔记第 14 节）：走了 16 步还没完的对交给整个 warp，先证明它返回 1，证不出来再接着走。
// 有了分离平面的界（笔记 22 第 3 节）以后，ACCD 很少走到 16 步：下面两种运动里贴着平面滑动第一步就返回 1，
// 沉下去几步就截断，用不上尾巴。这里只确认交不交给 warp 结果都逐位相同、和 CPU 版一致。
// 两个盒子一起绕 y 轴倾斜 0.3 rad（倾斜的面上，三角形的盒子在 z 方向有厚度，扫掠盒才会重叠；水平的面上，
// 平移的滑动早被盒子规则剔掉了），上面的盒子离下面 0.1 mm，沿下面盒子的顶面滑 3 cm：贴着滑的对要走约 330 步。
//   - 只滑：距离一直是 0.1 mm，证得出返回 1；
//   - 滑的同时沿法向沉 0.2 mm：中途会碰上，证不出来，接着串行走完。
TEST(Ccd, TailMatchesSerial) {
  REQUIRE_GPU();
  const Model M = load("stack.xml");
  StepParams P;
  P.dhat = 1e-3;
  gpu::Stepper G(M);
  const double th = 0.3, gap = 1e-4;
  const Eigen::Quaterniond tilt(Eigen::AngleAxisd(th, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond up = tilt * Eigen::Quaterniond(Eigen::AngleAxisd(EIGEN_PI / 4, Eigen::Vector3d::UnitZ()));
  const Eigen::Vector3d centre(0, 0, 1.0);  // 离地面远一点
  const Eigen::Vector3d nrm = tilt * Eigen::Vector3d::UnitZ(), tan = tilt * Eigen::Vector3d::UnitX();
  VectorXd q = M.qpos0;
  q.segment<3>(0) = centre;
  q.segment<4>(3) << tilt.w(), tilt.x(), tilt.y(), tilt.z();
  q.segment<3>(7) = centre + (0.2 + gap) * nrm;
  q.segment<4>(10) << up.w(), up.x(), up.y(), up.z();
  for (int k = 0; k < 2; ++k) {
    VectorXd p = VectorXd::Zero(M.nv);
    p.segment<3>(6) = 0.03 * tan - (k == 1 ? 2 * gap : 0.0) * nrm;
    const double b = std::min(curved_ccd(M, q, p, P.dhat, P.c), limit_ccd(M, q, p, P.c));
    G.set_ccd_tail(false);
    const double a0 = G.ccd(P, q, p);
    G.set_ccd_tail(true);
    G.reset_timers();
    const double a1 = G.ccd(P, q, p);
    EXPECT_EQ(a0, a1) << "direction " << k;
    EXPECT_NEAR(a1, b, 1e-12) << "direction " << k;
    if (k == 0) EXPECT_EQ(a1, 1.0);
    if (k == 1) EXPECT_LT(a1, 0.6);  // 真的截断了
    std::printf("  direction %d: tail %ld, proved %ld, serial %ld\n", k, G.timers().ccd_tail, G.timers().ccd_tail_proved,
                G.timers().ccd_tail_serial);
  }
}

// 分离平面的界（笔记 22 第 3 节）：搅拌机里球 0 贴着叶片 0 的面（间隙 1e-7 m），转子转 theta，
//   球沿法向跟着叶片在它那里的速度走直线。转动的平面几步就停；交不交给 warp 逐位相同、和 CPU 版一致。
TEST(Ccd, PlaneBoundTailMatchesSerial) {
  REQUIRE_GPU();
  const Model M = load("blender.xml");
  StepParams P;
  gpu::Stepper G(M);
  VectorXd q = M.qpos0;
  const double gap = 1e-7, rb = 0.15;
  q.segment<3>(1) = Eigen::Vector3d(rb, 0.0015 + 0.012 + gap, 0.036 + 0.005);  // 球 0 的球心
  {
    for (const double theta : {0.02, 0.06, 0.13}) {
      for (const double sink : {0.0, 2 * gap}) {
        VectorXd p = VectorXd::Zero(M.nv);
        p[0] = theta;                          // 转子
        p[2] = rb * std::sin(theta) - sink;    // 球 0 沿 y 跟着叶片走（sink：再往叶片里压一点）
        const double b = std::min(curved_ccd(M, q, p, P.dhat, P.c), limit_ccd(M, q, p, P.c));
        G.set_ccd_tail(false);
        const double a0 = G.ccd(P, q, p);
        G.set_ccd_tail(true);
        G.reset_timers();
        const double a1 = G.ccd(P, q, p);
        EXPECT_EQ(a0, a1) << "theta " << theta << " sink " << sink;
        // 和 CPU 版只差舍入，但这里被放大了：快停下时二次式的常数项 A = n·m - level 是两个约 0.012 m（球的半径）
        //   的数之差，只剩 1e-10 m 左右，1e-18 m 的舍入（GPU 合并乘加，CPU 不合并）就是 1e-8 的相对误差，
        //   最后一步的步长跟着差。两边的结果都是安全的；交不交给 warp 仍然逐位相同（上面）。
        EXPECT_NEAR(a1, b, 1e-8 * std::max(1.0, b)) << "theta " << theta << " sink " << sink;
        EXPECT_LT(a1, 1.0) << "theta " << theta << " sink " << sink;
        std::printf("  theta %.2f sink %.0e: alpha %.6f, tail %ld, proved %ld, serial %ld\n", theta, sink, a1,
                    G.timers().ccd_tail, G.timers().ccd_tail_proved, G.timers().ccd_tail_serial);
      }
    }
  }
}

// ---------- 整步 ----------

// GPU 和 CPU 跑同一段，最后的状态一致（只差舍入的累积），每步的 Newton 次数一样。
// GPU 的两种 Newton 迭代都比：CUDA graph 版（默认）和逐个发射的。small_buffers 时把缓冲区容量改得很小，
// 让图版走“放不下、扩容、重录、重做这次迭代”的路径。
void expect_same_run(const Model& M, const StepParams& P, VectorXd q, VectorXd v, int steps, double tol,
                     const char* what, bool small_buffers = false) {
  for (bool graphs : {true, false}) {
    gpu::Stepper G(M);
    G.set_graphs(graphs);
    if (small_buffers) G.set_capacities(2, 2, 4, 4);
    VectorXd qc = q, vc = v, qg = q, vg = v;
    SolverState Sc, Sg;
    int newton_diff = 0;
    for (int i = 0; i < steps; ++i) {
      const StepResult rc = step(M, P, qc, vc, &Sc);
      const StepResult rg = G.step(P, qg, vg, &Sg);
      newton_diff += rc.newton != rg.newton;
      qc = rc.q;
      vc = rc.v;
      qg = rg.q;
      vg = rg.v;
      EXPECT_GT(rg.min_distance, 0.0) << what << " step " << i << " graphs " << graphs;
      if (graphs && G.set_cooperative_pcg(true)) EXPECT_TRUE(G.used_graphs()) << what << " step " << i;
    }
    EXPECT_LT((qg - qc).norm(), tol) << what << " graphs " << graphs;
    EXPECT_LE(newton_diff, steps / 20) << what << " graphs " << graphs;
  }
}

TEST(Step, BoxDropMatchesCpu) {
  REQUIRE_GPU();
  const Model M = load("drop.xml");
  expect_same_run(M, StepParams{}, M.qpos0, VectorXd::Zero(6), 400, 1e-8, "drop");
}

// 焊在世界上的底座（models/welded_base.xml）：自由盒子带着焊在它下面的球落到地面，底座的盒子在地面下 0.03 mm。
TEST(Step, WeldedBaseMatchesCpu) {
  REQUIRE_GPU();
  const Model M = load("welded_base.xml");
  expect_same_run(M, StepParams{}, M.qpos0, VectorXd::Zero(M.nv), 100, 1e-8, "welded base");
}

TEST(Step, StackMatchesCpu) {
  REQUIRE_GPU();
  const Model M = load("stack.xml");
  expect_same_run(M, StepParams{}, M.qpos0, VectorXd::Zero(12), 300, 1e-6, "stack");
}

TEST(Step, RagdollMatchesCpu) {
  REQUIRE_GPU();
  const Model M = load("ragdoll.xml");
  expect_same_run(M, StepParams{}, M.qpos0, VectorXd::Zero(M.nv), 200, 1e-5, "ragdoll");
}

// 阶段 12：驱动、阻尼、armature（drives.xml，控制量按关键帧插值），GPU 和 CPU 逐步一致。drives.xml 没有顶点，
// 图版用不上（use_graph 要求有顶点），两种设置走的都是逐个发射；图版里的驱动由齿轮、夹爪的场景测试来比。
// G1 之后：关节的干摩擦和 actuatorfrcrange（joint_friction.xml）。它有一个球（碰不到别的东西），所以图版用得上。
TEST(Step, DrivesMatchCpu) {
  REQUIRE_GPU();
  for (const char* file : {"drives.xml", "joint_friction.xml"}) {
    const Model M = load(file);
    for (bool graphs : {true, false}) {
      gpu::Stepper G(M);
      G.set_graphs(graphs);
      StepParams P;
      P.h = M.timestep;
      P.eps_v = 0.01;
      VectorXd qc = M.qpos0, vc = VectorXd::Zero(M.nv), qg = qc, vg = vc;
      SolverState Sc, Sg;
      for (int i = 0; i < 300; ++i) {
        P.ctrl = ctrl_at(M, (i + 1) * P.h);
        const StepResult rc = step(M, P, qc, vc, &Sc);
        const StepResult rg = G.step(P, qg, vg, &Sg);
        EXPECT_EQ(rc.newton, rg.newton) << file << " step " << i << " graphs " << graphs;
        qc = rc.q;
        vc = rc.v;
        qg = rg.q;
        vg = rg.v;
        ASSERT_LT((qg - qc).norm(), 1e-9) << file << " step " << i << " graphs " << graphs;
      }
      if (graphs && M.nvertex() > 0 && G.set_cooperative_pcg(true)) EXPECT_TRUE(G.used_graphs()) << file;
    }
  }
}

// 机构场景（笔记 notes/13_mechanisms.md）：夹爪夹起方块，GPU 两种 Newton 迭代都和 CPU 逐步接近（接触、摩擦、
// 三个 position 驱动）。夹爪有顶点，图版确实用上了：驱动在 CUDA graph 里也对。
TEST(Mechanism, GripperMatchesCpu) {
  REQUIRE_GPU();
  const Model M = load("gripper.xml");
  for (bool graphs : {true, false}) {
    gpu::Stepper G(M);
    G.set_graphs(graphs);
    StepParams P;
    P.h = M.timestep;
    P.mu = 0.5;
    VectorXd qc = M.qpos0, vc = VectorXd::Zero(M.nv), qg = qc, vg = vc;
    SolverState Sc, Sg;
    int newton_diff = 0;
    for (int i = 0; i < 250; ++i) {
      P.ctrl = ctrl_at(M, (i + 1) * P.h);
      const StepResult rc = step(M, P, qc, vc, &Sc);
      const StepResult rg = G.step(P, qg, vg, &Sg);
      newton_diff += rc.newton != rg.newton;
      qc = rc.q;
      vc = rc.v;
      qg = rg.q;
      vg = rg.v;
      EXPECT_GT(rg.min_distance, 0.0) << "step " << i;
      if (graphs && G.set_cooperative_pcg(true)) EXPECT_TRUE(G.used_graphs()) << "step " << i;
    }
    EXPECT_LT((qg - qc).norm(), 1e-5) << "graphs " << graphs;
    EXPECT_LE(newton_diff, 250 / 20) << "graphs " << graphs;
    EXPECT_GT(qg[2], 0.15);  // 方块被提起来了
  }
}

// 齿轮组（tools/make_gears.py）：GPU 跑 1 s，不穿模；一样的齿轮相邻啮合，第 k 个转了主动轮的 (-1)^k 倍
// （差一个侧隙，所以比 1 小一点）；主动轮最后接近目标转速 -100°/s。CPU 上一步约 0.25 s，所以只在 GPU 上跑。
TEST(Mechanism, GearsTransmit) {
  REQUIRE_GPU();
  const Model M = load("gears.xml");
  ASSERT_EQ(M.nv, 4);
  gpu::Stepper G(M);
  StepParams P;
  P.h = M.timestep;
  P.mu = 0;
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  SolverState S;
  double dmin = 1e300;
  for (int i = 0; i < 100; ++i) {
    P.ctrl = ctrl_at(M, (i + 1) * P.h);
    const StepResult r = G.step(P, q, v, &S);
    dmin = std::min(dmin, r.min_distance);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(q[0], -1.4);  // 1 s 里主动轮转了 80° 以上（0.2 s 加速）
  for (int k = 1; k < 4; ++k) {
    const double ratio = q[k] / q[0];
    EXPECT_LT(ratio * (k % 2 ? -1 : 1), 1.0) << "gear " << k;
    EXPECT_GT(ratio * (k % 2 ? -1 : 1), 0.9) << "gear " << k;
  }
  EXPECT_NEAR(v[0], -100 * 3.14159265358979323846 / 180, 0.05);
}

// 容量从很小开始：每种缓冲区都要在 Newton 迭代中间扩容（图版是作废这次迭代、扩容、重录、重做），结果照样和 CPU 一致。
TEST(Step, GrowsBuffersMidStep) {
  REQUIRE_GPU();
  const Model M = load("stack.xml");
  expect_same_run(M, StepParams{}, M.qpos0, VectorXd::Zero(12), 300, 1e-6, "stack, small buffers", true);
}

// 8 个网格布娃娃（libuipc 的对比场景，d̂ = 1 cm，半隐式停止）：GPU 跑 150 步，每步结束都不穿透。
TEST(Step, MeshPileStaysSeparated) {
  REQUIRE_GPU();
  const Model M = load(kMesh8);
  StepParams P;
  P.h = M.timestep;
  P.dhat = 0.01;
  P.eps_v = 0.01;
  P.tol = 0.05;
  P.semi_implicit = true;
  gpu::Stepper G(M);
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  SolverState S;
  for (int i = 0; i < 150; ++i) {
    const StepResult r = G.step(P, q, v, &S);
    ASSERT_GT(r.min_distance, 0.0) << "step " << i;
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(S.kappa, 0.0);
  (void)kCaps8;
}

// ---------------------------------------------------------------------------------------------------------------------
// 笔记 notes/14_visible.md：穿透会改变拓扑的两个场景（tools/make_stress.py 生成）。

// 两个闭合多边形的环绕数（同 tools/topology.py 的 linking_number）：沿一个一般的方向投影，
// 数两条曲线之间的交叉，上面那段的方向 x 下面那段的方向和视线同向记 +1，总和除以 2。
static double linking_number(const std::vector<Vector3d>& P, const std::vector<Vector3d>& Q) {
  const Vector3d view = Vector3d(0.3127, 0.5413, 0.7805).normalized();
  const Vector3d e1 = view.cross(Vector3d::UnitX()).normalized(), e2 = view.cross(e1);
  const auto flat = [&](const Vector3d& x) { return Eigen::Vector2d(x.dot(e1), x.dot(e2)); };
  const int nP = static_cast<int>(P.size()), nQ = static_cast<int>(Q.size());
  int total = 0;
  for (int i = 0; i < nP; ++i) {
    const Vector3d &A0 = P[i], &A1 = P[(i + 1) % nP];
    const Eigen::Vector2d a0 = flat(A0), ua = flat(A1) - a0;
    for (int j = 0; j < nQ; ++j) {
      const Vector3d &B0 = Q[j], &B1 = Q[(j + 1) % nQ];
      const Eigen::Vector2d b0 = flat(B0), ub = flat(B1) - b0;
      const double den = ua.x() * ub.y() - ua.y() * ub.x();
      if (std::abs(den) < 1e-18) continue;
      const Eigen::Vector2d w = b0 - a0;
      const double s = (w.x() * ub.y() - w.y() * ub.x()) / den, t = (w.x() * ua.y() - w.y() * ua.x()) / den;
      if (s < 0 || s >= 1 || t < 0 || t >= 1) continue;
      const bool p_over = (A0 + s * (A1 - A0)).dot(view) > (B0 + t * (B1 - B0)).dot(view);
      const Vector3d up = p_over ? Vector3d(A1 - A0) : Vector3d(B1 - B0), dn = p_over ? Vector3d(B1 - B0) : Vector3d(A1 - A0);
      total += up.cross(dn).dot(view) > 0 ? 1 : -1;
    }
  }
  return total / 2.0;
}

// 锁链（models/ring_chain.xml）：24 个环，每个环 12 段胶囊。跑 0.3 s（150 步，h = 2 ms）：
// 每 10 步算一次所有环对的环绕数，和起点比，一对都不能变；每步结束都不穿透。
// MuJoCo 用默认的接触参数时，这一段里（0.245 s）就有一对环互相穿过、脱开了（笔记第 3 节）。
TEST(Topology, RingChainStaysLinked) {
  REQUIRE_GPU();
  const Model M = load("ring_chain.xml");
  // 环：有 12 条边的刚体。胶囊的边是 (轴的终点, 起点)，按边的顺序取起点就是环的 12 边形。
  //   胶囊端点的退化边（v0 == v1，点–边接触用）不是环的边，跳过。
  std::vector<std::vector<int>> ring_verts(M.nbody());
  for (const Edge& e : M.edges)
    if (e.v0 != e.v1) ring_verts[e.body].push_back(e.v1);
  std::vector<int> rings;
  for (int b = 1; b < M.nbody(); ++b)
    if (ring_verts[b].size() == 12) rings.push_back(b);
  ASSERT_EQ(rings.size(), 24u);
  const int n = static_cast<int>(rings.size());
  const auto links = [&](const VectorXd& q) {
    const Frames F = forward_kinematics(M, q);
    std::vector<std::vector<Vector3d>> poly(n);
    for (int i = 0; i < n; ++i)
      for (int v : ring_verts[rings[i]]) poly[i].push_back(world_point(F, rings[i], M.vertices[v].local));
    MatrixXd L = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) L(i, j) = linking_number(poly[i], poly[j]);
    return L;
  };
  StepParams P;
  P.h = 0.002;
  P.mu = 0.3;
  P.eps_v = 0.01;
  P.tol = 0.002;
  P.semi_implicit = true;
  gpu::Stepper G(M);
  VectorXd q = M.key_qpos, v = VectorXd::Zero(M.nv);
  const MatrixXd L0 = links(q);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) EXPECT_EQ(std::abs(L0(i, j)), j == i + 1 ? 1.0 : 0.0) << "rings " << i << " " << j;
  SolverState S;
  double vmax = 0;
  for (int k = 0; k < 150; ++k) {
    P.ctrl = ctrl_at(M, (k + 1) * P.h);
    const StepResult r = G.step(P, q, v, &S);
    ASSERT_GT(r.min_distance, 0.0) << "step " << k;
    q = r.q;
    v = r.v;
    vmax = std::max(vmax, v.cwiseAbs().maxCoeff());
    if ((k + 1) % 10 == 0) {
      const MatrixXd L = links(q);
      ASSERT_EQ((L - L0).cwiseAbs().maxCoeff(), 0.0) << "linking changed by step " << k + 1;
    }
  }
  EXPECT_GT(vmax, 3.0);  // 链的末端已经甩起来了（这一段测的不是静止的链）
}

// 搅拌机（models/blender.xml）：前 0.24 s，转子加速到约 100 rad/s，球落到 3 mm 厚的叶片上，
// 一半以上被打到 5 m/s 以上（最快约 30 m/s），撞 4 mm 厚的墙。每个球都还在罐子里，每步结束都不穿透。
// MuJoCo 用默认的接触参数时，0.222 s 第一个球穿出了罐子（笔记第 4 节）。
TEST(Topology, BlenderKeepsBallsInside) {
  REQUIRE_GPU();
  const Model M = load("blender.xml");
  // 罐子的尺寸同 tools/make_stress.py：墙的内表面半径 0.3 m、厚 4 mm（正 12 边形），盖子上表面 0.354 m。
  const double r_out = 0.304, z_top = 0.354, sector = 2 * kPi / 12;
  std::vector<int> balls;  // 球：带自由关节的刚体（转子是铰链）
  for (int b = 1; b < M.nbody(); ++b)
    if (M.bodies[b].joint >= 0 && M.joints[M.bodies[b].joint].type == JointType::kFree) balls.push_back(b);
  ASSERT_EQ(balls.size(), 20u);
  StepParams P;
  P.h = 0.002;
  P.mu = 0.3;
  P.eps_v = 0.01;
  P.tol = 0.002;
  P.semi_implicit = true;
  gpu::Stepper G(M);
  VectorXd q = M.key_qpos, v = VectorXd::Zero(M.nv);
  SolverState S;
  std::vector<double> top_speed(balls.size(), 0.0);
  double rotor_max = 0.0;
  for (int k = 0; k < 120; ++k) {
    P.ctrl = ctrl_at(M, (k + 1) * P.h);
    const StepResult r = G.step(P, q, v, &S);
    ASSERT_GT(r.min_distance, 0.0) << "step " << k;
    q = r.q;
    v = r.v;
    const Frames F = forward_kinematics(M, q);
    for (size_t i = 0; i < balls.size(); ++i) {
      const Vector3d& x = F.xpos[balls[i]];
      const double phi = std::round(std::atan2(x.y(), x.x()) / sector) * sector;
      ASSERT_LT(x.x() * std::cos(phi) + x.y() * std::sin(phi), r_out) << "ball " << i << " step " << k;
      ASSERT_LT(x.z(), z_top) << "ball " << i << " step " << k;
      ASSERT_GT(x.z(), 0.0) << "ball " << i << " step " << k;
      const int d = M.joints[M.bodies[balls[i]].joint].dadr;  // 自由关节：前 3 个自由度是平移
      top_speed[i] = std::max(top_speed[i], v.segment<3>(d).norm());
    }
    rotor_max = std::max(rotor_max, v[0]);
  }
  // 转子确实转起来了：整段的最大角速度。低于 100 rad/s 是因为惯性项的阻力（阶段 12 笔记第 4 节）让稳态停在约 97，
  // 而且 0.2 s 以后一直在打球。不看最后一步的值：打球是混沌的，GPU 每次的求和顺序不同，0.24 s 时在 64–88 rad/s 之间。
  EXPECT_GT(rotor_max, 80.0);
  EXPECT_GE(std::count_if(top_speed.begin(), top_speed.end(), [](double s) { return s > 5.0; }), 10);
  EXPECT_GT(*std::max_element(top_speed.begin(), top_speed.end()), 15.0);
}

// 摆锤（tools/make_wrecking_ball.py）：30 kg 的钢块挂在 10 节 23 g 的钢条连成的 1 m 长的链上，从水平位置释放。
// 链竖直时摆锤底面比矮墙的墙顶高 2 cm。libuipc 的罚函数关节和仿射体在这里把链拉长约 10 cm，摆锤撞墙；
// 我们的关节就是坐标、刚体由正运动学生成，链长不变，摆锤越过去。h = 10 ms（libuipc 的默认步长）。
// 阶段 13 之前这里 Newton 每步都撞上限（惯性 Hessian 只有 M / h^2，notes/16_inertia_curvature.md），现在每步 2–3 次。
TEST(Drift, WreckingBallClearsWall) {
  REQUIRE_GPU();
  const Model M = load("wrecking_ball.xml");
  const int wb = M.nbody() - 1;  // 摆锤是最后一个刚体（链的末端）
  ASSERT_EQ(M.nbody(), 12);
  // 摆锤的立方体：中心在刚体坐标系的 (0.015 + 0.08, 0, 0)，半边长 0.08；墙：|x| <= 0.03，墙顶 z = 0.3。
  const Vector3d c_local(0.095, 0.0, 0.0);
  const double half = 0.08, wall_x = 0.03, wall_top = 0.3;
  StepParams P;
  P.h = 0.01;
  P.mu = 0.2;
  P.eps_v = 0.01;
  P.tol = 0.01;
  P.semi_implicit = true;
  gpu::Stepper G(M);
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);  // 没有关键帧：从 MJCF 里写的姿态（链水平）开始
  SolverState S;
  double clearance = 1e9;
  bool started_right = false, crossed = false;
  int newton_max = 0;
  for (int k = 0; k < 80; ++k) {
    const StepResult r = G.step(P, q, v, &S);
    ASSERT_GT(r.min_distance, 0.0) << "step " << k;
    newton_max = std::max(newton_max, r.newton);
    q = r.q;
    v = r.v;
    const Frames F = forward_kinematics(M, q);
    const Vector3d c = world_point(F, wb, c_local);
    if (k == 0) started_right = c.x() > 0.5;
    crossed = crossed || c.x() < -half - wall_x;
    double low = 1e9, xmin = 1e9, xmax = -1e9;
    for (int sx = -1; sx <= 1; sx += 2)
      for (int sy = -1; sy <= 1; sy += 2)
        for (int sz = -1; sz <= 1; sz += 2) {
          const Vector3d x = world_point(F, wb, c_local + half * Vector3d(sx, sy, sz));
          low = std::min(low, x.z());
          xmin = std::min(xmin, x.x());
          xmax = std::max(xmax, x.x());
        }
    if (xmin < wall_x && xmax > -wall_x) clearance = std::min(clearance, low - wall_top);  // 摆锤在墙的正上方
  }
  EXPECT_TRUE(started_right);
  EXPECT_TRUE(crossed);           // 0.8 s 内越过了墙
  EXPECT_GT(clearance, 0.005);    // 名义 2 cm；摆锤绕自己的铰链转一点，最低的角低几毫米
  EXPECT_LE(newton_max, 10);      // 阶段 13 之前是 50（上限）
}

// ---------- 原始-对偶障碍（StepParams::primal_dual，GPU 版的 DPd） ----------

namespace {
// drop.xml 的盒子 4 个下角离地 gap，静止；kappa 选成平衡间隙正好是 dstar（同 test_primal_dual.cc）。
VectorXd pd_box_at(const Model& M, double gap) {
  VectorXd q = M.qpos0;
  q[2] = 0.2 + gap;
  return q;
}
double pd_kappa_for_rest_gap(double dstar, double dhat) {
  return 9.81 / (4 * std::abs(barrier(dstar * dstar, dhat * dhat).db) * 2 * dstar);
}
}  // namespace

// test_primal_dual.cc 的 SqueezedGapSameSolutionFewerIterations 搬到 GPU：间隙压到平衡值的 1/10，kappa 固定。
// 原始障碍和原始-对偶两种，GPU 和 CPU 的迭代次数一样、走到同一个点；GPU 上原始-对偶也比原始方法少迭代
// （pd 在 GPU 上真的起了作用）。图版和逐个发射的都比。
TEST(PrimalDual, SqueezedGapMatchesCpu) {
  REQUIRE_GPU();
  const Model M = load("drop.xml");
  const double dhat = 1e-3, dstar = 0.02 * dhat;
  const double kappa = pd_kappa_for_rest_gap(dstar, dhat);
  for (bool graphs : {true, false}) {
    gpu::Stepper G(M);
    G.set_graphs(graphs);
    int newton_g[2] = {0, 0};
    for (int pd : {0, 1}) {
      StepParams P;
      P.adapt_kappa = false;
      P.primal_dual = pd != 0;
      SolverState Sc, Sg;
      Sc.kappa = Sg.kappa = kappa;
      const VectorXd q0 = pd_box_at(M, 0.1 * dstar), v0 = VectorXd::Zero(6);
      const StepResult rc = step(M, P, q0, v0, &Sc);
      const StepResult rg = G.step(P, q0, v0, &Sg);
      newton_g[pd] = rg.newton;
      EXPECT_EQ(rg.newton, rc.newton) << "pd " << pd << " graphs " << graphs;
      EXPECT_LT((rg.q - rc.q).norm(), 1e-9) << "pd " << pd << " graphs " << graphs;
      EXPECT_FALSE(rg.capped) << "pd " << pd << " graphs " << graphs;
      if (graphs && G.set_cooperative_pcg(true)) EXPECT_TRUE(G.used_graphs()) << "pd " << pd;
    }
    EXPECT_LT(newton_g[1], newton_g[0]) << "graphs " << graphs;
  }
}

// 整段仿真（默认参数：原始-对偶、两阶段曲率、分离平面的界）：布娃娃、两个叠着的盒子落地、12 节链条，GPU 和 CPU
// 走同一条轨迹。图版、逐个发射，第三遍把容量改得很小：接触的缓冲区一步中间扩容，lambda 的表换成大表时重新散列，
// 结果不变。链条最吃两阶段曲率（上界在深链上太硬），GPU 在接触内核里算秩 1 行，开关每次迭代写进 DIter::crow。
TEST(Step, DefaultSolverMatchesCpu) {
  REQUIRE_GPU();
  struct Case {
    const char* file;
    int steps;
    double tol;
  };
  for (const Case& k : {Case{"ragdoll.xml", 200, 1e-5}, Case{"stack.xml", 300, 1e-6}, Case{"chain.xml", 300, 1e-4}}) {
    const Model M = load(k.file);
    StepParams P;
    P.h = M.timestep;
    for (int pass = 0; pass < 3; ++pass) {
      const bool graphs = pass != 1;
      gpu::Stepper G(M);
      G.set_graphs(graphs);
      if (pass == 2) G.set_capacities(2, 2, 4, 4);
      VectorXd qc = M.qpos0, vc = VectorXd::Zero(M.nv), qg = qc, vg = vc;
      SolverState Sc, Sg;
      int newton_diff = 0;
      long newton_c = 0, newton_g = 0;
      for (int i = 0; i < k.steps; ++i) {
        const StepResult rc = step(M, P, qc, vc, &Sc);
        const StepResult rg = G.step(P, qg, vg, &Sg);
        newton_diff += rc.newton != rg.newton;
        newton_c += rc.newton;
        newton_g += rg.newton;
        qc = rc.q;
        vc = rc.v;
        qg = rg.q;
        vg = rg.v;
        EXPECT_GT(rg.min_distance, 0.0) << k.file << " step " << i << " pass " << pass;
        if (graphs && G.set_cooperative_pcg(true)) EXPECT_TRUE(G.used_graphs()) << k.file << " step " << i;
      }
      std::printf("[default gpu] %s pass %d: newton CPU %ld GPU %ld, steps differing %d, |dq| %.3g\n", k.file, pass,
                  newton_c, newton_g, newton_diff, (qg - qc).norm());
      EXPECT_LE(newton_diff, k.steps / 10) << k.file << " pass " << pass;
      EXPECT_LT((qg - qc).norm(), k.tol) << k.file << " pass " << pass;
    }
  }
}
