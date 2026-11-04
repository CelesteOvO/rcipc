// 阶段 11：GPU 版和 CPU 版逐项对比（笔记 notes/11_gpu.md 第 7 节）。
//
// GPU 版的每一项都对应 CPU 版的一个函数，所以这里一项一项地比：
//   Fk.*      正运动学（zoo.xml，四种关节）
//   Terms.*   能量、梯度、稠密 Hessian（惯性、接触三种、mollifier、摩擦、曲率上界、限位、世界上的静态几何）
//   Pcg.*     PCG 的解和稠密 LDLT 的解
//   Ccd.*     CCD 的步长上限（地面、刚体之间、网格）
//   Step.*    整步：落地盒子、堆叠、布娃娃，和 CPU 版的轨迹一致；网格布娃娃堆不穿透
// GPU 的原子加让求和顺序不固定，所以都按相对误差比（1e-9 量级），不要求逐位相同。没有显卡时整组跳过。
#include "rcipc/gpu.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/friction.h"
#include "rcipc/inertia.h"
#include "rcipc/kinematics.h"
#include "rcipc/limits.h"
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
gpu::Stepper::Terms cpu_terms(const Model& M, const StepParams& P, const VectorXd& q0, const VectorXd& v0, double kappa,
                              const VectorXd& q) {
  const VectorXd q_prev = integrate(M, q0, v0, -P.h);
  const Frames F0 = forward_kinematics(M, q0);
  const InertiaTargets T = inertia_targets(M, F0, forward_kinematics(M, q_prev), P.h);
  const std::vector<FrictionContact> FC =
      P.mu > 0 ? lag_friction(M, F0, find_contacts(M, F0, P.dhat), kappa, P.dhat) : std::vector<FrictionContact>();
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, P.dhat);
  gpu::Stepper::Terms out;
  out.energy = inertia_energy(M, F, T, P.h) + contact_energy(M, C, kappa, P.dhat) +
               friction_energy(M, F, FC, P.h, P.mu, P.eps_v) + limit_energy(M, q, P.limit_k, P.limit_dhat);
  out.g = inertia_gradient(M, F, T, P.h) + contact_gradient(M, F, C, kappa, P.dhat) +
          friction_gradient(M, F, FC, P.h, P.mu, P.eps_v) + limit_gradient(M, q, P.limit_k, P.limit_dhat);
  out.H = inertia_hessian(M, F, P.h) + contact_curvature(M, q, C, kappa, P.dhat, P.curv_scale) +
          contact_hessian(M, F, C, kappa, P.dhat) + friction_hessian(M, F, FC, P.h, P.mu, P.eps_v);
  out.H.diagonal() += limit_hessian_diagonal(M, q, P.limit_k, P.limit_dhat);
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

void expect_terms_match(const Model& M, const StepParams& P, const State& s, const VectorXd& q, const char* what) {
  gpu::Stepper G(M);
  const gpu::Stepper::Terms g = G.terms(P, s.q, s.v, s.S.kappa, q);
  const gpu::Stepper::Terms c = cpu_terms(M, P, s.q, s.v, s.S.kappa, q);
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

// 布娃娃（球、胶囊带半径，关节限位，深的链）。
TEST(Terms, RagdollWithLimits) {
  REQUIRE_GPU();
  const Model M = load("ragdoll.xml");
  StepParams P;
  const State s = cpu_run(M, P, 300, M.qpos0, VectorXd::Zero(M.nv));
  expect_terms_match(M, P, s, s.q, "ragdoll");
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
// 两个盒子一起绕 y 轴倾斜 0.3 rad（倾斜的面上，三角形的盒子在 z 方向有厚度，扫掠盒才会重叠；水平的面上，
// 平移的滑动早被盒子规则剔掉了），上面的盒子离下面 0.1 mm，沿下面盒子的顶面滑 3 cm：贴着滑的对要走约 330 步。
//   - 只滑：距离一直是 0.1 mm，证得出返回 1；
//   - 滑的同时沿法向沉 0.2 mm：中途会碰上，证不出来，接着串行走完。
// 交不交给 warp，结果都要逐位相同，并且和 CPU 版一致。
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
  long tail = 0, proved = 0, serial = 0;
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
    tail += G.timers().ccd_tail;
    proved += G.timers().ccd_tail_proved;
    serial += G.timers().ccd_tail_serial;
  }
  EXPECT_GT(tail, 0);
  EXPECT_GT(proved, 0);
  EXPECT_GT(serial, 0);
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
