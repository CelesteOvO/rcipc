// 阶段 9：求解器（笔记 notes/09_solver.md）。
//
//   Tree.*          第 1 节：树的编号
//   Rows.*          第 2 节：行形式和稠密 Hessian 一致（障碍行、摩擦行）
//   Pcg.*           第 3 节：PCG 的解和 LDLT 一致；单棵树时预条件器精确，一次收敛
//   Stopping.*      第 5 节：半隐式停止
//   Kappa.*         第 4 节：kappa 按间隙调整
//   Simulation.*    PCG 和 LDLT 跑同一个场景，结果一致
#include "rcipc/solver.h"

#include <cmath>

#include <gtest/gtest.h>

#include "rcipc/barrier.h"
#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/friction.h"
#include "rcipc/inertia.h"
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

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// 堆叠场景里上面的盒子落到距离 gap 处（转 45 度），再加一点倾斜：两棵树之间有边–边接触，地面有顶点–平面接触。
VectorXd stacked_pose(const Model& M, double gap) {
  VectorXd q = M.qpos0;
  q[7 + 2] = 0.1005 + 0.1 + gap + 0.1;
  quat_store(Quaterniond(Eigen::AngleAxisd(0.7, Vector3d(0, 0, 1)) * Eigen::AngleAxisd(0.05, Vector3d(1, 0, 0))),
             &q[7 + 3]);
  return q;
}

// 在姿态 q 上组装 Newton 系统（惯性 + 曲率上界 + 障碍行 + 摩擦行），同时返回稠密的参照 Hessian。
struct Assembled {
  NewtonSystem sys;
  MatrixXd dense_ref;
};
Assembled assemble(const Model& M, const VectorXd& q, double kappa, double dhat, double h, double mu, double eps_v,
                   const VectorXd& q_prev_for_friction) {
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  const Frames Fp = forward_kinematics(M, q_prev_for_friction);
  const std::vector<FrictionContact> FC = lag_friction(M, Fp, find_contacts(M, Fp, dhat), kappa, dhat);
  Assembled a;
  a.sys.Hk = inertia_hessian(M, F, h) + contact_curvature(M, q, C, kappa, dhat, 1.0);
  a.sys.rows = contact_rows(M, F, C, kappa, dhat);
  const std::vector<Row> fr = friction_rows(M, F, FC, h, mu, eps_v);
  a.sys.rows.insert(a.sys.rows.end(), fr.begin(), fr.end());
  a.dense_ref = a.sys.Hk + contact_hessian(M, F, C, kappa, dhat) + friction_hessian(M, F, FC, h, mu, eps_v);
  return a;
}

}  // namespace

// ---------- 第 7 节：障碍行的权重 ----------

// b'' + b' / (2 d^2) 在 (0, dhat) 上处处为正（模型仍然半正定）；d 很小时它约是 b'' 的一半（GN 偏硬一倍）；
// 它就是 b(d^2) 对 d 的二阶导数除以 (2d)^2，用有限差分核对。
TEST(Weight, ExactNormalCurvatureIsPositiveAndHalfOfGaussNewtonNearContact) {
  const double dhat = 1e-3, eps = 1e-9;
  for (int i = 1; i < 1000; ++i) {
    const double d = dhat * i / 1000.0;
    const double w = barrier_row_weight(d, dhat);
    EXPECT_GT(w, 0.0) << "d " << d;
    const auto b = [&](double dd) { return barrier(dd * dd, dhat * dhat).b; };
    const double d2 = (b(d + eps) - 2 * b(d) + b(d - eps)) / (eps * eps);  // d^2/dd^2 b(d^2)
    EXPECT_NEAR(w, d2 / (4 * d * d), 1e-3 * std::abs(d2 / (4 * d * d)) + 1e-6) << "d " << d;
  }
  const double d = 1e-6;
  EXPECT_NEAR(barrier_row_weight(d, dhat) / barrier(d * d, dhat * dhat).ddb, 0.5, 0.01);
}

// ---------- 第 1 节：树 ----------

// zoo.xml：两棵树（torso 和 post）；每个自由度的树和它所在刚体的树一致；stack.xml 两棵树各 6 个自由度。
TEST(Tree, BodiesAndDofsAreGrouped) {
  const Model Z = load("zoo.xml");
  EXPECT_EQ(Z.ntree, 2);
  EXPECT_EQ(Z.body_tree[0], -1);
  for (int d = 0; d < Z.nv; ++d) EXPECT_EQ(Z.dof_tree[d], Z.body_tree[Z.dof_body[d]]);
  int n0 = 0, n1 = 0;
  for (int d = 0; d < Z.nv; ++d) (Z.dof_tree[d] == 0 ? n0 : n1)++;
  EXPECT_EQ(n0 + n1, Z.nv);
  EXPECT_EQ(n1, 4);  // post（铰链 1）+ bob（球关节 3）
  const Model S = load("stack.xml");
  EXPECT_EQ(S.ntree, 2);
  for (int d = 0; d < 6; ++d) EXPECT_EQ(S.dof_tree[d], 0);
  for (int d = 6; d < 12; ++d) EXPECT_EQ(S.dof_tree[d], 1);
}

// ---------- 第 2 节：行形式 ----------

// H_k + sum D r^T r 和稠密的“惯性 + 曲率 + 障碍 Hessian + 摩擦 Hessian”一致；apply 和稠密乘向量一致。
TEST(Rows, RowFormEqualsDenseHessian) {
  const Model M = load("stack.xml");
  const double dhat = 2e-3, kappa = 1e9, h = 0.002, mu = 0.3, eps_v = 1e-3;
  const VectorXd q = stacked_pose(M, 0.5e-3);
  VectorXd d = VectorXd::Zero(M.nv);
  d[6] = 1e-6;
  d[8] = -0.5e-6;  // 让摩擦有一点相对速度
  const Assembled a = assemble(M, integrate(M, q, d), kappa, dhat, h, mu, eps_v, q);
  ASSERT_GT(a.sys.rows.size(), 4u);
  const MatrixXd H = dense(a.sys);
  EXPECT_LT((H - a.dense_ref).norm(), 1e-9 * a.dense_ref.norm());
  VectorXd p(M.nv);
  for (int i = 0; i < M.nv; ++i) p[i] = std::sin(1.0 + i);
  EXPECT_LT((apply(a.sys, p) - a.dense_ref * p).norm(), 1e-9 * (a.dense_ref * p).norm());
  // 跨树的行：至少有一行在两棵树上都非零。
  int cross = 0;
  for (const Row& r : a.sys.rows) cross += r.r.head(6).norm() > 0 && r.r.tail(6).norm() > 0;
  EXPECT_GT(cross, 0);
}

// ---------- 第 3 节：PCG ----------

// 两个盒子（跨树接触）：PCG（tol 1e-12）的解和 LDLT 的解一致；需要不止一次迭代。
TEST(Pcg, MatchesDenseSolveWithCrossTreeRows) {
  const Model M = load("stack.xml");
  const double dhat = 2e-3, kappa = 1e9, h = 0.002;
  const VectorXd q = stacked_pose(M, 0.5e-3);
  const Assembled a = assemble(M, q, kappa, dhat, h, 0.3, 1e-3, q);
  VectorXd g(M.nv);
  for (int i = 0; i < M.nv; ++i) g[i] = std::cos(0.3 * i) * 100;
  const VectorXd x_ref = a.dense_ref.ldlt().solve(-g);
  const PcgResult sol = pcg(a.sys, build_preconditioner(M, a.sys), -g, 1e-12, 200);
  EXPECT_TRUE(sol.converged);
  EXPECT_GT(sol.iterations, 1);
  EXPECT_LE(sol.iterations, M.nv);  // CG 最多 nv 次（精确算术）
  EXPECT_LT((sol.x - x_ref).norm(), 1e-6 * x_ref.norm());
  // 松一点的容差（默认 1e-4）：方向仍然很接近。
  const PcgResult loose = pcg(a.sys, build_preconditioner(M, a.sys), -g, 1e-4, 200);
  EXPECT_LT((loose.x - x_ref).norm(), 5e-2 * x_ref.norm());
}

// 单棵树（落地的盒子，只有地面接触）：预条件块就是 H 本身，PCG 一次收敛，解和 LDLT 一致到舍入误差。
TEST(Pcg, ExactInOneIterationForASingleTree) {
  const Model M = load("drop.xml");
  VectorXd q = M.qpos0;
  q[2] = 0.2 + 0.4e-3;
  const Assembled a = assemble(M, q, 1e9, 1e-3, 0.002, 0.3, 1e-3, q);
  ASSERT_GE(a.sys.rows.size(), 4u);
  VectorXd g(6);
  g << 1, 2, 3, 0.1, 0.2, 0.3;
  const PcgResult sol = pcg(a.sys, build_preconditioner(M, a.sys), -g, 1e-4, 200);
  EXPECT_EQ(sol.iterations, 1);
  EXPECT_TRUE(sol.converged);
  const VectorXd x_ref = a.dense_ref.ldlt().solve(-g);
  EXPECT_LT((sol.x - x_ref).norm(), 1e-8 * x_ref.norm());
  // 右端为 0：解是 0，不迭代。
  const PcgResult zero = pcg(a.sys, build_preconditioner(M, a.sys), VectorXd::Zero(6), 1e-4, 200);
  EXPECT_EQ(zero.iterations, 0);
  EXPECT_EQ(zero.x.norm(), 0.0);
}

// ---------- 第 4 节：kappa 调整 ----------

// 静止在地面上的盒子：kappa 从初值（夹到上限）一路减半，直到间隙落到 1e-2 dhat 以下附近；仍然力平衡、不穿透。
TEST(Kappa, SoftensUntilTheGapIsInTheBand) {
  const Model M = load("drop.xml");
  StepParams P;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(6);
  double kappa_first = 0;
  for (int i = 0; i < 750; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    if (kappa_first == 0 && r.kappa > 0) kappa_first = r.kappa;
    EXPECT_GT(r.min_distance, 0.0);
    q = r.q;
    v = r.v;
  }
  const KappaRange R = kappa_range(M, P.dhat);
  EXPECT_GT(kappa_first, 0.0);
  EXPECT_LE(S.kappa, 0.01 * R.max);  // 初值被夹在上限（阶段 4），之后一路减半
  EXPECT_GE(S.kappa, R.min);
  const Frames F = forward_kinematics(M, q);
  const double gap = min_distance(M, F);
  EXPECT_GT(gap, 1e-3 * P.dhat);
  EXPECT_LT(gap, 0.3 * P.dhat);  // 减半在 gap > 1e-2 dhat 时触发：间隙从接近 dhat 压到 1e-2 dhat 的量级
  EXPECT_LT(v.norm(), 1e-4);
  // 关掉调整：kappa 保持初值，间隙停在接近 dhat 的地方（阶段 4 的结果）。
  StepParams P2;
  P2.adapt_kappa = false;
  SolverState S2;
  q = M.qpos0;
  v.setZero();
  for (int i = 0; i < 750; ++i) {
    const StepResult r = step(M, P2, q, v, &S2);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(S2.kappa, 100 * S.kappa);
  EXPECT_GT(min_distance(M, forward_kinematics(M, q)), 0.5 * P.dhat);
}

// ---------- 第 5 节：半隐式停止 ----------

// 堆叠场景：半隐式规则下每步的 Newton 次数不多于完全收敛，仍然不穿透，并且至少有一些步是被它停下的。
TEST(Stopping, SemiImplicitStopsEarlierAndStaysSeparated) {
  const Model M = load("stack.xml");
  StepParams P;
  P.semi_implicit = true;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(12);
  long newton_semi = 0;
  int semi_stops = 0;
  double dmin = 1e9;
  for (int i = 0; i < 300; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 8));
    newton_semi += r.newton;
    semi_stops += r.semi_stopped;
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_GT(semi_stops, 0);
  StepParams P2;
  SolverState S2;
  q = M.qpos0;
  v.setZero();
  long newton_full = 0;
  for (int i = 0; i < 300; ++i) {
    const StepResult r = step(M, P2, q, v, &S2);
    newton_full += r.newton;
    q = r.q;
    v = r.v;
  }
  EXPECT_LE(newton_semi, newton_full);
}

// ---------- 第 6 节：CCD 的保证是逐次迭代的 ----------

// kappa 调整把静止的间隙压到 1e-2 dhat 量级以后，“从步首到步末画直线再采样”可能钻到地面以下（那条直线不是真实轨迹，
// 中间的迭代可能先抬起再落下）。CCD 保证的是每一次接受的迭代的轨迹：打开 check_iterates，逐次迭代沿真实轨迹采样，
// 台阶和翻滚的盒子都 > 0；步末的最小距离也 > 0。
TEST(Iterates, EveryAcceptedIterateStaysSeparated) {
  for (const char* file : {"ledge.xml", "drop.xml"}) {
    const Model M = load(file);
    StepParams P;
    P.check_iterates = true;
    SolverState S;
    VectorXd q = M.qpos0, v = VectorXd::Zero(6);
    if (std::string(file) == "drop.xml") {
      v[3] = 6.0;  // 翻滚着落下
      v[4] = 2.0;
    }
    double dmin = 1e9;
    for (int i = 0; i < 750; ++i) {
      const StepResult r = step(M, P, q, v, &S);
      dmin = std::min(dmin, r.iterate_min_distance);
      EXPECT_GT(r.min_distance, 0.0) << file << " step " << i;
      q = r.q;
      v = r.v;
    }
    EXPECT_GT(dmin, 0.0) << file;
  }
}

// ---------- 整段仿真 ----------

// PCG 和 LDLT 跑同一个堆叠场景 300 步：轨迹一致（PCG 的容差 1e-4 带来的差别远小于间隙），都不穿透。
TEST(Simulation, PcgAndLdltAgree) {
  const Model M = load("stack.xml");
  VectorXd q_end[2];
  long pcg_total = 0;
  for (int s = 0; s < 2; ++s) {
    StepParams P;
    P.solver = s == 0 ? LinearSolver::kPcg : LinearSolver::kLdlt;
    SolverState S;
    VectorXd q = M.qpos0, v = VectorXd::Zero(12);
    for (int i = 0; i < 300; ++i) {
      const StepResult r = step(M, P, q, v, &S);
      EXPECT_GT(r.min_distance, 0.0);
      pcg_total += r.pcg;
      q = r.q;
      v = r.v;
    }
    q_end[s] = q;
  }
  EXPECT_LT((q_end[0] - q_end[1]).norm(), 1e-4);
  EXPECT_GT(pcg_total, 0);
}
