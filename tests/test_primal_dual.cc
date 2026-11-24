// 原始-对偶障碍（StepParams::primal_dual，默认打开；scene_demo 的 pd=0 关掉）。推导见 notes/22_primal_dual.md 第 1、2 节。
//
//   PrimalDual.LocalPower*       障碍的局部幂次 g = -d F'/F（第 2.3 节的表；混合权重 w = min(1, 1/g)）
//   PrimalDual.Squeezed*         间隙被压到平衡值的 1/10：解不变，原始-对偶的迭代次数少（第 1 节的一维算例）
//   PrimalDual.Drop*             盒子落地：不穿透、静止、力平衡，和原始方法一样
//   PrimalDual.Truncation*       截断区（d 接近 dhat）：w 很小，退回原始方法，走到同一个点
#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#include "rcipc/barrier.h"
#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::VectorXd;

namespace {

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// drop.xml：1 kg 的盒子，半高 0.2 m。4 个下角离地 gap，静止。
VectorXd box_at(const Model& M, double gap) {
  VectorXd q = M.qpos0;
  q[2] = 0.2 + gap;
  return q;
}

// 选 kappa，让 4 个角在间隙 dstar 上的障碍力之和正好等于重力：静止时的平衡间隙就是 dstar。
double kappa_for_rest_gap(double dstar, double dhat) {
  return 9.81 / (4 * std::abs(barrier(dstar * dstar, dhat * dhat).db) * 2 * dstar);
}

// 障碍的局部幂次 g = -d F'/F。F = -2 kappa b'(d^2) d，取对数再求导得 g = -1 - 2 x b''(x) / b'(x)，x = d^2。
double local_power(double d, double dhat) {
  const BarrierValue b = barrier(d * d, dhat * dhat);
  return -1 - 2 * d * d * b.ddb / b.db;
}

// 从 gap 静止出发，固定 kappa（不调整）走一步。pd = false 是原始障碍。
StepResult squeezed_step(const Model& M, double gap, double kappa, bool pd) {
  StepParams P;
  P.adapt_kappa = false;
  P.primal_dual = pd;
  SolverState S;
  S.kappa = kappa;  // 非 0：不再按梯度平衡初始化
  return step(M, P, box_at(M, gap), VectorXd::Zero(6), &S);
}

}  // namespace

// ---------- 局部幂次 g ----------

// 笔记 22 第 2.3 节的表：对数区 g 约等于 1，截断区很快变大；g = 1 恰在 0.3226 dhat（从这里往外混合权重 w < 1）。
TEST(PrimalDual, LocalPowerMatchesTheTable) {
  const double dhat = 1e-3;
  const double r[] = {0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95};
  const double g[] = {0.997, 0.905, 0.954, 1.745, 7.452, 17.388, 37.359};
  for (int i = 0; i < 7; ++i) EXPECT_NEAR(local_power(r[i] * dhat, dhat), g[i], 1e-3) << "d/dhat = " << r[i];
  EXPECT_LT(local_power(0.320 * dhat, dhat), 1.0);
  EXPECT_GT(local_power(0.325 * dhat, dhat), 1.0);
  EXPECT_NEAR(local_power(0.3226 * dhat, dhat), 1.0, 1e-3);
}

// 两头的极限：d -> 0 时回到纯对数 g = 1；d -> dhat 时 F 是二阶零点，g -> 2d/(dhat - d) - 1。
TEST(PrimalDual, LocalPowerLimits) {
  const double dhat = 1e-3;
  EXPECT_NEAR(local_power(1e-6 * dhat, dhat), 1.0, 1e-3);
  for (double r : {0.99, 0.999}) {
    const double approx = 2 * r / (1 - r) - 1;
    EXPECT_NEAR(local_power(r * dhat, dhat) / approx, 1.0, 0.02) << "d/dhat = " << r;
  }
}

// ---------- 间隙被压扁 ----------

// 笔记 22 第 1 节的一维算例搬到盒子上：kappa 固定，平衡间隙 d* = 0.02 dhat（对数区），从 0.1 d* 静止出发。
// 两种方法解的是同一个方程（右端都是原始梯度），所以走到同一个点；
// 原始方法每次迭代间隙最多翻倍，原始-对偶第一次迭代学到力，之后沿割线走。
TEST(PrimalDual, SqueezedGapSameSolutionFewerIterations) {
  const Model M = load("drop.xml");
  const double dhat = 1e-3, dstar = 0.02 * dhat;
  const double kappa = kappa_for_rest_gap(dstar, dhat);
  const StepResult r0 = squeezed_step(M, 0.1 * dstar, kappa, false);
  const StepResult r1 = squeezed_step(M, 0.1 * dstar, kappa, true);
  std::printf("[squeezed] newton primal %d, primal-dual %d; gap end %.4g m (d* %.4g m)\n", r0.newton, r1.newton,
              r0.min_distance, dstar);
  for (const StepResult* r : {&r0, &r1}) {
    EXPECT_FALSE(r->capped);
    EXPECT_GT(r->min_distance, 0.0);
  }
  // 收敛判据是位移 / h < 1e-6 m/s，位置的差应在 1e-9 m 量级以内。
  EXPECT_LT((r1.q - r0.q).norm(), 1e-9);
  EXPECT_LT(r1.newton, r0.newton);
}

// ---------- 落地 ----------

// 和 test_contact 的 Drop.NeverPenetratesAndRestsInForceBalance 一样（默认参数就是原始-对偶）：
// 轨迹不穿透，最后静止，障碍力之和等于重力，没有一步撞上限。
TEST(PrimalDual, DropRestsInForceBalance) {
  const Model M = load("drop.xml");
  StepParams P;
  ASSERT_TRUE(P.primal_dual);
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(6);
  double dmin = 1e9;
  int nmax = 0;
  for (int i = 0; i < 750; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 64));
    nmax = std::max(nmax, r.newton);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(nmax, P.max_newton);
  EXPECT_LT(v.norm(), 1e-5);
  const std::vector<Contact> C = find_contacts(M, forward_kinematics(M, q), P.dhat);
  ASSERT_EQ(C.size(), 4u);
  double force = 0;
  for (const Contact& c : C) force += S.kappa * std::abs(barrier(c.d * c.d, P.dhat * P.dhat).db) * 2 * c.d;
  EXPECT_NEAR(force, 9.81, 2e-3);
}

// ---------- 截断区 ----------

// 平衡间隙 d* = 0.9 dhat，从 0.7 dhat 出发：整步都在截断区（g 约 4–17，w = 1/g 约 0.06–0.25），
// 组装几乎就是原始的 Hessian。两种方法走到同一个点，都不撞上限。
TEST(PrimalDual, TruncationZoneFallsBackToPrimal) {
  const Model M = load("drop.xml");
  const double dhat = 1e-3, dstar = 0.9 * dhat;
  const double kappa = kappa_for_rest_gap(dstar, dhat);
  const StepResult r0 = squeezed_step(M, 0.7 * dhat, kappa, false);
  const StepResult r1 = squeezed_step(M, 0.7 * dhat, kappa, true);
  std::printf("[truncation] newton primal %d, primal-dual %d; gap end %.4g m\n", r0.newton, r1.newton, r0.min_distance);
  for (const StepResult* r : {&r0, &r1}) EXPECT_FALSE(r->capped);
  EXPECT_LT((r1.q - r0.q).norm(), 1e-9);
}
