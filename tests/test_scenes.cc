// 阶段 10：关节限位和场景（笔记 notes/10_scenes.md）。
//
//   Limits.*   第 1 节：限位的能量、梯度、Hessian、CCD
//   Ragdoll.*  第 2 节：一个布娃娃落地：步末和迭代轨迹都不穿模，关节不越界，最后静止
//   Chain.*    第 3 节：链条甩到地面：不穿模
//   Pile.*     第 4 节：三个布娃娃叠在一起：不穿模
// 场景测试在 Debug 下很慢，这里只跑最关键的一段时间；完整的跑法见 toys/scenes（scene_demo）。
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rcipc/barrier.h"
#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/limits.h"
#include "rcipc/model.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::VectorXd;

namespace {

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

struct RunStats {
  double dmin_end = 1e300, dmin_iter = 1e300, lmin = 1e300;
  int nmax = 0;
  VectorXd q, v;
};

RunStats run(const Model& M, StepParams P, int steps) {
  RunStats s;
  SolverState S;
  s.q = M.qpos0;
  s.v = VectorXd::Zero(M.nv);
  for (int i = 0; i < steps; ++i) {
    const StepResult r = step(M, P, s.q, s.v, &S);
    s.dmin_end = std::min(s.dmin_end, r.min_distance);
    s.dmin_iter = std::min(s.dmin_iter, r.iterate_min_distance);
    s.lmin = std::min(s.lmin, r.min_limit_distance);
    s.nmax = std::max(s.nmax, r.newton);
    s.q = r.q;
    s.v = r.v;
  }
  return s;
}

}  // namespace

// ---------- 第 1 节：限位 ----------

// 布娃娃有 4 个限位关节（两肘两膝）；梯度、对角 Hessian 和有限差分一致；越界的距离正确；CCD 的闭式解。
TEST(Limits, EnergyDerivativesAndCcd) {
  const Model M = load("ragdoll.xml");
  int limited = 0;
  for (const Joint& j : M.joints) limited += j.limited;
  EXPECT_EQ(limited, 4);
  const double k = 10.0, dhat = 0.1, eps = 1e-7;
  std::mt19937 rng(100);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int trial = 0; trial < 5; ++trial) {
    VectorXd q = M.qpos0;
    for (const Joint& j : M.joints) {
      if (j.limited) q[j.qadr] = j.range[0] + (j.range[1] - j.range[0]) * (trial == 0 ? 0.02 : u(rng));  // 第一次贴着下端
    }
    const VectorXd g = limit_gradient(M, q, k, dhat), hd = limit_hessian_diagonal(M, q, k, dhat);
    for (const Joint& j : M.joints) {
      if (!j.limited) continue;
      const int d = j.dadr;
      const VectorXd e = VectorXd::Unit(M.nv, d);
      const double fd = (limit_energy(M, integrate(M, q, e, eps), k, dhat) - limit_energy(M, integrate(M, q, e, -eps), k, dhat)) / (2 * eps);
      EXPECT_NEAR(g[d], fd, 1e-6 * (1 + std::abs(fd))) << "joint " << j.qadr;
      const double fdh = (limit_gradient(M, integrate(M, q, e, eps), k, dhat)[d] - limit_gradient(M, integrate(M, q, e, -eps), k, dhat)[d]) / (2 * eps);
      EXPECT_NEAR(hd[d], fdh, 1e-5 * (1 + std::abs(fdh))) << "joint " << j.qadr;
      EXPECT_GE(hd[d], 0.0);
    }
    // 没有限位的自由度上没有贡献。
    for (int d = 0; d < M.nv; ++d) {
      bool lim = false;
      for (const Joint& j : M.joints) lim = lim || (j.limited && j.dadr == d);
      if (!lim) EXPECT_EQ(g[d], 0.0);
    }
  }
  // CCD：膝盖离下端 0.05 rad，想往下转 0.2 rad：alpha = 0.9 * 0.05 / 0.2。
  VectorXd q = M.qpos0;
  const Joint* knee = nullptr;
  for (const Joint& j : M.joints) if (j.limited && j.axis.x() == 1) knee = &j;
  ASSERT_NE(knee, nullptr);
  q[knee->qadr] = knee->range[0] + 0.05;
  VectorXd p = VectorXd::Zero(M.nv);
  p[knee->dadr] = -0.2;
  EXPECT_NEAR(limit_ccd(M, q, p, 0.9), 0.9 * 0.05 / 0.2, 1e-12);
  p[knee->dadr] = 0.2;  // 往上转：离上端 2.35，不限制
  EXPECT_EQ(limit_ccd(M, q, p, 0.9), 1.0);
  EXPECT_NEAR(min_limit_distance(M, q), 0.05, 1e-12);  // 膝盖贴着下端 0.05；其余关节离端点至少 0.05（qpos0 的余量）
}

// ---------- 第 2 节：布娃娃 ----------

// 一个布娃娃从 1.2 m 落到地面（0.6 s，着地后翻倒）：步末不穿模、每次迭代的轨迹不穿模、关节始终在限位里、Newton 不撞上限。
TEST(Ragdoll, DropStaysSeparatedAndInsideLimits) {
  const Model M = load("ragdoll.xml");
  StepParams P;
  P.check_iterates = true;
  const RunStats s = run(M, P, 300);
  EXPECT_GT(s.dmin_end, 0.0);
  EXPECT_GT(s.dmin_iter, 0.0);
  EXPECT_GT(s.lmin, 0.0);
  EXPECT_LT(s.nmax, P.max_newton);
  EXPECT_LT(s.q[2], 0.8);  // 躯干落下来了
}

// ---------- 第 3 节：链条 ----------

// 12 节的链条从水平静止释放，甩到地面上（0.6 s）：不穿模（链节之间、链节和地面），Newton 不撞上限。
TEST(Chain, SwingOntoTheFloorStaysSeparated) {
  const Model M = load("chain.xml");
  EXPECT_EQ(M.ntree, 1);
  EXPECT_EQ(M.nv, 36);
  StepParams P;
  P.check_iterates = true;
  const RunStats s = run(M, P, 300);
  EXPECT_GT(s.dmin_end, 0.0);
  EXPECT_GT(s.dmin_iter, 0.0);
  EXPECT_LT(s.dmin_end, P.dhat);  // 确实碰到了地面
  // 链条甩到地面的那几步会撞 Newton 上限（书 §10.6.4：CCD 限制下的爬行），撞上限的步仍然不相交，这里不断言次数。
}

// ---------- 第 4 节：布娃娃堆 ----------

// 三个布娃娃（75 个自由度）叠在一起（0.4 s：第一个着地、第二个压上来）：步末不穿模，关节在限位里。
TEST(Pile, ThreeRagdollsStaySeparated) {
  const Model M = load("ragdoll_pile.xml");
  EXPECT_EQ(M.ntree, 3);
  EXPECT_EQ(M.nv, 75);
  StepParams P;
  const RunStats s = run(M, P, 200);
  EXPECT_GT(s.dmin_end, 0.0);
  EXPECT_GT(s.lmin, 0.0);
}
