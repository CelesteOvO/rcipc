// 阶段 4、5：CCD（笔记 notes/04_ground.md 第 5 节，notes/05_curved_ccd.md）。
//
//   ChordCcd.*      阶段 4：直弦 CCD 的规则，和阶段 1 的一维版本一致
//   Flail.Chord*    阶段 4 的反例：高速转动的杆，两端都在地面上方，弧线却穿过了地面
//   Acceleration.*  阶段 5：每个刚体的加速度上界，和真实轨迹的二阶导数比
//   Sagitta.*       阶段 5：弧线偏离弦不超过 A ds^2 / 8
//   CurvedCcd.*     阶段 5：弦加矢高抓住了反例；纯平移时和直弦一样；整段仿真不穿透
#include "rcipc/ccd.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// 甩杆的几何（flail.xml）：杆端的球心绕铰链转，半径 R = 0.5，球的半径 0.01；铰链离地 H = 0.5075。
// 球面到地面的距离是 (H - 0.01) - R cos(角)（角从正下方量起），最低点 H - 0.01 - R = -2.5 mm。
struct FlailSetup {
  Model M = load("flail.xml");
  double H = 0.5075, R = 0.5, rad = 0.01;
  double phi = std::acos((H - rad) / R);  // 在地面以下的那段弧的半角：acos(0.995) ≈ 0.1 rad
  StepParams P;
  VectorXd q = VectorXd::Zero(1), v = VectorXd::Zero(1);
  FlailSetup() {
    P.h = 0.01;
    q[0] = kPi / 2 - phi - 0.05;  // 起点：杆在“指向正下方”之前 phi + 0.05 rad 处，杆端离地约 3 mm
    v[0] = 40.0;
  }
};

// zoo.xml 的随机姿态（同 test_contact.cc）。
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

VectorXd random_p(int nv, std::mt19937& rng, double scale) {
  std::uniform_real_distribution<double> u(-scale, scale);
  VectorXd p(nv);
  for (int i = 0; i < nv; ++i) p[i] = u(rng);
  return p;
}

}  // namespace

// ---------- 阶段 4：直弦 ----------

// 盒子竖直平移：远离地面不限制；靠近时 alpha = c * d / (d - d_end)，和阶段 1 的 ccd_max_step 一样。
TEST(ChordCcd, MatchesTheOneDimensionalRule) {
  const Model M = load("drop.xml");
  const VectorXd q = M.qpos0;  // 下面 4 个角离地 0.5
  VectorXd up = VectorXd::Zero(6), down = VectorXd::Zero(6);
  up[2] = 0.3;
  down[2] = -1.0;  // 整步会穿过地面
  const Frames F = forward_kinematics(M, q);
  EXPECT_EQ(chord_ccd_ground(M, F, forward_kinematics(M, integrate(M, q, up)), 0.9), 1.0);
  EXPECT_NEAR(chord_ccd_ground(M, F, forward_kinematics(M, integrate(M, q, down)), 0.9), 0.9 * 0.5 / 1.0, 1e-12);
  // 整步闭合不到 c 倍时不限制：往下 0.3，闭合 60% < 90%。
  down[2] = -0.3;
  EXPECT_EQ(chord_ccd_ground(M, F, forward_kinematics(M, integrate(M, q, down)), 0.9), 1.0);
  // 不是从 1 开始截：往下 0.49（闭合 98%），alpha = 0.9 * 0.5 / 0.49。
  down[2] = -0.49;
  EXPECT_NEAR(chord_ccd_ground(M, F, forward_kinematics(M, integrate(M, q, down)), 0.9), 0.9 * 0.5 / 0.49, 1e-12);
}

// 走到 CCD 允许的步长处，最近的顶点离地正好是 (1 - c) 倍的起始距离。
TEST(ChordCcd, StopsAtOneMinusCOfTheGap) {
  const Model M = load("drop.xml");
  const VectorXd q = M.qpos0;
  VectorXd p = VectorXd::Zero(6);
  p[2] = -2.0;
  p[3] = 0.3;  // 加一点转动，4 个角不再同高
  const Frames F0 = forward_kinematics(M, q);
  const double alpha = chord_ccd_ground(M, F0, forward_kinematics(M, integrate(M, q, p)), 0.9);
  ASSERT_LT(alpha, 1.0);
  // 平移是线性的，转动是弧线：弧线上的距离和弦上的距离略有不同，所以这里只要求数量级一致，
  // 并且真实轨迹上没有穿透（转得慢，矢高很小）。
  const double d_end = min_distance(M, forward_kinematics(M, integrate(M, q, p, alpha)));
  EXPECT_NEAR(d_end, 0.1 * 0.5, 0.02);
  EXPECT_GT(trajectory_min_distance(M, q, p, alpha, 64), 0.0);
}

// 阶段 4 的反例（书 §9.1 的甩杆，改成地面）：杆端以 40 rad/s 从地面上方 3 mm 的位置一步转到另一侧 7 mm 的位置，
// 两端都在地面以上、都不在 dhat 之内，直弦 CCD 和线搜索都放行；真实的圆弧在中间低到地面以下 2.5 mm。
// 这个测试记录的是直弦 CCD 的**失败**（P.ccd = kChord）；下面 CurvedCcd.CatchesTheArc 是阶段 5 的修正。
TEST(Flail, ChordCcdMissesTheArcThroughTheFloor) {
  FlailSetup S;
  S.P.ccd = CcdMode::kChord;
  const double d_start = min_distance(S.M, forward_kinematics(S.M, S.q));
  EXPECT_NEAR(d_start, (S.H - S.rad) - S.R * std::cos(S.phi + 0.05), 1e-9);
  EXPECT_GT(d_start, S.P.dhat);

  std::vector<NewtonIterate> trace;
  const StepResult r = step(S.M, S.P, S.q, S.v, nullptr, &trace);
  // 一步转过的角度（约 0.35 rad：隐式 Euler 的耗散让它比 h w = 0.4 小）跨过了整段 2 phi ≈ 0.2 rad 的弧。
  const double turned = r.q[0] - S.q[0];
  EXPECT_GT(turned, 2 * S.phi + 0.05);
  // 终点在地面之上、dhat 之外。Newton 要迭代十来次（转 0.35 rad 时 FK 的非线性很强，Gauss-Newton 是线性收敛），
  // 但每一次 CCD 都没有截断、线搜索都走了满步：弦从来没有靠近地面。
  EXPECT_GT(r.min_distance, S.P.dhat);
  EXPECT_LT(r.newton, S.P.max_newton);
  ASSERT_FALSE(trace.empty());
  for (const NewtonIterate& it : trace) {
    EXPECT_EQ(it.alpha_max, 1.0);
    EXPECT_EQ(it.alpha, 1.0);
  }
  // 但真实轨迹的最低点在地面以下：直弦 CCD 漏掉了这次穿透。
  const double dmin = trajectory_min_distance(S.M, S.q, differentiate(S.M, S.q, r.q), 1.0, 64);
  EXPECT_LT(dmin, 0.0);
  EXPECT_NEAR(dmin, S.H - S.rad - S.R, 1e-4);
}

// ---------- 阶段 5：加速度界 ----------

// 单个铰链：A = w^2 R，R 是锚点到顶点的静态力臂（0.5 + 0.01）。
TEST(Acceleration, SingleHingeIsCentripetal) {
  const Model M = load("flail.xml");
  VectorXd p(1);
  p[0] = 0.4;
  const VectorXd A = acceleration_bounds(M, M.qpos0, p);
  EXPECT_NEAR(A[1], 0.4 * 0.4 * 0.51, 1e-12);
  EXPECT_EQ(A[0], 0.0);
  // 自由关节只平移：没有曲率。
  const Model D = load("drop.xml");
  VectorXd pd = VectorXd::Zero(6);
  pd[2] = -3.0;
  EXPECT_EQ(acceleration_bounds(D, D.qpos0, pd)[1], 0.0);
  // 自由关节转动：A = w^2 * 刚体半径。
  pd.setZero();
  pd[4] = 0.5;
  EXPECT_NEAR(acceleration_bounds(D, D.qpos0, pd)[1], 0.25 * D.bodies[1].radius, 1e-12);
}

// zoo.xml（自由、球、铰链、滑动都有，链深 4）：沿 q ⊕ s p 的真实二阶导数（中心差分）在 s = 0, 0.25, ..., 1 处
// 都不超过 A_b，对每个顶点。
TEST(Acceleration, BoundsTheTrueSecondDerivative) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(60);
  const double eps = 1e-4;
  for (int i = 0; i < 5; ++i) {
    const VectorXd q = random_q(M, rng);
    const VectorXd p = random_p(M.nv, rng, 0.5);
    const VectorXd A = acceleration_bounds(M, q, p);
    for (int v = 0; v < M.nvertex(); ++v) {
      const Vertex& V = M.vertices[v];
      for (int k = 0; k <= 4; ++k) {
        const double s = 0.25 * k;
        const auto x = [&](double ss) {
          return world_point(forward_kinematics(M, integrate(M, q, p, ss)), V.body, V.local);
        };
        const Vector3d xpp = (x(s + eps) - 2 * x(s) + x(s - eps)) / (eps * eps);
        EXPECT_LE(xpp.norm(), A[V.body] * (1 + 1e-6) + 1e-6) << "pose " << i << ", vertex " << v << ", s " << s;
      }
    }
  }
}

// ---------- 阶段 5：矢高 ----------

// 甩杆转 0.4 rad：弧线偏离弦最多 R (1 - cos(0.2)) ≈ R 0.4^2 / 8，不超过界 A / 8。
TEST(Sagitta, ArcStaysWithinTheBound) {
  const Model M = load("flail.xml");
  VectorXd p(1);
  p[0] = 0.4;
  const VectorXd q = M.qpos0;
  const double A = acceleration_bounds(M, q, p)[1];
  const Vertex& tip = M.vertices[0].local.norm() > M.vertices[1].local.norm() ? M.vertices[0] : M.vertices[1];
  const Vector3d xa = world_point(forward_kinematics(M, q), tip.body, tip.local);
  const Vector3d xb = world_point(forward_kinematics(M, integrate(M, q, p)), tip.body, tip.local);
  double dev_max = 0;
  for (int k = 0; k <= 64; ++k) {
    const double s = k / 64.0;
    const Vector3d x = world_point(forward_kinematics(M, integrate(M, q, p, s)), tip.body, tip.local);
    dev_max = std::max(dev_max, (x - (xa + s * (xb - xa))).norm());
  }
  EXPECT_NEAR(dev_max, 0.5 * (1 - std::cos(0.2)), 1e-6);  // 球心的半径是 0.5
  EXPECT_LE(dev_max, A / 8);
  EXPECT_GT(dev_max, 0.9 * A / 8);  // 单铰链时界几乎是紧的（差一个 0.5 对 0.51 的半径）
}

// ---------- 阶段 5：曲线 CCD ----------

// 纯平移：没有曲率，曲线 CCD 和直弦 CCD 给出同样的步长。
TEST(CurvedCcd, EqualsChordForPureTranslation) {
  const Model M = load("drop.xml");
  const VectorXd q = M.qpos0;
  VectorXd p = VectorXd::Zero(6);
  p[2] = -1.0;
  const double chord = chord_ccd_ground(M, forward_kinematics(M, q), forward_kinematics(M, integrate(M, q, p)), 0.9);
  EXPECT_NEAR(curved_ccd(M, q, p, 1e-3, 0.9), chord, 1e-12);
  p[2] = 0.3;
  EXPECT_EQ(curved_ccd(M, q, p, 1e-3, 0.9), 1.0);
}

// 阶段 4 的反例，换成曲线 CCD：第一次迭代就被截断，整条真实轨迹都在地面之上。
TEST(CurvedCcd, CatchesTheArcThroughTheFloor) {
  FlailSetup S;
  std::vector<NewtonIterate> trace;
  const StepResult r = step(S.M, S.P, S.q, S.v, nullptr, &trace);
  ASSERT_FALSE(trace.empty());
  EXPECT_LT(trace[0].alpha_max, 1.0);
  EXPECT_GT(r.min_distance, 0.0);
  EXPECT_LT(r.min_distance, S.P.dhat);  // 停在了障碍里
  EXPECT_GT(trajectory_min_distance(S.M, S.q, differentiate(S.M, S.q, r.q), 1.0, 256), 0.0);
  // 杆没有跨过去：终点还在正下方之前。
  EXPECT_LT(r.q[0], kPi / 2);
}

// 甩杆连续 100 步（40 rad/s 撞向地面，之后被弹回或停住）：每一步的真实轨迹都不穿透。
TEST(CurvedCcd, FlailNeverPenetratesOverManySteps) {
  FlailSetup S;
  S.P.adapt_kappa = false;  // 阶段 9 的 kappa 调整会把间隙压到 1e-2 dhat，这里保持阶段 5 的设置
  SolverState state;
  VectorXd q = S.q, v = S.v;
  double dmin = 1e9;
  int nmax = 0;
  for (int i = 0; i < 100; ++i) {
    const StepResult r = step(S.M, S.P, q, v, &state);
    dmin = std::min(dmin, trajectory_min_distance(S.M, q, differentiate(S.M, q, r.q), 1.0, 64));
    nmax = std::max(nmax, r.newton);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(nmax, S.P.max_newton);
}

// 落地的盒子带着转动落下（4 个角不同高）：曲线 CCD 下每步都不穿透。
// （没有摩擦，它落地后还会慢慢滑和晃，所以不检查静止；摩擦是阶段 8。）
TEST(CurvedCcd, TumblingBoxNeverPenetrates) {
  const Model M = load("drop.xml");
  StepParams P;
  P.adapt_kappa = false;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(6);
  v[3] = 6.0;  // 绕 x 轴 6 rad/s
  v[4] = 2.0;
  double dmin = 1e9;
  for (int i = 0; i < 1000; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 32));
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(v.norm(), 1.0);  // 落地时的速度是几 m/s，2 s 后只剩下慢慢的滑动
}
