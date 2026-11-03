// 阶段 4：直弦 CCD（笔记 notes/04_ground.md 第 5 节）。
//
//   ChordCcd.*   直弦 CCD 的规则：远离不限制、靠近最多闭合 c 倍、和阶段 1 的一维版本一致
//   Flail.*      反例：高速转动的杆，两端都在地面上方，弧线却穿过了地面（阶段 5 要修的）
#include "rcipc/ccd.h"

#include <cmath>

#include <gtest/gtest.h>

#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

}  // namespace

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
  const double d_end = min_ground_distance(M, forward_kinematics(M, integrate(M, q, p, alpha)));
  EXPECT_NEAR(d_end, 0.1 * 0.5, 0.02);
  EXPECT_GT(trajectory_min_distance(M, q, p, alpha, 64), 0.0);
}

// 反例（书 §9.1 的甩杆，改成地面）：杆端以 40 rad/s 从地面上方 3 mm 的位置一步转到另一侧 2 mm 的位置，
// 两端都在地面以上、都不在 dhat 之内，直弦 CCD 和线搜索都放行；真实的圆弧在中间低到地面以下 2.5 mm。
// 这个测试记录的是**失败**：阶段 5 把 CCD 换成弦加矢高以后，EXPECT_LT 改成 EXPECT_GT。
TEST(Flail, ChordCcdMissesTheArcThroughTheFloor) {
  const Model M = load("flail.xml");
  // 杆端的球心绕铰链转，半径 R = 0.5，球的半径 0.01；铰链离地 H = 0.5075。
  // 球面到地面的距离是 (H - 0.01) - R cos(角)（角从正下方量起），最低点 H - 0.01 - R = -2.5 mm。
  const double H = 0.5075, R = 0.5, rad = 0.01;
  const double phi = std::acos((H - rad) / R);  // 在地面以下的那段弧的半角：acos(0.995) ≈ 0.1 rad
  // 起点：杆在“指向正下方”之前 phi + 0.05 rad 处，杆端离地约 3 mm。
  StepParams P;
  P.h = 0.01;
  VectorXd q(1), v(1);
  q[0] = kPi / 2 - phi - 0.05;
  v[0] = 40.0;
  const Frames F0 = forward_kinematics(M, q);
  const double d_start = min_ground_distance(M, F0);
  EXPECT_NEAR(d_start, (H - rad) - R * std::cos(phi + 0.05), 1e-9);
  EXPECT_GT(d_start, P.dhat);

  std::vector<NewtonIterate> trace;
  const StepResult r = step(M, P, q, v, nullptr, &trace);
  // 一步转过的角度（约 0.35 rad：隐式 Euler 的耗散让它比 h w = 0.4 小）跨过了整段 2 phi ≈ 0.2 rad 的弧。
  const double turned = r.q[0] - q[0];
  EXPECT_GT(turned, 2 * phi + 0.05);
  // 终点在地面之上、dhat 之外。Newton 要迭代十来次（转 0.35 rad 时 FK 的非线性很强，Gauss-Newton 是线性收敛），
  // 但每一次 CCD 都没有截断、线搜索都走了满步：弦从来没有靠近地面。
  EXPECT_GT(r.min_distance, P.dhat);
  EXPECT_LT(r.newton, P.max_newton);
  ASSERT_FALSE(trace.empty());
  for (const NewtonIterate& it : trace) {
    EXPECT_EQ(it.alpha_max, 1.0);
    EXPECT_EQ(it.alpha, 1.0);
  }
  // 但真实轨迹的最低点在地面以下：直弦 CCD 漏掉了这次穿透。
  const double dmin = trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 64);
  EXPECT_LT(dmin, 0.0);
  EXPECT_NEAR(dmin, H - rad - R, 1e-4);
}
