// 阶段 12：驱动、关节阻尼和 armature（笔记 notes/12_drives.md）。
//
//   Drive.ReadsModel        读模型：阻尼、armature、四个驱动、关键帧
//   Drive.CtrlSchedule      关键帧的分段线性插值
//   Drive.Derivatives       能量、梯度、对角 Hessian 和有限差分一致（含力限制的两侧）
//   Drive.DampingArmature   只有阻尼和 armature 的滑块：每步速度乘以 (m + a) / (m + a + b h)
//   Drive.PositionServo     PD 伺服（带力限制）：和隐式欧拉的闭式递推逐步一致
//   Drive.ForceLimit        motor 的力被夹在 ±2 N：匀加速
//   Drive.VelocityServo     速度伺服：角速度收敛到目标
//   Drive.GravityBalance    摆在重力下由 position 驱动托住：停在力矩平衡的角度
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rcipc/drive.h"
#include "rcipc/model.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::VectorXd;

namespace {

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// 自由度和驱动的编号（drives.xml 里的顺序）。
constexpr int kDamped = 0, kServo = 1, kPushed = 2, kSpin = 3, kArm = 4;

StepParams params(const Model& M) {
  StepParams P;
  P.h = M.timestep;
  P.tol = 1e-9;
  P.pcg_tol = 1e-14;
  return P;
}

}  // namespace

TEST(Drive, ReadsModel) {
  const Model M = load("drives.xml");
  ASSERT_EQ(M.nv, 5);
  ASSERT_EQ(M.nu(), 4);
  EXPECT_EQ(M.dof_damping[kDamped], 3.0);
  EXPECT_EQ(M.dof_armature[kDamped], 0.5);
  // position：gain kp，bias (0, -kp, -kv)；velocity：gain kv，bias (0, 0, -kv)；motor：gain 1，没有 bias。
  const Actuator& s = M.actuators[0];
  EXPECT_EQ(s.dof, kServo);
  EXPECT_EQ(s.gain, 50.0);
  EXPECT_EQ(s.bias[1], -50.0);
  EXPECT_EQ(s.bias[2], -4.0);
  EXPECT_TRUE(s.force_limited);
  const Actuator& v = M.actuators[2];
  EXPECT_EQ(v.gain, 5.0);
  EXPECT_EQ(v.bias[1], 0.0);
  EXPECT_EQ(v.bias[2], -5.0);
  EXPECT_TRUE(M.actuators[1].ctrl_limited);
  ASSERT_EQ(M.ctrl_keys.size(), 2u);
  EXPECT_TRUE(M.ignored.empty());
}

TEST(Drive, CtrlSchedule) {
  const Model M = load("drives.xml");
  EXPECT_NEAR(ctrl_at(M, -1.0)[0], 0.0, 1e-15);
  EXPECT_NEAR(ctrl_at(M, 0.05)[0], 0.075, 1e-15);
  EXPECT_NEAR(ctrl_at(M, 0.2)[0], 0.3, 1e-15);
  EXPECT_NEAR(ctrl_at(M, 5.0)[0], 0.3, 1e-15);
  EXPECT_NEAR(ctrl_at(M, 0.1)[2], 2.0, 1e-15);
}

// 有限差分：每个驱动都取几个点，让 position 驱动的力分别落在范围里和范围外。
TEST(Drive, Derivatives) {
  const Model M = load("drives.xml");
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-1, 1);
  for (int trial = 0; trial < 20; ++trial) {
    VectorXd q0(M.nq), v0(M.nv), q(M.nq);
    for (int i = 0; i < M.nq; ++i) q0[i] = 0.3 * u(rng);
    for (int i = 0; i < M.nv; ++i) v0[i] = u(rng);
    for (int i = 0; i < M.nq; ++i) q[i] = q0[i] + 0.2 * u(rng);
    VectorXd ctrl(4);
    ctrl << 0.3 * u(rng), 30 * u(rng), 2 * u(rng), u(rng);
    const DriveTerms D = drive_terms(M, q0, v0, ctrl, M.timestep);
    const VectorXd g = drive_gradient(M, D, q), H = drive_hessian_diagonal(M, D, q);
    for (int d = 0; d < M.nv; ++d) {
      const double eps = 1e-6;
      VectorXd qp = q, qm = q;
      qp[d] += eps;
      qm[d] -= eps;
      const double fd = (drive_energy(M, D, qp) - drive_energy(M, D, qm)) / (2 * eps);
      EXPECT_NEAR(g[d], fd, 1e-6 * (1 + std::abs(fd))) << "trial " << trial << " dof " << d;
      const double fdh = (drive_gradient(M, D, qp)[d] - drive_gradient(M, D, qm)[d]) / (2 * eps);
      // 力限制的拐点上差分会跨过两段，跳过离拐点太近的点。
      bool kink = false;
      for (const DriveTerms::Act& a : D.act) {
        if (a.dof != d || a.C == 0) continue;
        const double y = a.A + a.C * q[d];
        kink = kink || std::abs(y - a.lo) < 1e-3 || std::abs(y - a.hi) < 1e-3;
      }
      if (!kink) EXPECT_NEAR(H[d], fdh, 1e-4 * (1 + std::abs(fdh))) << "trial " << trial << " dof " << d;
      EXPECT_GE(H[d], 0.0);
    }
  }
}

// 只有阻尼和 armature：(m + a)(v1 - v0) / h = -b v1，所以 v1 = v0 (m + a) / (m + a + b h)。
TEST(Drive, DampingArmature) {
  const Model M = load("drives.xml");
  const StepParams P = params(M);
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  v[kDamped] = 1.0;
  const double m = 2, a = 0.5, b = 3, rho = (m + a) / (m + a + b * P.h);
  double expect = 1.0;
  for (int i = 0; i < 100; ++i) {
    const StepResult r = step(M, P, q, v);
    q = r.q;
    v = r.v;
    expect *= rho;
    ASSERT_NEAR(v[kDamped], expect, 1e-10) << "step " << i;
  }
}

// PD 伺服：m (v1 - v0) / h = clamp(kp (u - x1) - kv v1, -5, 5) - b v1，x1 = x0 + h v1。f 单调，先按不夹解，越界再按常力解。
TEST(Drive, PositionServo) {
  const Model M = load("drives.xml");
  StepParams P = params(M);
  const double m = 1.5, kp = 50, kv = 4, b = 0.5, h = P.h, fmax = 5;
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  double x = 0, w = 0;  // 闭式递推的位置、速度
  int clamped = 0;
  for (int i = 0; i < 800; ++i) {
    P.ctrl = ctrl_at(M, (i + 1) * h);
    const double u = P.ctrl[0];
    double w1 = (m * w / h + kp * (u - x)) / (m / h + kp * h + kv + b);
    const double f = kp * (u - x - h * w1) - kv * w1;
    if (f > fmax) w1 = (m * w / h + fmax) / (m / h + b);
    if (f < -fmax) w1 = (m * w / h - fmax) / (m / h + b);
    clamped += std::abs(f) > fmax;
    x += h * w1;
    w = w1;
    const StepResult r = step(M, P, q, v);
    q = r.q;
    v = r.v;
    ASSERT_NEAR(q[kServo], x, 1e-10) << "step " << i;
  }
  EXPECT_GT(clamped, 0);              // 开始时目标跳得快，力确实被夹过
  EXPECT_NEAR(q[kServo], 0.3, 1e-3);  // 最后停在目标上
}

// motor：控制量 10，力夹在 ±2 N：a = 2 / 4 m/s^2。
TEST(Drive, ForceLimit) {
  const Model M = load("drives.xml");
  StepParams P = params(M);
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  for (int i = 0; i < 50; ++i) {
    P.ctrl = ctrl_at(M, (i + 1) * P.h);
    const StepResult r = step(M, P, q, v);
    q = r.q;
    v = r.v;
  }
  EXPECT_NEAR(v[kPushed], 50 * P.h * 2.0 / 4.0, 1e-10);
}

// 速度伺服：稳态时驱动力矩 kv (u - w) 正好抵掉惯性项的“拖慢”。阶段 3 的惯性目标把每个质点按直线外推
// （x~ = 2 x0 - x_prev），匀速转动时目标比真实的转角落后一点，梯度是 (2 I / h^2)(1 - cos t) sin t，t = h w
// （笔记 12 第 4 节）。所以稳态满足 (2 I / h^2)(1 - cos(h w)) sin(h w) = kv (u - w)，w 比 u 小约 (h w)^2 / 2 倍的 u。
TEST(Drive, VelocityServo) {
  const Model M = load("drives.xml");
  StepParams P = params(M);
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  for (int i = 0; i < 300; ++i) {
    P.ctrl = ctrl_at(M, (i + 1) * P.h);
    const StepResult r = step(M, P, q, v);
    q = r.q;
    v = r.v;
  }
  const double I = 0.05, kv = 5, u = 2, h = P.h;
  double w = u;
  for (int k = 0; k < 50; ++k) {
    const double t = h * w;
    const double f = 2 * I / (h * h) * (1 - std::cos(t)) * std::sin(t) - kv * (u - w);
    const double df = 2 * I / h * (std::sin(t) * std::sin(t) + (1 - std::cos(t)) * std::cos(t)) + kv;
    w -= f / df;
  }
  EXPECT_LT(w, u);
  EXPECT_NEAR(v[kSpin], w, 1e-9);
}

// 摆：质量 1、质心在杆端 0.5 m，position 驱动 kp = 20、目标 0.8 rad，关节阻尼 1。
// 停下来时驱动力矩和重力力矩平衡：kp (u - th) = m g l sin(th)。用牛顿法解出 th 再比较。
TEST(Drive, GravityBalance) {
  const Model M = load("drives.xml");
  StepParams P = params(M);
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  for (int i = 0; i < 1500; ++i) {
    P.ctrl = ctrl_at(M, (i + 1) * P.h);
    const StepResult r = step(M, P, q, v);
    q = r.q;
    v = r.v;
  }
  const double kp = 20, u = 0.8, mgl = 1 * 9.81 * 0.5;
  double th = 0.5;
  for (int k = 0; k < 50; ++k) th -= (kp * (u - th) - mgl * std::sin(th)) / (-kp - mgl * std::cos(th));
  // 绕 y 轴转正角时，杆端（在 -z 方向）向 -x 方向摆起：重力力矩是 -mgl sin(th)。
  EXPECT_NEAR(q[M.joints[kArm].qadr], th, 1e-6);
  EXPECT_NEAR(v[kArm], 0.0, 1e-6);
}
