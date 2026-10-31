// 阶段 2：四元数的 exp 和 log（笔记 notes/02_kinematics.md 第 2 节）。
//
// 只用 Eigen 做参照：Eigen::AngleAxisd(角度, 单位轴) 也能把“轴加角度”变成四元数。
#include "rcipc/quat.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

using namespace rcipc;
using Eigen::Quaterniond;
using Eigen::Vector3d;

namespace {

constexpr double kPi = 3.14159265358979323846;

// 两个四元数的分量差（不考虑 q 和 -q 是同一个转动）。
double quat_diff(const Quaterniond& a, const Quaterniond& b) { return (a.coeffs() - b.coeffs()).norm(); }

// 长度在 [0, max_angle) 之间、方向随机的旋转向量。
Vector3d random_rotvec(std::mt19937& rng, double max_angle) {
  std::normal_distribution<double> n;
  std::uniform_real_distribution<double> u(0, max_angle);
  return Vector3d(n(rng), n(rng), n(rng)).normalized() * u(rng);
}

}  // namespace

// exp(0) 是单位四元数，不能是 NaN（0/0）。
TEST(Quat, ExpOfZeroIsIdentity) {
  const Quaterniond q = quat_exp(Vector3d::Zero());
  EXPECT_EQ(q.w(), 1.0);
  EXPECT_EQ(q.vec(), Vector3d::Zero());
}

// exp(theta * n) = (cos(theta/2), sin(theta/2) n)，和 Eigen 的轴角转换一致。
TEST(Quat, ExpMatchesAxisAngle) {
  const Vector3d n = Vector3d(1, -2, 0.5).normalized();
  for (double theta : {0.3, 1.0, 2.5, kPi, 4.0}) {
    const Quaterniond expect(Eigen::AngleAxisd(theta, n));
    EXPECT_LT(quat_diff(quat_exp(theta * n), expect), 1e-15) << "theta = " << theta;
  }
}

// 很小的转角：结果仍是单位四元数，向量部分约等于 w / 2。
TEST(Quat, ExpIsAccurateForTinyAngles) {
  for (double t : {1e-3, 1e-6, 1e-9, 1e-200}) {
    const Vector3d w = Vector3d(0.6, 0, 0.8) * t;
    const Quaterniond q = quat_exp(w);
    EXPECT_NEAR(q.norm(), 1.0, 1e-15) << "t = " << t;
    EXPECT_LT((q.vec() - 0.5 * w).norm(), 1e-6 * t) << "t = " << t;  // 误差是 t^3 / 48 量级
  }
}

// exp 得到的总是单位四元数。
TEST(Quat, ExpIsUnit) {
  std::mt19937 rng(1);
  for (int i = 0; i < 100; ++i) EXPECT_NEAR(quat_exp(random_rotvec(rng, 10.0)).norm(), 1.0, 1e-15);
}

// |w| < pi 时 log(exp(w)) = w。
TEST(Quat, LogInvertsExp) {
  std::mt19937 rng(2);
  for (int i = 0; i < 100; ++i) {
    const Vector3d w = random_rotvec(rng, kPi - 1e-3);
    EXPECT_LT((quat_log(quat_exp(w)) - w).norm(), 1e-13) << "w = " << w.transpose();
  }
}

// log(1) = 0，并且 log 不要求输入是单位四元数。
TEST(Quat, LogOfIdentityAndScaledQuaternions) {
  EXPECT_EQ(quat_log(Quaterniond::Identity()), Vector3d::Zero());
  const Vector3d w(0.3, -0.2, 0.9);
  Quaterniond q = quat_exp(w);
  q.coeffs() *= 3.0;
  EXPECT_LT((quat_log(q) - w).norm(), 1e-14);
}

// q 和 -q 是同一个转动，log 要给出同一个结果。
TEST(Quat, LogIsTheSameForMinusQ) {
  std::mt19937 rng(3);
  for (int i = 0; i < 100; ++i) {
    const Quaterniond q = quat_exp(random_rotvec(rng, kPi - 1e-3));
    Quaterniond minus_q = q;
    minus_q.coeffs() *= -1.0;
    EXPECT_LT((quat_log(minus_q) - quat_log(q)).norm(), 1e-13);
  }
}

// 转角超过 pi 时，log 返回反方向的短转动：exp(theta n) 的 log 是 -(2 pi - theta) n。
TEST(Quat, LogPicksTheShortRotation) {
  const Vector3d n = Vector3d(0.2, 0.9, -0.4).normalized();
  for (double theta : {3.5, 5.0, 2 * kPi - 0.1}) {
    const Vector3d w = quat_log(quat_exp(theta * n));
    EXPECT_LE(w.norm(), kPi + 1e-15) << "theta = " << theta;
    EXPECT_LT((w - (theta - 2 * kPi) * n).norm(), 1e-13) << "theta = " << theta;
  }
}
