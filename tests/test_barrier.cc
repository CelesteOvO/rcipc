// 阶段 1：障碍函数 b(x) = -(x - xhat)^2 ln(x / xhat) 和它的导数（笔记 notes/01_ipc1d.md 第 2 节）。
#include "rcipc/barrier.h"

#include <cmath>

#include <gtest/gtest.h>

using rcipc::barrier;

// 每条性质都在两种尺度上检查：
//   distance：自变量是距离，xhat = dhat = 1 mm（阶段 1 的用法）；
//   squared： 自变量是平方距离，xhat = dhat^2 = 1e-6 m^2（阶段 4 起的用法）。
class BarrierTest : public ::testing::TestWithParam<double> {};
INSTANTIATE_TEST_SUITE_P(Scales, BarrierTest, ::testing::Values(1e-3, 1e-6),
                         [](const auto& info) { return info.param == 1e-3 ? "distance" : "squared"; });

// 激活区之外（x >= xhat），三项都严格等于 0。
TEST_P(BarrierTest, ZeroOutsideActivation) {
  const double xhat = GetParam();
  for (double r : {1.0, 1.5, 10.0}) {
    auto v = barrier(r * xhat, xhat);
    EXPECT_EQ(v.b, 0.0) << "x / xhat = " << r;
    EXPECT_EQ(v.db, 0.0) << "x / xhat = " << r;
    EXPECT_EQ(v.ddb, 0.0) << "x / xhat = " << r;
  }
}

// 激活区之内：b > 0（有能量），b' < 0（力 -b' 把间隙推开），b'' > 0（凸）。
TEST_P(BarrierTest, PositiveRepulsiveConvexInside) {
  const double xhat = GetParam();
  for (double r : {1e-6, 0.01, 0.1, 0.5, 0.9, 0.999}) {
    auto v = barrier(r * xhat, xhat);
    EXPECT_GT(v.b, 0.0) << "x / xhat = " << r;
    EXPECT_LT(v.db, 0.0) << "x / xhat = " << r;
    EXPECT_GT(v.ddb, 0.0) << "x / xhat = " << r;
  }
}

// 在 xhat 处 b、b'、b'' 都趋于 0，所以障碍和外侧的 0 函数 C^2 光滑地接上。
// 设 x = xhat (1 - eps)，则 b ≈ eps^3 xhat^2，b' ≈ -3 eps^2 xhat，b'' ≈ 6 eps。这里取 eps = 1e-6。
TEST_P(BarrierTest, SmoothAtActivation) {
  const double xhat = GetParam();
  const double eps = 1e-6;
  auto v = barrier(xhat * (1 - eps), xhat);
  EXPECT_LT(std::abs(v.b) / (xhat * xhat), 1e-15);
  EXPECT_LT(std::abs(v.db) / xhat, 1e-10);
  EXPECT_LT(std::abs(v.ddb), 1e-5);
}

// 用中心差分检查 b' 和 b''：(f(x+e) - f(x-e)) / (2e) 应该等于 f'(x)，相对误差小于 1e-6。
TEST_P(BarrierTest, DerivativesMatchFiniteDifferences) {
  const double xhat = GetParam();
  for (double r : {0.01, 0.1, 0.3, 0.5, 0.9, 0.99}) {
    const double x = r * xhat, e = 1e-6 * x;
    auto v = barrier(x, xhat), vp = barrier(x + e, xhat), vm = barrier(x - e, xhat);
    const double db_fd = (vp.b - vm.b) / (2 * e);
    const double ddb_fd = (vp.db - vm.db) / (2 * e);
    EXPECT_NEAR(v.db, db_fd, 1e-6 * std::abs(v.db)) << "x / xhat = " << r;
    EXPECT_NEAR(v.ddb, ddb_fd, 1e-6 * std::abs(v.ddb)) << "x / xhat = " << r;
  }
}

// 间隙越来越小时，能量和斥力都单调增长、没有上界：有限的载荷压不到 x = 0。
TEST_P(BarrierTest, GrowsWithoutBoundAtZero) {
  const double xhat = GetParam();
  double b_prev = 0, f_prev = 0;
  for (double r : {1e-2, 1e-4, 1e-8, 1e-12}) {
    auto v = barrier(r * xhat, xhat);
    EXPECT_GT(v.b, b_prev) << "x / xhat = " << r;
    EXPECT_GT(-v.db, f_prev) << "x / xhat = " << r;
    b_prev = v.b;
    f_prev = -v.db;
  }
  // x 接近 0 时，-b'(x) 约等于 xhat^2 / x。
  auto v = barrier(1e-12 * xhat, xhat);
  EXPECT_GT(-v.db, 0.9e12 * xhat);
}

// 书 §3.8 的静止间隙（1 kg，kappa = 1e4 N/m，dhat = 1 mm）是 0.5359 mm：
// 在那里障碍力 -kappa b'(d) 正好托住重力 m g = 9.81 N。
// 书上的间隙只保留 4 位，带来约 3e-3 N 的不确定度，所以容差取 0.01 N。
TEST(Barrier, BookRestGapBalancesWeight) {
  auto v = barrier(0.5359e-3, 1e-3);
  EXPECT_NEAR(-1e4 * v.db, 9.81, 0.01);
}
