// Stage 1: the barrier b(x) = -(x - xhat)^2 ln(x / xhat) and its derivatives.
#include "rcipc/barrier.h"

#include <cmath>

#include <gtest/gtest.h>

using rcipc::barrier;

// Every property is checked at two scales: a distance (dhat = 1 mm, stage 1) and a squared distance
// (dhat^2 = 1e-6 m^2, from stage 4 on).
class BarrierTest : public ::testing::TestWithParam<double> {};
INSTANTIATE_TEST_SUITE_P(Scales, BarrierTest, ::testing::Values(1e-3, 1e-6),
                         [](const auto& info) { return info.param == 1e-3 ? "distance" : "squared"; });

TEST_P(BarrierTest, ZeroOutsideActivation) {
  const double xhat = GetParam();
  for (double r : {1.0, 1.5, 10.0}) {
    auto v = barrier(r * xhat, xhat);
    EXPECT_EQ(v.b, 0.0) << "x / xhat = " << r;
    EXPECT_EQ(v.db, 0.0) << "x / xhat = " << r;
    EXPECT_EQ(v.ddb, 0.0) << "x / xhat = " << r;
  }
}

TEST_P(BarrierTest, PositiveRepulsiveConvexInside) {
  const double xhat = GetParam();
  for (double r : {1e-6, 0.01, 0.1, 0.5, 0.9, 0.999}) {
    auto v = barrier(r * xhat, xhat);
    EXPECT_GT(v.b, 0.0) << "x / xhat = " << r;
    EXPECT_LT(v.db, 0.0) << "x / xhat = " << r;   // the force -b' pushes the gap open
    EXPECT_GT(v.ddb, 0.0) << "x / xhat = " << r;  // convex in x
  }
}

// b, b' and b'' all go to 0 at xhat, so the barrier joins the zero function outside with C^2
// continuity. With x = xhat (1 - eps): b ~ eps^3 xhat^2, b' ~ -3 eps^2 xhat and b'' ~ 6 eps.
TEST_P(BarrierTest, SmoothAtActivation) {
  const double xhat = GetParam();
  const double eps = 1e-6;
  auto v = barrier(xhat * (1 - eps), xhat);
  EXPECT_LT(std::abs(v.b) / (xhat * xhat), 1e-15);
  EXPECT_LT(std::abs(v.db) / xhat, 1e-10);
  EXPECT_LT(std::abs(v.ddb), 1e-5);
}

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

// As the gap closes the energy and the force grow without bound: no finite load can push x to 0.
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
  // Near 0, -b'(x) is about xhat^2 / x.
  auto v = barrier(1e-12 * xhat, xhat);
  EXPECT_GT(-v.db, 0.9e12 * xhat);
}

// The book's rest gap (section 3.8: 1 kg, kappa 1e4 N/m, dhat 1 mm) is 0.5359 mm, where the barrier
// force -kappa b'(d) balances the weight m g = 9.81 N. The gap is given to 4 digits; that leaves an
// uncertainty of about 3e-3 N.
TEST(Barrier, BookRestGapBalancesWeight) {
  auto v = barrier(0.5359e-3, 1e-3);
  EXPECT_NEAR(-1e4 * v.db, 9.81, 0.01);
}
