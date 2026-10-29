// Stage 1: the one-dimensional IPC step, checked against its definition and the book's numbers.
#include "ipc1d/ipc1d.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rcipc/barrier.h"

using namespace rcipc::ipc1d;

namespace {

struct DropRun {
  double xmin = 1e300, x = 0, v = 0;
  int nmax = 0;
  double nmean = 0;
};

// The book's drop: T seconds from gap x0 and velocity v0.
DropRun simulate(const Params& P, double x0, double v0, double T) {
  DropRun r;
  double x = x0, v = v0, t = 0;
  long n = 0, steps = 0;
  while (t < T - 1e-12) {
    StepResult s = step(P, x, v);
    x = s.x;
    v = s.v;
    t += P.h;
    r.xmin = std::min(r.xmin, x);
    r.nmax = std::max(r.nmax, s.newton);
    n += s.newton;
    ++steps;
  }
  r.x = x;
  r.v = v;
  r.nmean = double(n) / steps;
  return r;
}

}  // namespace

// ---------- notes section 1: the incremental potential ----------

TEST(Energy, PredictedIsImplicitEulerWithoutTheGround) {
  Params P;
  EXPECT_DOUBLE_EQ(predicted(P, 0.45, 0.0), 0.45 - P.h * P.h * P.g);
  EXPECT_DOUBLE_EQ(predicted(P, 0.45, -2.0), 0.45 - 2.0 * P.h - P.h * P.h * P.g);
}

TEST(Energy, InertiaOnlyAboveActivation) {
  Params P;
  const double xt = 0.1;
  for (double x : {P.dhat, 0.05, 0.2}) {
    const double expected = 0.5 * P.m / (P.h * P.h) * (x - xt) * (x - xt);
    EXPECT_NEAR(energy(P, x, xt), expected, 1e-14 * expected) << "x = " << x;
  }
}

TEST(Energy, IncludesTheBarrierInside) {
  Params P;
  const double x = 0.3e-3, xt = -1e-3;
  const double inertia = 0.5 * P.m / (P.h * P.h) * (x - xt) * (x - xt);
  const double expected = inertia + P.kappa * rcipc::barrier(x, P.dhat).b;
  EXPECT_NEAR(energy(P, x, xt), expected, 1e-14 * expected);
}

TEST(Energy, DerivativesMatchFiniteDifferences) {
  Params P;
  const double xt = -3.04e-3;  // the book's impact step
  for (double x : {1e-5, 1e-4, 0.5e-3, 0.99e-3, 2e-3, 0.1}) {
    const double e = 1e-6 * x;
    const double g_fd = (energy(P, x + e, xt) - energy(P, x - e, xt)) / (2 * e);
    const double H_fd = (gradient(P, x + e, xt) - gradient(P, x - e, xt)) / (2 * e);
    EXPECT_NEAR(gradient(P, x, xt), g_fd, 1e-6 * std::abs(g_fd) + 1e-9) << "x = " << x;
    EXPECT_NEAR(hessian(P, x, xt), H_fd, 1e-6 * std::abs(H_fd)) << "x = " << x;
  }
}

// In one dimension the Hessian m / h^2 + kappa b'' is always positive, so every Newton direction
// p = -g / H points downhill. (From stage 4 on this will need work: notes/01_ipc1d.md, question 4.)
TEST(Energy, HessianIsPositive) {
  Params P;
  for (double x : {1e-9, 1e-5, 0.5e-3, 2e-3}) EXPECT_GT(hessian(P, x, 0.0), 0.0) << "x = " << x;
}

// ---------- notes section 3.2: CCD ----------

TEST(Ccd, MovingAwayTakesTheFullStep) {
  EXPECT_EQ(ccd_max_step(1e-3, +5e-3, 0.9), 1.0);
  EXPECT_EQ(ccd_max_step(1e-3, 0.0, 0.9), 1.0);
}

TEST(Ccd, ShortApproachTakesTheFullStep) {
  // The full step closes half of the gap, less than c = 90%.
  EXPECT_EQ(ccd_max_step(1e-3, -0.5e-3, 0.9), 1.0);
}

TEST(Ccd, LongApproachStopsAtFractionC) {
  // From 3 mm towards -6 mm: stop where 90% of the gap is closed, at 0.3 mm.
  const double x = 3e-3, p = -6e-3;
  const double a = ccd_max_step(x, p, 0.9);
  EXPECT_NEAR(a, 0.9 * x / -p, 1e-15);
  EXPECT_NEAR(x + a * p, 0.1 * x, 1e-15);
}

TEST(Ccd, NeverClosesMoreThanFractionC) {
  for (double x : {1e-9, 1e-5, 1e-3, 1.0}) {
    for (double p : {-1e-12, -1e-6, -1e-3, -1.0, -1e3}) {
      for (double c : {0.5, 0.9, 0.99}) {
        const double a = ccd_max_step(x, p, c);
        EXPECT_GT(a, 0.0);
        EXPECT_LE(a, 1.0);
        EXPECT_GE(x + a * p, (1 - c) * x * (1 - 1e-12)) << "x " << x << " p " << p << " c " << c;
      }
    }
  }
}

// ---------- notes section 3.4: the step ----------

// Far from the ground the energy is a quadratic with its minimum at x~, so the first Newton step
// lands on it exactly and the second iteration only confirms convergence.
TEST(Step, FreeFallIsOneNewtonStep) {
  Params P;
  StepResult r = step(P, 0.45, -1.0);
  EXPECT_NEAR(r.x, predicted(P, 0.45, -1.0), 1e-15);
  EXPECT_NEAR(r.v, (r.x - 0.45) / P.h, 1e-12);
  EXPECT_EQ(r.newton, 2);
}

// The book's impact step (section 3.5.3): from 3 mm at -3 m/s with x~ = -3.04 mm.
//   it  gap      p         alpha_max  alpha   new gap (mm)
//   1   3.0000  -6.0392    0.4471     0.4471  0.3000
//   2   0.3000  -1.9002    0.1421     0.1421  0.0300
//   3   0.0300  -0.0320    0.8449     0.4224  0.0165
//   4   0.0165  -0.0025    1          1       0.0140
//   5   0.0140  +0.0003    1          1       0.0143
//   6   converged at 0.0143
// With tol = 1e-6 m/s there is one more iteration: at 6, p = +7.5 nm, so |p| / h = 3.8e-6 m/s is
// still above tol (the book's table stops there; notes/01_ipc1d.md, question 5). It converges at 7.
TEST(Step, ImpactMatchesTheBook) {
  Params P;
  const double x0 = 3e-3, v0 = -3.0, xt = predicted(P, x0, v0);
  std::vector<Iterate> tr;
  StepResult r = step(P, x0, v0, &tr);

  ASSERT_EQ(tr.size(), 6u);
  EXPECT_EQ(r.newton, 7);
  EXPECT_NEAR(tr[5].p, 7.5e-9, 0.1e-9);
  const double gap_mm[] = {3.0, 0.3, 0.03, 0.0165, 0.0140};
  const double p_mm[] = {-6.0392, -1.9002, -0.0320, -0.0025, +0.0003};
  const double amax[] = {0.4471, 0.1421, 0.8449, 1.0, 1.0};
  const double alpha[] = {0.4471, 0.1421, 0.4224, 1.0, 1.0};
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(tr[i].x * 1e3, gap_mm[i], 1e-4) << "iteration " << i + 1;
    EXPECT_NEAR(tr[i].p * 1e3, p_mm[i], 1e-4) << "iteration " << i + 1;
    EXPECT_NEAR(tr[i].alpha_max, amax[i], 1e-4) << "iteration " << i + 1;
    EXPECT_NEAR(tr[i].alpha, alpha[i], 1e-4) << "iteration " << i + 1;
  }
  EXPECT_NEAR(r.x * 1e3, 0.0143, 1e-4);

  // The trace is consistent: each iteration starts where the previous one ended, every iterate is
  // inside the CCD bound, and the energy never goes up.
  for (size_t i = 0; i < tr.size(); ++i) {
    const double next = i + 1 < tr.size() ? tr[i + 1].x : r.x;
    EXPECT_NEAR(next, tr[i].x + tr[i].alpha * tr[i].p, 1e-18);
    EXPECT_LE(tr[i].alpha, tr[i].alpha_max);
    EXPECT_GT(next, 0.0);
    EXPECT_LE(energy(P, next, xt), energy(P, tr[i].x, xt));
  }
  // The end of the step is a minimum: the gradient is (almost) zero.
  EXPECT_LT(std::abs(gradient(P, r.x, xt)) / hessian(P, r.x, xt), P.tol * P.h);
}

// ---------- notes section 3.5: the drop ----------

// The book's run: 1 kg from 0.45 m, h 2 ms, kappa 1e4, 1.5 s. Book: min gap 0.0078 mm, final gap
// 0.5359 mm, final v 0, Newton per step max 9, mean 1.44.
TEST(Drop, NeverPenetratesAndComesToRest) {
  Params P;
  DropRun r = simulate(P, 0.45, 0.0, 1.5);
  EXPECT_GT(r.xmin, 0.0);
  EXPECT_NEAR(r.xmin * 1e3, 0.0078, 0.6e-4);  // the book rounds to 1e-4 mm
  EXPECT_NEAR(r.x * 1e3, 0.5359, 0.6e-4);
  EXPECT_EQ(r.v, 0.0);
  // At rest the barrier carries the weight: -kappa b'(x) = m g.
  EXPECT_NEAR(-P.kappa * rcipc::barrier(r.x, P.dhat).db, P.m * P.g, 1e-3);
  EXPECT_LE(r.nmax, 10);
  EXPECT_NEAR(r.nmean, 1.44, 0.05);
}

// The book's sweep (section 3.8). kappa sets where the ball rests, never whether it goes through:
// even at 40 m/s with h = 10 ms (40 cm per step) the gap stays positive, and the rest gap depends
// on kappa only.
struct SweepRow {
  const char* name;
  double kappa, h, v0, xmin_mm, xrest_mm;
};
void PrintTo(const SweepRow& w, std::ostream* os) { *os << w.name; }

class DropSweep : public ::testing::TestWithParam<SweepRow> {};
INSTANTIATE_TEST_SUITE_P(Book, DropSweep,
                         ::testing::Values(SweepRow{"kappa1e3_h2ms", 1e3, 0.002, 0, 0.0007, 0.1244},
                                           SweepRow{"kappa1e4_h2ms", 1e4, 0.002, 0, 0.0078, 0.5359},
                                           SweepRow{"kappa1e5_h2ms", 1e5, 0.002, 0, 0.0997, 0.8301},
                                           SweepRow{"kappa1e4_h10ms", 1e4, 0.010, 0, 0.0496, 0.5359},
                                           SweepRow{"kappa1e4_h10ms_v40", 1e4, 0.010, -40, 0.0029, 0.5359}),
                         [](const auto& info) { return std::string(info.param.name); });

TEST_P(DropSweep, MatchesTheBook) {
  const SweepRow& w = GetParam();
  Params P;
  P.kappa = w.kappa;
  P.h = w.h;
  DropRun r = simulate(P, 0.45, w.v0, 1.5);
  EXPECT_GT(r.xmin, 0.0);
  EXPECT_NEAR(r.xmin * 1e3, w.xmin_mm, 0.6e-4);  // the book rounds to 1e-4 mm
  EXPECT_NEAR(r.x * 1e3, w.xrest_mm, 0.6e-4);
}
