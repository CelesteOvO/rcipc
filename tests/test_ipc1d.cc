// 阶段 1：一维 IPC 的时间步。先按定义检查每个函数，再拿完整的仿真和书上的数字比。
//
// 测试分四组，和笔记 notes/01_ipc1d.md 的小节对应：
//   Energy.*         第 1 节：预测位置、增量势和它的导数
//   Ccd.*            第 3.2 节：CCD 的步长上限
//   Step.*           第 3.4 节：一个时间步（自由落体、书上的撞击那一步）
//   Drop.*、Book/*   第 3.5 节：完整的落地仿真
#include "ipc1d/ipc1d.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rcipc/barrier.h"

using namespace rcipc::ipc1d;

namespace {

// 一次落地仿真的统计量。
struct DropRun {
  double xmin = 1e300;  // 整个过程中的最小间隙
  double x = 0, v = 0;  // 结束时的间隙和速度
  int nmax = 0;         // 单步最多的 Newton 次数
  double nmean = 0;     // 每步平均 Newton 次数
};

// 从间隙 x0、速度 v0 出发，连续调用 step() 模拟 T 秒。
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

// ---------- 笔记第 1 节：增量势 ----------

// x~ = x0 + h v0 - h^2 g，试两个初速度。
TEST(Energy, PredictedIsImplicitEulerWithoutTheGround) {
  Params P;
  EXPECT_DOUBLE_EQ(predicted(P, 0.45, 0.0), 0.45 - P.h * P.h * P.g);
  EXPECT_DOUBLE_EQ(predicted(P, 0.45, -2.0), 0.45 - 2.0 * P.h - P.h * P.h * P.g);
}

// 间隙 >= dhat 时障碍为 0，E 只剩惯性项 m / (2 h^2) (x - xt)^2。
TEST(Energy, InertiaOnlyAboveActivation) {
  Params P;
  const double xt = 0.1;
  for (double x : {P.dhat, 0.05, 0.2}) {
    const double expected = 0.5 * P.m / (P.h * P.h) * (x - xt) * (x - xt);
    EXPECT_NEAR(energy(P, x, xt), expected, 1e-14 * expected) << "x = " << x;
  }
}

// 间隙 < dhat 时，E = 惯性项 + kappa * b(x)。
TEST(Energy, IncludesTheBarrierInside) {
  Params P;
  const double x = 0.3e-3, xt = -1e-3;
  const double inertia = 0.5 * P.m / (P.h * P.h) * (x - xt) * (x - xt);
  const double expected = inertia + P.kappa * rcipc::barrier(x, P.dhat).b;
  EXPECT_NEAR(energy(P, x, xt), expected, 1e-14 * expected);
}

// 用中心差分检查 gradient 和 hessian：(f(x+e) - f(x-e)) / (2e) 应该等于 f'(x)。
// 取的点覆盖了障碍区内（1e-5 ... 0.99e-3）和障碍区外（2e-3、0.1）。
TEST(Energy, DerivativesMatchFiniteDifferences) {
  Params P;
  const double xt = -3.04e-3;  // 书上撞击那一步的预测位置
  for (double x : {1e-5, 1e-4, 0.5e-3, 0.99e-3, 2e-3, 0.1}) {
    const double e = 1e-6 * x;
    const double g_fd = (energy(P, x + e, xt) - energy(P, x - e, xt)) / (2 * e);
    const double H_fd = (gradient(P, x + e, xt) - gradient(P, x - e, xt)) / (2 * e);
    EXPECT_NEAR(gradient(P, x, xt), g_fd, 1e-6 * std::abs(g_fd) + 1e-9) << "x = " << x;
    EXPECT_NEAR(hessian(P, x, xt), H_fd, 1e-6 * std::abs(H_fd)) << "x = " << x;
  }
}

// 一维里 Hessian m / h^2 + kappa b'' 总是正的，所以 Newton 方向 p = -g / H 一定是下降方向。
// 从阶段 4 起这一点不再自动成立（笔记思考题 4）。
TEST(Energy, HessianIsPositive) {
  Params P;
  for (double x : {1e-9, 1e-5, 0.5e-3, 2e-3}) EXPECT_GT(hessian(P, x, 0.0), 0.0) << "x = " << x;
}

// ---------- 笔记第 3.2 节：CCD ----------

// 远离地面（p > 0）或不动（p = 0）：走整步。
TEST(Ccd, MovingAwayTakesTheFullStep) {
  EXPECT_EQ(ccd_max_step(1e-3, +5e-3, 0.9), 1.0);
  EXPECT_EQ(ccd_max_step(1e-3, 0.0, 0.9), 1.0);
}

// 靠近地面，但整步只闭合一半间隙，没有超过 c = 90%：还是走整步。
TEST(Ccd, ShortApproachTakesTheFullStep) {
  EXPECT_EQ(ccd_max_step(1e-3, -0.5e-3, 0.9), 1.0);
}

// 从 3 mm 往 -6 mm 走：整步会穿过地面，所以停在闭合了 90% 的地方，也就是 0.3 mm。
TEST(Ccd, LongApproachStopsAtFractionC) {
  const double x = 3e-3, p = -6e-3;
  const double a = ccd_max_step(x, p, 0.9);
  EXPECT_NEAR(a, 0.9 * x / -p, 1e-15);
  EXPECT_NEAR(x + a * p, 0.1 * x, 1e-15);
}

// 各种间隙、方向、c 的组合：步长总在 (0, 1] 里，走完以后间隙总是 >= (1 - c) x。
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

// ---------- 笔记第 3.4 节：一个时间步 ----------

// 离地面很远时障碍为 0，E 是一个最小点在 x~ 的二次函数。
// Newton 法对二次函数一步就精确落到最小点，第二次迭代只是确认收敛，所以 newton = 2。
TEST(Step, FreeFallIsOneNewtonStep) {
  Params P;
  StepResult r = step(P, 0.45, -1.0);
  EXPECT_NEAR(r.x, predicted(P, 0.45, -1.0), 1e-15);
  EXPECT_NEAR(r.v, (r.x - 0.45) / P.h, 1e-12);
  EXPECT_EQ(r.newton, 2);
}

// 书 §3.5.3 的撞击那一步：从 3 mm 以 -3 m/s 下落，x~ = -3.04 mm 在地面以下。书上的表：
//   次数  间隙     p         alpha_max  alpha   新间隙（mm）
//   1     3.0000  -6.0392    0.4471     0.4471  0.3000    障碍还没激活，全靠 CCD 截断
//   2     0.3000  -1.9002    0.1421     0.1421  0.0300    障碍激活了，但 Newton 仍低估它，CCD 又截一次
//   3     0.0300  -0.0320    0.8449     0.4224  0.0165    CCD 允许 84%，但那里能量更高，线搜索减半一次
//   4     0.0165  -0.0025    1          1       0.0140    进入二次收敛区
//   5     0.0140  +0.0003    1          1       0.0143
//   6     在 0.0143 收敛
// 用 tol = 1e-6 m/s 时还要多一次：第 6 次 p = +7.5 nm，|p| / h = 3.8e-6 m/s，仍大于 tol
// （书上的表到这里就停了，见笔记思考题 5），第 7 次才收敛。
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

  // 记录本身要自洽：每次迭代从上一次结束的地方开始；alpha 不超过 CCD 的上限；间隙始终为正；能量从不上升。
  for (size_t i = 0; i < tr.size(); ++i) {
    const double next = i + 1 < tr.size() ? tr[i + 1].x : r.x;
    EXPECT_NEAR(next, tr[i].x + tr[i].alpha * tr[i].p, 1e-18);
    EXPECT_LE(tr[i].alpha, tr[i].alpha_max);
    EXPECT_GT(next, 0.0);
    EXPECT_LE(energy(P, next, xt), energy(P, tr[i].x, xt));
  }
  // 时间步结束时在最小点上：剩下的 Newton 修正 |g / H| 小于 tol * h。
  EXPECT_LT(std::abs(gradient(P, r.x, xt)) / hessian(P, r.x, xt), P.tol * P.h);
}

// ---------- 笔记第 3.5 节：落地仿真 ----------

// 书 §3.8 的仿真：1 kg 的球从 0.45 m 静止落下，h = 2 ms，kappa = 1e4，模拟 1.5 s。
// 书上的结果：最小间隙 0.0078 mm，静止间隙 0.5359 mm，最终速度 0，每步 Newton 最多 9 次、平均 1.44 次。
TEST(Drop, NeverPenetratesAndComesToRest) {
  Params P;
  DropRun r = simulate(P, 0.45, 0.0, 1.5);
  EXPECT_GT(r.xmin, 0.0);
  EXPECT_NEAR(r.xmin * 1e3, 0.0078, 0.6e-4);  // 书上的数字保留到 1e-4 mm
  EXPECT_NEAR(r.x * 1e3, 0.5359, 0.6e-4);
  EXPECT_EQ(r.v, 0.0);
  // 静止时障碍力托住重力：-kappa b'(x) = m g。
  EXPECT_NEAR(-P.kappa * rcipc::barrier(r.x, P.dhat).db, P.m * P.g, 1e-3);
  EXPECT_LE(r.nmax, 10);
  EXPECT_NEAR(r.nmean, 1.44, 0.05);
}

// 书 §3.8 末尾的参数扫描。kappa 决定球停在多远，但不决定会不会穿透：
// 即使以 40 m/s 撞地、h = 10 ms（每步走 40 cm），间隙也始终为正；静止间隙只和 kappa 有关。
struct SweepRow {
  const char* name;
  double kappa, h, v0, xmin_mm, xrest_mm;
};
void PrintTo(const SweepRow& w, std::ostream* os) { *os << w.name; }  // 让测试输出显示名字而不是字节

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
  EXPECT_NEAR(r.xmin * 1e3, w.xmin_mm, 0.6e-4);  // 书上的数字保留到 1e-4 mm
  EXPECT_NEAR(r.x * 1e3, w.xrest_mm, 0.6e-4);
}
