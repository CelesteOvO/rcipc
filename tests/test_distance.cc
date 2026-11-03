// 阶段 7：最近点（笔记 notes/07_contact.md 第 1 节）。
//
//   PointTriangle.*    点–三角形：和在三角形上密集采样得到的最小距离一致；7 个 Voronoi 区域都碰到
//   SegmentSegment.*   线段–线段：和在两条线段上密集采样一致；退化成点、平行的情形
#include "rcipc/distance.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

using namespace rcipc;
using Eigen::Vector3d;

namespace {

Vector3d random_point(std::mt19937& rng, double scale) {
  std::uniform_real_distribution<double> u(-scale, scale);
  return Vector3d(u(rng), u(rng), u(rng));
}

// 三角形上的采样点里离 p 最近的距离（n x n 个重心坐标）。
double sampled_point_triangle(const Vector3d& p, const Vector3d& a, const Vector3d& b, const Vector3d& c, int n) {
  double best = 1e300;
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; i + j <= n; ++j) {
      const double u = double(i) / n, v = double(j) / n;
      best = std::min(best, (p - ((1 - u - v) * a + u * b + v * c)).norm());
    }
  }
  return best;
}

}  // namespace

// 随机点和随机三角形：闭式最近点的距离 <= 采样的最小距离（采样只能更大），并且相差不超过采样间距。
TEST(PointTriangle, MatchesDenseSampling) {
  std::mt19937 rng(80);
  for (int i = 0; i < 200; ++i) {
    const Vector3d a = random_point(rng, 1), b = random_point(rng, 1), c = random_point(rng, 1);
    const Vector3d p = random_point(rng, 1.5);
    const Vector3d lam = point_triangle_closest(p, a, b, c);
    EXPECT_GE(lam.minCoeff(), 0.0);
    EXPECT_NEAR(lam.sum(), 1.0, 1e-12);
    const double d = (p - (lam[0] * a + lam[1] * b + lam[2] * c)).norm();
    const double ds = sampled_point_triangle(p, a, b, c, 200);
    const double spacing = std::max({(a - b).norm(), (b - c).norm(), (c - a).norm()}) / 200;
    EXPECT_LE(d, ds + 1e-12);
    EXPECT_GE(d, ds - spacing);
  }
}

// 7 个区域各一个例子：三角形 (0,0,0), (1,0,0), (0,1,0)。
TEST(PointTriangle, HitsEveryVoronoiRegion) {
  const Vector3d a(0, 0, 0), b(1, 0, 0), c(0, 1, 0);
  const auto closest = [&](const Vector3d& p) {
    const Vector3d lam = point_triangle_closest(p, a, b, c);
    return Vector3d(lam[0] * a + lam[1] * b + lam[2] * c);
  };
  EXPECT_LT((closest(Vector3d(-1, -1, 3)) - a).norm(), 1e-15);          // 角 a
  EXPECT_LT((closest(Vector3d(2, -1, 3)) - b).norm(), 1e-15);           // 角 b
  EXPECT_LT((closest(Vector3d(-1, 2, 3)) - c).norm(), 1e-15);           // 角 c
  EXPECT_LT((closest(Vector3d(0.5, -1, 3)) - Vector3d(0.5, 0, 0)).norm(), 1e-15);   // 边 ab
  EXPECT_LT((closest(Vector3d(-1, 0.5, 3)) - Vector3d(0, 0.5, 0)).norm(), 1e-15);   // 边 ac
  EXPECT_LT((closest(Vector3d(1, 1, 3)) - Vector3d(0.5, 0.5, 0)).norm(), 1e-15);    // 边 bc
  EXPECT_LT((closest(Vector3d(0.2, 0.3, 3)) - Vector3d(0.2, 0.3, 0)).norm(), 1e-15);  // 面内
}

// 随机两条线段：闭式最近点对的距离和密集采样一致。
TEST(SegmentSegment, MatchesDenseSampling) {
  std::mt19937 rng(81);
  for (int i = 0; i < 200; ++i) {
    const Vector3d p0 = random_point(rng, 1), p1 = random_point(rng, 1), q0 = random_point(rng, 1),
                   q1 = random_point(rng, 1);
    double s, t;
    segment_segment_closest(p0, p1, q0, q1, s, t);
    EXPECT_GE(s, 0.0);
    EXPECT_LE(s, 1.0);
    EXPECT_GE(t, 0.0);
    EXPECT_LE(t, 1.0);
    const double d = (((1 - s) * p0 + s * p1) - ((1 - t) * q0 + t * q1)).norm();
    double ds = 1e300;
    const int n = 200;
    for (int a = 0; a <= n; ++a) {
      for (int b = 0; b <= n; ++b) {
        const double u = double(a) / n, v = double(b) / n;
        ds = std::min(ds, (((1 - u) * p0 + u * p1) - ((1 - v) * q0 + v * q1)).norm());
      }
    }
    const double spacing = std::max((p1 - p0).norm(), (q1 - q0).norm()) / n;
    EXPECT_LE(d, ds + 1e-12);
    EXPECT_GE(d, ds - spacing);
  }
}

// 退化：一条或两条线段是一个点（球心）；平行的线段。
TEST(SegmentSegment, DegenerateAndParallel) {
  double s, t;
  // 点对线段：最近参数是投影，夹到 [0, 1]。
  segment_segment_closest(Vector3d(0.3, 1, 0), Vector3d(0.3, 1, 0), Vector3d(0, 0, 0), Vector3d(1, 0, 0), s, t);
  EXPECT_EQ(s, 0.0);
  EXPECT_NEAR(t, 0.3, 1e-15);
  segment_segment_closest(Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(2, 1, 0), Vector3d(2, 1, 0), s, t);
  EXPECT_NEAR(s, 1.0, 1e-15);
  EXPECT_EQ(t, 0.0);
  // 两个点。
  segment_segment_closest(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector3d(1, 1, 1), Vector3d(1, 1, 1), s, t);
  EXPECT_EQ(s, 0.0);
  EXPECT_EQ(t, 0.0);
  // 平行：距离仍然正确（1）。
  segment_segment_closest(Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(0.5, 1, 0), Vector3d(1.5, 1, 0), s, t);
  EXPECT_NEAR((((1 - s) * Vector3d(0, 0, 0) + s * Vector3d(1, 0, 0)) -
               ((1 - t) * Vector3d(0.5, 1, 0) + t * Vector3d(1.5, 1, 0)))
                  .norm(),
              1.0, 1e-15);
}
