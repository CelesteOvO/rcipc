#include "rcipc/distance.h"

#include <algorithm>

namespace rcipc {

using Eigen::Vector3d;

Vector3d point_triangle_closest(const Vector3d& p, const Vector3d& a, const Vector3d& b, const Vector3d& c) {
  // Ericson 5.1.5。思路：p 相对三角形落在 7 个 Voronoi 区域之一（3 个角、3 条边、1 个面），
  // 用几个点积判断在哪个区域，然后直接写出最近点的重心坐标。
  const Vector3d ab = b - a, ac = c - a, ap = p - a;
  const double d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) return Vector3d(1, 0, 0);  // 角 a 的区域

  const Vector3d bp = p - b;
  const double d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0 && d4 <= d3) return Vector3d(0, 1, 0);  // 角 b 的区域

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {  // 边 ab 的区域
    const double v = d1 / (d1 - d3);
    return Vector3d(1 - v, v, 0);
  }

  const Vector3d cp = p - c;
  const double d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0 && d5 <= d6) return Vector3d(0, 0, 1);  // 角 c 的区域

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {  // 边 ac 的区域
    const double w = d2 / (d2 - d6);
    return Vector3d(1 - w, 0, w);
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {  // 边 bc 的区域
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return Vector3d(0, 1 - w, w);
  }

  // 面内：重心坐标就是三个子三角形的面积比。
  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom, w = vc * denom;
  return Vector3d(1 - v - w, v, w);
}

void segment_segment_closest(const Vector3d& p0, const Vector3d& p1, const Vector3d& q0, const Vector3d& q1,
                             double& s, double& t) {
  // Ericson 5.1.9。记 d1 = p1 - p0，d2 = q1 - q0，r = p0 - q0。
  const Vector3d d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
  const double a = d1.squaredNorm(), e = d2.squaredNorm(), f = d2.dot(r);
  constexpr double kEps = 1e-30;  // 平方长度小于它就当作一个点

  if (a <= kEps && e <= kEps) {  // 两条都退化成点
    s = t = 0;
    return;
  }
  if (a <= kEps) {  // 第一条是点：t 是 q 上离 p0 最近的参数
    s = 0;
    t = std::clamp(f / e, 0.0, 1.0);
    return;
  }
  const double c = d1.dot(r);
  if (e <= kEps) {  // 第二条是点
    t = 0;
    s = std::clamp(-c / a, 0.0, 1.0);
    return;
  }

  // 一般情形。两条直线的最近点满足 s a - t b = -c，s b - t e = -f，其中 b = d1 . d2。
  const double b = d1.dot(d2);
  const double denom = a * e - b * b;  // >= 0，平行时为 0
  s = denom > kEps ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;  // 平行：s 随便取，这里取 0
  // 给定 s，q 上的最近参数 t = (b s + f) / e；夹到 [0, 1] 后，若夹过了就重新解 s。
  const double tnom = b * s + f;
  if (tnom < 0) {
    t = 0;
    s = std::clamp(-c / a, 0.0, 1.0);
  } else if (tnom > e) {
    t = 1;
    s = std::clamp((b - c) / a, 0.0, 1.0);
  } else {
    t = tnom / e;
  }
}

}  // namespace rcipc
