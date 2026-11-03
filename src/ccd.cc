#include "rcipc/ccd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "rcipc/chain.h"
#include "rcipc/contact.h"
#include "rcipc/distance.h"

// CCD。直弦版本的推导见 notes/04_ground.md 第 5 节，曲线版本见 notes/05_curved_ccd.md，
// 刚体之间的 ACCD 见 notes/07_contact.md 第 4 节；测试见 tests/test_ccd.cc。
namespace rcipc {

using Eigen::Vector3d;
using Eigen::VectorXd;

double chord_ccd_ground(const Model& model, const Frames& a, const Frames& b, double c) {
  // 对每个 (顶点, 平面) 对单独推进，取最小（书 §9.4：逐对，不用全局最小距离）。
  double alpha = 1.0;
  for (int v = 0; v < model.nvertex(); ++v) {
    if (model.vertices[v].body == 0) continue;  // 世界上的静态几何不和平面比
    for (int p = 0; p < static_cast<int>(model.planes.size()); ++p) {
      const double ya = ground_distance(model, a, v, p);
      const double yb = ground_distance(model, b, v, p);
      // 情况 1：远离平面（或不动），整步安全。
      if (yb >= ya) continue;
      // 情况 2：靠近平面。沿弦距离线性减少，最多闭合 c * ya：s <= c ya / (ya - yb)。
      //   这个值可能大于 1（整步也闭合不到 c * ya），那就不限制。和阶段 1 的 ccd_max_step 完全一样。
      alpha = std::min(alpha, c * ya / (ya - yb));
    }
  }
  return alpha;
}

VectorXd acceleration_bounds(const Model& model, const VectorXd& q, const VectorXd& p) {
  // 静态加速度界（阶段 5 笔记第 2 节，书 §9.3.1）。
  //
  // 推导的骨架：刚体上一点 x 的速度是链上各关节的贡献之和
  //   x' = sum_j w_j a_j x (x - o_j) + sum_s u_s a_s          （转动组 j：轴 a_j、锚点 o_j；滑动 s：轴 a_s）
  // 再求一次导数，每一项分两部分：
  //   - 轴在转：a_j、a_s 被更靠近根的关节带着转，|a'| <= W_{<j}；转动项的力臂 |x - o_j| <= R_j；
  //   - 点相对锚点在动：|(x - o_j)'| 只来自 j 和它下游的关节，<= sum_{k>=j} (w_k R_k + u_k)。
  // 合起来 |x''| <= sum_j w_j (W_{<j} R_j + sum_{k>=j} (w_k R_k + u_k)) + sum_s W_{<s} u_s。
  // R_j 用静态力臂（chain.h）：固定长度的线段首尾相接，三角不等式保证在任何姿态下 |x - o_j| 都不超过它们之和。
  VectorXd A = VectorXd::Zero(model.nbody());
  std::vector<double> w, u, R;  // 链上的关节，从根到刚体的顺序：转动组的 w、滑动的 u、静态力臂 R
  for (int b = 1; b < model.nbody(); ++b) {
    // 第 1 步：链上的关节和静态力臂。每个关节的 w 或 u 是 p 在它那组自由度上的长度。
    const std::vector<ChainJoint> chain = chain_joints(model, q, b);
    const int n = static_cast<int>(chain.size());
    w.assign(n, 0.0);
    u.assign(n, 0.0);
    R.assign(n, 0.0);
    for (int j = 0; j < n; ++j) {
      const ChainJoint& cj = chain[j];
      const double mag = p.segment(cj.dadr, cj.nv).norm();
      (cj.rotation ? w[j] : u[j]) = mag;
      R[j] = cj.reach;
    }
    // 滑动关节在这一步内最多再伸 u_s：它上游各关节的力臂都要加上这一段。
    for (int s = 0; s < n; ++s) {
      if (chain[s].rotation) continue;
      for (int j = 0; j < s; ++j) R[j] += u[s];
    }

    // 第 2 步：套公式。W_below 是更靠近根的各转动组的 w 之和。
    double a = 0, W_below = 0;
    for (int j = 0; j < n; ++j) {
      double s = W_below * R[j];
      for (int k = j; k < n; ++k) s += w[k] * R[k] + u[k];
      a += w[j] * s + W_below * u[j];
      W_below += w[j];
    }
    A[b] = a;
  }
  return A;
}

namespace {

// 所有顶点的世界坐标。
std::vector<Vector3d> world_vertices(const Model& model, const Frames& frames) {
  std::vector<Vector3d> X(model.nvertex());
  for (int v = 0; v < model.nvertex(); ++v) X[v] = world_point(frames, model.vertices[v].body, model.vertices[v].local);
  return X;
}

// 一对图元在弦上 t 处的距离（已减半径）。ids 是 4 个顶点：点–三角形是 (p, a, b, c)，边–边是 (p0, p1, q0, q1)。
double pair_distance_at(const std::vector<Vector3d>& Xa, const std::vector<Vector3d>& Xb, const int ids[4], bool pt,
                        double radii, double t) {
  Vector3d P[4];
  for (int i = 0; i < 4; ++i) P[i] = (1 - t) * Xa[ids[i]] + t * Xb[ids[i]];
  if (pt) {
    const Vector3d lam = point_triangle_closest(P[0], P[1], P[2], P[3]);
    return (P[0] - (lam[0] * P[1] + lam[1] * P[2] + lam[2] * P[3])).norm() - radii;
  }
  double s, u;
  segment_segment_closest(P[0], P[1], P[2], P[3], s, u);
  return (((1 - s) * P[0] + s * P[1]) - ((1 - u) * P[2] + u * P[3])).norm() - radii;
}

// 一对图元的加性 CCD（书 §9.2 的 accd_one，笔记第 4 节）。
//   xi 是这一对的矢高；返回这一段里的安全比例 t（1 表示整段安全）。
//   如果这一段对这一对太长（xi > d(0) / 2），把需要的段长写进 ds_need，返回 1。
double accd_pair(const std::vector<Vector3d>& Xa, const std::vector<Vector3d>& Xb, const int ids[4], bool pt,
                 double radii, double xi, double A_pair, double c, double& ds_need) {
  const double d0 = pair_distance_at(Xa, Xb, ids, pt, radii, 0.0);
  if (d0 <= 0) return 1.0;  // 不应该发生（上一个迭代点不相交）；真发生了也不能让 CCD 卡死
  if (xi > 0.5 * d0) {
    // 段太长：由 A len^2 / 8 <= d0 / 2 解出 len <= sqrt(4 d0 / A)。
    ds_need = std::min(ds_need, 0.9 * std::sqrt(4 * d0 / A_pair));
    return 1.0;
  }
  // 相对运动的上界 l：先减去 4 个顶点的平均位移（整体平移不改变距离），再取两侧最大位移之和。
  //   点–三角形：A 侧是顶点 0，B 侧是 1、2、3；边–边：A 侧是 0、1，B 侧是 2、3。
  Vector3d m = Vector3d::Zero();
  for (int i = 0; i < 4; ++i) m += (Xb[ids[i]] - Xa[ids[i]]) * 0.25;
  double la = 0, lb = 0;
  for (int i = 0; i < 4; ++i) {
    const double l = (Xb[ids[i]] - Xa[ids[i]] - m).norm();
    const bool in_a = pt ? i == 0 : i <= 1;
    (in_a ? la : lb) = std::max(in_a ? la : lb, l);
  }
  const double ell = la + lb;
  if (ell <= 0) return 1.0;  // 没有相对运动
  // 推进：每次最多走 (d - xi) / l，再留 10% 余量；距离减到起始余量的 (1 - c) 倍就停。
  double t = 0, d = d0;
  const double stop = (1 - c) * (d0 - xi);
  for (int it = 0; it < 1000; ++it) {
    const double tn = t + c * (d - xi) / ell;
    if (tn >= 1.0) return 1.0;
    t = tn;
    d = pair_distance_at(Xa, Xb, ids, pt, radii, t);
    if (d - xi <= stop) break;
  }
  return t;
}

// 刚体在一段两端的包围盒（顶点位置按半径放大）。
struct Aabb {
  Vector3d lo = Vector3d::Constant(std::numeric_limits<double>::infinity());
  Vector3d hi = Vector3d::Constant(-std::numeric_limits<double>::infinity());
  void add(const Vector3d& x, double r) {
    lo = lo.cwiseMin(x - Vector3d::Constant(r));
    hi = hi.cwiseMax(x + Vector3d::Constant(r));
  }
  bool overlaps(const Aabb& o, double margin) const {
    return (lo.array() <= o.hi.array() + margin).all() && (o.lo.array() <= hi.array() + margin).all();
  }
};

}  // namespace

double curved_ccd(const Model& model, const VectorXd& q, const VectorXd& p, double dhat, double c) {
  // 弦加矢高（阶段 5 笔记第 3、4 节，书 §9.2、§9.5；刚体之间：阶段 7 笔记第 4 节）。
  const VectorXd A = acceleration_bounds(model, q, p);
  const double A_max = A.size() > 0 ? A.maxCoeff() : 0.0;
  const bool pairs = model.nedge() > 0 || model.nface() > 0;

  // 第 1 步：段长。最大矢高 A_max ds^2 / 8 不超过 dhat，所以 ds <= sqrt(8 dhat / A_max)。没有转动时整段一次检查。
  double ds = A_max > 0 ? std::min(1.0, std::sqrt(8 * dhat / A_max)) : 1.0;

  // 第 2 步：逐段推进。每段先在两端做 FK，然后对每个候选对做带余量的保守推进。
  double s0 = 0;
  std::vector<Vector3d> Xa = world_vertices(model, forward_kinematics(model, q));
  for (int guard = 0; guard < 10000 && s0 < 1.0; ++guard) {
    const double s1 = std::min(1.0, s0 + ds);
    const double len = s1 - s0;
    const std::vector<Vector3d> Xb = world_vertices(model, forward_kinematics(model, integrate(model, q, p, s1)));

    double t_min = 1.0;    // 这一段里最早被截断的位置（段内比例）
    double ds_need = len;  // 有候选对要求更短的段时，缩短到多少

    // 顶点–平面：弦上距离线性，闭式解。
    for (int v = 0; v < model.nvertex(); ++v) {
      const Vertex& V = model.vertices[v];
      if (V.body == 0) continue;  // 世界上的静态几何不和平面比
      const double xi = A[V.body] * len * len / 8;  // 这个顶点在这一段里偏离弦的上界（矢高）
      for (int pl = 0; pl < static_cast<int>(model.planes.size()); ++pl) {
        const Plane& P = model.planes[pl];
        const double ya = P.normal.dot(Xa[v]) - P.offset - V.radius;
        const double yb = P.normal.dot(Xb[v]) - P.offset - V.radius;
        // 段对这个顶点太长（余量吃掉了一半以上的距离）：要求 A len^2 / 8 <= ya / 2，即 len <= sqrt(4 ya / A)。
        if (xi > 0.5 * ya) {
          ds_need = std::min(ds_need, 0.9 * std::sqrt(4 * ya / A[V.body]));
          continue;
        }
        // 远离平面：弦上距离不减少，整段安全（真实距离 >= yb - xi >= ya - xi > 0）。
        if (yb >= ya) continue;
        // 靠近平面：沿弦“距离 - 矢高”线性减少，最多闭合 c 倍：t = c (ya - xi) / (ya - yb)。
        t_min = std::min(t_min, c * (ya - xi) / (ya - yb));
      }
    }

    // 刚体之间：宽相（两端包围盒按矢高放大）加逐对 ACCD。
    if (pairs) {
      std::vector<Aabb> box(model.nbody());
      for (int v = 0; v < model.nvertex(); ++v) {
        const Vertex& V = model.vertices[v];
        box[V.body].add(Xa[v], V.radius);
        box[V.body].add(Xb[v], V.radius);
      }
      for (int a = 0; a < model.nbody(); ++a) {
        for (int b = a + 1; b < model.nbody(); ++b) {
          if (!model.collide(a, b)) continue;
          const double A_pair = A[a] + A[b];
          const double xi = A_pair * len * len / 8;
          if (!box[a].overlaps(box[b], xi)) continue;
          int ids[4];
          // 点–三角形，两个方向。
          for (int v = 0; v < model.nvertex(); ++v) {
            const int bv = model.vertices[v].body;
            if (bv != a && bv != b) continue;
            for (int f = 0; f < model.nface(); ++f) {
              const Face& F = model.faces[f];
              if (F.body != (bv == a ? b : a)) continue;
              ids[0] = v;
              ids[1] = F.v0;
              ids[2] = F.v1;
              ids[3] = F.v2;
              const double radii = model.vertices[v].radius + model.vertices[F.v0].radius;
              t_min = std::min(t_min, accd_pair(Xa, Xb, ids, true, radii, xi, A_pair, c, ds_need));
            }
          }
          // 边–边。
          for (int e1 = 0; e1 < model.nedge(); ++e1) {
            if (model.edges[e1].body != a) continue;
            for (int e2 = 0; e2 < model.nedge(); ++e2) {
              if (model.edges[e2].body != b) continue;
              ids[0] = model.edges[e1].v0;
              ids[1] = model.edges[e1].v1;
              ids[2] = model.edges[e2].v0;
              ids[3] = model.edges[e2].v1;
              const double radii = model.vertices[ids[0]].radius + model.vertices[ids[2]].radius;
              t_min = std::min(t_min, accd_pair(Xa, Xb, ids, false, radii, xi, A_pair, c, ds_need));
            }
          }
        }
      }
    }

    if (ds_need < len) {
      // 第 3 步：有候选对嫌这段太长，缩短后重做这一段（Xa 不变，Xb 要重算）。
      ds = std::min(0.5 * len, ds_need);
      continue;
    }
    if (t_min < 1.0) return s0 + t_min * len;  // 这一段里被截断：之前的段都安全，答案就在这里
    s0 = s1;
    Xa = Xb;
  }
  return std::min(s0, 1.0);
}

double trajectory_min_distance(const Model& model, const VectorXd& q, const VectorXd& p, double alpha, int samples) {
  double dmin = std::numeric_limits<double>::infinity();
  for (int k = 0; k <= samples; ++k) {
    const double s = alpha * k / samples;
    dmin = std::min(dmin, min_distance(model, forward_kinematics(model, integrate(model, q, p, s))));
  }
  return dmin;
}

}  // namespace rcipc
