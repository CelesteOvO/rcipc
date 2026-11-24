#include "rcipc/ccd.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
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
    if (model.weld[model.vertices[v].body] == 0) continue;  // 焊在世界上的静态几何不和平面比（同 for_each_candidate）
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

// 整段 s ∈ [0, 1] 上 |x'''| 的界 T，x 是刚体 b 上的任意一点（紧界的第 2 部分；证明在阶段 5 笔记第 6.1–6.4 节）。
// chain 是 b 到根的链上的关节（根在前，都是转动的），rate 是它们的转速。F 是 s = 0 处的 FK。
//
// 沿 q ⊕ s p，每个关节绕一根在父坐标系里固定的轴匀速转（铰链显然；球关节、自由关节的转动是 R0 exp(s w)，
// 在父坐标系里的角速度 R0 w 不随 s 变）。记 w_j 是关节 j 的转动向量，th_j = |w_j|，o_j 是锚点，r_j = x - o_j，
// D_j 是“在关节 j 的子坐标系里求导”。逐层只有一条恒等式：
//   D_{j-1} = D_j + w_j x                    （父坐标系里的导数 = 子坐标系里的导数 + 被 j 带着转）
// w_j 在父、子两个坐标系里都是常向量，右边两个算子可交换，三次方按二项式展开：
//   D_{j-1}^3 r_j = D_j^3 r_j + 3 w_j x D_j^2 r_j + 3 w_j x (w_j x D_j r_j) + w_j x (w_j x (w_j x r_j))
// D_j^k r_j 是把 j 和它之前的关节都冻住时点的 k 阶导数。记 v_j、a_j、t_j 为把 j 之前的关节冻住时
// 点的速度、加速度、三阶导数在整段上的界，逐项取长度：
//   c_j = th_j (d_j + lever + v_{j+1})       c_j >= |w_j x r_j|：关节 j 单独带给点的速度
//   t_j = t_{j+1} + 3 th_j a_{j+1} + 3 th_j^2 v_{j+1} + th_j^2 c_j
//   a_j = a_{j+1} + 2 th_j v_{j+1} + th_j c_j
//   v_j = v_{j+1} + c_j
// T = t_1（最靠近根的关节；x''' 就是世界系里 r_1 的三阶导数，自由关节的平移是直线，不贡献）。
// c_j 里的括号是整段上点到 j 的轴的距离的界：d_j 是 s = 0 处刚体原点到轴的距离（球关节、自由关节的轴方向跟着 p 变，
// 取到锚点的距离），lever 是刚体半径（刚体上的点离原点不超过它），v_{j+1} 是这段距离每单位 s 最多长多少
// （站在 j 的子刚体上看，轴不动，点的速度不超过 v_{j+1}）。
//
// 2026-11-12 之前这里是书 §9.3.2 的式子 T = sum_j th_j (2 B_<j rho_j + 4 A_<j^2 rho_j + 4 A_<j M_j + R_j)，
// 它也成立（笔记第 6.6 节），但同一个量被三角不等式拆了好几次：两个关节时中间两个系数是 7、5，这里是 3、3。
//
// 从刚体往根走（最深的关节先）。terms 不为空时，把每个关节的中间量记下来（顺序是从刚体往根）。
double third_derivative_bound(const Model& model, const Frames& F, int b, const std::vector<int>& chain,
                              const std::vector<double>& rate, std::vector<TightTerm>* terms) {
  const Vector3d centre = F.xpos[b];
  const double lever = model.bodies[b].radius;
  double v = 0, a = 0, t = 0;  // 进循环体时是 v_{j+1}、a_{j+1}、t_{j+1}，出来时是 v_j、a_j、t_j
  for (int k = static_cast<int>(chain.size()) - 1; k >= 0; --k) {
    const double th = rate[k];
    if (th == 0) continue;
    const Joint& J = model.joints[chain[k]];
    const Vector3d anchor = J.type == JointType::kFree ? F.xpos[J.body] : F.dof_anchor[J.dadr];
    const Vector3d off = centre - anchor;
    double dist = off.norm();  // 铰链：到轴的距离；球关节、自由关节：到锚点的距离
    if (J.type == JointType::kHinge) {
      const Vector3d ax = F.dof_axis[J.dadr].normalized();
      dist = (off - ax * off.dot(ax)).norm();
    }
    const double c = th * (dist + lever + v);
    t += th * (3 * a + th * (3 * v + c));  // 先 t，再 a，再 v：右边用的都是 j+1 的值
    a += th * (2 * v + c);
    v += c;
    if (terms) {
      const bool free = J.type == JointType::kFree, hinge = J.type == JointType::kHinge;
      terms->push_back({free ? J.dadr + 3 : J.dadr, hinge ? 1 : 3, th, c, v, a, t});
    }
  }
  return t;
}

}  // namespace

void tight_acceleration_parts(const Model& model, const Frames& F, const VectorXd& p, VectorXd* a0, VectorXd* T3) {
  // 紧的加速度界（阶段 5 笔记第 6 节；a0 的递推同书 §9.3.2，T 见上面的 third_derivative_bound）。
  //
  // 静态界松在两处：力臂取了“任何姿态下最坏的路径长度”，而且把 x'' 在整段上的每一项都按最坏情况放大。
  // 紧界换一个思路：s = 0 处的二阶导数可以精确算出来，只需要再给“从 0 到 s 它最多变多少”一个界（三阶导数的界 T）：
  //   a0_b = max_{v 在 b 上} |x_v''(0)|，    中值定理：|x''(s)| <= a0_b + T_b s，s 在 [0, 1] 上。
  // curved_ccd 整步用一个界，取 s = 1（A_b = a0_b + T_b）。
  // 链上有滑动关节的刚体不算（推导只有转动），两部分都返回 +无穷，由调用者和静态界取较小的那个。
  const double inf = std::numeric_limits<double>::infinity();
  *a0 = VectorXd::Constant(model.nbody(), inf);
  *T3 = VectorXd::Constant(model.nbody(), inf);
  (*a0)[0] = 0;
  (*T3)[0] = 0;
  std::vector<int> chain;  // 刚体到根的链上的关节，根在前
  for (int b = 1; b < model.nbody(); ++b) {
    chain.clear();
    bool slide = false;
    for (int k = b; k != 0; k = model.bodies[k].parent) {
      const int j = model.bodies[k].joint;
      if (j < 0) continue;
      chain.push_back(j);
      slide = slide || model.joints[j].type == JointType::kSlide;
    }
    if (slide) continue;
    std::reverse(chain.begin(), chain.end());

    // 第 1 部分：s = 0 处的二阶导数，沿链递推（都按单位 s 计）。W 角速度、Al 角加速度、O 参考点、AO 参考点的加速度。
    //   自由关节：参考点是根的原点，它的平移对 s 是线性的（AO = 0），转速 W 在世界系里是常数（A = 0）；
    //   铰链、球关节（锚点 P、转动向量 wr）：先把参考点移到 P（刚体运动学：a_P = a_O + A x r + W x (W x r)），
    //   再叠上这个关节的转动：A <- A + W x wr（wr 被上游带着转），W <- W + wr。
    Vector3d W = Vector3d::Zero(), Al = Vector3d::Zero(), O = Vector3d::Zero(), AO = Vector3d::Zero();
    std::vector<double> rate(chain.size());  // 每个关节的转速 |wr|
    for (size_t i = 0; i < chain.size(); ++i) {
      const Joint& J = model.joints[chain[i]];
      const int d0 = J.type == JointType::kFree ? J.dadr + 3 : J.dadr;
      const int n = J.type == JointType::kHinge ? 1 : 3;
      Vector3d wr = Vector3d::Zero();
      for (int d = d0; d < d0 + n; ++d) wr += p[d] * F.dof_axis[d];
      rate[i] = p.segment(d0, n).norm();
      if (J.type == JointType::kFree) {
        O = F.xpos[J.body];
        AO.setZero();
        W = wr;
        Al.setZero();
      } else {
        const Vector3d P = F.dof_anchor[d0];
        const Vector3d r = P - O;
        AO += Al.cross(r) + W.cross(W.cross(r));
        Al += W.cross(wr);
        W += wr;
        O = P;
      }
    }
    double p2 = 0;  // 顶点上 |x''(0)| 的最大值（x'' 对 x 是仿射的，图元上的点不会更大）
    for (int v = 0; v < model.nvertex(); ++v) {
      if (model.vertices[v].body != b) continue;
      const Vector3d r = world_point(F, b, model.vertices[v].local) - O;
      const Vector3d xdd = AO + Al.cross(r) + W.cross(W.cross(r));
      p2 = std::max(p2, xdd.norm());
    }

    // 第 2 部分：整段上 |x'''| 的界 T。
    (*a0)[b] = p2;
    (*T3)[b] = third_derivative_bound(model, F, b, chain, rate, nullptr);
  }
}

std::vector<TightTerm> tight_terms(const Model& model, const Frames& F, const VectorXd& p, int body) {
  std::vector<int> chain;
  for (int k = body; k != 0; k = model.bodies[k].parent) {
    const int j = model.bodies[k].joint;
    if (j < 0) continue;
    if (model.joints[j].type == JointType::kSlide) return {};
    chain.push_back(j);
  }
  std::reverse(chain.begin(), chain.end());
  std::vector<double> rate(chain.size());
  for (size_t i = 0; i < chain.size(); ++i) {
    const Joint& J = model.joints[chain[i]];
    rate[i] = p.segment(J.type == JointType::kFree ? J.dadr + 3 : J.dadr, J.type == JointType::kHinge ? 1 : 3).norm();
  }
  std::vector<TightTerm> terms;
  third_derivative_bound(model, F, body, chain, rate, &terms);
  std::reverse(terms.begin(), terms.end());  // 循环是从刚体往根走的，这里翻成根在前
  return terms;
}

VectorXd tight_acceleration_bounds(const Model& model, const Frames& F, const VectorXd& p) {
  VectorXd a0, T3;
  tight_acceleration_parts(model, F, p, &a0, &T3);
  return a0 + T3;
}

VectorXd ccd_acceleration_bounds(const Model& model, const VectorXd& q, const Frames& F, const VectorXd& p) {
  // 两个都是整段上的上界，取较小的那个。单个铰链时静态界就是精确的向心加速度，比紧界还紧；链深时紧界紧得多。
  return acceleration_bounds(model, q, p).cwiseMin(tight_acceleration_bounds(model, F, p));
}

namespace {

// 所有顶点的世界坐标。
std::vector<Vector3d> world_vertices(const Model& model, const Frames& frames) {
  std::vector<Vector3d> X(model.nvertex());
  for (int v = 0; v < model.nvertex(); ++v) X[v] = world_point(frames, model.vertices[v].body, model.vertices[v].local);
  return X;
}

// 一对图元在弦上 t 处的最近点之差 r = x_A - x_B（A 侧是点–三角形的点、边–边的第一条边）。ids 是 4 个顶点：
// 点–三角形是 (p, a, b, c)，边–边是 (p0, p1, q0, q1)。P 返回这 4 个顶点在 t 处的位置。
Vector3d pair_offset_at(const std::vector<Vector3d>& Xa, const std::vector<Vector3d>& Xb, const int ids[4], bool pt,
                        double t, Vector3d P[4]) {
  for (int i = 0; i < 4; ++i) P[i] = (1 - t) * Xa[ids[i]] + t * Xb[ids[i]];
  if (pt) {
    const Vector3d lam = point_triangle_closest(P[0], P[1], P[2], P[3]);
    return P[0] - (lam[0] * P[1] + lam[1] * P[2] + lam[2] * P[3]);
  }
  double s, u;
  segment_segment_closest(P[0], P[1], P[2], P[3], s, u);
  return ((1 - s) * P[0] + s * P[1]) - ((1 - u) * P[2] + u * P[3]);
}

// 一对图元在弦上 t 处的距离（已减半径）。
double pair_distance_at(const std::vector<Vector3d>& Xa, const std::vector<Vector3d>& Xb, const int ids[4], bool pt,
                        double radii, double t) {
  Vector3d P[4];
  return pair_offset_at(Xa, Xb, ids, pt, t, P).norm() - radii;
}

constexpr int kAccdMax = 1000;  // ACCD 一对最多走几步（GPU 版是 kernels.h 的 kAccdMax）

// A + B tau + C tau^2（A > 0）的第一个正根，没有就返回 +无穷。
double first_positive_root(double A, double B, double C) {
  const double inf = std::numeric_limits<double>::infinity();
  if (C == 0) return B < 0 ? A / -B : inf;
  const double disc = B * B - 4 * A * C;
  if (disc < 0) return inf;  // 只在 C > 0 时出现：处处为正
  const double qq = -0.5 * (B + (B >= 0 ? 1.0 : -1.0) * std::sqrt(disc));
  double best = inf;
  for (const double rt : {qq / C, A / qq})
    if (rt > 0) best = std::min(best, rt);
  return best;
}

// 转动的分离平面的转速 nu（垂直于 n）。两侧图元里各取“边”（点–三角形：三角形的两条边；边–边：两条边各一条），
//   要求 n(tau) = n + tau nu 和每条边 (u, w) 在一阶上保持同样的夹角：nu·(P_w - P_u) = -n·(D_w - D_u)。
//   两个方程、两个未知数（nu 在 n 的切平面里），边平行时退化，用最小二乘加一点正则。nu 取什么界都成立，这里只管紧不紧。
Vector3d plane_rate(const Vector3d P[4], const Vector3d D[4], bool pt, const Vector3d& n) {
  const Vector3d a = std::abs(n.x()) < 0.9 ? Vector3d::UnitX() : Vector3d::UnitY();
  const Vector3d u1 = n.cross(a).normalized(), u2 = n.cross(u1);
  const int e[2][2] = {{pt ? 1 : 0, pt ? 2 : 1}, {pt ? 1 : 2, 3}};
  Eigen::Matrix2d M = Eigen::Matrix2d::Zero();
  Eigen::Vector2d rhs = Eigen::Vector2d::Zero();
  for (const auto& k : e) {
    const Vector3d ed = P[k[1]] - P[k[0]];
    const Eigen::Vector2d g(ed.dot(u1), ed.dot(u2));
    M += g * g.transpose();
    rhs -= n.dot(D[k[1]] - D[k[0]]) * g;
  }
  const double reg = 1e-9 * M.trace() + 1e-300;
  M(0, 0) += reg;
  M(1, 1) += reg;
  const double det = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
  const Eigen::Vector2d x(M(1, 1) * rhs[0] - M(0, 1) * rhs[1], M(0, 0) * rhs[1] - M(1, 0) * rhs[0]);
  if (!(det > 0) || !(x / det).allFinite()) return Vector3d::Zero();
  return (x[0] * u1 + x[1] * u2) / det;
}

// 分离平面的界（笔记 notes/22_primal_dual.md 第 3 节）。对任何单位向量 n 和两侧图元上的点 x、y 都有
//   |x - y| >= n·(x - y)，而图元是它的顶点的凸包，所以
//     d(t_k + tau) + radii >= min_{j in A, i in B} n(tau)·(P_j(tau) - P_i(tau))，
//   P(tau) = P + tau D 是顶点沿弦的位置（D = Xb - Xa）。n(tau) 可以随 tau 变，只要是单位向量。
//   取 n(tau) = (n + tau nu) / |n + tau nu|，n 是 t_k 处的一个分离方向（r 归一化），nu 垂直于 n：
//     (n + tau nu)·(m + tau s) = n·m + tau (n·s + nu·m) + tau^2 nu·s，   m = P_j - P_i，s = D_j - D_i，
//     |n + tau nu| = sqrt(1 + tau^2 |nu|^2) <= 1 + tau^2 |nu|^2 / 2，
//   所以只要每一对顶点都有 n·m - L + tau (n·s + nu·m) + tau^2 (nu·s - L |nu|^2 / 2) >= 0，距离就 >= L - radii。
//   nu = 0（rotating = false）：固定的平面，每对顶点一条直线；贴着平面平移滑动时斜率都是 0，整段安全。
//   nu = plane_rate：平面跟着图元转，一阶项变成接触点的法向相对速度（转着滑动时是 0），剩下的是二阶的真实闭合。
// 返回所有顶点对都满足的最长的 tau（都不往下走时返回 +无穷）；level = L。
double plane_step(const Vector3d P[4], const Vector3d D[4], bool pt, const Vector3d& r, double level, bool rotating) {
  const double rn = r.norm();
  if (!(rn > 0)) return 0.0;
  const Vector3d n = r / rn;
  const Vector3d nu = rotating ? plane_rate(P, D, pt, n) : Vector3d::Zero();
  const double half_nu2 = 0.5 * nu.squaredNorm();
  double tau = std::numeric_limits<double>::infinity();
  const int na = pt ? 1 : 2;  // A 侧的顶点是 0..na-1，B 侧是 na..3
  for (int j = 0; j < na; ++j) {
    for (int i = na; i < 4; ++i) {
      const Vector3d m = P[j] - P[i], s = D[j] - D[i];
      const double A = n.dot(m) - level;
      if (!(A > 0)) return 0.0;
      tau = std::min(tau, first_positive_root(A, n.dot(s) + nu.dot(m), nu.dot(s) - level * half_nu2));
    }
  }
  return tau;
}

// 两个分离方向，固定的平面和转动的平面，取走得最远的那个：
//   1. 最近点的方向。间隙很小、又没有半径时它算不准：r 只有 1e-9 m，舍入 1e-17 m 就让方向偏 1e-8 rad，
//      0.36 m 长的边的另一端在 n 上就差出 4e-9 m，和间隙一样大。
//   2. 图元自己的法向：三角形的法向，两条边的叉积，朝 A 那一侧。它只用到边的方向，和间隙无关；
//      最近点在三角形里面、两条边交叉时，它就是准确的分离方向。GPU 版尾巴的证明（chord_lower_bound）也试这两个。
double plane_steps(const Vector3d P[4], const Vector3d D[4], bool pt, const Vector3d& r, double level) {
  double best = plane_step(P, D, pt, r, level, false);
  best = std::max(best, plane_step(P, D, pt, r, level, true));
  const Vector3d e1 = pt ? Vector3d(P[2] - P[1]) : Vector3d(P[1] - P[0]);
  const Vector3d e2 = pt ? Vector3d(P[3] - P[1]) : Vector3d(P[3] - P[2]);
  const Vector3d N = e1.cross(e2);
  const double side = N.dot(r);
  // 平行的边、退化的三角形（sin < 1e-6），或者 r 正好在面内：没有这个方向。
  if (!(N.norm() > 1e-6 * e1.norm() * e2.norm()) || side == 0) return best;
  const Vector3d n = side > 0 ? N : Vector3d(-N);
  best = std::max(best, plane_step(P, D, pt, n, level, false));
  best = std::max(best, plane_step(P, D, pt, n, level, true));
  return best;
}

// 一对图元的加性 CCD（书 §9.2 的 accd_one，笔记 07 第 4 节），每步再用分离平面的界（笔记 22 第 3 节）。
//   xi 是这一对的矢高；返回这一段里的安全比例 t（1 表示整段安全）。
//   如果这一段对这一对太长（xi > d(0) / 2），把它的 d(0) 记进 d0_long（取最小），返回 1；
//   调用者按这一对刚体里最小的 d(0) 算需要的段长（两侧刚体相同的对，d(0) 越小要求的段越短）。
//   radii 里已经加上了最小分离 sep，所以这里的“距离”都是 d - sep。iters、capped 不为空时给走了几步、有没有走满上限。
double accd_pair(const std::vector<Vector3d>& Xa, const std::vector<Vector3d>& Xb, const int ids[4], bool pt,
                 double radii, double xi, double c, double& d0_long, int& stalled, int* iters = nullptr,
                 bool* capped = nullptr) {
  const double d0 = pair_distance_at(Xa, Xb, ids, pt, radii, 0.0);
  // 起点已经在最小分离以内：弦加矢高给不出任何保证，卡住。原来这里返回 1（放行），
  // 浮点把 1e-16 m 的间隙算成负数时整步就穿过去了（笔记 22 第 5 节）。
  // IPC Toolkit 的 additive CCD 在起点距离不超过最小分离时也是返回碰撞时刻 0。
  if (d0 <= 0) {
    ++stalled;
    return 0.0;
  }
  if (xi > 0.5 * d0) {
    d0_long = std::min(d0_long, d0);
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
  // 推进：每一步要保住的下界是 level（距离，已减半径），它不低于停下的门槛 stop = (1 - c)(d0 - xi)，也不低于当前余量的
  //   (1 - c) 倍。两个步长都保证这一点，取走得远的那个：相对位移的界 (d - level) / l，和分离平面的界（plane_steps）。
  //   贴着表面滑动时相对位移的界每步只走 c (d - xi) / l，分离平面的界整段安全。离门槛 10% 以内就停：一次最多闭合 c。
  double t = 0, d = d0;
  const double stop = (1 - c) * (d0 - xi);
  Vector3d P[4], D[4];
  for (int i = 0; i < 4; ++i) D[i] = Xb[ids[i]] - Xa[ids[i]];
  Vector3d r = pair_offset_at(Xa, Xb, ids, pt, 0.0, P);
  for (int it = 0; it < kAccdMax; ++it) {
    const double level = xi + std::max(stop, (1 - c) * (d - xi));
    double step = (d - level) / ell;
    step = std::max(step, plane_steps(P, D, pt, r, radii + level));
    const double tn = t + step;
    if (tn >= 1.0) return 1.0;
    t = tn;
    r = pair_offset_at(Xa, Xb, ids, pt, t, P);
    d = r.norm() - radii;
    if (iters) *iters = it + 1;
    if (d - xi <= 1.1 * stop) return t;
  }
  // 走满上限还没停下：返回走到的位置。它是保守的（之前的每一步都证明了安全），只是可能很短。
  if (capped) *capped = true;
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

double accd_chord(const Vector3d x0[4], const Vector3d x1[4], bool pt, double radii, double xi, double c, int* iters,
                  bool* capped) {
  const std::vector<Vector3d> Xa(x0, x0 + 4), Xb(x1, x1 + 4);
  const int ids[4] = {0, 1, 2, 3};
  double d0_long = std::numeric_limits<double>::infinity();
  int stalled = 0;
  if (iters) *iters = 0;
  if (capped) *capped = false;
  const double t = accd_pair(Xa, Xb, ids, pt, radii, xi, c, d0_long, stalled, iters, capped);
  return d0_long < std::numeric_limits<double>::infinity() ? -1.0 : t;
}

double curved_ccd(const Model& model, const VectorXd& q, const VectorXd& p, double dhat, double c, bool tight,
                  double sep) {
  // 弦加矢高（阶段 5 笔记第 3、4 节，书 §9.2、§9.5；刚体之间：阶段 7 笔记第 4 节）。
  const double inf = std::numeric_limits<double>::infinity();
  // 只有带顶点的刚体会碰到东西：分段是为了让每个顶点偏离弦的距离不超过 dhat，没有顶点的刚体不该缩短段长。
  //   运动的刚体全都没有顶点时，没有东西要查，整步放行。
  std::vector<char> has_vertex(model.nbody(), 0);
  for (const Vertex& V : model.vertices) has_vertex[V.body] = 1;
  has_vertex[0] = 0;
  if (std::find(has_vertex.begin(), has_vertex.end(), 1) == has_vertex.end()) return 1.0;
  const Frames F0 = forward_kinematics(model, q);
  const VectorXd A_static = acceleration_bounds(model, q, p);
  VectorXd A = A_static;
  if (tight) {  // 静态界和紧界逐个刚体取小（两个都对整步成立）
    VectorXd a0, T3;
    tight_acceleration_parts(model, F0, p, &a0, &T3);
    A = A_static.cwiseMin(a0 + T3);
  }
  const bool pairs = model.nedge() > 0 || model.nface() > 0;

  // 第 1 步：段长。最大矢高 A_max ds^2 / 8 不超过 dhat，所以 ds <= sqrt(8 dhat / A_max)。没有转动时整段一次检查。
  double ds = 1.0;
  double A_max = 0;
  for (int b = 1; b < model.nbody(); ++b)
    if (has_vertex[b]) A_max = std::max(A_max, A[b]);
  if (A_max > 0) ds = std::min(1.0, std::sqrt(8 * dhat / A_max));

  // 某一对（刚体 a、b；顶点–平面时 b = -1）的矢高超过它当前距离 d0 的一半时，这一段要缩到多短：
  //   由 A len^2 / 8 <= d0 / 2 解出 len <= sqrt(4 d0 / A)，再留 10%。
  const auto shorter = [&](int a, int b, double d0) {
    const double A_now = b >= 0 ? A[a] + A[b] : A[a];
    return 0.9 * std::sqrt(4 * d0 / A_now);
  };

  // 第 2 步：逐段推进。每段先在两端做 FK，然后对每个候选对做带余量的保守推进。
  double s0 = 0;
  std::vector<Vector3d> Xa = world_vertices(model, F0);
  for (int guard = 0; guard < 10000 && s0 < 1.0; ++guard) {
    const double s1 = std::min(1.0, s0 + ds);
    const double len = s1 - s0;
    const Frames Fb = forward_kinematics(model, integrate(model, q, p, s1));
    const std::vector<Vector3d> Xb = world_vertices(model, Fb);

    double t_min = 1.0;    // 这一段里最早被截断的位置（段内比例）
    double ds_need = len;  // 有候选对要求更短的段时，缩短到多少
    int stalled = 0;       // 起点已经在最小分离以内的对

    // 顶点–平面：弦上距离线性，闭式解。距离都减去最小分离 sep。
    for (int v = 0; v < model.nvertex(); ++v) {
      const Vertex& V = model.vertices[v];
      if (model.weld[V.body] == 0) continue;  // 焊在世界上的静态几何不和平面比（同 for_each_candidate）
      const double xi = A[V.body] * len * len / 8;  // 这个顶点在这一段里偏离弦的上界（矢高）
      for (int pl = 0; pl < static_cast<int>(model.planes.size()); ++pl) {
        const Plane& P = model.planes[pl];
        const double ya = P.normal.dot(Xa[v]) - P.offset - V.radius - sep;
        const double yb = P.normal.dot(Xb[v]) - P.offset - V.radius - sep;
        // 起点已经在最小分离以内：卡住。原来 ya < 0 时下一行的 sqrt(4 ya / A) 是 NaN，
        // std::min 忽略 NaN，这个顶点就不再受约束，等于放行。
        if (ya <= 0) {
          ++stalled;
          t_min = 0;
          continue;
        }
        // 段对这个顶点太长（余量吃掉了一半以上的距离）：要求 A len^2 / 8 <= ya / 2，即 len <= sqrt(4 ya / A)。
        if (xi > 0.5 * ya) {
          ds_need = std::min(ds_need, shorter(V.body, -1, ya));
          continue;
        }
        // 远离平面：弦上距离不减少，整段安全（真实距离 >= yb - xi >= ya - xi > 0）。
        if (yb >= ya) continue;
        // 靠近平面：沿弦“距离 - 矢高”线性减少，最多闭合 c 倍：t = c (ya - xi) / (ya - yb)。
        t_min = std::min(t_min, c * (ya - xi) / (ya - yb));
      }
    }

    // 刚体之间：宽相加逐对 ACCD。宽相两层（书 §9.7）：
    //   1. 刚体对：两端的包围盒（扫掠盒）按矢高放大后不重叠的，跳过；
    //   2. 图元对（阶段 11 前加的）：两个图元各自的扫掠盒（顶点在段两端的位置）按矢高放大后不重叠的，跳过。
    //      理由和第 1 层一样：盒子沿某根轴隔开超过 xi，弦上的点就隔开超过 xi，弧线上的点（离弦不超过各自的矢高）就不会碰。
    //      网格的刚体有上万个图元对，绝大多数离得远，这一层省掉了它们的 ACCD。GPU 版用同样的规则，两边的 alpha_max 才一致。
    if (pairs) {
      std::vector<Aabb> box(model.nbody()), vbox(model.nvertex());
      for (int v = 0; v < model.nvertex(); ++v) {
        const Vertex& V = model.vertices[v];
        vbox[v].add(Xa[v], V.radius);
        vbox[v].add(Xb[v], V.radius);
        box[V.body].add(Xa[v], V.radius);
        box[V.body].add(Xb[v], V.radius);
      }
      // 图元的扫掠盒 = 它的顶点的扫掠盒的并。
      const auto merge = [&](std::initializer_list<int> ids) {
        Aabb out;
        for (int v : ids) {
          out.lo = out.lo.cwiseMin(vbox[v].lo);
          out.hi = out.hi.cwiseMax(vbox[v].hi);
        }
        return out;
      };
      for (int a = 0; a < model.nbody(); ++a) {
        for (int b = a + 1; b < model.nbody(); ++b) {
          if (!model.collide(a, b)) continue;
          const double xi = (A[a] + A[b]) * len * len / 8;
          const double margin = xi + sep;  // 盒子隔开超过 xi + sep，弧线上的真实距离就 > sep
          if (!box[a].overlaps(box[b], margin)) continue;
          int ids[4];
          double d0_long = inf;  // 这一对刚体里，嫌这一段太长的图元对的最小距离
          // 点–三角形，两个方向。
          for (int v = 0; v < model.nvertex(); ++v) {
            const int bv = model.vertices[v].body;
            if (bv != a && bv != b) continue;
            if (!vbox[v].overlaps(box[bv == a ? b : a], margin)) continue;  // 顶点离对方整个刚体都远
            for (int f = 0; f < model.nface(); ++f) {
              const Face& F = model.faces[f];
              if (F.body != (bv == a ? b : a)) continue;
              if (!vbox[v].overlaps(merge({F.v0, F.v1, F.v2}), margin)) continue;
              ids[0] = v;
              ids[1] = F.v0;
              ids[2] = F.v1;
              ids[3] = F.v2;
              const double radii = model.vertices[v].radius + model.vertices[F.v0].radius + sep;
              t_min = std::min(t_min, accd_pair(Xa, Xb, ids, true, radii, xi, c, d0_long, stalled));
            }
          }
          // 边–边。
          for (int e1 = 0; e1 < model.nedge(); ++e1) {
            if (model.edges[e1].body != a) continue;
            const Aabb b1 = merge({model.edges[e1].v0, model.edges[e1].v1});
            if (!b1.overlaps(box[b], margin)) continue;
            for (int e2 = 0; e2 < model.nedge(); ++e2) {
              if (model.edges[e2].body != b) continue;
              if (!b1.overlaps(merge({model.edges[e2].v0, model.edges[e2].v1}), margin)) continue;
              ids[0] = model.edges[e1].v0;
              ids[1] = model.edges[e1].v1;
              ids[2] = model.edges[e2].v0;
              ids[3] = model.edges[e2].v1;
              const double radii = model.vertices[ids[0]].radius + model.vertices[ids[2]].radius + sep;
              t_min = std::min(t_min, accd_pair(Xa, Xb, ids, false, radii, xi, c, d0_long, stalled));
            }
          }
          if (d0_long < inf) ds_need = std::min(ds_need, shorter(a, b, d0_long));
        }
      }
    }

    // 有一对卡住了：段的起点 s0 之前都已经查过是安全的，s0 之后一点也不能走（不用再缩短重做）。
    if (stalled > 0) return s0;
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
