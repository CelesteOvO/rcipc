// GPU 版的曲线 CCD（阶段 11，笔记 notes/11_gpu.md 第 4 节）。对应 CPU 版 ccd.cc 的 acceleration_bounds 和 curved_ccd 的一段。
//
// 一段 [s0, s0 + len] 的检查分两部分，并行方式和接触检测一样：
//   - 顶点–平面：一个线程一个顶点，闭式解；
//   - 刚体之间：先用扫掠盒找候选刚体对（launch_body_pairs，余量是两侧的矢高），
//     再剔除（narrow.cu，在刚体坐标系里），剩下的候选图元对一个线程一对：先按 CPU 版的盒子规则用双精度复核，
//     再做 ACCD（k_ccd_exact）。
//     前两步可以按窗口做（launch_ccd_cull 的 window）：一批候选给窗口里的好几段用，每段只重做复核和 ACCD（笔记第 13 节）。
// 截断位置和需要的更短段长都用原子 min 汇总。分段、缩短重做的循环在主机上（stepper.cc），和 CPU 版一样。
#include <algorithm>
#include <cfloat>

#include "devmath.cuh"
#include "kernels.h"

namespace rcipc::gpu {

namespace {

constexpr int kThreads = 128;
inline int blocks(int n) { return (n + kThreads - 1) / kThreads; }

// 紧界（ccd.cc 的 tight_acceleration_bounds，阶段 5 笔记第 6 节）：s = 0 处的二阶导数沿链递推，取刚体顶点上的最大值，
// 再加整段上三阶导数的界 T（笔记第 6.1–6.4 节）。jid 是链上的关节（根在前），th 是它们的转速；链上没有滑动关节。
__device__ double tight_bound(const DModel& m, const DFrames& F, const double* p, int b, const int* jid,
                              const double* th, int n) {
  V3 W = v3(0, 0, 0), Al = v3(0, 0, 0), O = v3(0, 0, 0), AO = v3(0, 0, 0);
  for (int i = 0; i < n; ++i) {
    const int j = jid[i], type = m.jtype[j];
    const int d0 = type == kFree ? m.jdadr[j] + 3 : m.jdadr[j];
    const int nd = type == kHinge ? 1 : 3;
    V3 wr = v3(0, 0, 0);
    for (int d = d0; d < d0 + nd; ++d) wr = wr + p[d] * v3load(F.daxis + 3 * d);
    if (type == kFree) {
      O = v3load(F.xpos + 3 * m.jbody[j]);
      AO = v3(0, 0, 0);
      W = wr;
      Al = v3(0, 0, 0);
    } else {
      const V3 P = v3load(F.danchor + 3 * d0);
      const V3 r = P - O;
      AO = AO + cross(Al, r) + cross(W, cross(W, r));
      Al = Al + cross(W, wr);
      W = W + wr;
      O = P;
    }
  }
  double p2 = 0;
  for (int v = m.bvert0[b]; v < m.bvert0[b] + m.bnvert[b]; ++v) {
    const V3 r = v3load(F.X + 3 * v) - O;
    p2 = fmax(p2, norm(AO + cross(Al, r) + cross(W, cross(W, r))));
  }
  // T：逐层的递推（ccd.cc 的 third_derivative_bound，算式和顺序一样）。v、a、t 是把 j 之前的关节冻住时
  // 点的速度、加速度、三阶导数的界，从刚体往根走。
  const V3 centre = v3load(F.xpos + 3 * b);
  const double lever = m.radius[b];
  double v = 0, a = 0, t = 0;
  for (int k = n - 1; k >= 0; --k) {
    if (th[k] == 0) continue;
    const int j = jid[k], type = m.jtype[j];
    const V3 anchor = type == kFree ? v3load(F.xpos + 3 * m.jbody[j]) : v3load(F.danchor + 3 * m.jdadr[j]);
    const V3 off = centre - anchor;
    double dist = norm(off);
    if (type == kHinge) {
      const V3 ax = v3load(F.daxis + 3 * m.jdadr[j]);
      const V3 axn = (1.0 / norm(ax)) * ax;
      dist = norm(off - dot(off, axn) * axn);
    }
    const double c = th[k] * (dist + lever + v);
    t += th[k] * (3 * a + th[k] * (3 * v + c));
    a += th[k] * (2 * v + c);
    v += c;
  }
  return p2 + t;
}

__global__ void k_accel_bounds(DModel m, const double* q, DFrames F, const double* p, int tight, double* A) {
  // ccd_acceleration_bounds：每个线程一个刚体。静态界（acceleration_bounds）和紧界（tight_acceleration_bounds）
  // 取较小的那个。F 是 q 处的 FK。
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= m.nbody) return;
  if (b == 0) {
    A[b] = 0;
    return;
  }
  // 没有顶点的刚体碰不到东西，不该缩短段长（同 CPU 版 curved_ccd 开头）：界记成 0。它没有图元，也不在任何刚体对里。
  if (m.bnvert[b] == 0) {
    A[b] = 0;
    return;
  }
  // 第 1 步：沿链往上收集（和 chain_joints 一样，力臂里含滑动关节现在的伸长），再反过来变成根在前。
  double w[kMaxDepth], u[kMaxDepth], R[kMaxDepth];
  int rot[kMaxDepth], jid[kMaxDepth];
  int n = 0;
  bool slide = false;
  double path = m.radius[b];
  for (int k = b; k != 0 && n < kMaxDepth; k = m.parent[k]) {
    const int j = m.bjnt[k];
    if (j >= 0) {
      const int type = m.jtype[j];
      const int dadr = type == kFree ? m.jdadr[j] + 3 : m.jdadr[j];
      const int nvj = (type == kSlide || type == kHinge) ? 1 : 3;
      if (type == kSlide) path += fabs(q[m.jqadr[j]] - m.jref[j]);
      double mag = 0;
      for (int d = dadr; d < dadr + nvj; ++d) mag += p[d] * p[d];
      mag = sqrt(mag);
      rot[n] = type != kSlide;
      jid[n] = j;
      slide = slide || type == kSlide;
      w[n] = rot[n] ? mag : 0.0;
      u[n] = rot[n] ? 0.0 : mag;
      R[n] = path + norm(v3load(m.jpos + 3 * j));
      ++n;
    }
    path += norm(v3load(m.bpos + 3 * k));
  }
  for (int i = 0; i < n / 2; ++i) {
    double t = w[i];
    w[i] = w[n - 1 - i];
    w[n - 1 - i] = t;
    t = u[i];
    u[i] = u[n - 1 - i];
    u[n - 1 - i] = t;
    t = R[i];
    R[i] = R[n - 1 - i];
    R[n - 1 - i] = t;
    const int r = rot[i];
    rot[i] = rot[n - 1 - i];
    rot[n - 1 - i] = r;
    const int jj = jid[i];
    jid[i] = jid[n - 1 - i];
    jid[n - 1 - i] = jj;
  }
  // 滑动关节在这一步内最多再伸 u_s：它上游各关节的力臂都要加上这一段。
  for (int s = 0; s < n; ++s) {
    if (rot[s]) continue;
    for (int j = 0; j < s; ++j) R[j] += u[s];
  }
  // 第 2 步：A_b = sum_j w_j (W_{<j} R_j + sum_{k>=j} (w_k R_k + u_k)) + sum_s W_{<s} u_s。
  double a = 0, W_below = 0;
  for (int j = 0; j < n; ++j) {
    double s = W_below * R[j];
    for (int k = j; k < n; ++k) s += w[k] * R[k] + u[k];
    a += w[j] * s + W_below * u[j];
    W_below += w[j];
  }
  if (tight && !slide) a = fmin(a, tight_bound(m, F, p, b, jid, w, n));
  A[b] = a;
}

__global__ void k_ground_ccd(DModel m, const double* Xa, const double* Xb, const double* A, DScalar lenp, double c,
                             double sep, double* tmin, double* ds_need) {
  const double len = lenp.get();
  // curved_ccd 的顶点–平面部分：弦上距离线性，带矢高余量的闭式解。
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= m.nvert) return;
  const int b = m.vbody[v];
  if (m.weld[b] == 0) return;  // 焊在世界上的静态顶点不和平面比（同 k_ground_detect）
  const double xi = A[b] * len * len / 8;
  const V3 xa = v3load(Xa + 3 * v), xb = v3load(Xb + 3 * v);
  for (int pl = 0; pl < m.nplane; ++pl) {
    const V3 n = v3load(m.pn + 3 * pl);
    const double ya = dot(n, xa) - m.poff[pl] - m.vradius[v] - sep;
    const double yb = dot(n, xb) - m.poff[pl] - m.vradius[v] - sep;
    if (ya <= 0) {  // 起点已经在最小分离以内：卡住（同 curved_ccd）。原来这里 sqrt 得 NaN，atomic_min 会把 NaN 写进 ds_need
      atomic_min(tmin, 0.0);
      continue;
    }
    if (xi > 0.5 * ya) {
      atomic_min(ds_need, 0.9 * sqrt(4 * ya / A[b]));
      continue;
    }
    if (yb >= ya) continue;
    atomic_min(tmin, c * (ya - xi) / (ya - yb));
  }
}

// pair_offset_at：一对图元在弦上 t 处的最近点之差 r = x_A - x_B，P 返回 4 个顶点在 t 处的位置（同 CPU 版 ccd.cc）。
__device__ V3 pair_offset_at(const double* Xa, const double* Xb, const int ids[4], bool pt, double t, V3 P[4]) {
  for (int i = 0; i < 4; ++i) P[i] = (1 - t) * v3load(Xa + 3 * ids[i]) + t * v3load(Xb + 3 * ids[i]);
  if (pt) {
    const V3 lam = point_triangle_closest(P[0], P[1], P[2], P[3]);
    return P[0] - (lam.x * P[1] + lam.y * P[2] + lam.z * P[3]);
  }
  double s, u;
  segment_segment_closest(P[0], P[1], P[2], P[3], s, u);
  return ((1 - s) * P[0] + s * P[1]) - ((1 - u) * P[2] + u * P[3]);
}

// pair_distance_at：一对图元在弦上 t 处的距离（已减半径）。
__device__ double pair_distance_at(const double* Xa, const double* Xb, const int ids[4], bool pt, double radii,
                                   double t) {
  V3 P[4];
  return norm(pair_offset_at(Xa, Xb, ids, pt, t, P)) - radii;
}

// ---- 分离平面的界（CPU 版 ccd.cc 的 plane_step、plane_steps，笔记 notes/22_primal_dual.md 第 3 节） ----
// 对任何单位向量 n(tau)，d(t + tau) + radii >= min_{j in A, i in B} n(tau)·(P_j(tau) - P_i(tau))。
// n(tau) = (n + tau nu) / |n + tau nu|：每对顶点一个二次式 A + B tau + C tau^2 >= 0，走到第一个正根。
// 算式和 CPU 版逐条对应（GPU 上乘加会合并，结果只差舍入）。

// A + B tau + C tau^2（A > 0）的第一个正根，没有就返回 +无穷。
__device__ double first_positive_root(double A, double B, double C) {
  if (C == 0) return B < 0 ? A / -B : DBL_MAX * 2;  // DBL_MAX * 2 = +无穷
  const double disc = B * B - 4 * A * C;
  if (disc < 0) return DBL_MAX * 2;
  const double qq = -0.5 * (B + (B >= 0 ? 1.0 : -1.0) * sqrt(disc));
  double best = DBL_MAX * 2;
  const double r1 = qq / C, r2 = A / qq;
  if (r1 > 0) best = fmin(best, r1);
  if (r2 > 0) best = fmin(best, r2);
  return best;
}

// 转动的分离平面的转速 nu（垂直于 n）：每条“边” (u, w) 要求 nu·(P_w - P_u) = -n·(D_w - D_u)，最小二乘加一点正则。
__device__ V3 plane_rate(const V3 P[4], const V3 D[4], bool pt, V3 n) {
  const V3 a = fabs(n.x) < 0.9 ? v3(1, 0, 0) : v3(0, 1, 0);
  V3 u1 = cross(n, a);
  u1 = u1 / norm(u1);
  const V3 u2 = cross(n, u1);
  const int e0[2] = {pt ? 1 : 0, pt ? 1 : 2}, e1[2] = {pt ? 2 : 1, 3};
  double m00 = 0, m01 = 0, m11 = 0, r0 = 0, r1 = 0;
  for (int k = 0; k < 2; ++k) {
    const V3 ed = P[e1[k]] - P[e0[k]];
    const double g0 = dot(ed, u1), g1 = dot(ed, u2);
    const double sk = dot(n, D[e1[k]] - D[e0[k]]);
    m00 += g0 * g0;
    m01 += g0 * g1;
    m11 += g1 * g1;
    r0 -= sk * g0;
    r1 -= sk * g1;
  }
  const double reg = 1e-9 * (m00 + m11) + 1e-300;
  m00 += reg;
  m11 += reg;
  const double det = m00 * m11 - m01 * m01;
  const double x0 = m11 * r0 - m01 * r1, x1 = m00 * r1 - m01 * r0;
  if (!(det > 0) || !isfinite(x0 / det) || !isfinite(x1 / det)) return v3(0, 0, 0);
  return (x0 * u1 + x1 * u2) / det;
}

// 一个分离方向 r（归一化）、固定（rotating = false）或转动的平面：所有顶点对都满足的最长的 tau。
__device__ double plane_step(const V3 P[4], const V3 D[4], bool pt, V3 r, double level, bool rotating) {
  const double rn = norm(r);
  if (!(rn > 0)) return 0.0;
  const V3 n = r / rn;
  const V3 nu = rotating ? plane_rate(P, D, pt, n) : v3(0, 0, 0);
  const double half_nu2 = 0.5 * norm2(nu);
  double tau = DBL_MAX * 2;
  const int na = pt ? 1 : 2;
  for (int j = 0; j < na; ++j) {
    for (int i = na; i < 4; ++i) {
      const V3 m = P[j] - P[i], s = D[j] - D[i];
      const double A = dot(n, m) - level;
      if (!(A > 0)) return 0.0;
      tau = fmin(tau, first_positive_root(A, dot(n, s) + dot(nu, m), dot(nu, s) - level * half_nu2));
    }
  }
  return tau;
}

// 两个分离方向（最近点的方向、图元的法向），固定的平面和转动的平面，取走得最远的那个。
__device__ double plane_steps(const V3 P[4], const V3 D[4], bool pt, V3 r, double level) {
  double best = plane_step(P, D, pt, r, level, false);
  best = fmax(best, plane_step(P, D, pt, r, level, true));
  const V3 e1 = pt ? P[2] - P[1] : P[1] - P[0];
  const V3 e2 = pt ? P[3] - P[1] : P[3] - P[2];
  const V3 N = cross(e1, e2);
  const double side = dot(N, r);
  if (!(norm(N) > 1e-6 * norm(e1) * norm(e2)) || side == 0) return best;
  const V3 n = side > 0 ? N : -N;
  best = fmax(best, plane_step(P, D, pt, n, level, false));
  best = fmax(best, plane_step(P, D, pt, n, level, true));
  return best;
}

// 一对的 ACCD 状态（accd_begin 先走 split 步以后）。r 是 t 处的最近点之差（分离平面的界用）。
struct AccdState {
  double t = 0, d = 0, d0 = 0, ell = 0, stop = 0;
  int it = 0;
  V3 r = V3{0, 0, 0};
};

// ACCD 的循环：从 S 的 (t, d, r)、第 it 步接着走，最多走到第 it_end 步（CPU 版一共 kAccdMax 步）。
// 返回 0：这一对返回 1（走到了 1，或者别的对已经截得更早）；1：停下了，结果是 t；2：走到第 it_end 步还没完。
// 分两次走（先 split 步，再走完）和一次走完，每一步的算式一样，结果逐位相同。
// 每步再用分离平面的界，目标不低于停下的门槛，离门槛 10% 以内就停（同 CPU 版 accd_pair）。
// __noinline__：两处调用（accd_begin 里、尾巴证不出来以后接着走）用同一份机器码。内联成两份时编译器可能把
//   乘加合并得不一样，交不交给 warp 的结果就差在末位（2026-11-14，Ccd.TailMatchesSerial 差 6e-14）。
__device__ __noinline__ int accd_steps(const double* Xa, const double* Xb, const int ids[4], bool pt, double radii, double xi,
                          double c, const double* tmin, AccdState& S, int it_end) {
  V3 D[4];
  for (int i = 0; i < 4; ++i) D[i] = v3load(Xb + 3 * ids[i]) - v3load(Xa + 3 * ids[i]);
  for (; S.it < it_end; ++S.it) {
    V3 P[4];
    for (int i = 0; i < 4; ++i) P[i] = (1 - S.t) * v3load(Xa + 3 * ids[i]) + S.t * v3load(Xb + 3 * ids[i]);
    const double level = xi + fmax(S.stop, (1 - c) * (S.d - xi));
    const double step = fmax((S.d - level) / S.ell, plane_steps(P, D, pt, S.r, radii + level));
    const double tn = S.t + step;
    if (tn >= 1.0) return 0;
    // t 只增不减：别的对已经找到的截断位置 *tmin 不比 tn 大时，这一对最后的结果也不会更小，
    // 不影响所有对的最小值，可以提前停（只是少算，结果和 CPU 版一样）。
    if (tmin && tn >= *static_cast<const volatile double*>(tmin)) return 0;
    S.t = tn;
    S.r = pair_offset_at(Xa, Xb, ids, pt, S.t, P);
    S.d = norm(S.r) - radii;
    if (S.d - xi <= 1.1 * S.stop) return 1;
  }
  return 2;
}

// accd_pair 的前半段：起点的距离、矢高太大时要求缩短段长、ell，然后串行走 split 步。
// 返回 0：这一对返回 1；1：截断，结果是 S.t（停下了，或者走满 kAccdMax 步，和 CPU 版一样）；2：还没走完。
__device__ int accd_begin(const double* Xa, const double* Xb, const int ids[4], bool pt, double radii, double xi,
                          double A_pair, double c, double* ds_need, const double* tmin, int split, AccdState& S) {
  S.d0 = pair_distance_at(Xa, Xb, ids, pt, radii, 0.0);
  if (S.d0 <= 0) {  // radii 里含最小分离：起点已经 <= sep，卡住（同 accd_pair）。原来返回 0 = 这一对放行
    S.t = 0;
    return 1;
  }
  if (xi > 0.5 * S.d0) {
    atomic_min(ds_need, 0.9 * sqrt(4 * S.d0 / A_pair));
    return 0;
  }
  V3 mvec = v3(0, 0, 0);
  for (int i = 0; i < 4; ++i) mvec = mvec + (v3load(Xb + 3 * ids[i]) - v3load(Xa + 3 * ids[i])) * 0.25;
  double la = 0, lb = 0;
  for (int i = 0; i < 4; ++i) {
    const double l = norm(v3load(Xb + 3 * ids[i]) - v3load(Xa + 3 * ids[i]) - mvec);
    const bool in_a = pt ? i == 0 : i <= 1;
    if (in_a) la = fmax(la, l);
    else lb = fmax(lb, l);
  }
  S.ell = la + lb;
  if (S.ell <= 0) return 0;
  S.t = 0;
  S.d = S.d0;
  S.it = 0;
  S.stop = (1 - c) * (S.d0 - xi);
  V3 P[4];
  S.r = pair_offset_at(Xa, Xb, ids, pt, 0.0, P);
  const int st = accd_steps(Xa, Xb, ids, pt, radii, xi, c, tmin, S, split);
  return st == 2 && S.it >= kAccdMax ? 1 : st;
}

// 一段弦 [ta, tb] 上，一对图元的距离（减去半径）的下界（第 14 节）。
// 对任何单位向量 u，u 方向上把两个图元分开的距离
//   m_u(t) = min_{i in A} u . x_i(t) - max_{j in B} u . x_j(t)
// 都不超过真实距离（A 是点或第一条边，B 是三角形或第二条边）。顶点沿弦线性运动，m_u 是凹的分段线性函数，
// 最小值在两端，所以 min(m_u(ta), m_u(tb)) 是 [ta, tb] 上距离的下界。试两个方向，取大的：
//   1. ta 处最近点的连线方向：m_u(ta) 就是 ta 处的距离；
//   2. 三角形的法向（点–三角形）或两条边的公垂线方向（边–边）：贴着滑过边缘时，最近点的连线会转到侧面，
//      而侧向的滑动很快把方向 1 的下界拉到 0 以下，这个方向却还隔着原来的间隙。
// 算不出来（距离为 0、非有限数）时返回 -1。
__device__ inline double separation(const V3 X[4], int na, V3 u) {
  double lo = DBL_MAX, hi = -DBL_MAX;
  for (int i = 0; i < 4; ++i) {
    const double v = dot(u, X[i]);
    if (i < na) lo = fmin(lo, v);
    else hi = fmax(hi, v);
  }
  return lo - hi;
}

__device__ double chord_lower_bound(const double* Xa, const double* Xb, const int ids[4], bool pt, double radii,
                                    double ta, double tb) {
  V3 P[4], Q[4];
  for (int i = 0; i < 4; ++i) {
    const V3 a = v3load(Xa + 3 * ids[i]), b = v3load(Xb + 3 * ids[i]);
    P[i] = (1 - ta) * a + ta * b;
    Q[i] = (1 - tb) * a + tb * b;
  }
  V3 w, n;
  if (pt) {
    const V3 lam = point_triangle_closest(P[0], P[1], P[2], P[3]);
    w = P[0] - (lam.x * P[1] + lam.y * P[2] + lam.z * P[3]);
    n = cross(P[2] - P[1], P[3] - P[1]);
  } else {
    double s, u;
    segment_segment_closest(P[0], P[1], P[2], P[3], s, u);
    w = ((1 - s) * P[0] + s * P[1]) - ((1 - u) * P[2] + u * P[3]);
    n = cross(P[1] - P[0], P[3] - P[2]);
  }
  const double m0 = norm(w);
  if (!(m0 > 0) || !isfinite(m0)) return -1;
  const int na = pt ? 1 : 2;
  const V3 u = (1.0 / m0) * w;
  double best = fmin(m0, separation(Q, na, u));
  const double nn = norm(n);
  if (nn > 0 && isfinite(nn)) {
    const V3 v = (dot(n, w) >= 0 ? 1.0 / nn : -1.0 / nn) * n;  // 指向 A 的那一侧
    best = fmax(best, fmin(separation(P, na, v), separation(Q, na, v)));
  }
  if (!(best == best)) return -1;
  return best - radii;
}

__device__ inline bool overlaps(V3 alo, V3 ahi, V3 blo, V3 bhi, double margin) {
  return alo.x <= bhi.x + margin && alo.y <= bhi.y + margin && alo.z <= bhi.z + margin && blo.x <= ahi.x + margin &&
         blo.y <= ahi.y + margin && blo.z <= ahi.z + margin;
}

__global__ void k_body_disp(DModel m, const double* Xa, const double* Xb, double* D) {
  // 每个刚体的顶点在这一段里的最大位移 |Xb - Xa|（一个线程一个顶点，原子 max；D 事先清零）。
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= m.nvert) return;
  atomic_max(D + m.vbody[v], norm(v3load(Xb + 3 * v) - v3load(Xa + 3 * v)));
}

__global__ void k_ccd_init(double* tmin, double* ds_need, double* seen, DScalar lenp) {
  const double len = lenp.get();
  // 一段开始时的初值（一个线程）：截断位置 1、需要的段长 len、候选、刚体对、工作项的个数 0。用内核写，不用同步的拷贝。
  *tmin = 1.0;
  *ds_need = len;
  seen[0] = seen[1] = seen[2] = 0;
}

// 顶点 v 在 [Xa, Xb] 上的扫掠盒（按半径放大），和 CPU 版 curved_ccd 的一样。
__device__ inline void vbox(const DModel& m, const double* Xa, const double* Xb, int v, V3& lo, V3& hi) {
  const double r = m.vradius[v];
  const V3 a = v3load(Xa + 3 * v), b = v3load(Xb + 3 * v), rr = v3(r, r, r);
  lo = vmin(a, b) - rr;
  hi = vmax(a, b) + rr;
}
// 几个顶点的扫掠盒的并。
__device__ inline void prim_box(const DModel& m, const double* Xa, const double* Xb, const int* ids, int n, V3& lo,
                                V3& hi) {
  vbox(m, Xa, Xb, ids[0], lo, hi);
  for (int k = 1; k < n; ++k) {
    V3 l2, h2;
    vbox(m, Xa, Xb, ids[k], l2, h2);
    lo = vmin(lo, l2);
    hi = vmax(hi, h2);
  }
}

// 候选拆成四个顶点（点–三角形：点、三角形的三个顶点；边–边：两条边的端点）和两个刚体。返回是不是边–边。
__device__ inline bool decode_cand(const DModel& m, DCand cd, int ids[4], int& ba, int& bb) {
  if (cd.b >= 0) {
    ids[0] = cd.a;
    ids[1] = m.fv[3 * cd.b];
    ids[2] = m.fv[3 * cd.b + 1];
    ids[3] = m.fv[3 * cd.b + 2];
    ba = m.vbody[cd.a];
    bb = m.fbody[cd.b];
    return false;
  }
  const int e2 = ~cd.b;
  ids[0] = m.ev[2 * cd.a];
  ids[1] = m.ev[2 * cd.a + 1];
  ids[2] = m.ev[2 * e2];
  ids[3] = m.ev[2 * e2 + 1];
  ba = m.ebody[cd.a];
  bb = m.ebody[e2];
  return true;
}

// 单精度预筛（笔记第 14.2 节）。xa、xb 是这一段两端顶点的单精度世界坐标（launch_to_float）。
// ACCD 第一步就返回 1、也不要求缩短段长的条件是 d0 >= max(2 xi, xi + ell / c)。用单精度算 d0 和 ell：
// 距离少算一个余量、ell 多算四个，条件仍然成立时，双精度的 ACCD 一定也是这个结果，这一对什么也不改，
// 盒子规则、距离都不用再算。余量和 narrow.cu 的一样，1e-5 (1 + R) 米，R 是这几个顶点坐标的最大绝对值，
// 比单精度的舍入（约 1e-7 R）大两个数量级以上。形状太差的三角形 pt_dist_f 返回 0，不会被跳过。
constexpr float kFloatPad = 1e-5f;

__device__ bool float_trivial(const float* xa, const float* xb, const int ids[4], bool pt, double radii, double xi,
                              double c) {
  F3 a[4], d[4];
  float R = 0;
  for (int i = 0; i < 4; ++i) {
    a[i] = f3(xa + 3 * ids[i]);
    const F3 b = f3(xb + 3 * ids[i]);
    d[i] = b - a[i];
    R = fmaxf(R, fmaxf(fmaxf(fabsf(a[i].x), fabsf(a[i].y)), fabsf(a[i].z)));
    R = fmaxf(R, fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fabsf(b.z)));
  }
  const float df = pt ? pt_dist_f(a[0], a[1], a[2], a[3]) : ee_dist_f(a[0], a[1], a[2], a[3]);
  const F3 mv = 0.25f * (d[0] + d[1] + d[2] + d[3]);
  float la = 0, lb = 0;
  for (int i = 0; i < 4; ++i) {
    const F3 e = d[i] - mv;
    const float l = sqrtf(fdot(e, e));
    if (i < (pt ? 1 : 2)) la = fmaxf(la, l);
    else lb = fmaxf(lb, l);
  }
  const double pad = kFloatPad * (1.0 + R);
  const double d0 = static_cast<double>(df) - radii - pad;
  const double ell = static_cast<double>(la + lb) + 4 * pad;
  return d0 >= 1.000001 * fmax(2 * xi, xi + ell / c);  // NaN 时为假
}

// CPU 版的盒子规则：点–三角形是顶点对三角形所在的刚体、对三角形；边–边是第一条边对第二条边的刚体、对第二条边。
__device__ inline bool box_rule(const DModel& m, const double* Xa, const double* Xb, const double* boxes,
                                const int ids[4], bool ee, int bb, double xi) {
  V3 lo1, hi1, lo2, hi2;
  prim_box(m, Xa, Xb, ids, ee ? 2 : 1, lo1, hi1);
  if (!overlaps(lo1, hi1, v3load(boxes + 6 * bb), v3load(boxes + 6 * bb + 3), xi)) return false;
  prim_box(m, Xa, Xb, ids + (ee ? 2 : 1), ee ? 2 : 3, lo2, hi2);
  return overlaps(lo1, hi1, lo2, hi2, xi);
}

// 尾巴的证明（第 14 节）：整个 warp 一起算 [t, tend] 分成 kTailPieces 段的距离下界 L，判断这一对一定返回 1。
// 参数整个 warp 一样。tend = min(1, tmin)：之后的步一定因为 tn >= tmin 或 tn >= 1 停下、返回 1，不用管。
// L - xi 比停下的门槛 stop 大（留一点余量盖住舍入）时，每一步都停不下来，而且每步至少前进 c (L - xi) / ell；
// 照这个步长在第 kAccdMax 步之前走得到 tend，这一对就一定返回 1。
// 分离平面的界：停下的门槛是 1.1 stop；每步的 ell 那一部分是 (d - level) / ell，
//   level = xi + max(stop, (1 - c)(d - xi))，所以至少前进 min(gap - stop, c gap) / ell（gap = L - xi <= d - xi），
//   平面的界只会走得更远。
constexpr int kTailPieces = 128;

__device__ bool warp_proves_one(const double* Xa, const double* Xb, const int ids[4], bool pt, double radii, double xi,
                                double c, const AccdState& S, double tend, int lane) {
  if (S.t >= tend) return true;  // 下一步的 tn > t >= tmin
  double L = DBL_MAX;
  for (int q = lane; q < kTailPieces; q += 32) {
    const double ta = S.t + (tend - S.t) * q / kTailPieces;
    const double tb = q + 1 == kTailPieces ? tend : S.t + (tend - S.t) * (q + 1) / kTailPieces;
    L = fmin(L, chord_lower_bound(Xa, Xb, ids, pt, radii, ta, tb));
  }
  for (int o = 16; o > 0; o >>= 1) L = fmin(L, __shfl_xor_sync(0xffffffffu, L, o));
  const double gap = L - xi;
  const double pad = 1e-12 + 1e-9 * (fabs(S.d0) + xi);
  const double adv = fmin(gap - S.stop, c * gap);
  return gap - 1.1 * S.stop > pad && adv > 0 && S.it + (tend - S.t) * S.ell / adv + 2 < kAccdMax;
}

__global__ void k_ccd_exact(DModel m, const double* Xa, const double* Xb, const double* boxes, const double* A,
                            DScalar lenp, double c, double sep, DCands C, const int* npairs_d, const int* work_n,
                            const float* xfa,
                            const float* xfb, DTail tail, double* tmin, double* ds_need, double* seen) {
  const double len = lenp.get();
  // 一个线程一个候选（narrow.cu 剔下来的）。剔除只保证“段起点的距离不小于阈值 T 的对一定被剔掉”，所以这里先按
  // CPU 版 curved_ccd 的规则用双精度复核（box_rule），通过的才做 ACCD。做 ACCD 的对是 CPU 版的一个子集，
  // 少掉的都是 ACCD 什么也不改的（笔记第 4 节）。xi、A_pair 由两个图元所在的刚体算。
  // xfa 不为空时，先用单精度预筛（float_trivial，笔记第 14.2 节）：第一步一定返回 1 的对直接跳过。
  // ACCD 分三步（笔记第 14 节）：
  //   1. 每个线程串行走 tail.split 步，绝大多数对在这里就有结果；
  //   2. 这个 warp 里还没走完的对，整个 warp 一起证明它返回 1（warp_proves_one），一次一对；
  //   3. 证不出来的，各自接着串行走完。
  // 每一步的算式和一次走完一样，所以结果逐位相同；tail.split >= kAccdMax 时第 2、3 步什么也不做。
  // seen[0..2]：候选、刚体对、工作项的总数（可能超过容量），主机和 tmin、ds_need 一起读回来，超过时扩容重做这一段。
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    seen[0] = static_cast<double>(*C.count);
    seen[1] = static_cast<double>(*npairs_d);
    seen[2] = static_cast<double>(*work_n);
  }
  const int n = cand_count(C);
  const int lane = threadIdx.x & 31;
  const int split = min(tail.split, kAccdMax);
  // 网格跨步的循环按 warp 对齐，整个 warp 走同样多圈（第 2 步要整个 warp 一起）。
  for (int base = blockIdx.x * blockDim.x + (threadIdx.x & ~31); base < n; base += gridDim.x * blockDim.x) {
    const int k = base + lane;
    int ids[4] = {0, 0, 0, 0}, ba = 0, bb = 0;
    bool ee = false, pending = false, skip = false;
    double xi = 0, radii = 0;
    AccdState S;
    if (k < n) {
      ee = decode_cand(m, C.item[k], ids, ba, bb);
      const double A_pair = A[ba] + A[bb];
      xi = A_pair * len * len / 8;
      radii = m.vradius[ids[0]] + m.vradius[ids[ee ? 2 : 1]] + sep;  // 距离都是 d - sep（同 curved_ccd）
      skip = xfa && float_trivial(xfa, xfb, ids, !ee, radii, xi, c);
      if (!skip && box_rule(m, Xa, Xb, boxes, ids, ee, bb, xi + sep)) {  // 盒子的余量也加 sep
        const int st = accd_begin(Xa, Xb, ids, !ee, radii, xi, A_pair, c, ds_need, tmin, split, S);
        if (st == 1 && S.t < 1.0) atomic_min(tmin, S.t);
        pending = st == 2;
      }
    }
    const unsigned sk = __ballot_sync(0xffffffffu, skip);
    if (sk && lane == 0 && tail.stats) atomicAdd(tail.stats + 3, static_cast<unsigned long long>(__popc(sk)));
    unsigned todo = __ballot_sync(0xffffffffu, pending);
    if (todo && lane == 0 && tail.stats) atomicAdd(tail.stats, static_cast<unsigned long long>(__popc(todo)));
    while (todo) {
      const int j = __ffs(todo) - 1;
      todo &= todo - 1;
      int jids[4];
      for (int i = 0; i < 4; ++i) jids[i] = __shfl_sync(0xffffffffu, ids[i], j);
      const bool jee = __shfl_sync(0xffffffffu, static_cast<int>(ee), j) != 0;
      AccdState J;
      J.t = __shfl_sync(0xffffffffu, S.t, j);
      J.d0 = __shfl_sync(0xffffffffu, S.d0, j);
      J.ell = __shfl_sync(0xffffffffu, S.ell, j);
      J.stop = __shfl_sync(0xffffffffu, S.stop, j);
      J.it = __shfl_sync(0xffffffffu, S.it, j);
      const double jxi = __shfl_sync(0xffffffffu, xi, j), jradii = __shfl_sync(0xffffffffu, radii, j);
      double tm = 0;
      if (lane == j) tm = *static_cast<const volatile double*>(tmin);
      const double tend = fmin(1.0, __shfl_sync(0xffffffffu, tm, j));  // 整个 warp 用同一个值
      const bool ok = warp_proves_one(Xa, Xb, jids, !jee, jradii, jxi, c, J, tend, lane);
      if (ok && lane == j) pending = false;
      if (ok && lane == 0 && tail.stats) atomicAdd(tail.stats + 1, 1ull);
    }
    if (pending) {
      if (tail.stats) atomicAdd(tail.stats + 2, 1ull);
      const int st = accd_steps(Xa, Xb, ids, !ee, radii, xi, c, tmin, S, kAccdMax);
      if (st != 0 && S.t < 1.0) atomic_min(tmin, S.t);
    }
  }
}

}  // namespace

void launch_accel_bounds(const DModel& m, const double* q, const DFrames& F, const double* p, bool tight, double* A) {
  k_accel_bounds<<<blocks(m.nbody), kThreads>>>(m, q, F, p, tight ? 1 : 0, A);
}

void launch_ground_ccd(const DModel& m, const double* Xa, const double* Xb, const double* A, DScalar len, double c,
                       double sep, double* tmin, double* ds_need) {
  if (m.nplane > 0 && m.nvert > 0) k_ground_ccd<<<blocks(m.nvert), kThreads>>>(m, Xa, Xb, A, len, c, sep, tmin, ds_need);
}

void launch_body_disp(const DModel& m, const double* Xa, const double* Xb, double* D) {
  cudaMemsetAsync(D, 0, sizeof(double) * m.nbody);
  if (m.nvert > 0) k_body_disp<<<blocks(m.nvert), kThreads>>>(m, Xa, Xb, D);
}

void launch_ccd_init(double* tmin, double* ds_need, double* seen, DScalar len) {
  k_ccd_init<<<1, 1>>>(tmin, ds_need, seen, len);
}

void launch_ccd_cull(const DModel& m, const DFrames& Fa, const double* boxes, const double* D, const double* A,
                     DScalar len, double c, double sep, const int* pairs, const int* npairs_d, int pair_cap, DNarrow N,
                     DCands C, bool window) {
  launch_narrow(m, Fa, boxes, pairs, npairs_d, pair_cap, sep, A, D, len, c, N, C, window);  // CCD 时 dmax 的位置传 sep
}

void launch_ccd_exact(const DModel& m, const double* Xa, const double* Xb, const double* boxes, const double* A,
                      DScalar len, double c, double sep, const int* npairs_d, int pair_cap, DNarrow N, DCands C,
                      DTail tail, double* tmin, double* ds_need, double* seen) {
  if (pair_cap > 0)
    k_ccd_exact<<<std::min(blocks(C.cap), 1024), kThreads>>>(m, Xa, Xb, boxes, A, len, c, sep, C, npairs_d, N.work_n, N.xa,
                                                             N.xb, tail,
                                                           tmin, ds_need, seen);
}

}  // namespace rcipc::gpu
