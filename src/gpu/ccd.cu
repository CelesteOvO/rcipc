// GPU 版的曲线 CCD（阶段 11，笔记 notes/11_gpu.md 第 4 节）。对应 CPU 版 ccd.cc 的 acceleration_bounds 和 curved_ccd 的一段。
//
// 一段 [s0, s0 + len] 的检查分两部分，并行方式和接触检测一样：
//   - 顶点–平面：一个线程一个顶点，闭式解；
//   - 刚体之间：先用扫掠盒找候选刚体对（launch_body_pairs，余量是两侧的矢高），
//     再剔除（narrow.cu，在刚体坐标系里），剩下的候选图元对一个线程一对：先按 CPU 版的盒子规则用双精度复核，
//     再做 ACCD（k_ccd_exact）。
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
// 再加整段上三阶导数的界 T。jid 是链上的关节（根在前），th 是它们的转速；链上没有滑动关节。
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
  double S = 0, TB = 0;
  for (int k = 0; k < n; ++k) {
    TB += th[k] * S;
    S += th[k];
  }
  const V3 centre = v3load(F.xpos + 3 * b);
  const double lever = m.radius[b];
  double below = 0, sx = 0, sy = 0, Pth = 0, PthA = 0, T = 0;
  for (int k = n - 1; k >= 0; --k) {
    if (th[k] == 0) continue;
    const int j = jid[k], type = m.jtype[j];
    Pth += th[k];
    const double above = S - Pth;
    PthA += th[k] * above;
    const double Bv = TB - PthA;
    const V3 anchor = type == kFree ? v3load(F.xpos + 3 * m.jbody[j]) : v3load(F.danchor + 3 * m.jdadr[j]);
    const V3 off = centre - anchor;
    const double full = norm(off);
    double dist = full;
    if (type == kHinge) {
      const V3 ax = v3load(F.daxis + 3 * m.jdadr[j]);
      const V3 axn = (1.0 / norm(ax)) * ax;
      dist = norm(off - dot(off, axn) * axn);
    }
    const double cj = th[k] * (dist + lever + below);
    const double rho = full + lever + below;
    const double M = below + cj;
    below += cj;
    sx += th[k] * (2 * above * rho + M);
    sy += th[k] * 2 * rho;
    const double Rge = sx - above * sy;
    T += th[k] * (2 * Bv * rho + 4 * above * above * rho + 4 * above * M + Rge);
  }
  return p2 + T;
}

__global__ void k_accel_bounds(DModel m, const double* q, DFrames F, const double* p, int tight, double* A) {
  // ccd_acceleration_bounds：每个线程一个刚体。静态界（acceleration_bounds）和紧界（tight_acceleration_bounds）
  // 取较小的那个。F 是 q 处的 FK。
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= m.nbody) return;
  if (b == 0) {
    A[0] = 0;
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
                             double* tmin, double* ds_need) {
  const double len = lenp.get();
  // curved_ccd 的顶点–平面部分：弦上距离线性，带矢高余量的闭式解。
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= m.nvert) return;
  const int b = m.vbody[v];
  if (b == 0) return;
  const double xi = A[b] * len * len / 8;
  const V3 xa = v3load(Xa + 3 * v), xb = v3load(Xb + 3 * v);
  for (int pl = 0; pl < m.nplane; ++pl) {
    const V3 n = v3load(m.pn + 3 * pl);
    const double ya = dot(n, xa) - m.poff[pl] - m.vradius[v];
    const double yb = dot(n, xb) - m.poff[pl] - m.vradius[v];
    if (xi > 0.5 * ya) {
      atomic_min(ds_need, 0.9 * sqrt(4 * ya / A[b]));
      continue;
    }
    if (yb >= ya) continue;
    atomic_min(tmin, c * (ya - xi) / (ya - yb));
  }
}

// pair_distance_at：一对图元在弦上 t 处的距离（已减半径）。
__device__ double pair_distance_at(const double* Xa, const double* Xb, const int ids[4], bool pt, double radii,
                                   double t) {
  V3 P[4];
  for (int i = 0; i < 4; ++i) P[i] = (1 - t) * v3load(Xa + 3 * ids[i]) + t * v3load(Xb + 3 * ids[i]);
  if (pt) {
    const V3 lam = point_triangle_closest(P[0], P[1], P[2], P[3]);
    return norm(P[0] - (lam.x * P[1] + lam.y * P[2] + lam.z * P[3])) - radii;
  }
  double s, u;
  segment_segment_closest(P[0], P[1], P[2], P[3], s, u);
  return norm(((1 - s) * P[0] + s * P[1]) - ((1 - u) * P[2] + u * P[3])) - radii;
}

// accd_pair：一对图元的加性 CCD，返回段内的安全比例。
__device__ double accd_pair(const double* Xa, const double* Xb, const int ids[4], bool pt, double radii, double xi,
                            double A_pair, double c, double* ds_need, const double* tmin) {
  const double d0 = pair_distance_at(Xa, Xb, ids, pt, radii, 0.0);
  if (d0 <= 0) return 1.0;
  if (xi > 0.5 * d0) {
    atomic_min(ds_need, 0.9 * sqrt(4 * d0 / A_pair));
    return 1.0;
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
  const double ell = la + lb;
  if (ell <= 0) return 1.0;
  double t = 0, d = d0;
  const double stop = (1 - c) * (d0 - xi);
  for (int it = 0; it < 1000; ++it) {
    const double tn = t + c * (d - xi) / ell;
    if (tn >= 1.0) return 1.0;
    // t 只增不减：别的对已经找到的截断位置 *tmin 不比 tn 大时，这一对最后的结果也不会更小，
    // 不影响所有对的最小值，可以提前停（只是少算，结果和 CPU 版一样）。
    if (tmin && tn >= *static_cast<const volatile double*>(tmin)) return 1.0;
    t = tn;
    d = pair_distance_at(Xa, Xb, ids, pt, radii, t);
    if (d - xi <= stop) break;
  }
  return t;
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

__global__ void k_ccd_exact(DModel m, const double* Xa, const double* Xb, const double* boxes, const double* A,
                            DScalar lenp, double c, DCands C, const int* npairs_d, const int* work_n, double* tmin,
                            double* ds_need, double* seen) {
  const double len = lenp.get();
  // 一个线程一个候选（narrow.cu 剔下来的）。剔除只保证“段起点的距离不小于阈值 T 的对一定被剔掉”，所以这里先按
  // CPU 版 curved_ccd 的规则用双精度复核（第一个图元的扫掠盒和对方刚体、和第二个图元都按 xi 重叠），
  // 通过的才做 ACCD（accd_pair）。做 ACCD 的对是 CPU 版的一个子集，少掉的都是 ACCD 什么也不改的（笔记第 4 节）。
  // xi、A_pair 由两个图元所在的刚体算。
  // seen[0..2]：候选、刚体对、工作项的总数（可能超过容量），主机和 tmin、ds_need 一起读回来，超过时扩容重做这一段。
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    seen[0] = static_cast<double>(*C.count);
    seen[1] = static_cast<double>(*npairs_d);
    seen[2] = static_cast<double>(*work_n);
  }
  const int n = cand_count(C);
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
    const DCand cd = C.item[k];
    const bool ee = cd.b < 0;
    int ids[4], ba, bb;
    if (!ee) {
      ids[0] = cd.a;
      ids[1] = m.fv[3 * cd.b];
      ids[2] = m.fv[3 * cd.b + 1];
      ids[3] = m.fv[3 * cd.b + 2];
      ba = m.vbody[cd.a];
      bb = m.fbody[cd.b];
    } else {
      const int e2 = ~cd.b;
      ids[0] = m.ev[2 * cd.a];
      ids[1] = m.ev[2 * cd.a + 1];
      ids[2] = m.ev[2 * e2];
      ids[3] = m.ev[2 * e2 + 1];
      ba = m.ebody[cd.a];
      bb = m.ebody[e2];
    }
    const double A_pair = A[ba] + A[bb];
    const double xi = A_pair * len * len / 8;
    {
      // CPU 版的盒子规则：点–三角形是顶点对三角形所在的刚体、对三角形；边–边是第一条边对第二条边的刚体、对第二条边。
      V3 lo1, hi1, lo2, hi2;
      prim_box(m, Xa, Xb, ids, ee ? 2 : 1, lo1, hi1);
      if (!overlaps(lo1, hi1, v3load(boxes + 6 * bb), v3load(boxes + 6 * bb + 3), xi)) continue;
      prim_box(m, Xa, Xb, ids + (ee ? 2 : 1), ee ? 2 : 3, lo2, hi2);
      if (!overlaps(lo1, hi1, lo2, hi2, xi)) continue;
    }
    const double radii = m.vradius[ids[0]] + m.vradius[ids[ee ? 2 : 1]];
    const double t = accd_pair(Xa, Xb, ids, !ee, radii, xi, A_pair, c, ds_need, tmin);
    if (t < 1.0) atomic_min(tmin, t);
  }
}

}  // namespace

void launch_accel_bounds(const DModel& m, const double* q, const DFrames& F, const double* p, bool tight, double* A) {
  k_accel_bounds<<<blocks(m.nbody), kThreads>>>(m, q, F, p, tight ? 1 : 0, A);
}

void launch_ground_ccd(const DModel& m, const double* Xa, const double* Xb, const double* A, DScalar len, double c,
                       double* tmin, double* ds_need) {
  if (m.nplane > 0 && m.nvert > 0) k_ground_ccd<<<blocks(m.nvert), kThreads>>>(m, Xa, Xb, A, len, c, tmin, ds_need);
}

void launch_body_disp(const DModel& m, const double* Xa, const double* Xb, double* D) {
  cudaMemsetAsync(D, 0, sizeof(double) * m.nbody);
  if (m.nvert > 0) k_body_disp<<<blocks(m.nvert), kThreads>>>(m, Xa, Xb, D);
}

void launch_ccd_init(double* tmin, double* ds_need, double* seen, DScalar len) {
  k_ccd_init<<<1, 1>>>(tmin, ds_need, seen, len);
}

void launch_pair_ccd(const DModel& m, const DFrames& Fa, const double* Xb, const double* boxes, const double* D,
                     const double* A, DScalar len, double c, const int* pairs, const int* npairs_d, int pair_cap,
                     DNarrow N, DCands C, double* tmin, double* ds_need, double* seen) {
  launch_narrow(m, Fa, boxes, pairs, npairs_d, pair_cap, 0.0, A, D, len, c, N, C);
  if (pair_cap > 0)
    k_ccd_exact<<<std::min(blocks(C.cap), 1024), kThreads>>>(m, Fa.X, Xb, boxes, A, len, c, C, npairs_d, N.work_n, tmin,
                                                           ds_need, seen);
}

}  // namespace rcipc::gpu
