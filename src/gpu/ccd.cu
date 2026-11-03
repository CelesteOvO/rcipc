// GPU 版的曲线 CCD（阶段 11，笔记 notes/11_gpu.md 第 4 节）。对应 CPU 版 ccd.cc 的 acceleration_bounds 和 curved_ccd 的一段。
//
// 一段 [s0, s0 + len] 的检查分两部分，并行方式和接触检测一样：
//   - 顶点–平面：一个线程一个顶点，闭式解；
//   - 刚体之间：先用扫掠盒找候选刚体对（launch_body_pairs，余量是两侧的矢高），
//     再剔除（k_pair_ccd，一个 warp 一对刚体，narrow.cuh），剩下的候选图元对一个线程一对做 ACCD（k_ccd_exact）。
// 截断位置和需要的更短段长都用原子 min 汇总。分段、缩短重做的循环在主机上（stepper.cc），和 CPU 版一样。
#include <algorithm>
#include <cfloat>

#include "devmath.cuh"
#include "kernels.h"
#include "narrow.cuh"

namespace rcipc::gpu {

namespace {

constexpr int kThreads = 128;
inline int blocks(int n) { return (n + kThreads - 1) / kThreads; }

__global__ void k_accel_bounds(DModel m, const double* q, const double* p, double* A) {
  // acceleration_bounds：每个线程一个刚体。链上的关节（根在前）、静态力臂，然后套公式。
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= m.nbody) return;
  if (b == 0) {
    A[0] = 0;
    return;
  }
  // 第 1 步：沿链往上收集（和 chain_joints 一样，力臂里含滑动关节现在的伸长），再反过来变成根在前。
  double w[kMaxDepth], u[kMaxDepth], R[kMaxDepth];
  int rot[kMaxDepth];
  int n = 0;
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
  A[b] = a;
}

__global__ void k_ground_ccd(DModel m, const double* Xa, const double* Xb, const double* A, double len, double c,
                             double* tmin, double* ds_need) {
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

__global__ void k_pair_ccd(DModel m, const double* boxes, PrimBufs P, const double* D, const double* A, double len,
                           double c, const int* pairs, const int* npairs_d, int pair_cap, DCands C) {
  // 剔除（一个 warp 一对刚体，narrow.cuh）。CPU 版 curved_ccd 的刚体之间部分的规则：
  //   点–三角形：顶点的扫掠盒和对方刚体的扫掠盒、和三角形的扫掠盒都按 xi 重叠，才做 ACCD；
  //   边–边：第一条边的扫掠盒和对方刚体、和第二条边都按 xi 重叠，才做 ACCD。
  // 在这之前先用单精度剔两遍，剔掉的都是做了 ACCD 也没有任何作用的对：
  //   1. 单精度的扫掠盒离对方刚体超过 xi 的图元（只会多留）；
  //   2. 段起点的距离 d0 足够大的图元对。accd_pair 在 d0 >= 2 xi 并且 c (d0 - xi) >= ell 时第一步就返回 1，
  //      什么也不改（不截断、不要求缩短段长）。ell = la + lb，la、lb 都不超过 2 Dmax（Dmax 是两个刚体的顶点
  //      在这一段里的最大位移，k_body_disp），所以 d0 >= T = max(2 xi, xi + 4 Dmax / c) 就可以跳过（T 再放大 1%，
  //      盖住双精度的舍入）。d0 用单精度算，减去余量。
  // 通过的再用双精度的盒子（P.dbox）把 CPU 版的条件原样查一遍，写进候选数组，由 k_ccd_exact 做 ACCD。
  // 所以做 ACCD 的组合是 CPU 版的一个子集，少掉的都是 ACCD 什么也不改的，结果和 CPU 版完全一样。
  const int warp = threadIdx.x >> 5;
  const int pi = blockIdx.x * kPairWarps + warp;
  __shared__ WarpLists lists[kPairWarps];
  if (pi >= min(*npairs_d, pair_cap)) return;  // 刚体对的个数在显存里，按容量发线程块
  const int body[2] = {pairs[2 * pi], pairs[2 * pi + 1]};
  const double xi = (A[body[0]] + A[body[1]]) * len * len / 8;
  const double T = 1.01 * fmax(2 * xi, xi + 4 * fmax(D[body[0]], D[body[1]]) / c);
  const float pad = pair_pad(boxes, body);
  bool full[6];
  int len6[6];
  warp_fill(m, P, boxes, body, xi, lists[warp], full, len6);
  warp_scan(m, P, lists[warp], full, body, len6, C, [&](int L1, int id1, int id2, float df) {
    const bool pt = L1 != 4;
    if (df - pad >= __double2float_ru(T + pair_radii(m, id1, id2, !pt))) return false;
    // 双精度复核：第一个图元（顶点或 a 的边）和对方刚体、和第二个图元。
    const int other = body[1 - L1 % 2];
    const double* b1 = P.dbox + 6 * static_cast<size_t>(prim_slot(m, pt ? 0 : 1, id1));
    const double* b2 = P.dbox + 6 * static_cast<size_t>(prim_slot(m, pt ? 2 : 1, id2));
    const V3 lo1 = v3load(b1), hi1 = v3load(b1 + 3);
    if (!overlaps(lo1, hi1, v3load(boxes + 6 * other), v3load(boxes + 6 * other + 3), xi)) return false;
    return overlaps(lo1, hi1, v3load(b2), v3load(b2 + 3), xi);
  });
}

__global__ void k_ccd_init(double* tmin, double* ds_need, double* seen, double len) {
  // 一段开始时的初值（一个线程）：截断位置 1、需要的段长 len、候选和刚体对的个数 0。用内核写，不用同步的拷贝。
  *tmin = 1.0;
  *ds_need = len;
  seen[0] = seen[1] = 0;
}

__global__ void k_ccd_exact(DModel m, const double* Xa, const double* Xb, const double* A, double len, double c,
                            DCands C, const int* npairs_d, double* tmin, double* ds_need, double* seen) {
  // 一个线程一个候选：ACCD（accd_pair）。xi、A_pair 由两个图元所在的刚体算，和 k_pair_ccd 里的一样。
  // seen[0]、seen[1]：候选总数、刚体对总数（可能超过容量），主机和 tmin、ds_need 一起读回来，超过时扩容重做这一段。
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    seen[0] = static_cast<double>(*C.count);
    seen[1] = static_cast<double>(*npairs_d);
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
    const double radii = m.vradius[ids[0]] + m.vradius[ids[ee ? 2 : 1]];
    const double t = accd_pair(Xa, Xb, ids, !ee, radii, xi, A_pair, c, ds_need, tmin);
    if (t < 1.0) atomic_min(tmin, t);
  }
}

}  // namespace

void launch_accel_bounds(const DModel& m, const double* q, const double* p, double* A) {
  k_accel_bounds<<<blocks(m.nbody), kThreads>>>(m, q, p, A);
}

void launch_ground_ccd(const DModel& m, const double* Xa, const double* Xb, const double* A, double len, double c,
                       double* tmin, double* ds_need) {
  if (m.nplane > 0 && m.nvert > 0) k_ground_ccd<<<blocks(m.nvert), kThreads>>>(m, Xa, Xb, A, len, c, tmin, ds_need);
}

void launch_body_disp(const DModel& m, const double* Xa, const double* Xb, double* D) {
  cudaMemsetAsync(D, 0, sizeof(double) * m.nbody);
  if (m.nvert > 0) k_body_disp<<<blocks(m.nvert), kThreads>>>(m, Xa, Xb, D);
}

void launch_ccd_init(double* tmin, double* ds_need, double* seen, double len) {
  k_ccd_init<<<1, 1>>>(tmin, ds_need, seen, len);
}

void launch_pair_ccd(const DModel& m, const double* Xa, const double* Xb, const double* boxes, const PrimBufs& P,
                     const double* D, const double* A, double len, double c, const int* pairs, const int* npairs_d,
                     int pair_cap, DCands C, double* tmin, double* ds_need, double* seen) {
  cudaMemsetAsync(C.count, 0, sizeof(int));
  if (pair_cap > 0) {
    k_pair_ccd<<<(pair_cap + kPairWarps - 1) / kPairWarps, 32 * kPairWarps>>>(m, boxes, P, D, A, len, c, pairs,
                                                                             npairs_d, pair_cap, C);
    k_ccd_exact<<<std::min(blocks(C.cap), 1024), kThreads>>>(m, Xa, Xb, A, len, c, C, npairs_d, tmin, ds_need, seen);
  }
}

}  // namespace rcipc::gpu
