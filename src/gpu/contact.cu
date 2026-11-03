// GPU 版的接触（阶段 11，笔记 notes/11_gpu.md 第 3 节）。对应 CPU 版的 contact.cc、friction.cc。
//
// 检测分三层，和 CPU 版的 for_each_candidate 规则一样（所以找到的接触一样）：
//   1. 刚体对：一个线程一对刚体，包围盒按 dmax 放大后重叠的写进候选列表；
//   2. 剔除（narrow.cu）：在刚体坐标系里用事先分好组的盒子和单精度距离剔，剩下的写进候选数组；
//   3. 图元对：一个线程一个候选，双精度算最近点，d < dmax 的原子地追加到接触列表（k_detect_exact）。
// 单精度的盒子向外取整、距离带余量，只会多留不会漏，最后的距离用双精度算。
// 每个接触的能量、梯度、行、曲率上界由一个线程算，往全局数组里原子加。
#include <algorithm>
#include <cfloat>

#include "devmath.cuh"
#include "kernels.h"
#include "rows.cuh"

namespace rcipc::gpu {

namespace {

constexpr int kThreads = 128;
inline int blocks(int n) { return (n + kThreads - 1) / kThreads; }

// ---------- 包围盒和刚体对 ----------

__global__ void k_body_boxes(DModel m, const double* X, const double* Xb, double* boxes) {
  // 每个线程一个刚体：它的顶点（按半径放大）的包围盒；Xb 不为空时是两个位置的并（扫掠盒）。
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= m.nbody) return;
  V3 lo = v3(DBL_MAX, DBL_MAX, DBL_MAX), hi = v3(-DBL_MAX, -DBL_MAX, -DBL_MAX);
  for (int v = m.bvert0[b]; v < m.bvert0[b] + m.bnvert[b]; ++v) {
    const V3 r = v3(m.vradius[v], m.vradius[v], m.vradius[v]);
    const V3 x = v3load(X + 3 * v);
    lo = vmin(lo, x - r);
    hi = vmax(hi, x + r);
    if (Xb) {
      const V3 y = v3load(Xb + 3 * v);
      lo = vmin(lo, y - r);
      hi = vmax(hi, y + r);
    }
  }
  v3store(lo, boxes + 6 * b);
  v3store(hi, boxes + 6 * b + 3);
}

__device__ inline bool overlaps(V3 alo, V3 ahi, V3 blo, V3 bhi, double margin) {
  return alo.x <= bhi.x + margin && alo.y <= bhi.y + margin && alo.z <= bhi.z + margin && blo.x <= ahi.x + margin &&
         blo.y <= ahi.y + margin && blo.z <= ahi.z + margin;
}

__device__ inline bool collide(const DModel& m, int a, int b) {
  // Model::collide。
  if (a == b) return false;
  if (m.filter_parent && m.parent[a] == b && b != 0) return false;
  if (m.filter_parent && m.parent[b] == a && a != 0) return false;
  return true;
}

__global__ void k_body_pairs(DModel m, const double* boxes, double margin, const double* A, double len, int* pairs,
                             int cap, int* count) {
  // 一个线程一对 (a, b)，a < b。A 不为空时（CCD）每个刚体的余量再加上它的矢高 A_b len^2 / 8。
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<long long>(m.nbody) * m.nbody) return;
  const int a = static_cast<int>(idx / m.nbody), b = static_cast<int>(idx % m.nbody);
  if (b <= a || !collide(m, a, b) || m.bnvert[a] == 0 || m.bnvert[b] == 0) return;
  const double mg = margin + (A ? A[a] * len * len / 8 + A[b] * len * len / 8 : 0.0);
  if (!overlaps(v3load(boxes + 6 * a), v3load(boxes + 6 * a + 3), v3load(boxes + 6 * b), v3load(boxes + 6 * b + 3), mg))
    return;
  const int k = atomicAdd(count, 1);
  if (k < cap) {
    pairs[2 * k] = a;
    pairs[2 * k + 1] = b;
  }
}

// ---------- 窄相 ----------

__device__ inline void append(DContact* out, int cap, int* count, const DContact& c) {
  const int k = atomicAdd(count, 1);
  if (k < cap) out[k] = c;
}

__global__ void k_ground_detect(DModel m, const double* X, double dmax, DContact* out, int cap, int* count) {
  // 顶点–平面（for_each_candidate 的第一段）。世界上的静态顶点不和平面比。
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= m.nvert || m.vbody[v] == 0) return;
  const V3 x = v3load(X + 3 * v);
  for (int p = 0; p < m.nplane; ++p) {
    const V3 n = v3load(m.pn + 3 * p);
    const double d = dot(n, x) - m.poff[p] - m.vradius[v];
    if (d >= dmax) continue;
    DContact c{};
    c.type = kGround;
    c.nst = 1;
    c.v[0] = v;
    c.w[0] = 1;
    v3store(n, c.n);
    c.d = d;
    c.bodyA = m.vbody[v];
    c.bodyB = 0;
    c.plane = p;
    c.moll = 1;
    c.dmoll = 0;
    append(out, cap, count, c);
  }
}

// make_pt：顶点 v 对三角形 f。
__device__ bool make_pt(const DModel& m, const double* X, int v, int f, double dmax, DContact& c) {
  const int f0 = m.fv[3 * f], f1 = m.fv[3 * f + 1], f2 = m.fv[3 * f + 2];
  const V3 p = v3load(X + 3 * v), a = v3load(X + 3 * f0), b = v3load(X + 3 * f1), cc = v3load(X + 3 * f2);
  const V3 lam = point_triangle_closest(p, a, b, cc);
  const V3 delta = p - (lam.x * a + lam.y * b + lam.z * cc);
  const double len = norm(delta);
  const double d = len - m.vradius[v] - m.vradius[f0];
  if (d >= dmax || len <= 0) return false;
  c.type = kPointTriangle;
  c.nst = 4;
  c.v[0] = v;
  c.v[1] = f0;
  c.v[2] = f1;
  c.v[3] = f2;
  c.w[0] = 1;
  c.w[1] = -lam.x;
  c.w[2] = -lam.y;
  c.w[3] = -lam.z;
  v3store(delta / len, c.n);
  c.d = d;
  c.bodyA = m.vbody[v];
  c.bodyB = m.fbody[f];
  c.plane = -1;
  c.moll = 1;
  c.dmoll = 0;
  return true;
}

// make_ee：边 e1 对边 e2（带 mollifier）。
__device__ bool make_ee(const DModel& m, const double* X, int e1, int e2, double dmax, DContact& c) {
  const int p0 = m.ev[2 * e1], p1 = m.ev[2 * e1 + 1], q0 = m.ev[2 * e2], q1 = m.ev[2 * e2 + 1];
  const V3 P0 = v3load(X + 3 * p0), P1 = v3load(X + 3 * p1), Q0 = v3load(X + 3 * q0), Q1 = v3load(X + 3 * q1);
  double s, t;
  segment_segment_closest(P0, P1, Q0, Q1, s, t);
  const V3 delta = ((1 - s) * P0 + s * P1) - ((1 - t) * Q0 + t * Q1);
  const double len = norm(delta);
  const double d = len - m.vradius[p0] - m.vradius[q0];
  if (d >= dmax || len <= 0) return false;
  c.type = kEdgeEdge;
  c.nst = 4;
  c.v[0] = p0;
  c.v[1] = p1;
  c.v[2] = q0;
  c.v[3] = q1;
  c.w[0] = 1 - s;
  c.w[1] = s;
  c.w[2] = -(1 - t);
  c.w[3] = -t;
  v3store(delta / len, c.n);
  c.d = d;
  c.bodyA = m.ebody[e1];
  c.bodyB = m.ebody[e2];
  c.plane = -1;
  const V3 ea = P1 - P0, eb = Q1 - Q0;
  const double cx = norm2(cross(ea, eb));
  const double eps = 1e-3 * norm2(ea) * norm2(eb);
  if (eps > 0 && cx < eps) {
    c.moll = (2 - cx / eps) * (cx / eps);
    c.dmoll = 2 * (1 - cx / eps) / eps;
  } else {
    c.moll = 1;
    c.dmoll = 0;
  }
  return true;
}


__global__ void k_detect_exact(DModel m, const double* X, DCands C, double dmax, DContact* out, int cap, int* count) {
  // 一个线程一个候选：双精度的最近点，d < dmax 的追加到接触列表（make_pt / make_ee）。
  const int n = cand_count(C);
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
    const DCand cd = C.item[k];
    DContact c;
    const bool hit = cd.b < 0 ? make_ee(m, X, cd.a, ~cd.b, dmax, c) : make_pt(m, X, cd.a, cd.b, dmax, c);
    if (hit) append(out, cap, count, c);
  }
}

// ---------- 链上的 Jacobian ----------

// 一侧（同一个刚体上的若干模板顶点）合并后的行：P = sum w_v X_v，W = sum w_v（书第 171 页）。
// 对方向 dir，第 d 个自由度的值：转动 dir . (a_d x (P - W o_d))，平移 W (dir . a_d)。
// 写到 idx/val 的第 slot 个槽起，返回写了几个。
__device__ int side_entries(const DModel& m, const DFrames& F, int body, V3 P, double W, V3 dir, double scale, int* idx,
                            double* val) {
  if (body == 0) return 0;
  const int n = m.chain_n[body];
  for (int k = 0; k < n; ++k) {
    const int d = m.chain_dof[body * m.CH + k];
    const V3 a = v3load(F.daxis + 3 * d);
    idx[k] = d;
    val[k] = scale * (m.drot[d] ? dot(dir, cross(a, P - W * v3load(F.danchor + 3 * d))) : W * dot(dir, a));
  }
  return n;
}

// 模板的两侧：地面只有 A 侧（顶点 0）；点–三角形 A 侧是顶点 0、B 侧是 1–3；边–边 A 侧是 0–1、B 侧是 2–3。
__device__ inline int side_a_count(const DContact& c) { return c.type == kEdgeEdge ? 2 : 1; }

__device__ void side_sums(const DContact& c, const double* X, int first, int last, V3& P, double& W) {
  P = v3(0, 0, 0);
  W = 0;
  for (int i = first; i < last; ++i) {
    P = P + c.w[i] * v3load(X + 3 * c.v[i]);
    W += c.w[i];
  }
}

// chain_joints：刚体到根的链上的关节（根在前），带静态力臂。返回个数。
struct CJ {
  int dadr, nv, rot;
  double reach;
};
__device__ int chain_joints(const DModel& m, const double* q, int body, CJ* out) {
  int n = 0;
  double path = m.radius[body];
  for (int k = body; k != 0 && n < kMaxDepth; k = m.parent[k]) {
    const int j = m.bjnt[k];
    if (j >= 0) {
      const int type = m.jtype[j];
      CJ cj;
      cj.rot = type != kSlide;
      cj.dadr = type == kFree ? m.jdadr[j] + 3 : m.jdadr[j];
      cj.nv = (type == kSlide || type == kHinge) ? 1 : 3;
      if (type == kSlide) path += fabs(q[m.jqadr[j]] - m.jref[j]);
      cj.reach = path + norm(v3load(m.jpos + 3 * j));
      out[n++] = cj;
    }
    path += norm(v3load(m.bpos + 3 * k));
  }
  for (int i = 0; i < n / 2; ++i) {
    const CJ tmp = out[i];
    out[i] = out[n - 1 - i];
    out[n - 1 - i] = tmp;
  }
  return n;
}

__global__ void k_contact_terms(DModel m, DFrames F, const double* q, const DContact* cs, int nc, double kappa,
                                double dhat, double curv_scale, double* e, double* g, double* Hk, DRows rows) {
  // 一个线程一个接触：contact_energy、contact_gradient、contact_rows、contact_curvature 里属于它的那一份。
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nc) return;
  const DContact c = cs[i];
  const BV bv = barrier(c.d * c.d, dhat * dhat);
  if (e) atomicAdd(e, kappa * c.moll * bv.b);
  if (!g && !rows.D && !(Hk && curv_scale > 0)) return;  // 线搜索只要能量

  // 接触行 r = 2 d n^T sum_v w_v J(X_v)，按两侧合并。
  int idx[2 * 64];
  double val[2 * 64];
  const int na = side_a_count(c);
  V3 PA, PB;
  double WA, WB;
  side_sums(c, F.X, 0, na, PA, WA);
  side_sums(c, F.X, na, c.nst, PB, WB);
  const V3 n = v3load(c.n);
  int cnt = side_entries(m, F, c.bodyA, PA, WA, n, 2 * c.d, idx, val);
  cnt += side_entries(m, F, c.bodyB, PB, WB, n, 2 * c.d, idx + cnt, val + cnt);

  if (g) {
    for (int k = 0; k < cnt; ++k) atomicAdd(g + idx[k], kappa * c.moll * bv.db * val[k]);
    if (c.dmoll != 0) {
      // mollifier 的导数（mollifier_row）：c_x = |u x v|^2 对四个顶点的导数，逐个顶点经 Jacobian 投影。
      const V3 X0 = v3load(F.X + 3 * c.v[0]), X1 = v3load(F.X + 3 * c.v[1]);
      const V3 X2 = v3load(F.X + 3 * c.v[2]), X3 = v3load(F.X + 3 * c.v[3]);
      const V3 u = X1 - X0, v = X3 - X2, w = cross(u, v);
      const V3 du = 2 * cross(v, w), dv = 2 * cross(w, u);
      const V3 dX[4] = {-du, du, -dv, dv};
      const V3 Xs[4] = {X0, X1, X2, X3};
      for (int s = 0; s < 4; ++s) {
        const int body = m.vbody[c.v[s]];
        if (body == 0) continue;
        for (int k = 0; k < m.chain_n[body]; ++k) {
          const int d = m.chain_dof[body * m.CH + k];
          const V3 a = v3load(F.daxis + 3 * d);
          const double jv = m.drot[d] ? dot(dX[s], cross(a, Xs[s] - v3load(F.danchor + 3 * d))) : dot(dX[s], a);
          atomicAdd(g + d, kappa * bv.b * c.dmoll * jv);
        }
      }
    }
  }
  if (rows.D) {
    // 行按力旋量存（rows.cuh）：两侧各是 (刚体, f = 2d W n, tau = 2d P x n)。
    rows.body[2 * i] = c.bodyA;
    rows.body[2 * i + 1] = c.bodyB;
    side_wrench(PA, WA, n, 2 * c.d, rows.w + 12 * static_cast<size_t>(i));
    side_wrench(PB, WB, n, 2 * c.d, rows.w + 12 * static_cast<size_t>(i) + 6);
    rows.D[i] = kappa * c.moll * barrier_row_weight(c.d, dhat);
  }
  if (Hk && curv_scale > 0) {
    // contact_curvature：f C_j 加到两侧链上每个转动组的对角。
    const double f = 2 * kappa * c.moll * fabs(bv.db) * c.d;
    for (int side = 0; side < 2; ++side) {
      const int body = side == 0 ? c.bodyA : c.bodyB;
      if (body == 0) continue;
      CJ chain[kMaxDepth];
      const int nj = chain_joints(m, q, body, chain);
      for (int j = 0; j < nj; ++j) {
        if (!chain[j].rot) continue;
        double C = 0;
        for (int k = 0; k < nj; ++k) {
          if (chain[k].rot) C += chain[k <= j ? j : k].reach;
        }
        for (int d = chain[j].dadr; d < chain[j].dadr + chain[j].nv; ++d) {
          const int t = m.dtree[d], l = d - m.tdof0[t];
          atomicAdd(Hk + t * m.TD * m.TD + l * m.TD + l, curv_scale * f * C);
        }
      }
    }
  }
}

__global__ void k_rows_expand(DModel m, DFrames F, DRows rows, int nrow, int* idx, double* val) {
  // 测试用：把每行展开成逐个自由度的值，每行 2 CH 个槽，用不到的 idx = -1。
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrow) return;
  const int RW = 2 * m.CH;
  const int n = row_expand(m, F, rows, r, idx + static_cast<size_t>(r) * RW, val + static_cast<size_t>(r) * RW);
  for (int k = n; k < RW; ++k) {
    idx[static_cast<size_t>(r) * RW + k] = -1;
    val[static_cast<size_t>(r) * RW + k] = 0;
  }
}

__global__ void k_min_distance(const DContact* cs, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomic_min(out, cs[i].d);
}

// ---------- 摩擦 ----------

__device__ inline double f0(double y, double eps) { return y >= eps ? y : -y * y * y / (3 * eps * eps) + y * y / eps + eps / 3; }
__device__ inline double f1_over_y(double y, double eps) { return y >= eps ? 1.0 / y : 2 / eps - y / (eps * eps); }

__global__ void k_friction_lag(DModel m, const double* X0, const DContact* cs, int nc, double kappa, double dhat,
                               DFriction* out, int* count) {
  // lag_friction：步首的法向力、切向基、模板顶点的位置。
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nc) return;
  const DContact c = cs[i];
  const double lambda = -kappa * c.moll * barrier(c.d * c.d, dhat * dhat).db * 2 * c.d;
  if (lambda <= 0) return;
  DFriction f;
  f.c = c;
  f.lambda = lambda;
  const V3 n = v3load(c.n);
  const V3 helper = fabs(n.x) > 0.9 ? v3(0, 1, 0) : v3(1, 0, 0);
  V3 t1 = cross(n, helper);
  t1 = t1 / norm(t1);
  v3store(t1, f.t1);
  v3store(cross(n, t1), f.t2);
  for (int s = 0; s < c.nst; ++s) v3store(v3load(X0 + 3 * c.v[s]), f.X0 + 3 * s);
  out[atomicAdd(count, 1)] = f;
}

__global__ void k_friction_terms(DModel m, DFrames F, const DFriction* fs, int nf, double h, double mu, double eps_v,
                                 double* e, double* g, DRows rows, int row0) {
  // friction_energy、friction_gradient、friction_rows 里属于这个摩擦接触的那一份。
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nf) return;
  const DFriction f = fs[i];
  const DContact& c = f.c;
  V3 V = v3(0, 0, 0);
  for (int s = 0; s < c.nst; ++s) V = V + c.w[s] * (v3load(F.X + 3 * c.v[s]) - v3load(f.X0 + 3 * s)) / h;
  const V3 t1 = v3load(f.t1), t2 = v3load(f.t2);
  const double u0 = dot(t1, V), u1 = dot(t2, V);
  const double y = sqrt(u0 * u0 + u1 * u1);
  if (e) atomicAdd(e, h * mu * f.lambda * f0(y, eps_v));
  if (!g && !rows.D) return;  // 线搜索只要能量

  // 两条切向行 rho_1、rho_2（接触行里把 n 换成 t_j，不乘 2d）。
  const int na = side_a_count(c);
  V3 PA, PB;
  double WA, WB;
  side_sums(c, F.X, 0, na, PA, WA);
  side_sums(c, F.X, na, c.nst, PB, WB);
  const double coef = mu * f.lambda * f1_over_y(y, eps_v);
  if (g) {
    // 梯度 coef (u0 rho_1 + u1 rho_2)：就是方向 u0 t1 + u1 t2 的那一行乘 coef。
    int idx[2 * 64];
    double val[2 * 64];
    const V3 dir = u0 * t1 + u1 * t2;
    int cnt = side_entries(m, F, c.bodyA, PA, WA, dir, coef, idx, val);
    cnt += side_entries(m, F, c.bodyB, PB, WB, dir, coef, idx + cnt, val + cnt);
    for (int k = 0; k < cnt; ++k) atomicAdd(g + idx[k], val[k]);
  }
  if (rows.D) {
    // 切平面里的 2 x 2 刚度 K = (f1/|u|)(1 - u^ u^T) + f1' u^ u^T 的两个特征方向：u^（特征值 f1'）和它的垂直方向（f1/|u|）。
    const double radial = f1_over_y(y, eps_v);
    const double along = y < eps_v ? 2 / eps_v - 2 * y / (eps_v * eps_v) : 0.0;
    double ev[2][2], k2[2];
    if (y > 1e-12) {
      ev[0][0] = u0 / y;
      ev[0][1] = u1 / y;
      ev[1][0] = -u1 / y;
      ev[1][1] = u0 / y;
      k2[0] = along;
      k2[1] = radial;
    } else {
      ev[0][0] = 1;
      ev[0][1] = 0;
      ev[1][0] = 0;
      ev[1][1] = 1;
      k2[0] = k2[1] = radial;
    }
    for (int r = 0; r < 2; ++r) {
      // 第 r 个特征方向的行：方向 ev[r][0] t1 + ev[r][1] t2，按力旋量存。
      const int row = row0 + 2 * i + r;
      const V3 dir = ev[r][0] * t1 + ev[r][1] * t2;
      rows.body[2 * row] = c.bodyA;
      rows.body[2 * row + 1] = c.bodyB;
      side_wrench(PA, WA, dir, 1.0, rows.w + 12 * static_cast<size_t>(row));
      side_wrench(PB, WB, dir, 1.0, rows.w + 12 * static_cast<size_t>(row) + 6);
      rows.D[row] = mu * f.lambda * fmax(k2[r], 0.0) / h;
    }
  }
}

}  // namespace

void launch_body_boxes(const DModel& m, const double* X, double* boxes) {
  k_body_boxes<<<blocks(m.nbody), kThreads>>>(m, X, nullptr, boxes);
}

void launch_swept_boxes(const DModel& m, const double* Xa, const double* Xb, double* boxes) {
  k_body_boxes<<<blocks(m.nbody), kThreads>>>(m, Xa, Xb, boxes);
}

void launch_body_pairs(const DModel& m, const double* boxes, double margin, const double* A, double len, int* pairs,
                       int cap, int* d_count) {
  cudaMemsetAsync(d_count, 0, sizeof(int));
  const long long total = static_cast<long long>(m.nbody) * m.nbody;
  k_body_pairs<<<static_cast<int>((total + kThreads - 1) / kThreads), kThreads>>>(m, boxes, margin, A, len, pairs, cap,
                                                                                    d_count);
}

void launch_detect(const DModel& m, const DFrames& F, const double* boxes, const int* pairs, int pair_cap,
                   double dmax, DNarrow N, DCands C, DContact* out, int cap, int* d_cnt, int h_cnt[4]) {
  int* d_contacts = d_cnt + 1;
  cudaMemsetAsync(d_contacts, 0, sizeof(int));
  if (m.nplane > 0 && m.nvert > 0) k_ground_detect<<<blocks(m.nvert), kThreads>>>(m, F.X, dmax, out, cap, d_contacts);
  launch_narrow(m, F, boxes, pairs, d_cnt, pair_cap, dmax, nullptr, nullptr, 0.0, 0.0, N, C);
  if (pair_cap > 0) k_detect_exact<<<std::min(blocks(C.cap), 1024), kThreads>>>(m, F.X, C, dmax, out, cap, d_contacts);
  cudaMemcpy(h_cnt, d_cnt, 4 * sizeof(int), cudaMemcpyDeviceToHost);  // 刚体对、接触、候选、工作项的个数，一次读回
}

void launch_contact_terms(const DModel& m, const DFrames& F, const double* q, const DContact* c, int n, double kappa,
                          double dhat, double curv_scale, double* e, double* g, double* Hk, DRows rows) {
  if (n > 0) k_contact_terms<<<blocks(n), kThreads>>>(m, F, q, c, n, kappa, dhat, curv_scale, e, g, Hk, rows);
}

void launch_rows_expand(const DModel& m, const DFrames& F, DRows rows, int nrow, int* idx, double* val) {
  if (nrow > 0) k_rows_expand<<<blocks(nrow), kThreads>>>(m, F, rows, nrow, idx, val);
}

void launch_min_distance(const DContact* c, int n, double* out) {
  if (n > 0) k_min_distance<<<blocks(n), kThreads>>>(c, n, out);
}

int launch_friction_lag(const DModel& m, const double* X0, const DContact* c, int n, double kappa, double dhat,
                        DFriction* out, int* d_count) {
  cudaMemsetAsync(d_count, 0, sizeof(int));
  if (n > 0) k_friction_lag<<<blocks(n), kThreads>>>(m, X0, c, n, kappa, dhat, out, d_count);
  int k = 0;
  cudaMemcpy(&k, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  return k;
}

void launch_friction_terms(const DModel& m, const DFrames& F, const DFriction* f, int nf, double h, double mu,
                           double eps_v, double* e, double* g, DRows rows, int row0) {
  if (nf > 0) k_friction_terms<<<blocks(nf), kThreads>>>(m, F, f, nf, h, mu, eps_v, e, g, rows, row0);
}

}  // namespace rcipc::gpu
