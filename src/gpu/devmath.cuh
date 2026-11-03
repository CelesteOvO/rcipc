#pragma once

// GPU 版的小数学库（阶段 11，笔记 notes/11_gpu.md 第 1 节）。
//
// 设备代码里不用 Eigen，所以这里自己写三维向量、四元数、3x3 矩阵。每个运算都照抄 Eigen 的写法（同样的运算顺序），
// 这样 GPU 和 CPU 的结果只差舍入，测试可以逐项对比。也照抄了 CPU 版的障碍函数和两个最近点函数。
#include <cuda_runtime.h>

#include <cmath>

namespace rcipc::gpu {

#define RC_HD __host__ __device__ inline

struct V3 {
  double x, y, z;
};
RC_HD V3 v3(double x, double y, double z) { return V3{x, y, z}; }
RC_HD V3 v3load(const double* p) { return V3{p[0], p[1], p[2]}; }
RC_HD void v3store(V3 a, double* p) {
  p[0] = a.x;
  p[1] = a.y;
  p[2] = a.z;
}
RC_HD V3 operator+(V3 a, V3 b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
RC_HD V3 operator-(V3 a, V3 b) { return V3{a.x - b.x, a.y - b.y, a.z - b.z}; }
RC_HD V3 operator-(V3 a) { return V3{-a.x, -a.y, -a.z}; }
RC_HD V3 operator*(double s, V3 a) { return V3{s * a.x, s * a.y, s * a.z}; }
RC_HD V3 operator*(V3 a, double s) { return V3{s * a.x, s * a.y, s * a.z}; }
RC_HD V3 operator/(V3 a, double s) { return V3{a.x / s, a.y / s, a.z / s}; }
RC_HD double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
RC_HD V3 cross(V3 a, V3 b) { return V3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
RC_HD double norm2(V3 a) { return dot(a, a); }
RC_HD double norm(V3 a) { return sqrt(dot(a, a)); }
RC_HD V3 vmin(V3 a, V3 b) { return V3{fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z)}; }
RC_HD V3 vmax(V3 a, V3 b) { return V3{fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z)}; }

// 四元数，(w, x, y, z)。
struct Q4 {
  double w, x, y, z;
};
RC_HD Q4 q4load(const double* p) { return Q4{p[0], p[1], p[2], p[3]}; }
RC_HD void q4store(Q4 q, double* p) {
  p[0] = q.w;
  p[1] = q.x;
  p[2] = q.y;
  p[3] = q.z;
}
RC_HD V3 qvec(Q4 q) { return V3{q.x, q.y, q.z}; }
// Hamilton 乘积，同 Eigen 的 operator*。
RC_HD Q4 qmul(Q4 a, Q4 b) {
  const V3 av = qvec(a), bv = qvec(b);
  const V3 v = a.w * bv + b.w * av + cross(av, bv);
  return Q4{a.w * b.w - dot(av, bv), v.x, v.y, v.z};
}
RC_HD Q4 qconj(Q4 q) { return Q4{q.w, -q.x, -q.y, -q.z}; }
RC_HD Q4 qnormalize(Q4 q) {
  const double n = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  return Q4{q.w / n, q.x / n, q.y / n, q.z / n};
}
// 用四元数转动向量，同 Eigen 的 _transformVector：uv = q.vec x v; uv += uv; v + w uv + q.vec x uv。
RC_HD V3 qrot(Q4 q, V3 v) {
  V3 uv = cross(qvec(q), v);
  uv = uv + uv;
  return v + q.w * uv + cross(qvec(q), uv);
}
// 指数映射，同 quat.cc 的 quat_exp。
RC_HD Q4 qexp(V3 w) {
  const double t = norm(w);
  const double k = t < 1e-4 ? 0.5 - t * t / 48 : sin(0.5 * t) / t;
  return Q4{cos(0.5 * t), k * w.x, k * w.y, k * w.z};
}

// 3x3 矩阵，按行存。
struct M3 {
  double m[9];
};
// 同 Eigen 的 toRotationMatrix。
RC_HD M3 qmat(Q4 q) {
  const double tx = 2 * q.x, ty = 2 * q.y, tz = 2 * q.z;
  const double twx = tx * q.w, twy = ty * q.w, twz = tz * q.w;
  const double txx = tx * q.x, txy = ty * q.x, txz = tz * q.x;
  const double tyy = ty * q.y, tyz = tz * q.y, tzz = tz * q.z;
  return M3{{1 - (tyy + tzz), txy - twz, txz + twy, txy + twz, 1 - (txx + tzz), tyz - twx, txz - twy, tyz + twx,
             1 - (txx + tyy)}};
}
RC_HD V3 mv(const M3& a, V3 v) {
  return V3{a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z, a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
            a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z};
}
RC_HD M3 mm(const M3& a, const M3& b) {
  M3 c;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) c.m[3 * i + j] = a.m[3 * i] * b.m[j] + a.m[3 * i + 1] * b.m[3 + j] + a.m[3 * i + 2] * b.m[6 + j];
  return c;
}
RC_HD M3 mt(const M3& a) { return M3{{a.m[0], a.m[3], a.m[6], a.m[1], a.m[4], a.m[7], a.m[2], a.m[5], a.m[8]}}; }
RC_HD M3 mload(const double* p) {
  M3 a;
  for (int i = 0; i < 9; ++i) a.m[i] = p[i];
  return a;
}

// 障碍函数，同 barrier.cc：返回 b、b'、b''。
struct BV {
  double b, db, ddb;
};
RC_HD BV barrier(double x, double xhat) {
  if (x >= xhat) return BV{0, 0, 0};
  const double r = x - xhat;
  const double L = log(x / xhat);
  return BV{-r * r * L, -2 * r * L - r * r / x, -2 * L - 4 * r / x + r * r / (x * x)};
}
// 障碍行的权重 b'' + b' / (2 d^2)，同 contact.cc 的 barrier_row_weight（阶段 9 笔记第 7 节）。
RC_HD double barrier_row_weight(double d, double dhat) {
  const BV bv = barrier(d * d, dhat * dhat);
  return fmax(bv.ddb + bv.db / (2 * d * d), 0.0);
}

// 点到三角形的最近点（重心坐标），同 distance.cc。
RC_HD V3 point_triangle_closest(V3 p, V3 a, V3 b, V3 c) {
  const V3 ab = b - a, ac = c - a, ap = p - a;
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return V3{1, 0, 0};
  const V3 bp = p - b;
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return V3{0, 1, 0};
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const double v = d1 / (d1 - d3);
    return V3{1 - v, v, 0};
  }
  const V3 cp = p - c;
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return V3{0, 0, 1};
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const double w = d2 / (d2 - d6);
    return V3{1 - w, 0, w};
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return V3{0, 1 - w, w};
  }
  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom, w = vc * denom;
  return V3{1 - v - w, v, w};
}

RC_HD double clamp01(double x) { return fmin(fmax(x, 0.0), 1.0); }

// 线段到线段的最近点参数，同 distance.cc。
RC_HD void segment_segment_closest(V3 p0, V3 p1, V3 q0, V3 q1, double& s, double& t) {
  const V3 d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
  const double a = norm2(d1), e = norm2(d2), f = dot(d2, r);
  const double kEps = 1e-30;
  if (a <= kEps && e <= kEps) {
    s = t = 0;
    return;
  }
  if (a <= kEps) {
    s = 0;
    t = clamp01(f / e);
    return;
  }
  const double c = dot(d1, r);
  if (e <= kEps) {
    t = 0;
    s = clamp01(-c / a);
    return;
  }
  const double b = dot(d1, d2);
  const double denom = a * e - b * b;
  s = denom > kEps ? clamp01((b * f - c * e) / denom) : 0.0;
  const double tnom = b * s + f;
  if (tnom < 0) {
    t = 0;
    s = clamp01(-c / a);
  } else if (tnom > e) {
    t = 1;
    s = clamp01((b - c) / a);
  } else {
    t = tnom / e;
  }
}

// 单精度的包围盒（宽相的快速剔除用）。从双精度转过来时向外取整（lo 向下、hi 向上），重叠测试里的加法向上取整，
// 所以“双精度的盒子重叠”一定推出“单精度的盒子重叠”：单精度测试只会多留，不会漏。通过以后再用双精度精确判断。
struct FBox {
  float lo[3], hi[3];
};
__device__ inline FBox make_fbox(V3 lo, V3 hi) {
  return FBox{{__double2float_rd(lo.x), __double2float_rd(lo.y), __double2float_rd(lo.z)},
              {__double2float_ru(hi.x), __double2float_ru(hi.y), __double2float_ru(hi.z)}};
}
__device__ inline FBox fbox_load(const float* p) { return FBox{{p[0], p[1], p[2]}, {p[3], p[4], p[5]}}; }
__device__ inline void fbox_store(const FBox& b, float* p) {
  for (int i = 0; i < 3; ++i) {
    p[i] = b.lo[i];
    p[3 + i] = b.hi[i];
  }
}
// margin 是已经向上取整的单精度余量。
__device__ inline bool fbox_overlap(const FBox& a, const FBox& b, float margin) {
  for (int i = 0; i < 3; ++i) {
    if (a.lo[i] > __fadd_ru(b.hi[i], margin) || b.lo[i] > __fadd_ru(a.hi[i], margin)) return false;
  }
  return true;
}

// ---------- 单精度的距离（只用来剔除，narrow.cuh） ----------
//
// 和上面两个最近点函数是同一个算法（Ericson 5.1.5、5.1.9），换成单精度：消费级显卡的单精度比双精度快几十倍。
// 结果只当作“大概的距离”：调用者要加一个余量（narrow.cuh 的 kFloatPad），离得比阈值加余量还远才跳过。
struct F3 {
  float x, y, z;
};
__device__ inline F3 f3(const float* p) { return F3{p[0], p[1], p[2]}; }
__device__ inline F3 operator+(F3 a, F3 b) { return F3{a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ inline F3 operator-(F3 a, F3 b) { return F3{a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline F3 operator*(float s, F3 a) { return F3{s * a.x, s * a.y, s * a.z}; }
__device__ inline float fdot(F3 a, F3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline F3 fcross(F3 a, F3 b) { return F3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

// 点到三角形的距离。形状太差的三角形（最小角大约小于 6 度）单精度算不准，返回 0（调用者就不会跳过它）。
__device__ inline float pt_dist_f(F3 p, F3 a, F3 b, F3 c) {
  const F3 ab = b - a, ac = c - a, bc = c - b;
  const F3 n = fcross(ab, ac);
  const float sum = fdot(ab, ab) + fdot(ac, ac) + fdot(bc, bc);
  if (!(fdot(n, n) >= 0.0025f * sum * sum)) return 0.0f;  // |n| >= 0.05 (三条边长的平方和)，NaN 也算差
  const F3 ap = p - a;
  const float d1 = fdot(ab, ap), d2 = fdot(ac, ap);
  F3 q;
  if (d1 <= 0 && d2 <= 0) {
    q = a;
  } else {
    const F3 bp = p - b;
    const float d3 = fdot(ab, bp), d4 = fdot(ac, bp);
    const float vc = d1 * d4 - d3 * d2;
    const F3 cp = p - c;
    const float d5 = fdot(ab, cp), d6 = fdot(ac, cp);
    const float vb = d5 * d2 - d1 * d6;
    const float va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3) q = b;
    else if (vc <= 0 && d1 >= 0 && d3 <= 0) q = a + (d1 / (d1 - d3)) * ab;
    else if (d6 >= 0 && d5 <= d6) q = c;
    else if (vb <= 0 && d2 >= 0 && d6 <= 0) q = a + (d2 / (d2 - d6)) * ac;
    else if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) q = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * bc;
    else {
      const float den = 1.0f / (va + vb + vc);
      q = a + (vb * den) * ab + (vc * den) * ac;
    }
  }
  const F3 d = p - q;
  return sqrtf(fdot(d, d));
}

// 线段到线段的距离。
__device__ inline float ee_dist_f(F3 p0, F3 p1, F3 q0, F3 q1) {
  const F3 d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
  const float a = fdot(d1, d1), e = fdot(d2, d2), f = fdot(d2, r);
  float s, t;
  if (a <= 1e-30f && e <= 1e-30f) {
    s = t = 0;
  } else if (a <= 1e-30f) {
    s = 0;
    t = fminf(fmaxf(f / e, 0.0f), 1.0f);
  } else {
    const float c = fdot(d1, r);
    if (e <= 1e-30f) {
      t = 0;
      s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
    } else {
      const float b = fdot(d1, d2);
      const float den = a * e - b * b;
      s = den > 1e-6f * a * e ? fminf(fmaxf((b * f - c * e) / den, 0.0f), 1.0f) : 0.0f;
      const float tn = b * s + f;
      if (tn < 0) {
        t = 0;
        s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
      } else if (tn > e) {
        t = 1;
        s = fminf(fmaxf((b - c) / a, 0.0f), 1.0f);
      } else {
        t = tn / e;
      }
    }
  }
  const F3 d = (p0 + s * d1) - (q0 + t * d2);
  return sqrtf(fdot(d, d));
}

// 双精度的原子 min / max（CUDA 只有原子加）：比较并交换的循环。
__device__ inline double atomic_min(double* addr, double val) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(static_cast<long long>(assumed)) <= val) break;
    old = atomicCAS(a, assumed, static_cast<unsigned long long>(__double_as_longlong(val)));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
}
__device__ inline double atomic_max(double* addr, double val) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(static_cast<long long>(assumed)) >= val) break;
    old = atomicCAS(a, assumed, static_cast<unsigned long long>(__double_as_longlong(val)));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
}

}  // namespace rcipc::gpu
