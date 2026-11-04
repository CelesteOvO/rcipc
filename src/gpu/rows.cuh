#pragma once

// 力旋量形式的行（阶段 11，笔记 notes/11_gpu.md 第 8 节）。
//
// 一条接触行（或摩擦行）在两侧各是一个“力旋量”：A 侧的模板顶点在刚体 b_A 上，合起来是 P = sum w X、W = sum w，
// 方向 dir、系数 scale 时，这一侧在自由度 d（d 在 b_A 的链上）上的值是
//     转动：scale dir . (a_d x (P - W o_d)) = a_d . (tau - o_d x f)，
//     滑动：scale W (dir . a_d)            = a_d . f，
// 其中 f = scale W dir、tau = scale P x dir（关于世界原点的力和力矩）。所以一侧只要存 (b, f, tau) 六个数，
// 不用存它链上每个自由度的值（布娃娃的手链上有 13 个）。
//
// 这样 PCG 里的 H p 的行部分 sum_r D_r r_r (r_r . p) 可以这样算：
//   1. 每个刚体的速度旋量 (v_b, omega_b)：omega_b = sum_{转动 d} p_d a_d，v_b = sum_{转动 d} p_d (o_d x a_d) + sum_{滑动 d} p_d a_d；
//      这一侧的 r . p 就是 f . v_b + tau . omega_b；
//   2. coef = D (r . p)，把 coef (f, tau) 原子加到刚体上（每行两侧 12 次原子加）；
//   3. 每个刚体的子树之和 S_b = sum_{b' 在 b 的子树里} W_b'，自由度 d 得到 a_d . (T - o_d x F)（转动）或 a_d . F（滑动），
//      (F, T) = S_{d 所在的刚体}（d 在 b' 的链上，当且仅当 b' 在 d 所在刚体的子树里）。
// 原来逐个自由度原子加时，一个布娃娃的所有行都要加到它的 6 个根自由度上，几百次原子加挤在同几个地址上排队；
// 现在每行只加到它自己的两个刚体上。上一个分支的协作 PCG 也是这样做的（wrench scatter）。
#include "devmath.cuh"
#include "device_model.h"

namespace rcipc::gpu {

// 力旋量 (f, tau) 在自由度 d 上的值。
__device__ inline double wrench_on_dof(const DModel& m, const DFrames& F, int d, V3 f, V3 tau) {
  const V3 a = v3load(F.daxis + 3 * d);
  return m.drot[d] ? dot(a, tau - cross(v3load(F.danchor + 3 * d), f)) : dot(a, f);
}
__device__ inline double wrench_on_dof(const DModel& m, const DFrames& F, int d, const double* w) {
  return wrench_on_dof(m, F, d, v3load(w), v3load(w + 3));
}

// 一侧的力旋量：f = scale W dir，tau = scale P x dir，写到 out（6 个）。
__device__ inline void side_wrench(V3 P, double W, V3 dir, double scale, double* out) {
  v3store((scale * W) * dir, out);
  v3store(scale * cross(P, dir), out + 3);
}

// 刚体 b 在广义速度 p 下的速度旋量：out[0..2] = v_b，out[3..5] = omega_b（和力旋量的 f、tau 对应着点乘）。
__device__ inline void body_twist(const DModel& m, const DFrames& F, int b, const double* p, double* out) {
  V3 v = v3(0, 0, 0), w = v3(0, 0, 0);
  for (int k = 0; k < m.chain_n[b]; ++k) {
    const int d = m.chain_dof[b * m.CH + k];
    const V3 a = v3load(F.daxis + 3 * d);
    if (m.drot[d]) {
      w = w + p[d] * a;
      v = v + p[d] * cross(v3load(F.danchor + 3 * d), a);
    } else {
      v = v + p[d] * a;
    }
  }
  v3store(v, out);
  v3store(w, out + 3);
}

__device__ inline double dot6(const double* a, const double* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5];
}

// 一行的 r . p（两侧的速度旋量 tw 已经算好，每个刚体 6 个）。
__device__ inline double row_dot(const DRows& rows, int r, const double* tw) {
  double s = 0;
  for (int side = 0; side < 2; ++side) {
    const int b = rows.body[2 * r + side];
    if (b > 0) s += dot6(rows.w + 12 * static_cast<size_t>(r) + 6 * side, tw + 6 * b);
  }
  return s;
}

// coef (f, tau) 原子加到两侧的刚体上（wacc 每个刚体 6 个）。
__device__ inline void row_scatter(const DRows& rows, int r, double coef, double* wacc) {
  for (int side = 0; side < 2; ++side) {
    const int b = rows.body[2 * r + side];
    if (b <= 0) continue;
    const double* w = rows.w + 12 * static_cast<size_t>(r) + 6 * side;
    for (int k = 0; k < 6; ++k) atomicAdd(wacc + 6 * b + k, coef * w[k]);
  }
}

// 刚体 b 的子树里所有刚体的力旋量之和（DFS 序，子树是 [b, bsub_end[b])）。
__device__ inline void subtree_wrench(const DModel& m, int b, const double* wacc, double* out) {
  double s[6] = {0, 0, 0, 0, 0, 0};
  for (int c = b; c < m.bsub_end[b]; ++c)
    for (int k = 0; k < 6; ++k) s[k] += wacc[6 * c + k];
  for (int k = 0; k < 6; ++k) out[k] = s[k];
}

// 自由度 d 关于参考点 ref 的空间轴 s（6 个）：转动 ((o_d - ref) x a_d, a_d)，滑动 (a_d, 0)。
// 速度旋量是 sum_d p_d s_d（body_twist 是 ref = 0 的情形），力旋量 (f, tau - ref x f) 在 d 上的值是 s . w
// （和 wrench_on_dof 相同）。所以“每个刚体累加一个 6x6，再按子树求和”的写法里，树块的元素都是 s_i^T A s_j
// （惯性 Hessian 和预条件器的行都这样算，笔记第 11 节）。参考点取树根，坐标小，舍入小。
__device__ inline void dof_axis6(const DModel& m, const DFrames& F, int d, V3 ref, double* s) {
  const V3 a = v3load(F.daxis + 3 * d);
  if (m.drot[d]) {
    v3store(cross(v3load(F.danchor + 3 * d) - ref, a), s);
    v3store(a, s + 3);
  } else {
    v3store(a, s);
    v3store(v3(0, 0, 0), s + 3);
  }
}

// s^T A t，A 是 6x6（按行存 36 个）。
__device__ inline double form6(const double* s, const double* A, const double* t) {
  double acc = 0;
  for (int i = 0; i < 6; ++i) {
    if (s[i] == 0) continue;
    double r = 0;
    for (int j = 0; j < 6; ++j) r += A[6 * i + j] * t[j];
    acc += s[i] * r;
  }
  return acc;
}

// 6x6 对称矩阵按上三角存 21 个时 (i, j) 的位置（i <= j）。
__device__ inline int sym21(int i, int j) { return i * 6 - i * (i - 1) / 2 + (j - i); }

// 把一行展开成逐个自由度的值（idx/val，最多 2 CH 个，返回个数）。预条件器和测试用。
__device__ inline int row_expand(const DModel& m, const DFrames& F, const DRows& rows, int r, int* idx, double* val) {
  int n = 0;
  for (int side = 0; side < 2; ++side) {
    const int b = rows.body[2 * r + side];
    if (b <= 0) continue;
    const double* w = rows.w + 12 * static_cast<size_t>(r) + 6 * side;
    for (int k = 0; k < m.chain_n[b]; ++k) {
      const int d = m.chain_dof[b * m.CH + k];
      idx[n] = d;
      val[n] = wrench_on_dof(m, F, d, w);
      ++n;
    }
  }
  return n;
}

}  // namespace rcipc::gpu
