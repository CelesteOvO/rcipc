// GPU 版的运动学、惯性项、关节限位（阶段 11，笔记 notes/11_gpu.md 第 2 节）。
//
// 每个内核对应 CPU 版的一个函数（kinematics.cc、inertia.cc、limits.cc），运算顺序照抄，结果只差舍入。
// 并行的方式：
//   - FK：一个线程一个刚体。每个线程从根沿链往下自己算一遍（重复计算祖先，但链很短，省掉了按层同步）；
//   - 顶点、惯性目标、能量、梯度：一个线程一个刚体或顶点，往全局数组里原子加；
//   - 树块 M / h^2：一个线程块一棵树，块里一个线程一个 (i, j) 元素；先在子树上把 6x6 的空间惯量加起来
//     （组合刚体的写法，笔记第 11 节），每个元素只是一次二次型。
#include <cfloat>

#include "devmath.cuh"
#include "kernels.h"
#include "rows.cuh"

namespace rcipc::gpu {

namespace {

constexpr int kThreads = 128;
inline int blocks(int n) { return (n + kThreads - 1) / kThreads; }

__global__ void k_integrate(DModel m, const double* q, const double* p, DScalar sp, double* out) {
  // integrate：每个线程一个关节，只改写这个关节的那几个数。
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m.njnt) return;
  const double s = sp.get();
  const int a = m.jqadr[j], d = m.jdadr[j];
  switch (m.jtype[j]) {
    case kHinge:
    case kSlide:
      out[a] = q[a] + s * p[d];
      break;
    case kBall: {
      const Q4 r = qnormalize(q4load(q + a));
      q4store(qmul(r, qexp(s * v3load(p + d))), out + a);
      break;
    }
    case kFree: {
      for (int k = 0; k < 3; ++k) out[a + k] = q[a + k] + s * p[d + k];
      const Q4 r = qnormalize(q4load(q + a + 3));
      q4store(qmul(r, qexp(s * v3load(p + d + 3))), out + a + 3);
      break;
    }
  }
}

__global__ void k_fk(DModel m, const double* q, DFrames F) {
  // forward_kinematics：每个线程一个刚体 b。先沿父刚体收集到根的链，再从根往下逐个刚体套 CPU 版的第 1–3 步。
  // 只有 b 自己的关节写 dof_axis / dof_anchor（祖先的由祖先自己的线程写）。
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= m.nbody) return;
  if (b == 0) {
    v3store(v3(0, 0, 0), F.xpos);
    q4store(Q4{1, 0, 0, 0}, F.xquat);
    return;
  }
  int chain[kMaxDepth];
  int n = 0;
  for (int k = b; k != 0 && n < kMaxDepth; k = m.parent[k]) chain[n++] = k;
  V3 x = v3(0, 0, 0);
  Q4 r = Q4{1, 0, 0, 0};  // 父刚体的世界位姿（姿态已归一化）
  for (int i = n - 1; i >= 0; --i) {
    const int k = chain[i];
    const bool own = k == b;
    // 第 1 步：固定偏移。
    V3 xk = x + qrot(r, v3load(m.bpos + 3 * k));
    Q4 rk = qmul(r, q4load(m.bquat + 4 * k));
    // 第 2 步：关节。
    const int j = m.bjnt[k];
    if (j >= 0) {
      const int a = m.jqadr[j], d = m.jdadr[j];
      const V3 jpos = v3load(m.jpos + 3 * j), jaxis = v3load(m.jaxis + 3 * j);
      switch (m.jtype[j]) {
        case kHinge: {
          const V3 o = xk + qrot(rk, jpos);
          if (own) {
            v3store(qrot(rk, jaxis), F.daxis + 3 * d);
            v3store(o, F.danchor + 3 * d);
          }
          rk = qmul(rk, qexp((q[a] - m.jref[j]) * jaxis));
          xk = o - qrot(rk, jpos);
          break;
        }
        case kBall: {
          const V3 o = xk + qrot(rk, jpos);
          rk = qmul(rk, qnormalize(q4load(q + a)));
          xk = o - qrot(rk, jpos);
          if (own) {
            for (int c = 0; c < 3; ++c) {
              v3store(qrot(rk, v3(c == 0, c == 1, c == 2)), F.daxis + 3 * (d + c));
              v3store(o, F.danchor + 3 * (d + c));
            }
          }
          break;
        }
        case kSlide: {
          const V3 axis = qrot(rk, jaxis);
          if (own) {
            v3store(axis, F.daxis + 3 * d);
            v3store(xk + qrot(rk, jpos), F.danchor + 3 * d);
          }
          xk = xk + axis * (q[a] - m.jref[j]);
          break;
        }
        case kFree: {
          xk = v3load(q + a);
          rk = qnormalize(q4load(q + a + 3));
          if (own) {
            for (int c = 0; c < 3; ++c) {
              v3store(v3(c == 0, c == 1, c == 2), F.daxis + 3 * (d + c));
              v3store(xk, F.danchor + 3 * (d + c));
              v3store(qrot(rk, v3(c == 0, c == 1, c == 2)), F.daxis + 3 * (d + 3 + c));
              v3store(xk, F.danchor + 3 * (d + 3 + c));
            }
          }
          break;
        }
      }
    }
    // 第 3 步：保存（归一化），作为下一层的父刚体位姿。
    x = xk;
    r = qnormalize(rk);
  }
  v3store(x, F.xpos + 3 * b);
  q4store(r, F.xquat + 4 * b);
}

__global__ void k_vertices(DModel m, DFrames F) {
  // world_point：每个线程一个顶点。
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= m.nvert) return;
  const int b = m.vbody[v];
  v3store(v3load(F.xpos + 3 * b) + qrot(q4load(F.xquat + 4 * b), v3load(m.vlocal + 3 * v)), F.X + 3 * v);
}

__device__ V3 com(const DModel& m, const DFrames& F, int b) {
  return v3load(F.xpos + 3 * b) + qrot(q4load(F.xquat + 4 * b), v3load(m.ipos + 3 * b));
}

__global__ void k_targets(DModel m, DFrames now, DFrames prev, double h, double* tc, double* tA) {
  // inertia_targets：c~ = 2 c^n - c^{n-1} + g h^2，A~ = 2 R^n - R^{n-1}。
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= m.nbody) return;
  if (b == 0) {
    for (int i = 0; i < 3; ++i) tc[i] = 0;
    for (int i = 0; i < 9; ++i) tA[i] = 0;
    return;
  }
  const V3 g = v3(m.gravity[0], m.gravity[1], m.gravity[2]);
  v3store(2 * com(m, now, b) - com(m, prev, b) + g * (h * h), tc + 3 * b);
  const M3 Rn = qmat(q4load(now.xquat + 4 * b)), Rp = qmat(q4load(prev.xquat + 4 * b));
  for (int i = 0; i < 9; ++i) tA[9 * b + i] = 2 * Rn.m[i] - Rp.m[i];
}

__global__ void k_inertia_energy(DModel m, DFrames F, const double* tc, const double* tA, double h, double* e) {
  // inertia_energy：每个刚体 m |c - c~|^2 + tr(D S D^T)，D = R - A~，再除以 2 h^2。
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < 1 || b >= m.nbody) return;
  const V3 dc = com(m, F, b) - v3load(tc + 3 * b);
  double sum = m.mass[b] * norm2(dc);
  const M3 R = qmat(q4load(F.xquat + 4 * b));
  M3 D;
  for (int i = 0; i < 9; ++i) D.m[i] = R.m[i] - tA[9 * b + i];
  const M3 DS = mm(D, mload(m.S + 9 * b));
  for (int i = 0; i < 9; ++i) sum += DS.m[i] * D.m[i];  // tr(D S D^T) = sum_ij (D S)_ij D_ij
  atomicAdd(e, sum / (2 * h * h));
}

__global__ void k_inertia_gradient(DModel m, DFrames F, const double* tc, const double* tA, double h, double* g) {
  // inertia_gradient：刚体上的力 f 和力矩 tau，沿链投影到每个自由度上。
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < 1 || b >= m.nbody) return;
  const V3 c = com(m, F, b);
  const M3 R = qmat(q4load(F.xquat + 4 * b));
  const V3 f = (m.mass[b] * (c - v3load(tc + 3 * b))) / (h * h);
  M3 Dt;  // (R - A~)^T
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) Dt.m[3 * i + k] = R.m[3 * k + i] - tA[9 * b + 3 * k + i];
  const M3 A = mm(mm(R, mload(m.S + 9 * b)), Dt);
  const V3 tau = v3(A.m[5] - A.m[7], A.m[6] - A.m[2], A.m[1] - A.m[3]) / (h * h);  // vex(A) / h^2
  for (int k = 0; k < m.chain_n[b]; ++k) {
    const int d = m.chain_dof[b * m.CH + k];
    const V3 a = v3load(F.daxis + 3 * d);
    const double val = m.drot[d] ? dot(cross(a, c - v3load(F.danchor + 3 * d)), f) + dot(a, tau) : dot(a, f);
    atomicAdd(g + d, val);
  }
}

__global__ void k_inertia_hessian_crb(DModel m, DFrames F, double h, double* Hk) {
  // inertia_hessian 的组合刚体写法（CRBA）：一个线程块一棵树 t，参考点 ref 取树根。
  //   H_ij = 1/h^2 s_i^T Ibar_lo s_j，lo 是 i、j 所在刚体里较深的那个（两者在一条链上时），s 是空间轴（dof_axis6），
  //   Ibar_b = sum_{b' 在 b 的子树里} I_b'，I_b' 是关于 ref 的 6x6 空间惯量（速度旋量 (v, omega) 的动能矩阵）：
  //     [[m 1, -m [c]x], [m [c]x, I_c + m (|c|^2 1 - c c^T)]]，c = 质心 - ref，I_c 是世界系里关于质心的惯量。
  // 和 k_inertia_hessian 是同一个和（m V_i . V_j + a_i . I_c a_j，V = v - [c]x omega 是质心的速度），只是先在子树上
  // 把惯量加起来：每个元素从“子树里每个刚体一次叉乘和 3x3 乘法”变成一次 6x6 的二次型。结果只差舍入。
  extern __shared__ double sh[];
  const int t = blockIdx.x;
  const int TD = m.TD, n = m.tndof[t], d0 = m.tdof0[t], b0 = m.tbody0[t], nb = m.tnbody[t];
  double* I6 = sh;                 // nb x 36
  double* Ibar = I6 + 36 * m.TB;   // nb x 36
  double* s6 = Ibar + 36 * m.TB;   // n x 6
  const V3 ref = v3load(F.xpos + 3 * b0);
  for (int k = threadIdx.x; k < nb; k += blockDim.x) {
    const int b = b0 + k;
    const V3 c = com(m, F, b) - ref;
    const M3 R = qmat(q4load(F.xquat + 4 * b));
    const M3 Iw = mm(mm(R, mload(m.inertia + 9 * b)), mt(R));
    const double ms = m.mass[b], cc = dot(c, c);
    const double cv[3] = {c.x, c.y, c.z};
    // [c]x 的第 (i, j) 个元素
    const double X[9] = {0, -c.z, c.y, c.z, 0, -c.x, -c.y, c.x, 0};
    double* A = I6 + 36 * k;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        A[6 * i + j] = i == j ? ms : 0.0;
        A[6 * i + 3 + j] = -ms * X[3 * i + j];
        A[6 * (3 + i) + j] = ms * X[3 * i + j];
        A[6 * (3 + i) + 3 + j] = Iw.m[3 * i + j] + ms * ((i == j ? cc : 0.0) - cv[i] * cv[j]);
      }
  }
  for (int li = threadIdx.x; li < n; li += blockDim.x) dof_axis6(m, F, d0 + li, ref, s6 + 6 * li);
  __syncthreads();
  for (int e = threadIdx.x; e < nb * 36; e += blockDim.x) {
    const int k = e / 36, a = e % 36, b = b0 + k;
    double s = 0;
    for (int c = b; c < m.bsub_end[b]; ++c) s += I6[36 * (c - b0) + a];
    Ibar[e] = s;
  }
  __syncthreads();
  const double ih2 = 1.0 / (h * h);
  for (int e = threadIdx.x; e < TD * TD; e += blockDim.x) {
    const int li = e / TD, lj = e % TD;
    if (lj < li) continue;  // 下三角由对称的那个线程写
    double acc = 0;
    if (lj < n) {
      const int bi = m.dbody[d0 + li], bj = m.dbody[d0 + lj];
      const int lo = max(bi, bj);
      if (lo < min(m.bsub_end[bi], m.bsub_end[bj])) acc = form6(s6 + 6 * li, Ibar + 36 * (lo - b0), s6 + 6 * lj) * ih2;
    }
    Hk[t * TD * TD + li * TD + lj] = acc;
    Hk[t * TD * TD + lj * TD + li] = acc;
  }
}

__global__ void k_inertia_hessian(DModel m, DFrames F, double h, double* Hk) {
  // inertia_hessian：一个线程块一棵树 t。（树太大、组合刚体版放不进共享内存时用这个。）
  //   H_ij = 1/h^2 sum_b ( m V_i . V_j + W_i . (I_world W_j) )，只对链上同时有 i 和 j 的刚体 b 求和。
  //   自由度 i 在 b 的链上 <=> i 所在的刚体是 b 的祖先或 b 自己 <=> dbody[i] <= b < bsub_end[dbody[i]]。
  // 先把这棵树每个刚体的质心 c_b、世界系的惯量 R I R^T，和每个自由度的轴、锚点放进共享内存（每个只算一次），
  // 再由块里的线程分摊上三角的 (i, j)（H 对称，下三角照抄）。
  extern __shared__ double sh[];
  const int t = blockIdx.x;
  const int TD = m.TD, n = m.tndof[t], d0 = m.tdof0[t], b0 = m.tbody0[t], nb = m.tnbody[t];
  double* sc = sh;                  // nb x 3
  double* sI = sc + 3 * m.TB;       // nb x 9
  double* sa = sI + 9 * m.TB;       // n x 3
  double* so = sa + 3 * TD;         // n x 3
  for (int k = threadIdx.x; k < nb; k += blockDim.x) {
    const int b = b0 + k;
    v3store(com(m, F, b), sc + 3 * k);
    const M3 R = qmat(q4load(F.xquat + 4 * b));
    const M3 Iw = mm(mm(R, mload(m.inertia + 9 * b)), mt(R));
    for (int e = 0; e < 9; ++e) sI[9 * k + e] = Iw.m[e];
  }
  for (int li = threadIdx.x; li < n; li += blockDim.x) {
    for (int c = 0; c < 3; ++c) {
      sa[3 * li + c] = F.daxis[3 * (d0 + li) + c];
      so[3 * li + c] = F.danchor[3 * (d0 + li) + c];
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < TD * TD; e += blockDim.x) {
    const int li = e / TD, lj = e % TD;
    if (lj < li) continue;  // 下三角由对称的那个线程写
    double acc = 0;
    if (lj < n) {
      const int i = d0 + li, j = d0 + lj;
      const int bi = m.dbody[i], bj = m.dbody[j];
      const bool ri = m.drot[i], rj = m.drot[j];
      const V3 ai = v3load(sa + 3 * li), aj = v3load(sa + 3 * lj);
      const V3 oi = v3load(so + 3 * li), oj = v3load(so + 3 * lj);
      // 链上同时有 i 和 j 的刚体：两个子树的交，也就是较深的那个刚体的子树（不在一条链上时为空）。
      const int lo = max(bi, bj), hi = min(m.bsub_end[bi], m.bsub_end[bj]);
      for (int b = lo; b < hi; ++b) {
        const V3 c = v3load(sc + 3 * (b - b0));
        const V3 Vi = ri ? cross(ai, c - oi) : ai;
        const V3 Vj = rj ? cross(aj, c - oj) : aj;
        acc += m.mass[b] * dot(Vi, Vj);
        if (ri && rj) {
          M3 Iw;
          for (int k = 0; k < 9; ++k) Iw.m[k] = sI[9 * (b - b0) + k];
          acc += dot(ai, mv(Iw, aj));
        }
      }
    }
    Hk[t * TD * TD + li * TD + lj] = acc / (h * h);
    Hk[t * TD * TD + lj * TD + li] = acc / (h * h);
  }
}

__device__ inline double rot_rot(V3 a, const double* G, V3 o, V3 f, V3 b) {
  // a^T Q b - (a . b) tr Q，Q = G - o f^T（inertia.cc 的 rot_rot，Q 写成子树之和 G 减去锚点那一项）。
  const double aGb = a.x * (G[0] * b.x + G[1] * b.y + G[2] * b.z) + a.y * (G[3] * b.x + G[4] * b.y + G[5] * b.z) +
                     a.z * (G[6] * b.x + G[7] * b.y + G[8] * b.z);
  const double trQ = G[0] + G[4] + G[8] - dot(o, f);
  return aGb - dot(a, o) * dot(f, b) - dot(a, b) * trQ;
}
// a^T P b - (a . b) tr P（P 按行存 9 个）。
__device__ inline double rot_rot_plain(V3 a, const double* P, V3 b) {
  const double aPb = a.x * (P[0] * b.x + P[1] * b.y + P[2] * b.z) + a.y * (P[3] * b.x + P[4] * b.y + P[5] * b.z) +
                     a.z * (P[6] * b.x + P[7] * b.y + P[8] * b.z);
  return aPb - dot(a, b) * (P[0] + P[4] + P[8]);
}

__global__ void k_inertia_curvature(DModel m, DFrames F, const double* tc, const double* tA, double h, double rot,
                                    double floor, double* Hk, double* K, unsigned long long* dropped, const double* CB,
                                    int inertia) {
  // 阶段 13：inertia_curvature + add_inertia_curvature（inertia.cc）。一个线程块一棵树 t。
  //   1. 每个刚体 b：f_b = m (c - c~) / h^2，G_b = R S (R - A~)^T / h^2 + c_b f_b^T；再对子树求和 Gbar、fbar。
  //      刚体 b 的 Q(o) = G_b - o f_b^T，子树上的和就是 Gbar - o fbar^T：每个元素 O(1)，不用逐个刚体加（同 CRBA）。
  //   2. K 的每个元素（i、j 在一条链上；dn 是刚体较深的那个，求和的范围是 dn 的子树）：
  //        同一个关节、都是转动：1/2 (rot_rot(a_i, Q, a_j) + rot_rot(a_j, Q, a_i))，o 是这个关节的锚点；
  //        不同关节、上游转动：下游转动是 rot_rot(a_up, Q(o_dn), a_dn)，下游平移是 (a_up x a_dn) . fbar；
  //        其余是 0。写到 K（ntree x TD x TD）。
  //   3. B = S (floor Hk + K) S（S = diag(Hk)^-1/2；这时 Hk 里只有 M / h^2）在共享内存里做 LDL^T（同 k_precond_factor_sh），
  //      主元都 > 0 才 Hk += K；否则这棵树不加，*dropped 加一。
  extern __shared__ double sh[];
  const int t = blockIdx.x;
  const int TD = m.TD, n = m.tndof[t], d0 = m.tdof0[t], b0 = m.tbody0[t], nb = m.tnbody[t];
  // 阶段 14：每个刚体多存 9 个——接触的 P = sum (X - xpos) f^T（CB 的后 9 个），它已经在旋量块里，K 里要减掉。
  double* G = sh;               // TB x 21：每个刚体的 G_b（9 个）、f_b（3 个）、P_b（9 个）
  double* Gs = G + 21 * m.TB;   // TB x 21：子树之和
  double* sa = Gs + 21 * m.TB;  // TD x 3：轴
  double* so = sa + 3 * TD;     // TD x 3：锚点
  double* sc = so + 3 * TD;     // TD：缩放
  double* B = sc + TD;          // n x n
  __shared__ int ok;
  double* Kt = K + static_cast<size_t>(t) * TD * TD;
  double* Ht = Hk + static_cast<size_t>(t) * TD * TD;
  const double ih2 = 1.0 / (h * h);
  for (int k = threadIdx.x; k < nb; k += blockDim.x) {
    const int b = b0 + k;
    const V3 c = com(m, F, b);
    const V3 f = (m.mass[b] * (c - v3load(tc + 3 * b))) * ih2;
    const M3 R = qmat(q4load(F.xquat + 4 * b));
    M3 Dt;  // (R - A~)^T
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) Dt.m[3 * i + j] = R.m[3 * j + i] - tA[9 * b + 3 * j + i];
    const M3 P = mm(mm(R, mload(m.S + 9 * b)), Dt);
    const double cv[3] = {c.x, c.y, c.z}, fv[3] = {f.x, f.y, f.z};
    double* g = G + 21 * k;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        g[3 * i + j] = inertia ? rot * P.m[3 * i + j] * ih2 + cv[i] * fv[j] : 0.0;  // rot = 0：不要转动那一部分
    g[9] = inertia ? f.x : 0.0;
    g[10] = inertia ? f.y : 0.0;
    g[11] = inertia ? f.z : 0.0;
    for (int i = 0; i < 9; ++i) g[12 + i] = 0;
    if (CB) {  // 阶段 14：接触的那一份（G = P + xpos F^T 和惯性的 c f^T 同一个形式，所以直接加）
      const double* cb = CB + 21 * static_cast<size_t>(b);
      for (int i = 0; i < 12; ++i) g[i] += cb[i];
      for (int i = 0; i < 9; ++i) g[12 + i] = cb[12 + i];
    }
  }
  for (int li = threadIdx.x; li < n; li += blockDim.x) {
    for (int c = 0; c < 3; ++c) {
      sa[3 * li + c] = F.daxis[3 * (d0 + li) + c];
      so[3 * li + c] = F.danchor[3 * (d0 + li) + c];
    }
    const double dg = Ht[li * TD + li];
    sc[li] = dg > 0 ? 1.0 / sqrt(dg) : 1.0;
  }
  if (threadIdx.x == 0) ok = 1;
  __syncthreads();
  for (int e = threadIdx.x; e < nb * 21; e += blockDim.x) {
    const int k = e / 21, a = e % 21, b = b0 + k;
    double s = 0;
    for (int c = b; c < m.bsub_end[b]; ++c) s += G[21 * (c - b0) + a];
    Gs[e] = s;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {
    const int li = e / n, lj = e % n;
    if (lj < li) continue;  // 上三角算，两边都写
    const int i = d0 + li, j = d0 + lj;
    const int bi = m.dbody[i], bj = m.dbody[j];
    const int lo = max(bi, bj);
    double v = 0;
    if (lo < min(m.bsub_end[bi], m.bsub_end[bj])) {
      const double* gs = Gs + 21 * (lo - b0);
      const V3 fs = v3(gs[9], gs[10], gs[11]);
      if (bi == bj) {  // 同一个关节
        if (m.drot[i] && m.drot[j]) {
          const V3 ai = v3load(sa + 3 * li), aj = v3load(sa + 3 * lj), o = v3load(so + 3 * lj);
          v = 0.5 * (rot_rot(ai, gs, o, fs, aj) + rot_rot(aj, gs, o, fs, ai));
          if (CB) v -= 0.5 * (rot_rot_plain(ai, gs + 12, aj) + rot_rot_plain(aj, gs + 12, ai));
        }
      } else {
        const int up = bi < bj ? li : lj, dn = bi < bj ? lj : li;
        if (m.drot[d0 + up]) {
          const V3 au = v3load(sa + 3 * up), ad = v3load(sa + 3 * dn);
          if (m.drot[d0 + dn]) {
            v = rot_rot(au, gs, v3load(so + 3 * dn), fs, ad);
            // 阶段 14：旋量块里已经有对称化的 rot_rot(a, P, b)（contact_chain_curvature 的第二条）
            if (CB) v -= 0.5 * (rot_rot_plain(au, gs + 12, ad) + rot_rot_plain(ad, gs + 12, au));
          } else {
            v = dot(cross(au, ad), fs);
          }
        }
      }
    }
    Kt[li * TD + lj] = v;
    Kt[lj * TD + li] = v;
    B[lj * n + li] = sc[lj] * (floor * Ht[lj * TD + li] + v) * sc[li];  // 下三角
  }
  __syncthreads();
  for (int j = 0; j < n; ++j) {
    const double d = B[j * n + j];
    if (!(d > 1e-12)) {  // 不正定（或 NaN）：所有线程读到的是同一个主元，一起退出
      if (threadIdx.x == 0) ok = 0;
      break;
    }
    const double rd = 1.0 / d;
    const int w = n - 1 - j;
    for (int e = threadIdx.x; e < w * w; e += blockDim.x) {
      const int i = j + 1 + e / w, k = j + 1 + e % w;
      if (k > i) continue;
      B[i * n + k] -= B[i * n + j] * B[k * n + j] * rd;
    }
    __syncthreads();
  }
  __syncthreads();
  if (!ok) {
    if (threadIdx.x == 0) atomicAdd(dropped, 1ULL);
    return;
  }
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {
    const int li = e / n, lj = e % n;
    Ht[li * TD + lj] += Kt[li * TD + lj];
  }
}

__global__ void k_limits(DModel m, const double* q, double k, double dhat_hinge, double dhat_slide, double* e, double* g,
                         double* Hk) {
  // limits.cc：到两端的距离各一个一维障碍，激活距离按关节类型（LimitBarrier::dhat）。
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m.njnt || !m.jlimited[j]) return;
  const int a = m.jqadr[j], d = m.jdadr[j];
  const double dhat = m.jtype[j] == kSlide ? dhat_slide : dhat_hinge;
  const BV lo = barrier(q[a] - m.jrange[2 * j], dhat), hi = barrier(m.jrange[2 * j + 1] - q[a], dhat);
  if (e) atomicAdd(e, k * (lo.b + hi.b));
  if (g) atomicAdd(g + d, k * (lo.db - hi.db));
  if (Hk) {
    const int t = m.dtree[d], l = d - m.tdof0[t];
    atomicAdd(Hk + t * m.TD * m.TD + l * m.TD + l, k * (lo.ddb + hi.ddb));
  }
}

__global__ void k_drives(DModel m, DDrive D, const double* q, double* e, double* g, double* Hk) {
  // drive.cc：前 nv 个线程是阻尼和 armature（K/2 (q - r)^2）加干摩擦（h F f0(|v|)），
  // 后 nact 个是驱动（Huber 形状的一元凸函数）。
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  int d = -1;
  double de = 0, dg = 0, dh = 0;
  if (i < D.nv) {
    const double F = D.F ? D.F[i] : 0.0;
    if (D.K[i] == 0 && F == 0) return;
    d = i;
    const double x = q[D.dqadr[i]] - D.r[i];
    de = 0.5 * D.K[i] * x * x;
    dg = D.K[i] * x;
    dh = D.K[i];
    if (F != 0) {
      // friction.h 的 f0、f1 和 f1'（宽度 eps_v），v = (q - c) / h。
      const double v = (q[D.dqadr[i]] - D.c[i]) / D.h, y = fabs(v), eps = D.eps_v;
      const bool in = y < eps;
      de += D.h * F * (in ? -y * y * y / (3 * eps * eps) + y * y / eps + eps / 3 : y);
      dg += F * (in ? 2 * y / eps - y * y / (eps * eps) : 1.0) * (v < 0 ? -1.0 : 1.0);
      if (in) dh += F * 2 * (eps - y) / (eps * eps) / D.h;
    }
  } else if (i < D.nv + D.nact) {
    const int k = i - D.nv;
    d = D.adof[k];
    const double* a = D.act + 6 * k;
    const double s = a[0], A = a[1], C = a[2], w = a[3], lo = a[4], hi = a[5];
    const double qd = q[D.dqadr[d]];
    if (C == 0) {
      const double f = fmin(fmax(A, lo), hi);
      de = -s * f * qd;
      dg = -s * f;
    } else {
      const double y = A + C * qd, c = fmin(fmax(y, lo), hi);
      de = -(0.5 * c * c + c * (y - c)) / w;
      dg = -s * c;
      if (y > lo && y < hi) dh = -s * C;
    }
  } else {
    return;
  }
  if (e) atomicAdd(e, de);
  if (g) atomicAdd(g + d, dg);
  if (Hk && dh != 0) {
    const int t = m.dtree[d], l = d - m.tdof0[t];
    atomicAdd(Hk + t * m.TD * m.TD + l * m.TD + l, dh);
  }
}

__global__ void k_limit_ccd(DModel m, const double* q, const double* p, double c, double sep, double* tmin) {
  // limit_ccd：和阶段 1 的 ccd_max_step 一样的闭式解，距离减去最小分离 sep，夹到 >= 0。
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m.njnt || !m.jlimited[j]) return;
  const int a = m.jqadr[j], d = m.jdadr[j];
  const double dlo = q[a] - m.jrange[2 * j], dhi = m.jrange[2 * j + 1] - q[a];
  if (p[d] < 0) atomic_min(tmin, fmax(0.0, c * (dlo - sep) / -p[d]));
  if (p[d] > 0) atomic_min(tmin, fmax(0.0, c * (dhi - sep) / p[d]));
}

__global__ void k_fill(double* p, double v) { *p = v; }

__global__ void k_max_displacement(const double* Xa, const double* Xb, int n, double* out) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  atomic_max(out, norm(v3load(Xb + 3 * v) - v3load(Xa + 3 * v)));
}

// 没有顶点的模型的收敛判据（同 CPU 版 step.cc 的 max_vertex_displacement）：刚体质量的均方根位移
//   d_b^2 = |dc|^2 + th^T (R I R^T) th / m，th 是从 Fa 到 Fb 的转动向量。没有质量的刚体不算。
__global__ void k_max_body_rms(DModel m, DFrames Fa, DFrames Fb, double* out) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (b >= m.nbody || m.mass[b] <= 0) return;
  const Q4 qa = q4load(Fa.xquat + 4 * b);
  Q4 dq = qmul(q4load(Fb.xquat + 4 * b), qconj(qa));
  if (dq.w < 0) dq = Q4{-dq.w, -dq.x, -dq.y, -dq.z};
  const double s = norm(qvec(dq));
  const V3 th = s > 0 ? (2 * atan2(s, dq.w) / s) * qvec(dq) : 2.0 * qvec(dq);
  const V3 tb = qrot(qconj(qa), th);  // 转到刚体坐标系：th^T R I R^T th = tb^T I tb
  const M3 I = mload(m.inertia + 9 * b);
  const double d2 = norm2(com(m, Fb, b) - com(m, Fa, b)) + dot(tb, mv(I, tb)) / m.mass[b];
  atomic_max(out, sqrt(d2));
}

}  // namespace

void launch_integrate(const DModel& m, const double* q, const double* p, DScalar s, double* out) {
  if (m.njnt > 0) k_integrate<<<blocks(m.njnt), kThreads>>>(m, q, p, s, out);
}

void launch_fk(const DModel& m, const double* q, const DFrames& F) {
  k_fk<<<blocks(m.nbody), kThreads>>>(m, q, F);
  if (m.nvert > 0) k_vertices<<<blocks(m.nvert), kThreads>>>(m, F);
}

void launch_targets(const DModel& m, const DFrames& now, const DFrames& prev, double h, double* tc, double* tA) {
  k_targets<<<blocks(m.nbody), kThreads>>>(m, now, prev, h, tc, tA);

}

void launch_inertia_energy(const DModel& m, const DFrames& F, const double* tc, const double* tA, double h, double* e) {
  k_inertia_energy<<<blocks(m.nbody), kThreads>>>(m, F, tc, tA, h, e);
}

void launch_inertia_gradient(const DModel& m, const DFrames& F, const double* tc, const double* tA, double h, double* g) {
  k_inertia_gradient<<<blocks(m.nbody), kThreads>>>(m, F, tc, tA, h, g);
}

void launch_inertia_hessian(const DModel& m, const DFrames& F, double h, double* Hk) {
  if (m.ntree == 0) return;
  const size_t crb = sizeof(double) * (72 * static_cast<size_t>(m.TB) + 6 * static_cast<size_t>(m.TD));
  if (crb <= 48 * 1024) {
    k_inertia_hessian_crb<<<m.ntree, 256, crb>>>(m, F, h, Hk);
    return;
  }
  const size_t smem = sizeof(double) * (12 * static_cast<size_t>(m.TB) + 6 * static_cast<size_t>(m.TD));
  k_inertia_hessian<<<m.ntree, 256, smem>>>(m, F, h, Hk);
}

size_t inertia_curvature_smem(const DModel& m) {
  return sizeof(double) * (42 * static_cast<size_t>(m.TB) + 7 * static_cast<size_t>(m.TD) +
                           static_cast<size_t>(m.TD) * m.TD);
}

bool inertia_curvature_fits(const DModel& m) {
  static int optin = -1;
  if (optin < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  const size_t need = inertia_curvature_smem(m);
  if (need > static_cast<size_t>(optin)) return false;
  if (need > 48 * 1024)
    cudaFuncSetAttribute(k_inertia_curvature, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(need));
  return true;
}

void launch_inertia_curvature(const DModel& m, const DFrames& F, const double* tc, const double* tA, double h, bool rot,
                              double floor, double* Hk, double* K, unsigned long long* dropped, const double* CB,
                              bool inertia) {
  if (m.ntree == 0) return;
  k_inertia_curvature<<<m.ntree, 256, inertia_curvature_smem(m)>>>(m, F, tc, tA, h, rot ? 1.0 : 0.0, floor, Hk, K, dropped,
                                                                   CB, inertia ? 1 : 0);
}

void launch_drives(const DModel& m, const DDrive& D, const double* q, double* e, double* g, double* Hk) {
  const int n = D.nv + D.nact;
  if (n > 0) k_drives<<<(n + 127) / 128, 128>>>(m, D, q, e, g, Hk);
}

void launch_limits(const DModel& m, const double* q, double k, double dhat_hinge, double dhat_slide, double* e, double* g,
                   double* Hk) {
  if (m.njnt > 0) k_limits<<<blocks(m.njnt), kThreads>>>(m, q, k, dhat_hinge, dhat_slide, e, g, Hk);
}

void launch_limit_ccd(const DModel& m, const double* q, const double* p, double c, double sep, double* tmin) {
  if (m.njnt > 0) k_limit_ccd<<<blocks(m.njnt), kThreads>>>(m, q, p, c, sep, tmin);
}

void launch_fill(double* p, double v) { k_fill<<<1, 1>>>(p, v); }

void launch_max_displacement(const double* Xa, const double* Xb, int n, double* out) {
  if (n > 0) k_max_displacement<<<blocks(n), kThreads>>>(Xa, Xb, n, out);
}

void launch_max_body_rms(const DModel& m, const DFrames& Fa, const DFrames& Fb, double* out) {
  if (m.nbody > 1) k_max_body_rms<<<blocks(m.nbody - 1), kThreads>>>(m, Fa, Fb, out);
}

}  // namespace rcipc::gpu
