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

__global__ void k_limits(DModel m, const double* q, double k, double dhat, double* e, double* g, double* Hk) {
  // limits.cc：到两端的距离各一个一维障碍。
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m.njnt || !m.jlimited[j]) return;
  const int a = m.jqadr[j], d = m.jdadr[j];
  const BV lo = barrier(q[a] - m.jrange[2 * j], dhat), hi = barrier(m.jrange[2 * j + 1] - q[a], dhat);
  if (e) atomicAdd(e, k * (lo.b + hi.b));
  if (g) atomicAdd(g + d, k * (lo.db - hi.db));
  if (Hk) {
    const int t = m.dtree[d], l = d - m.tdof0[t];
    atomicAdd(Hk + t * m.TD * m.TD + l * m.TD + l, k * (lo.ddb + hi.ddb));
  }
}

__global__ void k_drives(DModel m, DDrive D, const double* q, double* e, double* g, double* Hk) {
  // drive.cc：前 nv 个线程是阻尼和 armature（K/2 (q - r)^2），后 nact 个是驱动（Huber 形状的一元凸函数）。
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  int d = -1;
  double de = 0, dg = 0, dh = 0;
  if (i < D.nv) {
    if (D.K[i] == 0) return;
    d = i;
    const double x = q[D.dqadr[i]] - D.r[i];
    de = 0.5 * D.K[i] * x * x;
    dg = D.K[i] * x;
    dh = D.K[i];
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

__global__ void k_limit_ccd(DModel m, const double* q, const double* p, double c, double* tmin) {
  // limit_ccd：和阶段 1 的 ccd_max_step 一样的闭式解。
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m.njnt || !m.jlimited[j]) return;
  const int a = m.jqadr[j], d = m.jdadr[j];
  const double dlo = q[a] - m.jrange[2 * j], dhi = m.jrange[2 * j + 1] - q[a];
  if (p[d] < 0) atomic_min(tmin, c * dlo / -p[d]);
  if (p[d] > 0) atomic_min(tmin, c * dhi / p[d]);
}

__global__ void k_fill(double* p, double v) { *p = v; }

__global__ void k_max_displacement(const double* Xa, const double* Xb, int n, double* out) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  atomic_max(out, norm(v3load(Xb + 3 * v) - v3load(Xa + 3 * v)));
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

void launch_drives(const DModel& m, const DDrive& D, const double* q, double* e, double* g, double* Hk) {
  const int n = D.nv + D.nact;
  if (n > 0) k_drives<<<(n + 127) / 128, 128>>>(m, D, q, e, g, Hk);
}

void launch_limits(const DModel& m, const double* q, double k, double dhat, double* e, double* g, double* Hk) {
  if (m.njnt > 0) k_limits<<<blocks(m.njnt), kThreads>>>(m, q, k, dhat, e, g, Hk);
}

void launch_limit_ccd(const DModel& m, const double* q, const double* p, double c, double* tmin) {
  if (m.njnt > 0) k_limit_ccd<<<blocks(m.njnt), kThreads>>>(m, q, p, c, tmin);
}

void launch_fill(double* p, double v) { k_fill<<<1, 1>>>(p, v); }

void launch_max_displacement(const double* Xa, const double* Xb, int n, double* out) {
  if (n > 0) k_max_displacement<<<blocks(n), kThreads>>>(Xa, Xb, n, out);
}

}  // namespace rcipc::gpu
