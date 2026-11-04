// GPU 版的线性求解（阶段 11，笔记 notes/11_gpu.md 第 5 节）。对应 CPU 版 solver.cc。
//
//   H p = 树块 p + sum_r D_r r_r (r_r . p)：树块一个线程一行；接触行按力旋量算（rows.cuh）：
//     刚体的速度旋量（一个线程一个刚体）、行（一个线程一行，力旋量原子加到刚体上）、子树之和（一个线程一个刚体）、
//     投影到自由度（一个线程一个自由度）。
//   预条件器：每棵树一块 P_t = H_k,t + sum_r D_r [r_r^T r_r]_tt，Jacobi 缩放、加 1e-10 的对角岭、分解。
//     分解完再把逆显式算出来存着：P_t^-1 = S (L D L^T)^-1 S。PCG 每次迭代都要用一次 P^-1，
//     用逆就是一个小的稠密矩阵乘向量（一个线程一行，全并行），不用一行一行地前代、回代（那是串行的）。
//     行的部分不再逐行把外积原子加进树块（一个布娃娃的几百行都要加到它根部的那几个元素上，排队）：
//       只落在一侧的部分，先在每个刚体上累加 6x6 的 sum D w w^T，再按子树求和，每个元素是一次 s_i^T W s_j
//       （和惯性 Hessian 的组合刚体写法一样，rows.cuh 的 dof_axis6）；两侧在同一棵树里的行另有交叉项，单独加。
//     分解在共享内存里做，LDL^T，一列一次同步，块里的线程分摊后面的整块更新（树太大放不进共享内存时用原来的写法）。
//   PCG：主机上的循环，每次迭代几个内核加两次读回（p^T H p 和 r^T z），和 CPU 版的 pcg 一行一行对应。
//     协作内核版（整个循环一个内核）在 pcg_coop.cu。
#include <algorithm>
#include <cmath>

#include "devmath.cuh"
#include "kernels.h"
#include "rows.cuh"

namespace rcipc::gpu {

namespace {

constexpr int kThreads = 128;
inline int blocks(int n) { return (n + kThreads - 1) / kThreads; }

__global__ void k_matvec_tree(DModel m, const double* Hk, const double* p, double* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m.nv) return;
  const int t = m.dtree[i], d0 = m.tdof0[t], n = m.tndof[t], li = i - d0;
  const double* row = Hk + t * m.TD * m.TD + li * m.TD;
  double s = 0;
  for (int lj = 0; lj < n; ++lj) s += row[lj] * p[d0 + lj];
  y[i] = s;
}

__global__ void k_twists(DModel m, DFrames F, const double* p, double* tw) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < m.nbody) body_twist(m, F, b, p, tw + 6 * b);
}

__global__ void k_rows_scatter(DRows rows, int nrow, const double* tw, double* wacc) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrow) return;
  const double coef = rows.D[r] * row_dot(rows, r, tw);
  if (coef != 0) row_scatter(rows, r, coef, wacc);
}

__global__ void k_subtree(DModel m, const double* wacc, double* wsub) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < m.nbody) subtree_wrench(m, b, wacc, wsub + 6 * b);
}

__global__ void k_project(DModel m, DFrames F, const double* wsub, double* y) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < m.nv) y[d] += wrench_on_dof(m, F, d, wsub + 6 * m.dbody[d]);
}

__global__ void k_precond_rows(DModel m, DFrames F, DRows rows, DCount nrows, double* P) {
  const int nrow = nrows.get();
  // 每行在同一棵树里的那些元素对 (e, f)：P_t[le][lf] += D val_e val_f。一个 warp 一行：
  //   1. lane k 算这一行的第 k 个元素（行展开成逐个自由度的值，rows.cuh）：先数两侧各有几个，再按位置算；
  //   2. lane e 拿着自己的 (idx_e, D val_e)，依次用 shuffle 取到每个 val_f，同一棵树、lf <= le 的就原子加。
  // 只加下三角：分解只读下三角，上三角留着 H_k 的值也没关系。
  const int lane = threadIdx.x & 31;
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (r >= nrow) return;  // 整个 warp 一起退出
  const double D = rows.D[r];
  if (D == 0) return;
  const int bA = rows.body[2 * r], bB = rows.body[2 * r + 1];
  const int nA = bA > 0 ? m.chain_n[bA] : 0, nB = bB > 0 ? m.chain_n[bB] : 0, n = nA + nB;
  for (int e0 = 0; e0 < n; e0 += 32) {  // 2 CH 不超过 32 时只有一轮
    const int e = e0 + lane;
    int ide = -1;
    double ve = 0;
    if (e < n) {
      const int side = e < nA ? 0 : 1, k = e < nA ? e : e - nA, b = side == 0 ? bA : bB;
      ide = m.chain_dof[b * m.CH + k];
      ve = wrench_on_dof(m, F, ide, rows.w + 12 * static_cast<size_t>(r) + 6 * side);
    }
    const int te = ide >= 0 ? m.dtree[ide] : -1;
    const int le = ide >= 0 ? ide - m.tdof0[te] : 0;
    double* Prow = te >= 0 ? P + static_cast<size_t>(te) * m.TD * m.TD + le * m.TD : nullptr;
    const double De = D * ve;
    for (int f0 = 0; f0 < n; f0 += 32) {
      // 第 f0..f0+31 个元素：lane 自己算一份（和上面一样），再广播。
      const int f = f0 + lane;
      int idf = -1;
      double vf = 0;
      if (f < n) {
        const int side = f < nA ? 0 : 1, k = f < nA ? f : f - nA, b = side == 0 ? bA : bB;
        idf = m.chain_dof[b * m.CH + k];
        vf = f0 == e0 ? ve : wrench_on_dof(m, F, idf, rows.w + 12 * static_cast<size_t>(r) + 6 * side);
      }
      const int cnt = min(32, n - f0);
      for (int s = 0; s < cnt; ++s) {
        const int id = __shfl_sync(0xffffffffu, idf, s);
        const double v = __shfl_sync(0xffffffffu, vf, s);
        if (te < 0 || id < 0 || m.dtree[id] != te) continue;
        const int lf = id - m.tdof0[te];
        if (lf <= le) atomicAdd(Prow + lf, De * v);
      }
    }
  }
}

__global__ void k_precond_factor(DModel m, double* P, double* scale) {
  // 一个线程块一棵树。P 的前一半（ntree 块）进来时是 B = P_t，出去时是 S (L L^T)^-1 S；后一半是工作区 Y。
  //   1. Jacobi 缩放 B <- S B S + 1e-10 I，S = diag(1 / sqrt(B_ii))；
  //   2. Cholesky B = L L^T（原地，下三角），一列一列地：先算对角元，再由块里的线程分摊这一列下面的元素；
  //   3. Y = L^-1：一个线程一列，前代；
  //   4. P^-1 = S Y^T Y S：一个线程一个元素（Y 是下三角，所以只加 k >= max(i, j) 的项）。
  const int t = blockIdx.x;
  if (t >= m.ntree) return;
  const int n = m.tndof[t], d0 = m.tdof0[t], TD = m.TD;
  double* B = P + static_cast<size_t>(t) * TD * TD;
  double* Y = P + static_cast<size_t>(m.ntree + t) * TD * TD;
  for (int i = threadIdx.x; i < n; i += blockDim.x) scale[d0 + i] = B[i * TD + i] > 0 ? 1.0 / sqrt(B[i * TD + i]) : 1.0;
  __syncthreads();
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {
    const int i = e / n, j = e % n;
    if (j > i) continue;  // 只用下三角（k_precond_rows 也只加了下三角）
    B[i * TD + j] = scale[d0 + i] * B[i * TD + j] * scale[d0 + j] + (i == j ? 1e-10 : 0.0);
  }
  __syncthreads();
  __shared__ double Ljj;
  for (int j = 0; j < n; ++j) {
    if (threadIdx.x == 0) {
      double d = B[j * TD + j];
      for (int k = 0; k < j; ++k) d -= B[j * TD + k] * B[j * TD + k];
      Ljj = sqrt(fmax(d, 1e-300));
      B[j * TD + j] = Ljj;
    }
    __syncthreads();
    for (int i = j + 1 + threadIdx.x; i < n; i += blockDim.x) {
      double s = B[i * TD + j];
      for (int k = 0; k < j; ++k) s -= B[i * TD + k] * B[j * TD + k];
      B[i * TD + j] = s / Ljj;
    }
    __syncthreads();
  }
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    for (int i = 0; i < j; ++i) Y[i * TD + j] = 0;
    for (int i = j; i < n; ++i) {
      double s = i == j ? 1.0 : 0.0;
      for (int k = j; k < i; ++k) s -= B[i * TD + k] * Y[k * TD + j];
      Y[i * TD + j] = s / B[i * TD + i];
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {
    const int i = e / n, j = e % n;
    double s = 0;
    for (int k = i > j ? i : j; k < n; ++k) s += Y[k * TD + i] * Y[k * TD + j];
    B[i * TD + j] = scale[d0 + i] * s * scale[d0 + j];
  }
}

// ---------- 预条件器的新写法（笔记第 11 节）----------

// 两个刚体最近的公共祖先（同一棵树里）。MuJoCo 的刚体按深度优先排序，祖先的编号总比后代小。
__device__ inline int common_ancestor(const DModel& m, int a, int b) {
  while (a != b) {
    if (a > b) a = m.parent[a];
    else b = m.parent[b];
  }
  return a;
}

// 自由度 d 在刚体 L 的链上：d 所在的刚体是 L 的祖先或 L 自己。
__device__ inline bool on_chain(const DModel& m, int d, int L) {
  const int bd = m.dbody[d];
  return bd <= L && L < m.bsub_end[bd];
}

__global__ void k_precond_outer(DModel m, DFrames F, DRows rows, DCount nrows, DPrecondWork W) {
  const int nrow = nrows.get();
  // 一个线程一行。每一侧（刚体 b > 0）的力旋量换到树根 ref 上：w = (f, tau - ref x f)，
  // 把 D w w^T 的上三角原子加到 Wb[b]。
  // 两侧 A、B 在同一棵树里时还有交叉项 D (u v'^T + v' u^T)（u、v' 是两侧在各自链上的值）。两条链共有的那一段
  // （公共祖先 L 的链）上，交叉项是 s_d^T D (w_A w_B^T + w_B w_A^T) s_e，和单侧的项同一个形状，所以也加到 Wb[L]，
  // 交给子树求和；剩下至少有一个自由度在 L 下面的那些对，记进 cross，由 k_precond_cross 加。
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrow) return;
  const double D = rows.D[r];
  if (D == 0) return;
  double ws[2][6];
  for (int side = 0; side < 2; ++side) {
    const int b = rows.body[2 * r + side];
    if (b <= 0) continue;
    const V3 ref = v3load(F.xpos + 3 * m.tbody0[m.btree[b]]);
    const double* w = rows.w + 12 * static_cast<size_t>(r) + 6 * side;
    const V3 f = v3load(w);
    v3store(f, ws[side]);
    v3store(v3load(w + 3) - cross(ref, f), ws[side] + 3);
    double* out = W.Wb + 21 * static_cast<size_t>(b);
    for (int i = 0; i < 6; ++i) {
      const double Di = D * ws[side][i];
      for (int j = i; j < 6; ++j) atomicAdd(out + sym21(i, j), Di * ws[side][j]);
    }
  }
  const int bA = rows.body[2 * r], bB = rows.body[2 * r + 1];
  if (bA > 0 && bB > 0 && m.btree[bA] == m.btree[bB]) {
    const int L = common_ancestor(m, bA, bB);
    double* out = W.Wb + 21 * static_cast<size_t>(L);
    for (int i = 0; i < 6; ++i)
      for (int j = i; j < 6; ++j) atomicAdd(out + sym21(i, j), D * (ws[0][i] * ws[1][j] + ws[1][i] * ws[0][j]));
    const int slot = atomicAdd(W.ncross, 1);
    if (slot < W.cap) W.cross[slot] = r;
  }
}

__global__ void k_precond_cross(DModel m, DFrames F, DRows rows, DPrecondWork W, double* P) {
  // 两侧在同一棵树 t 里的行，交叉项 D (u v'^T + v' u^T) 里 d、e 不都在公共祖先 L 的链上的那些对（L 的链上的
  // 已经由 k_precond_outer 加进了 Wb[L]）：对 A 链上的每个 d、B 链上的每个 e，把 D u_d v'_e 加到下三角的 (max, min)。
  // 剩下的对里 d != e（d = e 的自由度两条链都有，一定在 L 的链上）。一个 warp 一行，网格跨步。
  // 相邻的两个刚体（A 是 B 的子刚体）时 L = B，剩下的只有 A 自己的关节那几个自由度乘 B 的链。
  const int lane = threadIdx.x & 31;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int n = min(*W.ncross, W.cap);
  for (int c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; c < n; c += nwarp) {
    const int r = W.cross[c];
    const double D = rows.D[r];
    const int bA = rows.body[2 * r], bB = rows.body[2 * r + 1];
    const int L = common_ancestor(m, bA, bB);
    const int t = m.btree[bA], d0 = m.tdof0[t];
    const int nA = m.chain_n[bA], nB = m.chain_n[bB];
    const double* wA = rows.w + 12 * static_cast<size_t>(r);
    const double* wB = wA + 6;
    double* Pt = P + static_cast<size_t>(t) * m.TD * m.TD;
    for (int e0 = 0; e0 < nA; e0 += 32) {
      const int e = e0 + lane;
      const int ide = e < nA ? m.chain_dof[bA * m.CH + e] : -1;
      const bool inA = ide >= 0 && on_chain(m, ide, L);
      const double ue = ide >= 0 ? D * wrench_on_dof(m, F, ide, wA) : 0.0;
      const int le = ide - d0;
      for (int f0 = 0; f0 < nB; f0 += 32) {
        const int f = f0 + lane;
        const int idf = f < nB ? m.chain_dof[bB * m.CH + f] : -1;
        const int inB = idf >= 0 && on_chain(m, idf, L);
        const double vf = idf >= 0 ? wrench_on_dof(m, F, idf, wB) : 0.0;
        const int cnt = min(32, nB - f0);
        for (int s = 0; s < cnt; ++s) {
          const int id = __shfl_sync(0xffffffffu, idf, s);
          const double v = __shfl_sync(0xffffffffu, vf, s);
          const int inL = __shfl_sync(0xffffffffu, inB, s);
          if (ide < 0 || (inA && inL)) continue;
          const int lf = id - d0;
          atomicAdd(Pt + max(le, lf) * m.TD + min(le, lf), ue * v);
        }
      }
    }
  }
}

__global__ void k_precond_factor_sh(DModel m, DFrames F, const double* Wb, double* P, double* scale) {
  // 一个线程块一棵树 t，全部在共享内存里：
  //   1. Wbar_k = 子树里各刚体 Wb 之和（展开成 36 个），s_d = 自由度的空间轴（关于树根）；
  //      B = P_t（树块 H_k 加上交叉项，下三角）+ s_i^T Wbar_lo s_j（i、j 在一条链上时，lo 是较深的刚体）；
  //   2. Jacobi 缩放 B <- S B S + 1e-10 I；
  //   3. LDL^T（右视）：第 j 列的主元 d_j = B_jj，后面的整块 B_ik -= B_ij B_kj / d_j（i >= k > j），一列一次同步；
  //   4. Y = L^-1（L 是单位下三角，L_ij = B_ij / d_j）：一个线程一列，前代；
  //   5. P_t^-1 = S Y^T D^-1 Y S，写回 P（两个三角都写，k_precond_apply 读整行）。
  extern __shared__ double sh[];
  const int t = blockIdx.x;
  const int n = m.tndof[t], d0 = m.tdof0[t], b0 = m.tbody0[t], nb = m.tnbody[t], TD = m.TD;
  double* Wbar = sh;              // TB x 36
  double* s6 = Wbar + 36 * m.TB;  // TD x 6
  double* sc = s6 + 6 * TD;       // TD：Jacobi 缩放
  double* rinv = sc + TD;         // TD：主元的倒数 1 / d_j（双精度除法在这张卡上是软件实现，每个只算一次）
  double* B = rinv + TD;          // n x n
  double* Y = B + n * n;          // n x n
  double* Pt = P + static_cast<size_t>(t) * TD * TD;
  const V3 ref = v3load(F.xpos + 3 * b0);
  for (int e = threadIdx.x; e < nb * 36; e += blockDim.x) {
    const int k = e / 36, i = (e % 36) / 6, j = e % 6;
    const int b = b0 + k, a = sym21(min(i, j), max(i, j));
    double s = 0;
    for (int c = b; c < m.bsub_end[b]; ++c) s += Wb[21 * static_cast<size_t>(c) + a];
    Wbar[e] = s;
  }
  for (int li = threadIdx.x; li < n; li += blockDim.x) dof_axis6(m, F, d0 + li, ref, s6 + 6 * li);
  __syncthreads();
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {
    const int li = e / n, lj = e % n;
    if (lj > li) continue;
    double v = Pt[li * TD + lj];
    const int bi = m.dbody[d0 + li], bj = m.dbody[d0 + lj];
    const int lo = max(bi, bj);
    if (lo < min(m.bsub_end[bi], m.bsub_end[bj])) v += form6(s6 + 6 * li, Wbar + 36 * (lo - b0), s6 + 6 * lj);
    B[li * n + lj] = v;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) sc[i] = B[i * n + i] > 0 ? 1.0 / sqrt(B[i * n + i]) : 1.0;
  __syncthreads();
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {
    const int i = e / n, j = e % n;
    if (j <= i) B[e] = sc[i] * B[e] * sc[j] + (i == j ? 1e-10 : 0.0);
  }
  __syncthreads();
  if (threadIdx.x == 0 && n > 0) rinv[0] = 1.0 / fmax(B[0], 1e-300);
  __syncthreads();
  for (int j = 0; j < n; ++j) {
    const double rd = rinv[j];
    const int w = n - 1 - j;  // 后面那一块是 w x w
    for (int e = threadIdx.x; e < w * w; e += blockDim.x) {
      const int i = j + 1 + e / w, k = j + 1 + e % w;
      if (k > i) continue;
      B[i * n + k] -= B[i * n + j] * B[k * n + j] * rd;
      if (e == 0) rinv[j + 1] = 1.0 / fmax(B[i * n + k], 1e-300);  // (j+1, j+1) 这时已经是下一列的主元
    }
    __syncthreads();
  }
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {  // L_ij = B_ij / d_j（i > j）
    const int i = e / n, j = e % n;
    if (j < i) B[e] *= rinv[j];
  }
  __syncthreads();
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    Y[j * n + j] = 1.0;
    for (int i = j + 1; i < n; ++i) {
      double s = -B[i * n + j];
      for (int k = j + 1; k < i; ++k) s -= B[i * n + k] * Y[k * n + j];
      Y[i * n + j] = s;
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {
    const int i = e / n, j = e % n;
    if (j > i) continue;
    double s = 0;
    for (int k = i; k < n; ++k) s += Y[k * n + i] * Y[k * n + j] * rinv[k];
    s *= sc[i] * sc[j];
    Pt[i * TD + j] = s;
    Pt[j * TD + i] = s;
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) scale[d0 + i] = sc[i];
}

__global__ void k_precond_apply(DModel m, const double* P, const double* r, double* z) {
  // z = P^-1 r：P^-1 是树块对角的，一个线程一行，和树块的矩阵乘向量一样。
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m.nv) return;
  const int t = m.dtree[i], d0 = m.tdof0[t], n = m.tndof[t], li = i - d0;
  const double* row = P + static_cast<size_t>(t) * m.TD * m.TD + li * m.TD;
  double s = 0;
  for (int lj = 0; lj < n; ++lj) s += row[lj] * r[d0 + lj];
  z[i] = s;
}

__global__ void k_dot(const double* a, const double* b, int n, double* out) {
  __shared__ double buf[kThreads];
  double s = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) s += a[i] * b[i];
  buf[threadIdx.x] = s;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) buf[threadIdx.x] += buf[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, buf[0]);
}

__global__ void k_cg_update(int n, double alpha, const double* p, const double* Ap, double* x, double* r) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] += alpha * p[i];
  r[i] -= alpha * Ap[i];
}

__global__ void k_cg_direction(int n, double beta, const double* z, double* p) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = z[i] + beta * p[i];
}

__global__ void k_scale(int n, double s, double* x) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}

double dot_sync(const double* a, const double* b, int n, double* scal) {
  cudaMemsetAsync(scal, 0, sizeof(double));
  launch_dot(a, b, n, scal);
  double h = 0;
  cudaMemcpy(&h, scal, sizeof(double), cudaMemcpyDeviceToHost);
  return h;
}

}  // namespace

void launch_scale(double* x, int n, double s) {
  if (n > 0) k_scale<<<blocks(n), kThreads>>>(n, s, x);
}

void launch_dot(const double* a, const double* b, int n, double* out) {
  const int nb = std::min(blocks(n), 256);
  if (n > 0) k_dot<<<nb, kThreads>>>(a, b, n, out);
}

void launch_matvec(const DModel& m, const DFrames& F, const double* Hk, DRows rows, int nrow, const double* p,
                   double* y, double* bw) {
  k_matvec_tree<<<blocks(m.nv), kThreads>>>(m, Hk, p, y);
  if (nrow == 0) return;
  double* tw = bw;
  double* wacc = bw + 6 * m.nbody;
  double* wsub = bw + 12 * m.nbody;
  k_twists<<<blocks(m.nbody), kThreads>>>(m, F, p, tw);
  cudaMemsetAsync(wacc, 0, sizeof(double) * 6 * m.nbody);
  k_rows_scatter<<<blocks(nrow), kThreads>>>(rows, nrow, tw, wacc);
  k_subtree<<<blocks(m.nbody), kThreads>>>(m, wacc, wsub);
  k_project<<<blocks(m.nv), kThreads>>>(m, F, wsub, y);
}

size_t precond_smem(const DModel& m) {
  return sizeof(double) * (36 * static_cast<size_t>(m.TB) + 8 * static_cast<size_t>(m.TD) +
                           2 * static_cast<size_t>(m.TD) * m.TD);
}

// 共享内存版的分解能不能用（树块和工作区放得进一个线程块的共享内存），能用时顺便把上限调上去。
bool precond_shared_ok(const DModel& m) {
  static int optin = -1;
  if (optin < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  const size_t need = precond_smem(m);
  if (need > static_cast<size_t>(optin)) return false;
  if (need > 48 * 1024)
    cudaFuncSetAttribute(k_precond_factor_sh, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(need));
  return true;
}

void launch_precond_build(const DModel& m, const DFrames& F, const double* Hk, DRows rows, DCount nrow, double* P,
                          double* scale, const DPrecondWork& W) {
  cudaMemcpyAsync(P, Hk, sizeof(double) * m.ntree * m.TD * m.TD, cudaMemcpyDeviceToDevice);
  if (m.ntree == 0) return;
  if (!precond_shared_ok(m)) {  // 树太大：原来的写法（逐行外积，全局内存里分解）
    if (nrow.grid() > 0) k_precond_rows<<<(nrow.grid() + 3) / 4, 128>>>(m, F, rows, nrow, P);  // 一个 warp 一行
    k_precond_factor<<<m.ntree, 64>>>(m, P, scale);
    return;
  }
  cudaMemsetAsync(W.Wb, 0, sizeof(double) * 21 * m.nbody);
  cudaMemsetAsync(W.ncross, 0, sizeof(int));
  if (nrow.grid() > 0) {
    k_precond_outer<<<blocks(nrow.grid()), kThreads>>>(m, F, rows, nrow, W);
    k_precond_cross<<<std::max(1, std::min(blocks(32 * nrow.grid()), 256)), kThreads>>>(m, F, rows, W, P);
  }
  k_precond_factor_sh<<<m.ntree, 256, precond_smem(m)>>>(m, F, W.Wb, P, scale);
}

void launch_precond_apply(const DModel& m, const double* P, const double* r, double* z) {
  k_precond_apply<<<blocks(m.nv), kThreads>>>(m, P, r, z);
}

int run_pcg(const DModel& m, const DFrames& F, const double* Hk, DRows rows, int nrow, const double* P,
            const double* rhs, double* x, double tol, int max_iter, double* work, double* scal, bool* converged) {
  // pcg（solver.cc）：x0 = 0，r0 = rhs，z0 = P^-1 r0，p0 = z0；rz 准则；p^T H p <= 0 或 NaN 时停。
  const int n = m.nv;
  double* r = work;
  double* z = work + n;
  double* p = work + 2 * n;
  double* Ap = work + 3 * n;
  cudaMemsetAsync(x, 0, sizeof(double) * n);
  cudaMemcpyAsync(r, rhs, sizeof(double) * n, cudaMemcpyDeviceToDevice);
  launch_precond_apply(m, P, r, z);
  cudaMemcpyAsync(p, z, sizeof(double) * n, cudaMemcpyDeviceToDevice);
  double rz = dot_sync(r, z, n, scal);
  const double rz0 = std::fabs(rz);
  *converged = false;
  if (rz0 == 0) {
    *converged = true;
    return 0;
  }
  int it = 0;
  for (; it < max_iter;) {
    launch_matvec(m, F, Hk, rows, nrow, p, Ap, work + 4 * n);
    const double pAp = dot_sync(p, Ap, n, scal);
    if (!(pAp > 0) || !std::isfinite(pAp)) break;
    const double alpha = rz / pAp;
    k_cg_update<<<blocks(n), kThreads>>>(n, alpha, p, Ap, x, r);
    launch_precond_apply(m, P, r, z);
    const double rz_new = dot_sync(r, z, n, scal);
    ++it;
    if (std::fabs(rz_new) <= tol * rz0) {
      *converged = true;
      break;
    }
    if (!std::isfinite(rz_new)) break;
    k_cg_direction<<<blocks(n), kThreads>>>(n, rz_new / rz, z, p);
    rz = rz_new;
  }
  return it;
}

}  // namespace rcipc::gpu
