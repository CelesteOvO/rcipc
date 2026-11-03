// GPU 版的线性求解（阶段 11，笔记 notes/11_gpu.md 第 5 节）。对应 CPU 版 solver.cc。
//
//   H p = 树块 p + sum_r D_r r_r (r_r . p)：树块一个线程一行；接触行按力旋量算（rows.cuh）：
//     刚体的速度旋量（一个线程一个刚体）、行（一个线程一行，力旋量原子加到刚体上）、子树之和（一个线程一个刚体）、
//     投影到自由度（一个线程一个自由度）。
//   预条件器：每棵树一块 P_t = H_k,t + sum_r D_r [r_r^T r_r]_tt，Jacobi 缩放、加 1e-10 的对角岭、Cholesky。
//     CPU 版用 LDLT，这里用 LLT（块对称正定，两者的解只差舍入）。一个线程块一棵树。
//     分解完再把逆显式算出来存着：P_t^-1 = S L^-T L^-1 S。PCG 每次迭代都要用一次 P^-1，
//     用逆就是一个小的稠密矩阵乘向量（一个线程一行，全并行），不用一行一行地前代、回代（那是串行的）。
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

__global__ void k_precond_rows(DModel m, DFrames F, DRows rows, int nrow, double* P) {
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

void launch_precond_build(const DModel& m, const DFrames& F, const double* Hk, DRows rows, int nrow, double* P,
                          double* scale) {
  cudaMemcpyAsync(P, Hk, sizeof(double) * m.ntree * m.TD * m.TD, cudaMemcpyDeviceToDevice);
  if (nrow > 0) k_precond_rows<<<(nrow + 3) / 4, 128>>>(m, F, rows, nrow, P);  // 一个 warp 一行
  if (m.ntree > 0) k_precond_factor<<<m.ntree, 64>>>(m, P, scale);
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
