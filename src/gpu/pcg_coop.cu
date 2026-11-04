// 协作内核版的 PCG（阶段 11 的优化，笔记 notes/11_gpu.md 第 8 节）。
//
// solver.cu 的 run_pcg 每次迭代要发好几个内核、读回 2 个点积；60 个布娃娃每个 Newton 迭代 15 次 PCG 迭代时，
// 时间几乎全花在发内核和同步上。这里把整个 PCG 循环放进一个内核：
//   - 用协作启动（cudaLaunchCooperativeKernel），块数不超过能同时驻留的块数，于是可以用 grid.sync() 做全局同步；
//   - 一个 warp 负责固定的几棵树（通常一棵），这些树的 p、r、Ap 和每个自由度的运动向量 S_d 放在这个 warp 的共享内存里，
//     整个求解期间不离开（只有这个 warp 读写它们），不用每次迭代往显存里来回搬；
//   - 每次迭代只有 3 个全局同步，分成三段：
//       A（一个 warp 一棵树）：p = z + beta p；这棵树每个刚体的速度旋量（rows.cuh），并把刚体上的力旋量清零；
//          树块的 Ap = H_k p 和 p^T H_k p；
//       B（一个线程一行）：s = r . p（两侧的力旋量点乘速度旋量），p^T H p 加上 D s^2，D s (f, tau) 原子加到两侧的刚体上；
//       C（一个 warp 一棵树）：每个刚体的子树之和，投影到自由度加进 Ap；x += alpha p，r -= alpha Ap，
//          z = P^-1 r（P^-1 是显式的逆，solver.cu），r^T z。
//     p^T H p 直接按 p^T H_k p + sum D (r . p)^2 累加，不用等 Ap 算完再做一次点积；
//   - 运动向量 S_d：转动自由度 (o_d x a_d, a_d)，滑动自由度 (a_d, 0)。刚体的速度旋量是它链上的 sum p_d S_d，
//     力旋量 W 在自由度 d 上的值是 S_d . W（rows.cuh 的 wrench_on_dof 展开以后就是这个），求解开始时算一次；
//   - 树块的乘法按列读（H_k、P^-1 都对称）：warp 里相邻的线程读相邻的地址，访存能合并；
//   - 点积：每个 warp 先用 shuffle 归约，再原子加一次。每次迭代用 acc 里新的两格（调用者事先清零），循环里不用清零；
//   - 收敛与否、是否 breakdown 由同步之后读到的全局量决定，所有线程走同一个分支，grid.sync() 的次数一致。
// 算法和 run_pcg 一样（同样的初值、rz 准则、breakdown 保护），只是同步的方式和求和的顺序不同。
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "devmath.cuh"
#include "kernels.h"
#include "rows.cuh"

namespace cg = cooperative_groups;

namespace rcipc::gpu {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// warp 内求和，lane 0 原子加到 *out。整个 warp 都要调用。
__device__ inline void warp_add(double v, double* out) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0 && v != 0) atomicAdd(out, v);
}

// 对称的树块 M_t 的第 li 行乘 v（按列读：sum_lj M[lj][li] v[lj]，相邻线程读相邻地址）。
__device__ inline double tree_row(const double* M, int TD, int li, int n, const double* v) {
  double s = 0;
  for (int lj = 0; lj < n; ++lj) s += M[lj * TD + li] * v[lj];
  return s;
}

// 一棵树在共享内存里的数据（每个 warp 有 slots 份，每份 9 TD 个 double）。
struct TreeShared {
  double *S, *p, *r, *Ap;  // S：6 TD（每个自由度的运动向量），p、r、Ap：各 TD
};
__device__ inline TreeShared tree_shared(double* base, int TD, int slot) {
  double* b = base + static_cast<size_t>(slot) * 9 * TD;
  return TreeShared{b, b + 6 * TD, b + 7 * TD, b + 8 * TD};
}

__global__ void k_pcg_coop(DModel m, DFrames F, const double* Hk, DRows rows, DCount nrows, const double* P,
                           const double* rhs, double* x, double* z, double* tw, double* wacc, double* wsub, double* acc,
                           double tol, int max_iter, int slots, double* out) {
  const int nrow = nrows.get();
  extern __shared__ double smem[];
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, nth = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31, warp = tid >> 5, nwarp = nth >> 5;
  const int TD = m.TD;
  const size_t TD2 = static_cast<size_t>(TD) * TD;
  double* wbase = smem + static_cast<size_t>(threadIdx.x >> 5) * slots * 9 * TD;  // 这个 warp 的共享内存

  // 初值：S_d；x = 0，r = rhs，z = P^-1 r，rz0 = r . z。树 t = warp + j nwarp 放在第 j 份。
  {
    double local = 0;
    for (int j = 0; j < slots; ++j) {
      const int t = warp + j * nwarp;
      if (t >= m.ntree) break;
      const TreeShared T = tree_shared(wbase, TD, j);
      const int d0 = m.tdof0[t], n = m.tndof[t];
      for (int li = lane; li < n; li += 32) {
        const int d = d0 + li;
        const V3 a = v3load(F.daxis + 3 * d);
        const V3 lin = m.drot[d] ? cross(v3load(F.danchor + 3 * d), a) : a;
        const V3 ang = m.drot[d] ? a : v3(0, 0, 0);
        v3store(lin, T.S + 6 * li);
        v3store(ang, T.S + 6 * li + 3);
        x[d] = 0;
        T.r[li] = rhs[d];
      }
      __syncwarp();
      for (int li = lane; li < n; li += 32) {
        const double zi = tree_row(P + t * TD2, TD, li, n, T.r);
        z[d0 + li] = zi;
        local += T.r[li] * zi;
      }
      __syncwarp();
    }
    warp_add(local, acc);
  }
  grid.sync();
  double rz = acc[0], beta = 0;
  const double rz0 = fabs(rz);
  bool conv = rz0 == 0;
  int it = 0;
  while (!conv && it < max_iter) {
    double* pAp_acc = acc + 1 + 2 * it;
    double* rz_acc = acc + 2 + 2 * it;
    // A：p = z + beta p（第一次 p = z）；速度旋量 sum p_d S_d；清零力旋量；Ap = H_k p，累加 p^T H_k p。
    double local = 0;
    for (int j = 0; j < slots; ++j) {
      const int t = warp + j * nwarp;
      if (t >= m.ntree) break;
      const TreeShared T = tree_shared(wbase, TD, j);
      const int d0 = m.tdof0[t], n = m.tndof[t], b0 = m.tbody0[t], nb = m.tnbody[t];
      for (int li = lane; li < n; li += 32) T.p[li] = it == 0 ? z[d0 + li] : z[d0 + li] + beta * T.p[li];
      __syncwarp();
      for (int lb = lane; lb < nb; lb += 32) {
        const int b = b0 + lb;
        double s6[6] = {0, 0, 0, 0, 0, 0};
        for (int k = 0; k < m.chain_n[b]; ++k) {
          const int l = m.chain_dof[b * m.CH + k] - d0;
          const double pl = T.p[l];
          for (int c = 0; c < 6; ++c) s6[c] += pl * T.S[6 * l + c];
        }
        for (int c = 0; c < 6; ++c) {
          tw[6 * b + c] = s6[c];
          wacc[6 * b + c] = 0;
        }
      }
      for (int li = lane; li < n; li += 32) {
        const double s = tree_row(Hk + t * TD2, TD, li, n, T.p);
        T.Ap[li] = s;
        local += T.p[li] * s;
      }
      __syncwarp();
    }
    warp_add(local, pAp_acc);
    grid.sync();
    // B：接触行。
    local = 0;
    for (int rr = tid; rr < nrow; rr += nth) {
      const double s = row_dot(rows, rr, tw);
      const double coef = rows.D[rr] * s;
      if (coef == 0) continue;
      local += coef * s;
      row_scatter(rows, rr, coef, wacc);
    }
    warp_add(local, pAp_acc);
    grid.sync();
    const double pAp = *static_cast<volatile double*>(pAp_acc);
    if (!(pAp > 0) || !isfinite(pAp)) break;  // breakdown：保留当前的 x
    const double alpha = rz / pAp;
    // C：行的部分投影回自由度（S_d . 子树的力旋量）；x += alpha p，r -= alpha Ap，z = P^-1 r，累加 r . z。
    local = 0;
    for (int j = 0; j < slots; ++j) {
      const int t = warp + j * nwarp;
      if (t >= m.ntree) break;
      const TreeShared T = tree_shared(wbase, TD, j);
      const int d0 = m.tdof0[t], n = m.tndof[t], b0 = m.tbody0[t], nb = m.tnbody[t];
      for (int lb = lane; lb < nb; lb += 32) subtree_wrench(m, b0 + lb, wacc, wsub + 6 * (b0 + lb));
      __syncwarp();
      for (int li = lane; li < n; li += 32) {
        const int d = d0 + li;
        const double* W = wsub + 6 * m.dbody[d];
        double ap = T.Ap[li];
        for (int c = 0; c < 6; ++c) ap += T.S[6 * li + c] * W[c];
        x[d] += alpha * T.p[li];
        T.r[li] -= alpha * ap;
      }
      __syncwarp();
      for (int li = lane; li < n; li += 32) {
        const double zi = tree_row(P + t * TD2, TD, li, n, T.r);
        z[d0 + li] = zi;
        local += T.r[li] * zi;
      }
      __syncwarp();
    }
    warp_add(local, rz_acc);
    grid.sync();
    const double rz_new = *static_cast<volatile double*>(rz_acc);
    ++it;
    if (fabs(rz_new) <= tol * rz0) {
      conv = true;
      break;
    }
    if (!isfinite(rz_new)) break;
    beta = rz_new / rz;
    rz = rz_new;
  }
  if (tid == 0) {
    out[0] = it;
    out[1] = conv ? 1.0 : 0.0;
  }
}

}  // namespace

bool pcg_coop_supported() {
  int dev = 0, attr = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  cudaDeviceGetAttribute(&attr, cudaDevAttrCooperativeLaunch, dev);
  return attr != 0;
}

int run_pcg_coop(const DModel& m, const DFrames& F, const double* Hk, DRows rows, DCount nrow, const double* P,
                 const double* rhs, double* x, double tol, int max_iter, double* work, double* acc, double* res) {
  // 块数和每个 warp 放几棵树（slots）：块数不超过能同时驻留的块数（和共享内存的用量有关），
  // 也不超过任务需要的量（每棵树一个 warp，每行一个线程）；warp 不够分时一个 warp 放几棵树。
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int want = std::max({(m.ntree + kWarps - 1) / kWarps, (nrow.grid() + kThreads - 1) / kThreads, 1});
  int nblocks = 0, slots = 1;
  size_t smem = 0;
  for (;; ++slots) {
    smem = static_cast<size_t>(kWarps) * slots * 9 * m.TD * sizeof(double);
    if (smem > 48 * 1024) return -1;  // 树太大，放不进共享内存：调用者改用 run_pcg
    int per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_pcg_coop, kThreads, smem);
    nblocks = std::min(sms * per_sm, want);
    if (nblocks >= 1 && static_cast<long long>(nblocks) * kWarps * slots >= m.ntree) break;
  }
  const int n = m.nv;
  double* z = work + n;
  double* tw = work + 4 * n;
  double* wacc = tw + 6 * m.nbody;
  double* wsub = wacc + 6 * m.nbody;
  cudaMemsetAsync(acc, 0, sizeof(double) * (2 * static_cast<size_t>(max_iter) + 4));
  DModel mm = m;
  DFrames FF = F;
  void* args[] = {&mm, &FF, &Hk, &rows, &nrow, &P, &rhs, &x, &z, &tw, &wacc, &wsub, &acc, &tol, &max_iter, &slots, &res};
  const cudaError_t e = cudaLaunchCooperativeKernel(reinterpret_cast<void*>(k_pcg_coop), nblocks, kThreads, args, smem, 0);
  if (e != cudaSuccess) throw std::runtime_error(std::string("cooperative PCG launch: ") + cudaGetErrorString(e));
  return 0;  // 结果在 res 里，不在这里等
}

}  // namespace rcipc::gpu
