// CUDA graph 版的 Newton 迭代里，设备上做的判断（笔记 notes/11_gpu.md 第 12 节）。
//
// 主机版的 step()（stepper.cc）每次 CCD 分段、每次线搜索试探都要把几个数读回主机，再决定下一步：
// 一个 Newton 迭代约 6 次同步，每次 GPU 都要空等主机发下一批内核。这里把这些判断写成单线程的小内核，
// 读写显存里的 DIter，循环用条件节点（WHILE）在设备上转，主机每个迭代只发一次图、读回一次。
// 每个判断都和主机版 step() 里的那几行一一对应（注释里写了是哪几行），所以两种写法走的是同一个算法。
#include <cfloat>

#include "devmath.cuh"
#include "iter.h"

namespace rcipc::gpu {

namespace {

__device__ inline void set_cont(DIter* it, cudaGraphConditionalHandle h, int v) {
  it->cont = v;
  if (h) cudaGraphSetConditional(h, v);
}

__global__ void k_it_begin(DIter* it) {
  // 组装之前：行数 = 每接触的行数 x 接触 + 2 x 摩擦，计数清零。
  it->nrow = it->rpc * it->nc + 2 * it->nf;
  it->row0f = it->rpc * it->nc;
  it->nfull = it->rpc == 12 ? it->nc : 0;
  it->status = kIterRunning;
  it->segs = it->windows = it->trials = 0;
  it->adv = 0;
  for (int k = 0; k < 4; ++k) it->need[k] = 0;
  it->t_ccd = -1;
}

__global__ void k_it_converge(DIter* it, const double* scal, double h, double tol) {
  // step() 第 2 步：组装的能量 scal[0]、整步里顶点的最大位移 scal[7]、PCG 的迭代次数 scal[12]。
  it->e0 = scal[0];
  it->pcg = static_cast<int>(scal[12]);
  it->size = scal[7] / h;
  if (it->status == kIterRunning && it->size < tol) it->status = kIterConverged;  // 组装时溢出过的不改
}

__global__ void k_ccd_setup(DIter* it, const double* A, int nbody, double dhat, double* scal,
                            cudaGraphConditionalHandle h) {
  // ccd() 的开头：第一段的长度 sqrt(8 dhat / A_max)，限位 CCD 的结果 scal[3] 先设成 1。一个线程块。
  __shared__ double red[256];
  double a = 0;
  for (int b = threadIdx.x; b < nbody; b += blockDim.x) a = fmax(a, A[b]);
  red[threadIdx.x] = a;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + k]);
    __syncthreads();
  }
  if (threadIdx.x != 0) return;
  const double A_max = red[0];
  it->ds = A_max > 0 ? fmin(1.0, sqrt(8 * dhat / A_max)) : 1.0;
  it->s0 = 0;
  it->t_ccd = -1;
  it->w1 = -1;
  scal[3] = 1.0;
  set_cont(it, h, it->status == kIterRunning ? 1 : 0);
}

__global__ void k_ccd_seg_begin(DIter* it, int reuse, double pair_len, cudaGraphConditionalHandle hf) {
  // 一段 [s0, s1]：s1 = min(1, s0 + ds)。integrate 读 it->s。
  // 落在窗口 [w0, w1] 里的段直接用窗口的候选；不在的段自己当新的窗口，重新求候选（条件节点 hf 打开）。
  // 和 ccd() 里 fresh 那几行一样。
  const double s0 = it->s0, s1 = fmin(1.0, s0 + it->ds);
  it->len = s1 - s0;
  it->s = s1;
  const int fresh = !reuse || !(s0 >= it->w0 && s1 <= it->w1);
  if (fresh) {
    ++it->windows;
    it->w0 = s0;
    it->w1 = s1;
    it->plen = pair_len * it->len;
  }
  if (hf) cudaGraphSetConditional(hf, fresh);
}

__global__ void k_ccd_seg_end(DIter* it, const double* scal, int pair_cap, int work_cap, int cand_cap,
                              cudaGraphConditionalHandle h) {
  // ccd() 每一段读回之后的那几行：scal[1] 截断位置，scal[2] 需要的段长，scal[9..11] 候选、刚体对、工作项的个数。
  const double t_min = scal[1], ds_need = scal[2];
  const int cands = static_cast<int>(scal[9]), pairs = static_cast<int>(scal[10]), work = static_cast<int>(scal[11]);
  it->adv = 0;
  if (pairs > pair_cap || work > work_cap || cands > cand_cap) {  // 放不下：这次迭代作废，主机扩容后重来
    it->status = kIterOverflow;
    it->need[0] = max(it->need[0], pairs);
    it->need[2] = max(it->need[2], cands);
    it->need[3] = max(it->need[3], work);
    set_cont(it, h, 0);
    return;
  }
  ++it->segs;
  if (it->segs > 10000) {  // 和 ccd() 的 guard 一样
    set_cont(it, h, 0);
    return;
  }
  if (t_min <= 0) {  // 有一对卡在最小分离以内（只有它会给出 0）：段起点之后一点也不能走，不用再缩短重做
    it->t_ccd = it->s0;
    set_cont(it, h, 0);
    return;
  }
  if (ds_need < it->len) {  // 某个顶点嫌这一段太长：缩短，重做这一段
    it->ds = fmin(0.5 * it->len, ds_need);
    set_cont(it, h, 1);
    return;
  }
  if (t_min < 1.0) {  // 截断在这一段里
    it->t_ccd = it->s0 + t_min * it->len;
    set_cont(it, h, 0);
    return;
  }
  it->s0 = it->s;  // 这一段安全，下一段从它的终点开始
  it->adv = 1;
  set_cont(it, h, it->s0 < 1.0 ? 1 : 0);
}

__global__ void k_copy_frames_if(const DIter* it, DFrames from, DFrames to, int nbody, int nv, int nvert) {
  // adv 时把这一段终点的 FK 拷成下一段的起点（主机版是交换 Fa、Fb 的指针；图里的指针是固定的，只能拷）。
  if (!it->adv) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 3 * nbody) to.xpos[i] = from.xpos[i];
  if (i < 4 * nbody) to.xquat[i] = from.xquat[i];
  if (i < 3 * nv) {
    to.daxis[i] = from.daxis[i];
    to.danchor[i] = from.danchor[i];
  }
  if (i < 3 * nvert) to.X[i] = from.X[i];
}

__global__ void k_ls_setup(DIter* it, const double* scal, cudaGraphConditionalHandle h) {
  // ccd() 的结尾和 step() 第 4 步的开头：alpha_max = min(截断位置, 限位 CCD)，线搜索从 alpha_max 开始。
  if (it->status != kIterRunning) {
    set_cont(it, h, 0);
    return;
  }
  const double result = it->t_ccd < 0 ? fmin(it->s0, 1.0) : it->t_ccd;
  it->alpha_max = fmin(result, scal[3]);
  it->alpha = it->alpha_max;
  it->s = it->alpha;
  set_cont(it, h, 1);
}

__global__ void k_ls_end(DIter* it, const double* scal, const int* cnt, int pair_cap, int contact_cap, int cand_cap,
                         int work_cap, cudaGraphConditionalHandle h) {
  // step() 第 4 步每次试探之后：能量 scal[0]、最小距离 scal[4]；检测的四个计数放得下才算数。
  if (cnt[0] > pair_cap || cnt[1] > contact_cap || cnt[2] > cand_cap || cnt[3] > work_cap) {
    it->status = kIterOverflow;
    for (int k = 0; k < 4; ++k) it->need[k] = max(it->need[k], cnt[k]);
    set_cont(it, h, 0);
    return;
  }
  ++it->trials;
  it->e_trial = scal[0];
  it->dmin = scal[4];
  if (it->e_trial > it->e0 && it->alpha > 1e-12) {
    it->alpha *= 0.5;
    it->s = it->alpha;
    set_cont(it, h, 1);
    return;
  }
  if (!isfinite(it->e_trial) && it->alpha > 0) {
    // 试探点穿进去了（能量 +无穷）：这一次一步也不走。再试一次 alpha = 0（就是当前点），接受它的 FK 和接触。
    it->alpha = 0;
    it->s = 0;
    set_cont(it, h, 1);
    return;
  }
  it->status = kIterDone;
  set_cont(it, h, 0);
}

__global__ void k_it_accept(DIter* it, const double* qtrial, double* q, int nq, DFrames from, DFrames to, int nbody,
                            int nv, int nvert, const int* cnt) {
  // step() 第 5 步：接受最后一次试探的点（q = qtrial，FK 和接触都用它的）。
  if (it->status != kIterDone) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nq) q[i] = qtrial[i];
  if (i < 3 * nbody) to.xpos[i] = from.xpos[i];
  if (i < 4 * nbody) to.xquat[i] = from.xquat[i];
  if (i < 3 * nv) {
    to.daxis[i] = from.daxis[i];
    to.danchor[i] = from.danchor[i];
  }
  if (i < 3 * nvert) to.X[i] = from.X[i];
  if (i == 0) it->nc = cnt[1];
}

int span(int nq, int nbody, int nv, int nvert) {
  int n = nq;
  n = n > 4 * nbody ? n : 4 * nbody;
  n = n > 3 * nv ? n : 3 * nv;
  n = n > 3 * nvert ? n : 3 * nvert;
  return n;
}

constexpr int kThreads = 256;

}  // namespace

void launch_it_begin(DIter* it) { k_it_begin<<<1, 1>>>(it); }
void launch_it_converge(DIter* it, const double* scal, double h, double tol) {
  k_it_converge<<<1, 1>>>(it, scal, h, tol);
}
void launch_ccd_setup(DIter* it, const double* A, int nbody, double dhat, double* scal, cudaGraphConditionalHandle h) {
  k_ccd_setup<<<1, 256>>>(it, A, nbody, dhat, scal, h);
}
void launch_ccd_seg_begin(DIter* it, bool reuse, double pair_len, cudaGraphConditionalHandle hf) {
  k_ccd_seg_begin<<<1, 1>>>(it, reuse ? 1 : 0, pair_len, hf);
}
void launch_ccd_seg_end(DIter* it, const double* scal, int pair_cap, int work_cap, int cand_cap,
                        cudaGraphConditionalHandle h) {
  k_ccd_seg_end<<<1, 1>>>(it, scal, pair_cap, work_cap, cand_cap, h);
}
void launch_copy_frames_if(const DIter* it, const DFrames& from, const DFrames& to, int nbody, int nv, int nvert) {
  const int n = span(0, nbody, nv, nvert);
  k_copy_frames_if<<<(n + kThreads - 1) / kThreads, kThreads>>>(it, from, to, nbody, nv, nvert);
}
void launch_ls_setup(DIter* it, const double* scal, cudaGraphConditionalHandle h) { k_ls_setup<<<1, 1>>>(it, scal, h); }
void launch_ls_end(DIter* it, const double* scal, const int* cnt, int pair_cap, int contact_cap, int cand_cap,
                   int work_cap, cudaGraphConditionalHandle h) {
  k_ls_end<<<1, 1>>>(it, scal, cnt, pair_cap, contact_cap, cand_cap, work_cap, h);
}
void launch_it_accept(DIter* it, const double* qtrial, double* q, int nq, const DFrames& from, const DFrames& to,
                      int nbody, int nv, int nvert, const int* cnt) {
  const int n = span(nq, nbody, nv, nvert);
  k_it_accept<<<(n + kThreads - 1) / kThreads, kThreads>>>(it, qtrial, q, nq, from, to, nbody, nv, nvert, cnt);
}

}  // namespace rcipc::gpu
