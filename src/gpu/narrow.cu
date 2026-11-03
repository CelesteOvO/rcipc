// 检测和 CCD 共用的窄相剔除（阶段 11，笔记 notes/11_gpu.md 第 3、4、8 节）。
//
// 输入是候选刚体对（launch_body_pairs），输出是候选图元对（DCands），由调用者用双精度精确判断
// （检测：k_detect_exact；CCD：k_ccd_exact）。剔除只用单精度，而且只会多留不会漏，所以结果和 CPU 版一样。
//
// 关键是在刚体坐标系里剔：刚体是刚性的，它的三角形、边在自己的坐标系里永远不动，所以它们的盒子、
// 分组（chunk）都在上传模型时算一次（stepper.cc 的 build_chunks），以后不用每次检测都重算世界坐标系里的盒子。
// 要查的另一个刚体的顶点或边，每次换到这个刚体的坐标系里（一对刚体一个相对变换）。这是上一个分支的做法（pq_chunk）。
//
// 两个内核：
//   1. k_pair_cull：一个 warp 一对刚体 (a, b)。先算两个相对变换（a 到 b、b 到 a），再把三种查询
//      （a 的顶点对 b 的三角形、b 的顶点对 a 的三角形、a 的边对 b 的边）换到对方的坐标系里，
//      和对方全部三角形（或边）的总盒子比，通过的写进工作列表（用 __ballot_sync 压紧）；
//   2. k_work_cull：一个线程一个工作项。先和对方的分组盒子比，命中的组里再和每个图元的盒子比，
//      再算单精度距离，减去余量还在阈值以内的写成候选。
// 阈值：检测是 dmax；CCD 是 T = max(2 xi, xi + 4 Dmax / c)（再放大 1%）：段起点的距离不小于 T 的对，
// ACCD 第一步就返回 1，什么也不改（笔记第 4 节）。两者都再加上两个图元的半径。
#include <algorithm>
#include <cfloat>

#include "devmath.cuh"
#include "kernels.h"

namespace rcipc::gpu {

namespace {

// 单精度距离的余量（米），乘以 (1 + R)，R 是这对刚体的世界坐标的最大绝对值。相对变换和刚体坐标系里的坐标
// 都比 R 小，单精度的舍入误差是 1e-7 R 的量级，这里留了两个数量级以上的余地（上一个分支的 EPS32 也是 1e-5 米）。
constexpr float kFloatPad = 1e-5f;
constexpr int kPairWarps = 4;  // k_pair_cull 一个线程块几个 warp（几对刚体）
constexpr int kThreads = 128;
constexpr int kLanes = 8;  // k_work_cull 一个工作项几个线程

// 从刚体 from 的坐标系到刚体 to 的坐标系的变换（单精度）：x_to = R x_from + t。
struct Xf {
  float R[9], t[3];
};
__device__ inline F3 xf_apply(const Xf& T, F3 v) {
  return F3{T.R[0] * v.x + T.R[1] * v.y + T.R[2] * v.z + T.t[0], T.R[3] * v.x + T.R[4] * v.y + T.R[5] * v.z + T.t[1],
            T.R[6] * v.x + T.R[7] * v.y + T.R[8] * v.z + T.t[2]};
}
__device__ inline Xf xf_load(const float* p) {
  Xf T;
  for (int k = 0; k < 9; ++k) T.R[k] = p[k];
  for (int k = 0; k < 3; ++k) T.t[k] = p[9 + k];
  return T;
}
// R = R_to^T R_from，t = R_to^T (x_from - x_to)，双精度算好再转成单精度。
__device__ inline void xf_store(const DFrames& F, int from, int to, float* out) {
  const M3 Rf = qmat(q4load(F.xquat + 4 * from)), Rt = qmat(q4load(F.xquat + 4 * to));
  const M3 R = mm(mt(Rt), Rf);
  const V3 t = mv(mt(Rt), v3load(F.xpos + 3 * from) - v3load(F.xpos + 3 * to));
  for (int k = 0; k < 9; ++k) out[k] = static_cast<float>(R.m[k]);
  out[9] = static_cast<float>(t.x);
  out[10] = static_cast<float>(t.y);
  out[11] = static_cast<float>(t.z);
}

__device__ inline F3 vlf(const DModel& m, int v) { return f3(m.vlf + 3 * v); }

// 点（带半径 r）的盒子。
__device__ inline FBox point_box(F3 p, float r) {
  return FBox{{p.x - r, p.y - r, p.z - r}, {p.x + r, p.y + r, p.z + r}};
}
__device__ inline FBox seg_box(F3 a, F3 b, float r) {
  return FBox{{fminf(a.x, b.x) - r, fminf(a.y, b.y) - r, fminf(a.z, b.z) - r},
              {fmaxf(a.x, b.x) + r, fmaxf(a.y, b.y) + r, fmaxf(a.z, b.z) + r}};
}

// 一对图元的半径之和（make_pt / make_ee / accd 用的那两个）。
__device__ inline double pair_radii(const DModel& m, int id1, int id2, bool ee) {
  return ee ? m.vradius[m.ev[2 * id1]] + m.vradius[m.ev[2 * id2]] : m.vradius[id1] + m.vradius[m.fv[3 * id2]];
}

// 查询：kind 0 = a 的顶点对 b 的三角形，1 = b 的顶点对 a 的三角形，2 = a 的边对 b 的边。
// 查询的刚体、目标刚体、目标的坐标系用的是哪个变换（每对存两个：0 是 a 到 b，1 是 b 到 a）。
__device__ inline int query_xf(int kind) { return kind == 1 ? 1 : 0; }

__global__ void k_pair_cull(DModel m, DFrames F, const double* boxes, const int* pairs, const int* npairs_d,
                            int pair_cap, double dmax, const double* A, const double* D, double len, double c,
                            DNarrow N) {
  const int lane = threadIdx.x & 31;
  const unsigned below = (1u << lane) - 1;
  const int pi = blockIdx.x * kPairWarps + (threadIdx.x >> 5);
  if (pi >= min(*npairs_d, pair_cap)) return;  // 刚体对的个数在显存里，按容量发线程块；整个 warp 一起退出
  const int body[2] = {pairs[2 * pi], pairs[2 * pi + 1]};
  // 阈值和余量（整个 warp 一样）。
  double T = dmax;
  if (A) {
    const double xi = (A[body[0]] + A[body[1]]) * len * len / 8;
    T = 1.01 * fmax(2 * xi, xi + 4 * fmax(D[body[0]], D[body[1]]) / c);
  }
  double R = 0;
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < 6; ++k) R = fmax(R, fabs(boxes[6 * body[s] + k]));
  const float pad = kFloatPad * (1.0f + static_cast<float>(R));
  if (lane == 0) {
    N.pT[pi] = T;
    N.ppad[pi] = pad;
    if (A) N.pxm[pi] = __fadd_ru(__double2float_ru((A[body[0]] + A[body[1]]) * len * len / 8), pad);
  }
  if (lane < 2) xf_store(F, body[lane], body[1 - lane], N.pxf + 24 * static_cast<size_t>(pi) + 12 * lane);
  __syncwarp();
  const Xf X[2] = {xf_load(N.pxf + 24 * static_cast<size_t>(pi)), xf_load(N.pxf + 24 * static_cast<size_t>(pi) + 12)};
  const float mf = __fadd_ru(__double2float_ru(T), pad);
  for (int kind = 0; kind < 3; ++kind) {
    const int q = kind == 1 ? body[1] : body[0], t = kind == 1 ? body[0] : body[1];
    const FBox tb = fbox_load((kind == 2 ? m.blbox_e : m.blbox_f) + 6 * t);
    if (tb.lo[0] > tb.hi[0]) continue;  // 目标刚体没有三角形（或边）
    const Xf& Tq = X[query_xf(kind)];
    const int first = kind == 2 ? m.bedge0[q] : m.bvert0[q], num = kind == 2 ? m.bnedge[q] : m.bnvert[q];
    for (int i0 = 0; i0 < num; i0 += 32) {
      const int i = i0 + lane;
      bool pass = false;
      if (i < num) {
        const int id = first + i;
        if (kind == 2) {
          const int v0 = m.ev[2 * id], v1 = m.ev[2 * id + 1];
          pass = fbox_overlap(seg_box(xf_apply(Tq, vlf(m, v0)), xf_apply(Tq, vlf(m, v1)), static_cast<float>(m.vradius[v0])),
                              tb, mf);
        } else {
          pass = fbox_overlap(point_box(xf_apply(Tq, vlf(m, id)), static_cast<float>(m.vradius[id])), tb, mf);
        }
      }
      const unsigned mask = __ballot_sync(0xffffffffu, pass);
      if (mask == 0) continue;
      int base = 0;
      if (lane == 0) base = atomicAdd(N.work_n, __popc(mask));
      base = __shfl_sync(0xffffffffu, base, 0);
      const int k = base + __popc(mask & below);
      if (pass && k < N.work_cap) N.work[k] = DWork{3 * pi + kind, first + i};
    }
  }
}

// 图元（种类 kind：0 顶点、1 边、2 三角形）在 CCD 一段里的单精度扫掠盒：顶点在段起点、终点的位置按半径放大，取并。
__device__ inline FBox swept_fbox(const DModel& m, const DNarrow& N, int kind, int id) {
  const int n = kind == 0 ? 1 : (kind == 1 ? 2 : 3);
  FBox b{{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
  for (int k = 0; k < n; ++k) {
    const int v = kind == 0 ? id : (kind == 1 ? m.ev[2 * id + k] : m.fv[3 * id + k]);
    const float r = static_cast<float>(m.vradius[v]);
    for (int a = 0; a < 3; ++a) {
      const float x = N.xa[3 * v + a], y = N.xb[3 * v + a];
      b.lo[a] = fminf(b.lo[a], fminf(x, y) - r);
      b.hi[a] = fmaxf(b.hi[a], fmaxf(x, y) + r);
    }
  }
  return b;
}

__global__ void k_work_cull(DModel m, const double* boxes, const int* pairs, DNarrow N, DCands C) {
  // 一个工作项 kLanes 个线程：都做同样的分组测试，命中的组里按 lane 分摊图元（每个 lane 每隔 kLanes 个取一个）。
  // 一个线程串行地查一组的 32 个图元时，时间几乎都在等一个接一个的访存；分给几个线程就能同时发出去。
  const int n = min(*N.work_n, N.work_cap);
  const int lane = threadIdx.x % kLanes;
  for (int w = (blockIdx.x * blockDim.x + threadIdx.x) / kLanes; w < n; w += gridDim.x * blockDim.x / kLanes) {
    const DWork item = N.work[w];
    const int pi = item.a / 3, kind = item.a % 3, id = item.b;
    const int t = kind == 1 ? pairs[2 * pi] : pairs[2 * pi + 1];  // 目标刚体
    const Xf Tq = xf_load(N.pxf + 24 * static_cast<size_t>(pi) + 12 * query_xf(kind));
    const double T = N.pT[pi];
    const float pad = N.ppad[pi], mf = __fadd_ru(__double2float_ru(T), pad);
    // 查询的图元在目标坐标系里的位置和盒子。
    F3 p0, p1;
    FBox qb;
    if (kind == 2) {
      p0 = xf_apply(Tq, vlf(m, m.ev[2 * id]));
      p1 = xf_apply(Tq, vlf(m, m.ev[2 * id + 1]));
      qb = seg_box(p0, p1, static_cast<float>(m.vradius[m.ev[2 * id]]));
    } else {
      p0 = xf_apply(Tq, vlf(m, id));
      qb = point_box(p0, static_cast<float>(m.vradius[id]));
    }
    const bool ee = kind == 2;
    const int* order = ee ? m.ech_order : m.fch_order;
    const int* range = ee ? m.ech_range : m.fch_range;
    const float* cbox = ee ? m.ech_box : m.fch_box;
    const float* ibox = ee ? m.elbox : m.flbox;
    const int* bch = ee ? m.bech : m.bfch;
    for (int ch = bch[2 * t]; ch < bch[2 * t] + bch[2 * t + 1]; ++ch) {
      if (!fbox_overlap(qb, fbox_load(cbox + 6 * ch), mf)) continue;
      for (int k = range[2 * ch] + lane; k < range[2 * ch] + range[2 * ch + 1]; k += kLanes) {
        const int it = order[k];
        if (!fbox_overlap(qb, fbox_load(ibox + 6 * it), mf)) continue;
        const float df = ee ? ee_dist_f(p0, p1, vlf(m, m.ev[2 * it]), vlf(m, m.ev[2 * it + 1]))
                            : pt_dist_f(p0, vlf(m, m.fv[3 * it]), vlf(m, m.fv[3 * it + 1]), vlf(m, m.fv[3 * it + 2]));
        if (df - pad >= __double2float_ru(T + pair_radii(m, id, it, ee))) continue;  // df 是 NaN 时也留下
        if (N.xa) {
          // CCD：再用单精度的世界坐标把 CPU 版的盒子规则先查一遍（第一个图元的扫掠盒和对方刚体的扫掠盒、
          // 和第二个图元的扫掠盒都按 xi 重叠）。余量盖住单精度的舍入，只会多留；精确的复核在 k_ccd_exact 里。
          const FBox qs = swept_fbox(m, N, ee ? 1 : 0, id);
          const float xm = N.pxm[pi];
          if (!fbox_overlap(qs, make_fbox(v3load(boxes + 6 * t), v3load(boxes + 6 * t + 3)), xm)) continue;
          if (!fbox_overlap(qs, swept_fbox(m, N, ee ? 1 : 2, it), xm)) continue;
        }
        const int c = atomicAdd(C.count, 1);
        if (c < C.cap) C.item[c] = DCand{id, ee ? ~it : it};
      }
    }
  }
}

}  // namespace

void launch_narrow(const DModel& m, const DFrames& F, const double* boxes, const int* pairs, const int* npairs_d,
                   int pair_cap, double dmax, const double* A, const double* D, double len, double c, DNarrow N,
                   DCands C) {
  cudaMemsetAsync(N.work_n, 0, sizeof(int));
  cudaMemsetAsync(C.count, 0, sizeof(int));
  if (pair_cap <= 0) return;
  k_pair_cull<<<(pair_cap + kPairWarps - 1) / kPairWarps, 32 * kPairWarps>>>(m, F, boxes, pairs, npairs_d, pair_cap,
                                                                            dmax, A, D, len, c, N);
  k_work_cull<<<std::min((kLanes * N.work_cap + kThreads - 1) / kThreads, 4096), kThreads>>>(m, boxes, pairs, N, C);
}

__global__ void k_to_float(const double* x, int n, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = static_cast<float>(x[i]);
}

void launch_to_float(const double* x, int n, float* out) {
  if (n > 0) k_to_float<<<(n + kThreads - 1) / kThreads, kThreads>>>(x, n, out);
}

}  // namespace rcipc::gpu
