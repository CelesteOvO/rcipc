#pragma once

// 检测和 CCD 共用的窄相工具（阶段 11，笔记 notes/11_gpu.md 第 8 节）。
//
// 一个 warp（32 个线程）处理一对刚体 (a, b)，一个线程块 kPairWarps 对：
//   1. 用每个图元的单精度盒子（launch_prim_boxes 事先算好，向外取整）和对方刚体的盒子比，剔掉离得远的，
//      剩下的编号放进这个 warp 在共享内存里的六张表（用 __ballot_sync 压紧，不用原子操作）；
//   2. warp 的 32 个线程分摊三种组合（a 的顶点 x b 的三角形，b 的顶点 x a 的三角形，a 的边 x b 的边），
//      用单精度坐标（launch_prim_boxes 顺便存的）算一个大概的距离，交给调用者的 keep 决定要不要精确地算；
//   3. 要算的写进全局的候选数组（DCands）。精确判断（双精度）由另一个内核做，一个线程一个候选。
// 单精度只用来剔除：盒子向外取整，距离带余量 kFloatPad，只会多留不会漏；留下的都用双精度、按 CPU 版的规则判断，
// 所以结果和 CPU 版一样。这是上一个分支（Warp 版 RC-IPC）的做法：消费级显卡的双精度只有单精度的 1/64。
//
// 为什么一对刚体用一个 warp 而不是一个线程块：一堆布娃娃里每次有一千多对候选刚体，可剔完以后每对平均只剩几百个
// 组合、几个要精确算的对。一个线程块一对时，块里的同步和访存延迟占了绝大部分时间；一个 warp 一对只需要 __syncwarp，
// 共享内存也小得多，一个 SM 上能同时跑的对多得多。
// 为什么分成两个内核：精确判断（最近点、ACCD）要很多寄存器，放在剔除的内核里会把它的占用率拖下来
// （一个线程 140 多个寄存器，一个 SM 只能跑十几个 warp）；而且只有少数线程有候选，其余线程空等。
// 分开以后剔除的内核寄存器少、占用率高，精确判断的内核每个线程都有活干。
#include "devmath.cuh"
#include "device_model.h"

namespace rcipc::gpu {

constexpr int kPairWarps = 4;  // 一个线程块几个 warp（几对刚体）
constexpr int kListCap = 96;   // 每张表最多放这么多；放不下就退回那个刚体的全部图元
// 单精度距离的余量（米），乘以 (1 + R)，R 是这对刚体的坐标的最大绝对值。坐标的单精度舍入误差是 6e-8 R 的量级，
// 距离的误差也是这个量级，这里留了两个数量级以上的余地（上一个分支的 EPS32 也是 1e-5 米）。
constexpr float kFloatPad = 1e-5f;

// 一个 warp 的六张表：a 的顶点、b 的顶点、a 的三角形、b 的三角形、a 的边、b 的边（L = 0..5）。
struct WarpLists {
  int ids[6][kListCap];
};

__device__ inline int list_kind(int L) { return L < 2 ? 0 : (L < 4 ? 2 : 1); }  // 0 顶点、1 边、2 三角形

__device__ inline int prim_first(const DModel& m, int kind, int body) {
  return kind == 0 ? m.bvert0[body] : (kind == 1 ? m.bedge0[body] : m.bface0[body]);
}
__device__ inline int prim_count(const DModel& m, int kind, int body) {
  return kind == 0 ? m.bnvert[body] : (kind == 1 ? m.bnedge[body] : m.bnface[body]);
}
// 图元在 PrimBufs 的盒子数组里的位置：顶点、边、三角形依次排。
__device__ inline int prim_slot(const DModel& m, int kind, int id) {
  return kind == 0 ? id : (kind == 1 ? m.nvert + id : m.nvert + m.nedge + id);
}

// 单精度距离的余量：kFloatPad (1 + 两个刚体盒子的坐标的最大绝对值)。
__device__ inline float pair_pad(const double* boxes, const int body[2]) {
  double R = 0;
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < 6; ++k) R = fmax(R, fabs(boxes[6 * body[s] + k]));
  return kFloatPad * (1.0f + static_cast<float>(R));
}

// 填六张表：每张只留单精度盒子和对方刚体的盒子按 margin 重叠的图元。整个 warp 一起调用。
// 返回后 len[L] 是每张表的长度；某张放不下时 full[L] = true，那张表退回完整的范围。
__device__ inline void warp_fill(const DModel& m, const PrimBufs& P, const double* boxes, const int body[2],
                                 double margin, WarpLists& wl, bool full[6], int len[6]) {
  const int lane = threadIdx.x & 31;
  const unsigned below = (1u << lane) - 1;
  const float mf = __double2float_ru(margin);
  for (int L = 0; L < 6; ++L) {
    const int kind = list_kind(L), self = body[L % 2], other = body[1 - L % 2];
    const int first = prim_first(m, kind, self), num = prim_count(m, kind, self);
    const FBox ob = make_fbox(v3load(boxes + 6 * other), v3load(boxes + 6 * other + 3));
    int cnt = 0;  // 整个 warp 一样
    for (int i0 = 0; i0 < num; i0 += 32) {
      const int i = i0 + lane;
      const bool pass = i < num && fbox_overlap(fbox_load(P.fbox + 6 * prim_slot(m, kind, first + i)), ob, mf);
      const unsigned mask = __ballot_sync(0xffffffffu, pass);
      const int pos = cnt + __popc(mask & below);
      if (pass && pos < kListCap) wl.ids[L][pos] = first + i;
      cnt += __popc(mask);
    }
    full[L] = cnt > kListCap;
    len[L] = full[L] ? num : cnt;
  }
  __syncwarp();
}

__device__ inline int list_item(const DModel& m, const WarpLists& wl, const bool full[6], const int body[2], int L,
                                int i) {
  return full[L] ? prim_first(m, list_kind(L), body[L % 2]) + i : wl.ids[L][i];
}

// 把第 w 个组合拆成 (表 L1 的第 i1 个, 表 L2 的第 i2 个)。总数是 len[0] len[3] + len[1] len[2] + len[4] len[5]
// （假设不超过 2^32）。
__device__ inline void split_work(unsigned w, const int len[6], int& L1, int& L2, int& i1, int& i2) {
  const unsigned n1 = static_cast<unsigned>(len[0]) * len[3], n2 = static_cast<unsigned>(len[1]) * len[2];
  if (w < n1) {
    L1 = 0, L2 = 3, i1 = w / len[3], i2 = w % len[3];
  } else if (w < n1 + n2) {
    const unsigned u = w - n1;
    L1 = 1, L2 = 2, i1 = u / len[2], i2 = u % len[2];
  } else {
    const unsigned u = w - n1 - n2;
    L1 = 4, L2 = 5, i1 = u / len[5], i2 = u % len[5];
  }
}

__device__ inline unsigned work_count(const int len[6]) {
  return static_cast<unsigned>(len[0]) * len[3] + static_cast<unsigned>(len[1]) * len[2] +
         static_cast<unsigned>(len[4]) * len[5];
}

// 一对图元的半径之和（make_pt / make_ee / accd 用的那两个）。
__device__ inline double pair_radii(const DModel& m, int id1, int id2, bool ee) {
  return ee ? m.vradius[m.ev[2 * id1]] + m.vradius[m.ev[2 * id2]] : m.vradius[id1] + m.vradius[m.fv[3 * id2]];
}

__device__ inline F3 xf_load(const float* xf, int v) { return f3(xf + 3 * v); }

// 遍历三种组合，用单精度坐标算大概的距离 df（没减半径），keep(L1, id1, id2, df) 为真的写进候选数组 C。
// 整个 warp 一起调用（每轮 32 个组合，所有线程走同样多轮）。调用前要先 warp_fill。
// 写候选时每轮只做一次原子加：__ballot_sync 数出这一轮有几个，lane 0 一次占好位置，各线程按自己的名次写。
template <class Keep>
__device__ void warp_scan(const DModel& m, const PrimBufs& P, const WarpLists& wl, const bool full[6],
                          const int body[2], const int len[6], DCands C, Keep keep) {
  const int lane = threadIdx.x & 31;
  const unsigned below = (1u << lane) - 1;
  const unsigned nw = work_count(len);
  for (unsigned base = 0; base < nw; base += 32) {
    const unsigned w = base + lane;
    bool take = false, ee = false;
    int id1 = 0, id2 = 0;
    if (w < nw) {
      int L1, L2, i1, i2;
      split_work(w, len, L1, L2, i1, i2);
      id1 = list_item(m, wl, full, body, L1, i1);
      id2 = list_item(m, wl, full, body, L2, i2);
      ee = L1 == 4;
      float df;
      if (ee) {
        df = ee_dist_f(xf_load(P.xf, m.ev[2 * id1]), xf_load(P.xf, m.ev[2 * id1 + 1]), xf_load(P.xf, m.ev[2 * id2]),
                       xf_load(P.xf, m.ev[2 * id2 + 1]));
      } else {
        df = pt_dist_f(xf_load(P.xf, id1), xf_load(P.xf, m.fv[3 * id2]), xf_load(P.xf, m.fv[3 * id2 + 1]),
                       xf_load(P.xf, m.fv[3 * id2 + 2]));
      }
      take = keep(L1, id1, id2, df);
    }
    const unsigned mask = __ballot_sync(0xffffffffu, take);
    if (mask == 0) continue;
    int first = 0;
    if (lane == 0) first = atomicAdd(C.count, __popc(mask));
    first = __shfl_sync(0xffffffffu, first, 0);
    const int k = first + __popc(mask & below);
    if (take && k < C.cap) C.item[k] = DCand{id1, ee ? ~id2 : id2};
  }
}

// 精确判断的内核用：显存里的候选个数（不超过 cap）。一个线程一个候选，网格跨步，不用回主机读个数。
__device__ inline int cand_count(const DCands& C) { return min(*C.count, C.cap); }

}  // namespace rcipc::gpu
