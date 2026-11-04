#pragma once

// CUDA graph 版的 Newton 迭代里，设备上做的判断（iter.cu）。每个函数发一个小内核，读写显存里的 DIter；
// h 是条件节点（WHILE）的句柄，不为 0 时内核把“循环要不要继续”写进去（逐个发射时传 0，主机读 it->cont）。
#include <cuda_runtime.h>

#include "device_model.h"

namespace rcipc::gpu {

// 组装之前：nrow = nc + 2 nf，计数清零。
void launch_it_begin(DIter* it);
// 收敛判断：读组装的能量、PCG 的次数、整步的最大位移（scal[0]、scal[12]、scal[7]）。
void launch_it_converge(DIter* it, const double* scal, double h, double tol);
// CCD 的开头：第一段的长度（A 是每个刚体的加速度界），scal[3] = 1；没收敛时让 CCD 的循环开始。
void launch_ccd_setup(DIter* it, const double* A, int nbody, double dhat, double* scal, cudaGraphConditionalHandle h);
// 一段的开始：it->s = 段终点，it->len = 段长。
void launch_ccd_seg_begin(DIter* it);
// 一段的结束：缩短重做、截断、或者前进到下一段；放不下时标记溢出。
void launch_ccd_seg_end(DIter* it, const double* scal, int pair_cap, int work_cap, int cand_cap,
                        cudaGraphConditionalHandle h);
// it->adv 时把 from 拷到 to（下一段的起点）。
void launch_copy_frames_if(const DIter* it, const DFrames& from, const DFrames& to, int nbody, int nv, int nvert);
// 线搜索的开头：alpha = alpha_max = min(CCD 的截断位置, 限位 CCD)。
void launch_ls_setup(DIter* it, const double* scal, cudaGraphConditionalHandle h);
// 一次试探之后：能量没降就折半再试；检测放不下时标记溢出。
void launch_ls_end(DIter* it, const double* scal, const int* cnt, int pair_cap, int contact_cap, int cand_cap,
                   int work_cap, cudaGraphConditionalHandle h);
// 线搜索成功时接受试探点：q = qtrial，to = from（FK），nc = cnt[1]。
void launch_it_accept(DIter* it, const double* qtrial, double* q, int nq, const DFrames& from, const DFrames& to,
                      int nbody, int nv, int nvert, const int* cnt);

}  // namespace rcipc::gpu
