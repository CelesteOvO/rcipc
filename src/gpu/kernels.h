#pragma once

// GPU 版的内核启动函数（阶段 11）。实现在 kin.cu、contact.cu、ccd.cu、solver.cu。
//
// 约定：
//   - 所有指针都指向显存，都在默认流上排队；需要把结果读回主机的函数会同步，并在注释里说明；
//   - 随 Newton 迭代变化的标量和个数用 DScalar / DCount（device_model.h）：逐个发射时直接传值，
//     CUDA graph 版传显存里的地址（图录好以后参数不能再改）；
//   - “累加”的输出（能量、梯度、树块）用原子加累加到调用者清零过的缓冲区里，所以几个函数可以往同一个缓冲区里加；
//   - 每个函数都对应 CPU 版的一个函数，名字和注释里写了是哪一个。
#include "device_model.h"

namespace rcipc::gpu {

// ---------- kin.cu：运动学、惯性项、关节限位 ----------

// out = q ⊕ s p（integrate）。
void launch_integrate(const DModel& m, const double* q, const double* p, DScalar s, double* out);
// 正运动学和顶点的世界坐标（forward_kinematics + world_point）。
void launch_fk(const DModel& m, const double* q, const DFrames& F);
// 惯性目标（inertia_targets）：tc nbody x 3，tA nbody x 9。
void launch_targets(const DModel& m, const DFrames& now, const DFrames& prev, double h, double* tc, double* tA);
// 惯性能量（inertia_energy）累加到 *e；梯度（inertia_gradient）累加到 g。
void launch_inertia_energy(const DModel& m, const DFrames& F, const double* tc, const double* tA, double h, double* e);
void launch_inertia_gradient(const DModel& m, const DFrames& F, const double* tc, const double* tA, double h, double* g);
// 树块 M / h^2（inertia_hessian），覆盖写 Hk（ntree x TD x TD）。
void launch_inertia_hessian(const DModel& m, const DFrames& F, double h, double* Hk);
// 关节限位（limits.cc）：能量累加到 *e、梯度累加到 g、对角累加到 Hk（哪个为空就不算）。
void launch_limits(const DModel& m, const double* q, double k, double dhat, double* e, double* g, double* Hk);
// 限位的 CCD（limit_ccd）：结果 atomic_min 到 *tmin（调用者先设成 1）。
void launch_limit_ccd(const DModel& m, const double* q, const double* p, double c, double* tmin);
// *p = v（一个线程的内核）。用它代替从主机拷一个数过去：那样的拷贝要等前面的内核都做完。
void launch_fill(double* p, double v);
// 两组顶点坐标之间的最大位移，atomic_max 到 *out（调用者先设成 0）。
void launch_max_displacement(const double* Xa, const double* Xb, int n, double* out);

// ---------- contact.cc 对应的部分：contact.cu ----------

// 每个刚体的包围盒（nbody x 6：lo, hi），顶点按半径放大；没有顶点的刚体是空盒（lo > hi）。
void launch_body_boxes(const DModel& m, const double* X, double* boxes);
// 扫掠盒：顶点在 Xa 和 Xb 两个位置的包围盒（CCD 用）。
void launch_swept_boxes(const DModel& m, const double* Xa, const double* Xb, double* boxes);
// 候选刚体对：collide(a, b) 并且两个盒子按 margin（A 不为空时再加两侧的矢高 A len^2 / 8）放大后重叠。
// 写到 pairs（每对两个 int），个数写到显存里的 *d_count（不读回主机）；个数可能大于 cap，调用者事后检查、扩容重来。
void launch_body_pairs(const DModel& m, const double* boxes, double margin, const double* A, DScalar len, int* pairs,
                       int cap, int* d_count);
// 接触检测（find_contacts）：顶点–平面，加上候选刚体对之间的点–三角形、边–边。只留 d < dmax 的。
// 返回总数（同步）；可能大于 cap，调用者扩容后重来。
// 接触检测：F 是这个位置的 FK（要 X、xpos、xquat）。d_cnt 是显存里的 4 个计数：[0] 是 launch_body_pairs 数的刚体对，
// [1] 接触、[2] 候选（C.count 要指向 d_cnt + 2）、[3] 剔除的工作项（N.work_n 要指向 d_cnt + 3）。
// 四个计数一次读回 h_cnt（同步）；任何一个超过容量时，调用者扩容后重来（这时接触不全）。
// pair_cap = 0 表示没有刚体对（模型里没有边和三角形）。
void launch_detect(const DModel& m, const DFrames& F, const double* boxes, const int* pairs, int pair_cap,
                   double dmax, DNarrow N, DCands C, DContact* out, int cap, int* d_cnt, int h_cnt[4]);
// 同上，但不读回：四个计数留在 d_cnt 里（CUDA graph 版由设备上的小内核检查是否放得下）。
void launch_detect_async(const DModel& m, const DFrames& F, const double* boxes, const int* pairs, int pair_cap,
                         double dmax, DNarrow N, DCands C, DContact* out, int cap, int* d_cnt);
// 窄相的剔除（narrow.cu）：候选刚体对里，在刚体坐标系里剔掉“距离一定不小于阈值”的图元对，剩下的写进 C。
// 检测时 A 为空，阈值是 dmax；CCD 时阈值是 max(2 xi, xi + 4 Dmax / c)（A 是加速度界，D 是每个刚体的最大位移）。
// 都再加上两个图元的半径。F 是剔除用的位置的 FK（CCD 时是段起点）。window 见 launch_ccd_cull。
void launch_narrow(const DModel& m, const DFrames& F, const double* boxes, const int* pairs, const int* npairs_d,
                   int pair_cap, double dmax, const double* A, const double* D, DScalar len, double c, DNarrow N,
                   DCands C, bool window = false);
// out[i] = (float) x[i]，i < n（CCD 的单精度扫掠盒用）。
void launch_to_float(const double* x, int n, float* out);
// 接触的各项（contact_energy、contact_gradient、contact_rows、contact_curvature）：
//   能量累加到 *e，梯度累加到 g，曲率上界累加到 Hk 的对角，行写到 rows 的第 0..n-1 行（哪个为空就不算）。
void launch_contact_terms(const DModel& m, const DFrames& F, const double* q, const DContact* c, DCount n, DScalar kappa,
                          double dhat, double curv_scale, double* e, double* g, double* Hk, DRows rows);
// 测试用：把行展开成逐个自由度的值（每行 2 CH 个槽，用不到的 idx = -1）。
void launch_rows_expand(const DModel& m, const DFrames& F, DRows rows, int nrow, int* idx, double* val);
// 接触里最小的距离，atomic_min 到 *out（调用者先设成 +无穷）。
void launch_min_distance(const DContact* c, DCount n, double* out);
// 滞后摩擦（lag_friction）：返回摩擦接触的个数（同步）。out 至少要 n 个。
int launch_friction_lag(const DModel& m, const double* X0, const DContact* c, int n, double kappa, double dhat,
                        DFriction* out, int* d_count);
// 摩擦的各项（friction_energy、friction_gradient、friction_rows）：行写到 rows 的第 row0.. row0 + 2 nf - 1 行。
void launch_friction_terms(const DModel& m, const DFrames& F, const DFriction* f, DCount nf, double h, double mu,
                           double eps_v, double* e, double* g, DRows rows, DCount row0);

// ---------- ccd.cc 对应的部分：ccd.cu ----------

// 每个刚体的加速度上界（ccd_acceleration_bounds：静态界和紧界取小），A 长度 nbody。F 是 q 处的 FK。
// tight = false 时只用静态界。
void launch_accel_bounds(const DModel& m, const double* q, const DFrames& F, const double* p, bool tight, double* A);
// 一段 [s0, s0 + len] 里的 CCD（curved_ccd 的第 2 步）：顶点–平面的闭式解，和候选刚体对之间的逐对 ACCD。
//   段内的截断位置 atomic_min 到 *tmin（调用者先设成 1），需要的更短段长 atomic_min 到 *ds_need（先设成 len）。
void launch_ground_ccd(const DModel& m, const double* Xa, const double* Xb, const double* A, DScalar len, double c,
                       double* tmin, double* ds_need);
// 刚体之间的 CCD 分两步。
// 1. 剔除（launch_narrow）：Fa 是段起点的 FK，boxes 是这一段每个刚体的扫掠盒，D 是 launch_body_disp 算的每个刚体的
//    最大位移，npairs_d 是显存里的刚体对个数（launch_body_pairs）。window 时 [段起点, 段起点 + len] 当作窗口剔，
//    候选盖住窗口里的任何一段（这时刚体对的余量也要放大：段长按 kWindowPairLen len 传给 launch_body_pairs，笔记第 13 节）。
void launch_ccd_cull(const DModel& m, const DFrames& Fa, const double* boxes, const double* D, const double* A,
                     DScalar len, double c, const int* pairs, const int* npairs_d, int pair_cap, DNarrow N, DCands C,
                     bool window);
// 窗口的刚体对：余量 2 xi 要求段长放大 sqrt(2) 倍，再多留一点盖住舍入。
constexpr double kWindowPairLen = 1.42;
// 2. 复核和 ACCD：C 里的候选（上一次剔除留下的，可以是窗口的）在这一段 [Xa, Xb] 上做，boxes 是这一段的扫掠盒。
//    seen[0..2] 写候选、刚体对、工作项的总数（double），超过容量时调用者扩容后重做这一段。
//    tail.split < kAccdMax 时，ACCD 走了 split 步还没完的对由整个 warp 一起先试着证明它返回 1，
//    证不出来再接着走；结果不变（笔记第 14 节）。
void launch_ccd_exact(const DModel& m, const double* Xa, const double* Xb, const double* boxes, const double* A,
                      DScalar len, double c, const int* npairs_d, int pair_cap, DNarrow N, DCands C, DTail tail,
                      double* tmin, double* ds_need, double* seen);
// ACCD 最多走几步（和 CPU 版的 accd 一样），和默认先串行走几步再交给尾巴的内核。
constexpr int kAccdMax = 1000;
constexpr int kAccdSplit = 16;
// 一段 CCD 的初值：*tmin = 1，*ds_need = len，seen[0..1] = 0（一个小内核，不用同步的拷贝）。
void launch_ccd_init(double* tmin, double* ds_need, double* seen, DScalar len);
// 每个刚体的顶点从 Xa 到 Xb 的最大位移，写到 D（nbody 个，先清零）。
void launch_body_disp(const DModel& m, const double* Xa, const double* Xb, double* D);

// ---------- solver.cc 对应的部分：solver.cu ----------

// y = H p，H = 树块 + sum D r^T r（apply）。行按力旋量算（rows.cuh），要 F（当前位置的 FK）和
// 18 nbody 个 double 的工作区 bw（速度旋量、刚体上的力旋量、子树之和）。
void launch_matvec(const DModel& m, const DFrames& F, const double* Hk, DRows rows, int nrow, const double* p,
                   double* y, double* bw);
// 树块预条件器（build_preconditioner）：分解以后显式求逆。P 至少 2 ntree TD^2 个 double：前一半出来时是
// 每棵树的 P_t^-1 = S (L D L^T)^-1 S（ntree x TD x TD），后一半是工作区。scale 是 Jacobi 缩放（nv）。
// W 是行的那部分用的工作区（Wb 至少 21 nbody 个，cross 至少 nrow 个）。
void launch_precond_build(const DModel& m, const DFrames& F, const double* Hk, DRows rows, DCount nrow, double* P,
                          double* scale, const DPrecondWork& W);
// z = P^-1 r（树块对角的稠密矩阵乘向量）。
void launch_precond_apply(const DModel& m, const double* P, const double* r, double* z);
// PCG（pcg）：解 H x = rhs。work 至少 4 nv + 18 nbody 个 double，scal 至少 4 个。返回迭代次数，
// *converged 写是否收敛。
int run_pcg(const DModel& m, const DFrames& F, const double* Hk, DRows rows, int nrow, const double* P,
            const double* rhs, double* x, double tol, int max_iter, double* work, double* scal, bool* converged);
// 同一个 PCG，整个循环放在一个协作内核里（pcg_coop.cu），每次迭代不用回主机。work 同 run_pcg，
// acc 至少 2 max_iter + 4 个 double。设备不支持协作启动时 pcg_coop_supported() 返回 false。
bool pcg_coop_supported();
// 和 run_pcg 不同，这里不等结果：迭代次数和是否收敛写到显存里的 res[0]、res[1]（double），调用者以后和别的数一起读回。
// 返回 0；树太大、放不进共享内存时返回 -1（什么也没做）。
int run_pcg_coop(const DModel& m, const DFrames& F, const double* Hk, DRows rows, DCount nrow, const double* P,
                 const double* rhs, double* x, double tol, int max_iter, double* work, double* acc, double* res);
// out += a . b（点积，原子加到 *out）。
void launch_dot(const double* a, const double* b, int n, double* out);
// x *= s。
void launch_scale(double* x, int n, double s);

}  // namespace rcipc::gpu
