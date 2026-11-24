#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"

// 连续碰撞检测（CCD）。阶段 4 的直弦版本（笔记 notes/04_ground.md 第 5 节，书 §9.5）、
// 阶段 5 的曲线版本：弦加矢高（notes/05_curved_ccd.md，书 §9.2–9.4），阶段 7 扩展到刚体之间（notes/07_contact.md 第 4 节）。
//
// 问题：Newton 想从 q 走到 q ⊕ p。线搜索只看终点的能量，而终点可能已经在障碍的另一侧（能量又是有限的）。
// 所以走之前先问：沿 q ⊕ s p（s 从 0 到 1）走，最多能走到哪里不穿透？
namespace rcipc {

// 直弦 CCD（阶段 4，只管顶点–平面，留着做对比）：假装每个顶点从 s = 0 的位置沿直线走到 s = 1 的位置。
//   平面距离沿直线是线性的：y(s) = y_a + s (y_b - y_a)。按阶段 1 的保守推进规则，
//   每个顶点最多闭合当前距离的 c 倍：y(s) >= (1 - c) y_a，即 s <= c y_a / (y_a - y_b)（y_b < y_a 时）。
//   返回所有 (顶点, 平面) 对的最小值，再和 1 取较小。远离平面的顶点不限制。
// a 是 q 处的 FK，b 是 q ⊕ p 处的 FK。前提：a 处所有距离 > 0。
//
// 它会漏什么：关节转动时顶点走的是弧线，弧线可能在中间穿过平面而两端都在上面（阶段 4 笔记第 5 节的反例）。
double chord_ccd_ground(const Model& model, const Frames& a, const Frames& b, double c);

// 每个刚体的加速度上界（阶段 5，笔记第 2 节，书 §9.3.1 的静态界）：沿 q ⊕ s p 走时，刚体 b 上任何一点的
// |x''(s)| 都不超过 A_b（单位 m，按 s 计）。用的是静态力臂（固定长度的线段之和），所以对整段 s 都成立。
//   A_b = sum_j w_j ( W_{<j} R_j + sum_{k >= j} (w_k R_k + u_k) ) + sum_s W_{<s} u_s
// j 遍历 b 到根的链上的转动组（铰链 1 个自由度，球关节和自由关节的转动 3 个），w_j = |p 在这组上的分量|，
// W_{<j} 是更靠近根的各组的 w 之和，R_j 是从关节 j 的锚点沿链走到 b 的顶点的路径长度（静态力臂）。
// 滑动关节 s 的速度 u_s = |p_s| 进两处：它被上游转动带着转（W_{<s} u_s），它改变下游点相对上游锚点的速度。
// 单个铰链时 A = w^2 R，就是向心加速度。自由关节的平移是直线，不贡献。q 只用来算滑动关节的当前伸长。
Eigen::VectorXd acceleration_bounds(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p);

// 紧的加速度界（阶段 5 笔记第 6 节；想法来自书 §9.3.2）：A_b = max_v |x_v''(0)| + T_b。s = 0 处的二阶导数沿链精确递推，
// T_b 是整段上 |x'''| 的界（逐层的递推，用现在姿态下的力臂：刚体原点到铰链轴、到锚点的距离，加刚体半径）。
// F 是 q 处的 FK。链上有滑动关节的刚体不算，返回 +无穷。
Eigen::VectorXd tight_acceleration_bounds(const Model& model, const Frames& F, const Eigen::VectorXd& p);

// 紧界的两部分分开给：a0_b = max_v |x_v''(0)|，T_b = |x'''| 的界，所以 |x''(s)| <= a0_b + T_b s（s 从 F 的姿态量起）。
// tight_acceleration_bounds 是 a0 + T（s 取到 1）。链上有滑动关节的刚体两部分都是 +无穷。
void tight_acceleration_parts(const Model& model, const Frames& F, const Eigen::VectorXd& p, Eigen::VectorXd* a0,
                              Eigen::VectorXd* T3);

// T_b 的中间量（只给测试用）。T_b 的证明（笔记第 6.1–6.4 节）说的是：下面每个量在整段 s ∈ [0, 1] 上
// 盖住真实轨迹上的一个量；测试 Acceleration.TightTerms* 拿真实轨迹逐个比。每个转速不为 0 的关节一组。
//   o_j 是关节 j 的锚点，w_j 是它的转动向量（世界系），x 是刚体 b 上的任意一点；
//   “冻住上游”= 把 j 之前（更靠近根）的关节都冻住，只让 j 和它下游的关节动。
struct TightTerm {
  int dadr;      // 这个关节的转动自由度在 delta 里的起始下标（自由关节是后 3 个）
  int nv;        // 铰链 1，球关节和自由关节的转动 3
  double rate;   // 转速 th_j = |p 在这一组上的分量|
  double speed;  // c_j >= |w_j x (x - o_j)|：这个关节单独带给点的速度
  double v;      // v_j：冻住上游时点的速度的界
  double a;      // a_j：冻住上游时点的加速度的界
  double t;      // t_j：冻住上游时点的三阶导数的界；最靠近根的那个关节的 t 就是 T_b
};

// 刚体 body 的链上各关节的中间量，从根到刚体的顺序。链上有滑动关节时返回空（紧界不算这样的刚体）。
std::vector<TightTerm> tight_terms(const Model& model, const Frames& F, const Eigen::VectorXd& p, int body);

// curved_ccd 用的界：静态界和紧界逐个刚体取较小的那个（两个都对整段成立）。F 是 q 处的 FK。
Eigen::VectorXd ccd_acceleration_bounds(const Model& model, const Eigen::VectorXd& q, const Frames& F,
                                        const Eigen::VectorXd& p);

// 一对图元沿弦的 ACCD（只给测试用）：x0、x1 是 4 个顶点在这一段两端的位置（点–三角形 (p, a, b, c)，
//   边–边 (p0, p1, q0, q1)），radii 是要减掉的半径之和，xi 是矢高。返回安全比例 t ∈ [0, 1]；
//   iters、capped 是走了几步、有没有走满上限（1000 步）。xi > d0 / 2 时（curved_ccd 会缩短这一段）返回 -1。
double accd_chord(const Eigen::Vector3d x0[4], const Eigen::Vector3d x1[4], bool pt, double radii, double xi, double c,
                  int* iters = nullptr, bool* capped = nullptr);

// 曲线 CCD（阶段 5、7）：弦加矢高。把 [0, 1] 切成若干段，每段长 ds，刚体 b 上的顶点在段内偏离弦不超过矢高
//   xi_b = A_b ds^2 / 8，
// 于是真实距离 >= 弦上的距离 - xi（一对接触的 xi 是两侧刚体的 xi 之和）。沿弦做保守推进，把“距离”换成
// “弦上的距离 - xi”：
//   顶点–平面（阶段 5）：弦上距离是线性的，闭式解 t = c (y_a - xi) / (y_a - y_b)；
//   点–三角形、边–边（阶段 7，书 §9.2 的 ACCD）：弦上距离不是线性的，但它的变化率不超过两侧相对位移之和 l，
//     反复推进 t <- t + c (d(t) - xi) / l，直到 t >= 1 或 d(t) - xi <= (1 - c) (d(0) - xi)。
//     每步再用分离平面的界（笔记 notes/22_primal_dual.md 第 3 节）：一个分离方向上两侧顶点沿弦的投影给出距离的下界，
//     贴着表面滑动时它整段成立，而相对位移的界每步只走 c (d - xi) / l。两个都成立，取走得远的那个。
// 段长先取 s_end = min(1, sqrt(8 dhat / A_max))（最大矢高不超过 dhat，相当于一个信赖域）；
// 某对的 xi 超过它当前距离的一半时，这段太长，缩短后重算。宽相：两侧刚体在段两端的包围盒按 xi 放大后不重叠的对跳过。
// 返回 alpha_max ∈ [0, 1]。
// tight = false 时只用静态界（acceleration_bounds），true 时用 ccd_acceleration_bounds。
//
// sep（最小分离，笔记 22 第 5 节）：上面所有的“距离”都换成 d - sep，宽相的盒子余量也加上 sep。
// 于是截断点上真实距离 >= sep + (1 - c)(d0 - sep - xi) > sep。一对的起点距离 d0 <= sep 时，弦加矢高给不出任何保证，
// 这一对返回 0（卡住）。原来 d0 <= 0 时返回 1（放行）：浮点把 1e-16 m 的间隙算成负数时，整步就穿过去了。
double curved_ccd(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p, double dhat, double c,
                  bool tight = true, double sep = 1e-12);

// 沿 q ⊕ s p（s 从 0 到 alpha 等分 samples 段）做 FK，返回路上所有候选对的最小距离。
// 它是独立于 CCD 的检查：CCD 说安全的那段路，真实轨迹是不是真的没碰到。测试和演示程序用它抓穿透。
double trajectory_min_distance(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p, double alpha,
                               int samples);

}  // namespace rcipc
