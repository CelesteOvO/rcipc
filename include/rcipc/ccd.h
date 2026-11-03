#pragma once

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

// 曲线 CCD（阶段 5、7）：弦加矢高。把 [0, 1] 切成若干段，每段长 ds，刚体 b 上的顶点在段内偏离弦不超过矢高
//   xi_b = A_b ds^2 / 8，
// 于是真实距离 >= 弦上的距离 - xi（一对接触的 xi 是两侧刚体的 xi 之和）。沿弦做保守推进，把“距离”换成
// “弦上的距离 - xi”：
//   顶点–平面（阶段 5）：弦上距离是线性的，闭式解 t = c (y_a - xi) / (y_a - y_b)；
//   点–三角形、边–边（阶段 7，书 §9.2 的 ACCD）：弦上距离不是线性的，但它的变化率不超过两侧相对位移之和 l，
//     反复推进 t <- t + c (d(t) - xi) / l，直到 t >= 1 或 d(t) - xi <= (1 - c) (d(0) - xi)。
// 段长先取 s_end = min(1, sqrt(8 dhat / A_max))（最大矢高不超过 dhat，相当于一个信赖域）；
// 某对的 xi 超过它当前距离的一半时，这段太长，缩短后重算。宽相：两侧刚体在段两端的包围盒按 xi 放大后不重叠的对跳过。
// 返回 alpha_max ∈ (0, 1]。
double curved_ccd(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p, double dhat, double c);

// 沿 q ⊕ s p（s 从 0 到 alpha 等分 samples 段）做 FK，返回路上所有候选对的最小距离。
// 它是独立于 CCD 的检查：CCD 说安全的那段路，真实轨迹是不是真的没碰到。测试和演示程序用它抓穿透。
double trajectory_min_distance(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p, double alpha,
                               int samples);

}  // namespace rcipc
