#pragma once

#include <Eigen/Core>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"

// 连续碰撞检测（CCD）。阶段 4 先做直弦版本（笔记 notes/04_ground.md 第 5 节，书 §9.5），
// 阶段 5 再加上弧线偏离弦的余量（矢高）。
//
// 问题：Newton 想从 q 走到 q ⊕ p。线搜索只看终点的能量，而终点可能已经在障碍的另一侧（能量又是有限的）。
// 所以走之前先问：沿 q ⊕ s p（s 从 0 到 1）走，最多能走到哪里不穿透？
namespace rcipc {

// 直弦 CCD（阶段 4）：假装每个顶点从 s = 0 的位置沿直线走到 s = 1 的位置。
//   平面距离沿直线是线性的：y(s) = y_a + s (y_b - y_a)。按阶段 1 的保守推进规则，
//   每个顶点最多闭合当前距离的 c 倍：y(s) >= (1 - c) y_a，即 s <= c y_a / (y_a - y_b)（y_b < y_a 时）。
//   返回所有 (顶点, 平面) 对的最小值，再和 1 取较小。远离平面的顶点不限制。
// a 是 q 处的 FK，b 是 q ⊕ p 处的 FK。前提：a 处所有距离 > 0。
//
// 它会漏什么：关节转动时顶点走的是弧线，弧线可能在中间穿过平面而两端都在上面（笔记第 5 节的反例）。
double chord_ccd_ground(const Model& model, const Frames& a, const Frames& b, double c);

// 沿 q ⊕ s p（s 从 0 到 alpha 等分 samples 段）做 FK，返回路上所有 (顶点, 平面) 对的最小距离。
// 它是独立于 CCD 的检查：CCD 说安全的那段路，真实轨迹是不是真的没碰到。测试和演示程序用它抓穿透。
double trajectory_min_distance(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p, double alpha,
                               int samples);

}  // namespace rcipc
