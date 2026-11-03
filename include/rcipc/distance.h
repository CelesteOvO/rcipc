#pragma once

#include <Eigen/Core>

// 图元之间的最近点（阶段 7，笔记 notes/07_contact.md 第 1 节，书 §3.3、§8.1）。
//
// 只做两种：点–三角形和线段–线段。它们覆盖了两个凸多面体之间所有可能的最近点组合
// （点–点、点–边是它们的特殊情形：最近点落在三角形的边、角上，或线段的端点上）。
// 返回的是最近点的参数（重心坐标、线段参数），不开方；距离由调用者从两个最近点之差算。
namespace rcipc {

// 点 p 到三角形 (a, b, c) 的最近点，用重心坐标表示：closest = lam[0] a + lam[1] b + lam[2] c，lam >= 0 且和为 1。
// 算法是按 Voronoi 区域分情况（Ericson《Real-Time Collision Detection》5.1.5）：
// 先看三个角，再看三条边，最后才是面内。
Eigen::Vector3d point_triangle_closest(const Eigen::Vector3d& p, const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                                       const Eigen::Vector3d& c);

// 线段 (p0, p1) 和线段 (q0, q1) 之间的最近点对：p0 + s (p1 - p0) 和 q0 + t (q1 - q0)，s、t 在 [0, 1] 里。
// 线段可以退化成一个点（p0 == p1：球心），这时 s = 0。平行时取其中一对最近点（距离仍然正确）。
// 算法：先解无约束的两条直线，再把 s 夹到 [0, 1] 并重新解 t，再把 t 夹回去并重新解 s（Ericson 5.1.9）。
void segment_segment_closest(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& q0,
                             const Eigen::Vector3d& q1, double& s, double& t);

}  // namespace rcipc
