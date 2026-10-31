#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

// 四元数的小工具（阶段 2，笔记 notes/02_kinematics.md 第 2 节）。
//
// 四元数直接用 Eigen::Quaterniond：
//   - 乘法 a * b 是 Hamilton 乘积，和 MuJoCo 的 mju_mulQuat 一样；
//   - q * v（v 是 Vector3d）是用 q 转动向量 v，和 mju_rotVecQuat 一样；
//   - q.conjugate() 是共轭，单位四元数的共轭就是逆；q.normalized() 归一化。
//
// 要小心的只有一处：分量的顺序。
//   - MuJoCo 的数组顺序是 (w, x, y, z)；
//   - Eigen 的构造函数 Quaterniond(w, x, y, z) 也是这个顺序，
//     但 q.coeffs() 返回的是 (x, y, z, w)。
// 所以和 MuJoCo 的数组互相转换时，一律用下面的 quat_load 和 quat_store，不要直接碰 coeffs()。
namespace rcipc {

// 从 MuJoCo 顺序 (w, x, y, z) 的数组读一个四元数（不归一化）。
inline Eigen::Quaterniond quat_load(const double* wxyz) {
  return Eigen::Quaterniond(wxyz[0], wxyz[1], wxyz[2], wxyz[3]);
}

// 把四元数按 MuJoCo 顺序 (w, x, y, z) 写进数组。
inline void quat_store(const Eigen::Quaterniond& q, double* wxyz) {
  wxyz[0] = q.w();
  wxyz[1] = q.x();
  wxyz[2] = q.y();
  wxyz[3] = q.z();
}

// 指数映射：旋转向量 w（方向是转轴，长度是转角，单位 rad）-> 单位四元数。
//
//   exp(w) = ( cos(|w|/2),  sin(|w|/2) * w / |w| )
//
// |w| = 0 时按极限取单位四元数 (1, 0, 0, 0)。
Eigen::Quaterniond quat_exp(const Eigen::Vector3d& w);

// 对数映射：exp 的逆。返回旋转向量 w，满足 exp(w) 和 q 表示同一个转动，并且 |w| <= pi
// （取“最短”的那个转动；q 和 -q 返回同一个结果）。q 不必是单位四元数。
Eigen::Vector3d quat_log(const Eigen::Quaterniond& q);

}  // namespace rcipc
