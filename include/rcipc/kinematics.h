#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "rcipc/model.h"

// 关节坐标的运动学（阶段 2，笔记 notes/02_kinematics.md）。
//
// 记号：q 是位置（长度 nq，MuJoCo 的 qpos），delta 是切空间里的增量（长度 nv，和速度 qvel 同一个空间）。
namespace rcipc {

// 正运动学的结果：每个刚体在世界里的位姿，以及每个自由度在世界里的轴和锚点。
struct Frames {
  std::vector<Eigen::Vector3d> xpos;      // 每个刚体坐标系的原点（长度 nbody），同 MuJoCo 的 xpos
  std::vector<Eigen::Quaterniond> xquat;  // 每个刚体坐标系的姿态（长度 nbody），同 MuJoCo 的 xquat
  std::vector<Eigen::Vector3d> dof_axis;    // 每个自由度的轴 a_d（世界系，长度 nv）
  std::vector<Eigen::Vector3d> dof_anchor;  // 每个转动自由度的锚点 o_d（世界系，长度 nv；平移自由度用不到）
};

// 正运动学（第 3 节）：给定 q，从根往下逐层算出每个刚体的位姿和每个自由度的轴、锚点。
Frames forward_kinematics(const Model& model, const Eigen::VectorXd& q);

// 刚体 body 上一个点：local 是它在刚体坐标系里的坐标，返回世界坐标 x_b + R_b * local。
Eigen::Vector3d world_point(const Frames& frames, int body, const Eigen::Vector3d& local);

// 位置的“加法”（第 2 节）：返回 q ⊕ s * delta，同 MuJoCo 的 mj_integratePos(q, delta, s)。
//   铰链、滑动、自由关节的平移部分：直接相加；
//   球关节、自由关节的转动部分：quat * exp(s * delta_rot)（右乘，增量写在刚体自己的坐标系里）。
Eigen::VectorXd integrate(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& delta,
                          double s = 1.0);

// 位置的“减法”（第 2 节）：返回 q1 ⊖ q0，即满足 q0 ⊕ delta = q1 的 delta。
// 同 MuJoCo 的 mj_differentiatePos(qvel, dt = 1, q0, q1)。转动部分取最短的那个（长度 <= pi）。
Eigen::VectorXd differentiate(const Model& model, const Eigen::VectorXd& q0, const Eigen::VectorXd& q1);

// 点的 Jacobian（第 4 节）：刚体 body 上、世界坐标为 x 的点，对 delta 的导数，3 x nv。
//   第 d 列 = a_d x (x - o_d)   若 d 是转动自由度，并且在 body 到根的链上；
//           = a_d               若 d 是平移自由度，并且在链上；
//           = 0                 若 d 不在链上。
// 含义：沿 q ⊕ s * delta 运动时，这个点的速度 dx/ds（s = 0 处）= J * delta。
// frames 必须是在当前 q 上算的 forward_kinematics。
Eigen::Matrix3Xd point_jacobian(const Model& model, const Frames& frames, int body, const Eigen::Vector3d& x);

// 刚体的角速度 Jacobian（阶段 3 加的），3 x nv：沿 q ⊕ s * delta 运动时，刚体 body 的角速度（世界系）= W * delta。
//   第 d 列 = a_d   若 d 是链上的转动自由度；
//           = 0     若 d 是平移自由度，或者不在链上。
// 和 point_jacobian 一样沿父刚体往上走。它对应 mj_jac 的 jacr。
Eigen::Matrix3Xd angular_jacobian(const Model& model, const Frames& frames, int body);

}  // namespace rcipc
