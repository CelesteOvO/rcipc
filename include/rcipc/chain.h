#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/model.h"

// 一个刚体到根的链上的关节，以及每个关节到这个刚体的静态力臂（阶段 5 的加速度界和阶段 6 的曲率上界都用它）。
namespace rcipc {

struct ChainJoint {
  int joint;      // 关节编号
  int dadr;       // 这一组自由度在 delta 里的起始下标：转动组是转动自由度（自由关节是后 3 个），滑动是它自己
  int nv;         // 这一组有几个自由度：铰链 1，球关节 3，自由关节的转动 3，滑动 1
  bool rotation;  // 转动组还是滑动
  double reach;   // 静态力臂 R：从这个关节的锚点沿链走到刚体顶点的路径长度（m），在任何姿态下都 >= 真实距离
};

// 刚体 body 到根的链上的关节，从根到刚体的顺序。自由关节只算它的转动部分（平移是直线，不产生曲率）。
// 静态力臂 = |jnt_pos|（锚点到关节所在刚体的原点）+ 链上每段 body_pos 的长度 + 滑动关节在 q 处的伸长 + 刚体半径。
// 这些都是固定长度的线段首尾相接，三角不等式保证任何姿态下 |顶点 - 锚点| 都不超过它们之和。
std::vector<ChainJoint> chain_joints(const Model& model, const Eigen::VectorXd& q, int body);

}  // namespace rcipc
