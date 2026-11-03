#include "rcipc/chain.h"

#include <algorithm>
#include <cmath>

namespace rcipc {

std::vector<ChainJoint> chain_joints(const Model& model, const Eigen::VectorXd& q, int body) {
  // 沿链往上走，先按“从刚体到根”的顺序收集，最后反过来。
  // path 是“从当前走到的刚体原点到 body 的顶点”的路径长度，一路往上累加。
  std::vector<ChainJoint> out;
  double path = model.bodies[body].radius;
  for (int k = body; k != 0; k = model.bodies[k].parent) {
    const int j = model.bodies[k].joint;
    if (j >= 0) {
      const Joint& J = model.joints[j];
      ChainJoint cj;
      cj.joint = j;
      cj.rotation = J.type != JointType::kSlide;
      cj.dadr = J.type == JointType::kFree ? J.dadr + 3 : J.dadr;  // 自由关节：只要转动的 3 个
      cj.nv = J.type == JointType::kSlide || J.type == JointType::kHinge ? 1 : 3;
      // 滑动关节的锚点是没伸出去时的位置（FK 里 dof_anchor 在平移之前算），所以它自己的力臂也要加上现在的伸长。
      if (J.type == JointType::kSlide) path += std::abs(q[J.qadr] - J.ref);
      cj.reach = path + J.pos.norm();  // 锚点到 k 的原点，再沿路径到顶点
      out.push_back(cj);
    }
    path += model.bodies[k].pos.norm();  // 往上一段：k 的原点到父刚体原点的固定偏移
  }
  std::reverse(out.begin(), out.end());
  return out;
}

}  // namespace rcipc
