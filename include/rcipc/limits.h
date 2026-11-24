#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/contact.h"
#include "rcipc/model.h"

// 关节限位（阶段 10，笔记 notes/10_scenes.md 第 1 节，书第 216 页的 E_limit）。
//
// 铰链和滑动关节的 q 必须保持在 (lo, hi) 里。它和接触是同一类问题（一个不等式约束），所以用同一个办法：
// 到两端的距离 d_lo = q - lo、d_hi = hi - q 各放一个障碍，
//
//   E_limit = k sum_j [ b(d_lo, dhat) + b(d_hi, dhat) ]
//
// b 是阶段 1 的障碍函数，这里直接用距离（弧度或米）做自变量，不用平方（一维里不需要）。
//
// 激活距离 dhat 和距离同单位，所以按关节类型取（LimitBarrier）：
//   滑动关节的 q 是米，到端点的距离和接触的间隙是同一种量（一个刚体沿一条轴靠近一个挡块），用接触的 dhat；
//   铰链的 q 是弧度，没有通用的长度换算，用它自己的 StepParams::limit_dhat（弧度）。
// 原来两种关节共用 limit_dhat = 0.1：对铰链是 5.7 度，对滑动关节是 10 cm。Panda 的手指行程只有 4 cm，
// 两端的障碍在整个行程里都是活跃的，张开的目标 0.04 被顶回 0.0277（notes/22_primal_dual.md 第 5 节）。
// q 对关节增量是线性的（q + alpha p），所以限位的 CCD 有闭式解，和阶段 1 的 ccd_max_step 一样。
// 梯度和 Hessian 只落在那一个自由度上：Hessian 是对角的，进树块 H_k。
namespace rcipc {

// 限位障碍的参数。k 是刚度（铰链 J/rad^2，滑动 J/m^2）；激活距离按关节类型取（见上）。
struct LimitBarrier {
  double k = 10.0;
  double dhat_hinge = 0.1;   // 铰链，rad（StepParams::limit_dhat）
  double dhat_slide = 1e-3;  // 滑动，m（StepParams::dhat，和接触同一个）
  double dhat(JointType type) const { return type == JointType::kSlide ? dhat_slide : dhat_hinge; }
};

double limit_energy(const Model& model, const Eigen::VectorXd& q, const LimitBarrier& L);
Eigen::VectorXd limit_gradient(const Model& model, const Eigen::VectorXd& q, const LimitBarrier& L);
Eigen::VectorXd limit_hessian_diagonal(const Model& model, const Eigen::VectorXd& q, const LimitBarrier& L);

// 有没有活跃的限位：某个关节到某一端的距离小于它自己的激活距离。
bool limits_active(const Model& model, const Eigen::VectorXd& q, const LimitBarrier& L);

// 限位的 CCD：沿 q + s p，每个有限位的自由度最多闭合到两端距离减 sep 的 c 倍（sep 是最小分离，同 curved_ccd）。
// 返回 alpha_max ∈ [0, 1]：距离已经 <= sep 而且还往那一端走时是 0（q 沿 p 是直线，往回走总是安全的）。
double limit_ccd(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p, double c, double sep = 1e-12);

// 到最近的限位的距离（弧度和米混在一起取最小，只用来检查没有越界：它 > 0 就是都在范围里）；没有限位时 +无穷。
// 判断限位是否活跃要按关节各自的激活距离比，用 limits_active。
double min_limit_distance(const Model& model, const Eigen::VectorXd& q);

}  // namespace rcipc
