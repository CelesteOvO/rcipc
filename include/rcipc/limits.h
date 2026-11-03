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
// q 对关节增量是线性的（q + alpha p），所以限位的 CCD 有闭式解，和阶段 1 的 ccd_max_step 一样。
// 梯度和 Hessian 只落在那一个自由度上：Hessian 是对角的，进树块 H_k。
namespace rcipc {

double limit_energy(const Model& model, const Eigen::VectorXd& q, double k, double dhat);
Eigen::VectorXd limit_gradient(const Model& model, const Eigen::VectorXd& q, double k, double dhat);
Eigen::VectorXd limit_hessian_diagonal(const Model& model, const Eigen::VectorXd& q, double k, double dhat);

// 限位的 CCD：沿 q + s p，每个有限位的自由度最多闭合到两端距离的 c 倍。返回 alpha_max ∈ (0, 1]。
double limit_ccd(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p, double c);

// 到最近的限位的距离（弧度或米）；没有限位时 +无穷。每步结束后用它检查没有越界。
double min_limit_distance(const Model& model, const Eigen::VectorXd& q);

}  // namespace rcipc
