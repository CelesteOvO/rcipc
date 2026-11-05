#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"

// body 形式的惯性项（阶段 3，笔记 notes/03_inertia.md，书 §7.5 第 152–157 页）。
//
// 每个刚体各做一次隐式 Euler：它的所有质点“想”按惯性走到各自的目标，刚体选一个最接近这些目标的位姿。
// 再把刚体位姿写成关节坐标 q 的函数（经过 FK），就得到 q 的能量：
//
//   E_inert(q) = 1/(2h^2) * sum_b [ m_b |c_b(q) - c~_b|^2 + tr((R_b(q) - A~_b) S_b (R_b(q) - A~_b)^T) ]
//
// 重力藏在质心的目标 c~_b 里，所以不用另写重力项。
namespace rcipc {

// 每个刚体的惯性目标（笔记第 2 节）。0 号（世界）不用，留着只是为了下标和刚体编号一致。
struct InertiaTargets {
  std::vector<Eigen::Vector3d> c;  // 质心的目标 c~_b = 2 c_b^n - c_b^{n-1} + g h^2
  std::vector<Eigen::Matrix3d> A;  // 转动的目标 A~_b = 2 R_b^n - R_b^{n-1}（一般不是转动矩阵）
};

// 刚体 b 的质心的世界坐标：c_b = x_b + R_b * ipos_b。
Eigen::Vector3d body_com(const Model& model, const Frames& frames, int body);

// 由这一步开始时的位姿 now（q_n 处的 FK）和上一步的位姿 prev（q_{n-1} 处的 FK）算目标。
InertiaTargets inertia_targets(const Model& model, const Frames& now, const Frames& prev, double h);

// 惯性能量 E_inert（单位 J）。frames 是当前迭代点 q 处的 FK。
double inertia_energy(const Model& model, const Frames& frames, const InertiaTargets& targets, double h);

// 梯度 dE/d(delta)，长度 nv（笔记第 3 节）：每个刚体的“力” f_b 和“力矩” tau_b，经 Jacobian 投影到关节上。
Eigen::VectorXd inertia_gradient(const Model& model, const Frames& frames, const InertiaTargets& targets,
                                 double h);

// Gauss-Newton Hessian，nv x nv（笔记第 4 节）。它正好等于 M(q) / h^2，M 是关节空间的质量矩阵。
// 和目标无关，所以不需要 targets。
Eigen::MatrixXd inertia_hessian(const Model& model, const Frames& frames, double h);

// 阶段 13（笔记 notes/16_inertia_curvature.md）：Gauss-Newton 丢掉的那一项 K，nv x nv，对称。
//   精确的 Hessian（E(q ⊕ delta) 在 delta = 0 处的二阶导数）= inertia_hessian + K。
//   K_ij = sum_p f_p . d^2 x_p / (d delta_i d delta_j)，x_p 是刚体上的质点，f_p = m_p (x_p - x~_p) / h^2 是它的“残差力”。
//   刚体正好在目标上时 K = 0；链被拉紧时 K 是绳子的几何刚度（半正定）；被压缩时可能不定。
//   rot = false 时只要质心那一部分（f_b . d^2 c_b，力经过链传下去的几何刚度），不要转动那一部分（笔记 16 第 5 节）。
Eigen::MatrixXd inertia_curvature(const Model& model, const Frames& frames, const InertiaTargets& targets, double h,
                                  bool rot = true);

// 把 K 一棵树一棵树地加进 H（nv x nv）：只有 floor * Mh + K 在这棵树的块上正定时才加，否则这棵树退回 Gauss-Newton。
//   Mh 是 inertia_hessian（M / h^2）。H 里其余的项（接触、限位、驱动）都是半正定的，所以加了以后 H 仍然正定，
//   而且 K 最多抵掉质量矩阵 floor 那么多的曲率（默认 99%；笔记 16 第 5 节比较了 0.01）。返回没有加 K 的树的个数。
int add_inertia_curvature(const Model& model, const Eigen::MatrixXd& Mh, const Eigen::MatrixXd& K, Eigen::MatrixXd* H,
                          double floor = 0.99);

}  // namespace rcipc
