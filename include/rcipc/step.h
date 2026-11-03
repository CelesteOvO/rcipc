#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/model.h"

// 关节坐标里的一个时间步（阶段 3，笔记 notes/03_inertia.md 第 5 节）。
//
// 和阶段 1 的 ipc1d::step 是同一个算法，只是未知量从一个数 x 换成了关节坐标 q：
//   x + alpha * p   ->  integrate(q, p, alpha)        （q ⊕ alpha p）
//   p = -E' / E''   ->  解 H p = -g                    （H 是 nv x nv 的矩阵）
//   (x - x0) / h    ->  differentiate(q0, q) / h      （(q ⊖ q0) / h）
// 现在能量里只有惯性项（含重力），还没有接触，所以也还没有 CCD：每次迭代的步长上限是 1。
namespace rcipc {

struct StepParams {
  double h = 0.002;     // 时间步长（s）
  double tol = 1e-6;    // 收敛判断：max_i |p_i| / h < tol（单位 m/s 或 rad/s）
  int max_newton = 50;  // 最多迭代次数
};

// 一次 Newton 迭代的记录，用来看收敛过程。
struct NewtonIterate {
  double energy;  // 迭代开始时的能量 E(q)
  double step;    // max_i |p_i| / h，收敛判断用的量
  double alpha;   // 线搜索接受的步长
};

struct StepResult {
  Eigen::VectorXd q;  // 新位置 q_{n+1}
  Eigen::VectorXd v;  // 新速度 v_{n+1} = (q_{n+1} ⊖ q_n) / h
  int newton = 0;     // 迭代次数，含最后那次只检查不移动的迭代（同阶段 1）
};

// 从 (q0, v0) 走一个时间步。trace 不为空时，把每次移动的迭代记进去。
StepResult step(const Model& model, const StepParams& params, const Eigen::VectorXd& q0,
                const Eigen::VectorXd& v0, std::vector<NewtonIterate>* trace = nullptr);

// 动能 1/2 v^T M(q) v 和重力势能 -sum_b m_b g . c_b（单位 J），用来检查能量守恒。
double kinetic_energy(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& v);
double potential_energy(const Model& model, const Eigen::VectorXd& q);

}  // namespace rcipc
