#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/model.h"

// 驱动、关节阻尼和 armature（阶段 12，笔记 notes/12_drives.md）。
//
// 三者都只作用在单个自由度（铰链、滑动）上，能量是这个自由度的一元凸函数，所以梯度和 Hessian 只落在对角上，
// 和关节限位一样进树块 H_k。都按隐式欧拉处理（速度取这一步的 (q - q0) / h）：
//
//   阻尼 b、armature a：
//     E = b/(2h) (q - q0)^2 + a/(2h^2) (q - q0 - h v0)^2 = K/2 (q - r)^2 + 常数，   K = b/h + a/h^2
//     （a 就是加在质量矩阵对角上的转子惯量，所以和惯性项同一个形状；阻尼是瑞利耗散函数乘 h。）
//
//   驱动（MuJoCo 的 general 模型，L = s q，s 是传动比）：
//     驱动力 f = g u + b0 + b1 L + b2 dL/dt，关节力矩 tau = s f。把 dL/dt 换成 s (q - q0) / h，f 就是 q 的一次函数
//       f = A + C q,   A = g u + b0 - b2 s q0 / h,   C = s (b1 + b2 / h)。
//     能量取 E = -∫ tau dq，梯度就是 -tau。b1、b2 <= 0（读模型时检查）时 dtau/dq = s^2 (b1 + b2/h) <= 0，E 是凸的。
//     有力的上下限时 tau = s clamp(f, lo, hi)，E 变成 Huber 的形状：范围里二次，范围外一次，仍然凸、一阶连续。
//     position 驱动（b1 = -kp，b2 = -kv）就是隐式的 PD 控制；velocity（b2 = -kv）是隐式的速度伺服；motor 是常力。
namespace rcipc {

struct DriveTerms {
  Eigen::VectorXd K, r;  // 阻尼和 armature：每个自由度一个（球、自由关节的是 0）
  struct Act {
    int dof;
    double s, A, C, w;   // w = b1 + b2 / h（<= 0），C = s w
    double lo, hi;       // 力的范围（没有限制时是正负无穷）
  };
  std::vector<Act> act;
};

// 一步开始时准备：q0、v0 是步首的状态，ctrl 是这一步的控制量（长度 nu；空表示全 0）。这一步里都是常数。
DriveTerms drive_terms(const Model& model, const Eigen::VectorXd& q0, const Eigen::VectorXd& v0,
                       const Eigen::VectorXd& ctrl, double h);

double drive_energy(const Model& model, const DriveTerms& D, const Eigen::VectorXd& q);
Eigen::VectorXd drive_gradient(const Model& model, const DriveTerms& D, const Eigen::VectorXd& q);
Eigen::VectorXd drive_hessian_diagonal(const Model& model, const DriveTerms& D, const Eigen::VectorXd& q);

// 每个驱动在 q 处的关节力矩 tau = s clamp(f)（长度 nu），测试和演示用。
Eigen::VectorXd actuator_torques(const Model& model, const DriveTerms& D, const Eigen::VectorXd& q);

// 时刻 t 的控制量（长度 nu）：按关键帧（Model::ctrl_keys）分段线性插值，第一帧之前、最后一帧之后取端点的值；
// 没有关键帧时全 0。
Eigen::VectorXd ctrl_at(const Model& model, double t);

}  // namespace rcipc
