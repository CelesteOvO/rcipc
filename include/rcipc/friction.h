#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"

// 滞后的光滑化 Coulomb 摩擦（阶段 8，笔记 notes/08_friction.md，书 §3.6、§8.7 第 186–189 页）。
//
// Coulomb 摩擦不是任何势能的梯度：它依赖法向力和切平面，而这两者又依赖未知的 q。IPC 的做法：
//   1. 滞后：在步首把法向力 lambda、切向基 (t1, t2)、模板和权重固定下来，这一步里当常数；
//   2. 耗散功当势能：一步里摩擦耗散的功 h mu lambda |u|（u 是相对切向速度）就是势能；
//   3. 光滑化：|u| 在 0 处不可导，在 eps_v 以内换成 f0(|u|)，f0' 从 0 光滑地升到 1。
// 于是 E_fric = sum_k h mu lambda_k f0(|u_k|)，可以和惯性、障碍一起放进 Newton。
namespace rcipc {

// 光滑化函数（书第 186 页）：
//   f0(y) = -y^3 / (3 eps^2) + y^2 / eps + eps / 3   (y < eps)，    f0(y) = y   (y >= eps)
//   f1 = f0' = 2 y / eps - y^2 / eps^2               (y < eps)，    f1 = 1      (y >= eps)
// f1(0) = 0（静止时没有摩擦力），f1(eps) = 1、f1'(eps) = 0（光滑地接上滑动摩擦）。
double friction_f0(double y, double eps);
double friction_f1(double y, double eps);
// f1(y) / y：y -> 0 时趋于 2 / eps，没有除零问题。梯度和 Hessian 的系数用它。
double friction_f1_over_y(double y, double eps);

// 一个滞后的摩擦接触：步首的接触加上固定下来的量。
struct FrictionContact {
  Contact contact;        // 步首的模板、权重、两侧刚体（n 和 d 也是步首的，不再用）
  double lambda;          // 步首的法向力 -kappa m b'(x) 2d（N）
  Eigen::Vector3d t1, t2; // 切向基，垂直于步首的法向
  Eigen::Vector3d X0[4];  // 模板顶点在步首的世界坐标
};

// 步首：把活跃的接触冻结成摩擦接触（书第 186 页“对每个步首接触”）。kappa = 0 时（还没初始化）返回空。
std::vector<FrictionContact> lag_friction(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                                          double kappa, double dhat);

// 相对切向速度 u = (t1 . V, t2 . V)，V = sum_v w_v (X_v(q) - X_v^n) / h（隐式 Euler：位移除以 h）。
Eigen::Vector2d friction_velocity(const Model& model, const Frames& frames, const FrictionContact& f, double h);

// 能量 sum h mu lambda f0(|u|)、梯度 sum_j c u_j rho_j^T（c = mu lambda f1(|u|) / |u|），
//   rho_j = d(t_j . P)/d(delta) 是“切向位移对关节的导数”（接触行里把 n 换成 t_j）。
// Hessian 用切平面里的精确二阶导数 (mu lambda / h) R^T [ (f1 / |u|)(1 - u^ u^T) + f1' u^ u^T ] R（R 的两行是 rho_1、rho_2），
//   它本来就是半正定的。书上的代码用 (f1 / |u|) 1 代替方括号（一个上界），见 friction.cc 里的说明。
double friction_energy(const Model& model, const Frames& frames, const std::vector<FrictionContact>& fc, double h,
                       double mu, double eps_v);
Eigen::VectorXd friction_gradient(const Model& model, const Frames& frames, const std::vector<FrictionContact>& fc,
                                  double h, double mu, double eps_v);
Eigen::MatrixXd friction_hessian(const Model& model, const Frames& frames, const std::vector<FrictionContact>& fc,
                                 double h, double mu, double eps_v);

// Hessian 的行形式（阶段 9）：每个摩擦接触两行。把 2 x 2 的 K = (f1 / |u|)(1 - u^ u^T) + f1' u^ u^T 按特征向量拆开，
//   K = sum_i k_i v_i v_i^T，于是 (mu lambda / h) R^T K R = sum_i (mu lambda k_i / h) (v_i^T R)^T (v_i^T R)：
// 行是 v_i^T R（两条切向行的线性组合），权重 mu lambda k_i / h。
std::vector<Row> friction_rows(const Model& model, const Frames& frames, const std::vector<FrictionContact>& fc, double h,
                               double mu, double eps_v);

}  // namespace rcipc
