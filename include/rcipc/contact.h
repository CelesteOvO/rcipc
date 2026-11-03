#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"

// 地面接触（阶段 4，笔记 notes/04_ground.md，书 §8.1–8.3 第 169–175 页）。
//
// 每个顶点 v 到每个平面的带符号距离 d = n . x_v - offset - radius_v。接触能量用的是平方距离 x = d^2：
//
//   E_contact(q) = kappa * sum_{活跃的 (v, 平面)} b(d^2, dhat^2)
//
// b 是阶段 1 的障碍函数（barrier.h），d >= dhat 时为 0。“活跃”就是 d < dhat。
// 用 d^2 而不是 d 的原因：到阶段 7，点-三角形、边-边的平方距离是顶点坐标的多项式，处处可导；
// 这里先把接口定成和那时一样，阶段 7 只换距离函数，不换能量、梯度和 Hessian 的形式。
namespace rcipc {

// 一个活跃的接触：顶点 v 离平面 p 不到 dhat。
struct GroundContact {
  int vertex;         // 顶点编号
  int plane;          // 平面编号
  Eigen::Vector3d x;  // 顶点的世界坐标（球心；半径已经在 d 里减掉）
  double d;           // 带符号距离（m），CCD 保证它 > 0
};

// 顶点 v 到平面 p 的带符号距离（已减半径）。
double ground_distance(const Model& model, const Frames& frames, int vertex, int plane);

// 所有活跃的接触（d < dhat）。每次 FK 之后重新找：换了位置，活跃的接触就可能变了。
std::vector<GroundContact> ground_contacts(const Model& model, const Frames& frames, double dhat);

// 所有 (顶点, 平面) 对里最小的距离；没有平面或顶点时返回 +无穷。每步结束后用它检查不穿透。
double min_ground_distance(const Model& model, const Frames& frames);

// 接触“行”（笔记第 2 节）：r = d(d^2)/d(delta) = 2 d n^T J(x)，1 x nv。J 是顶点的 point_jacobian。
// 含义：关节沿 delta 动时，平方距离的变化率。梯度和 Hessian 都由它组装。
Eigen::RowVectorXd contact_row(const Model& model, const Frames& frames, const GroundContact& c);

// 能量 kappa * sum b(d^2)、梯度 kappa * sum b' r^T、Gauss-Newton Hessian kappa * sum max(b'', 0) r^T r。
double contact_energy(const Model& model, const std::vector<GroundContact>& contacts, double kappa, double dhat);
Eigen::VectorXd contact_gradient(const Model& model, const Frames& frames, const std::vector<GroundContact>& contacts,
                                 double kappa, double dhat);
Eigen::MatrixXd contact_hessian(const Model& model, const Frames& frames, const std::vector<GroundContact>& contacts,
                                double kappa, double dhat);

// 曲率上界（阶段 6，笔记 notes/06_curvature.md，书 §8.4）。Gauss-Newton 丢掉的那一项里有顶点轨迹的曲率
//   2 kappa b'(x) Delta . Delta''，
// 它的绝对值不超过 f |Delta''|，f = 2 kappa |b'(x)| d 是法向力；而 |Delta''| <= sum_j C_j w_j^2（w_j 是第 j 个转动组
// 的增量大小，C_j = sum_k R_max(j, k) 由链上的静态力臂算出）。所以在顶点所在链上每个转动组的对角块加 scale * f * C_j，
// 得到的模型在每个方向上都不比真实 Hessian 弯得少（scale = 1 是可证明的上界，0 是纯 Gauss-Newton）。
// 只动 Hessian，不动梯度，所以收敛点不变。返回 nv x nv 的对角矩阵。
Eigen::MatrixXd contact_curvature(const Model& model, const Eigen::VectorXd& q, const std::vector<GroundContact>& contacts,
                                  double kappa, double dhat, double scale);

// kappa 的取值范围（笔记第 4 节）。参考值 kappa_ref 让“一个障碍在 dhat/2 处的斥力”等于一个平均刚体的重量：
//   kappa_ref * |b'(dhat^2 / 4)| * dhat = m_avg * |g|
// 下限是 kappa_ref / 100，上限是 100 kappa_ref（上限 / 下限 = 1e4，同书第 174 页）。
struct KappaRange {
  double ref, min, max;
};
KappaRange kappa_range(const Model& model, double dhat);

// kappa 的初值（笔记第 4 节，书 §8.3）：让障碍梯度和惯性梯度尽量抵消，
//   kappa = -(g_b . g_e) / |g_b|^2，   g_b 是 kappa = 1 时、在 frames 处的接触梯度，g_e 是惯性梯度，
// 再夹到 kappa_range 里。g_b = 0（没有活跃接触）或者算出来不是正数时，返回 kappa_ref。
// step() 传的 g_e 是步首 q_n 处的惯性梯度 -m (v / h + g)（这一步要刹住的动量加重力），不是当前迭代点的。
double initial_kappa(const Model& model, const Frames& frames, const std::vector<GroundContact>& contacts,
                     const Eigen::VectorXd& g_inertia, double dhat);

}  // namespace rcipc
