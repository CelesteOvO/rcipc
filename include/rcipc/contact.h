#pragma once

#include <vector>

#include <Eigen/Core>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"

// 接触（阶段 4：顶点–地面，笔记 notes/04_ground.md；阶段 7：点–三角形、边–边，notes/07_contact.md；
// 书 §8.1–8.3 第 169–175 页）。
//
// 每个接触是“两个最近点之间的向量” Delta = sum_v w_v X_v（最多 4 个顶点，A 侧权重之和为 +1，B 侧为 -1），
// 距离 d = |Delta| - 两侧的半径，接触能量用的是 d^2：
//
//   E_contact(q) = kappa * sum_{活跃的接触} b(d^2, dhat^2)
//
// b 是阶段 1 的障碍函数（barrier.h），d >= dhat 时为 0。“活跃”就是 d < dhat。
// 三种接触的区别只在最近点怎么算；能量、梯度、Hessian 的形式完全一样。
namespace rcipc {

enum class ContactType {
  kGround,         // 阶段 4：顶点对平面。模板 1 个顶点，n 是平面法向
  kPointTriangle,  // 阶段 7：顶点 p 对三角形 (a, b, c)。模板 (p, a, b, c)，权重 (1, -lam0, -lam1, -lam2)
  kEdgeEdge,       // 阶段 7：边 (p0, p1) 对边 (q0, q1)。模板 (p0, p1, q0, q1)，权重 (1-s, s, -(1-t), -t)
};

struct Contact {
  ContactType type;
  int nstencil;       // 模板里的顶点个数：地面 1，其余 4
  int verts[4];       // 顶点编号
  double w[4];        // 权重
  Eigen::Vector3d n;  // 单位向量，从 B 侧的最近点指向 A 侧的最近点（地面：平面法向），Delta = |Delta| n
  double d;           // 距离（已减半径，m），CCD 保证它 > 0
  int bodyA, bodyB;   // 两侧的刚体（地面：bodyB = 0）
  int plane;          // 地面接触的平面编号，其余 -1
  // 边–边的 mollifier（笔记 notes/07_contact.md 第 2.3 节，IPC 原文 §4.2）：两条边接近平行时最近点不唯一，
  // 距离有“折角”，Newton 会在折角两侧来回跳。把边–边的能量乘上 m(c)，c = |e_a x e_b|^2，
  // c >= eps 时 m = 1，c < eps 时 m = (2 - c / eps) c / eps，平行时 m = 0（这时点–三角形接触会接管）。
  double moll = 1;   // m(c)
  double dmoll = 0;  // dm/dc（c >= eps 时 0）
};

// 顶点 v 到平面 p 的带符号距离（已减半径）。
double ground_distance(const Model& model, const Frames& frames, int vertex, int plane);

// 所有活跃的接触（d < dhat）：顶点–平面、点–三角形、边–边（只在 model.collide 允许的刚体对之间）。
// 每次 FK 之后重新找：换了位置，活跃的接触就可能变了。
std::vector<Contact> find_contacts(const Model& model, const Frames& frames, double dhat);

// 所有候选对里最小的距离；什么都没有时返回 +无穷。每步结束后用它检查不穿透。
double min_distance(const Model& model, const Frames& frames);

// 在另一个姿态 frames 上重新算同一个接触（同一对图元）的距离：最近点参数重新求。测试用它做有限差分。
double contact_distance(const Model& model, const Frames& frames, const Contact& c);

// 接触“行”（阶段 4 笔记第 2 节，书 §8.2 第 171 页）：r = d(d^2)/d(delta) = 2 d n^T sum_v w_v J(X_v)，1 x nv。
// J 是顶点的 point_jacobian（世界上的顶点 J = 0）。含义：关节沿 delta 动时，平方距离的变化率。
Eigen::RowVectorXd contact_row(const Model& model, const Frames& frames, const Contact& c);

// 能量 kappa * sum b(d^2)、梯度 kappa * sum b' r^T、Gauss-Newton Hessian kappa * sum max(b'', 0) r^T r。
double contact_energy(const Model& model, const std::vector<Contact>& contacts, double kappa, double dhat);
Eigen::VectorXd contact_gradient(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                                 double kappa, double dhat);
Eigen::MatrixXd contact_hessian(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                                double kappa, double dhat);

// 曲率上界（阶段 6，笔记 notes/06_curvature.md，书 §8.4）。Gauss-Newton 丢掉的那一项里有顶点轨迹的曲率
//   2 kappa b'(x) Delta . Delta''，
// 它的绝对值不超过 f |Delta''|，f = 2 kappa |b'(x)| d 是法向力；而 |Delta''| <= sum_j C_j w_j^2（w_j 是第 j 个转动组
// 的增量大小，C_j = sum_k R_max(j, k) 由链上的静态力臂算出）。所以在两侧刚体的链上每个转动组的对角块加 scale * f * C_j，
// 得到的模型在每个方向上都不比真实 Hessian 弯得少（scale = 1 是可证明的上界，0 是纯 Gauss-Newton）。
// 只动 Hessian，不动梯度，所以收敛点不变。返回 nv x nv 的对角矩阵。
Eigen::MatrixXd contact_curvature(const Model& model, const Eigen::VectorXd& q, const std::vector<Contact>& contacts,
                                  double kappa, double dhat, double scale);

// kappa 的取值范围（阶段 4 笔记第 4 节）。参考值 kappa_ref 让“一个障碍在 dhat/2 处的斥力”等于一个平均刚体的重量：
//   kappa_ref * |b'(dhat^2 / 4)| * dhat = m_avg * |g|
// 下限是 kappa_ref / 100，上限是 100 kappa_ref（上限 / 下限 = 1e4，同书第 174 页）。
struct KappaRange {
  double ref, min, max;
};
KappaRange kappa_range(const Model& model, double dhat);

// kappa 的初值（阶段 4 笔记第 4 节，书 §8.3）：让障碍梯度和惯性梯度尽量抵消，
//   kappa = -(g_b . g_e) / |g_b|^2，   g_b 是 kappa = 1 时、在 frames 处的接触梯度，g_e 是惯性梯度，
// 再夹到 kappa_range 里。g_b = 0（没有活跃接触）或者算出来不是正数时，返回 kappa_ref。
// step() 传的 g_e 是步首 q_n 处的惯性梯度 -m (v / h + g)（这一步要刹住的动量加重力），不是当前迭代点的。
double initial_kappa(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                     const Eigen::VectorXd& g_inertia, double dhat);

}  // namespace rcipc
