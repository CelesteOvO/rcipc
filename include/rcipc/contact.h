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
  int verts[4] = {-1, -1, -1, -1};  // 顶点编号（前 nstencil 个有效，其余 -1）
  double w[4] = {0, 0, 0, 0};        // 权重（前 nstencil 个有效，其余 0）
  Eigen::Vector3d n;  // 单位向量，从 B 侧的最近点指向 A 侧的最近点（地面：平面法向），Delta = |Delta| n
  double d;           // 距离（已减半径，m），CCD 保证它 > 0
  int bodyA, bodyB;   // 两侧的刚体（地面：bodyB = 0）
  int plane = -1;     // 地面接触的平面编号，其余 -1
  // 边–边的 mollifier（笔记 notes/07_contact.md 第 2.3 节，IPC 原文 §4.2）：两条边接近平行时最近点不唯一，
  // 距离有“折角”，Newton 会在折角两侧来回跳。把边–边的能量乘上 m(c)，c = |e_a x e_b|^2，
  // c >= eps 时 m = 1，c < eps 时 m = (2 - c / eps) c / eps，平行时 m = 0（这时点–三角形接触会接管）。
  double moll = 1;   // m(c)
  double dmoll = 0;  // dm/dc（c >= eps 时 0）
  // 这个接触的障碍刚度是 kappa * kw：能量、梯度、Hessian 都只是乘上它（contact.cc、friction.cc 的 lambda）。
  //   find_contacts 给 1。原始-对偶障碍（StepParams::primal_dual）组装时在接触的副本上把它乘以 lambda_eff / F，
  //   只改 Hessian（笔记 notes/22_primal_dual.md 第 2 节）。
  double kw = 1;
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

// Newton 系统里的一“行”（阶段 9，书第 152 页、§10）：Hessian 加 D r^T r。障碍行 D = kappa m b''，摩擦行见 friction.h。
// 一行只在两侧刚体的链上非零，所以 H = H_k + sum_r D_r r_r^T r_r 不用组装成稠密矩阵就能乘向量（solver.h）。
struct Row {
  Eigen::RowVectorXd r;
  double D;
};

// 障碍行的权重：阶段 4 是 max(b'', 0)；阶段 9 起是 b(d^2) 对 d 的精确二阶导数的系数 max(b'' + b' / (2 d^2), 0)
// （contact.cc 里有说明）。两者都半正定，后者让 Newton 在法向上二次收敛。
double barrier_row_weight(double d, double dhat);

// 能量 kappa * sum b(d^2)、梯度 kappa * sum b' r^T、Hessian kappa * sum weight r^T r。
// contact_rows 给出 Hessian 的行形式（contact_hessian 就是这些行的 sum D r^T r，测试拿两者对比）。
double contact_energy(const Model& model, const std::vector<Contact>& contacts, double kappa, double dhat);
std::vector<Row> contact_rows(const Model& model, const Frames& frames, const std::vector<Contact>& contacts, double kappa,
                              double dhat);
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

// ---------- 阶段 14：完整的接触 Hessian（笔记 notes/17_contact_hessian.md） ----------
//
// 到阶段 13 为止，接触的 Hessian 是 Gauss-Newton 行 kappa w r^T r（w 补回了法向的二阶项）加阶段 6 的对角上界。
// 精确的二阶导数还有 kappa b' 乘上 d^2 对 delta 的二阶导数里剩下的部分：最近点的参数随顶点变化（包络定理）、
// 顶点沿关节走弧线（轨迹曲率）、两条边的 mollifier。这些项不定号，不能逐项加。这里把一个接触的全部二阶导数在
// 两侧刚体的旋量空间里算成 12 x 12 的对称矩阵（每侧 6 维：刚体原点的平移 v 和转动 omega，顶点 X 的速度是
// v + omega x (X - xpos)），做特征分解，只留正的特征值；每个特征向量就是两侧各一个力旋量 (f, tau)，正好是 GPU 上
// 行的存法（src/gpu/rows.cuh）。关节坐标对旋量不是线性的那一截（铰链锚点不在刚体原点、链上有上游关节）另算，
// 和惯性的曲率项一样经正定检查进树块；自由刚体上它恰好是 0。

// 平方距离 y = |Delta|^2 对模板顶点坐标的精确二阶导数（3 nstencil x 3 nstencil）。固定权重时是 2 (w w^T) ⊗ I；
// 最近点在三角形内部、边内部或线段内部时参数会随顶点动，包络定理给出要减掉的一项 f_{X lam} f_{lam lam}^-1 f_{lam X}。
// 地面：2 n n^T。X 是模板顶点的世界坐标。
Eigen::MatrixXd squared_distance_hessian(const Contact& c, const Eigen::Vector3d* X);

// 一个接触的能量 kappa m b(d^2) 对两侧旋量 (v_A, omega_A, v_B, omega_B) 的精确二阶导数（12 x 12，对称，不定号），
// 含顶点随转动走弧线的几何项。地面接触的 B 侧是世界，后 6 行 6 列为 0。
Eigen::Matrix<double, 12, 12> contact_wrench_hessian(const Model& model, const Frames& frames, const Contact& c,
                                                     double kappa, double dhat);

// 旋量对 delta 的 Jacobian（12 x nv）：前 6 行是刚体 A 的 (v, omega)，后 6 行是 B 的；旋量以各自的原点 xpos 为参考点。
// 转动自由度 d 在它刚体的链上时那一列是 ((o_d - xpos) x a_d, a_d)，滑动是 (a_d, 0)。
// 顶点的 point_jacobian 等于 [I, -[X - xpos]x] 乘它，所以 sum_c J_c^T H12_c J_c 加 contact_chain_curvature 就是精确的 Hessian。
Eigen::MatrixXd contact_twist_jacobian(const Model& model, const Frames& frames, const Contact& c);

// 完整 Hessian 的行形式：每个接触的 12 x 12 做特征分解，特征值 > rel_tol * 最大特征值的每一对变成一行
// （r = 特征向量作为力旋量在各自由度上的值，D = 特征值），追加到 rows。一个接触最多 12 行。
void contact_rows_full(const Model& model, const Frames& frames, const std::vector<Contact>& contacts, double kappa,
                       double dhat, std::vector<Row>* rows, double rel_tol = 1e-12);

// 链上的剩余项（nv x nv，对称）：精确的 delta 空间 Hessian 减去 sum_c J_c^T H12_c J_c。只在每侧刚体链上的自由度之间
// 非零：同一个关节的两个转动（锚点 o 不在原点时是合力关于 (xpos - o) 的几何项）、上游转动对下游转动或滑动。
// 不定号，step() 把它和惯性的曲率项加在一起经 add_inertia_curvature 的正定检查进树块。
Eigen::MatrixXd contact_chain_curvature(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                                        double kappa, double dhat);

// 测试用：不投影的精确 Hessian sum_c J_c^T H12_c J_c + contact_chain_curvature，和能量的有限差分一致。
Eigen::MatrixXd contact_hessian_exact(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                                      double kappa, double dhat);

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

// ---------- 秩 1 曲率行（两阶段曲率，笔记 notes/22_primal_dual.md 第 4 节；书 §8.6） ----------

// 刚体 body 上的点 X（世界坐标）沿自由度方向 v（长度 nv）的二阶导数 X''[v, v] = sum_{j,k} v_j v_k d^2 X / dq_j dq_k。O(链长)。
Eigen::Vector3d point_acceleration_along(const Model& model, const Frames& frames, int body, const Eigen::Vector3d& X,
                                         const Eigen::VectorXd& v);

// 接触 c 的秩 1 曲率权重：Gauss-Newton 丢掉的轨迹曲率项 S（阶段 6 用对角上界盖住的那一项）只取沿它自己那一行 r 的方向，
//   max(0, r^T S r) / |r|^4，加到这一行的权重 D 上（行数不变）。r 是 contact_row。
double contact_curvature_row_weight(const Model& model, const Frames& frames, const Contact& c,
                                    const Eigen::RowVectorXd& r, double kappa, double dhat);

}  // namespace rcipc
