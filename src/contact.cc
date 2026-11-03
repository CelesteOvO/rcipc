#include "rcipc/contact.h"

#include <cmath>
#include <limits>

#include "rcipc/barrier.h"
#include "rcipc/chain.h"

// 地面接触。推导见 notes/04_ground.md，测试见 tests/test_contact.cc。
namespace rcipc {

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

double ground_distance(const Model& model, const Frames& frames, int vertex, int plane) {
  // 带符号距离（笔记第 1 节）：先把顶点转到世界系，再投影到法向上，最后减掉半径。
  const Vertex& V = model.vertices[vertex];
  const Plane& P = model.planes[plane];
  const Vector3d x = world_point(frames, V.body, V.local);
  return P.normal.dot(x) - P.offset - V.radius;
}

std::vector<GroundContact> ground_contacts(const Model& model, const Frames& frames, double dhat) {
  // 遍历所有 (顶点, 平面) 对。顶点和平面都很少，直接双重循环；阶段 7 的刚体间接触才需要宽相。
  std::vector<GroundContact> out;
  for (int v = 0; v < model.nvertex(); ++v) {
    const Vertex& V = model.vertices[v];
    const Vector3d x = world_point(frames, V.body, V.local);
    for (int p = 0; p < static_cast<int>(model.planes.size()); ++p) {
      const double d = model.planes[p].normal.dot(x) - model.planes[p].offset - V.radius;
      if (d < dhat) out.push_back({v, p, x, d});
    }
  }
  return out;
}

double min_ground_distance(const Model& model, const Frames& frames) {
  double dmin = std::numeric_limits<double>::infinity();
  for (int v = 0; v < model.nvertex(); ++v) {
    for (int p = 0; p < static_cast<int>(model.planes.size()); ++p) {
      dmin = std::min(dmin, ground_distance(model, frames, v, p));
    }
  }
  return dmin;
}

RowVectorXd contact_row(const Model& model, const Frames& frames, const GroundContact& c) {
  // 接触行（笔记第 2 节，书 §8.1 第 171 页）。链式法则分三层：
  //   d(d^2)/d(delta) = 2 d * dd/dx * dx/d(delta) = 2 d * n^T * J(x)
  //   - d^2 对 d 的导数是 2d；
  //   - 平面距离 d = n . x - offset - radius 对顶点位置 x 的导数是 n（半径是常数）；
  //   - 顶点位置对关节的导数是阶段 2 的 point_jacobian（转动自由度那一列是 a x (x - o)）。
  // 结果是 1 x nv 的行向量；不在顶点所在链上的自由度那一列是 0。
  const Vertex& V = model.vertices[c.vertex];
  const Vector3d& n = model.planes[c.plane].normal;
  return 2 * c.d * (n.transpose() * point_jacobian(model, frames, V.body, c.x));
}

double contact_energy(const Model& model, const std::vector<GroundContact>& contacts, double kappa, double dhat) {
  // 每个活跃接触贡献 kappa * b(d^2, dhat^2)。阶段 1 传的是 (d, dhat)，现在传的是平方（barrier.h 里说过）。
  (void)model;
  double e = 0;
  for (const GroundContact& c : contacts) e += kappa * barrier(c.d * c.d, dhat * dhat).b;
  return e;
}

VectorXd contact_gradient(const Model& model, const Frames& frames, const std::vector<GroundContact>& contacts,
                          double kappa, double dhat) {
  // 梯度（笔记第 3 节）：d/d(delta) [kappa b(d^2)] = kappa b'(d^2) * r^T。
  //   物理含义：-kappa b'(d^2) * 2d 是法向力的大小（b' < 0，所以是斥力），r / (2d) = n^T J 把它投影到关节上。
  VectorXd g = VectorXd::Zero(model.nv);
  for (const GroundContact& c : contacts) {
    const double db = barrier(c.d * c.d, dhat * dhat).db;
    g += kappa * db * contact_row(model, frames, c).transpose();
  }
  return g;
}

MatrixXd contact_hessian(const Model& model, const Frames& frames, const std::vector<GroundContact>& contacts,
                         double kappa, double dhat) {
  // Gauss-Newton Hessian（笔记第 3 节）。精确的二阶导数是
  //   kappa b'' r^T r  +  kappa b' d^2(d^2)/d(delta)^2
  // 第一项是“行的外积”，半正定；第二项含 b' < 0 和顶点轨迹的曲率，可能不定。这里只保留第一项，
  // 并且用 max(b'', 0) 保险（b'' 在 (0, dhat) 上本来就是正的）。丢掉第二项的后果是阶段 6 的内容。
  MatrixXd H = MatrixXd::Zero(model.nv, model.nv);
  for (const GroundContact& c : contacts) {
    const double ddb = std::max(barrier(c.d * c.d, dhat * dhat).ddb, 0.0);
    const RowVectorXd r = contact_row(model, frames, c);
    H += kappa * ddb * r.transpose() * r;
  }
  return H;
}

MatrixXd contact_curvature(const Model& model, const VectorXd& q, const std::vector<GroundContact>& contacts,
                           double kappa, double dhat, double scale) {
  // 曲率上界（笔记第 3 节，书 §8.4 第 176–177 页）。推导的三步：
  //   1. 被丢掉的项沿方向 delta 是 2 kappa b' Delta . Delta''，Cauchy-Schwarz 给 |.| <= 2 kappa |b'| d |Delta''| = f |Delta''|。
  //   2. 顶点的二阶导 Delta'' = sum_{j,k} delta_j delta_k d^2X/dq_j dq_k，两个转动的二阶导形如 a_j x (a_k x (X - o_k))，
  //      大小不超过到较深那个锚点的力臂 R_max(j,k)。于是 |Delta''| <= sum_{j,k} w_j w_k R_max(j,k)。
  //   3. w_j w_k <= (w_j^2 + w_k^2) / 2 把交叉项摊到对角上：|Delta''| <= sum_j C_j w_j^2，C_j = sum_k R_max(j,k)。
  // 合起来 |被丢掉的项| <= delta^T (f sum_j C_j P_j) delta，P_j 是第 j 组自由度上的单位对角块。
  MatrixXd H = MatrixXd::Zero(model.nv, model.nv);
  if (scale <= 0) return H;
  for (const GroundContact& c : contacts) {
    const double f = 2 * kappa * std::abs(barrier(c.d * c.d, dhat * dhat).db) * c.d;  // 法向力的大小
    const std::vector<ChainJoint> chain = chain_joints(model, q, model.vertices[c.vertex].body);
    const int n = static_cast<int>(chain.size());
    for (int j = 0; j < n; ++j) {
      if (!chain[j].rotation) continue;
      // C_j = sum_k R_max(j, k)：k 比 j 靠近根时较深的是 j，用 R_j；k 比 j 深时用 R_k。
      double C = 0;
      for (int k = 0; k < n; ++k) {
        if (!chain[k].rotation) continue;
        C += chain[k <= j ? j : k].reach;
      }
      for (int d = chain[j].dadr; d < chain[j].dadr + chain[j].nv; ++d) H(d, d) += scale * f * C;
    }
  }
  return H;
}

KappaRange kappa_range(const Model& model, double dhat) {
  // 参考值：一个障碍在 d = dhat/2 处的斥力 kappa |b'(d^2)| 2d 等于一个平均刚体的重量 m_avg |g|。
  //   解出 kappa_ref = m_avg |g| / (|b'(dhat^2/4)| dhat)。b'(dhat^2/4) = -4.33 dhat^2，所以 kappa_ref ≈ m_avg |g| / (4.33 dhat^3)。
  //   单位：N / m^3 = J / m^4，和 kappa b（b 的单位是 m^4）相乘得到焦耳。
  // 它只是一个量级，真正的初值由 initial_kappa 的梯度平衡决定；这里给的是夹的范围。
  double mass = 0;
  for (int b = 1; b < model.nbody(); ++b) mass += model.bodies[b].mass;
  const double m_avg = mass / std::max(1, model.nbody() - 1);
  const double gnorm = model.gravity.norm() > 0 ? model.gravity.norm() : 9.81;  // 没有重力时按 1 g 算一个量级
  const double db = std::abs(barrier(0.25 * dhat * dhat, dhat * dhat).db);
  const double ref = m_avg * gnorm / (db * dhat);
  return {ref, 0.01 * ref, 100.0 * ref};
}

double initial_kappa(const Model& model, const Frames& frames, const std::vector<GroundContact>& contacts,
                     const VectorXd& g_inertia, double dhat) {
  // 梯度平衡（笔记第 4 节，书 §8.3 第 174 页）：选 kappa 使 |g_e + kappa g_b|^2 最小。
  //   对 kappa 求导令其为 0：2 g_b . (g_e + kappa g_b) = 0，得 kappa = -(g_b . g_e) / |g_b|^2。
  //   含义：障碍一出现就和惯性（含重力）“势均力敌”，Newton 的第一步不会被一个太软或太硬的障碍带偏。
  const KappaRange range = kappa_range(model, dhat);
  const VectorXd g_b = contact_gradient(model, frames, contacts, 1.0, dhat);
  const double gb2 = g_b.squaredNorm();
  double kappa = range.ref;
  if (gb2 > 0) {
    const double balance = -g_b.dot(g_inertia) / gb2;
    if (balance > 0) kappa = balance;  // 惯性在把物体拉离障碍时 balance <= 0，这时没有平衡可言，用参考值
  }
  return std::min(std::max(kappa, range.min), range.max);
}

}  // namespace rcipc
