#include "rcipc/contact.h"

#include <cmath>
#include <initializer_list>
#include <limits>

#include "rcipc/barrier.h"
#include "rcipc/chain.h"
#include "rcipc/distance.h"

// 接触。推导见 notes/04_ground.md（地面）和 notes/07_contact.md（刚体之间），测试见 tests/test_contact.cc。
namespace rcipc {

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

// 所有顶点的世界坐标，一次算好。
std::vector<Vector3d> world_vertices(const Model& model, const Frames& frames) {
  std::vector<Vector3d> X(model.nvertex());
  for (int v = 0; v < model.nvertex(); ++v) X[v] = world_point(frames, model.vertices[v].body, model.vertices[v].local);
  return X;
}

// 每个刚体的包围盒（顶点位置，按顶点半径放大），宽相用。
struct Aabb {
  Vector3d lo = Vector3d::Constant(std::numeric_limits<double>::infinity());
  Vector3d hi = Vector3d::Constant(-std::numeric_limits<double>::infinity());
  bool overlaps(const Aabb& o, double margin) const {
    return (lo.array() <= o.hi.array() + margin).all() && (o.lo.array() <= hi.array() + margin).all();
  }
};

std::vector<Aabb> body_aabbs(const Model& model, const std::vector<Vector3d>& X) {
  std::vector<Aabb> box(model.nbody());
  for (int v = 0; v < model.nvertex(); ++v) {
    Aabb& b = box[model.vertices[v].body];
    const double r = model.vertices[v].radius;
    b.lo = b.lo.cwiseMin(X[v] - Vector3d::Constant(r));
    b.hi = b.hi.cwiseMax(X[v] + Vector3d::Constant(r));
  }
  return box;
}

// 点–三角形接触：顶点 v 对三角形 f。距离 >= dmax 时返回 false。
bool make_pt(const Model& model, const std::vector<Vector3d>& X, int v, int f, double dmax, Contact& c) {
  const Face& F = model.faces[f];
  const Vector3d lam = point_triangle_closest(X[v], X[F.v0], X[F.v1], X[F.v2]);
  const Vector3d delta = X[v] - (lam[0] * X[F.v0] + lam[1] * X[F.v1] + lam[2] * X[F.v2]);
  const double len = delta.norm();
  const double d = len - model.vertices[v].radius - model.vertices[F.v0].radius;
  if (d >= dmax || len <= 0) return false;
  c.type = ContactType::kPointTriangle;
  c.nstencil = 4;
  c.verts[0] = v;
  c.verts[1] = F.v0;
  c.verts[2] = F.v1;
  c.verts[3] = F.v2;
  c.w[0] = 1;
  c.w[1] = -lam[0];
  c.w[2] = -lam[1];
  c.w[3] = -lam[2];
  c.n = delta / len;
  c.d = d;
  c.bodyA = model.vertices[v].body;
  c.bodyB = F.body;
  c.plane = -1;
  return true;
}

// 边–边接触：边 e1 对边 e2。
bool make_ee(const Model& model, const std::vector<Vector3d>& X, int e1, int e2, double dmax, Contact& c) {
  const Edge& P = model.edges[e1];
  const Edge& Q = model.edges[e2];
  double s, t;
  segment_segment_closest(X[P.v0], X[P.v1], X[Q.v0], X[Q.v1], s, t);
  const Vector3d delta = ((1 - s) * X[P.v0] + s * X[P.v1]) - ((1 - t) * X[Q.v0] + t * X[Q.v1]);
  const double len = delta.norm();
  const double d = len - model.vertices[P.v0].radius - model.vertices[Q.v0].radius;
  if (d >= dmax || len <= 0) return false;
  c.type = ContactType::kEdgeEdge;
  c.nstencil = 4;
  c.verts[0] = P.v0;
  c.verts[1] = P.v1;
  c.verts[2] = Q.v0;
  c.verts[3] = Q.v1;
  c.w[0] = 1 - s;
  c.w[1] = s;
  c.w[2] = -(1 - t);
  c.w[3] = -t;
  c.n = delta / len;
  c.d = d;
  c.bodyA = P.body;
  c.bodyB = Q.body;
  c.plane = -1;
  // mollifier：c_x = |e_a x e_b|^2，阈值 eps = 1e-3 |e_a|^2 |e_b|^2（边是刚的，现在的长度就是静止长度）。
  const Vector3d ea = X[P.v1] - X[P.v0], eb = X[Q.v1] - X[Q.v0];
  const double cx = ea.cross(eb).squaredNorm();
  const double eps = 1e-3 * ea.squaredNorm() * eb.squaredNorm();
  if (eps > 0 && cx < eps) {
    c.moll = (2 - cx / eps) * (cx / eps);
    c.dmoll = 2 * (1 - cx / eps) / eps;
  } else {
    c.moll = 1;
    c.dmoll = 0;
  }
  return true;
}

// mollifier 的行：dc_x/d(delta)，1 x nv。c_x = |u x v|^2，u = X1 - X0，v = X3 - X2，w = u x v：
//   dc_x/du = 2 v x w，dc_x/dv = 2 w x u（对 w . w 用链式法则，再用 [v]x 的转置是 -[v]x）。
RowVectorXd mollifier_row(const Model& model, const Frames& frames, const Contact& c) {
  Vector3d P[4];
  for (int i = 0; i < 4; ++i) P[i] = world_point(frames, model.vertices[c.verts[i]].body, model.vertices[c.verts[i]].local);
  const Vector3d u = P[1] - P[0], v = P[3] - P[2], w = u.cross(v);
  const Vector3d du = 2 * v.cross(w), dv = 2 * w.cross(u);
  const Vector3d dX[4] = {-du, du, -dv, dv};
  RowVectorXd row = RowVectorXd::Zero(model.nv);
  for (int i = 0; i < 4; ++i) {
    const Vertex& V = model.vertices[c.verts[i]];
    if (V.body == 0) continue;
    row += dX[i].transpose() * point_jacobian(model, frames, V.body, P[i]);
  }
  return row;
}

// 遍历所有候选：顶点–平面，以及宽相通过的刚体对之间的点–三角形（两个方向）和边–边。
// dmax 是距离的上限（接触检测传 dhat，求最小距离传 +无穷）；每个候选调用 visit(c)。
template <class Visit>
void for_each_candidate(const Model& model, const Frames& frames, double dmax, Visit visit) {
  const std::vector<Vector3d> X = world_vertices(model, frames);
  // 顶点–平面。世界上的静态几何不和平面比（它贴着地面放时距离是 0，障碍会变成无穷大；书 §8.1）。
  for (int v = 0; v < model.nvertex(); ++v) {
    if (model.vertices[v].body == 0) continue;
    for (int p = 0; p < static_cast<int>(model.planes.size()); ++p) {
      const double d = model.planes[p].normal.dot(X[v]) - model.planes[p].offset - model.vertices[v].radius;
      if (d >= dmax) continue;
      Contact c;
      c.type = ContactType::kGround;
      c.nstencil = 1;
      c.verts[0] = v;
      c.w[0] = 1;
      c.n = model.planes[p].normal;
      c.d = d;
      c.bodyA = model.vertices[v].body;
      c.bodyB = 0;
      c.plane = p;
      visit(c);
    }
  }
  // 刚体之间。宽相：包围盒按 dmax 放大后不重叠的刚体对直接跳过（可以多留，绝不能漏）。
  if (model.nedge() == 0 && model.nface() == 0) return;
  const std::vector<Aabb> box = body_aabbs(model, X);
  const double margin = std::isfinite(dmax) ? dmax : 1e300;
  for (int a = 0; a < model.nbody(); ++a) {
    for (int b = a + 1; b < model.nbody(); ++b) {
      if (!model.collide(a, b) || !box[a].overlaps(box[b], margin)) continue;
      Contact c;
      // a 的顶点对 b 的三角形，b 的顶点对 a 的三角形。
      //   第二层（阶段 11 前加的）：离对方整个刚体的包围盒超过 dmax 的顶点、边先剔掉。这是精确的（d < dmax 的对一个都不会漏），
      //   只是省掉网格上成千上万次注定失败的最近点计算。
      Aabb pbox;
      const auto near = [&](const Aabb& other, std::initializer_list<int> ids) {
        pbox = Aabb();
        for (int v : ids) {
          const double r = model.vertices[v].radius;
          pbox.lo = pbox.lo.cwiseMin(X[v] - Vector3d::Constant(r));
          pbox.hi = pbox.hi.cwiseMax(X[v] + Vector3d::Constant(r));
        }
        return pbox.overlaps(other, margin);
      };
      for (int v = 0; v < model.nvertex(); ++v) {
        const int bv = model.vertices[v].body;
        if (bv != a && bv != b) continue;
        if (!near(box[bv == a ? b : a], {v})) continue;
        for (int f = 0; f < model.nface(); ++f) {
          if (model.faces[f].body != (bv == a ? b : a)) continue;
          if (make_pt(model, X, v, f, dmax, c)) visit(c);
        }
      }
      // a 的边对 b 的边。
      for (int e1 = 0; e1 < model.nedge(); ++e1) {
        if (model.edges[e1].body != a) continue;
        if (!near(box[b], {model.edges[e1].v0, model.edges[e1].v1})) continue;
        for (int e2 = 0; e2 < model.nedge(); ++e2) {
          if (model.edges[e2].body != b) continue;
          if (make_ee(model, X, e1, e2, dmax, c)) visit(c);
        }
      }
    }
  }
}

}  // namespace

double ground_distance(const Model& model, const Frames& frames, int vertex, int plane) {
  // 带符号距离（阶段 4 笔记第 1 节）：先把顶点转到世界系，再投影到法向上，最后减掉半径。
  const Vertex& V = model.vertices[vertex];
  const Plane& P = model.planes[plane];
  const Vector3d x = world_point(frames, V.body, V.local);
  return P.normal.dot(x) - P.offset - V.radius;
}

std::vector<Contact> find_contacts(const Model& model, const Frames& frames, double dhat) {
  std::vector<Contact> out;
  for_each_candidate(model, frames, dhat, [&](const Contact& c) { out.push_back(c); });
  return out;
}

double min_distance(const Model& model, const Frames& frames) {
  double dmin = std::numeric_limits<double>::infinity();
  for_each_candidate(model, frames, std::numeric_limits<double>::infinity(),
                     [&](const Contact& c) { dmin = std::min(dmin, c.d); });
  return dmin;
}

double contact_distance(const Model& model, const Frames& frames, const Contact& c) {
  if (c.type == ContactType::kGround) return ground_distance(model, frames, c.verts[0], c.plane);
  Vector3d P[4];
  for (int i = 0; i < 4; ++i) P[i] = world_point(frames, model.vertices[c.verts[i]].body, model.vertices[c.verts[i]].local);
  const double radii = model.vertices[c.verts[0]].radius + model.vertices[c.verts[2]].radius;
  if (c.type == ContactType::kPointTriangle) {
    const Vector3d lam = point_triangle_closest(P[0], P[1], P[2], P[3]);
    return (P[0] - (lam[0] * P[1] + lam[1] * P[2] + lam[2] * P[3])).norm() - radii;
  }
  double s, t;
  segment_segment_closest(P[0], P[1], P[2], P[3], s, t);
  return (((1 - s) * P[0] + s * P[1]) - ((1 - t) * P[2] + t * P[3])).norm() - radii;
}

RowVectorXd contact_row(const Model& model, const Frames& frames, const Contact& c) {
  // 接触行（阶段 4 笔记第 2 节，书 §8.2 第 171 页）。链式法则分三层：
  //   d(d^2)/d(delta) = 2 d * dd/dX * dX/d(delta)
  //   - d^2 对 d 的导数是 2d；
  //   - 距离 d = |Delta| - 半径 对模板顶点的导数：d|Delta|/dX_v = w_v n（最近点的参数由包络定理当作常数）；
  //   - 顶点位置对关节的导数是阶段 2 的 point_jacobian（转动自由度那一列是 a x (x - o)）。
  // 结果是 1 x nv 的行向量；不在两侧链上的自由度那一列是 0。
  RowVectorXd r = RowVectorXd::Zero(model.nv);
  for (int i = 0; i < c.nstencil; ++i) {
    const Vertex& V = model.vertices[c.verts[i]];
    if (V.body == 0) continue;  // 世界上的顶点不动
    const Vector3d x = world_point(frames, V.body, V.local);
    r += c.w[i] * (c.n.transpose() * point_jacobian(model, frames, V.body, x));
  }
  return 2 * c.d * r;
}

double contact_energy(const Model& model, const std::vector<Contact>& contacts, double kappa, double dhat) {
  // 每个活跃接触贡献 kappa * b(d^2, dhat^2)。阶段 1 传的是 (d, dhat)，现在传的是平方（barrier.h 里说过）。
  // 边–边的能量再乘上 mollifier m（其余接触 m = 1）。
  (void)model;
  double e = 0;
  for (const Contact& c : contacts) e += kappa * c.moll * barrier(c.d * c.d, dhat * dhat).b;
  return e;
}

VectorXd contact_gradient(const Model& model, const Frames& frames, const std::vector<Contact>& contacts, double kappa,
                          double dhat) {
  // 梯度（阶段 4 笔记第 3 节）：d/d(delta) [kappa b(d^2)] = kappa b'(d^2) * r^T。
  //   物理含义：-kappa b'(d^2) * 2d 是法向力的大小（b' < 0，所以是斥力），r / (2d) 把它投影到关节上。
  // 带 mollifier 时是乘积的导数：d/d(delta) [kappa m b] = kappa (m b' r + b m' dc_x/d(delta))。
  VectorXd g = VectorXd::Zero(model.nv);
  for (const Contact& c : contacts) {
    const BarrierValue bv = barrier(c.d * c.d, dhat * dhat);
    g += kappa * c.moll * bv.db * contact_row(model, frames, c).transpose();
    if (c.dmoll != 0) g += kappa * bv.b * c.dmoll * mollifier_row(model, frames, c).transpose();
  }
  return g;
}

double barrier_row_weight(double d, double dhat) {
  // 行的权重（阶段 4 用 b''；阶段 9 起用 b(d^2) 对 d 的精确二阶导数，笔记 notes/09_solver.md 第 7 节）。
  //   沿法向，d^2 的二阶导数是 2，所以 d^2/dd^2 [b(d^2)] = b'' (2d)^2 + 2 b'。写成行的外积 r^T r = 4 d^2 (n J)^T (n J)
  //   的系数：b'' + b' / (2 d^2)。它就是把 Gauss-Newton 丢掉的 2 kappa b' |Delta'|^2 里沿法向的那一部分补回来：
  //   d 很小时 b'' ≈ x^2/d^4 而 b'/(2 d^2) ≈ -x^2/(2 d^4)，GN 的权重比精确值大一倍，Newton 在法向上每次只走一半
  //   （线性收敛，比例 1/2）；补回来以后法向上是二次收敛。在 (0, dhat) 上它始终 > 0（测试 Weight.*），仍然半正定。
  //   切向的那部分 2 kappa b' |Delta'_t|^2 仍然丢掉（它 <= 0，丢掉只会让模型更硬，安全）。
  const BarrierValue bv = barrier(d * d, dhat * dhat);
  return std::max(bv.ddb + bv.db / (2 * d * d), 0.0);
}

std::vector<Row> contact_rows(const Model& model, const Frames& frames, const std::vector<Contact>& contacts, double kappa,
                              double dhat) {
  // 每个活跃接触一行：r 是接触行，权重 D = kappa m max(b'', 0)。
  std::vector<Row> rows;
  rows.reserve(contacts.size());
  for (const Contact& c : contacts) rows.push_back({contact_row(model, frames, c), kappa * c.moll * barrier_row_weight(c.d, dhat)});
  return rows;
}

MatrixXd contact_hessian(const Model& model, const Frames& frames, const std::vector<Contact>& contacts, double kappa,
                         double dhat) {
  // Gauss-Newton Hessian（阶段 4 笔记第 3 节）。精确的二阶导数是
  //   kappa b'' r^T r  +  kappa b' d^2(d^2)/d(delta)^2
  // 第一项是“行的外积”，半正定；第二项含 b' < 0 和顶点轨迹的曲率，可能不定。这里只保留第一项，
  // 并且用 max(b'', 0) 保险（b'' 在 (0, dhat) 上本来就是正的）。丢掉第二项的后果由阶段 6 的上界处理。
  // 带 mollifier 时只保留 m b'' r^T r：mollifier 自己的二阶项（b m'' rho^T rho，m'' < 0）和交叉项
  // （b' m' (r^T rho + rho^T r)，不定）都丢掉，模型仍然半正定；线搜索保证能量下降。
  MatrixXd H = MatrixXd::Zero(model.nv, model.nv);
  for (const Contact& c : contacts) {
    const RowVectorXd r = contact_row(model, frames, c);
    H += kappa * c.moll * barrier_row_weight(c.d, dhat) * r.transpose() * r;
  }
  return H;
}

MatrixXd contact_curvature(const Model& model, const VectorXd& q, const std::vector<Contact>& contacts, double kappa,
                           double dhat, double scale) {
  // 曲率上界（阶段 6 笔记第 3 节，书 §8.4 第 176–177 页）。推导的三步：
  //   1. 被丢掉的项沿方向 delta 是 2 kappa b' Delta . Delta''，Cauchy-Schwarz 给 |.| <= 2 kappa |b'| d |Delta''| = f |Delta''|。
  //   2. Delta'' 是两侧最近点的加速度之差，每侧 |X''| <= sum_{j,k} w_j w_k R_max(j,k)（两个转动的二阶导形如
  //      a_j x (a_k x (X - o_k))，大小不超过到较深那个锚点的力臂）。
  //   3. w_j w_k <= (w_j^2 + w_k^2) / 2 把交叉项摊到对角上：|X''| <= sum_j C_j w_j^2，C_j = sum_k R_max(j,k)。
  // 合起来 |被丢掉的项| <= delta^T (f sum_{两侧} sum_j C_j P_j) delta，P_j 是第 j 组自由度上的单位对角块。
  MatrixXd H = MatrixXd::Zero(model.nv, model.nv);
  if (scale <= 0) return H;
  for (const Contact& c : contacts) {
    const double f = 2 * kappa * c.moll * std::abs(barrier(c.d * c.d, dhat * dhat).db) * c.d;  // 法向力的大小
    for (int body : {c.bodyA, c.bodyB}) {
      if (body == 0) continue;
      const std::vector<ChainJoint> chain = chain_joints(model, q, body);
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

double initial_kappa(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                     const VectorXd& g_inertia, double dhat) {
  // 梯度平衡（阶段 4 笔记第 4 节，书 §8.3 第 174 页）：选 kappa 使 |g_e + kappa g_b|^2 最小。
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
