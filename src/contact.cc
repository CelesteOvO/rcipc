#include "rcipc/contact.h"

#include <cmath>
#include <initializer_list>
#include <limits>

#include <Eigen/Eigenvalues>

#include "rcipc/barrier.h"
#include "rcipc/chain.h"
#include "rcipc/distance.h"

// 接触。推导见 notes/04_ground.md（地面）和 notes/07_contact.md（刚体之间），测试见 tests/test_contact.cc。
namespace rcipc {

using Eigen::Matrix3d;
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

// ---------- 阶段 14：完整的接触 Hessian（笔记 notes/17_contact_hessian.md） ----------

namespace {

using Matrix12d = Eigen::Matrix<double, 12, 12>;
using Vector12d = Eigen::Matrix<double, 12, 1>;

// 模板顶点的世界坐标和两侧的半径之和（同 contact_distance）。
void stencil_points(const Model& model, const Frames& frames, const Contact& c, Vector3d* X, double* radii) {
  for (int i = 0; i < c.nstencil; ++i) {
    const Vertex& V = model.vertices[c.verts[i]];
    X[i] = world_point(frames, V.body, V.local);
  }
  *radii = c.type == ContactType::kGround ? model.vertices[c.verts[0]].radius
                                           : model.vertices[c.verts[0]].radius + model.vertices[c.verts[2]].radius;
}

// 模板顶点 i 在哪一侧（0 = A，1 = B）；世界上的顶点返回 -1。
int stencil_side(const Model& model, const Contact& c, int i) {
  const int body = model.vertices[c.verts[i]].body;
  if (body == 0) return -1;
  return body == c.bodyA ? 0 : 1;
}

// [v]x：v x w = skew(v) w。
Matrix3d skew(const Vector3d& v) {
  Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

// 几何项：f . (a x (b x rho)) = a^T (rho f^T) b - (a . b) tr(rho f^T)（inertia.cc 的 rot_rot）。对称化以后作为 omega 的
// 二次型，矩阵是 (Q + Q^T) / 2 - tr(Q) I，Q = sum_v rho_v f_v^T。
Matrix3d rotation_curvature(const Matrix3d& Q) { return 0.5 * (Q + Q.transpose()) - Q.trace() * Matrix3d::Identity(); }

// 一个接触的全部一阶、二阶导数（对模板顶点坐标，kappa 已乘上）：f 是 3 nstencil 的梯度，H 是 3 nstencil x 3 nstencil。
void vertex_derivatives(const Contact& c, const Vector3d* X, double radii, double kappa, double dhat, VectorXd* f,
                        MatrixXd* H) {
  const int n = 3 * c.nstencil;
  // 第 1 层：y = |Delta|^2 对顶点的导数。地面：y = (n . X - offset)^2，梯度 2 ell n，二阶 2 n n^T。
  const double ell = c.d + radii;  // |Delta|
  VectorXd gy(n);
  for (int i = 0; i < c.nstencil; ++i) gy.segment<3>(3 * i) = 2 * ell * c.w[i] * c.n;
  const MatrixXd Hy = squared_distance_hessian(c, X);
  // 第 2 层：x = d^2 = (sqrt(y) - R)^2。dx/dy = d / ell，d^2x/dy^2 = R / (2 ell^3)。
  const double alpha = c.d / ell;
  const BarrierValue bv = barrier(c.d * c.d, dhat * dhat);
  const VectorXd gb = bv.db * alpha * gy;
  const MatrixXd Hb = (bv.ddb * alpha * alpha + bv.db * radii / (2 * ell * ell * ell)) * gy * gy.transpose() + bv.db * alpha * Hy;
  if (c.type != ContactType::kEdgeEdge || (c.moll == 1 && c.dmoll == 0)) {
    *f = kappa * gb;
    *H = kappa * Hb;
    return;
  }
  // 第 3 层（边–边）：乘上 mollifier m(c_x)，c_x = |u x v|^2 = |u|^2 |v|^2 - (u . v)^2，u = X1 - X0，v = X3 - X2。
  //   dc/du = 2 |v|^2 u - 2 (u . v) v，dc/dv 对称；
  //   d^2c/du^2 = 2 |v|^2 I - 2 v v^T，d^2c/dv^2 = 2 |u|^2 I - 2 u u^T，d^2c/du dv = 4 u v^T - 2 v u^T - 2 (u . v) I。
  //   m 的二阶导数：c_x < eps 时 m'' = -2 / eps^2（make_ee 里 eps = 1e-3 |u|^2 |v|^2），否则 0。
  const Vector3d u = X[1] - X[0], v = X[3] - X[2];
  const double uv = u.dot(v);
  Eigen::Matrix<double, 6, 1> gc;
  gc << 2 * v.squaredNorm() * u - 2 * uv * v, 2 * u.squaredNorm() * v - 2 * uv * u;
  Eigen::Matrix<double, 6, 6> Cuv;
  Cuv.block<3, 3>(0, 0) = 2 * v.squaredNorm() * Matrix3d::Identity() - 2 * v * v.transpose();
  Cuv.block<3, 3>(3, 3) = 2 * u.squaredNorm() * Matrix3d::Identity() - 2 * u * u.transpose();
  Cuv.block<3, 3>(0, 3) = 4 * u * v.transpose() - 2 * v * u.transpose() - 2 * uv * Matrix3d::Identity();
  Cuv.block<3, 3>(3, 0) = Cuv.block<3, 3>(0, 3).transpose();
  Eigen::Matrix<double, 6, 12> T = Eigen::Matrix<double, 6, 12>::Zero();  // (u, v) = T X
  T.block<3, 3>(0, 0) = -Matrix3d::Identity();
  T.block<3, 3>(0, 3) = Matrix3d::Identity();
  T.block<3, 3>(3, 6) = -Matrix3d::Identity();
  T.block<3, 3>(3, 9) = Matrix3d::Identity();
  const VectorXd gcX = T.transpose() * gc;
  const MatrixXd HcX = T.transpose() * Cuv * T;
  const double eps = 1e-3 * u.squaredNorm() * v.squaredNorm();
  const double ddm = c.dmoll != 0 && eps > 0 ? -2.0 / (eps * eps) : 0.0;
  // E / kappa = m b：梯度 m b' + b m'，二阶 m H_b + b (m'' gc gc^T + m' H_c) + m' (gc gb^T + gb gc^T)。
  *f = kappa * (c.moll * gb + bv.b * c.dmoll * gcX);
  *H = kappa * (c.moll * Hb + bv.b * (ddm * gcX * gcX.transpose() + c.dmoll * HcX) +
                c.dmoll * (gcX * gb.transpose() + gb * gcX.transpose()));
}

// 力旋量 (f, tau)（参考点 ref）在刚体 body 链上各自由度上的值，加到 r 里（rows.cuh 的 wrench_on_dof）。
void add_wrench_row(const Model& model, const Frames& frames, int body, const Vector3d& ref, const Vector3d& f,
                    const Vector3d& tau, RowVectorXd* r) {
  for (int b = body; b != 0; b = model.bodies[b].parent) {
    const int j = model.bodies[b].joint;
    if (j < 0) continue;
    for (int d = model.joints[j].dadr; d < model.joints[j].dadr + model.joints[j].nv; ++d) {
      const Vector3d& a = frames.dof_axis[d];
      (*r)[d] += model.dof_rot[d] ? a.dot(tau - (frames.dof_anchor[d] - ref).cross(f)) : a.dot(f);
    }
  }
}

}  // namespace

MatrixXd squared_distance_hessian(const Contact& c, const Vector3d* X) {
  if (c.type == ContactType::kGround) return 2 * c.n * c.n.transpose();
  const int n = 3 * c.nstencil;
  // 固定权重的那一项：y = |sum_v w_v X_v|^2，d^2y/dX_v dX_u = 2 w_v w_u I。
  MatrixXd H = MatrixXd::Zero(n, n);
  for (int v = 0; v < c.nstencil; ++v)
    for (int u = 0; u < c.nstencil; ++u) H.block<3, 3>(3 * v, 3 * u) = 2 * c.w[v] * c.w[u] * Matrix3d::Identity();
  // 包络定理：y(X) = min_lam F(X, lam)，F = |sum_v w_v(lam) X_v|^2，w 对 lam 是线性的。
  //   d^2y = F_XX - F_{X lam} F_{lam lam}^-1 F_{lam X}。
  //   F_{lam_k} = 2 Delta . E_k，E_k = sum_v (dw_v/dlam_k) X_v；F_{lam_k lam_l} = 2 E_k . E_l；
  //   F_{X_v lam_k} = 2 w_v E_k + 2 (dw_v/dlam_k) Delta。
  // 哪些参数在动：点–三角形里重心坐标 > 0 的那几个（一个：顶点，不动；两个：边上，一个参数；三个：面内，两个参数），
  // 边–边里 s、t 在 (0, 1) 内部的。被夹在边界上的参数当常数（和距离函数的分段结构一致）。
  Vector3d Delta = Vector3d::Zero();
  for (int v = 0; v < c.nstencil; ++v) Delta += c.w[v] * X[v];
  double dw[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  int np = 0;
  if (c.type == ContactType::kPointTriangle) {
    int active[3], na = 0;
    for (int i = 0; i < 3; ++i)
      if (-c.w[1 + i] > 0) active[na++] = i;
    // 以 active[0] 为基：lam_{i0} = 1 - sum 其余，所以 w[1 + i0] = -lam_{i0} 对 lam_{ik} 的导数是 +1，w[1 + ik] 的是 -1。
    for (int k = 1; k < na; ++k) {
      dw[np][1 + active[0]] = 1;
      dw[np][1 + active[k]] = -1;
      ++np;
    }
  } else {
    if (c.w[1] > 0 && c.w[1] < 1) {  // s 在内部：w = (1 - s, s, ., .)
      dw[np][0] = -1;
      dw[np][1] = 1;
      ++np;
    }
    if (-c.w[3] > 0 && -c.w[3] < 1) {  // t 在内部：w = (., ., -(1 - t), -t)
      dw[np][2] = 1;
      dw[np][3] = -1;
      ++np;
    }
  }
  if (np == 0) return H;
  Vector3d E[2];
  MatrixXd G(n, np);  // 第 k 列是 F_{X lam_k}
  for (int k = 0; k < np; ++k) {
    E[k] = Vector3d::Zero();
    for (int v = 0; v < c.nstencil; ++v) E[k] += dw[k][v] * X[v];
    for (int v = 0; v < c.nstencil; ++v) G.block<3, 1>(3 * v, k) = 2 * c.w[v] * E[k] + 2 * dw[k][v] * Delta;
  }
  MatrixXd Fll(np, np);
  for (int k = 0; k < np; ++k)
    for (int l = 0; l < np; ++l) Fll(k, l) = 2 * E[k].dot(E[l]);
  // F_{lam lam} 退化（两条边平行、或者边长为 0）时用伪逆：特征值小于最大值 1e-10 倍的方向不动（mollifier 让这种接触的能量趟近 0）。
  const Eigen::SelfAdjointEigenSolver<MatrixXd> es(Fll);
  MatrixXd Finv = MatrixXd::Zero(np, np);
  const double lmax = es.eigenvalues().maxCoeff();
  for (int k = 0; k < np; ++k) {
    const double l = es.eigenvalues()[k];
    if (l > 1e-10 * lmax) Finv += es.eigenvectors().col(k) * es.eigenvectors().col(k).transpose() / l;
  }
  H -= G * Finv * G.transpose();
  return H;
}

MatrixXd contact_twist_jacobian(const Model& model, const Frames& frames, const Contact& c) {
  MatrixXd J = MatrixXd::Zero(12, model.nv);
  for (int side = 0; side < 2; ++side) {
    const int body = side == 0 ? c.bodyA : c.bodyB;
    if (body == 0) continue;
    const Vector3d& ref = frames.xpos[body];
    for (int b = body; b != 0; b = model.bodies[b].parent) {
      const int j = model.bodies[b].joint;
      if (j < 0) continue;
      for (int d = model.joints[j].dadr; d < model.joints[j].dadr + model.joints[j].nv; ++d) {
        const Vector3d& a = frames.dof_axis[d];
        if (model.dof_rot[d]) {
          J.block<3, 1>(6 * side, d) = (frames.dof_anchor[d] - ref).cross(a);
          J.block<3, 1>(6 * side + 3, d) = a;
        } else {
          J.block<3, 1>(6 * side, d) = a;
        }
      }
    }
  }
  return J;
}

Matrix12d contact_wrench_hessian(const Model& model, const Frames& frames, const Contact& c, double kappa, double dhat) {
  Vector3d X[4];
  double radii;
  stencil_points(model, frames, c, X, &radii);
  VectorXd f;
  MatrixXd HX;
  vertex_derivatives(c, X, radii, kappa, dhat, &f, &HX);
  // 顶点速度对旋量：X' = v + omega x (X - ref)，即 [I, -[X - ref]x]。
  Eigen::Matrix<double, Eigen::Dynamic, 12> Jv = Eigen::Matrix<double, Eigen::Dynamic, 12>::Zero(3 * c.nstencil, 12);
  Matrix3d Q[2] = {Matrix3d::Zero(), Matrix3d::Zero()};
  for (int i = 0; i < c.nstencil; ++i) {
    const int side = stencil_side(model, c, i);
    if (side < 0) continue;
    const Vector3d rho = X[i] - frames.xpos[side == 0 ? c.bodyA : c.bodyB];
    Jv.block<3, 3>(3 * i, 6 * side) = Matrix3d::Identity();
    Jv.block<3, 3>(3 * i, 6 * side + 3) = -skew(rho);
    Q[side] += rho * f.segment<3>(3 * i).transpose();
  }
  Matrix12d H = Jv.transpose() * HX * Jv;
  // 几何项：顶点随 omega 走弧线，exp([omega]) rho 的二阶项是 omega x (omega x rho) / 2，乘上力 f 求和。
  for (int side = 0; side < 2; ++side) H.block<3, 3>(6 * side + 3, 6 * side + 3) += rotation_curvature(Q[side]);
  return H;
}

void contact_rows_full(const Model& model, const Frames& frames, const std::vector<Contact>& contacts, double kappa,
                       double dhat, std::vector<Row>* rows, double rel_tol) {
  for (const Contact& c : contacts) {
    const Matrix12d H = contact_wrench_hessian(model, frames, c, kappa, dhat);
    const Eigen::SelfAdjointEigenSolver<Matrix12d> es(H);
    const double lmax = es.eigenvalues().maxCoeff();
    if (!(lmax > 0)) continue;
    for (int k = 0; k < 12; ++k) {
      const double l = es.eigenvalues()[k];
      if (l <= rel_tol * lmax) continue;
      const Vector12d u = es.eigenvectors().col(k);
      Row row;
      row.r = RowVectorXd::Zero(model.nv);
      row.D = l;
      for (int side = 0; side < 2; ++side) {
        const int body = side == 0 ? c.bodyA : c.bodyB;
        if (body == 0) continue;
        add_wrench_row(model, frames, body, frames.xpos[body], u.segment<3>(6 * side), u.segment<3>(6 * side + 3), &row.r);
      }
      rows->push_back(std::move(row));
    }
  }
}

MatrixXd contact_chain_curvature(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                                 double kappa, double dhat) {
  // 精确的 delta 空间 Hessian 里的几何项（inertia_curvature 的公式，Q_j = (X - o_j) f^T）减去旋量块里已经有的
  // （rot_rot(a_i, Q, a_j) 对称化，Q = sum (X - xpos) f^T）。记 F = sum f、P = sum (X - xpos) f^T：
  //   同一个关节的两个转动 i、j（锚点 o）：精确 sym rot_rot(a_i, P + (xpos - o) F^T, a_j)，块里 sym rot_rot(a_i, P, a_j)，
  //     剩 sym rot_rot(a_i, (xpos - o) F^T, a_j)；
  //   上游转动 up、下游转动 dn：精确 rot_rot(a_up, P + (xpos - o_dn) F^T, a_dn)（不对称化），块里 sym rot_rot(a_up, P, a_dn)；
  //   上游转动、下游滑动：精确 (a_up x a_dn) . F，块里 0；
  //   其余 0。
  const auto rot_rot = [](const Vector3d& a, const Matrix3d& Q, const Vector3d& b) { return a.dot(Q * b) - a.dot(b) * Q.trace(); };
  MatrixXd K = MatrixXd::Zero(model.nv, model.nv);
  for (const Contact& c : contacts) {
    Vector3d X[4];
    double radii;
    stencil_points(model, frames, c, X, &radii);
    VectorXd f;
    MatrixXd HX;
    vertex_derivatives(c, X, radii, kappa, dhat, &f, &HX);
    for (int side = 0; side < 2; ++side) {
      const int body = side == 0 ? c.bodyA : c.bodyB;
      if (body == 0) continue;
      const Vector3d& ref = frames.xpos[body];
      Vector3d F = Vector3d::Zero();
      Matrix3d P = Matrix3d::Zero();
      for (int i = 0; i < c.nstencil; ++i) {
        if (stencil_side(model, c, i) != side) continue;
        F += f.segment<3>(3 * i);
        P += (X[i] - ref) * f.segment<3>(3 * i).transpose();
      }
      std::vector<int> chain;  // 从刚体往根
      for (int b = body; b != 0; b = model.bodies[b].parent) {
        const int j = model.bodies[b].joint;
        if (j < 0) continue;
        for (int k = 0; k < model.joints[j].nv; ++k) chain.push_back(model.joints[j].dadr + k);
      }
      for (size_t u = 0; u < chain.size(); ++u) {
        for (size_t w = u; w < chain.size(); ++w) {
          const int i = chain[u], j = chain[w];
          double term = 0;
          if (model.dof_joint[i] == model.dof_joint[j]) {
            if (model.dof_rot[i] && model.dof_rot[j]) {
              const Matrix3d Qo = (ref - frames.dof_anchor[j]) * F.transpose();
              term = 0.5 * (rot_rot(frames.dof_axis[i], Qo, frames.dof_axis[j]) + rot_rot(frames.dof_axis[j], Qo, frames.dof_axis[i]));
            }
          } else {
            const int up = j, dn = i;  // chain 从刚体往根排，后面的在上游
            const Vector3d& au = frames.dof_axis[up];
            const Vector3d& ad = frames.dof_axis[dn];
            if (model.dof_rot[up]) {
              if (model.dof_rot[dn]) {
                term = rot_rot(au, P + (ref - frames.dof_anchor[dn]) * F.transpose(), ad) -
                       0.5 * (rot_rot(au, P, ad) + rot_rot(ad, P, au));
              } else {
                term = au.cross(ad).dot(F);
              }
            }
          }
          K(i, j) += term;
          if (i != j) K(j, i) += term;
        }
      }
    }
  }
  return K;
}

MatrixXd contact_hessian_exact(const Model& model, const Frames& frames, const std::vector<Contact>& contacts, double kappa,
                               double dhat) {
  MatrixXd H = contact_chain_curvature(model, frames, contacts, kappa, dhat);
  for (const Contact& c : contacts) {
    const MatrixXd J = contact_twist_jacobian(model, frames, c);
    H += J.transpose() * contact_wrench_hessian(model, frames, c, kappa, dhat) * J;
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
