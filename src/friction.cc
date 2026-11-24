#include "rcipc/friction.h"

#include <cmath>

#include <Eigen/Eigenvalues>

#include "rcipc/barrier.h"

// 滞后摩擦。推导见 notes/08_friction.md，测试见 tests/test_friction.cc。
namespace rcipc {

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

double friction_f0(double y, double eps) {
  if (y >= eps) return y;
  return -y * y * y / (3 * eps * eps) + y * y / eps + eps / 3;
}

double friction_f1(double y, double eps) {
  if (y >= eps) return 1.0;
  return 2 * y / eps - y * y / (eps * eps);
}

double friction_f1_over_y(double y, double eps) {
  if (y >= eps) return 1.0 / y;
  return 2 / eps - y / (eps * eps);
}

std::vector<FrictionContact> lag_friction(const Model& model, const Frames& frames, const std::vector<Contact>& contacts,
                                          double kappa, double dhat) {
  // 步首：法向力 lambda = -kappa m b'(x) 2d（阶段 4 笔记第 3 节的力），切向基由步首的法向构造，模板顶点的位置记下来。
  std::vector<FrictionContact> out;
  if (kappa <= 0) return out;
  for (const Contact& c : contacts) {
    FrictionContact f;
    f.contact = c;
    f.lambda = -kappa * c.kw * c.moll * barrier(c.d * c.d, dhat * dhat).db * 2 * c.d;
    if (f.lambda <= 0) continue;
    // 切向基：先挑一个不和 n 平行的轴，叉乘得到 t1，再叉乘得到 t2（书第 187 页的 lag_friction）。
    const Vector3d& n = c.n;
    const Vector3d helper = std::abs(n.x()) > 0.9 ? Vector3d::UnitY() : Vector3d::UnitX();
    f.t1 = n.cross(helper).normalized();
    f.t2 = n.cross(f.t1);
    for (int i = 0; i < c.nstencil; ++i) {
      const Vertex& V = model.vertices[c.verts[i]];
      f.X0[i] = world_point(frames, V.body, V.local);
    }
    out.push_back(f);
  }
  return out;
}

Vector2d friction_velocity(const Model& model, const Frames& frames, const FrictionContact& f, double h) {
  // V = sum_v w_v (X_v(q) - X_v^n) / h：两侧最近点（按步首的权重）的相对速度；u 是它在切平面里的分量。
  Vector3d V = Vector3d::Zero();
  for (int i = 0; i < f.contact.nstencil; ++i) {
    const Vertex& Vt = model.vertices[f.contact.verts[i]];
    V += f.contact.w[i] * (world_point(frames, Vt.body, Vt.local) - f.X0[i]) / h;
  }
  return Vector2d(f.t1.dot(V), f.t2.dot(V));
}

namespace {

// 切向行 rho_j = d(t_j . P)/d(delta)：接触行的算法，把 n 换成 t_j，不乘 2d。
RowVectorXd tangent_row(const Model& model, const Frames& frames, const FrictionContact& f, const Vector3d& t) {
  RowVectorXd r = RowVectorXd::Zero(model.nv);
  for (int i = 0; i < f.contact.nstencil; ++i) {
    const Vertex& V = model.vertices[f.contact.verts[i]];
    if (V.body == 0) continue;
    const Vector3d x = world_point(frames, V.body, V.local);
    r += f.contact.w[i] * (t.transpose() * point_jacobian(model, frames, V.body, x));
  }
  return r;
}

}  // namespace

double friction_energy(const Model& model, const Frames& frames, const std::vector<FrictionContact>& fc, double h,
                       double mu, double eps_v) {
  // 耗散功 h mu lambda |u|，|u| 光滑化成 f0(|u|)。单位：s * N * m/s = J。
  double e = 0;
  for (const FrictionContact& f : fc) e += h * mu * f.lambda * friction_f0(friction_velocity(model, frames, f, h).norm(), eps_v);
  return e;
}

VectorXd friction_gradient(const Model& model, const Frames& frames, const std::vector<FrictionContact>& fc, double h,
                           double mu, double eps_v) {
  // 链式法则：d/d(delta) [h mu lambda f0(|u|)] = h mu lambda f1(|u|) (u / |u|) . du/d(delta)，而 du_j/d(delta) = rho_j / h，
  // 所以 = sum_j c u_j rho_j^T，c = mu lambda f1(|u|) / |u|。c 在 u -> 0 时趋于 2 mu lambda / eps_v，有限。
  VectorXd g = VectorXd::Zero(model.nv);
  for (const FrictionContact& f : fc) {
    const Vector2d u = friction_velocity(model, frames, f, h);
    const double c = mu * f.lambda * friction_f1_over_y(u.norm(), eps_v);
    g += c * u[0] * tangent_row(model, frames, f, f.t1).transpose();
    g += c * u[1] * tangent_row(model, frames, f, f.t2).transpose();
  }
  return g;
}

namespace {

// 切平面里的 2 x 2 刚度 K（friction_hessian 的说明）。
Eigen::Matrix2d tangent_stiffness(const Vector2d& u, double eps_v) {
  const double y = u.norm();
  const double radial = friction_f1_over_y(y, eps_v);                         // f1 / |u|
  const double along = y < eps_v ? 2 / eps_v - 2 * y / (eps_v * eps_v) : 0.0;  // f1'
  Eigen::Matrix2d K = radial * Eigen::Matrix2d::Identity();
  if (y > 1e-12) {
    const Vector2d uh = u / y;
    K += (along - radial) * uh * uh.transpose();
  }
  return K;
}

}  // namespace

std::vector<Row> friction_rows(const Model& model, const Frames& frames, const std::vector<FrictionContact>& fc, double h,
                               double mu, double eps_v) {
  // K 是对称半正定的 2 x 2 矩阵，特征分解 K = sum_i k_i v_i v_i^T（k_i >= 0），每个特征向量给一行 v_i^T R。
  std::vector<Row> rows;
  rows.reserve(2 * fc.size());
  for (const FrictionContact& f : fc) {
    const Vector2d u = friction_velocity(model, frames, f, h);
    const Eigen::Matrix2d K = tangent_stiffness(u, eps_v);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(K);
    const RowVectorXd r1 = tangent_row(model, frames, f, f.t1), r2 = tangent_row(model, frames, f, f.t2);
    for (int i = 0; i < 2; ++i) {
      const Vector2d v = es.eigenvectors().col(i);
      rows.push_back({v[0] * r1 + v[1] * r2, mu * f.lambda * std::max(es.eigenvalues()[i], 0.0) / h});
    }
  }
  return rows;
}

MatrixXd friction_hessian(const Model& model, const Frames& frames, const std::vector<FrictionContact>& fc, double h,
                          double mu, double eps_v) {
  // 切平面里 f0(|u|) 的精确二阶导数是 (f1 / |u|) (1 - u^ u^T) + f1' u^ u^T（书第 186 页）：
  //   垂直于 u 的方向刚度 f1 / |u|，沿 u 的方向刚度 f1'（滑动时 f1' = 0：滑动摩擦力的大小不随速度变）。
  //   两个系数都 >= 0，所以它是半正定的，可以直接用。再乘以 (1/h)^2（u 对 delta 的导数是 rho / h）和 h mu lambda。
  // 书上的代码用 (f1 / |u|) 1 代替它（径向不小于精确值，是一个上界）；但撞击时 lambda 很大而 |u| 刚过 eps_v，
  // 上界在滑动方向上比真实值硬几百倍，Newton 会在那个方向上爬行（台阶场景一步跑满 50 次），所以这里用精确值。
  MatrixXd H = MatrixXd::Zero(model.nv, model.nv);
  for (const FrictionContact& f : fc) {
    const Eigen::Matrix2d K = tangent_stiffness(friction_velocity(model, frames, f, h), eps_v);
    Eigen::Matrix<double, 2, Eigen::Dynamic> R(2, model.nv);
    R.row(0) = tangent_row(model, frames, f, f.t1);
    R.row(1) = tangent_row(model, frames, f, f.t2);
    H += (mu * f.lambda / h) * (R.transpose() * K * R);
  }
  return H;
}

}  // namespace rcipc
