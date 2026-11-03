#include "rcipc/inertia.h"

// body 形式的惯性项。推导见 notes/03_inertia.md，测试见 tests/test_inertia.cc。
//
// Eigen 的几个写法，下面会用到：
//   frames.xquat[b].toRotationMatrix()   刚体 b 的转动矩阵 R_b（3 x 3）
//   v.squaredNorm()                      |v|^2
//   A.trace()、A.transpose()             迹、转置
namespace rcipc {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

// vex(A) = (A23 - A32, A31 - A13, A12 - A21)，满足 tr([w]x A) = w . vex(A)（笔记第 3.2 节）。
// 公式里的下标从 1 开始，代码里从 0 开始。
Vector3d vex(const Matrix3d& A) {
  return Vector3d(A(1, 2) - A(2, 1), A(2, 0) - A(0, 2), A(0, 1) - A(1, 0));
}

}  // namespace

Vector3d body_com(const Model& model, const Frames& frames, int body) {
  // 质心是刚体上一个固定的点（在刚体坐标系里的坐标是 ipos），所以和顶点一样用 world_point 算。
  return world_point(frames, body, model.bodies[body].ipos);
}

InertiaTargets inertia_targets(const Model& model, const Frames& now, const Frames& prev, double h) {
  // 惯性目标（笔记第 2 节，书 §7.5 第 153 页）。
  //
  // 输入：now 是这一步开始时（q_n）的 FK，prev 是上一步（q_{n-1}）的 FK。
  // 0 号刚体是世界，不动，T.c[0]、T.A[0] 留着 0；循环从 1 号开始。
  const int nb = model.nbody();
  InertiaTargets T;
  T.c.assign(nb, Vector3d::Zero());
  T.A.assign(nb, Matrix3d::Zero());

  for (int b = 1; b < nb; ++b) {
    // 质心的目标：c~ = 2 c^n - c^{n-1} + g h^2。
    //   2 c^n - c^{n-1} = c^n + (c^n - c^{n-1})，即“按上一步的速度再走一步”；
    //   g h^2 是重力在这一步造成的位移（g 是向量，指向下方）。
    //   和阶段 1 的 x~ = x0 + h v0 - h^2 g 是同一个东西（那里 g 是标量，向上为正，所以是减号）。
    const Vector3d c_now = body_com(model, now, b);
    const Vector3d c_prev = body_com(model, prev, b);
    T.c[b] = 2 * c_now - c_prev + model.gravity * (h * h);

    // 转动的目标：A~ = 2 R^n - R^{n-1}，同样是“按上一步的变化再走一步”，只是作用在转动矩阵上。
    //   A~ 一般不是转动矩阵（两个转动矩阵的线性组合不在 SO(3) 里），这没关系：能量只要它是个 3 x 3 矩阵。
    //   最小化会自动选一个离它最近的转动（按 S 加权），思考题 1 讨论这一步“投影”带来的耗散。
    const Matrix3d R_now = now.xquat[b].toRotationMatrix();
    const Matrix3d R_prev = prev.xquat[b].toRotationMatrix();
    T.A[b] = 2 * R_now - R_prev;
  }
  return T;
}

double inertia_energy(const Model& model, const Frames& frames, const InertiaTargets& targets, double h) {
  // 惯性能量（笔记第 2 节）：
  //
  //   E = 1/(2 h^2) * sum_b [ m_b |c_b - c~_b|^2  +  tr( D S_b D^T ) ]，   D = R_b - A~_b
  //
  // 第 1 项就是阶段 1 的 m/(2h^2) (x - x~)^2 搬到三维；
  // 第 2 项是“刚体上所有质点离各自目标的加权距离”里和转动有关的那部分。两项的单位都是焦耳。
  double sum = 0;
  for (int b = 1; b < model.nbody(); ++b) {
    const Body& B = model.bodies[b];

    // 平移部分：质心离目标的距离平方，乘以质量。
    const Vector3d dc = body_com(model, frames, b) - targets.c[b];
    sum += B.mass * dc.squaredNorm();

    // 转动部分：D = R - A~ 是姿态离目标的“差”，S 是刚体的二阶矩（刚体坐标系，绕质心）。
    //   tr(D S D^T) = sum_p m_p |D X_p|^2：每个质点因为转动没到位而离目标的距离平方，加权求和。
    const Matrix3d D = frames.xquat[b].toRotationMatrix() - targets.A[b];
    sum += (D * B.S * D.transpose()).trace();
  }
  return sum / (2 * h * h);
}

VectorXd inertia_gradient(const Model& model, const Frames& frames, const InertiaTargets& targets, double h) {
  // 梯度 dE/d(delta)（笔记第 3 节，书 §7.5 第 154 页）。
  //
  // 思路：先在每个刚体上算“能量对刚体运动的导数”，再用 Jacobian 投影到关节上。
  //
  //   刚体 b 平移 dc、以角速度 w 转一点（dR = [w]x R）时，能量的变化是 dE_b = f . dc + tau . w，其中
  //     “力”   f   = m (c - c~) / h^2                 平移部分 m/(2h^2) |c - c~|^2 对 c 的导数
  //     “力矩” tau = vex(R S (R - A~)^T) / h^2         转动部分对 w 的导数（用 tr([w]x A) = w . vex(A)）
  //
  //   关节沿 delta 动时，质心的速度是 V delta、角速度是 W delta：
  //     V = point_jacobian(质心)，W = angular_jacobian(刚体)。
  //   代进去 dE_b = f . (V delta) + tau . (W delta) = (V^T f + W^T tau) . delta，
  //   所以这个刚体对梯度的贡献是 V^T f + W^T tau。
  //
  //   按列展开就是书上的写法：转动自由度 d 贡献 a_d . (tau + (c - o_d) x f)，平移自由度贡献 a_d . f。
  //   也就是“把刚体受的力和力矩沿链投影到每个关节轴上”。
  VectorXd g = VectorXd::Zero(model.nv);
  for (int b = 1; b < model.nbody(); ++b) {
    const Body& B = model.bodies[b];
    const Vector3d c = body_com(model, frames, b);
    const Matrix3d R = frames.xquat[b].toRotationMatrix();

    // 刚体上的力和力矩。
    const Vector3d f = B.mass * (c - targets.c[b]) / (h * h);
    const Vector3d tau = vex(R * B.S * (R - targets.A[b]).transpose()) / (h * h);

    // 投影到关节上。
    g += point_jacobian(model, frames, b, c).transpose() * f;
    g += angular_jacobian(model, frames, b).transpose() * tau;
  }
  return g;
}

MatrixXd inertia_hessian(const Model& model, const Frames& frames, double h) {
  // Gauss-Newton Hessian（笔记第 4 节，书 §7.5 第 155 页）。
  //
  // 把每个刚体的能量在当前点按 delta 展开到二阶。残差（c - c~、R - A~）对 delta 的一阶变化是
  // V delta 和 [W delta]x R；把这两个一阶变化代回平方项，得到
  //
  //   H = 1/h^2 * sum_b ( m_b V_b^T V_b  +  W_b^T I_b^world W_b )
  //
  // 被丢掉的是“残差乘以二阶导数”的项（Gauss-Newton 近似）。刚体正好在目标上时它是 0。
  //
  // 括号里正是关节空间的质量矩阵 M(q) 的定义：动能 1/2 v^T M v = sum_b (1/2 m |c'|^2 + 1/2 w^T I w)。
  // 所以 H = M(q) / h^2，测试拿它和 MuJoCo 的 mj_fullM 比较。
  // I_b^world = R I_b R^T 是世界系里绕质心的惯性张量（B.inertia 在刚体坐标系里）。
  MatrixXd H = MatrixXd::Zero(model.nv, model.nv);
  for (int b = 1; b < model.nbody(); ++b) {
    const Body& B = model.bodies[b];
    const Matrix3d R = frames.xquat[b].toRotationMatrix();
    const Matrix3d I_world = R * B.inertia * R.transpose();
    const Eigen::Matrix3Xd V = point_jacobian(model, frames, b, body_com(model, frames, b));
    const Eigen::Matrix3Xd W = angular_jacobian(model, frames, b);
    H += B.mass * V.transpose() * V + W.transpose() * I_world * W;
  }
  return H / (h * h);
}

}  // namespace rcipc
