#include "rcipc/inertia.h"

#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

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

namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

Matrix3d skew(const Vector3d& v) {
  Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

// 自由度 d 的运动子空间（世界系，在世界原点取的旋量 (w, v_O)，点 x 的速度是 v_O + w x x）：
//   转动：(a, o x a)（绕过 o、方向 a 的轴转：x 的速度 a x (x - o) = a x x + o x a）；平移：(0, a)。
Vector6d motion(const Model& model, const Frames& frames, int d) {
  const Vector3d& a = frames.dof_axis[d];
  Vector6d S;
  if (model.dof_rot[d]) S << a, frames.dof_anchor[d].cross(a);
  else S << Vector3d::Zero(), a;
  return S;
}

// a^T Q b - (a . b) tr Q = sum_p f_p . (a x (b x r_p))，Q = sum_p r_p f_p^T（笔记 16 第 2 节）。
double rot_rot(const Vector3d& a, const Matrix3d& Q, const Vector3d& b) {
  return a.dot(Q * b) - a.dot(b) * Q.trace();
}

}  // namespace

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
  //
  // 算法（复合刚体算法 CRBA）：直接按上式对每个刚体做整行宽的 V^T V，代价 O(刚体数 x nv^2)，163 节的链
  // （nv 489）一次要 96 ms。换成空间惯量：刚体 b 的动能写成 1/2 T^T I_b T，T = (w, v_O) 是它在世界原点的旋量，
  //   I_b = [[I_c + m [c]x^T [c]x, m [c]x], [m [c]x^T, m 1]]（I_c = R I R^T 是世界系里绕质心的惯性张量）；
  // T = sum_{d 在 b 的链上} S_d delta_d，所以 M_ij = sum_b S_i^T I_b S_j，b 取 i、j 两个自由度共同的下游刚体。
  // i 在 j 的上游（或同一个关节）时，共同的下游就是 j 的刚体的子树，于是 M_ij = S_i^T I^c S_j，
  // I^c = 子树里所有刚体的 I_b 之和（复合惯量，从叶子往根累加一遍）。不在同一条链上的两个自由度 M_ij = 0。
  // 代价 O(刚体数 + nv x 链的深度)。
  const int nb = model.nbody();
  std::vector<Matrix6d> Ic(nb, Matrix6d::Zero());
  for (int b = 1; b < nb; ++b) {
    const Body& B = model.bodies[b];
    const Matrix3d R = frames.xquat[b].toRotationMatrix();
    const Matrix3d cx = skew(body_com(model, frames, b));
    Matrix6d& I = Ic[b];
    I.topLeftCorner<3, 3>() = R * B.inertia * R.transpose() + B.mass * cx.transpose() * cx;
    I.topRightCorner<3, 3>() = B.mass * cx;
    I.bottomLeftCorner<3, 3>() = B.mass * cx.transpose();
    I.bottomRightCorner<3, 3>() = B.mass * Matrix3d::Identity();
  }
  for (int b = nb - 1; b >= 1; --b) {  // 父刚体的编号总比子刚体小，倒着扫一遍就是从叶子往根
    const int p = model.bodies[b].parent;
    if (p > 0) Ic[p] += Ic[b];
  }
  MatrixXd H = MatrixXd::Zero(model.nv, model.nv);
  for (int j = 0; j < model.nv; ++j) {
    const Vector6d F = Ic[model.dof_body[j]] * motion(model, frames, j);
    for (int x = model.dof_body[j]; x != 0; x = model.bodies[x].parent) {
      const int jt = model.bodies[x].joint;
      if (jt < 0) continue;
      for (int k = 0; k < model.joints[jt].nv; ++k) {
        const int i = model.joints[jt].dadr + k;
        H(i, j) = H(j, i) = motion(model, frames, i).dot(F);
      }
    }
  }
  return H / (h * h);
}

MatrixXd inertia_curvature(const Model& model, const Frames& frames, const InertiaTargets& targets, double h,
                          bool rot) {
  // Gauss-Newton 丢掉的那一项（笔记 16 第 2 节）。
  //
  // 把刚体看成一群质点：E_b = sum_p m_p / (2 h^2) |x_p - x~_p|^2（展开就是 inertia_energy 里的两项）。
  // 二阶导数 = sum_p m_p / h^2 (J_p^T J_p + (x_p - x~_p) . d^2 x_p)，前一半是 M / h^2，后一半就是 K：
  //   K_ij = sum_p f_p . d^2 x_p / (d delta_i d delta_j)，f_p = m_p (x_p - x~_p) / h^2。
  //
  // 质点的二阶导数（自由度 i 在 j 的上游，也就是 i 的刚体是 j 的刚体的祖先）：
  //   i 转动、j 转动：a_i x (a_j x (x - o_j))   （转 i 时，下游的轴 a_j、锚点 o_j、点 x 一起绕 a_i 转）
  //   i 转动、j 平移：a_i x a_j
  //   i 平移：0                                  （平移不改变下游的轴，也不改变 x - o_j）
  // 同一个关节里的两个转动自由度（球关节、自由关节）：增量是 R exp([delta])，二阶导数取对称的一半
  //   1/2 (a_i x (a_j x r) + a_j x (a_i x r))；铰链 i = j 就是 a x (a x r)。同一个关节里有平移的：0。
  //
  // 对质点求和：sum_p f_p . (a x (b x r_p)) = a^T Q b - (a . b) tr Q，Q = sum_p r_p f_p^T，r_p = x_p - o。
  //   x_p = c + R X_p、x~_p = c~ + A~ X_p，sum_p m_p X_p = 0，sum_p m_p X_p X_p^T = S，所以
  //   Q = (c - o) f^T + P，f = m (c - c~) / h^2（刚体的“力”，同 inertia_gradient），P = R S (R - A~)^T / h^2；
  //   sum_p f_p = f。
  //
  // 算法：一对自由度 (i 在上游, j 在下游) 的贡献是 j 的刚体的子树里所有刚体的 Q（或 f）之和。直接逐个刚体、对它链上的
  // 每一对自由度累加，代价 O(刚体数 x 深度^2)，163 节的链一次 33 ms。先从叶子往根累加子树和
  //   F_sub = sum f_b，C_sub = sum (c_b f_b^T + P_b)，
  // 那么 sum_b Q_b = sum_b ((c_b - o_j) f_b^T + P_b) = C_sub - o_j F_sub^T，每一对只算一次：O(刚体数 + nv x 深度)。
  const int nv = model.nv, nb = model.nbody();
  std::vector<Vector3d> Fs(nb, Vector3d::Zero());
  std::vector<Matrix3d> Cs(nb, Matrix3d::Zero());
  for (int b = 1; b < nb; ++b) {
    const Body& B = model.bodies[b];
    const Vector3d c = body_com(model, frames, b);
    const Matrix3d R = frames.xquat[b].toRotationMatrix();
    Fs[b] = B.mass * (c - targets.c[b]) / (h * h);
    Cs[b] = c * Fs[b].transpose();
    if (rot) Cs[b] += R * B.S * (R - targets.A[b]).transpose() / (h * h);
  }
  for (int b = nb - 1; b >= 1; --b) {
    const int p = model.bodies[b].parent;
    if (p > 0) {
      Fs[p] += Fs[b];
      Cs[p] += Cs[b];
    }
  }
  MatrixXd K = MatrixXd::Zero(nv, nv);
  for (int dn = 0; dn < nv; ++dn) {
    const int x = model.dof_body[dn], jn = model.dof_joint[dn];
    const Vector3d& ad = frames.dof_axis[dn];
    const Matrix3d Q = Cs[x] - frames.dof_anchor[dn] * Fs[x].transpose();  // 只在 dn 是转动时用
    // 同一个关节里编号不大于 dn 的自由度：只有两个都是转动时不为 0（锚点相同），二阶导数取对称的一半。
    if (model.dof_rot[dn]) {
      for (int i = model.joints[jn].dadr; i <= dn; ++i) {
        if (!model.dof_rot[i]) continue;
        const Vector3d& ai = frames.dof_axis[i];
        const double term = 0.5 * (rot_rot(ai, Q, ad) + rot_rot(ad, Q, ai));
        K(i, dn) += term;
        if (i != dn) K(dn, i) += term;
      }
    }
    // 上游关节的自由度 up：up 是平移时为 0。
    for (int y = model.bodies[x].parent; y != 0; y = model.bodies[y].parent) {
      const int ju = model.bodies[y].joint;
      if (ju < 0) continue;
      for (int k = 0; k < model.joints[ju].nv; ++k) {
        const int up = model.joints[ju].dadr + k;
        if (!model.dof_rot[up]) continue;
        const Vector3d& au = frames.dof_axis[up];
        const double term = model.dof_rot[dn] ? rot_rot(au, Q, ad) : au.cross(ad).dot(Fs[x]);
        K(up, dn) += term;
        K(dn, up) += term;
      }
    }
  }
  return K;
}

namespace {

// floor * Mh + K 在自由度 D（一棵树）的块上正定吗：Jacobi 缩放以后 LLT 成功。
bool tree_block_pd(const MatrixXd& Mh, const MatrixXd& K, const std::vector<int>& D, double floor) {
  const int n = static_cast<int>(D.size());
  MatrixXd B(n, n);
  VectorXd s(n);
  for (int i = 0; i < n; ++i) s[i] = 1.0 / std::sqrt(Mh(D[i], D[i]));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) B(i, j) = s[i] * (floor * Mh(D[i], D[j]) + K(D[i], D[j])) * s[j];
  return Eigen::LLT<MatrixXd>(B).info() == Eigen::Success;
}

void add_tree_block(const MatrixXd& K, const std::vector<int>& D, MatrixXd* H) {
  for (int i : D)
    for (int j : D) (*H)(i, j) += K(i, j);
}

std::vector<std::vector<int>> tree_dofs(const Model& model) {
  std::vector<std::vector<int>> dofs(model.ntree);
  for (int d = 0; d < model.nv; ++d) dofs[model.dof_tree[d]].push_back(d);
  return dofs;
}

}  // namespace

int add_inertia_curvature(const Model& model, const MatrixXd& Mh, const MatrixXd& K, MatrixXd* H, double floor) {
  // 一棵树一棵树地试：floor * Mh + K 在这棵树的块上正定（Jacobi 缩放以后 LLT 成功）才把 K 加进 H。
  int dropped = 0;
  for (const std::vector<int>& D : tree_dofs(model)) {
    if (!tree_block_pd(Mh, K, D, floor)) {
      ++dropped;
      continue;
    }
    add_tree_block(K, D, H);
  }
  return dropped;
}

}  // namespace rcipc
