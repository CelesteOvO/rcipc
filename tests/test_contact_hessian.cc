// 阶段 14：完整的接触 Hessian（笔记 notes/17_contact_hessian.md）。
//
//   Distance.*  第 1 节：平方距离对模板顶点的二阶导数和有限差分一致（点–三角形的面内、边上、顶点，边–边的内部和端点）
//   Exact.*     第 2 节：精确的 Hessian（旋量块 + 链上的剩余项）和能量的有限差分一致：zoo（四种关节、地面）、
//               两个盒子（自由刚体，剩余项为 0）、台阶上的 mollifier
//   Rows.*      第 3 节：行形式就是每个接触的旋量块的半正定部分
//   Step.*      第 4 节：开着完整 Hessian 的时间步不穿透、静止在力平衡
#include <cmath>
#include <random>

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "rcipc/contact.h"
#include "rcipc/distance.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/solver.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// zoo.xml 的随机姿态，不相交（同 test_pairs.cc）。
VectorXd random_q(const Model& model, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  std::normal_distribution<double> n;
  VectorXd q = model.qpos0;
  for (int attempt = 0; attempt < 100; ++attempt) {
    for (const Joint& j : model.joints) {
      switch (j.type) {
        case JointType::kFree:
          q.segment<3>(j.qadr) = Vector3d(u(rng), u(rng), 1.5 + 0.3 * u(rng));
          quat_store(Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized(), &q[j.qadr + 3]);
          break;
        case JointType::kBall:
          quat_store(Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized(), &q[j.qadr]);
          break;
        case JointType::kHinge:
          q[j.qadr] = j.ref + kPi * u(rng);
          break;
        case JointType::kSlide:
          q[j.qadr] = j.ref + 0.3 * u(rng);
          break;
      }
    }
    if (min_distance(model, forward_kinematics(model, q)) > 1e-3) break;
  }
  return q;
}

// 模板顶点在 X 处时这个接触（同一对图元）的平方距离 |Delta|^2，最近点重新求（地面：到平面的距离，不减半径）。
double squared_distance_at(const Contact& c, const Vector3d* X, const Model& M) {
  if (c.type == ContactType::kGround) {
    const Plane& P = M.planes[c.plane];
    const double ell = P.normal.dot(X[0]) - P.offset;
    return ell * ell;
  }
  if (c.type == ContactType::kPointTriangle) {
    const Vector3d lam = point_triangle_closest(X[0], X[1], X[2], X[3]);
    return (X[0] - (lam[0] * X[1] + lam[1] * X[2] + lam[2] * X[3])).squaredNorm();
  }
  double s, t;
  segment_segment_closest(X[0], X[1], X[2], X[3], s, t);
  return (((1 - s) * X[0] + s * X[1]) - ((1 - t) * X[2] + t * X[3])).squaredNorm();
}

// 能量 E(q ⊕ delta) 对 delta 的有限差分 Hessian（同 test_inertia.cc）。
MatrixXd fd_hessian(const Model& M, const VectorXd& q, double kappa, double dhat, double eps) {
  const int n = M.nv;
  const auto phi = [&](const VectorXd& d) {
    const Frames F = forward_kinematics(M, integrate(M, q, d));
    return contact_energy(M, find_contacts(M, F, dhat), kappa, dhat);
  };
  MatrixXd H(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) {
      const VectorXd ei = eps * VectorXd::Unit(n, i), ej = eps * VectorXd::Unit(n, j);
      H(i, j) = H(j, i) = (phi(ei + ej) - phi(ei - ej) - phi(-ei + ej) + phi(-ei - ej)) / (4 * eps * eps);
    }
  return H;
}

// 点–三角形的子类型：重心坐标里正的有几个（1 顶点、2 边、3 面）；边–边：s、t 里在内部的有几个。
int subtype(const Contact& c) {
  int n = 0;
  if (c.type == ContactType::kPointTriangle) {
    for (int i = 0; i < 3; ++i) n += -c.w[1 + i] > 0;
  } else if (c.type == ContactType::kEdgeEdge) {
    n += c.w[1] > 0 && c.w[1] < 1;
    n += -c.w[3] > 0 && -c.w[3] < 1;
  }
  return n;
}

}  // namespace

// ---------- 第 1 节：平方距离的二阶导数 ----------

// zoo 的随机姿态、dhat = 10 让所有候选对都活跃：每个接触的 squared_distance_hessian 和 |Delta|^2 对顶点坐标的
// 有限差分一致。盒子、球、胶囊之间什么子类型都有。
TEST(Distance, HessianMatchesFiniteDifferenceForEverySubtype) {
  const Model Z = load("zoo.xml"), S = load("stack.xml");
  std::mt19937 rng(140);
  const double eps = 1e-5;
  int seen_pt[4] = {0, 0, 0, 0}, seen_ee[3] = {0, 0, 0};
  // zoo 的 4 个随机姿态，加上两个盒子交叉着叠的姿态（盒子的边和边在内部相交，zoo 的随机姿态里很少）。
  for (int pose = 0; pose < 5; ++pose) {
    const Model& M = pose < 4 ? Z : S;
    VectorXd q;
    if (pose < 4) {
      q = random_q(M, rng);
    } else {
      q = M.qpos0;
      q[7 + 2] = 0.1005 + 0.1 + 0.02 + 0.1;
      quat_store(Quaterniond(Eigen::AngleAxisd(0.6, Vector3d(0, 0, 1)) * Eigen::AngleAxisd(0.1, Vector3d(1, 0, 0))), &q[7 + 3]);
    }
    const Frames F = forward_kinematics(M, q);
    for (const Contact& c : find_contacts(M, F, pose < 4 ? 10.0 : 0.05)) {
      Vector3d X[4];
      for (int i = 0; i < c.nstencil; ++i)
        X[i] = world_point(F, M.vertices[c.verts[i]].body, M.vertices[c.verts[i]].local);
      const int n = 3 * c.nstencil;
      const MatrixXd H = squared_distance_hessian(c, X);
      MatrixXd fd(n, n);
      for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
          const auto phi = [&](double si, double sj) {
            Vector3d Y[4];
            for (int k = 0; k < c.nstencil; ++k) Y[k] = X[k];
            Y[i / 3][i % 3] += si;
            Y[j / 3][j % 3] += sj;
            return squared_distance_at(c, Y, M);
          };
          fd(i, j) = fd(j, i) = (phi(eps, eps) - phi(eps, -eps) - phi(-eps, eps) + phi(-eps, -eps)) / (4 * eps * eps);
        }
      // 球的边退化成一个点（两个端点是同一个顶点）：差分把它们分开扰动，最近点参数乱跳，这样的接触跳过
      // （模型里它们永远一起动，Hessian 把它当一个点是对的）。子类型边界上（一个重心坐标正好是 0）差分会跨过折角，也跳过。
      bool boundary = c.type == ContactType::kEdgeEdge && (c.verts[0] == c.verts[1] || c.verts[2] == c.verts[3]);
      if (c.type == ContactType::kPointTriangle)
        for (int i = 0; i < 3; ++i) boundary = boundary || (-c.w[1 + i] > 0 && -c.w[1 + i] < 1e-4);
      if (c.type == ContactType::kEdgeEdge)
        boundary = boundary || (c.w[1] > 0 && c.w[1] < 1e-4) || (c.w[1] < 1 && c.w[1] > 1 - 1e-4) ||
                   (-c.w[3] > 0 && -c.w[3] < 1e-4) || (-c.w[3] < 1 && -c.w[3] > 1 - 1e-4);
      if (boundary) continue;
      EXPECT_LT((H - fd).norm(), 1e-5 * fd.norm()) << "pose " << pose << ", type " << int(c.type) << ", subtype " << subtype(c);
      if (c.type == ContactType::kPointTriangle) ++seen_pt[subtype(c)];
      if (c.type == ContactType::kEdgeEdge) ++seen_ee[subtype(c)];
    }
  }
  EXPECT_GT(seen_pt[1], 0);  // 顶点
  EXPECT_GT(seen_pt[2], 0);  // 边上
  EXPECT_GT(seen_pt[3], 0);  // 面内
  EXPECT_GT(seen_ee[0] + seen_ee[1], 0);  // 端点对端点、端点对内部
  EXPECT_GT(seen_ee[2], 0);               // 两条边的内部
}

// 面内的点–三角形：沿三角形平面平移顶点，距离不变，所以切向的二阶导数是 0（固定权重时是 2）。
TEST(Distance, FaceContactHasNoTangentialCurvature) {
  Contact c;
  c.type = ContactType::kPointTriangle;
  c.nstencil = 4;
  const Vector3d X[4] = {Vector3d(0.1, 0.2, 0.3), Vector3d(-1, -1, 0), Vector3d(2, -1, 0), Vector3d(0, 2, 0)};
  const Vector3d lam = point_triangle_closest(X[0], X[1], X[2], X[3]);
  c.w[0] = 1;
  for (int i = 0; i < 3; ++i) c.w[1 + i] = -lam[i];
  const Vector3d Delta = X[0] - (lam[0] * X[1] + lam[1] * X[2] + lam[2] * X[3]);
  c.n = Delta.normalized();
  c.d = Delta.norm();
  const MatrixXd H = squared_distance_hessian(c, X);
  // 顶点 p 的 3 x 3 块：2 n n^T。
  EXPECT_LT((H.block<3, 3>(0, 0) - 2 * c.n * c.n.transpose()).norm(), 1e-12);
  // 整个模板一起平移，距离不变：H 乘上平移方向是 0。一起转动也不变，但转动的生成元 r_v = e x X_v 本身随 X 变，
  // 对 y(R X) = y(X) 求两次导数得 H r = [e x grad_v y]_v（不是 0；grad_v y = 2 w_v Delta）。
  for (int k = 0; k < 3; ++k) {
    VectorXd t = VectorXd::Zero(12);
    for (int v = 0; v < 4; ++v) t[3 * v + k] = 1;
    EXPECT_LT((H * t).norm(), 1e-12);
    VectorXd r = VectorXd::Zero(12), expect(12);
    for (int v = 0; v < 4; ++v) {
      r.segment<3>(3 * v) = Vector3d::Unit(k).cross(X[v]);
      expect.segment<3>(3 * v) = Vector3d::Unit(k).cross(2 * c.w[v] * Delta);
    }
    EXPECT_LT((H * r - expect).norm(), 1e-12);
  }
}

// ---------- 第 2 节：精确的 Hessian ----------

// zoo：四种关节、锚点不在原点、焊死的刚体、地面；dhat = 1 让一部分接触活跃。精确的 Hessian 和能量的有限差分一致，
// Gauss-Newton 的差得多（测试本身有区分度）。
TEST(Exact, MatchesFiniteDifferenceOnTheZoo) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(141);
  const double dhat = 1.0, kappa = 3.0;
  for (int pose = 0; pose < 4; ++pose) {
    const VectorXd q = random_q(M, rng);
    const Frames F = forward_kinematics(M, q);
    const std::vector<Contact> C = find_contacts(M, F, dhat);
    ASSERT_FALSE(C.empty());
    const MatrixXd H = contact_hessian_exact(M, F, C, kappa, dhat);
    const MatrixXd fd = fd_hessian(M, q, kappa, dhat, 1e-4);
    EXPECT_LT((H - H.transpose()).norm(), 1e-10 * H.norm()) << "pose " << pose;
    EXPECT_LT((H - fd).norm(), 1e-5 * fd.norm()) << "pose " << pose;
    const MatrixXd gn = contact_hessian(M, F, C, kappa, dhat);
    EXPECT_GT((gn - fd).norm(), 1e-3 * fd.norm()) << "pose " << pose;
  }
}

// 两个自由的盒子（stack.xml，上面的歪一点）：链上的剩余项是 0，旋量块就是全部。
TEST(Exact, ChainTermVanishesForFreeBodies) {
  const Model M = load("stack.xml");
  const double dhat = 0.05, kappa = 1e3;
  VectorXd q = M.qpos0;
  q[2] += 0.1;  // 下面的盒子抬离地面（它原来离地 0.5 mm，差分在这么陡的障碍上不准）
  q[7 + 2] = 0.1005 + 0.1 + 0.1 + 0.02 + 0.1;
  quat_store(Quaterniond(Eigen::AngleAxisd(0.6, Vector3d(0, 0, 1)) * Eigen::AngleAxisd(0.1, Vector3d(1, 0, 0))), &q[7 + 3]);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  ASSERT_FALSE(C.empty());
  EXPECT_LT(contact_chain_curvature(M, F, C, kappa, dhat).norm(), 1e-9 * contact_hessian_exact(M, F, C, kappa, dhat).norm());
  const MatrixXd H = contact_hessian_exact(M, F, C, kappa, dhat);
  const MatrixXd fd = fd_hessian(M, q, kappa, dhat, 1e-5);
  EXPECT_LT((H - fd).norm(), 1e-5 * fd.norm());
}

// 台阶：盒子的底边和台阶的顶边几乎平行，mollifier 在起作用（同 test_pairs.cc 的 Row.Mollified*）。
// m、m'、m'' 和交叉项都在，Hessian 和有限差分一致。
TEST(Exact, MollifiedEdgeEdgeMatchesFiniteDifference) {
  const Model M = load("ledge.xml");
  const double dhat = 0.05, kappa = 1e3;
  VectorXd q = M.qpos0;
  q[0] = 0.4;
  q[2] = 0.2 + 0.1 + 0.02;
  quat_store(Quaterniond(Eigen::AngleAxisd(0.02, Vector3d(0, 0, 1))), &q[3]);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  int mollified = 0;
  for (const Contact& c : C) mollified += c.type == ContactType::kEdgeEdge && c.moll < 1;
  ASSERT_GT(mollified, 0);
  const MatrixXd H = contact_hessian_exact(M, F, C, kappa, dhat);
  const MatrixXd fd = fd_hessian(M, q, kappa, dhat, 1e-5);
  EXPECT_LT((H - fd).norm(), 1e-5 * fd.norm());
}

// ---------- 第 3 节：行形式 ----------

// 每个接触的行加起来是 J^T [H12]_+ J（只留正特征值的那部分）；全部行的和半正定，并且在 H12 本来就半正定的接触上等于精确值。
TEST(Rows, AreTheProjectedWrenchBlocks) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(142);
  const double dhat = 1.0, kappa = 3.0;
  const VectorXd q = random_q(M, rng);
  const Frames F = forward_kinematics(M, q);
  const std::vector<Contact> C = find_contacts(M, F, dhat);
  ASSERT_FALSE(C.empty());
  MatrixXd expect = MatrixXd::Zero(M.nv, M.nv);
  for (const Contact& c : C) {
    const Eigen::Matrix<double, 12, 12> H = contact_wrench_hessian(M, F, c, kappa, dhat);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es(H);
    Eigen::Matrix<double, 12, 12> Hp = Eigen::Matrix<double, 12, 12>::Zero();
    for (int k = 0; k < 12; ++k)
      if (es.eigenvalues()[k] > 1e-12 * es.eigenvalues().maxCoeff())
        Hp += es.eigenvalues()[k] * es.eigenvectors().col(k) * es.eigenvectors().col(k).transpose();
    const MatrixXd J = contact_twist_jacobian(M, F, c);
    expect += J.transpose() * Hp * J;
  }
  NewtonSystem sys;
  sys.Hk = MatrixXd::Zero(M.nv, M.nv);
  contact_rows_full(M, F, C, kappa, dhat, &sys.rows);
  EXPECT_LE(sys.rows.size(), 12 * C.size());
  const MatrixXd H = dense(sys);
  EXPECT_LT((H - expect).norm(), 1e-9 * expect.norm());
  const VectorXd eig = Eigen::SelfAdjointEigenSolver<MatrixXd>(H).eigenvalues();
  EXPECT_GE(eig.minCoeff(), -1e-12 * eig.maxCoeff());
}

// 顶点的 point_jacobian 等于 [I, -[X - xpos]x] 乘旋量 Jacobian。
TEST(Rows, TwistJacobianReproducesPointJacobians) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(143);
  const VectorXd q = random_q(M, rng);
  const Frames F = forward_kinematics(M, q);
  for (const Contact& c : find_contacts(M, F, 10.0)) {
    const MatrixXd J = contact_twist_jacobian(M, F, c);
    for (int i = 0; i < c.nstencil; ++i) {
      const Vertex& V = M.vertices[c.verts[i]];
      if (V.body == 0) continue;
      const int side = V.body == c.bodyA ? 0 : 1;
      const Vector3d X = world_point(F, V.body, V.local);
      const Vector3d rho = X - F.xpos[V.body];
      const MatrixXd Jp = point_jacobian(M, F, V.body, X);
      MatrixXd Jq = J.middleRows(6 * side, 3);
      for (int d = 0; d < M.nv; ++d) Jq.col(d) += J.block<3, 1>(6 * side + 3, d).cross(rho);
      EXPECT_LT((Jp - Jq).norm(), 1e-12 * (1 + Jp.norm()));
    }
  }
}

// ---------- 第 4 节：时间步 ----------

// 堆叠的两个盒子（stack.xml）开着完整 Hessian 走 100 步：不穿透，最后静止；Newton 次数不比 Gauss-Newton 明显多。
TEST(Step, StackRestsWithTheFullHessian) {
  const Model M = load("stack.xml");
  int newton[2] = {0, 0};
  for (int full = 0; full < 2; ++full) {
    StepParams P;
    P.h = 0.005;
    P.dhat = 0.01;
    P.mu = 0;
    P.full_contact_hessian = full != 0;
    SolverState S;
    VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
    for (int step = 0; step < 100; ++step) {
      const StepResult r = rcipc::step(M, P, q, v, &S);
      ASSERT_GT(r.min_distance, 0) << "full " << full << ", step " << step;
      q = r.q;
      v = r.v;
      newton[full] += r.newton;
    }
    EXPECT_LT(v.norm(), 1e-3) << "full " << full;
  }
  EXPECT_LE(newton[1], 1.1 * newton[0]);  // 两个盒子的堆叠太简单，两种 Hessian 都是每步 2–3 次，只要求不变差
}
