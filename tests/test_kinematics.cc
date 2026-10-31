// 阶段 2：关节坐标的运动学，拿 MuJoCo 当参照答案（笔记 notes/02_kinematics.md）。
//
// 测试分五组，和笔记的小节对应：
//   Model.*          第 1 节：从 mjModel 读出的树结构
//   Integrate.*、Differentiate.*  第 2 节：q 的加法和减法，对比 mj_integratePos、mj_differentiatePos
//   Fk.*             第 3 节：正运动学，对比 mj_kinematics 的 xpos、xquat
//   Jacobian.*       第 4 节：点的 Jacobian，对比 mj_jac，再用有限差分独立检查
//
// 模型是 models/zoo.xml：四种关节、两棵树、有分叉、有偏移和非零 ref。
// 每个测试都在 20 个随机姿态上比较，随机数种子固定，所以每次运行结果一样。
#include "rcipc/kinematics.h"

#include <cmath>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "rcipc/model.h"
#include "rcipc/quat.h"

using namespace rcipc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kPoses = 20;

// 一个 MuJoCo 模型，加上它的 mjData 和我们的 Model。
struct Zoo {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/zoo.xml");
  mjData* d = mj_makeData(m.get());
  Model model = from_mujoco(m.get());
  ~Zoo() { mj_deleteData(d); }

  // 名字 -> 编号。
  int body(const char* name) const { return mj_name2id(m.get(), mjOBJ_BODY, name); }
  int dof(const char* joint_name) const { return m->jnt_dofadr[mj_name2id(m.get(), mjOBJ_JOINT, joint_name)]; }

  // 让 MuJoCo 在 q 上做正运动学（mj_comPos 是 mj_jac 要用的）。
  void mj_fk(const VectorXd& q) {
    for (int i = 0; i < m->nq; ++i) d->qpos[i] = q[i];
    mj_kinematics(m.get(), d);
    mj_comPos(m.get(), d);
  }
};

// 随机单位四元数：四个正态分布的数归一化，在所有转动上是均匀的。
Quaterniond random_quat(std::mt19937& rng) {
  std::normal_distribution<double> n;
  return Quaterniond(n(rng), n(rng), n(rng), n(rng)).normalized();
}

// 随机姿态：铰链在 ref 附近转 [-pi, pi]，滑动伸缩 [-0.3, 0.3] m，球关节和自由关节随机转动。
VectorXd random_q(const Model& model, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  VectorXd q = model.qpos0;
  for (const Joint& j : model.joints) {
    switch (j.type) {
      case JointType::kFree:
        q.segment<3>(j.qadr) = Vector3d(u(rng), u(rng), u(rng));
        quat_store(random_quat(rng), &q[j.qadr + 3]);
        break;
      case JointType::kBall:
        quat_store(random_quat(rng), &q[j.qadr]);
        break;
      case JointType::kHinge:
        q[j.qadr] = j.ref + kPi * u(rng);
        break;
      case JointType::kSlide:
        q[j.qadr] = j.ref + 0.3 * u(rng);
        break;
    }
  }
  return q;
}

// 随机增量：每个分量在 [-1, 1] 里，所以转动部分的长度 <= sqrt(3) < pi。
VectorXd random_delta(int nv, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  VectorXd delta(nv);
  for (int i = 0; i < nv; ++i) delta[i] = u(rng);
  return delta;
}

// 两个四元数表示的转动差多少（q 和 -q 算同一个）。
double quat_diff(const Quaterniond& a, const Quaterniond& b) {
  return std::min((a.coeffs() - b.coeffs()).norm(), (a.coeffs() + b.coeffs()).norm());
}

// 两个 q 差多少：标量部分直接比，四元数部分按 quat_diff 比。
double q_diff(const Model& model, const VectorXd& a, const VectorXd& b) {
  double err = 0;
  for (const Joint& j : model.joints) {
    switch (j.type) {
      case JointType::kFree:
        err = std::max(err, (a.segment<3>(j.qadr) - b.segment<3>(j.qadr)).norm());
        err = std::max(err, quat_diff(quat_load(&a[j.qadr + 3]), quat_load(&b[j.qadr + 3])));
        break;
      case JointType::kBall:
        err = std::max(err, quat_diff(quat_load(&a[j.qadr]), quat_load(&b[j.qadr])));
        break;
      default:
        err = std::max(err, std::abs(a[j.qadr] - b[j.qadr]));
    }
  }
  return err;
}

// 我们的 Frames 和 MuJoCo 的 xpos、xquat 差多少（所有刚体里最大的）。
double frames_vs_mujoco(const Frames& F, const Zoo& z) {
  double err = 0;
  for (int b = 0; b < z.m->nbody; ++b) {
    err = std::max(err, (F.xpos[b] - Vector3d(z.d->xpos + 3 * b)).norm());
    err = std::max(err, quat_diff(F.xquat[b], quat_load(z.d->xquat + 4 * b)));
  }
  return err;
}

}  // namespace

// ---------- 第 1 节：树结构 ----------

TEST(Model, ReadsTheZoo) {
  Zoo z;
  const Model& M = z.model;
  EXPECT_EQ(M.nbody(), 9);  // 世界 + 8 个刚体
  EXPECT_EQ(M.joints.size(), 7u);
  EXPECT_EQ(M.nq, 7 + 4 + 1 + 1 + 1 + 1 + 4);
  EXPECT_EQ(M.nv, 6 + 3 + 1 + 1 + 1 + 1 + 3);
  EXPECT_EQ(M.bodies[0].parent, -1);
  EXPECT_EQ(M.bodies[z.body("hand")].joint, -1);  // 焊死的刚体没有关节
  EXPECT_EQ(M.bodies[z.body("finger")].parent, z.body("hand"));
  for (int b = 1; b < M.nbody(); ++b) EXPECT_LT(M.bodies[b].parent, b);  // 父刚体编号总在前面

  const Joint& shoulder = M.joints[M.bodies[z.body("arm")].joint];
  EXPECT_EQ(shoulder.type, JointType::kHinge);
  EXPECT_DOUBLE_EQ(shoulder.ref, 0.3);
  EXPECT_NEAR(shoulder.axis.norm(), 1.0, 1e-15);  // MuJoCo 把轴 (0, 1, 1) 归一化了

  // 自由关节：前 3 个自由度是平移，后 3 个是转动。
  const int root = z.dof("root");
  for (int k = 0; k < 6; ++k) EXPECT_EQ(M.dof_rot[root + k], k >= 3);
  EXPECT_FALSE(M.dof_rot[z.dof("elbow")]);
  EXPECT_EQ(M.dof_body[z.dof("twist") + 2], z.body("bob"));
}

// ---------- 第 2 节：q 的加法和减法 ----------

// delta = 0 时 q ⊕ delta = q。
TEST(Integrate, ZeroStepChangesNothing) {
  Zoo z;
  std::mt19937 rng(10);
  const VectorXd q = random_q(z.model, rng);
  EXPECT_LT(q_diff(z.model, integrate(z.model, q, VectorXd::Zero(z.model.nv)), q), 1e-15);
}

// 和 mj_integratePos 一致（它是原地修改 qpos 的）。
TEST(Integrate, MatchesMujoco) {
  Zoo z;
  std::mt19937 rng(11);
  for (int i = 0; i < kPoses; ++i) {
    const VectorXd q = random_q(z.model, rng);
    const VectorXd delta = random_delta(z.model.nv, rng);
    const double s = 0.7;
    VectorXd expect = q;
    mj_integratePos(z.m.get(), expect.data(), delta.data(), s);
    EXPECT_LT(q_diff(z.model, integrate(z.model, q, delta, s), expect), 1e-14) << "pose " << i;
  }
}

// 和 mj_differentiatePos(dt = 1) 一致。
TEST(Differentiate, MatchesMujoco) {
  Zoo z;
  std::mt19937 rng(12);
  for (int i = 0; i < kPoses; ++i) {
    const VectorXd q0 = random_q(z.model, rng), q1 = random_q(z.model, rng);
    VectorXd expect(z.model.nv);
    mj_differentiatePos(z.m.get(), expect.data(), 1.0, q0.data(), q1.data());
    EXPECT_LT((differentiate(z.model, q0, q1) - expect).norm(), 1e-12) << "pose " << i;
  }
}

// (q ⊕ delta) ⊖ q = delta：减法是加法的逆（转动部分长度 < pi 时）。
TEST(Differentiate, InvertsIntegrate) {
  Zoo z;
  std::mt19937 rng(13);
  for (int i = 0; i < kPoses; ++i) {
    const VectorXd q = random_q(z.model, rng);
    const VectorXd delta = random_delta(z.model.nv, rng);
    const VectorXd back = differentiate(z.model, q, integrate(z.model, q, delta));
    EXPECT_LT((back - delta).norm(), 1e-13) << "pose " << i;
  }
}

// q ⊕ (q1 ⊖ q) = q1：反过来也成立，不需要任何条件。
TEST(Differentiate, IntegrateInvertsDifferentiate) {
  Zoo z;
  std::mt19937 rng(14);
  for (int i = 0; i < kPoses; ++i) {
    const VectorXd q0 = random_q(z.model, rng), q1 = random_q(z.model, rng);
    EXPECT_LT(q_diff(z.model, integrate(z.model, q0, differentiate(z.model, q0, q1)), q1), 1e-14) << "pose " << i;
  }
}

// ---------- 第 3 节：正运动学 ----------

// q = qpos0 时，每个刚体都在 MJCF 写的位置上。
TEST(Fk, MatchesMujocoAtQpos0) {
  Zoo z;
  z.mj_fk(z.model.qpos0);
  EXPECT_LT(frames_vs_mujoco(forward_kinematics(z.model, z.model.qpos0), z), 1e-15);
}

// 随机姿态下和 mj_kinematics 一致。
TEST(Fk, MatchesMujocoAtRandomPoses) {
  Zoo z;
  std::mt19937 rng(20);
  for (int i = 0; i < kPoses; ++i) {
    const VectorXd q = random_q(z.model, rng);
    z.mj_fk(q);
    EXPECT_LT(frames_vs_mujoco(forward_kinematics(z.model, q), z), 1e-14) << "pose " << i;
  }
}

// q 里的四元数没有归一化时（例如被乘了 3），结果和归一化以后一样。MuJoCo 也是这样做的。
TEST(Fk, NormalizesQuaternionsInQ) {
  Zoo z;
  std::mt19937 rng(21);
  const VectorXd q = random_q(z.model, rng);
  VectorXd scaled = q;
  for (const Joint& j : z.model.joints) {
    if (j.type == JointType::kFree) scaled.segment<4>(j.qadr + 3) *= 3.0;
    if (j.type == JointType::kBall) scaled.segment<4>(j.qadr) *= 3.0;
  }
  const Frames a = forward_kinematics(z.model, q), b = forward_kinematics(z.model, scaled);
  for (int k = 0; k < z.model.nbody(); ++k) {
    EXPECT_LT((a.xpos[k] - b.xpos[k]).norm(), 1e-14) << "body " << k;
    EXPECT_LT(quat_diff(a.xquat[k], b.xquat[k]), 1e-14) << "body " << k;
  }
}

// 没有关节的刚体（hand）相对父刚体（forearm）的位姿不随 q 改变。
TEST(Fk, WeldedBodyKeepsItsOffset) {
  Zoo z;
  const int hand = z.body("hand"), forearm = z.body("forearm");
  const Body& B = z.model.bodies[hand];
  std::mt19937 rng(22);
  for (int i = 0; i < kPoses; ++i) {
    const Frames F = forward_kinematics(z.model, random_q(z.model, rng));
    const Vector3d local_pos = F.xquat[forearm].conjugate() * (F.xpos[hand] - F.xpos[forearm]);
    const Quaterniond local_quat = F.xquat[forearm].conjugate() * F.xquat[hand];
    EXPECT_LT((local_pos - B.pos).norm(), 1e-14) << "pose " << i;
    EXPECT_LT(quat_diff(local_quat, B.quat), 1e-14) << "pose " << i;
  }
}

// 铰链和滑动的轴、锚点和 MuJoCo 的 xaxis、xanchor 一致。
TEST(Fk, HingeAndSlideFramesMatchMujoco) {
  Zoo z;
  std::mt19937 rng(23);
  for (int i = 0; i < kPoses; ++i) {
    const VectorXd q = random_q(z.model, rng);
    z.mj_fk(q);
    const Frames F = forward_kinematics(z.model, q);
    for (int j = 0; j < z.m->njnt; ++j) {
      const Joint& J = z.model.joints[j];
      if (J.type != JointType::kHinge && J.type != JointType::kSlide) continue;
      EXPECT_LT((F.dof_axis[J.dadr] - Vector3d(z.d->xaxis + 3 * j)).norm(), 1e-14) << "joint " << j;
      if (J.type == JointType::kHinge) {
        EXPECT_LT((F.dof_anchor[J.dadr] - Vector3d(z.d->xanchor + 3 * j)).norm(), 1e-14) << "joint " << j;
      }
    }
  }
}

// ---------- 第 4 节：点的 Jacobian ----------

// 每个刚体上取一个随机点，和 mj_jac 的平移部分比。
TEST(Jacobian, MatchesMujoco) {
  Zoo z;
  std::mt19937 rng(30);
  std::uniform_real_distribution<double> u(-0.2, 0.2);
  Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> expect(3, z.model.nv);  // mj_jac 按行存
  for (int i = 0; i < kPoses; ++i) {
    const VectorXd q = random_q(z.model, rng);
    z.mj_fk(q);
    const Frames F = forward_kinematics(z.model, q);
    for (int b = 1; b < z.model.nbody(); ++b) {
      const Vector3d x = world_point(F, b, Vector3d(u(rng), u(rng), u(rng)));
      mj_jac(z.m.get(), z.d, expect.data(), nullptr, x.data(), b);
      EXPECT_LT((point_jacobian(z.model, F, b, x) - expect).norm(), 1e-13) << "pose " << i << ", body " << b;
    }
  }
}

// 有限差分：J 的第 d 列 ≈ [X(q ⊕ eps e_d) - X(q ⊕ -eps e_d)] / (2 eps)，X 是刚体上固定的一点。
// 这个检查不依赖 MuJoCo：它说明 J 确实是“沿 q ⊕ s delta 运动时点的速度”，和 integrate 用的是同一个切空间。
TEST(Jacobian, MatchesFiniteDifference) {
  Zoo z;
  const Model& M = z.model;
  std::mt19937 rng(31);
  const double eps = 1e-6;
  for (int i = 0; i < 5; ++i) {
    const VectorXd q = random_q(M, rng);
    const Frames F = forward_kinematics(M, q);
    for (int b = 1; b < M.nbody(); ++b) {
      const Vector3d local(0.1, -0.05, 0.07);
      const Eigen::Matrix3Xd J = point_jacobian(M, F, b, world_point(F, b, local));
      for (int d = 0; d < M.nv; ++d) {
        const VectorXd e = VectorXd::Unit(M.nv, d);
        const Vector3d xp = world_point(forward_kinematics(M, integrate(M, q, e, eps)), b, local);
        const Vector3d xm = world_point(forward_kinematics(M, integrate(M, q, e, -eps)), b, local);
        const Vector3d fd = (xp - xm) / (2 * eps);
        EXPECT_LT((J.col(d) - fd).norm(), 1e-8) << "pose " << i << ", body " << b << ", dof " << d;
      }
    }
  }
}

// 不在链上的自由度，Jacobian 那一列是 0：手指的运动和头、摆都无关。
TEST(Jacobian, ColumnsOffTheChainAreZero) {
  Zoo z;
  std::mt19937 rng(32);
  const VectorXd q = random_q(z.model, rng);
  const Frames F = forward_kinematics(z.model, q);
  const int finger = z.body("finger");
  const Eigen::Matrix3Xd J = point_jacobian(z.model, F, finger, F.xpos[finger] + Vector3d(0.03, 0, 0));
  for (const char* off : {"neck", "swing", "twist"}) {
    const int d = z.dof(off);
    const int n = z.model.joints[z.model.dof_joint[d]].nv;
    EXPECT_EQ(J.middleCols(d, n).norm(), 0.0) << off;
  }
  for (const char* on : {"root", "shoulder", "elbow", "knuckle"}) EXPECT_GT(J.col(z.dof(on)).norm(), 0.0) << on;
}
