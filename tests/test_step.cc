// 阶段 3：关节坐标里的时间步（笔记 notes/03_inertia.md 第 5、6 节）。
//
//   JointStep.* 单个时间步：自由落体、静止、每次迭代能量下降
//   Spin.*      没有重力的旋转盒子：动能不能增加（书 §7.5 图 35）
//   Pendulum.*  双摆：能量不增加、和 MuJoCo 的 RK4 比较、一阶收敛
#include "rcipc/step.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"

using namespace rcipc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

// 一段仿真的统计量。
struct SimRun {
  VectorXd q, v;
  double e_start = 0, e_max = 0, e_end = 0;  // 总能量（动能 + 重力势能）
  int nmax = 0;
};

SimRun simulate(const Model& model, const StepParams& P, VectorXd q, VectorXd v, double T) {
  SimRun r;
  r.e_start = r.e_max = kinetic_energy(model, q, v) + potential_energy(model, q);
  for (double t = 0; t < T - 1e-12; t += P.h) {
    const StepResult s = step(model, P, q, v);
    q = s.q;
    v = s.v;
    r.e_max = std::max(r.e_max, kinetic_energy(model, q, v) + potential_energy(model, q));
    r.nmax = std::max(r.nmax, s.newton);
  }
  r.q = q;
  r.v = v;
  r.e_end = kinetic_energy(model, q, v) + potential_energy(model, q);
  return r;
}

// MuJoCo 的 RK4，步长 1e-5，当作精确解。返回 T 秒时的 qpos。
VectorXd mujoco_rk4(mjModel* m, const VectorXd& q0, double T) {
  mjData* d = mj_makeData(m);
  for (int i = 0; i < m->nq; ++i) d->qpos[i] = q0[i];
  m->opt.integrator = mjINT_RK4;
  m->opt.timestep = 1e-5;
  while (d->time < T - 1e-12) mj_step(m, d);
  VectorXd q = Eigen::Map<VectorXd>(d->qpos, m->nq);
  mj_deleteData(d);
  return q;
}

}  // namespace

// ---------- 单个时间步 ----------

// 自由落体：从静止开始一步，隐式 Euler 的答案是 x = x0 + g h^2，v = g h，姿态不变。
// 平移部分的能量是二次函数，Newton 一步就到：迭代次数是 2（求解一次、确认一次），和阶段 1 一样。
TEST(JointStep, FreeFallIsOneNewtonStep) {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/box.xml");
  const Model model = from_mujoco(m.get());
  StepParams P;
  const VectorXd q0 = model.qpos0, v0 = VectorXd::Zero(6);
  const StepResult r = step(model, P, q0, v0);
  EXPECT_EQ(r.newton, 2);
  EXPECT_LT((r.q.head<3>() - q0.head<3>() - model.gravity * P.h * P.h).norm(), 1e-12);
  EXPECT_LT((r.q.tail<4>() - q0.tail<4>()).norm(), 1e-12);
  EXPECT_LT((r.v.head<3>() - model.gravity * P.h).norm(), 1e-9);
  EXPECT_LT(r.v.tail<3>().norm(), 1e-9);
}

// 没有重力、静止：什么都不动，第一次迭代就收敛。
TEST(JointStep, RestStaysAtRest) {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/double_pendulum.xml");
  Model model = from_mujoco(m.get());
  model.gravity.setZero();
  const VectorXd q0 = Eigen::Vector2d(0.3, -0.7);
  const StepResult r = step(model, StepParams{}, q0, VectorXd::Zero(2));
  EXPECT_EQ(r.newton, 1);
  EXPECT_EQ(r.q, q0);
  EXPECT_EQ(r.v, VectorXd::Zero(2));
}

// 每次迭代能量都不升（线搜索保证），并且最后收敛（max |p| / h < tol）。
TEST(JointStep, EnergyDecreasesInEveryIteration) {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/double_pendulum.xml");
  const Model model = from_mujoco(m.get());
  StepParams P;
  P.h = 0.01;
  std::vector<NewtonIterate> trace;
  const StepResult r = step(model, P, Eigen::Vector2d(0.5, 1.0), Eigen::Vector2d(-3.0, 4.0), &trace);
  ASSERT_GE(trace.size(), 2u);
  for (size_t i = 1; i < trace.size(); ++i) EXPECT_LE(trace[i].energy, trace[i - 1].energy) << "iteration " << i;
  EXPECT_LT(r.newton, P.max_newton);
}

// ---------- 旋转的盒子 ----------

// 没有重力和接触，动能应当守恒。隐式 Euler 只会让它减少（数值耗散），body 形式保证它不会增加。
// 书 §7.5 图 35：关节空间的写法在这里能把动能放大上千倍。
TEST(Spin, KineticEnergyNeverGrows) {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/box.xml");
  Model model = from_mujoco(m.get());
  model.gravity.setZero();
  StepParams P;
  P.h = 0.01;
  VectorXd q = model.qpos0, v = VectorXd::Zero(6);
  v[3] = 0.05;  // 绕中间那根主轴（y）转 5 rad/s，加一点绕 x 的扰动
  v[4] = 5.0;
  const double k0 = kinetic_energy(model, q, v);
  double kmax = k0;
  for (int i = 0; i < 300; ++i) {
    const StepResult r = step(model, P, q, v);
    q = r.q;
    v = r.v;
    kmax = std::max(kmax, kinetic_energy(model, q, v));
  }
  EXPECT_LE(kmax, k0 * (1 + 1e-9));
  EXPECT_GT(kinetic_energy(model, q, v), 0.3 * k0);  // 3 s 后还剩一半左右（耗散，见思考题）
}

// ---------- 双摆 ----------

// 从两根杆都水平静止释放，1 秒内总能量从不高于初始值。
TEST(Pendulum, EnergyNeverGrows) {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/double_pendulum.xml");
  const Model model = from_mujoco(m.get());
  const SimRun r = simulate(model, StepParams{}, VectorXd::Zero(2), VectorXd::Zero(2), 1.0);
  EXPECT_LE(r.e_max, r.e_start + 1e-9);
  EXPECT_LT(r.e_end, r.e_start);
  EXPECT_LE(r.nmax, 5);
}

// 和 MuJoCo 的 RK4（步长 1e-5，当作精确解）比较 0.5 秒时的摆角。
// 隐式 Euler 是一阶方法：步长减半，误差也大约减半。
TEST(Pendulum, ConvergesAtFirstOrder) {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/double_pendulum.xml");
  const Model model = from_mujoco(m.get());
  const double T = 0.5;
  const VectorXd exact = mujoco_rk4(m.get(), VectorXd::Zero(2), T);
  double err[2];
  for (int i = 0; i < 2; ++i) {
    StepParams P;
    P.h = i == 0 ? 0.002 : 0.001;
    err[i] = (simulate(model, P, VectorXd::Zero(2), VectorXd::Zero(2), T).q - exact).norm();
  }
  EXPECT_LT(err[0], 0.05);
  EXPECT_GT(err[0] / err[1], 1.7);
  EXPECT_LT(err[0] / err[1], 2.3);
}
