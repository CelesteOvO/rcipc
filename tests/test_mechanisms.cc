// 机构和机器人场景（阶段 12 之后，笔记 notes/13_mechanisms.md）。
//
//   Cylinder.Geometry           圆柱做成 12 边形的棱柱：顶点、边、三角形的个数，三角形都朝外
//   Cylinder.RestsOnGroundAndBox 立着的、横着的、落到盒子上的圆柱：不穿模，停在该停的高度
//   Gripper.LiftsCube           夹爪夹起方块提 15 cm：不穿模，方块跟着走（几乎不滑）
// 齿轮组在 CPU 上一步约 0.25 s（网格大、检测是暴力的），它的测试在 test_gpu.cc（Mechanism.GearsTransmit）。
#include <cmath>

#include <gtest/gtest.h>

#include "rcipc/drive.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

}  // namespace

TEST(Cylinder, Geometry) {
  const Model M = load("cylinders.xml");
  // 三个圆柱，各 2 x 12 + 2 个顶点、6 x 12 条边、4 x 12 个三角形；世界上的盒子 8、12、12。
  EXPECT_EQ(M.nvertex(), 3 * 26 + 8);
  EXPECT_EQ(M.nedge(), 3 * 72 + 12);
  EXPECT_EQ(M.nface(), 3 * 48 + 12);
  // 每个三角形的法向都背离它所在圆柱的中心（刚体原点就是圆柱的中心）。
  for (const Face& f : M.faces) {
    if (f.body == 0) continue;
    const Vector3d a = M.vertices[f.v0].local, b = M.vertices[f.v1].local, c = M.vertices[f.v2].local;
    EXPECT_GT((b - a).cross(c - a).dot((a + b + c) / 3), 0.0);
  }
}

TEST(Cylinder, RestsOnGroundAndBox) {
  const Model M = load("cylinders.xml");
  StepParams P;
  P.h = M.timestep;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  double dmin = 1e300;
  for (int i = 0; i < 150; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, r.min_distance);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_NEAR(q[2], 0.05, 1e-3);                                  // 立着：中心在半长的高度
  EXPECT_NEAR(q[7 + 2], 0.04 * std::cos(3.14159265358979323846 / 12), 1e-3);   // 横着：一个侧面着地
  EXPECT_GT(q[14 + 2], 0.1 + 0.03 * 0.9);                         // 停在盒子上（盒子顶面 0.1）
  EXPECT_LT(v.norm(), 0.05);
}

TEST(Gripper, LiftsCube) {
  const Model M = load("gripper.xml");
  ASSERT_EQ(M.nu(), 3);
  StepParams P;
  P.h = M.timestep;
  P.mu = 0.5;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  double dmin = 1e300, cube_low = 0, lift_low = 0;
  for (int i = 0; i < 250; ++i) {
    P.ctrl = ctrl_at(M, (i + 1) * P.h);
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, r.min_distance);
    q = r.q;
    v = r.v;
    if (i + 1 == 110) {  // 1.1 s：夹紧了、还没开始提
      cube_low = q[2];
      lift_low = q[7];
    }
  }
  EXPECT_GT(dmin, 0.0);
  // 1.1 s 以后提了约 15.5 cm；方块跟着走，两者的差就是夹在手指里滑掉的距离。
  const double lifted = q[7] - lift_low, carried = q[2] - cube_low;
  EXPECT_GT(lifted, 0.15);
  EXPECT_NEAR(carried, lifted, 2e-3);
  EXPECT_NEAR(q[0], 0.0, 1e-3);  // 没有往一边歪
}
