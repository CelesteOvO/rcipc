// 阶段 4、5：CCD（笔记 notes/04_ground.md 第 5 节，notes/05_curved_ccd.md）。
//
//   ChordCcd.*      阶段 4：直弦 CCD 的规则，和阶段 1 的一维版本一致
//   Flail.Chord*    阶段 4 的反例：高速转动的杆，两端都在地面上方，弧线却穿过了地面
//   Acceleration.*  阶段 5：每个刚体的加速度上界，和真实轨迹的二阶导数比
//   Sagitta.*       阶段 5：弧线偏离弦不超过 A ds^2 / 8
//   CurvedCcd.*     阶段 5：弦加矢高抓住了反例；纯平移时和直弦一样；整段仿真不穿透
//   Acceleration.Parts*、Acceleration.Tight*  紧界的两部分 |x''(s)| <= a0 + T s，逐层的中间量
//   CurvedCcd.KeepsTheMinimumSeparation*、Step.Inside*、Scene.WithoutABarrier*  最小分离（笔记 22 第 5 节）
//   CurvedCcd.Sliding*、Accd.*  ACCD 的分离平面界（笔记 22 第 3 节）
#include "rcipc/ccd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rcipc/contact.h"
#include "rcipc/distance.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/step.h"

using namespace rcipc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

constexpr double kPi = 3.14159265358979323846;

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

// 甩杆的几何（flail.xml）：杆端的球心绕铰链转，半径 R = 0.5，球的半径 0.01；铰链离地 H = 0.5075。
// 球面到地面的距离是 (H - 0.01) - R cos(角)（角从正下方量起），最低点 H - 0.01 - R = -2.5 mm。
struct FlailSetup {
  Model M = load("flail.xml");
  double H = 0.5075, R = 0.5, rad = 0.01;
  double phi = std::acos((H - rad) / R);  // 在地面以下的那段弧的半角：acos(0.995) ≈ 0.1 rad
  StepParams P;
  VectorXd q = VectorXd::Zero(1), v = VectorXd::Zero(1);
  FlailSetup() {
    P.h = 0.01;
    q[0] = kPi / 2 - phi - 0.05;  // 起点：杆在“指向正下方”之前 phi + 0.05 rad 处，杆端离地约 3 mm
    v[0] = 40.0;
  }
};

// zoo.xml 的随机姿态（同 test_contact.cc）。
VectorXd random_q(const Model& model, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(-1, 1);
  std::normal_distribution<double> n;
  VectorXd q = model.qpos0;
  for (int attempt = 0; attempt < 100; ++attempt) {  // 相交的姿态重新抽（阶段 7 起刚体之间也算）
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

VectorXd random_p(int nv, std::mt19937& rng, double scale) {
  std::uniform_real_distribution<double> u(-scale, scale);
  VectorXd p(nv);
  for (int i = 0; i < nv; ++i) p[i] = u(rng);
  return p;
}

}  // namespace

// ---------- 阶段 4：直弦 ----------

// 盒子竖直平移：远离地面不限制；靠近时 alpha = c * d / (d - d_end)，和阶段 1 的 ccd_max_step 一样。
TEST(ChordCcd, MatchesTheOneDimensionalRule) {
  const Model M = load("drop.xml");
  const VectorXd q = M.qpos0;  // 下面 4 个角离地 0.5
  VectorXd up = VectorXd::Zero(6), down = VectorXd::Zero(6);
  up[2] = 0.3;
  down[2] = -1.0;  // 整步会穿过地面
  const Frames F = forward_kinematics(M, q);
  EXPECT_EQ(chord_ccd_ground(M, F, forward_kinematics(M, integrate(M, q, up)), 0.9), 1.0);
  EXPECT_NEAR(chord_ccd_ground(M, F, forward_kinematics(M, integrate(M, q, down)), 0.9), 0.9 * 0.5 / 1.0, 1e-12);
  // 整步闭合不到 c 倍时不限制：往下 0.3，闭合 60% < 90%。
  down[2] = -0.3;
  EXPECT_EQ(chord_ccd_ground(M, F, forward_kinematics(M, integrate(M, q, down)), 0.9), 1.0);
  // 不是从 1 开始截：往下 0.49（闭合 98%），alpha = 0.9 * 0.5 / 0.49。
  down[2] = -0.49;
  EXPECT_NEAR(chord_ccd_ground(M, F, forward_kinematics(M, integrate(M, q, down)), 0.9), 0.9 * 0.5 / 0.49, 1e-12);
}

// 走到 CCD 允许的步长处，最近的顶点离地正好是 (1 - c) 倍的起始距离。
TEST(ChordCcd, StopsAtOneMinusCOfTheGap) {
  const Model M = load("drop.xml");
  const VectorXd q = M.qpos0;
  VectorXd p = VectorXd::Zero(6);
  p[2] = -2.0;
  p[3] = 0.3;  // 加一点转动，4 个角不再同高
  const Frames F0 = forward_kinematics(M, q);
  const double alpha = chord_ccd_ground(M, F0, forward_kinematics(M, integrate(M, q, p)), 0.9);
  ASSERT_LT(alpha, 1.0);
  // 平移是线性的，转动是弧线：弧线上的距离和弦上的距离略有不同，所以这里只要求数量级一致，
  // 并且真实轨迹上没有穿透（转得慢，矢高很小）。
  const double d_end = min_distance(M, forward_kinematics(M, integrate(M, q, p, alpha)));
  EXPECT_NEAR(d_end, 0.1 * 0.5, 0.02);
  EXPECT_GT(trajectory_min_distance(M, q, p, alpha, 64), 0.0);
}

// 阶段 4 的反例（书 §9.1 的甩杆，改成地面）：杆端以 40 rad/s 从地面上方 3 mm 的位置一步转到另一侧 7 mm 的位置，
// 两端都在地面以上、都不在 dhat 之内，直弦 CCD 和线搜索都放行；真实的圆弧在中间低到地面以下 2.5 mm。
// 这个测试记录的是直弦 CCD 的**失败**（P.ccd = kChord）；下面 CurvedCcd.CatchesTheArc 是阶段 5 的修正。
TEST(Flail, ChordCcdMissesTheArcThroughTheFloor) {
  FlailSetup S;
  S.P.ccd = CcdMode::kChord;
  const double d_start = min_distance(S.M, forward_kinematics(S.M, S.q));
  EXPECT_NEAR(d_start, (S.H - S.rad) - S.R * std::cos(S.phi + 0.05), 1e-9);
  EXPECT_GT(d_start, S.P.dhat);

  std::vector<NewtonIterate> trace;
  const StepResult r = step(S.M, S.P, S.q, S.v, nullptr, &trace);
  // 一步转过的角度（约 0.35 rad：隐式 Euler 的耗散让它比 h w = 0.4 小）跨过了整段 2 phi ≈ 0.2 rad 的弧。
  const double turned = r.q[0] - S.q[0];
  EXPECT_GT(turned, 2 * S.phi + 0.05);
  // 终点在地面之上、dhat 之外。Newton 要迭代十来次（转 0.35 rad 时 FK 的非线性很强，Gauss-Newton 是线性收敛），
  // 但每一次 CCD 都没有截断、线搜索都走了满步：弦从来没有靠近地面。
  EXPECT_GT(r.min_distance, S.P.dhat);
  EXPECT_LT(r.newton, S.P.max_newton);
  ASSERT_FALSE(trace.empty());
  for (const NewtonIterate& it : trace) {
    EXPECT_EQ(it.alpha_max, 1.0);
    EXPECT_EQ(it.alpha, 1.0);
  }
  // 但真实轨迹的最低点在地面以下：直弦 CCD 漏掉了这次穿透。
  const double dmin = trajectory_min_distance(S.M, S.q, differentiate(S.M, S.q, r.q), 1.0, 64);
  EXPECT_LT(dmin, 0.0);
  EXPECT_NEAR(dmin, S.H - S.rad - S.R, 1e-4);
}

// ---------- 阶段 5：加速度界 ----------

// 单个铰链：A = w^2 R，R 是锚点到顶点的静态力臂（0.5 + 0.01）。
TEST(Acceleration, SingleHingeIsCentripetal) {
  const Model M = load("flail.xml");
  VectorXd p(1);
  p[0] = 0.4;
  const VectorXd A = acceleration_bounds(M, M.qpos0, p);
  EXPECT_NEAR(A[1], 0.4 * 0.4 * 0.51, 1e-12);
  EXPECT_EQ(A[0], 0.0);
  // 自由关节只平移：没有曲率。
  const Model D = load("drop.xml");
  VectorXd pd = VectorXd::Zero(6);
  pd[2] = -3.0;
  EXPECT_EQ(acceleration_bounds(D, D.qpos0, pd)[1], 0.0);
  // 自由关节转动：A = w^2 * 刚体半径。
  pd.setZero();
  pd[4] = 0.5;
  EXPECT_NEAR(acceleration_bounds(D, D.qpos0, pd)[1], 0.25 * D.bodies[1].radius, 1e-12);
}

// zoo.xml（自由、球、铰链、滑动都有，链深 4）：沿 q ⊕ s p 的真实二阶导数（中心差分）在 s = 0, 0.25, ..., 1 处
// 都不超过 A_b，对每个顶点。
TEST(Acceleration, BoundsTheTrueSecondDerivative) {
  const Model M = load("zoo.xml");
  std::mt19937 rng(60);
  const double eps = 1e-4;
  for (int i = 0; i < 5; ++i) {
    const VectorXd q = random_q(M, rng);
    const VectorXd p = random_p(M.nv, rng, 0.5);
    const VectorXd A = acceleration_bounds(M, q, p);
    for (int v = 0; v < M.nvertex(); ++v) {
      const Vertex& V = M.vertices[v];
      for (int k = 0; k <= 4; ++k) {
        const double s = 0.25 * k;
        const auto x = [&](double ss) {
          return world_point(forward_kinematics(M, integrate(M, q, p, ss)), V.body, V.local);
        };
        const Vector3d xpp = (x(s + eps) - 2 * x(s) + x(s - eps)) / (eps * eps);
        EXPECT_LE(xpp.norm(), A[V.body] * (1 + 1e-6) + 1e-6) << "pose " << i << ", vertex " << v << ", s " << s;
      }
    }
  }
}

// 紧界（笔记第 6 节）也要对整段成立：四个模型、随机姿态和方向，真实的 |x''(s)| 在 s = 0, 0.25, ..., 1 处
// 都不超过它（zoo.xml 里链上有滑动关节的刚体返回 +无穷，不比）。
TEST(Acceleration, TightBoundsTheTrueSecondDerivative) {
  std::mt19937 rng(61);
  const double eps = 1e-4;
  int checked = 0;
  for (const char* file : {"zoo.xml", "ragdoll.xml", "chain.xml", "flail.xml"}) {
    const Model M = load(file);
    for (int i = 0; i < 4; ++i) {
      const VectorXd q = random_q(M, rng);
      const VectorXd p = random_p(M.nv, rng, 0.8);
      const VectorXd A = tight_acceleration_bounds(M, forward_kinematics(M, q), p);
      for (int v = 0; v < M.nvertex(); ++v) {
        const Vertex& V = M.vertices[v];
        if (V.body == 0 || !std::isfinite(A[V.body])) continue;
        for (int k = 0; k <= 4; ++k) {
          const double s = 0.25 * k;
          const auto x = [&](double ss) {
            return world_point(forward_kinematics(M, integrate(M, q, p, ss)), V.body, V.local);
          };
          const Vector3d xpp = (x(s + eps) - 2 * x(s) + x(s - eps)) / (eps * eps);
          EXPECT_LE(xpp.norm(), A[V.body] * (1 + 1e-6) + 1e-6) << file << " pose " << i << ", vertex " << v << ", s " << s;
          ++checked;
        }
      }
    }
  }
  EXPECT_GT(checked, 500);
}

// 紧界松紧的两端：单个铰链时静态界就是精确的向心加速度，紧界还多一个三阶项，所以 CCD 取两者的较小值；
// 12 节链条的末端，每个自由度转 0.01 rad 左右（布娃娃堆里一个 Newton 迭代的常见量级）时，静态界的力臂是沿整条链的
// 路径长度，紧界用现在姿态下的真实距离，紧好几倍。转得多时反过来：紧界的余项是转速的三次方，静态界只是二次方，
// 每个自由度转 0.2 rad 时两者取小总是静态界（笔记第 6 节的表）。
TEST(Acceleration, TightIsTighterOnDeepChains) {
  const Model H = load("flail.xml");
  VectorXd ph(1);
  ph[0] = 0.4;
  const VectorXd Fs = acceleration_bounds(H, H.qpos0, ph);
  const VectorXd Ft = tight_acceleration_bounds(H, forward_kinematics(H, H.qpos0), ph);
  EXPECT_GT(Ft[1], Fs[1]);
  EXPECT_EQ(ccd_acceleration_bounds(H, H.qpos0, forward_kinematics(H, H.qpos0), ph)[1], Fs[1]);

  const Model C = load("chain.xml");
  std::mt19937 rng(62);
  double ratio = 0;
  const int tries = 8;
  for (int i = 0; i < tries; ++i) {
    const VectorXd q = random_q(C, rng);
    const VectorXd p = random_p(C.nv, rng, 0.01);
    const int tip = C.nbody() - 1;
    ratio += tight_acceleration_bounds(C, forward_kinematics(C, q), p)[tip] / acceleration_bounds(C, q, p)[tip];
  }
  EXPECT_LT(ratio / tries, 0.5) << "mean tight / static at the chain's tip";
}

// 紧界的两部分用的是比上面更强的一句话：|x''(s)| <= a0 + T s 对 [0, 1] 上的每个 s 都成立，
// 不只是 <= a0 + T。把锚点挪到 s0（在那里的姿态下重算两部分）以后，同一句话对 s0 之后的轨迹也要成立；
// 这一条同时检查了球关节和自由关节沿同一个 p 走时，中途的 FK 给出的轴和真实轨迹的速度是一致的。
TEST(Acceleration, PartsBoundTheSecondDerivativeFromAnyAnchor) {
  std::mt19937 rng(63);
  const double eps = 1e-4;
  int checked = 0;
  for (const char* file : {"zoo.xml", "ragdoll.xml", "chain.xml", "flail.xml"}) {
    const Model M = load(file);
    for (int i = 0; i < 4; ++i) {
      const VectorXd q = random_q(M, rng);
      const VectorXd p = random_p(M.nv, rng, 0.8);
      const auto frames = [&](double s) { return forward_kinematics(M, integrate(M, q, p, s)); };
      for (double s0 : {0.0, 0.3, 0.6}) {
        VectorXd a0, T;
        tight_acceleration_parts(M, frames(s0), p, &a0, &T);
        VectorXd at_anchor = VectorXd::Zero(M.nbody());  // 锚点处（u = 0）每个刚体顶点上 |x''| 的最大值，差分出来的
        for (double u : {0.0, 0.05, 0.1, 0.2, 0.4}) {
          const Frames Fm = frames(s0 + u - eps), Fc = frames(s0 + u), Fp = frames(s0 + u + eps);
          for (int v = 0; v < M.nvertex(); ++v) {
            const Vertex& V = M.vertices[v];
            if (V.body == 0 || !std::isfinite(T[V.body])) continue;
            const Vector3d xpp =
                (world_point(Fp, V.body, V.local) - 2 * world_point(Fc, V.body, V.local) + world_point(Fm, V.body, V.local)) /
                (eps * eps);
            EXPECT_LE(xpp.norm(), (a0[V.body] + T[V.body] * u) * (1 + 1e-6) + 1e-6)
                << file << " pose " << i << ", vertex " << v << ", anchor " << s0 << ", u " << u;
            if (u == 0.0) at_anchor[V.body] = std::max(at_anchor[V.body], xpp.norm());
            ++checked;
          }
        }
        // a0 不只是上界：它就是锚点处的最大值（所以上面的比较在 u = 0 处没有余量，轴或锚点算错了会被抓住）。
        for (int b = 1; b < M.nbody(); ++b) {
          if (std::isfinite(T[b])) EXPECT_NEAR(at_anchor[b], a0[b], 1e-5 * (1 + a0[b])) << file << " pose " << i << ", body " << b;
        }
      }
    }
  }
  EXPECT_GT(checked, 2000);
}

// T_b 本身（笔记第 6.5 节的定理）：真实的 |x'''(s)| 在整段上不超过 T_b。上面两个测试只通过 |x''| <= a0 + T s 间接用到 T，
// 这里直接比三阶导数（四个点的中心差分，误差 O(eps^2)）。
// 甩杆是单个铰链，真实的 |x'''| = th^3 * 0.5，T = th^3 * 0.51（多出来的是球的半径），用它检查这个比较不是空的。
TEST(Acceleration, ThirdDerivativeStaysBelowT) {
  std::mt19937 rng(64);
  const double eps = 1e-3;
  int checked = 0;
  for (const char* file : {"zoo.xml", "ragdoll.xml", "chain.xml", "flail.xml"}) {
    const Model M = load(file);
    double tightest = 0;  // |x'''| / T 的最大值
    for (int i = 0; i < 4; ++i) {
      const VectorXd q = random_q(M, rng);
      const VectorXd p = random_p(M.nv, rng, 0.8);
      VectorXd a0, T;
      tight_acceleration_parts(M, forward_kinematics(M, q), p, &a0, &T);
      for (int k = 0; k <= 4; ++k) {
        const double s = 0.25 * k;
        const double at[4] = {s - 2 * eps, s - eps, s + eps, s + 2 * eps};
        Frames F[4];
        for (int m = 0; m < 4; ++m) F[m] = forward_kinematics(M, integrate(M, q, p, at[m]));
        for (int v = 0; v < M.nvertex(); ++v) {
          const Vertex& V = M.vertices[v];
          if (V.body == 0 || !std::isfinite(T[V.body])) continue;
          const auto x = [&](int m) { return world_point(F[m], V.body, V.local); };
          const Vector3d xppp = (x(3) - 2 * x(2) + 2 * x(1) - x(0)) / (2 * eps * eps * eps);
          EXPECT_LE(xppp.norm(), T[V.body] * (1 + 1e-3) + 1e-5) << file << " pose " << i << ", vertex " << v << ", s " << s;
          if (T[V.body] > 1e-3) tightest = std::max(tightest, xppp.norm() / T[V.body]);
          ++checked;
        }
      }
    }
    if (std::string(file) == "flail.xml") EXPECT_GT(tightest, 0.95) << "the bound is nearly exact for one hinge";
  }
  EXPECT_GT(checked, 500);
}

// T_b 的证明（笔记第 6.2、6.3 节）逐层说的话，各拿真实轨迹比一遍。tight_terms 给出每个关节 j 的 c_j、v_j、a_j、t_j：
//   引理（距离）：整段上 |w_j x (x - o_j)| <= c_j，也就是点到 j 的轴的距离不超过 d_j + lever + v_{j+1}；
//   递推（冻住上游）：把 j 之前的关节冻住（p 里只留 j 和它下游的转动），点的速度 <= v_j，加速度 <= a_j，
//   三阶导数 <= t_j。
// 递推里每一层的三个量是不是真的盖住了它该盖住的东西，在这里分开检查；最后一层（t_1 = T_b）由上一个测试检查。
TEST(Acceleration, TightTermsBoundTheTrajectory) {
  std::mt19937 rng(65);
  const double eps = 1e-3;
  int checked = 0;
  for (const char* file : {"zoo.xml", "ragdoll.xml", "chain.xml", "flail.xml"}) {
    const Model M = load(file);
    for (int i = 0; i < 3; ++i) {
      const VectorXd q = random_q(M, rng);
      const VectorXd p = random_p(M.nv, rng, 0.8);
      const Frames F0 = forward_kinematics(M, q);
      VectorXd a0, T;
      tight_acceleration_parts(M, F0, p, &a0, &T);
      for (int b = 1; b < M.nbody(); ++b) {
        const std::vector<TightTerm> terms = tight_terms(M, F0, p, b);
        // 最靠近根的那一层的 t 就是 T_b（tight_terms 和 CCD 用的是同一个循环）。
        if (!terms.empty()) EXPECT_EQ(terms.front().t, T[b]) << file << " pose " << i << ", body " << b;
        std::vector<int> verts;
        for (int v = 0; v < M.nvertex(); ++v) {
          if (M.vertices[v].body == b) verts.push_back(v);
        }
        for (size_t j = 0; j < terms.size(); ++j) {
          const TightTerm& t = terms[j];
          VectorXd pm = VectorXd::Zero(M.nv);  // 只留 j 和它下游的转动：j 之前的关节不动
          for (size_t k = j; k < terms.size(); ++k) {
            pm.segment(terms[k].dadr, terms[k].nv) = p.segment(terms[k].dadr, terms[k].nv);
          }
          for (int k = 0; k <= 4; ++k) {
            const double s = 0.25 * k;
            const Frames Fs = forward_kinematics(M, integrate(M, q, p, s));  // 真实轨迹
            Frames Fz[5];  // 冻住上游的轨迹，在 s - 2 eps, ..., s + 2 eps 处
            for (int m = 0; m < 5; ++m) Fz[m] = forward_kinematics(M, integrate(M, q, pm, s + (m - 2) * eps));
            Vector3d w = Vector3d::Zero();  // 关节 j 的转动向量（世界系）
            for (int d = t.dadr; d < t.dadr + t.nv; ++d) w += p[d] * Fs.dof_axis[d];
            for (int v : verts) {
              const Vector3d local = M.vertices[v].local;
              const Vector3d r = world_point(Fs, b, local) - Fs.dof_anchor[t.dadr];
              Vector3d x[5];
              for (int m = 0; m < 5; ++m) x[m] = world_point(Fz[m], b, local);
              const Vector3d d1 = (x[3] - x[1]) / (2 * eps);
              const Vector3d d2 = (x[3] - 2 * x[2] + x[1]) / (eps * eps);
              const Vector3d d3 = (x[4] - 2 * x[3] + 2 * x[1] - x[0]) / (2 * eps * eps * eps);
              const std::string where = std::string(file) + " pose " + std::to_string(i) + ", body " + std::to_string(b) +
                                        ", joint " + std::to_string(j) + ", s " + std::to_string(s);
              EXPECT_LE(w.cross(r).norm(), t.speed * (1 + 1e-9) + 1e-12) << where;
              EXPECT_LE(d1.norm(), t.v * (1 + 1e-3) + 1e-6) << where;
              EXPECT_LE(d2.norm(), t.a * (1 + 1e-3) + 1e-6) << where;
              EXPECT_LE(d3.norm(), t.t * (1 + 1e-3) + 1e-5) << where;
              ++checked;
            }
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 2000);
}

// ---------- 阶段 5：矢高 ----------

// 甩杆转 0.4 rad：弧线偏离弦最多 R (1 - cos(0.2)) ≈ R 0.4^2 / 8，不超过界 A / 8。
TEST(Sagitta, ArcStaysWithinTheBound) {
  const Model M = load("flail.xml");
  VectorXd p(1);
  p[0] = 0.4;
  const VectorXd q = M.qpos0;
  const double A = acceleration_bounds(M, q, p)[1];
  const Vertex& tip = M.vertices[0].local.norm() > M.vertices[1].local.norm() ? M.vertices[0] : M.vertices[1];
  const Vector3d xa = world_point(forward_kinematics(M, q), tip.body, tip.local);
  const Vector3d xb = world_point(forward_kinematics(M, integrate(M, q, p)), tip.body, tip.local);
  double dev_max = 0;
  for (int k = 0; k <= 64; ++k) {
    const double s = k / 64.0;
    const Vector3d x = world_point(forward_kinematics(M, integrate(M, q, p, s)), tip.body, tip.local);
    dev_max = std::max(dev_max, (x - (xa + s * (xb - xa))).norm());
  }
  EXPECT_NEAR(dev_max, 0.5 * (1 - std::cos(0.2)), 1e-6);  // 球心的半径是 0.5
  EXPECT_LE(dev_max, A / 8);
  EXPECT_GT(dev_max, 0.9 * A / 8);  // 单铰链时界几乎是紧的（差一个 0.5 对 0.51 的半径）
}

// ---------- 阶段 5：曲线 CCD ----------

// 纯平移：没有曲率，曲线 CCD 和直弦 CCD 给出同样的步长。
TEST(CurvedCcd, EqualsChordForPureTranslation) {
  const Model M = load("drop.xml");
  const VectorXd q = M.qpos0;
  VectorXd p = VectorXd::Zero(6);
  p[2] = -1.0;
  const double chord = chord_ccd_ground(M, forward_kinematics(M, q), forward_kinematics(M, integrate(M, q, p)), 0.9);
  EXPECT_NEAR(curved_ccd(M, q, p, 1e-3, 0.9, true, 0.0), chord, 1e-12);  // 直弦 CCD 没有最小分离
  p[2] = 0.3;
  EXPECT_EQ(curved_ccd(M, q, p, 1e-3, 0.9), 1.0);
}

// 阶段 4 的反例，换成曲线 CCD：第一次迭代就被截断，整条真实轨迹都在地面之上。
TEST(CurvedCcd, CatchesTheArcThroughTheFloor) {
  FlailSetup S;
  std::vector<NewtonIterate> trace;
  const StepResult r = step(S.M, S.P, S.q, S.v, nullptr, &trace);
  ASSERT_FALSE(trace.empty());
  EXPECT_LT(trace[0].alpha_max, 1.0);
  EXPECT_GT(r.min_distance, 0.0);
  EXPECT_LT(r.min_distance, S.P.dhat);  // 停在了障碍里
  EXPECT_GT(trajectory_min_distance(S.M, S.q, differentiate(S.M, S.q, r.q), 1.0, 256), 0.0);
  // 杆没有跨过去：终点还在正下方之前。
  EXPECT_LT(r.q[0], kPi / 2);
}

// 甩杆连续 100 步（40 rad/s 撞向地面，之后被弹回或停住）：每一步的真实轨迹都不穿透。
TEST(CurvedCcd, FlailNeverPenetratesOverManySteps) {
  FlailSetup S;
  S.P.adapt_kappa = false;  // 阶段 9 的 kappa 调整会把间隙压到 1e-2 dhat，这里保持阶段 5 的设置
  SolverState state;
  VectorXd q = S.q, v = S.v;
  double dmin = 1e9;
  int nmax = 0;
  for (int i = 0; i < 100; ++i) {
    const StepResult r = step(S.M, S.P, q, v, &state);
    dmin = std::min(dmin, trajectory_min_distance(S.M, q, differentiate(S.M, q, r.q), 1.0, 64));
    nmax = std::max(nmax, r.newton);
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(nmax, S.P.max_newton);
}

// 落地的盒子带着转动落下（4 个角不同高）：曲线 CCD 下每步都不穿透。
// （没有摩擦，它落地后还会慢慢滑和晃，所以不检查静止；摩擦是阶段 8。）
TEST(CurvedCcd, TumblingBoxNeverPenetrates) {
  const Model M = load("drop.xml");
  StepParams P;
  P.adapt_kappa = false;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(6);
  v[3] = 6.0;  // 绕 x 轴 6 rad/s
  v[4] = 2.0;
  double dmin = 1e9;
  for (int i = 0; i < 1000; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin = std::min(dmin, trajectory_min_distance(M, q, differentiate(M, q, r.q), 1.0, 32));
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin, 0.0);
  EXPECT_LT(v.norm(), 1.0);  // 落地时的速度是几 m/s，2 s 后只剩下慢慢的滑动
}

// ---------- 最小分离（笔记 22 第 5 节） ----------
//
// CCD 把 d - sep 当距离用，起点已经 <= sep 的对返回 0（卡住）。原来 d <= 0 时返回 1（放行）：
// 浮点把 1e-16 m 的间隙算成负数时，整步就穿过去了（ragdoll_pile.xml mu=0.35 的最后一环）。

// capsule_cradle.xml：自由胶囊落在两根平行胶囊的槽里，四对“端点–边”的间隙都是 gap。
static VectorXd cradle_at_gap(const Model& M, double gap) {
  VectorXd q = M.qpos0;
  q[2] = 0.2 + std::sqrt((0.06 + gap) * (0.06 + gap) - 0.04 * 0.04);
  return q;
}

TEST(CurvedCcd, KeepsTheMinimumSeparationBetweenBodies) {
  const Model M = load("capsule_cradle.xml");
  const double sep = 1e-12;
  VectorXd p = VectorXd::Zero(M.nv);
  p[2] = -1e-6;  // 往槽里压 1 μm（间隙减少约 0.75 μm）
  // 1. 间隙 1 nm：截断以后，真实轨迹（纯平移，弦就是轨迹）上的距离 > sep。
  {
    const VectorXd q = cradle_at_gap(M, 1e-9);
    const double a = curved_ccd(M, q, p, 1e-3, 0.9, true, sep);
    EXPECT_GT(a, 0.0);
    EXPECT_LT(a, 1.0);
    EXPECT_GT(trajectory_min_distance(M, q, p, a, 64), sep);
  }
  // 2. 间隙 5e-13，已经在 sep 以内：往里、往外都返回 0（卡住）。
  {
    const VectorXd q = cradle_at_gap(M, 5e-13);
    EXPECT_EQ(curved_ccd(M, q, p, 1e-3, 0.9, true, sep), 0.0);
    EXPECT_EQ(curved_ccd(M, q, -p, 1e-3, 0.9, true, sep), 0.0);
  }
  // 3. 浮点那一环：算出来的距离已经是负的（-1e-15）。原来返回 1，现在即使 sep = 0 也返回 0。
  {
    const VectorXd q = cradle_at_gap(M, -1e-15);
    ASSERT_LE(min_distance(M, forward_kinematics(M, q)), 0.0);
    EXPECT_EQ(curved_ccd(M, q, p, 1e-3, 0.9, true, 0.0), 0.0);
  }
}

TEST(CurvedCcd, KeepsTheMinimumSeparationFromTheGround) {
  const Model M = load("drop.xml");
  const double d0 = min_distance(M, forward_kinematics(M, M.qpos0));
  const auto at_height = [&](double gap) {  // 盒子平着往下挪，最低的面离地 gap
    VectorXd q = M.qpos0;
    q[2] -= d0 - gap;
    return q;
  };
  VectorXd p = VectorXd::Zero(6);
  p[2] = -1.0;
  // 离地 1 μm：截在 0.9 (1e-6 - sep)，没有转动所以矢高是 0。
  EXPECT_NEAR(curved_ccd(M, at_height(1e-6), p, 1e-3, 0.9), 0.9 * (1e-6 - 1e-12), 1e-15);
  // 离地 5e-13，在 sep 以内：卡住。
  EXPECT_EQ(curved_ccd(M, at_height(5e-13), p, 1e-3, 0.9), 0.0);
  // 已经在地面以下 1e-14：原来 sqrt(4 ya / A) 得 NaN，std::min 把它忽略，这个顶点不再受约束，整步放行。
  const VectorXd q = at_height(-1e-14);
  ASSERT_LT(min_distance(M, forward_kinematics(M, q)), 0.0);
  EXPECT_EQ(curved_ccd(M, q, p, 1e-3, 0.9, true, 0.0), 0.0);
}

// 线搜索的第二道网：带半径的图元一旦重叠（d < 0），能量是 +无穷。原来 b(d^2) 对 d 是偶函数，d = -1e-9 和 +1e-9 一样。
TEST(Contact, EnergyIsInfiniteOnceThePrimitivesOverlap) {
  const Model M = load("capsule_cradle.xml");
  const auto energy = [&](double gap) {
    const Frames F = forward_kinematics(M, cradle_at_gap(M, gap));
    return contact_energy(M, find_contacts(M, F, 1e-3), 1e8, 1e-3);
  };
  EXPECT_TRUE(std::isfinite(energy(1e-9)));
  EXPECT_GT(energy(1e-9), 0.0);
  EXPECT_EQ(energy(-1e-9), std::numeric_limits<double>::infinity());
}

// 从最小分离以内出发：每次 CCD 都返回 0，这一步原地不动（撞上迭代上限），不穿透。这就是“卡住”的样子。
// 胶囊沿槽的方向以 0.1 m/s 滑动（在实数里不改变间隙）也一样被挡住：卡住的对不看方向。
TEST(Step, InsideTheMinimumSeparationItStallsInsteadOfPenetrating) {
  const Model M = load("capsule_cradle.xml");
  StepParams P;
  P.max_newton = 5;
  const VectorXd q0 = cradle_at_gap(M, 5e-13);
  VectorXd v0 = VectorXd::Zero(M.nv);
  v0[0] = 0.1;
  const StepResult r = step(M, P, q0, v0);
  EXPECT_LT((r.q - q0).norm(), 1e-15);
  EXPECT_EQ(r.newton, P.max_newton);
  EXPECT_TRUE(r.capped);
  EXPECT_GT(r.min_distance, 0.0);
}

// 模型本身有洞时 CCD 也不放行。去掉胶囊端点的退化边（2026-11-12 之前的模型），平行的胶囊之间没有任何障碍力，
// 自由胶囊一直往槽里掉：原来 0.1 s 内间隙变成 0、穿过去；现在 CCD 把它挡在最小分离上，卡住。
TEST(Scene, WithoutABarrierTheCcdStillNeverLetsGo) {
  Model M = load("capsule_cradle.xml");
  M.edges.erase(std::remove_if(M.edges.begin(), M.edges.end(), [](const Edge& e) { return e.v0 == e.v1; }),
                M.edges.end());
  StepParams P;
  P.check_iterates = true;
  SolverState S;
  VectorXd q = M.qpos0, v = VectorXd::Zero(M.nv);
  double dmin_iter = 1e9;
  int capped = 0;
  for (int i = 0; i < 100; ++i) {
    const StepResult r = step(M, P, q, v, &S);
    dmin_iter = std::min(dmin_iter, r.iterate_min_distance);
    capped += r.capped;
    q = r.q;
    v = r.v;
  }
  EXPECT_GT(dmin_iter, 0.0);
  EXPECT_GT(capped, 0);  // 卡住的步原地不动，撞上迭代上限
  std::printf("without end-point edges: min distance along iterates %.3e m, %d capped steps\n", dmin_iter, capped);
}

// ---------- 分离平面的界（笔记 22 第 3 节） ----------
//
// accd_chord：一对图元沿弦的 ACCD（相对位移的界加分离平面的界）。返回的 t 必须是安全的：弦上 [0, t] 里的距离 - xi 处处 > 0。
// 只用相对位移的界时，贴着表面滑动的一对每步只走 c (d - xi) / ell，1000 步的上限常常走不完一段；
// 分离平面的界在平移滑动时整段成立，转着滑动时只剩二阶的真实闭合。

namespace {

// 弦上 t 处的距离（已减 radii），和 ccd.cc 的 pair_distance_at 一样。
double chord_distance(const Vector3d x0[4], const Vector3d x1[4], bool pt, double radii, double t) {
  Vector3d P[4];
  for (int i = 0; i < 4; ++i) P[i] = (1 - t) * x0[i] + t * x1[i];
  if (pt) {
    const Vector3d lam = point_triangle_closest(P[0], P[1], P[2], P[3]);
    return (P[0] - (lam[0] * P[1] + lam[1] * P[2] + lam[2] * P[3])).norm() - radii;
  }
  double s, u;
  segment_segment_closest(P[0], P[1], P[2], P[3], s, u);
  return (((1 - s) * P[0] + s * P[1]) - ((1 - u) * P[2] + u * P[3])).norm() - radii;
}

// [0, t] 上（4001 个点）距离 - xi 的最小值。
double chord_margin(const Vector3d x0[4], const Vector3d x1[4], bool pt, double radii, double xi, double t) {
  double m = std::numeric_limits<double>::infinity();
  for (int k = 0; k <= 4000; ++k) m = std::min(m, chord_distance(x0, x1, pt, radii, t * k / 4000.0) - xi);
  return m;
}

}  // namespace

// 两个盒子面对面叠着，间隙 1 μm，上面的盒子沿接触面滑 5 mm：真实距离不变，整步都是安全的。
//   只用相对位移的界时 ACCD 每步只敢走 c (d - xi) / ell = 0.9 x 1e-6 / 5e-3 = 1.8e-4，1000 步走到 0.18 就停了；
//   分离平面的界在平移滑动时每对顶点的斜率都是 0，整步放行。两个盒子一起绕 x 轴转 0.5 rad：接触面不和坐标轴对齐，
//   宽相的包围盒才分不开它们（面水平时两个包围盒在 z 上隔开 1 μm，宽相直接放行）；上面的盒子再沿面错开 5 cm，
//   让角落落在对方的面内（正对着的角落，每个顶点–三角形对的包围盒也分得开）。
TEST(CurvedCcd, SlidingAtATinyGapPassesTheWholeStep) {
  const Model M = load("stack.xml");
  VectorXd q = M.qpos0;
  const Quaterniond R(Eigen::AngleAxisd(0.5, Vector3d::UnitX()));
  q.segment<3>(0) = Vector3d(0, 0, 1);  // 抬高，离开地面
  q.segment<4>(3) << R.w(), R.x(), R.y(), R.z();
  q.segment<3>(7) = Vector3d(0, 0, 1) + R * Vector3d(0, 0.05, 0.2 + 1e-6);  // 间隙 1 μm，沿面错开 5 cm：角落在对方的面内
  q.segment<4>(10) << R.w(), R.x(), R.y(), R.z();
  ASSERT_NEAR(min_distance(M, forward_kinematics(M, q)), 1e-6, 1e-12);
  VectorXd p = VectorXd::Zero(M.nv);
  p[6] = 5e-3;  // 上面的盒子沿 x 平移 5 mm（x 轴在接触面里）
  EXPECT_EQ(curved_ccd(M, q, p, 1e-3, 0.9), 1.0);
  EXPECT_GT(trajectory_min_distance(M, q, p, 1.0, 64), 0.99e-6);
}

// 随机的刚体运动：两侧图元各自绕一个随机的轴、随机的支点转 0–0.3 rad，再平移（沿接触面滑 0–5 cm，
//   法向靠近或离开 1e-9–1e-3 m）。起点间隙 1e-9–1e-3 m，radii 是 0 或 12 mm（球），矢高取到间隙的 0.4 倍。
//   返回的 t 都要安全，一次最多闭合 c = 0.9，而且几乎从不走满 1000 步的上限。
TEST(Accd, PlaneBoundsAreSafeOnRandomRigidMotions) {
  std::mt19937 rng(20261114);
  std::uniform_real_distribution<double> U(0, 1);
  const auto unit = [&] { return Vector3d(U(rng) - 0.5, U(rng) - 0.5, U(rng) - 0.5).normalized(); };
  const auto logu = [&](double lo, double hi) { return lo * std::pow(hi / lo, U(rng)); };
  int capped = 0, trials = 0;
  long iters = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    const bool pt = trial % 2 == 0;
    Vector3d x0[4];
    const double size = logu(0.01, 0.3), gap = logu(1e-9, 1e-3), radii = U(rng) < 0.5 ? 0.0 : 0.012;
    Vector3d n;
    if (pt) {
      x0[1] = Vector3d::Zero();
      x0[2] = size * unit();
      x0[3] = size * unit();
      n = (x0[2] - x0[1]).cross(x0[3] - x0[1]).normalized();
      double l1 = U(rng), l2 = U(rng);
      if (l1 + l2 > 1) l1 = 1 - l1, l2 = 1 - l2;
      x0[0] = l1 * x0[2] + l2 * x0[3] + (gap + radii) * n;  // 点在三角形里面的正上方
    } else {
      const Vector3d e1 = unit(), e2 = unit();
      n = e1.cross(e2).normalized();
      const Vector3d c = (gap + radii) * n;
      const double s1 = U(rng), s2 = U(rng);
      x0[0] = c - s1 * size * e1;
      x0[1] = c + (1 - s1) * size * e1;
      x0[2] = -s2 * size * e2;
      x0[3] = (1 - s2) * size * e2;
    }
    if (std::abs(chord_distance(x0, x0, pt, radii, 0) - gap) > 1e-3 * gap + 1e-15) continue;  // 最近点不在内部
    // 两侧各自的刚体运动。
    Vector3d x1[4];
    const int na = pt ? 1 : 2;
    for (int side = 0; side < 2; ++side) {
      const Eigen::AngleAxisd R(0.3 * U(rng), unit());
      const Vector3d pivot = size * unit() * U(rng);
      Vector3d tan = unit();
      tan -= tan.dot(n) * n;
      const Vector3d T = 0.05 * U(rng) * tan.normalized() + (U(rng) < 0.5 ? -1 : 1) * logu(1e-9, 1e-3) * n;
      for (int i = side == 0 ? 0 : na; i < (side == 0 ? na : 4); ++i) x1[i] = pivot + R * (x0[i] - pivot) + T;
    }
    const double xi = 0.4 * gap * U(rng);
    ++trials;
    int it = 0;
    bool cap = false;
    const double t = accd_chord(x0, x1, pt, radii, xi, 0.9, &it, &cap);
    ASSERT_GE(t, 0.0);
    capped += cap;
    iters += it;
    const double m = chord_margin(x0, x1, pt, radii, xi, t);
    ASSERT_GT(m, 0.0) << "trial " << trial << (pt ? " pt" : " ee") << " gap " << gap << " t " << t;
    // 一次最多闭合 c = 0.9：[0, t] 上的余量不低于起始余量的 10%（每步的目标不低于这个门槛）。
    const double floor = 0.1 * (chord_distance(x0, x1, pt, radii, 0) - xi);
    EXPECT_GE(m, floor * (1 - 1e-6)) << "trial " << trial;
  }
  std::printf("  %d trials; capped %d; mean iterations %.1f\n", trials, capped, double(iters) / trials);
  EXPECT_GT(trials, 2000);
  EXPECT_LT(capped, trials / 200);
}

// 搅拌机里卡住的那种：球（一个顶点，半径 12 mm）贴着转动的叶片（包含转轴的平面 y = 1.5 mm，半径 0.03–0.26 m），
//   间隙 1.8e-7 m。这一段里叶片转 0.011 rad（66 rad/s、h = 2 ms、段长 0.084），球心沿直线跟着叶片在它那里的
//   法向速度走。真实的弦上间隙只按二阶闭合：球心离转轴所在平面 y0 = 13.5 mm，转 phi 以后间隙少 y0 phi^2 / 2 左右。
//   只用相对位移的界每步只走 c d / ell（ell 是叶尖的位移 2.9 mm 左右），1000 步走到 0.087；转动的平面 2 步走到 0.466。
TEST(Accd, RotatingPlaneFollowsABladeSweepingABall) {
  const double theta = 0.011, gap = 1.8e-7, radius = 0.012, rb = 0.15, y0 = 0.0015 + radius + gap;
  Vector3d x0[4], x1[4];
  x0[0] = Vector3d(rb, y0, -0.01);
  x0[1] = Vector3d(0.03, 0.0015, -0.03);
  x0[2] = Vector3d(0.26, 0.0015, -0.03);
  x0[3] = Vector3d(0.26, 0.0015, 0.03);
  const Eigen::AngleAxisd R(theta, Vector3d::UnitZ());
  for (int i = 1; i < 4; ++i) x1[i] = R * x0[i];
  x1[0] = x0[0] + Vector3d(0, rb * std::sin(theta), 0);  // 直线，法向速度和叶片在 r = rb 处一样
  ASSERT_NEAR(chord_distance(x0, x0, true, radius, 0), gap, 1e-12);
  int it = 0;
  bool cap = false;
  const double t = accd_chord(x0, x1, true, radius, 0.0, 0.9, &it, &cap);
  EXPECT_GT(chord_margin(x0, x1, true, radius, 0.0, t), 0.0);
  std::printf("  t %.4f (%d iterations%s); gap at t %.2e\n", t, it, cap ? ", capped" : "",
              chord_distance(x0, x1, true, radius, t));
  EXPECT_FALSE(cap);
  EXPECT_LT(it, 20);
  EXPECT_GT(t, 0.4);
  // 停下的地方间隙闭合了 89%–90%：不是保守地早停，也没有多闭合（每步的目标不低于起始间隙的 10%）。
  EXPECT_LT(chord_distance(x0, x1, true, radius, t), 0.11 * gap + 1e-12);
  EXPECT_GE(chord_distance(x0, x1, true, radius, t), 0.1 * gap * (1 - 1e-6));
}

// 布娃娃堆（mu = 0.25，第 151 步左右）里只用相对位移的界时走满上限的一对：两个盒子的边–边，边长 0.36 m、夹角 59 度，
//   间隙 5.5e-9 m，没有半径。最近点的方向只准到 1e-8 rad（r 只有 5.5e-9 m），边的另一端在这个方向上差出
//   4e-9 m，有的顶点对的 n·m 已经小于停下的水平，这个方向的界给不出步子。两条边的叉积不依赖间隙，
//   用它就能走。坐标是 scene_demo debug=3 打出来的那一段两端。
TEST(Accd, EdgeNormalWorksWhereTheClosestDirectionIsTooNoisy) {
  Vector3d x0[4], x1[4];
  x0[0] = Vector3d(0.15000439408195548, 0.18439156975600057, 0.22224128592428397);
  x1[0] = Vector3d(0.15008966320759703, 0.18448363967881634, 0.22224385582889472);
  x0[1] = Vector3d(-0.011557369840391823, -0.13716287550576065, 0.23226800875384548);
  x1[1] = Vector3d(-0.011481489098090825, -0.13706607084203318, 0.23227113893899815);
  x0[2] = Vector3d(-0.0028378938923135089, -0.059229251821122486, 0.22382017930821563);
  x1[2] = Vector3d(-0.0029388129990224054, -0.059179469652762107, 0.22380266407029673);
  x0[3] = Vector3d(0.34964917953066799, -0.026454203953004907, 0.28923164089519049);
  x1[3] = Vector3d(0.34956916384728964, -0.026453749244027158, 0.28912611685296324);
  const double radii = 1e-12, xi = 2.2281780579841374e-09;
  int it = 0;
  bool cap = false;
  const double t = accd_chord(x0, x1, false, radii, xi, 0.9, &it, &cap);
  EXPECT_GT(chord_margin(x0, x1, false, radii, xi, t), 0.0);
  std::printf("  t %.4f (%d iterations%s)\n", t, it, cap ? ", capped" : "");
  EXPECT_FALSE(cap);
  EXPECT_LT(it, 20);
}
