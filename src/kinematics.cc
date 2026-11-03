#include "rcipc/kinematics.h"

#include "rcipc/quat.h"

// 关节坐标的运动学。推导见 notes/02_kinematics.md，测试见 tests/test_kinematics.cc。
//
// Eigen 的几个写法，下面会用到：
//   q.segment<3>(a)          q 里从下标 a 开始的 3 个数（可读可写）
//   Vector3d::Unit(k)        第 k 个坐标轴 e_k
//   r * v                    用四元数 r 转动向量 v
//   r1 * r2                  四元数乘法：先转 r2，再转 r1（在 r1 的坐标系里写 r2）
//   a.cross(b)               叉乘 a x b
//   quat_load(&q[a])         从 q[a..a+3] 读一个四元数（MuJoCo 顺序 w, x, y, z）
//   quat_store(r, &q[a])     把四元数写回 q[a..a+3]
namespace rcipc {

using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

Frames forward_kinematics(const Model& model, const VectorXd& q) {
  // 正运动学（笔记第 3 节，书 §7.3 第 147–148 页）：从根往下，逐个刚体算出世界位姿。
  //
  // 准备：F.xpos、F.xquat 各 nbody 个，F.dof_axis、F.dof_anchor 各 nv 个。
  //   用 assign 给初值（Eigen 的默认构造不初始化，里面是垃圾数）。
  //   世界（0 号刚体）的位姿就是这里的初值：原点和单位四元数，循环从 1 号开始。
  const int nb = model.nbody();
  Frames F;
  F.xpos.assign(nb, Vector3d::Zero());
  F.xquat.assign(nb, Quaterniond::Identity());
  F.dof_axis.assign(model.nv, Vector3d::Zero());
  F.dof_anchor.assign(model.nv, Vector3d::Zero());

  // 对每个刚体 b = 1, 2, ..., nbody - 1。因为 parent < b，处理 b 时它的父刚体已经算好了。
  for (int b = 1; b < nb; ++b) {
    const Body& B = model.bodies[b];
    const int p = B.parent;

    // 第 1 步：固定偏移。先假设关节不动，偏移是在父坐标系里写的，要先用父刚体的姿态转到世界。
    //   x、r 是这个刚体的位置和姿态，第 2 步还会修改它们，所以不加 const。
    Vector3d x = F.xpos[p] + F.xquat[p] * B.pos;
    Quaterniond r = F.xquat[p] * B.quat;

    // 第 2 步：关节。没有关节（B.joint == -1）的刚体焊死在父刚体上，第 1 步的 x、r 就是答案。
    if (B.joint >= 0) {
      const Joint& J = model.joints[B.joint];
      const int a = J.qadr;  // 这个关节在 q 里的起始下标
      const int d = J.dadr;  // 这个关节的第一个自由度的编号

      switch (J.type) {
        case JointType::kHinge: {
          // 铰链：绕锚点 o、沿轴 J.axis 转角 t = q[a] - J.ref。
          //   ref 是 MJCF 里写的参考角：q = ref 时刚体就在 MJCF 写的姿态上，所以实际转过的角度要减掉它。
          const Vector3d o = x + r * J.pos;  // 锚点的世界坐标，必须在转动之前算（这时 r 还是“关节不动”的姿态）
          F.dof_axis[d] = r * J.axis;        // 轴转到世界系。绕自己转不改变轴，所以转动前后算都一样
          F.dof_anchor[d] = o;

          const double t = q[a] - J.ref;
          r = r * quat_exp(t * J.axis);  // 右乘：在刚体自己的坐标系里绕 J.axis 转 t
          x = o - r * J.pos;             // 锚点不动。从锚点出发，沿“新姿态下锚点到原点”的方向倒推原点
          break;
        }

        case JointType::kBall: {
          // 球关节：和铰链一样绕锚点转，只是转动直接是 q 里的四元数（可能没归一化，先归一化）。
          const Vector3d o = x + r * J.pos;
          r = r * quat_load(&q[a]).normalized();
          x = o - r * J.pos;

          // 3 个自由度的轴是转动之后的刚体坐标轴（用新的 r），锚点都是 o。
          // 这对应“球关节的角速度写在刚体自己的坐标系里”。
          for (int k = 0; k < 3; ++k) {
            F.dof_axis[d + k] = r * Vector3d::Unit(k);
            F.dof_anchor[d + k] = o;
          }
          break;
        }

        case JointType::kSlide: {
          // 滑动：沿轴平移 q[a] - J.ref，不转动，所以 r 不变。
          F.dof_axis[d] = r * J.axis;
          F.dof_anchor[d] = x + r * J.pos;  // 平移自由度用不到锚点，填上只是为了和 MuJoCo 的 xanchor 一样
          x = x + F.dof_axis[d] * (q[a] - J.ref);
          break;
        }

        case JointType::kFree: {
          // 自由关节：不管父刚体和第 1 步的固定偏移，位姿直接取 q 里的 7 个数。
          x = q.segment<3>(a);
          r = quat_load(&q[a + 3]).normalized();

          for (int k = 0; k < 3; ++k) {
            // 前 3 个自由度（平移）：轴是世界坐标轴，因为自由关节的线速度写在世界系里。
            F.dof_axis[d + k] = Vector3d::Unit(k);
            F.dof_anchor[d + k] = x;
            // 后 3 个自由度（转动）：轴是刚体坐标轴，锚点是刚体原点。
            F.dof_axis[d + 3 + k] = r * Vector3d::Unit(k);
            F.dof_anchor[d + 3 + k] = x;
          }
          break;
        }
      }
    }

    // 第 3 步：保存。乘法本身保持长度，归一化只是把累积的舍入误差去掉（MuJoCo 也这样做）。
    F.xpos[b] = x;
    F.xquat[b] = r.normalized();
  }
  return F;
}

Vector3d world_point(const Frames& frames, int body, const Vector3d& local) {
  // 刚体上的点随刚体一起运动：先按刚体的姿态转，再加上刚体原点。
  return frames.xpos[body] + frames.xquat[body] * local;
}

VectorXd integrate(const Model& model, const VectorXd& q, const VectorXd& delta, double s) {
  // 位置的加法 q ⊕ s * delta（笔记第 2.2 节，书 §7.2 第 145 页）。
  //
  // 它不是物理意义上的“时间积分”：这里没有力、没有质量，只是一个几何运算——
  // 从位置 q 出发，沿切空间里的方向 delta 走 s 倍。名字跟 MuJoCo 的 mj_integratePos 一致：
  //   - 时间步末尾用 q_new = q ⊕ h * v，这时它确实是在“按速度积分位置”；
  //   - Newton 迭代里用 q_i ⊕ alpha * p，这时它只是“沿搜索方向走一步”。
  //
  // 为什么转动是右乘：右乘 exp(w) 表示“在刚体坐标系里转 w”，左乘表示“在世界坐标系里转 w”。
  //   MuJoCo 规定球关节和自由关节的角速度写在刚体坐标系里，所以是右乘。
  //   写成左乘的话，测试 Integrate.MatchesMujoco 会失败。

  // 先复制一份 q，然后每个关节只改写属于自己的那几个数。
  VectorXd out = q;

  for (const Joint& J : model.joints) {
    const int a = J.qadr;  // 这个关节在 q 里的起始下标
    const int d = J.dadr;  // 这个关节在 delta 里的起始下标

    switch (J.type) {
      case JointType::kHinge:
      case JointType::kSlide:
        // 铰链、滑动：q 和 delta 都是 1 个数，直接相加。
        out[a] = q[a] + s * delta[d];
        break;

      case JointType::kBall: {
        // 球关节：q 是 4 个数的四元数，delta 是 3 个数的旋转向量。
        //   先用 quat_exp 把旋转向量变成四元数，再右乘到原来的姿态上。
        const Quaterniond r = quat_load(&q[a]).normalized();
        const Quaterniond step = quat_exp(s * delta.segment<3>(d));
        quat_store(r * step, &out[a]);
        break;
      }

      case JointType::kFree: {
        // 自由关节：前 3 个数是位置，直接相加（线速度写在世界系，所以就是普通的向量加法）。
        out.segment<3>(a) = q.segment<3>(a) + s * delta.segment<3>(d);
        // 后面是转动，和球关节一样，只是四元数从 q[a + 3] 开始，旋转向量从 delta[d + 3] 开始。
        const Quaterniond r = quat_load(&q[a + 3]).normalized();
        const Quaterniond step = quat_exp(s * delta.segment<3>(d + 3));
        quat_store(r * step, &out[a + 3]);
        break;
      }
    }
  }
  return out;
}

VectorXd differentiate(const Model& model, const VectorXd& q0, const VectorXd& q1) {
  // 位置的减法 q1 ⊖ q0（笔记第 2.2 节）：找 delta，使 q0 ⊕ delta = q1。
  //
  // 它是 integrate 的逆：integrate 是“从 q0 沿 delta 走到哪里”，differentiate 是“从 q0 到 q1 要走多少”。
  // 结果有 nv 个数（不是 nq 个），和速度、Newton 方向在同一个空间里。
  // 一个时间步结束时用它回算速度：v = (q_new ⊖ q_old) / h，对应阶段 1 的 v = (x - x0) / h。

  VectorXd delta(model.nv);

  for (const Joint& J : model.joints) {
    const int a = J.qadr;  // 这个关节在 q 里的起始下标
    const int d = J.dadr;  // 这个关节在 delta 里的起始下标

    switch (J.type) {
      case JointType::kHinge:
      case JointType::kSlide:
        // 铰链、滑动：普通的减法。
        delta[d] = q1[a] - q0[a];
        break;

      case JointType::kBall: {
        // 球关节：要解 r0 * exp(w) = r1。两边左乘 r0 的逆（单位四元数的逆就是共轭）：
        //     exp(w) = r0^-1 * r1    =>    w = log(r0^-1 * r1)
        // r0^-1 * r1 是“从 r0 到 r1 还差的那个转动”，在 r0 的刚体坐标系里表示（和 integrate 的右乘对应）。
        // r0、r1 不必归一化：共轭乘积只差一个正的倍数，quat_log 只看比例。
        const Quaterniond r0 = quat_load(&q0[a]);
        const Quaterniond r1 = quat_load(&q1[a]);
        delta.segment<3>(d) = quat_log(r0.conjugate() * r1);
        break;
      }

      case JointType::kFree: {
        // 自由关节：位置部分是普通的减法，转动部分和球关节一样（四元数从 a + 3 开始，结果写到 d + 3）。
        delta.segment<3>(d) = q1.segment<3>(a) - q0.segment<3>(a);
        const Quaterniond r0 = quat_load(&q0[a + 3]);
        const Quaterniond r1 = quat_load(&q1[a + 3]);
        delta.segment<3>(d + 3) = quat_log(r0.conjugate() * r1);
        break;
      }
    }
  }
  return delta;
}

Eigen::Matrix3Xd point_jacobian(const Model& model, const Frames& frames, int body, const Vector3d& x) {
  // 点的 Jacobian（笔记第 4 节，书 §7.3 第 148 页）：刚体 body 上、世界坐标为 x 的点，对 delta 的导数。
  //
  // 第 d 列 = “只让第 d 个自由度以速度 1 运动时，这个点的速度”：
  //   转动自由度：整个子树绕过锚点 o、方向 a 的轴转，点的速度是 a x (x - o)
  //              （锚点自己速度为 0；离转轴越远动得越快，方向沿圆弧的切线）；
  //   平移自由度：整个子树沿 a 平移，每个点的速度都是 a，和锚点无关；
  //   不在 body 到根的链上的自由度：动不了这个点，这一列是 0。

  // 第 1 步：先全部填 0，下面只填链上的列。
  Eigen::Matrix3Xd jac = Eigen::Matrix3Xd::Zero(3, model.nv);

  // 第 2 步：从 body 沿父刚体往上走到世界（0 号）。路上每个刚体的关节都会带动这个点。
  for (int b = body; b != 0; b = model.bodies[b].parent) {
    const int j = model.bodies[b].joint;
    if (j < 0) continue;  // 焊死的刚体没有关节，不贡献列，但还要继续往上走
    const Joint& J = model.joints[j];

    // 第 3 步：这个关节的每个自由度填一列。轴和锚点在 FK 里已经算好（世界系）。
    for (int k = 0; k < J.nv; ++k) {
      const int d = J.dadr + k;
      const Vector3d& a = frames.dof_axis[d];
      if (model.dof_rot[d]) {
        jac.col(d) = a.cross(x - frames.dof_anchor[d]);  // 转动：a x (x - o)
      } else {
        jac.col(d) = a;                                   // 平移：a
      }
    }
  }
  return jac;
}

Eigen::Matrix3Xd angular_jacobian(const Model& model, const Frames& frames, int body) {
  // 刚体的角速度只来自链上的转动自由度：每个以速度 1 转动的自由度给刚体加上角速度 a_d。
  // 平移自由度不改变姿态，那一列是 0。结构和 point_jacobian 完全一样。
  Eigen::Matrix3Xd jac = Eigen::Matrix3Xd::Zero(3, model.nv);
  for (int b = body; b != 0; b = model.bodies[b].parent) {
    const int j = model.bodies[b].joint;
    if (j < 0) continue;
    const Joint& J = model.joints[j];
    for (int k = 0; k < J.nv; ++k) {
      const int d = J.dadr + k;
      if (model.dof_rot[d]) jac.col(d) = frames.dof_axis[d];
    }
  }
  return jac;
}

}  // namespace rcipc
