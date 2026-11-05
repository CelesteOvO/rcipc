#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mujoco/mujoco.h>

// 关节树的模型（阶段 2，笔记 notes/02_kinematics.md 第 1 节）。
//
// 只保存我们需要的东西，从 MuJoCo 的 mjModel 里抄出来：阶段 2 的树结构，阶段 3 的质量和惯量。
// 之后的阶段会继续往这里加字段（阶段 4：顶点）。我们自己的代码只读这个结构，不再碰 mjModel。
namespace rcipc {

// 四种关节。括号里是它在 q 和 v 里各占几个数。
enum class JointType {
  kFree,   // 自由关节（q 7 个：位置 3 + 四元数 4；v 6 个：世界系线速度 3 + 刚体系角速度 3）
  kBall,   // 球关节  （q 4 个：四元数；v 3 个：刚体系角速度）
  kSlide,  // 滑动关节（q 1 个；v 1 个），单位 m
  kHinge,  // 铰链    （q 1 个；v 1 个），单位 rad
};

struct Joint {
  JointType type;
  int body;              // 关节所在的刚体
  int qadr;              // 在 q 里的起始下标（MuJoCo 的 jnt_qposadr）
  int dadr;              // 在 v（以及增量 delta）里的起始下标（jnt_dofadr）
  int nq;                // 在 q 里占几个数：7、4、1、1
  int nv;                // 自由度个数：6、3、1、1
  Eigen::Vector3d pos;   // 锚点，在刚体自己的坐标系里（jnt_pos）
  Eigen::Vector3d axis;  // 铰链和滑动的轴，刚体坐标系，单位向量（jnt_axis）
  double ref;            // 铰链和滑动的参考值（qpos0）：q = ref 时，刚体就在 MJCF 里写的位置。其余类型为 0
  // 阶段 10：关节限位（MJCF 的 limited="true" range="lo hi"），只支持铰链和滑动。q 必须保持在 (lo, hi) 里，
  // 用和接触一样的障碍函数（limits.h）。
  bool limited = false;
  double range[2] = {0, 0};
};

struct Body {
  int parent;               // 父刚体的编号。0 号是世界，它的 parent 记为 -1
  Eigen::Vector3d pos;      // 在父刚体坐标系里的固定偏移（body_pos）
  Eigen::Quaterniond quat;  // 在父刚体坐标系里的固定姿态（body_quat）
  int joint;                // 这个刚体的关节编号；没有关节（焊死在父刚体上）时为 -1

  // 阶段 3：质量分布（笔记 notes/03_inertia.md 第 1 节）。下面三个量都在刚体自己的坐标系里。
  double mass = 0;                                    // 质量 m_b（kg）
  Eigen::Vector3d ipos = Eigen::Vector3d::Zero();     // 质心的位置（body_ipos）
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();  // 绕质心的惯性张量 I_b（kg m^2）
  Eigen::Matrix3d S = Eigen::Matrix3d::Zero();        // 绕质心的二阶矩 S_b = sum m_p X_p X_p^T，I = tr(S) 1 - S

  // 阶段 5：刚体半径，所有顶点到刚体原点的距离（加顶点半径）的最大值。CCD 的加速度界用它当力臂。
  double radius = 0;
};

// 阶段 4：接触用的几何。
//
// 顶点：刚体上的一个点，外加一个半径（笔记 notes/04_ground.md 第 1 节）。
//   盒子的 8 个角半径为 0；球是球心加半径；胶囊是两个端点加半径。
//   到障碍的距离 = 点到障碍的距离 - 半径，所以球和胶囊的圆弧面也能正确处理。
struct Vertex {
  int body;               // 所在刚体
  Eigen::Vector3d local;  // 刚体坐标系里的位置
  double radius;          // 半径（m），盒子的角是 0
};

// 平面障碍（地面）：自由的一侧是 normal . x - offset > 0。只允许挂在世界（0 号刚体）上，不动。
struct Plane {
  Eigen::Vector3d normal;  // 单位法向，指向自由的一侧
  double offset;           // normal . x = offset 是平面
};

// 阶段 7：刚体之间的接触用的边和面（笔记 notes/07_contact.md 第 1 节）。下标都指向 Model::vertices。
//   盒子：12 条边、12 个三角形；胶囊：1 条边（半径是端点的半径）；球：1 条退化的边（两端是同一个顶点）。
//   边和面的半径就是它们顶点的半径（同一个 geom 的顶点半径相同）。
struct Edge {
  int v0, v1;
  int body;
};
struct Face {
  int v0, v1, v2;
  int body;
};

// 阶段 12：驱动（MJCF 的 actuator，笔记 notes/12_drives.md）。只支持 MuJoCo 的 general 模型里的一部分：
//   传动是关节（铰链或滑动），没有激活动力学（dyntype none），增益固定，偏置是 none 或 affine。
//   position、velocity、motor 这几种快捷写法都在这个范围里。
//   驱动力 f = gain u + bias[0] + bias[1] L + bias[2] dL/dt，L = gear q 是驱动的“长度”，关节上的力矩是 gear f。
struct Actuator {
  int joint = -1;          // 驱动的关节
  int dof = -1, qadr = -1; // 那个关节的自由度、在 q 里的下标
  double gear = 1;         // 传动比（actuator_gear 的第一个数）
  double gain = 0;         // gainprm[0]
  double bias[3] = {0, 0, 0};  // biasprm[0..2]
  bool ctrl_limited = false;   // 控制量 u 夹在 ctrl_range 里
  double ctrl_range[2] = {0, 0};
  bool force_limited = false;  // 驱动力 f 夹在 force_range 里
  double force_range[2] = {0, 0};
  // 关节上的 actuatorfrcrange（G1 之后）：关节上只有这一个驱动时，把“关节力矩 gear f 夹在 [lo, hi]”
  // 换成“f 夹在 [lo, hi] / gear”，和 force_range 取交集，所以 force_range 已经包含它。
};

// 阶段 12：控制量的时间表，来自 MJCF 的关键帧 <key time="..." ctrl="..."/>，按时间分段线性插值（drive.h 的 ctrl_at）。
struct CtrlKey {
  double time;
  Eigen::VectorXd ctrl;  // 长度 nu
};

struct Model {
  int nq = 0;                    // q 的长度
  int nv = 0;                    // 自由度个数，也就是 v 和 delta 的长度
  std::vector<Body> bodies;      // 下标就是刚体编号，0 号是世界。保证 parent < 自己的编号
  std::vector<Joint> joints;
  std::vector<int> dof_joint;    // 每个自由度属于哪个关节（长度 nv）
  std::vector<int> dof_body;     // 每个自由度属于哪个刚体（长度 nv）
  std::vector<bool> dof_rot;     // 每个自由度是转动（true）还是平移（false）
  Eigen::VectorXd qpos0;         // 参考位置：MJCF 里写的姿态
  Eigen::VectorXd key_qpos;      // 第一个关键帧的 qpos（没有关键帧时为空）。scene_demo 有它就从它开始
  Eigen::Vector3d gravity{0, 0, -9.81};  // 重力加速度（MJCF 的 option gravity），阶段 3 起用
  double timestep = 0.002;       // MJCF 的 option timestep（s）
  std::vector<Vertex> vertices;  // 阶段 4：所有刚体的顶点（从 MJCF 的 geom 读出）
  std::vector<Plane> planes;     // 阶段 4：世界上的平面障碍（MJCF 里挂在 worldbody 上的 plane geom）
  std::vector<Edge> edges;       // 阶段 7：边
  std::vector<Face> faces;       // 阶段 7：三角形
  // 阶段 9：树。挂在世界上的每个刚体连同它的子孙是一棵运动学树；树内部的自由度通过质量矩阵紧密耦合，
  // 树之间只通过接触耦合。预条件器按树分块。
  int ntree = 0;
  std::vector<int> body_tree;    // 每个刚体属于哪棵树（世界是 -1）
  std::vector<int> dof_tree;     // 每个自由度属于哪棵树（长度 nv）
  // 阶段 12：驱动、关节阻尼和 armature（drive.h）。阻尼和 armature 只用在铰链和滑动的自由度上。
  std::vector<double> dof_damping;   // 长度 nv（dof_damping，N s/m 或 N m s/rad）
  std::vector<double> dof_armature;  // 长度 nv（dof_armature，加在质量矩阵对角上的转子惯量）
  std::vector<double> dof_frictionloss;  // 长度 nv：关节的干摩擦力（矩）（G1 之后，drive.h）。球、自由关节上的是 0
  std::vector<Actuator> actuators;
  std::vector<CtrlKey> ctrl_keys;    // 控制量的时间表（没有驱动或没有关键帧时为空）
  std::vector<std::string> ignored;  // 读到了、但不支持而被忽略的设置（比如关节的 frictionloss），演示程序会打印

  int nu() const { return static_cast<int>(actuators.size()); }
  int nbody() const { return static_cast<int>(bodies.size()); }
  int nvertex() const { return static_cast<int>(vertices.size()); }
  int nedge() const { return static_cast<int>(edges.size()); }
  int nface() const { return static_cast<int>(faces.size()); }

  // 两个刚体之间要不要检测接触（阶段 7）：不同的刚体，而且不是父子（MuJoCo 的默认规则；世界除外，
  // 挂在世界上的刚体和世界上的静态几何要检测）。
  //   阶段 12 之后按 MuJoCo 的写法，比的是“焊在一起的整体”（weld）：没有关节的刚体和它的父刚体是一个整体
  //   （body_weldid）。同一个整体里的两个刚体不检测；两个整体是父子时也不检测。Allegro 手的指尖焊在末节上，
  //   指尖和中节因此算父子，不检测（它们本来就重叠）。每个刚体都有关节时，和原来的规则一样。
  //   MJCF 里 <option><flag filterparent="disable"/></option> 关掉父子过滤（和 libuipc 对比时用：它检测所有刚体对）。
  bool filter_parent = true;
  // 阶段 12 之后：MJCF 的 <contact><exclude body1 body2/> 指定的刚体对也不检测（Allegro 手的手掌和拇指根部就是）。
  //   每对编码成 min(a, b) * nbody + max(a, b)，排好序。
  std::vector<int> exclude;
  std::vector<int> weld;  // 每个刚体所在的整体（MuJoCo 的 body_weldid），长度 nbody
  bool collide(int a, int b) const {
    if (a == b) return false;
    const int wa = weld[a], wb = weld[b];
    if (wa == wb) return false;
    if (filter_parent && wa != 0 && wb != 0 && (weld[bodies[wa].parent] == wb || weld[bodies[wb].parent] == wa)) return false;
    if (!exclude.empty()) {
      const int key = std::min(a, b) * nbody() + std::max(a, b);
      if (std::binary_search(exclude.begin(), exclude.end(), key)) return false;
    }
    return true;
  }
};

// 从 mjModel 抄出 Model。遇到不支持的结构时抛 std::runtime_error：
//   - 一个刚体上有多个关节（书里也要求每个刚体至多一个关节）；
//   - 自由关节不在顶层（父刚体不是世界）；
//   - mocap 刚体；
//   - 阶段 4 起：geom 不是 plane、box、sphere、capsule（阶段 12 之后加了 cylinder、mesh）；plane 不在世界上。
//     （MJCF 里 contype="0" conaffinity="0" 的 geom 只用来显示，会被跳过。）
//   阶段 7 起世界上也可以放 box、sphere、capsule，当作静止的障碍物。
//   阶段 12 起：驱动不在 Actuator 说明的范围里（传动不是铰链或滑动、有激活动力学、增益不固定、偏置不是 affine，
//     或者偏置让驱动变成负刚度、负阻尼）。
Model from_mujoco(const mjModel* m);

// mjModel 的智能指针：离开作用域时自动调用 mj_deleteModel。
struct MjModelDeleter {
  void operator()(mjModel* m) const { mj_deleteModel(m); }
};
using MjModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;

// 读一个 MJCF 文件；失败时抛 std::runtime_error，信息里带 MuJoCo 的报错。
MjModelPtr load_mjcf(const std::string& path);

}  // namespace rcipc
