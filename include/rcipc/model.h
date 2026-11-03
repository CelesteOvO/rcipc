#pragma once

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

struct Model {
  int nq = 0;                    // q 的长度
  int nv = 0;                    // 自由度个数，也就是 v 和 delta 的长度
  std::vector<Body> bodies;      // 下标就是刚体编号，0 号是世界。保证 parent < 自己的编号
  std::vector<Joint> joints;
  std::vector<int> dof_joint;    // 每个自由度属于哪个关节（长度 nv）
  std::vector<int> dof_body;     // 每个自由度属于哪个刚体（长度 nv）
  std::vector<bool> dof_rot;     // 每个自由度是转动（true）还是平移（false）
  Eigen::VectorXd qpos0;         // 参考位置：MJCF 里写的姿态
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

  int nbody() const { return static_cast<int>(bodies.size()); }
  int nvertex() const { return static_cast<int>(vertices.size()); }
  int nedge() const { return static_cast<int>(edges.size()); }
  int nface() const { return static_cast<int>(faces.size()); }

  // 两个刚体之间要不要检测接触（阶段 7）：不同的刚体，而且不是父子（MuJoCo 的默认规则；世界除外，
  // 挂在世界上的刚体和世界上的静态几何要检测）。
  bool collide(int a, int b) const {
    if (a == b) return false;
    if (bodies[a].parent == b && b != 0) return false;
    if (bodies[b].parent == a && a != 0) return false;
    return true;
  }
};

// 从 mjModel 抄出 Model。遇到不支持的结构时抛 std::runtime_error：
//   - 一个刚体上有多个关节（书里也要求每个刚体至多一个关节）；
//   - 自由关节不在顶层（父刚体不是世界）；
//   - mocap 刚体；
//   - 阶段 4 起：geom 不是 plane、box、sphere、capsule；plane 不在世界上。
//     （MJCF 里 contype="0" conaffinity="0" 的 geom 只用来显示，会被跳过。）
//   阶段 7 起世界上也可以放 box、sphere、capsule，当作静止的障碍物。
Model from_mujoco(const mjModel* m);

// mjModel 的智能指针：离开作用域时自动调用 mj_deleteModel。
struct MjModelDeleter {
  void operator()(mjModel* m) const { mj_deleteModel(m); }
};
using MjModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;

// 读一个 MJCF 文件；失败时抛 std::runtime_error，信息里带 MuJoCo 的报错。
MjModelPtr load_mjcf(const std::string& path);

}  // namespace rcipc
