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

  int nbody() const { return static_cast<int>(bodies.size()); }
};

// 从 mjModel 抄出 Model。遇到不支持的结构时抛 std::runtime_error：
//   - 一个刚体上有多个关节（书里也要求每个刚体至多一个关节）；
//   - 自由关节不在顶层（父刚体不是世界）；
//   - mocap 刚体。
Model from_mujoco(const mjModel* m);

// mjModel 的智能指针：离开作用域时自动调用 mj_deleteModel。
struct MjModelDeleter {
  void operator()(mjModel* m) const { mj_deleteModel(m); }
};
using MjModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;

// 读一个 MJCF 文件；失败时抛 std::runtime_error，信息里带 MuJoCo 的报错。
MjModelPtr load_mjcf(const std::string& path);

}  // namespace rcipc
