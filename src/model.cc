#include "rcipc/model.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "rcipc/quat.h"

namespace rcipc {

namespace {

// MuJoCo 的关节类型 -> 我们的 JointType，同时给出它在 q 和 v 里占几个数。
void joint_type(int mj_type, Joint& j) {
  switch (mj_type) {
    case mjJNT_FREE:  j.type = JointType::kFree;  j.nq = 7; j.nv = 6; return;
    case mjJNT_BALL:  j.type = JointType::kBall;  j.nq = 4; j.nv = 3; return;
    case mjJNT_SLIDE: j.type = JointType::kSlide; j.nq = 1; j.nv = 1; return;
    case mjJNT_HINGE: j.type = JointType::kHinge; j.nq = 1; j.nv = 1; return;
  }
  throw std::runtime_error("unknown joint type " + std::to_string(mj_type));
}

// 关节 j 的第 k 个自由度是不是转动：自由关节的后 3 个、球关节的 3 个、铰链是转动；其余是平移。
bool is_rotation(const Joint& j, int k) {
  switch (j.type) {
    case JointType::kFree:  return k >= 3;
    case JointType::kBall:  return true;
    case JointType::kSlide: return false;
    case JointType::kHinge: return true;
  }
  return false;
}

}  // namespace

Model from_mujoco(const mjModel* m) {
  Model M;
  M.nq = static_cast<int>(m->nq);  // mjtSize 是 64 位整数，我们的模型远小于 int 的范围
  M.nv = static_cast<int>(m->nv);
  M.qpos0 = Eigen::Map<const Eigen::VectorXd>(m->qpos0, m->nq);
  M.gravity = Eigen::Vector3d(m->opt.gravity);
  M.timestep = m->opt.timestep;

  // 关节：逐个抄出类型、地址、锚点和轴。
  M.joints.resize(m->njnt);
  for (int i = 0; i < m->njnt; ++i) {
    Joint& j = M.joints[i];
    joint_type(m->jnt_type[i], j);
    j.body = m->jnt_bodyid[i];
    j.qadr = m->jnt_qposadr[i];
    j.dadr = m->jnt_dofadr[i];
    j.pos = Eigen::Vector3d(m->jnt_pos + 3 * i);
    j.axis = Eigen::Vector3d(m->jnt_axis + 3 * i);  // MuJoCo 编译时已经归一化
    const bool scalar = j.type == JointType::kHinge || j.type == JointType::kSlide;
    j.ref = scalar ? m->qpos0[j.qadr] : 0.0;
    // 阶段 10：限位。MuJoCo 的 jnt_range 对铰链是弧度（compiler angle="radian"）。球关节的限位是转角的上限，不支持。
    j.limited = scalar && m->jnt_limited[i] != 0;
    if (m->jnt_limited[i] && !scalar) throw std::runtime_error("joint " + std::to_string(i) + ": only hinge and slide limits are supported");
    j.range[0] = m->jnt_range[2 * i];
    j.range[1] = m->jnt_range[2 * i + 1];
  }

  // 刚体：父刚体、固定偏移、关节编号。顺便检查我们不支持的结构。
  M.bodies.resize(m->nbody);
  for (int b = 0; b < m->nbody; ++b) {
    Body& B = M.bodies[b];
    B.parent = b == 0 ? -1 : m->body_parentid[b];
    B.pos = Eigen::Vector3d(m->body_pos + 3 * b);
    B.quat = quat_load(m->body_quat + 4 * b);
    B.joint = m->body_jntnum[b] > 0 ? m->body_jntadr[b] : -1;
    // 质量分布。MuJoCo 给的是主惯量 body_inertia（3 个数）和主轴的姿态 body_iquat（相对刚体坐标系）。
    //   主轴坐标系里惯性张量是对角阵 diag(I1, I2, I3)，二阶矩也是对角阵：S_k = tr(I)/2 - I_k
    //   （由 I = tr(S) 1 - S 两边取迹得 tr(I) = 2 tr(S)，再反解）。
    //   然后用主轴的转动矩阵 Q 转到刚体坐标系：I_b = Q diag(I) Q^T，S_b = Q diag(S) Q^T。
    B.mass = m->body_mass[b];
    B.ipos = Eigen::Vector3d(m->body_ipos + 3 * b);
    const Eigen::Vector3d Ip(m->body_inertia + 3 * b);
    const Eigen::Vector3d Sp = Eigen::Vector3d::Constant(0.5 * Ip.sum()) - Ip;
    const Eigen::Matrix3d Q = quat_load(m->body_iquat + 4 * b).normalized().toRotationMatrix();
    B.inertia = Q * Ip.asDiagonal() * Q.transpose();
    B.S = Q * Sp.asDiagonal() * Q.transpose();

    const std::string name = mj_id2name(m, mjOBJ_BODY, b) ? mj_id2name(m, mjOBJ_BODY, b) : std::to_string(b);
    if (m->body_jntnum[b] > 1) throw std::runtime_error("body " + name + " has more than one joint");
    if (m->body_mocapid[b] >= 0) throw std::runtime_error("body " + name + " is a mocap body");
    if (B.joint >= 0 && M.joints[B.joint].type == JointType::kFree && B.parent != 0) {
      throw std::runtime_error("body " + name + ": free joint below the top level");
    }
  }

  // 阶段 4：接触几何。每个 geom 变成几个顶点（笔记 notes/04_ground.md 第 1 节），世界上的平面变成障碍。
  //   MuJoCo 已经把 fromto、euler 等写法编译成了 geom_pos、geom_quat、geom_size（半尺寸）。
  //   contype 和 conaffinity 都是 0 的 geom 不参与碰撞（只用来显示），跳过。
  for (int gi = 0; gi < m->ngeom; ++gi) {
    if (m->geom_contype[gi] == 0 && m->geom_conaffinity[gi] == 0) continue;
    const int b = m->geom_bodyid[gi];
    const Eigen::Vector3d pos(m->geom_pos + 3 * gi);
    const Eigen::Quaterniond quat = quat_load(m->geom_quat + 4 * gi).normalized();
    const double* size = m->geom_size + 3 * gi;
    const std::string name = mj_id2name(m, mjOBJ_GEOM, gi) ? mj_id2name(m, mjOBJ_GEOM, gi) : std::to_string(gi);

    switch (m->geom_type[gi]) {
      case mjGEOM_PLANE: {
        // 平面：geom 坐标系的 z 轴是法向，自由的一侧是 +z。只支持挂在世界上的平面（它不动）。
        if (b != 0) throw std::runtime_error("plane " + name + " is not on the world body");
        const Eigen::Vector3d n = quat * Eigen::Vector3d::UnitZ();
        M.planes.push_back({n, n.dot(pos)});
        break;
      }
      case mjGEOM_BOX: {
        // 盒子：8 个角（半径 0）、12 条边、12 个三角形（每个面两个）。size 是三个半边长。
        // 角的编号：k = 4 ix + 2 iy + iz，i = 0 是负方向、1 是正方向。
        const int v0 = M.nvertex();
        for (int k = 0; k < 8; ++k) {
          const double sx = (k & 4) ? 1 : -1, sy = (k & 2) ? 1 : -1, sz = (k & 1) ? 1 : -1;
          M.vertices.push_back({b, pos + quat * Eigen::Vector3d(sx * size[0], sy * size[1], sz * size[2]), 0.0});
        }
        // 边：两个角只差一个坐标的符号。
        for (int k = 0; k < 8; ++k) {
          for (int bit : {1, 2, 4}) {
            if (!(k & bit)) M.edges.push_back({v0 + k, v0 + (k | bit), b});
          }
        }
        // 面：6 个面各两个三角形。每个面固定一个坐标，另两个坐标的 4 个角按 (00, 01, 11, 10) 绕一圈。
        for (int axis = 0; axis < 3; ++axis) {
          const int fixed = 4 >> axis;  // 固定的那一位：x 是 4，y 是 2，z 是 1
          const int u = (fixed == 4) ? 2 : 4, w = (fixed == 1) ? 2 : 1;  // 另外两位
          for (int side = 0; side < 2; ++side) {
            const int base = side ? fixed : 0;
            const int c00 = base, c01 = base | w, c11 = base | u | w, c10 = base | u;
            M.faces.push_back({v0 + c00, v0 + c01, v0 + c11, b});
            M.faces.push_back({v0 + c00, v0 + c11, v0 + c10, b});
          }
        }
        break;
      }
      case mjGEOM_SPHERE:
        // 球：球心加半径；一条退化的边（两端是同一个顶点），让球和球、球和边都走边–边的最近点。
        M.vertices.push_back({b, pos, size[0]});
        M.edges.push_back({M.nvertex() - 1, M.nvertex() - 1, b});
        break;
      case mjGEOM_CAPSULE:
        // 胶囊：轴是 geom 坐标系的 z 轴，size[1] 是半长，size[0] 是半径。两个端点（球心）加半径，一条边。
        M.vertices.push_back({b, pos + quat * Eigen::Vector3d(0, 0, size[1]), size[0]});
        M.vertices.push_back({b, pos - quat * Eigen::Vector3d(0, 0, size[1]), size[0]});
        M.edges.push_back({M.nvertex() - 2, M.nvertex() - 1, b});
        break;
      default:
        throw std::runtime_error("geom " + name + ": only plane, box, sphere and capsule are supported");
    }
  }

  // 阶段 5：刚体半径 = max（顶点到刚体原点的距离 + 顶点半径）。
  for (const Vertex& v : M.vertices) {
    Body& B = M.bodies[v.body];
    B.radius = std::max(B.radius, v.local.norm() + v.radius);
  }

  // 阶段 9：树。父刚体是世界的刚体开一棵新树，其余刚体继承父刚体的树（parent < 自己的编号，所以一遍就够）。
  M.body_tree.assign(m->nbody, -1);
  for (int b = 1; b < m->nbody; ++b) {
    M.body_tree[b] = M.bodies[b].parent == 0 ? M.ntree++ : M.body_tree[M.bodies[b].parent];
  }

  // 自由度：每个属于哪个关节、哪个刚体，是转动还是平移。
  M.dof_joint.resize(m->nv);
  M.dof_body.resize(m->nv);
  M.dof_rot.resize(m->nv);
  M.dof_tree.resize(m->nv);
  for (int i = 0; i < m->njnt; ++i) {
    const Joint& j = M.joints[i];
    for (int k = 0; k < j.nv; ++k) {
      M.dof_joint[j.dadr + k] = i;
      M.dof_body[j.dadr + k] = j.body;
      M.dof_rot[j.dadr + k] = is_rotation(j, k);
      M.dof_tree[j.dadr + k] = M.body_tree[j.body];
    }
  }
  return M;
}

MjModelPtr load_mjcf(const std::string& path) {
  char err[1000] = "";
  mjModel* m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
  if (m == nullptr) throw std::runtime_error("cannot load " + path + ": " + err);
  return MjModelPtr(m);
}

}  // namespace rcipc
