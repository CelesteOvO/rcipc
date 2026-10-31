#include "rcipc/model.h"

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
  }

  // 刚体：父刚体、固定偏移、关节编号。顺便检查我们不支持的结构。
  M.bodies.resize(m->nbody);
  for (int b = 0; b < m->nbody; ++b) {
    Body& B = M.bodies[b];
    B.parent = b == 0 ? -1 : m->body_parentid[b];
    B.pos = Eigen::Vector3d(m->body_pos + 3 * b);
    B.quat = quat_load(m->body_quat + 4 * b);
    B.joint = m->body_jntnum[b] > 0 ? m->body_jntadr[b] : -1;
    const std::string name = mj_id2name(m, mjOBJ_BODY, b) ? mj_id2name(m, mjOBJ_BODY, b) : std::to_string(b);
    if (m->body_jntnum[b] > 1) throw std::runtime_error("body " + name + " has more than one joint");
    if (m->body_mocapid[b] >= 0) throw std::runtime_error("body " + name + " is a mocap body");
    if (B.joint >= 0 && M.joints[B.joint].type == JointType::kFree && B.parent != 0) {
      throw std::runtime_error("body " + name + ": free joint below the top level");
    }
  }

  // 自由度：每个属于哪个关节、哪个刚体，是转动还是平移。
  M.dof_joint.resize(m->nv);
  M.dof_body.resize(m->nv);
  M.dof_rot.resize(m->nv);
  for (int i = 0; i < m->njnt; ++i) {
    const Joint& j = M.joints[i];
    for (int k = 0; k < j.nv; ++k) {
      M.dof_joint[j.dadr + k] = i;
      M.dof_body[j.dadr + k] = j.body;
      M.dof_rot[j.dadr + k] = is_rotation(j, k);
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
