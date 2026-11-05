#include "rcipc/drive.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "rcipc/friction.h"

// 驱动、关节阻尼和 armature（阶段 12），关节的干摩擦（G1 之后）。推导见 notes/12_drives.md 第 1–3、5 节；
// 测试见 tests/test_drives.cc。
namespace rcipc {

using Eigen::VectorXd;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

bool scalar_dof(const Model& model, int d) {
  const JointType t = model.joints[model.dof_joint[d]].type;
  return t == JointType::kHinge || t == JointType::kSlide;
}

// 驱动 k 在 q 处的 y = A + C q（夹之前的驱动力）。
double force_of(const DriveTerms::Act& a, const Model& model, const VectorXd& q) {
  const Joint& j = model.joints[model.dof_joint[a.dof]];
  return a.A + a.C * q[j.qadr];
}

// 干摩擦在自由度 d 上这一步的速度 v = (q - c) / h。
double friction_speed(const DriveTerms& D, const Model& model, const VectorXd& q, int d) {
  return (q[model.joints[model.dof_joint[d]].qadr] - D.c[d]) / D.h;
}

// f1 的导数：f1 = 2 y / eps - y^2 / eps^2（y < eps），所以 f1' = 2 (eps - y) / eps^2；y >= eps 时是 0。
double friction_f1_slope(double y, double eps) { return y < eps ? 2 * (eps - y) / (eps * eps) : 0.0; }

}  // namespace

DriveTerms drive_terms(const Model& model, const VectorXd& q0, const VectorXd& v0, const VectorXd& ctrl, double h,
                       double eps_v) {
  DriveTerms D;
  D.K = VectorXd::Zero(model.nv);
  D.r = VectorXd::Zero(model.nv);
  D.F = VectorXd::Zero(model.nv);
  D.c = VectorXd::Zero(model.nv);
  D.h = h;
  D.eps_v = eps_v;
  for (int d = 0; d < model.nv; ++d) {
    if (!scalar_dof(model, d)) continue;
    if (model.dof_frictionloss[d] > 0) {
      D.F[d] = model.dof_frictionloss[d];
      D.c[d] = q0[model.joints[model.dof_joint[d]].qadr];
    }
    const double b = model.dof_damping[d], a = model.dof_armature[d];
    if (b == 0 && a == 0) continue;
    // K/2 (q - r)^2 = b/(2h) (q - q0)^2 + a/(2h^2) (q - (q0 + h v0))^2 + 常数：K 是两个系数之和，r 是两个中心的加权平均。
    const double qd = q0[model.joints[model.dof_joint[d]].qadr];
    const double kb = b / h, ka = a / (h * h);
    D.K[d] = kb + ka;
    D.r[d] = (kb * qd + ka * (qd + h * v0[d])) / D.K[d];
  }
  for (int i = 0; i < model.nu(); ++i) {
    const Actuator& act = model.actuators[i];
    double u = ctrl.size() == model.nu() ? ctrl[i] : 0.0;
    if (act.ctrl_limited) u = std::min(std::max(u, act.ctrl_range[0]), act.ctrl_range[1]);
    DriveTerms::Act a;
    a.dof = act.dof;
    a.s = act.gear;
    a.w = act.bias[1] + act.bias[2] / h;
    a.C = a.s * a.w;
    a.A = act.gain * u + act.bias[0] - act.bias[2] * a.s * q0[act.qadr] / h;
    a.lo = act.force_limited ? act.force_range[0] : -kInf;
    a.hi = act.force_limited ? act.force_range[1] : kInf;
    D.act.push_back(a);
  }
  return D;
}

double drive_energy(const Model& model, const DriveTerms& D, const VectorXd& q) {
  double e = 0;
  for (int d = 0; d < model.nv; ++d) {
    if (D.F[d] != 0) e += D.h * D.F[d] * friction_f0(std::abs(friction_speed(D, model, q, d)), D.eps_v);
    if (D.K[d] == 0) continue;
    const double x = q[model.joints[model.dof_joint[d]].qadr] - D.r[d];
    e += 0.5 * D.K[d] * x * x;
  }
  for (const DriveTerms::Act& a : D.act) {
    const double qd = q[model.joints[model.dof_joint[a.dof]].qadr];
    if (a.C == 0) {
      // 常力（motor，或者偏置里没有 L、dL/dt 的项）：E = -tau q。
      e -= a.s * std::min(std::max(a.A, a.lo), a.hi) * qd;
      continue;
    }
    // E = -(1/w) Psi(y)，y = A + C q，Psi(y) = c^2/2 + c (y - c)，c = clamp(y)：Psi' = clamp(y)，所以 E' = -s clamp(y)。
    const double y = a.A + a.C * qd;
    const double c = std::min(std::max(y, a.lo), a.hi);
    e -= (0.5 * c * c + c * (y - c)) / a.w;
  }
  return e;
}

VectorXd drive_gradient(const Model& model, const DriveTerms& D, const VectorXd& q) {
  VectorXd g = VectorXd::Zero(model.nv);
  for (int d = 0; d < model.nv; ++d) {
    if (D.K[d] != 0) g[d] = D.K[d] * (q[model.joints[model.dof_joint[d]].qadr] - D.r[d]);
    if (D.F[d] != 0) {
      // d/dq [h F f0(|v|)] = F f1(|v|) sign(v)，dv/dq = 1/h。
      const double v = friction_speed(D, model, q, d);
      g[d] += D.F[d] * friction_f1(std::abs(v), D.eps_v) * (v < 0 ? -1.0 : 1.0);
    }
  }
  const VectorXd tau = actuator_torques(model, D, q);
  for (size_t i = 0; i < D.act.size(); ++i) g[D.act[i].dof] -= tau[static_cast<int>(i)];
  return g;
}

VectorXd drive_hessian_diagonal(const Model& model, const DriveTerms& D, const VectorXd& q) {
  VectorXd H = D.K;
  for (int d = 0; d < model.nv; ++d) {
    if (D.F[d] != 0) H[d] += D.F[d] * friction_f1_slope(std::abs(friction_speed(D, model, q, d)), D.eps_v) / D.h;
  }
  for (const DriveTerms::Act& a : D.act) {
    if (a.C == 0) continue;
    const double y = force_of(a, model, q);
    if (y > a.lo && y < a.hi) H[a.dof] -= a.s * a.C;  // -s C = -s^2 w >= 0；范围外力是常数，Hessian 为 0
  }
  return H;
}

VectorXd actuator_torques(const Model& model, const DriveTerms& D, const VectorXd& q) {
  VectorXd tau(static_cast<int>(D.act.size()));
  for (size_t i = 0; i < D.act.size(); ++i) {
    const DriveTerms::Act& a = D.act[i];
    tau[static_cast<int>(i)] = a.s * std::min(std::max(force_of(a, model, q), a.lo), a.hi);
  }
  return tau;
}

VectorXd ctrl_at(const Model& model, double t) {
  const std::vector<CtrlKey>& K = model.ctrl_keys;
  if (K.empty()) return VectorXd::Zero(model.nu());
  if (t <= K.front().time) return K.front().ctrl;
  if (t >= K.back().time) return K.back().ctrl;
  size_t k = 1;
  while (K[k].time < t) ++k;
  const double t0 = K[k - 1].time, t1 = K[k].time;
  const double w = t1 > t0 ? (t - t0) / (t1 - t0) : 1.0;
  return (1 - w) * K[k - 1].ctrl + w * K[k].ctrl;
}

}  // namespace rcipc
