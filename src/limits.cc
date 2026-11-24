#include "rcipc/limits.h"

#include <algorithm>
#include <limits>

#include "rcipc/barrier.h"

// 关节限位。推导见 notes/10_scenes.md 第 1 节，测试见 tests/test_scenes.cc。
namespace rcipc {

using Eigen::VectorXd;

namespace {

// 对每个有限位的关节，给出它的自由度编号、到两端的距离和它的类型，调用 visit(dof, d_lo, d_hi, type)。
template <class Visit>
void for_each_limit(const Model& model, const VectorXd& q, Visit visit) {
  for (const Joint& J : model.joints) {
    if (!J.limited) continue;
    visit(J.dadr, q[J.qadr] - J.range[0], J.range[1] - q[J.qadr], J.type);
  }
}

}  // namespace

double limit_energy(const Model& model, const VectorXd& q, const LimitBarrier& L) {
  double e = 0;
  for_each_limit(model, q, [&](int, double dlo, double dhi, JointType t) {
    const double dh = L.dhat(t);
    e += L.k * (barrier(dlo, dh).b + barrier(dhi, dh).b);
  });
  return e;
}

VectorXd limit_gradient(const Model& model, const VectorXd& q, const LimitBarrier& L) {
  // d_lo 随 q 增大（导数 +1），d_hi 随 q 减小（导数 -1）。
  VectorXd g = VectorXd::Zero(model.nv);
  for_each_limit(model, q, [&](int d, double dlo, double dhi, JointType t) {
    const double dh = L.dhat(t);
    g[d] += L.k * (barrier(dlo, dh).db - barrier(dhi, dh).db);
  });
  return g;
}

VectorXd limit_hessian_diagonal(const Model& model, const VectorXd& q, const LimitBarrier& L) {
  // 一维的精确二阶导数 b''，两端都是正的（阶段 1 思考题 4：一维里 E'' 永远是正的）。
  VectorXd h = VectorXd::Zero(model.nv);
  for_each_limit(model, q, [&](int d, double dlo, double dhi, JointType t) {
    const double dh = L.dhat(t);
    h[d] += L.k * (barrier(dlo, dh).ddb + barrier(dhi, dh).ddb);
  });
  return h;
}

bool limits_active(const Model& model, const VectorXd& q, const LimitBarrier& L) {
  bool active = false;
  for_each_limit(model, q, [&](int, double dlo, double dhi, JointType t) { active |= std::min(dlo, dhi) < L.dhat(t); });
  return active;
}

double limit_ccd(const Model& model, const VectorXd& q, const VectorXd& p, double c, double sep) {
  // 和阶段 1 的 ccd_max_step 一样：往下端走时 s <= c (d_lo - sep) / (-p)，往上端走时 s <= c (d_hi - sep) / p。
  // 距离已经 <= sep 时右边 <= 0，夹到 0（原来会得到负的步长）。
  double alpha = 1.0;
  for_each_limit(model, q, [&](int d, double dlo, double dhi, JointType) {
    if (p[d] < 0) alpha = std::min(alpha, std::max(0.0, c * (dlo - sep) / -p[d]));
    if (p[d] > 0) alpha = std::min(alpha, std::max(0.0, c * (dhi - sep) / p[d]));
  });
  return alpha;
}

double min_limit_distance(const Model& model, const VectorXd& q) {
  double dmin = std::numeric_limits<double>::infinity();
  for_each_limit(model, q, [&](int, double dlo, double dhi, JointType) { dmin = std::min({dmin, dlo, dhi}); });
  return dmin;
}

}  // namespace rcipc
