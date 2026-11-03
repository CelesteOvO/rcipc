#include "rcipc/limits.h"

#include <algorithm>
#include <limits>

#include "rcipc/barrier.h"

// 关节限位。推导见 notes/10_scenes.md 第 1 节，测试见 tests/test_scenes.cc。
namespace rcipc {

using Eigen::VectorXd;

namespace {

// 对每个有限位的关节，给出它的自由度编号和到两端的距离，调用 visit(dof, d_lo, d_hi)。
template <class Visit>
void for_each_limit(const Model& model, const VectorXd& q, Visit visit) {
  for (const Joint& J : model.joints) {
    if (!J.limited) continue;
    visit(J.dadr, q[J.qadr] - J.range[0], J.range[1] - q[J.qadr]);
  }
}

}  // namespace

double limit_energy(const Model& model, const VectorXd& q, double k, double dhat) {
  double e = 0;
  for_each_limit(model, q, [&](int, double dlo, double dhi) { e += k * (barrier(dlo, dhat).b + barrier(dhi, dhat).b); });
  return e;
}

VectorXd limit_gradient(const Model& model, const VectorXd& q, double k, double dhat) {
  // d_lo 随 q 增大（导数 +1），d_hi 随 q 减小（导数 -1）。
  VectorXd g = VectorXd::Zero(model.nv);
  for_each_limit(model, q, [&](int d, double dlo, double dhi) { g[d] += k * (barrier(dlo, dhat).db - barrier(dhi, dhat).db); });
  return g;
}

VectorXd limit_hessian_diagonal(const Model& model, const VectorXd& q, double k, double dhat) {
  // 一维的精确二阶导数 b''，两端都是正的（阶段 1 思考题 4：一维里 E'' 永远是正的）。
  VectorXd h = VectorXd::Zero(model.nv);
  for_each_limit(model, q, [&](int d, double dlo, double dhi) { h[d] += k * (barrier(dlo, dhat).ddb + barrier(dhi, dhat).ddb); });
  return h;
}

double limit_ccd(const Model& model, const VectorXd& q, const VectorXd& p, double c) {
  // 和阶段 1 的 ccd_max_step 完全一样：往下端走时 s <= c d_lo / (-p)，往上端走时 s <= c d_hi / p。
  double alpha = 1.0;
  for_each_limit(model, q, [&](int d, double dlo, double dhi) {
    if (p[d] < 0) alpha = std::min(alpha, c * dlo / -p[d]);
    if (p[d] > 0) alpha = std::min(alpha, c * dhi / p[d]);
  });
  return alpha;
}

double min_limit_distance(const Model& model, const VectorXd& q) {
  double dmin = std::numeric_limits<double>::infinity();
  for_each_limit(model, q, [&](int, double dlo, double dhi) { dmin = std::min({dmin, dlo, dhi}); });
  return dmin;
}

}  // namespace rcipc
