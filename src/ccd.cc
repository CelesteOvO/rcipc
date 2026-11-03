#include "rcipc/ccd.h"

#include <algorithm>
#include <limits>

#include "rcipc/contact.h"

// CCD。推导见 notes/04_ground.md 第 5 节，测试见 tests/test_ccd.cc。
namespace rcipc {

double chord_ccd_ground(const Model& model, const Frames& a, const Frames& b, double c) {
  // 对每个 (顶点, 平面) 对单独推进，取最小（书 §9.4：逐对，不用全局最小距离）。
  double alpha = 1.0;
  for (int v = 0; v < model.nvertex(); ++v) {
    for (int p = 0; p < static_cast<int>(model.planes.size()); ++p) {
      const double ya = ground_distance(model, a, v, p);
      const double yb = ground_distance(model, b, v, p);
      // 情况 1：远离平面（或不动），整步安全。
      if (yb >= ya) continue;
      // 情况 2：靠近平面。沿弦距离线性减少，最多闭合 c * ya：s <= c ya / (ya - yb)。
      //   这个值可能大于 1（整步也闭合不到 c * ya），那就不限制。和阶段 1 的 ccd_max_step 完全一样。
      alpha = std::min(alpha, c * ya / (ya - yb));
    }
  }
  return alpha;
}

double trajectory_min_distance(const Model& model, const Eigen::VectorXd& q, const Eigen::VectorXd& p, double alpha,
                               int samples) {
  double dmin = std::numeric_limits<double>::infinity();
  for (int k = 0; k <= samples; ++k) {
    const double s = alpha * k / samples;
    dmin = std::min(dmin, min_ground_distance(model, forward_kinematics(model, integrate(model, q, p, s))));
  }
  return dmin;
}

}  // namespace rcipc
