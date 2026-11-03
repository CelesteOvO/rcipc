#include "rcipc/ccd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "rcipc/contact.h"

// CCD。直弦版本的推导见 notes/04_ground.md 第 5 节，曲线版本见 notes/05_curved_ccd.md；测试见 tests/test_ccd.cc。
namespace rcipc {

using Eigen::VectorXd;

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

VectorXd acceleration_bounds(const Model& model, const VectorXd& q, const VectorXd& p) {
  // 静态加速度界（笔记第 2 节，书 §9.3.1）。
  //
  // 推导的骨架：刚体上一点 x 的速度是链上各关节的贡献之和
  //   x' = sum_j w_j a_j x (x - o_j) + sum_s u_s a_s          （转动组 j：轴 a_j、锚点 o_j；滑动 s：轴 a_s）
  // 再求一次导数，每一项分两部分：
  //   - 轴在转：a_j、a_s 被更靠近根的关节带着转，|a'| <= W_{<j}；转动项的力臂 |x - o_j| <= R_j；
  //   - 点相对锚点在动：|(x - o_j)'| 只来自 j 和它下游的关节，<= sum_{k>=j} (w_k R_k + u_k)。
  // 合起来 |x''| <= sum_j w_j (W_{<j} R_j + sum_{k>=j} (w_k R_k + u_k)) + sum_s W_{<s} u_s。
  // R_j 用静态力臂：锚点到关节所在刚体原点的距离 |J.pos|，加上链上每段 body_pos 的长度（滑动关节再加上
  // 它的伸长 |q - ref| 和这一步最多再伸 |p|），加上刚体半径。这些都是固定长度的线段首尾相接，
  // 三角不等式保证在任何姿态下 |x - o_j| 都不超过它们之和。
  VectorXd A = VectorXd::Zero(model.nbody());
  std::vector<double> w, u, R;  // 链上的关节，从根到刚体的顺序：转动组的 w、滑动的 u、静态力臂 R
  for (int b = 1; b < model.nbody(); ++b) {
    // 第 1 步：沿链往上走，收集每个关节的 w、u 和静态力臂 R。先按“从刚体到根”的顺序收集，之后反过来。
    w.clear();
    u.clear();
    R.clear();
    double path = model.bodies[b].radius;  // 从当前走到的刚体原点到 b 的顶点的路径长度
    for (int k = b; k != 0; k = model.bodies[k].parent) {
      const int j = model.bodies[k].joint;
      if (j >= 0) {
        const Joint& J = model.joints[j];
        if (J.type == JointType::kSlide) {
          // 滑动：u = |p|，力臂还要加上它现在和这一步内的伸长。
          const double us = std::abs(p[J.dadr]);
          w.push_back(0.0);
          u.push_back(us);
          R.push_back(path + J.pos.norm());
          path += std::abs(q[J.qadr] - J.ref) + us;
        } else {
          // 转动组：铰链 1 个自由度（dadr），球关节 3 个（dadr..），自由关节后 3 个（dadr+3..）。
          double wj = 0;
          for (int d = J.dadr; d < J.dadr + J.nv; ++d) {
            if (model.dof_rot[d]) wj += p[d] * p[d];
          }
          w.push_back(std::sqrt(wj));
          u.push_back(0.0);
          R.push_back(path + J.pos.norm());  // 锚点到 k 的原点，再沿路径到 b 的顶点
        }
      }
      path += model.bodies[k].pos.norm();  // 往上一段：k 的原点到父刚体原点的固定偏移
    }
    std::reverse(w.begin(), w.end());
    std::reverse(u.begin(), u.end());
    std::reverse(R.begin(), R.end());

    // 第 2 步：套公式。W_below 是更靠近根的各转动组的 w 之和。
    const int n = static_cast<int>(w.size());
    double a = 0, W_below = 0;
    for (int j = 0; j < n; ++j) {
      double s = W_below * R[j];
      for (int k = j; k < n; ++k) s += w[k] * R[k] + u[k];
      a += w[j] * s + W_below * u[j];
      W_below += w[j];
    }
    A[b] = a;
  }
  return A;
}

double curved_ccd_ground(const Model& model, const VectorXd& q, const VectorXd& p, double dhat, double c) {
  // 弦加矢高（笔记第 3、4 节，书 §9.2、§9.5）。
  const VectorXd A = acceleration_bounds(model, q, p);
  const double A_max = A.size() > 0 ? A.maxCoeff() : 0.0;

  // 第 1 步：段长。最大矢高 A_max ds^2 / 8 不超过 dhat，所以 ds <= sqrt(8 dhat / A_max)。没有转动时整段一次检查。
  double ds = A_max > 0 ? std::min(1.0, std::sqrt(8 * dhat / A_max)) : 1.0;

  // 第 2 步：逐段推进。每段先在两端做 FK，然后对每个 (顶点, 平面) 对做带余量的保守推进。
  double s0 = 0;
  Frames Fa = forward_kinematics(model, q);
  for (int guard = 0; guard < 10000 && s0 < 1.0; ++guard) {
    const double s1 = std::min(1.0, s0 + ds);
    const double len = s1 - s0;
    const Frames Fb = forward_kinematics(model, integrate(model, q, p, s1));

    double t_min = 1.0;     // 这一段里最早被截断的位置（段内比例）
    double ds_need = len;   // 有顶点要求更短的段时，缩短到多少
    for (int v = 0; v < model.nvertex(); ++v) {
      const int b = model.vertices[v].body;
      const double xi = A[b] * len * len / 8;  // 这个顶点在这一段里偏离弦的上界（矢高）
      for (int pl = 0; pl < static_cast<int>(model.planes.size()); ++pl) {
        const double ya = ground_distance(model, Fa, v, pl);
        const double yb = ground_distance(model, Fb, v, pl);
        // 段对这个顶点太长（余量吃掉了一半以上的距离）：要求 A len^2 / 8 <= ya / 2，即 len <= sqrt(4 ya / A)。
        if (xi > 0.5 * ya) {
          ds_need = std::min(ds_need, 0.9 * std::sqrt(4 * ya / A[b]));
          continue;
        }
        // 远离平面：弦上距离不减少，整段安全（真实距离 >= yb - xi >= ya - xi > 0）。
        if (yb >= ya) continue;
        // 靠近平面：沿弦“距离 - 矢高”线性减少，最多闭合 c 倍：t = c (ya - xi) / (ya - yb)。
        t_min = std::min(t_min, c * (ya - xi) / (ya - yb));
      }
    }

    if (ds_need < len) {
      // 第 3 步：有顶点嫌这段太长，缩短后重做这一段（两端的 FK 里 Fa 不变，Fb 要重算）。
      ds = std::min(0.5 * len, ds_need);
      continue;
    }
    if (t_min < 1.0) return s0 + t_min * len;  // 这一段里被截断：之前的段都安全，答案就在这里
    s0 = s1;
    Fa = Fb;
  }
  return std::min(s0, 1.0);
}

double trajectory_min_distance(const Model& model, const VectorXd& q, const VectorXd& p, double alpha, int samples) {
  double dmin = std::numeric_limits<double>::infinity();
  for (int k = 0; k <= samples; ++k) {
    const double s = alpha * k / samples;
    dmin = std::min(dmin, min_ground_distance(model, forward_kinematics(model, integrate(model, q, p, s))));
  }
  return dmin;
}

}  // namespace rcipc
