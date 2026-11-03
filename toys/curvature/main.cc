// 阶段 6 的探索程序：曲率上界。同一个时间步，分别用 curv_scale = 0（纯 Gauss-Newton）和给定的 scale 走，
// 逐次迭代打印最小间隙、CCD 上限和线搜索步长，看间隙有没有被一次次挖小。
//
// 用法：curvature_demo [flail|box] [h=..] [w=..] [scale=..] [steps=..] [dhat=..]
//   flail（默认）：甩杆先以 w rad/s 撞上地面（第一步），然后看第二步（杆贴着地面还想往前转）的 Newton 过程。
//   box：          盒子一角着地、带着角速度往前倒。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/step.h"

namespace {

using namespace rcipc;
using Eigen::VectorXd;

struct Args {
  std::string mode = "flail";
  double h = 0.01, w = 40.0, scale = 1.0, dhat = 1e-3;
  int steps = 3;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto eq = s.find('=');
    if (eq == std::string::npos) {
      a.mode = s;
      continue;
    }
    std::string k = s.substr(0, eq), v = s.substr(eq + 1);
    if (k == "h") a.h = std::atof(v.c_str());
    else if (k == "w") a.w = std::atof(v.c_str());
    else if (k == "scale") a.scale = std::atof(v.c_str());
    else if (k == "steps") a.steps = std::atoi(v.c_str());
    else if (k == "dhat") a.dhat = std::atof(v.c_str());
    else std::fprintf(stderr, "unknown option %s\n", k.c_str());
  }
  return a;
}

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

void run(const Model& model, StepParams P, VectorXd q, VectorXd v, int steps, double scale) {
  P.curv_scale = scale;
  P.adapt_kappa = false;  // 阶段 6 的实验只看 Newton 模型，不让 kappa 在迭代之间变
  SolverState S;
  std::printf("curv_scale %g\n", scale);
  for (int i = 0; i < steps; ++i) {
    std::vector<NewtonIterate> trace;
    const StepResult r = step(model, P, q, v, &S, &trace);
    const double dmin = trajectory_min_distance(model, q, differentiate(model, q, r.q), 1.0, 64);
    std::printf("  step %d: newton %d, end gap %.4e m, trajectory min %.4e m, kappa %.3g\n", i + 1, r.newton,
                r.min_distance, dmin, r.kappa);
    for (size_t k = 0; k < trace.size(); ++k) {
      std::printf("    it %2zu: gap %.4e  |p|/h %.3e  alpha_max %.4f  alpha %.4f  E %.6e\n", k + 1,
                  trace[k].min_distance, trace[k].step, trace[k].alpha_max, trace[k].alpha, trace[k].energy);
    }
    q = r.q;
    v = r.v;
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  try {
    StepParams P;
    P.h = a.h;
    P.dhat = a.dhat;
    if (a.mode == "flail") {
      const Model model = load("flail.xml");
      VectorXd q(1), v(1);
      q[0] = 1.5707963267948966 - std::acos(0.4975 / 0.5) - 0.05;
      v[0] = a.w;
      run(model, P, q, v, a.steps, 0.0);
      run(model, P, q, v, a.steps, a.scale);
      return 0;
    }
    if (a.mode == "box") {
      // 盒子绕 x 轴转 atan(0.5)，让角 (0, -0.1, -0.2) 正好在质心正下方（距离 sqrt(0.1^2 + 0.2^2) = 0.2236），
      // 离地 dhat/2。再绕 x 轴以 w rad/s 继续倒：角的速度是水平的，沿着地面；弧线却往地面弯。
      const Model model = load("drop.xml");
      const double ang = std::atan(0.5), rc = std::sqrt(0.1 * 0.1 + 0.2 * 0.2);
      VectorXd q = model.qpos0, v = VectorXd::Zero(6);
      q[2] = rc + 0.5 * a.dhat;
      q[3] = std::cos(0.5 * ang);
      q[4] = std::sin(0.5 * ang);
      q[5] = q[6] = 0;
      v[3] = a.w;
      run(model, P, q, v, a.steps, 0.0);
      run(model, P, q, v, a.steps, a.scale);
      return 0;
    }
    if (a.mode == "lever") {
      // 顶杆（lever.xml）：球在弧线最高点、离地 dhat/2，杆以 w rad/s 开始转。
      const Model model = load("lever.xml");
      VectorXd q = VectorXd::Zero(1), v = VectorXd::Zero(1);
      v[0] = a.w;
      run(model, P, q, v, a.steps, 0.0);
      run(model, P, q, v, a.steps, a.scale);
      return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", a.mode.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
