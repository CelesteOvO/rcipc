// 阶段 4 的演示程序：地面接触。
//
// 用法（参数都可以省略，写成 名字=值）：
//   contact_demo [drop|flail|stack|ledge|slope] [h=..] [T=..] [dhat=..] [w=..] [kappa=..] [mu=..] [theta=..] [eps_v=..]
//                [ldlt=0|1] [semi=0|1] [adapt=0|1] [trace=0|1] [out=..]
//   阶段 9 的开关（stack、ledge、slope）：ldlt=1 用稠密 LDLT 代替 PCG；semi=1 打开半隐式提前停止；adapt=0 关掉 kappa 调整。
//
// drop（默认）：models/drop.xml，盒子从 0.5 m 静止落到地面，模拟 T 秒（默认 1.5）。
//               每步沿真实轨迹采样 64 点检查最小距离；打印最小间隙、静止间隙、kappa、Newton 次数、力平衡。
//               kappa 不给时由第一次接触的梯度平衡初始化；给了就固定用它（可以试试 kappa=2e9）。
//               CSV 每行：t, z, vz, min_distance, newton, kappa。
// flail：       models/flail.xml，杆以 w rad/s（默认 40）从地面上方 3 mm 处甩过正下方，一步，
//               直弦 CCD（阶段 4）和曲线 CCD（阶段 5）各走一遍。打印终点和真实弧线最低点的离地距离：
//               直弦看不见弧线穿过地面（书 §9.1 的反例），弦加矢高把这一步截在地面之前。
// stack：       阶段 7，models/stack.xml，转了 45 度的盒子落到另一个盒子上，模拟 T 秒（默认 1.5）。
// ledge：       阶段 7，models/ledge.xml，盒子落在世界上的静态台阶的边缘，翻下去掉到地面。
// slope：       阶段 8，models/slope.xml，斜面（坡度 tan theta，默认 0.3）上的盒子：mu 大于坡度时静止，否则匀加速下滑。
//
// 输出用英文，因为 Windows 控制台默认不是 UTF-8。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "rcipc/barrier.h"
#include "rcipc/ccd.h"
#include "rcipc/contact.h"
#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/step.h"

namespace {

using namespace rcipc;
using Eigen::VectorXd;

struct Args {
  std::string mode = "drop";
  double h = NAN, T = NAN, dhat = 1e-3, w = 40.0, kappa = 0;  // kappa = 0：由梯度平衡初始化
  double mu = 0.2, theta = 0.3, eps_v = 1e-3;  // 阶段 8：摩擦系数；斜面的坡度（tan theta）；摩擦光滑化的速度
  int trace = 0;
  int ldlt = 0, semi = 0, adapt = 1;  // 阶段 9：用稠密 LDLT 而不是 PCG；半隐式停止；kappa 调整
  std::string out;
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
    else if (k == "T") a.T = std::atof(v.c_str());
    else if (k == "dhat") a.dhat = std::atof(v.c_str());
    else if (k == "w") a.w = std::atof(v.c_str());
    else if (k == "kappa") a.kappa = std::atof(v.c_str());
    else if (k == "trace") a.trace = std::atoi(v.c_str());
    else if (k == "mu") a.mu = std::atof(v.c_str());
    else if (k == "theta") a.theta = std::atof(v.c_str());
    else if (k == "eps_v") a.eps_v = std::atof(v.c_str());
    else if (k == "ldlt") a.ldlt = std::atoi(v.c_str());
    else if (k == "semi") a.semi = std::atoi(v.c_str());
    else if (k == "adapt") a.adapt = std::atoi(v.c_str());
    else if (k == "out") a.out = v;
    else std::fprintf(stderr, "unknown option %s\n", k.c_str());
  }
  return a;
}

Model load(const char* file) { return from_mujoco(load_mjcf(std::string(RCIPC_MODEL_DIR "/") + file).get()); }

int drop(const Args& a) {
  const Model model = load("drop.xml");
  StepParams P;
  P.h = std::isnan(a.h) ? model.timestep : a.h;
  P.dhat = a.dhat;
  const double T = std::isnan(a.T) ? 1.5 : a.T;
  SolverState S;
  S.kappa = a.kappa;  // 0 表示第一次接触时按梯度平衡初始化

  VectorXd q = model.qpos0, v = VectorXd::Zero(6);
  double t = 0, dmin = 1e9, t_dmin = 0;
  int nmax = 0;
  long nsum = 0, nstep = 0;
  FILE* f = std::fopen((a.out.empty() ? "drop.csv" : a.out).c_str(), "w");
  if (f) std::fprintf(f, "t,z,vz,min_distance,newton,kappa\n");
  while (t < T - 1e-12) {
    const StepResult r = step(model, P, q, v, &S);
    const double d = trajectory_min_distance(model, q, differentiate(model, q, r.q), 1.0, 64);
    q = r.q;
    v = r.v;
    t += P.h;
    if (d < dmin) {
      dmin = d;
      t_dmin = t;
    }
    nmax = std::max(nmax, r.newton);
    nsum += r.newton;
    ++nstep;
    if (f) std::fprintf(f, "%.6f,%.17g,%.17g,%.17g,%d,%.6g\n", t, q[2], v[2], r.min_distance, r.newton, r.kappa);
  }
  if (f) std::fclose(f);

  // 静止时的力平衡：4 个角的障碍力之和应当等于重力。
  const Frames F = forward_kinematics(model, q);
  double force = 0, gap = 0;
  const std::vector<Contact> C = find_contacts(model, F, P.dhat);
  for (const Contact& c : C) {
    force += S.kappa * std::abs(barrier(c.d * c.d, P.dhat * P.dhat).db) * 2 * c.d;
    gap = c.d;
  }
  std::printf("box drop, h %g s, T %g s, dhat %g m\n", P.h, T, P.dhat);
  std::printf("min distance along all trajectories: %.4f mm at t = %.3f s%s\n", dmin * 1e3, t_dmin,
              dmin > 0 ? "" : "  <-- PENETRATION");
  std::printf("at rest: gap %.4f mm, |v| %.2e, %zu active contacts, barrier force %.4f N vs weight %.4f N\n",
              gap * 1e3, v.norm(), C.size(), force, model.bodies[1].mass * model.gravity.norm());
  std::printf("kappa %.4g J/m^4 (range %.3g .. %.3g)\n", S.kappa, kappa_range(model, P.dhat).min,
              kappa_range(model, P.dhat).max);
  std::printf("Newton iterations per step: max %d, mean %.2f\n", nmax, double(nsum) / nstep);
  if (f) std::printf("wrote %s\n", a.out.empty() ? "drop.csv" : a.out.c_str());
  return 0;
}

// 阶段 7 的场景：stack（两个盒子堆叠）、ledge（盒子落在台阶边缘）。每步沿真实轨迹采样检查最小距离；
// 打印最小距离、Newton 次数；Newton 跑满上限的步把每次迭代打出来（trace=1 时所有步都打）。
int scene(const Args& a, const char* file) {
  Model model = load(file);
  StepParams P;
  P.mu = a.mu;
  P.eps_v = a.eps_v;
  if (std::string(file) == "slope.xml") {
    // 斜面：转重力而不是转地面，盒子沿 +x 下滑（见 slope.xml 的说明）。
    const double th = std::atan(a.theta);
    model.gravity = 9.81 * Eigen::Vector3d(std::sin(th), 0, -std::cos(th));
    std::printf("slope tan(theta) = %g, mu = %g: %s; frictionless acceleration %.3f m/s^2, with friction %.3f m/s^2\n",
                a.theta, a.mu, a.theta < a.mu ? "should stick (creep below eps_v)" : "should slide",
                9.81 * std::sin(th), std::max(0.0, 9.81 * (std::sin(th) - a.mu * std::cos(th))));
  }
  P.h = std::isnan(a.h) ? model.timestep : a.h;
  P.dhat = a.dhat;
  const double T = std::isnan(a.T) ? 1.5 : a.T;
  SolverState S;
  S.kappa = a.kappa;

  VectorXd q = model.qpos0, v = VectorXd::Zero(model.nv);
  double t = 0, dmin = 1e9, t_dmin = 0;
  int nmax = 0, capped = 0, semi_stops = 0;
  long nsum = 0, nstep = 0, pcg_sum = 0;
  P.solver = a.ldlt ? LinearSolver::kLdlt : LinearSolver::kPcg;
  P.semi_implicit = a.semi != 0;
  P.adapt_kappa = a.adapt != 0;
  FILE* f = a.out.empty() ? nullptr : std::fopen(a.out.c_str(), "w");
  if (f) std::fprintf(f, "t,min_distance,newton,kappa\n");
  while (t < T - 1e-12) {
    std::vector<NewtonIterate> trace;
    const StepResult r = step(model, P, q, v, &S, &trace);
    const double d = trajectory_min_distance(model, q, differentiate(model, q, r.q), 1.0, 16);
    if (r.newton >= P.max_newton || a.trace) {
      const std::vector<Contact> Cend = find_contacts(model, forward_kinematics(model, r.q), P.dhat);
      std::printf("step %ld (t = %.3f): newton %d, min distance %.3e, %zu contacts:", nstep, t, r.newton,
                  r.min_distance, Cend.size());
      for (const Contact& c : Cend) std::printf(" [type %d d %.2e m %.3f]", int(c.type), c.d, c.moll);
      std::printf("\n");
      for (size_t k = 0; k < trace.size() && k < 12; ++k) {
        std::printf("    it %2zu: gap %.3e |p|/h %.3e alpha_max %.4f alpha %.4f E %.8e\n", k + 1,
                    trace[k].min_distance, trace[k].step, trace[k].alpha_max, trace[k].alpha, trace[k].energy);
      }
      if (r.newton >= P.max_newton) ++capped;
    }
    q = r.q;
    v = r.v;
    t += P.h;
    if (d < dmin) {
      dmin = d;
      t_dmin = t;
    }
    nmax = std::max(nmax, r.newton);
    nsum += r.newton;
    pcg_sum += r.pcg;
    semi_stops += r.semi_stopped;
    ++nstep;
    if (f) std::fprintf(f, "%.6f,%.17g,%d,%.6g\n", t, r.min_distance, r.newton, r.kappa);
  }
  if (f) std::fclose(f);
  std::printf("%s, h %g s, T %g s, dhat %g m, kappa %.4g\n", file, P.h, T, P.dhat, S.kappa);
  std::printf("min distance along all trajectories: %.4f mm at t = %.3f s%s\n", dmin * 1e3, t_dmin,
              dmin > 0 ? "" : "  <-- PENETRATION");
  std::printf("Newton iterations per step: max %d, mean %.2f, %d steps hit the cap; PCG iterations per step: mean %.2f\n",
              nmax, double(nsum) / nstep, capped, double(pcg_sum) / nstep);
  if (a.semi) std::printf("semi-implicit stop: %d of %ld steps stopped early\n", semi_stops, nstep);
  std::printf("final v:");
  for (int i = 0; i < model.nv; ++i) std::printf(" %.4f", v[i]);
  std::printf("\nfinal q:");
  for (int i = 0; i < model.nq; ++i) std::printf(" %.4f", q[i]);
  std::printf("\n");
  return 0;
}

int flail(const Args& a) {
  const Model model = load("flail.xml");
  StepParams P;
  P.h = std::isnan(a.h) ? model.timestep : a.h;
  P.dhat = a.dhat;
  const double H = 0.5075, R = 0.5, rad = 0.01;  // 铰链高度、球心的转动半径、球的半径（见 flail.xml）
  const double phi = std::acos((H - rad) / R);  // 杆端在地面以下的那段弧的半角

  VectorXd q(1), v(1);
  q[0] = 1.5707963267948966 - phi - 0.05;
  v[0] = a.w;
  const double d0 = min_distance(model, forward_kinematics(model, q));
  std::printf("flail, h %g s, w %g rad/s, dhat %g m: tip starts %+.3f mm above the floor\n", P.h, a.w, P.dhat,
              d0 * 1e3);
  std::printf("the arc dips below the floor for |angle - pi/2| < %.4f rad (lowest point %+.3f mm)\n", phi,
              (H - rad - R) * 1e3);

  // 同一步分别用直弦 CCD（阶段 4）和曲线 CCD（阶段 5）走一遍。
  for (CcdMode mode : {CcdMode::kChord, CcdMode::kCurved}) {
    P.ccd = mode;
    std::vector<NewtonIterate> trace;
    const StepResult r = step(model, P, q, v, nullptr, &trace);
    const VectorXd p = differentiate(model, q, r.q);
    const double dmin = trajectory_min_distance(model, q, p, 1.0, 256);
    std::printf("%s CCD: swept %.4f rad in %d Newton iterations, first alpha_max %.3f\n",
                mode == CcdMode::kChord ? "chord " : "curved", p[0], r.newton, trace.empty() ? 1.0 : trace[0].alpha_max);
    std::printf("  tip at the end %+.3f mm, lowest point of the true arc %+.3f mm%s\n", r.min_distance * 1e3, dmin * 1e3,
                dmin < 0 ? "  <-- PENETRATION" : "");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  try {
    if (a.mode == "drop") return drop(a);
    if (a.mode == "flail") return flail(a);
    if (a.mode == "stack") return scene(a, "stack.xml");
    if (a.mode == "ledge") return scene(a, "ledge.xml");
    if (a.mode == "slope") return scene(a, "slope.xml");
    std::fprintf(stderr, "unknown mode %s (drop, flail, stack, ledge or slope)\n", a.mode.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
