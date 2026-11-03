// 阶段 3 的演示程序：关节坐标里的时间步，只有惯性和重力，没有接触。
//
// 用法（参数都可以省略，写成 名字=值）：
//   dynamics_demo [pendulum|spin] [h=..] [T=..] [out=..]
//
// pendulum（默认）：models/double_pendulum.xml，从两根杆都水平（q = 0）静止释放，模拟 T 秒（默认 3）。
//                   打印总能量的变化、Newton 次数，并和 MuJoCo 的 RK4（步长 h/10，当作“精确解”）比较摆角。
//                   CSV 每行：t, q1, q2, v1, v2, energy, newton。
// spin：            models/box.xml，关掉重力，让盒子绕中间那根主轴转（w 默认 5 rad/s，加一点扰动），模拟 T 秒（默认 5）。
//                   没有外力，动能应当守恒；打印动能的最大、最小倍数（书 §7.5 图 35 的实验）。
//                   CSV 每行：t, wx, wy, wz（刚体系角速度）, kinetic, newton。
//
// 输出用英文，因为 Windows 控制台默认不是 UTF-8。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/quat.h"
#include "rcipc/step.h"

namespace {

using namespace rcipc;
using Eigen::VectorXd;

struct Args {
  std::string mode = "pendulum";
  double h = NAN, T = NAN, w = 5.0;  // NaN 表示“用这个模式的默认值”
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
    else if (k == "w") a.w = std::atof(v.c_str());
    else if (k == "out") a.out = v;
    else std::fprintf(stderr, "unknown option %s\n", k.c_str());
  }
  return a;
}

int pendulum(const Args& a) {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/double_pendulum.xml");
  const Model model = from_mujoco(m.get());
  StepParams P;
  P.h = std::isnan(a.h) ? model.timestep : a.h;
  const double T = std::isnan(a.T) ? 3.0 : a.T;

  // 我们的时间步。
  VectorXd q = VectorXd::Zero(2), v = VectorXd::Zero(2);
  const double e_start = kinetic_energy(model, q, v) + potential_energy(model, q);
  double e_max = e_start, e_min = e_start, t = 0;
  int nmax = 0;
  long nsum = 0, nstep = 0;
  FILE* f = std::fopen((a.out.empty() ? "pendulum.csv" : a.out).c_str(), "w");
  if (f) std::fprintf(f, "t,q1,q2,v1,v2,energy,newton\n");
  while (t < T - 1e-12) {
    StepResult r = step(model, P, q, v);
    q = r.q;
    v = r.v;
    t += P.h;
    const double e = kinetic_energy(model, q, v) + potential_energy(model, q);
    e_max = std::max(e_max, e);
    e_min = std::min(e_min, e);
    nmax = std::max(nmax, r.newton);
    nsum += r.newton;
    ++nstep;
    if (f) std::fprintf(f, "%.6f,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n", t, q[0], q[1], v[0], v[1], e, r.newton);
  }
  if (f) std::fclose(f);

  // MuJoCo 的 RK4，步长 h/10，当作精确解。
  mjData* d = mj_makeData(m.get());
  m->opt.integrator = mjINT_RK4;
  m->opt.timestep = P.h / 10;
  while (d->time < T - 1e-12) mj_step(m.get(), d);
  const double dq1 = q[0] - d->qpos[0], dq2 = q[1] - d->qpos[1];
  mj_deleteData(d);

  const double e = kinetic_energy(model, q, v) + potential_energy(model, q);
  std::printf("double pendulum, h %g s, T %g s, released from rest with both links horizontal\n", P.h, T);
  std::printf("total energy: start %.6f J, end %.6f J, max %+.3e J above start, min %+.3e J\n", e_start, e,
              e_max - e_start, e_min - e_start);
  std::printf("angles at T: ours (%.4f, %.4f) rad, MuJoCo RK4 h/10 differs by (%+.4f, %+.4f) rad\n", q[0], q[1],
              dq1, dq2);
  std::printf("Newton iterations per step: max %d, mean %.2f\n", nmax, double(nsum) / nstep);
  if (f) std::printf("wrote %s\n", a.out.empty() ? "pendulum.csv" : a.out.c_str());
  return 0;
}

int spin(const Args& a) {
  MjModelPtr m = load_mjcf(RCIPC_MODEL_DIR "/box.xml");
  Model model = from_mujoco(m.get());
  model.gravity.setZero();
  StepParams P;
  P.h = std::isnan(a.h) ? 0.01 : a.h;  // 书 §7.5 的实验用 dt = 0.01
  const double T = std::isnan(a.T) ? 5.0 : a.T;

  // 自由关节的速度：前 3 个是世界系线速度，后 3 个是刚体系角速度。
  // 盒子的三根主轴就是刚体坐标轴，y 轴（长 0.2 的那条边）是中间那根。绕它转，再加一点绕 x 的扰动。
  VectorXd q = model.qpos0, v = VectorXd::Zero(6);
  v[3] = 0.01 * a.w;
  v[4] = a.w;
  const double k0 = kinetic_energy(model, q, v);
  double kmax = k0, kmin = k0, t = 0;
  int nmax = 0;
  long nsum = 0, nstep = 0;
  FILE* f = std::fopen((a.out.empty() ? "spin.csv" : a.out).c_str(), "w");
  if (f) std::fprintf(f, "t,wx,wy,wz,kinetic,newton\n");
  while (t < T - 1e-12) {
    StepResult r = step(model, P, q, v);
    q = r.q;
    v = r.v;
    t += P.h;
    const double k = kinetic_energy(model, q, v);
    kmax = std::max(kmax, k);
    kmin = std::min(kmin, k);
    nmax = std::max(nmax, r.newton);
    nsum += r.newton;
    ++nstep;
    if (f) std::fprintf(f, "%.6f,%.17g,%.17g,%.17g,%.17g,%d\n", t, v[3], v[4], v[5], k, r.newton);
  }
  if (f) std::fclose(f);
  std::printf("spinning box, no gravity, h %g s, T %g s, |w| %g rad/s about the intermediate axis\n", P.h, T, a.w);
  std::printf("kinetic energy / initial: max %.6f, min %.6f, end %.6f\n", kmax / k0, kmin / k0,
              kinetic_energy(model, q, v) / k0);
  std::printf("Newton iterations per step: max %d, mean %.2f\n", nmax, double(nsum) / nstep);
  if (f) std::printf("wrote %s\n", a.out.empty() ? "spin.csv" : a.out.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  try {
    if (a.mode == "pendulum") return pendulum(a);
    if (a.mode == "spin") return spin(a);
    std::fprintf(stderr, "unknown mode %s (pendulum or spin)\n", a.mode.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
