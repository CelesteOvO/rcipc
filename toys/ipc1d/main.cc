// 阶段 1 的演示程序：书 §3.8 的小球落地，以及书 §3.5.3 的撞击那一步。
//
// 用法（参数都可以省略，写成 名字=值）：
//   ipc1d_demo [drop|impact] [kappa=1e4] [h=0.002] [dhat=1e-3] [c=0.9] [tol=1e-6] [x0=..] [v0=..]
//              [T=1.5] [out=ipc1d.csv]
//
// drop（默认）：从间隙 x0（默认 0.45 m）、速度 v0（默认 0）开始模拟 T 秒。打印最小间隙、最终间隙和
//               Newton 次数，并把每一步的 t、x、v、newton 写进 CSV（用 tools/plot_ipc1d.py 画图）。
// impact：      只算一个时间步，逐次打印 Newton 迭代（默认 x0 = 3 mm，v0 = -3 m/s，和书上的表一样）。
//
// 输出用英文，因为 Windows 控制台默认不是 UTF-8，中文会显示成乱码。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "ipc1d/ipc1d.h"

namespace {

using namespace rcipc::ipc1d;

// 命令行参数。
struct Args {
  std::string mode = "drop";
  Params P;
  double x0 = NAN, v0 = NAN, T = 1.5;  // x0、v0 是 NaN 表示“用这个模式的默认值”
  std::string out = "ipc1d.csv";
};

// 不带 = 的参数是模式（drop 或 impact），带 = 的是 名字=值。
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
    if (k == "kappa") a.P.kappa = std::atof(v.c_str());
    else if (k == "h") a.P.h = std::atof(v.c_str());
    else if (k == "dhat") a.P.dhat = std::atof(v.c_str());
    else if (k == "c") a.P.c = std::atof(v.c_str());
    else if (k == "tol") a.P.tol = std::atof(v.c_str());
    else if (k == "x0") a.x0 = std::atof(v.c_str());
    else if (k == "v0") a.v0 = std::atof(v.c_str());
    else if (k == "T") a.T = std::atof(v.c_str());
    else if (k == "out") a.out = v;
    else std::fprintf(stderr, "unknown option %s\n", k.c_str());
  }
  return a;
}

// drop 模式：一步接一步地调用 step()，记录最小间隙和 Newton 次数。
int drop(const Args& a) {
  const Params& P = a.P;
  double x = std::isnan(a.x0) ? 0.45 : a.x0, v = std::isnan(a.v0) ? 0.0 : a.v0, t = 0;
  double xmin = x, tmin = 0;
  int nmax = 0;
  long nsum = 0, nstep = 0;
  FILE* f = std::fopen(a.out.c_str(), "w");
  if (f) std::fprintf(f, "t,x,v,newton\n");
  while (t < a.T - 1e-12) {
    StepResult r = step(P, x, v);  // 上一步的结果就是这一步的起点
    x = r.x;
    v = r.v;
    t += P.h;
    if (x < xmin) {
      xmin = x;
      tmin = t;
    }
    nmax = std::max(nmax, r.newton);
    nsum += r.newton;
    ++nstep;
    if (f) std::fprintf(f, "%.6f,%.17g,%.17g,%d\n", t, x, v, r.newton);
  }
  if (f) std::fclose(f);
  std::printf("kappa %g N/m, h %g s, dhat %g m\n", P.kappa, P.h, P.dhat);
  std::printf("min gap %.4f mm at t = %.3f s\n", xmin * 1e3, tmin);
  std::printf("final gap %.4f mm, final v %+.2e m/s\n", x * 1e3, v);
  std::printf("Newton iterations per step: max %d, mean %.2f\n", nmax, double(nsum) / nstep);
  std::printf("book with the defaults: min 0.0078 mm, final 0.5359 mm, Newton max 9, mean 1.44\n");
  if (f) std::printf("wrote %s\n", a.out.c_str());
  return 0;
}

// impact 模式：只算一个时间步，用 trace 拿到每次迭代，打印成和书上一样的表。
int impact(const Args& a) {
  double x0 = std::isnan(a.x0) ? 3e-3 : a.x0, v0 = std::isnan(a.v0) ? -3.0 : a.v0;
  std::vector<Iterate> trace;
  StepResult r = step(a.P, x0, v0, &trace);
  std::printf("x0 %.4f mm, v0 %+.3f m/s, x~ %.4f mm\n", x0 * 1e3, v0, predicted(a.P, x0, v0) * 1e3);
  std::printf("it   gap (mm)       p (mm)   alpha_max   alpha   new gap (mm)\n");
  for (size_t i = 0; i < trace.size(); ++i) {
    const Iterate& it = trace[i];
    std::printf("%2zu  %9.4f  %+11.4e  %9.4f  %7.4f  %9.4f\n", i + 1, it.x * 1e3, it.p * 1e3, it.alpha_max,
                it.alpha, (it.x + it.alpha * it.p) * 1e3);
  }
  std::printf("%d Newton iterations (max %d): gap %.4f mm, v %+.4f m/s\n", r.newton, a.P.max_newton,
              r.x * 1e3, r.v);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  try {
    if (a.mode == "drop") return drop(a);
    if (a.mode == "impact") return impact(a);
    std::fprintf(stderr, "unknown mode %s (drop or impact)\n", a.mode.c_str());
    return 2;
  } catch (const std::exception& e) {  // 还没实现的函数会抛出 "not implemented: ..."
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
