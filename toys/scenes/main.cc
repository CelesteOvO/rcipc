// 阶段 10 的场景程序：跑任意一个 MJCF 场景，每步检查穿模，写出轨迹。
//
// 用法（参数都可以省略，写成 名字=值）：
//   scene_demo model=ragdoll.xml [T=2] [h=..] [dhat=..] [mu=..] [eps_v=..] [tol=..] [semi=0|1] [adapt=0|1] [ldlt=0|1]
//              [check=0|1] [debug=0|1|2] [gpu=0|1] [timing=0|1] [pcg_tol=..] [icurv=0|1|2] [icurv_floor=..] [pd=0|1]
//              [out=traj.csv]
//   gpu=1：用 GPU 版（阶段 11，需要编译时打开了 CUDA）。check 在 GPU 上不支持。
//   pd=0：关掉原始-对偶障碍（StepParams::primal_dual，notes/22_primal_dual.md），退回原始障碍，只用来对比。
//   icurv=0：惯性 Hessian 只用 M / h^2（阶段 13 之前的做法，notes/16_inertia_curvature.md）；icurv=2 连转动那一部分也加。
//   fullh=1：完整的接触 Hessian（每个接触 12 x 12 的旋量块做半正定投影，阶段 14，notes/17_contact_hessian.md）。
//     GPU 版要和 pd=0 一起用。
//   timing=0：GPU 版不做分段计时（分段计时每段都要等 GPU，会拖慢整步；量墙钟时间时关掉）。
//
//   model：models/ 下的文件名（或绝对路径）。有的场景：ragdoll.xml、ragdoll_pile.xml、chain.xml、stack.xml、ledge.xml，
//          阶段 12 之后：gears.xml（用 mu=0 gpu=1）、gripper.xml（用 mu=0.5）、cylinders.xml，
//          以及 tools/make_robot_pile.py 生成的 models/local/go2_pile.xml；穿透会改变拓扑的 ring_chain.xml、blender.xml
//          （tools/make_stress.py，notes/14_visible.md）；轻链吊重物的 wrecking_ball.xml（tools/make_wrecking_ball.py，
//          notes/15_drift.md）。
//   模型有关键帧时，从第一个关键帧的 qpos 开始（没有时从 qpos0）。
//   check=1：每次接受的 Newton 迭代都沿真实轨迹采样最小距离（慢，但验证的是 CCD 真正保证的东西）。
//   debug=1：第一次穿模的那一步打印每次迭代；debug=2 另外打印每个撞上 Newton 上限的步（间隙、|dx|/h、CCD 和线搜索的步长）；debug=3 每一步都打印。
//   out：每行 t, min_distance, min_limit, newton, pcg, 然后是 nq 个 qpos。用 tools/replay.py 在 MuJoCo 的窗口里回放。
//
// 阶段 12：有驱动的场景，每一步的控制量按 MJCF 的关键帧插值（ctrl_at，取这一步结束的时刻）。
//   模型里读到了但不支持而被忽略的设置（Model::ignored）在开头打印出来。
//
// 每 0.1 s 打印一行：时间、步末最小距离、最近的限位距离、Newton 和 PCG 的次数、每步耗时。
// 结束时打印整段的最小距离（步末、迭代轨迹）、限位、Newton 统计、总耗时。输出用英文（Windows 控制台）。
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "rcipc/contact.h"
#include "rcipc/drive.h"
#include "rcipc/kinematics.h"
#include "rcipc/limits.h"
#include "rcipc/model.h"
#include "rcipc/step.h"
#ifdef RCIPC_HAVE_CUDA
#include "rcipc/gpu.h"
#endif

namespace {

using namespace rcipc;
using Eigen::VectorXd;

struct Args {
  std::string model = "ragdoll.xml", out;
  double T = 2.0, h = NAN, dhat = 1e-3, mu = 0.2, eps_v = 1e-3, tol = 1e-6;
  int semi = 0, ldlt = 0, check = 0, debug = 0, adapt = 1, gpu = 0, timing = 1, graph = 1, tight = 1, reuse = 1, tail = 1, prefilter = 1;
  int icurv = 1, relax = 1, fullh = 0;  // fullh=1：完整的接触 Hessian（StepParams::full_contact_hessian，阶段 14）
  int pd = 1;  // pd=0：原始障碍（StepParams::primal_dual = false）
  double kappa = 0;  // > 0：kappa 的预设值（跳过第一次接触时的梯度平衡初始化）
  double icurv_floor = NAN, curv = NAN;  // curv：StepParams::curv_scale（1 可证明的上界，0 纯 Gauss-Newton）
  double pcg_tol = NAN;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto eq = s.find('=');
    if (eq == std::string::npos) {
      a.model = s;
      continue;
    }
    std::string k = s.substr(0, eq), v = s.substr(eq + 1);
    if (k == "model") a.model = v;
    else if (k == "T") a.T = std::atof(v.c_str());
    else if (k == "h") a.h = std::atof(v.c_str());
    else if (k == "dhat") a.dhat = std::atof(v.c_str());
    else if (k == "mu") a.mu = std::atof(v.c_str());
    else if (k == "eps_v") a.eps_v = std::atof(v.c_str());
    else if (k == "tol") a.tol = std::atof(v.c_str());
    else if (k == "adapt") a.adapt = std::atoi(v.c_str());
    else if (k == "relax") a.relax = std::atoi(v.c_str());
    else if (k == "fullh") a.fullh = std::atoi(v.c_str());
    else if (k == "pd") a.pd = std::atoi(v.c_str());
    else if (k == "kappa") a.kappa = std::atof(v.c_str());
    else if (k == "gpu") a.gpu = std::atoi(v.c_str());
    else if (k == "timing") a.timing = std::atoi(v.c_str());
    else if (k == "graph") a.graph = std::atoi(v.c_str());
    else if (k == "tight") a.tight = std::atoi(v.c_str());
    else if (k == "reuse") a.reuse = std::atoi(v.c_str());
    else if (k == "tail") a.tail = std::atoi(v.c_str());
    else if (k == "prefilter") a.prefilter = std::atoi(v.c_str());
    else if (k == "pcg_tol") a.pcg_tol = std::atof(v.c_str());
    else if (k == "semi") a.semi = std::atoi(v.c_str());
    else if (k == "ldlt") a.ldlt = std::atoi(v.c_str());
    else if (k == "check") a.check = std::atoi(v.c_str());
    else if (k == "debug") a.debug = std::atoi(v.c_str());
    else if (k == "icurv") a.icurv = std::atoi(v.c_str());
    else if (k == "icurv_floor") a.icurv_floor = std::atof(v.c_str());
    else if (k == "curv") a.curv = std::atof(v.c_str());
    else if (k == "out") a.out = v;
    else std::fprintf(stderr, "unknown option %s\n", k.c_str());
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  try {
    const std::string path = a.model.find(':') != std::string::npos || a.model[0] == '/' ? a.model
                                                                                            : std::string(RCIPC_MODEL_DIR "/") + a.model;
    const Model model = from_mujoco(load_mjcf(path).get());
    StepParams P;
    P.h = std::isnan(a.h) ? model.timestep : a.h;
    P.dhat = a.dhat;
    P.mu = a.mu;
    P.eps_v = a.eps_v;
    P.tol = a.tol;
    P.adapt_kappa = a.adapt != 0;
    P.kappa_relax = a.relax != 0;
    P.full_contact_hessian = a.fullh != 0;
    P.primal_dual = a.pd != 0;
    P.semi_implicit = a.semi != 0;
    P.inertia_curvature = a.icurv != 0;
    P.inertia_curvature_rot = a.icurv == 2;
    if (!std::isnan(a.icurv_floor)) P.inertia_curvature_floor = a.icurv_floor;
    if (!std::isnan(a.curv)) P.curv_scale = a.curv;
    P.solver = a.ldlt ? LinearSolver::kLdlt : LinearSolver::kPcg;
    P.check_iterates = a.check != 0;
    P.tight_bound = a.tight != 0;
    if (!std::isnan(a.pcg_tol)) P.pcg_tol = a.pcg_tol;
    SolverState S;
    if (a.kappa > 0) S.kappa = a.kappa;
#ifdef RCIPC_HAVE_CUDA
    std::unique_ptr<gpu::Stepper> G;
    if (a.gpu) {
      if (!gpu::available()) throw std::runtime_error("gpu=1 but no CUDA device");
      G = std::make_unique<gpu::Stepper>(model);
      G->set_timing(a.timing != 0);
      G->set_graphs(a.graph != 0);
      G->set_ccd_reuse(a.reuse != 0);
      G->set_ccd_tail(a.tail != 0);
      G->set_ccd_prefilter(a.prefilter != 0);
    }
#else
    if (a.gpu) throw std::runtime_error("gpu=1 but this build has no CUDA");
#endif

    std::printf("%s: %d bodies, %d trees, nv %d, %d vertices, %d edges, %d faces; h %g, T %g, dhat %g, mu %g, eps_v %g, "
                "tol %g m/s, %s, %s contact Hessian%s%s%s\n",
                a.model.c_str(), model.nbody() - 1, model.ntree, model.nv, model.nvertex(), model.nedge(), model.nface(),
                P.h, a.T, P.dhat, P.mu, P.eps_v, P.tol, a.gpu ? "GPU" : (a.ldlt ? "LDLT" : "PCG"),
                P.full_contact_hessian ? "full" : "Gauss-Newton", P.primal_dual ? "" : ", primal barrier (pd=0)",
                a.semi ? ", semi-implicit stop" : "",
                model.filter_parent ? "" : ", parent-child contact on");
    if (model.nu() > 0) std::printf("%d actuators, %zu control keyframes\n", model.nu(), model.ctrl_keys.size());
    for (const std::string& s : model.ignored) std::printf("ignored (not supported): %s\n", s.c_str());

    VectorXd q = model.key_qpos.size() == model.nq ? model.key_qpos : model.qpos0, v = VectorXd::Zero(model.nv);
    FILE* f = a.out.empty() ? nullptr : std::fopen(a.out.c_str(), "w");
    if (f) std::fprintf(f, "t,min_distance,min_limit,newton,pcg,qpos...\n");
    double t = 0, dmin_end = 1e300, dmin_iter = 1e300, lmin = 1e300, next_report = 0;
    int nmax = 0, capped = 0;
    long nsum = 0, pcg_sum = 0, nstep = 0, last_step = 0, dropped = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    while (t < a.T - 1e-12) {
      std::vector<NewtonIterate> trace;
      if (model.nu() > 0) P.ctrl = ctrl_at(model, t + P.h);
#ifdef RCIPC_HAVE_CUDA
      const StepResult r = G ? G->step(P, q, v, &S, a.debug ? &trace : nullptr)
                             : step(model, P, q, v, &S, a.debug ? &trace : nullptr);
#else
      const StepResult r = step(model, P, q, v, &S, a.debug ? &trace : nullptr);
#endif
      if (a.debug && r.min_distance <= 0 && dmin_end > 0) {
        // 第一次出现穿模的那一步：打印最近的那一对和每次迭代的间隙、步长，帮助定位是 CCD 漏了还是别的。
        const Frames Fe = forward_kinematics(model, r.q);
        const Contact* worst = nullptr;
        std::vector<Contact> all = find_contacts(model, Fe, 1e300);
        for (const Contact& c : all) if (worst == nullptr || c.d < worst->d) worst = &c;
        std::printf("first penetration at step %ld (t = %.4f): type %d bodies %d %d d %.3e verts %d %d %d %d\n", nstep, t + P.h,
                    int(worst->type), worst->bodyA, worst->bodyB, worst->d, worst->verts[0], worst->verts[1], worst->verts[2],
                    worst->verts[3]);
        for (size_t k = 0; k < trace.size(); ++k)
          std::printf("  it %2zu: gap %.3e |dx|/h %.3e alpha_max %.4f alpha %.4f pcg %d\n", k + 1, trace[k].min_distance,
                      trace[k].step, trace[k].alpha_max, trace[k].alpha, trace[k].pcg);
      }
      if (a.debug >= 3) {
        // debug=3：每一步都打印每次迭代（看静止时 Newton 为什么要走好几步）。
        std::printf("step %ld (t = %.4f): newton %d\n", nstep + 1, t + P.h, r.newton);
        for (size_t k = 0; k < trace.size(); ++k)
          std::printf("  it %2zu: gap %.3e |dx|/h %.3e alpha_max %.4f alpha %.4f pcg %d E %.12e\n", k + 1, trace[k].min_distance,
                      trace[k].step, trace[k].alpha_max, trace[k].alpha, trace[k].pcg, trace[k].energy);
      }
      if (a.debug >= 2 && r.capped) {
        // debug=2：撞上 Newton 上限的步，打印每次迭代的间隙、|dx|/h（收敛判据）、CCD 允许的最大步长和线搜索取的步长。
        std::printf("step %ld (t = %.4f) hit the Newton cap:\n", nstep + 1, t + P.h);
        for (size_t k = 0; k < trace.size(); ++k)
          std::printf("  it %2zu: gap %.3e |dx|/h %.3e alpha_max %.4f alpha %.4f pcg %d\n", k + 1, trace[k].min_distance,
                      trace[k].step, trace[k].alpha_max, trace[k].alpha, trace[k].pcg);
      }
      q = r.q;
      v = r.v;
      t += P.h;
      dmin_end = std::min(dmin_end, r.min_distance);
      dmin_iter = std::min(dmin_iter, r.iterate_min_distance);
      lmin = std::min(lmin, r.min_limit_distance);
      nmax = std::max(nmax, r.newton);
      capped += r.capped;
      nsum += r.newton;
      pcg_sum += r.pcg;
      dropped += r.curvature_dropped;
      ++nstep;
      if (f) {
        std::fprintf(f, "%.6f,%.9g,%.9g,%d,%d", t, r.min_distance, r.min_limit_distance, r.newton, r.pcg);
        for (int i = 0; i < model.nq; ++i) std::fprintf(f, ",%.17g", q[i]);
        std::fprintf(f, "\n");
      }
      if (t >= next_report - 1e-12) {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - last).count();
        std::printf("t %5.2f  gap %9.3e  limit %9.3e  newton %2d  pcg %3d  kappa %.2e  %6.1f ms/step%s\n", t,
                    r.min_distance, r.min_limit_distance, r.newton, r.pcg, r.kappa, ms / std::max(1L, nstep - last_step),
                    r.min_distance <= 0 ? "  <-- PENETRATION" : "");
        last = now;
        last_step = nstep;
        next_report += 0.1;
      }
    }
    if (f) std::fclose(f);
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("done: %ld steps in %.1f s (%.1f ms/step)\n", nstep, total, 1e3 * total / nstep);
    std::printf("min distance at step ends %.3e m%s", dmin_end, dmin_end > 0 ? "" : "  <-- PENETRATION");
    if (a.check) std::printf(", along accepted iterates %.3e m%s", dmin_iter, dmin_iter > 0 ? "" : "  <-- PENETRATION");
    std::printf("\nmin joint-limit distance %.3e; Newton per step max %d mean %.2f (%d hit the cap); PCG per step mean %.2f\n",
                lmin, nmax, double(nsum) / nstep, capped, double(pcg_sum) / nstep);
    if (P.inertia_curvature && !a.gpu)
      std::printf("inertia curvature left out %ld times (tree x iteration, not positive definite)\n", dropped);
#ifdef RCIPC_HAVE_CUDA
    if (G) {
      // GPU 各部分的耗时（每步平均，毫秒；timing=0 时没有分段计时）。
      const gpu::Stepper::Timers& T = G->timers();
      const double n = static_cast<double>(T.steps);
      if (a.timing) std::printf("GPU per step (ms): prepare %.2f, update %.2f, assemble %.2f, solve %.2f, converge %.2f, ccd %.2f, "
                  "line search %.2f, graph %.2f, total %.2f; energy evaluations per step %.2f\n",
                  T.prepare / n, T.update / n, T.assemble / n, T.solve / n, T.converge / n, T.ccd / n, T.line_search / n,
                  T.graph / n, T.total / n, T.energy_evals / n);
      const double nn = std::max(1.0, static_cast<double>(T.newton));
      std::printf("GPU per Newton iteration: %.0f contacts, %.0f rows, %.1f PCG iterations, %.2f CCD segments (%.2f culls); "
                  "%.0f body pairs per query\n",
                  T.contacts / nn, T.rows / nn, T.pcg / nn, T.ccd_segments / nn, T.ccd_windows / nn,
                  static_cast<double>(T.pair_sum) / std::max(1L, T.pair_calls));
      if (T.ccd_tail > 0)
        std::printf("ACCD tail: %ld pairs handed over (%.2f per CCD segment), %ld proved to return 1, %ld finished serially\n",
                    T.ccd_tail, T.ccd_tail / std::max(1.0, static_cast<double>(T.ccd_segments)), T.ccd_tail_proved,
                    T.ccd_tail_serial);
      if (P.inertia_curvature)
        std::printf("inertia curvature left out %ld times on the GPU (tree x iteration, not positive definite)\n",
                    T.curv_dropped);
      if (T.ccd_prefiltered > 0)
        std::printf("ACCD prefilter: %.0f pairs per CCD segment skipped in single precision\n",
                    T.ccd_prefiltered / std::max(1.0, static_cast<double>(T.ccd_segments)));
    }
#endif
    if (f) std::printf("wrote %s (replay: python tools/replay.py %s %s)\n", a.out.c_str(), path.c_str(), a.out.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
