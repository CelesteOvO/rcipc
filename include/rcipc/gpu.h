#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "rcipc/kinematics.h"
#include "rcipc/model.h"
#include "rcipc/step.h"

// GPU 版的时间步（阶段 11，笔记 notes/11_gpu.md）。
//
// 算法和 CPU 版的 step() 完全一样（同一个 Newton 循环、同样的能量、行、预条件器、CCD、线搜索、kappa 规则），
// 只是每一项都换成在显存里并行算。所以测试可以拿 CPU 版当参照，逐项对比（tests/test_gpu.cc）。
//
// 和 CPU 版的差别：
//   - 原子加的顺序不固定，求和的舍入每次不一样，所以 GPU 和 CPU（以及 GPU 两次运行之间）只在舍入的量级上一致，
//     混沌的场景跑久了轨迹会分开；
//   - StepParams::solver 被忽略（总是 PCG），check_iterates 不支持（iterate_min_distance 返回 +无穷）；
//   - StepResult::min_distance 只看 d̂ 以内的接触：没有接触时返回 +无穷（CPU 版返回真实的最小距离）。
//     kappa 规则只关心 d̂ 以内的距离，所以 kappa 的行为和 CPU 版一样。
namespace rcipc::gpu {

// 有没有能用的 CUDA 设备（编译时打开了 CUDA，并且运行时找到了显卡）。
bool available();

class Stepper {
 public:
  explicit Stepper(const Model& model);  // 把模型上传到显存
  ~Stepper();
  Stepper(const Stepper&) = delete;
  Stepper& operator=(const Stepper&) = delete;

  // 和 rcipc::step 一样的接口。
  StepResult step(const StepParams& P, const Eigen::VectorXd& q0, const Eigen::VectorXd& v0, SolverState* state = nullptr,
                  std::vector<NewtonIterate>* trace = nullptr);

  // 各部分的累计耗时（毫秒，每段结束时同步一次再读时钟），剖析用。几段互不重叠，加起来约等于 total。
  //   prepare：步首（倒推、两次 FK、目标、滞后摩擦）；update：新位置上的 FK 和接触检测；assemble：组装 g、树块、行；
  //   solve：预条件器和 PCG；converge：收敛判断；ccd：CCD；line_search：线搜索（每次试探都做 FK、检测、能量）。
  //   graph：CUDA graph 版的 Newton 迭代（组装到接受试探点整个在一张图里，分不开，见 set_graphs）。
  struct Timers {
    double prepare = 0, update = 0, assemble = 0, solve = 0, converge = 0, ccd = 0, line_search = 0, total = 0;
    double graph = 0;
    long newton = 0, pcg = 0, energy_evals = 0, steps = 0;
    long contacts = 0, rows = 0;         // 每次组装时的接触数、行数（接触 + 2 x 摩擦）之和
    long pair_calls = 0, pair_sum = 0;   // 候选刚体对：求了几次、个数之和（检测和 CCD 都算）
    long ccd_segments = 0;               // CCD 一共检查了几段（缩短重做的也算）
  };
  const Timers& timers() const;
  void reset_timers();
  // 分段计时要在每段结束时等 GPU 做完（cudaDeviceSynchronize），这本身会让 GPU 空等。默认开着；
  // 量整步的墙钟时间时关掉（关掉以后 Timers 里只有计数是对的）。
  void set_timing(bool on);
  // PCG 用协作内核（整个循环一个内核，默认，设备支持时）还是主机循环（每次迭代发几个内核、读回点积）。
  // 两者算法一样。返回实际用的是不是协作内核。
  bool set_cooperative_pcg(bool on);
  // Newton 迭代用 CUDA graph（默认）还是逐个发射。图版把一个迭代录成一张图，CCD 的分段和线搜索的回溯在设备上
  // 循环，每个迭代只回主机一次（笔记 notes/11_gpu.md 第 12 节）。算法一样，结果只差舍入。
  // 要协作 PCG；传了 trace（逐次迭代的记录）时总是逐个发射。
  void set_graphs(bool on);
  // 上一步是不是用 CUDA graph 跑的（set_graphs 打开、设备支持、录图成功时）。
  bool used_graphs() const;
  // 测试用：把刚体对、接触、候选、工作项的容量改小，让“放不下、扩容、重来”的路径也走到。
  void set_capacities(int pairs, int contacts, int candidates, int work);

  // ---------- 测试用：在给定的点上算各项，下载回来和 CPU 版对比 ----------

  // 正运动学。
  Frames fk(const Eigen::VectorXd& q);
  // 步首用 (q0, v0) 准备惯性目标和滞后摩擦（kappa 是摩擦用的），然后在 q 处组装 Newton 系统：
  //   energy = 惯性 + kappa 障碍 + 摩擦 + 限位，g 是梯度，H 是稠密的 Hessian（树块 + 曲率上界 + 所有行），
  //   ncontact 是 d̂ 以内的接触个数，nfriction 是摩擦接触个数。
  struct Terms {
    double energy = 0;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    int ncontact = 0, nfriction = 0;
  };
  Terms terms(const StepParams& P, const Eigen::VectorXd& q0, const Eigen::VectorXd& v0, double kappa,
              const Eigen::VectorXd& q);
  // 用最近一次 terms() 组装的系统，PCG 解 H x = rhs（容差 tol），返回解和迭代次数。
  Eigen::VectorXd solve(const Eigen::VectorXd& rhs, double tol, int* iterations = nullptr);
  // 用最近一次 terms() 组装的系统建树块预条件器，返回稠密的块对角 P^-1（nv x nv）。
  Eigen::MatrixXd precond_inverse();
  // CCD：curved_ccd 和 limit_ccd 的较小值。
  double ccd(const StepParams& P, const Eigen::VectorXd& q, const Eigen::VectorXd& p);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rcipc::gpu
