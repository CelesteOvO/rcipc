#pragma once

#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "rcipc/contact.h"
#include "rcipc/model.h"

// Newton 方程 H delta = -g 的求解（阶段 9，笔记 notes/09_solver.md，书 §10.3–10.4 第 219–224 页）。
//
// H = H_k + sum_r D_r r_r^T r_r：
//   H_k 是“树块”：惯性项 M / h^2 加上曲率上界的对角，只在同一棵树的自由度之间非零（阶段 3、6）；
//   每一行 r_r 是一个接触（障碍或摩擦），只在两侧刚体的链上非零（阶段 4、8）。
// 到现在为止代码把它们加成一个稠密矩阵再做 LDLT，代价 O(nv^3)。刚体一多就太慢；而 H 其实很稀疏：
// 树和树之间只靠少数几个接触行连接。这里改用预条件共轭梯度（PCG）：只需要 H 乘向量，不需要组装 H。
namespace rcipc {

// Newton 系统的两部分。
struct NewtonSystem {
  Eigen::MatrixXd Hk;      // 树块（nv x nv，但只有同树的项非零）
  std::vector<Row> rows;   // 接触行
};

// H p = H_k p + sum_r D_r r_r (r_r . p)：每行只要一次点积和一次 axpy。
Eigen::VectorXd apply(const NewtonSystem& sys, const Eigen::VectorXd& p);

// 组装成稠密矩阵（测试用，以及 LDLT 参照解）。
Eigen::MatrixXd dense(const NewtonSystem& sys);

// 树块预条件器（书 §10.4.1）：每棵树一块，P^(t) = H_k^(t) + sum_r D_r [r_r^T r_r]_tt，只保留两个自由度都在树 t 里的项；
// 跨树的项扔掉（块 Jacobi）。每块先做 Jacobi 缩放 S P S（S = diag(1 / sqrt(P_ii)）），加 1e-10 的对角保证正定，
// 再做 LDLT。应用时 z = P^-1 r 逐块解。树内部的耦合（质量矩阵）全在块里，所以单棵树、没有跨树接触时 P = H，PCG 一次就收敛。
struct TreePreconditioner {
  std::vector<std::vector<int>> dofs;             // 每棵树的自由度编号
  std::vector<Eigen::VectorXd> scale;             // 每块的 Jacobi 缩放
  std::vector<Eigen::LDLT<Eigen::MatrixXd>> ldlt; // 每块的分解
};
TreePreconditioner build_preconditioner(const Model& model, const NewtonSystem& sys);
Eigen::VectorXd apply(const TreePreconditioner& P, const Eigen::VectorXd& r);

// PCG（书 §10.3）。x0 = 0，r0 = rhs，z0 = P^-1 r0，p0 = z0；每次迭代
//   alpha = r^T z / p^T H p，x += alpha p，r -= alpha H p，z = P^-1 r，beta = r_new^T z_new / r^T z，p = z + beta p。
// 停止准则用 rz：|r^T z| <= tol |r_0^T z_0|（libuipc 的默认准则；rr 准则在硬接触主导残差时会过早停，书第 220 页）。
// p^T H p <= 0 或出现 NaN 时立即停（breakdown 保护，返回当前的 x）。
struct PcgResult {
  Eigen::VectorXd x;
  int iterations = 0;
  bool converged = false;
};
PcgResult pcg(const NewtonSystem& sys, const TreePreconditioner& P, const Eigen::VectorXd& rhs, double tol, int max_iter);

}  // namespace rcipc
