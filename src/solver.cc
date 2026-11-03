#include "rcipc/solver.h"

#include <cmath>

// Newton 方程的求解。推导见 notes/09_solver.md，测试见 tests/test_solver.cc。
namespace rcipc {

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd apply(const NewtonSystem& sys, const VectorXd& p) {
  // 树块乘向量，再逐行加 D r (r . p)。
  VectorXd y = sys.Hk * p;
  for (const Row& row : sys.rows) y += row.D * row.r.dot(p) * row.r.transpose();
  return y;
}

MatrixXd dense(const NewtonSystem& sys) {
  MatrixXd H = sys.Hk;
  for (const Row& row : sys.rows) H += row.D * row.r.transpose() * row.r;
  return H;
}

TreePreconditioner build_preconditioner(const Model& model, const NewtonSystem& sys) {
  TreePreconditioner P;
  P.dofs.assign(model.ntree, {});
  for (int d = 0; d < model.nv; ++d) P.dofs[model.dof_tree[d]].push_back(d);
  P.scale.resize(model.ntree);
  P.ldlt.resize(model.ntree);

  for (int t = 0; t < model.ntree; ++t) {
    const std::vector<int>& dofs = P.dofs[t];
    const int n = static_cast<int>(dofs.size());
    // 第 1 步：取出树块 H_k 里属于这棵树的子矩阵。
    MatrixXd B(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) B(i, j) = sys.Hk(dofs[i], dofs[j]);
    // 第 2 步：加上每一行在这棵树里的那一截的外积（跨树的项扔掉）。
    for (const Row& row : sys.rows) {
      VectorXd rt(n);
      bool any = false;
      for (int i = 0; i < n; ++i) {
        rt[i] = row.r[dofs[i]];
        any = any || rt[i] != 0;
      }
      if (any) B += row.D * rt * rt.transpose();
    }
    // 第 3 步：Jacobi 缩放 S B S，加 1e-10 的对角岭，LDLT。缩放让不同单位（rad 和 m）的自由度量级一致，分解更稳。
    VectorXd s(n);
    for (int i = 0; i < n; ++i) s[i] = B(i, i) > 0 ? 1.0 / std::sqrt(B(i, i)) : 1.0;
    const MatrixXd Bs = s.asDiagonal() * B * s.asDiagonal() + 1e-10 * MatrixXd::Identity(n, n);
    P.scale[t] = s;
    P.ldlt[t].compute(Bs);
  }
  return P;
}

VectorXd apply(const TreePreconditioner& P, const VectorXd& r) {
  // 逐块解：z_t = S (S B S)^-1 S r_t。
  VectorXd z = VectorXd::Zero(r.size());
  for (size_t t = 0; t < P.dofs.size(); ++t) {
    const std::vector<int>& dofs = P.dofs[t];
    const int n = static_cast<int>(dofs.size());
    VectorXd rt(n);
    for (int i = 0; i < n; ++i) rt[i] = P.scale[t][i] * r[dofs[i]];
    const VectorXd zt = P.ldlt[t].solve(rt);
    for (int i = 0; i < n; ++i) z[dofs[i]] = P.scale[t][i] * zt[i];
  }
  return z;
}

PcgResult pcg(const NewtonSystem& sys, const TreePreconditioner& P, const VectorXd& rhs, double tol, int max_iter) {
  // 标准的预条件共轭梯度（书第 220 页的公式）。
  PcgResult out;
  out.x = VectorXd::Zero(rhs.size());
  VectorXd r = rhs;
  VectorXd z = apply(P, r);
  VectorXd p = z;
  double rz = r.dot(z);
  const double rz0 = std::abs(rz);
  if (rz0 == 0) {  // 右端是 0：解就是 0
    out.converged = true;
    return out;
  }
  for (int it = 0; it < max_iter; ++it) {
    const VectorXd Hp = apply(sys, p);
    const double pHp = p.dot(Hp);
    if (!(pHp > 0) || !std::isfinite(pHp)) break;  // breakdown：H 不正定或出现 NaN，保留当前的 x
    const double alpha = rz / pHp;
    out.x += alpha * p;
    r -= alpha * Hp;
    z = apply(P, r);
    const double rz_new = r.dot(z);
    ++out.iterations;
    // rz 准则：|r^T z| 相对初值降到 tol。
    if (std::abs(rz_new) <= tol * rz0) {
      out.converged = true;
      break;
    }
    if (!std::isfinite(rz_new)) break;
    const double beta = rz_new / rz;
    p = z + beta * p;
    rz = rz_new;
  }
  return out;
}

}  // namespace rcipc
