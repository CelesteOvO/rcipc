// GPU 版的完整接触 Hessian（阶段 14，笔记 notes/17_contact_hessian.md）。对应 CPU 版 contact.cc 里的
// squared_distance_hessian、contact_wrench_hessian、contact_rows_full，以及 contact_chain_curvature 里每个接触的那一份。
//
// 一个线程一个接触：
//   1. 顶点空间的梯度 f（3 nst）和 Hessian H_X（3 nst x 3 nst），同 contact.cc 的 vertex_derivatives；
//   2. 投到两侧刚体的旋量上（参考点是各刚体的原点 xpos）：H12 = J^T H_X J 加顶点随转动走弧线的几何项；
//   3. 12 x 12 的循环 Jacobi 特征分解，正的特征值各写一行（rows 的第 12 i + k 行；用不到的行两侧刚体都是 0，D = 0）；
//   4. 梯度按两侧的力旋量加到 g；链上剩余项要的每刚体量 (G, F, P) 原子加到 CB（nbody x 21），
//      k_inertia_curvature（kin.cu）再把它们和惯性的那一份一起算、一起做正定检查。
// 能量和 k_contact_terms 算的一样，线搜索仍然用 k_contact_terms。
#include "devmath.cuh"
#include "kernels.h"
#include "rows.cuh"

namespace rcipc::gpu {

namespace {

constexpr int kThreads = 64;  // 每个线程的本地数组大一些（几个 12 x 12），块小一点
inline int blocks(int n) { return (n + kThreads - 1) / kThreads; }

// 循环 Jacobi（n <= 12）：A（n x n，按行存）出来时对角是特征值，V 的列是特征向量。
__device__ void sym_eig(int n, double* A, double* V, double* w) {
  for (int i = 0; i < n * n; ++i) V[i] = 0;
  for (int i = 0; i < n; ++i) V[i * n + i] = 1;
  for (int sweep = 0; sweep < 40; ++sweep) {
    double off = 0, dg = 0;
    for (int p = 0; p < n; ++p) {
      dg += A[p * n + p] * A[p * n + p];
      for (int q = p + 1; q < n; ++q) off += A[p * n + q] * A[p * n + q];
    }
    if (off <= 1e-30 * dg) break;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = A[p * n + q];
        if (apq == 0) continue;
        const double theta = 0.5 * (A[q * n + q] - A[p * n + p]) / apq;
        const double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1));
        const double c = 1 / sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < n; ++k) {  // A <- A P
          const double akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {  // A <- P^T A
          const double apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {  // V <- V P
          const double vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int i = 0; i < n; ++i) w[i] = A[i * n + i];
}

// 平方距离 y = |Delta|^2 对模板顶点的二阶导数（contact.cc 的 squared_distance_hessian）。H 是 N x N，N = 3 nst，步长 N。
__device__ void sq_dist_hessian(const DContact& c, const V3* X, double* H) {
  const int N = 3 * c.nst;
  if (c.type == kGround) {
    const double nn[3] = {c.n[0], c.n[1], c.n[2]};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) H[3 * i + j] = 2 * nn[i] * nn[j];
    return;
  }
  for (int v = 0; v < c.nst; ++v)
    for (int u = 0; u < c.nst; ++u)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) H[(3 * v + a) * N + 3 * u + b] = a == b ? 2 * c.w[v] * c.w[u] : 0.0;
  V3 Delta = v3(0, 0, 0);
  for (int v = 0; v < c.nst; ++v) Delta = Delta + c.w[v] * X[v];
  double dw[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  int np = 0;
  if (c.type == kPointTriangle) {
    int active[3], na = 0;
    for (int i = 0; i < 3; ++i)
      if (-c.w[1 + i] > 0) active[na++] = i;
    for (int k = 1; k < na; ++k) {
      dw[np][1 + active[0]] = 1;
      dw[np][1 + active[k]] = -1;
      ++np;
    }
  } else {
    if (c.w[1] > 0 && c.w[1] < 1) {
      dw[np][0] = -1;
      dw[np][1] = 1;
      ++np;
    }
    if (-c.w[3] > 0 && -c.w[3] < 1) {
      dw[np][2] = 1;
      dw[np][3] = -1;
      ++np;
    }
  }
  if (np == 0) return;
  V3 E[2];
  double G[2 * 12];  // 第 k 个参数的 F_{X lam_k}（N 个）
  for (int k = 0; k < np; ++k) {
    E[k] = v3(0, 0, 0);
    for (int v = 0; v < c.nst; ++v) E[k] = E[k] + dw[k][v] * X[v];
    for (int v = 0; v < c.nst; ++v) v3store(2 * c.w[v] * E[k] + 2 * dw[k][v] * Delta, G + 12 * k + 3 * v);
  }
  double Fll[4], V[4], w[2], Finv[4] = {0, 0, 0, 0};
  for (int k = 0; k < np; ++k)
    for (int l = 0; l < np; ++l) Fll[k * np + l] = 2 * dot(E[k], E[l]);
  sym_eig(np, Fll, V, w);
  double lmax = 0;
  for (int k = 0; k < np; ++k) lmax = fmax(lmax, w[k]);
  for (int k = 0; k < np; ++k) {
    if (!(w[k] > 1e-10 * lmax)) continue;  // 伪逆：退化的方向不动（两条边平行、边长为 0）
    for (int a = 0; a < np; ++a)
      for (int b = 0; b < np; ++b) Finv[a * np + b] += V[a * np + k] * V[b * np + k] / w[k];
  }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0;
      for (int k = 0; k < np; ++k)
        for (int l = 0; l < np; ++l) s += G[12 * k + i] * Finv[k * np + l] * G[12 * l + j];
      H[i * N + j] -= s;
    }
}

// contact.cc 的 vertex_derivatives：能量 kappa m b 对模板顶点的梯度 f（N）和 Hessian H（N x N，步长 N），kappa 已乘上。
// 返回能量。
__device__ double vertex_derivatives(const DContact& c, const V3* X, double radii, double kappa, double dhat, double* f,
                                     double* H) {
  const int N = 3 * c.nst;
  const double ell = c.d + radii;
  const V3 n = v3load(c.n);
  double gy[12], Hy[144];
  for (int i = 0; i < c.nst; ++i) v3store((2 * ell * c.w[i]) * n, gy + 3 * i);
  sq_dist_hessian(c, X, Hy);
  const double alpha = c.d / ell;
  const BV bv = barrier(c.d * c.d, dhat * dhat);
  double gb[12], Hb[144];
  const double cg = bv.ddb * alpha * alpha + bv.db * radii / (2 * ell * ell * ell);
  for (int i = 0; i < N; ++i) gb[i] = bv.db * alpha * gy[i];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) Hb[i * N + j] = cg * gy[i] * gy[j] + bv.db * alpha * Hy[i * N + j];
  if (c.type != kEdgeEdge || (c.moll == 1 && c.dmoll == 0)) {
    for (int i = 0; i < N; ++i) f[i] = kappa * gb[i];
    for (int i = 0; i < N * N; ++i) H[i] = kappa * Hb[i];
    return kappa * bv.b;
  }
  // 边–边的 mollifier m(c_x)，c_x = |u x v|^2，u = X1 - X0，v = X3 - X2（公式见 contact.cc）。
  const V3 u = X[1] - X[0], v = X[3] - X[2];
  const double uv = dot(u, v), uu = norm2(u), vv = norm2(v);
  const V3 gcu = 2 * vv * u - 2 * uv * v, gcv = 2 * uu * v - 2 * uv * u;
  double C[36];  // (u, v) 的 6 x 6 二阶导数
  const double ua[3] = {u.x, u.y, u.z}, va[3] = {v.x, v.y, v.z};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      C[a * 6 + b] = 2 * vv * (a == b) - 2 * va[a] * va[b];
      C[(3 + a) * 6 + 3 + b] = 2 * uu * (a == b) - 2 * ua[a] * ua[b];
      C[a * 6 + 3 + b] = 4 * ua[a] * va[b] - 2 * va[a] * ua[b] - 2 * uv * (a == b);
      C[(3 + b) * 6 + a] = C[a * 6 + 3 + b];
    }
  // (u, v) = T X：顶点 0、1 进 u（符号 -、+），2、3 进 v（符号 -、+）。
  const double sgn[4] = {-1, 1, -1, 1};
  const int blk[4] = {0, 0, 3, 3};
  double gcX[12], HcX[144];
  v3store(-gcu, gcX);
  v3store(gcu, gcX + 3);
  v3store(-gcv, gcX + 6);
  v3store(gcv, gcX + 9);
  for (int p = 0; p < 4; ++p)
    for (int q = 0; q < 4; ++q)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) HcX[(3 * p + a) * 12 + 3 * q + b] = sgn[p] * sgn[q] * C[(blk[p] + a) * 6 + blk[q] + b];
  const double eps = 1e-3 * uu * vv;
  const double ddm = c.dmoll != 0 && eps > 0 ? -2.0 / (eps * eps) : 0.0;
  for (int i = 0; i < 12; ++i) f[i] = kappa * (c.moll * gb[i] + bv.b * c.dmoll * gcX[i]);
  for (int i = 0; i < 12; ++i)
    for (int j = 0; j < 12; ++j)
      H[i * 12 + j] = kappa * (c.moll * Hb[i * 12 + j] + bv.b * (ddm * gcX[i] * gcX[j] + c.dmoll * HcX[i * 12 + j]) +
                               c.dmoll * (gcX[i] * gb[j] + gb[i] * gcX[j]));
  return kappa * c.moll * bv.b;
}

__global__ void k_contact_full(DModel m, DFrames F, const DContact* cs, DCount ncs, DScalar kp, double dhat, double* e,
                               double* g, DRows rows, double* CB) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int nc = ncs.get();
  const double kappa = kp.get();
  if (i >= nc) return;
  const DContact c = cs[i];
  const int N = 3 * c.nst;
  V3 X[4];
  for (int s = 0; s < c.nst; ++s) X[s] = v3load(F.X + 3 * c.v[s]);
  const double radii = c.type == kGround ? m.vradius[c.v[0]] : m.vradius[c.v[0]] + m.vradius[c.v[2]];
  double f[12], HX[144];
  const double E = vertex_derivatives(c, X, radii, kappa, dhat, f, HX);
  if (e) atomicAdd(e, E);

  // 两侧：参考点、每个顶点的 rho、合力 Fs、关于参考点的力矩 tau、几何项的 Q = sum rho f^T。
  const int body[2] = {c.bodyA, c.bodyB};
  V3 ref[2], Fs[2], tau[2];
  double Q[2][9];
  for (int s = 0; s < 2; ++s) {
    ref[s] = body[s] > 0 ? v3load(F.xpos + 3 * body[s]) : v3(0, 0, 0);
    Fs[s] = tau[s] = v3(0, 0, 0);
    for (int k = 0; k < 9; ++k) Q[s][k] = 0;
  }
  double Jv[144];  // N x 12：顶点速度对旋量，[I, -[rho]x]
  for (int k = 0; k < N * 12; ++k) Jv[k] = 0;
  for (int s = 0; s < c.nst; ++s) {
    const int vb = m.vbody[c.v[s]];
    if (vb == 0) continue;
    const int side = vb == c.bodyA ? 0 : 1;
    const V3 rho = X[s] - ref[side];
    const V3 fs = v3load(f + 3 * s);
    const double ra[3] = {rho.x, rho.y, rho.z}, fa[3] = {fs.x, fs.y, fs.z};
    for (int a = 0; a < 3; ++a) Jv[(3 * s + a) * 12 + 6 * side + a] = 1;
    // -[rho]x
    Jv[(3 * s + 0) * 12 + 6 * side + 4] = rho.z;
    Jv[(3 * s + 0) * 12 + 6 * side + 5] = -rho.y;
    Jv[(3 * s + 1) * 12 + 6 * side + 3] = -rho.z;
    Jv[(3 * s + 1) * 12 + 6 * side + 5] = rho.x;
    Jv[(3 * s + 2) * 12 + 6 * side + 3] = rho.y;
    Jv[(3 * s + 2) * 12 + 6 * side + 4] = -rho.x;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) Q[side][3 * a + b] += ra[a] * fa[b];
    Fs[side] = Fs[side] + fs;
    tau[side] = tau[side] + cross(rho, fs);
  }

  // 梯度：每侧的力旋量在链上各自由度上的值（和 sum_v f_v . J_v 相同）。
  if (g) {
    for (int s = 0; s < 2; ++s) {
      if (body[s] == 0) continue;
      for (int k = 0; k < m.chain_n[body[s]]; ++k) {
        const int d = m.chain_dof[body[s] * m.CH + k];
        const V3 a = v3load(F.daxis + 3 * d);
        const double val = m.drot[d] ? dot(a, tau[s] - cross(v3load(F.danchor + 3 * d) - ref[s], Fs[s])) : dot(a, Fs[s]);
        atomicAdd(g + d, val);
      }
    }
  }
  // 链上剩余项要的每刚体量：G = P + xpos F^T（和惯性的 c f^T 同一个形式）、F、P。
  if (CB) {
    for (int s = 0; s < 2; ++s) {
      if (body[s] == 0) continue;
      double* out = CB + 21 * static_cast<size_t>(body[s]);
      const double ra[3] = {ref[s].x, ref[s].y, ref[s].z}, fa[3] = {Fs[s].x, Fs[s].y, Fs[s].z};
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          atomicAdd(out + 3 * a + b, Q[s][3 * a + b] + ra[a] * fa[b]);
          atomicAdd(out + 12 + 3 * a + b, Q[s][3 * a + b]);
        }
      for (int a = 0; a < 3; ++a) atomicAdd(out + 9 + a, fa[a]);
    }
  }
  if (!rows.D) return;

  // H12 = Jv^T H_X Jv，加几何项 (Q + Q^T) / 2 - tr(Q) I 到 omega 块。
  double T[144];  // N x 12
  for (int r = 0; r < N; ++r)
    for (int col = 0; col < 12; ++col) {
      double s = 0;
      for (int k = 0; k < N; ++k) s += HX[r * N + k] * Jv[k * 12 + col];
      T[r * 12 + col] = s;
    }
  double H[144];
  for (int a = 0; a < 12; ++a)
    for (int b = 0; b < 12; ++b) {
      double s = 0;
      for (int r = 0; r < N; ++r) s += Jv[r * 12 + a] * T[r * 12 + b];
      H[a * 12 + b] = s;
    }
  for (int s = 0; s < 2; ++s) {
    if (body[s] == 0) continue;
    const double tr = Q[s][0] + Q[s][4] + Q[s][8];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        H[(6 * s + 3 + a) * 12 + 6 * s + 3 + b] += 0.5 * (Q[s][3 * a + b] + Q[s][3 * b + a]) - (a == b ? tr : 0.0);
  }
  for (int a = 0; a < 12; ++a)  // 对称化（舍入）
    for (int b = a + 1; b < 12; ++b) H[a * 12 + b] = H[b * 12 + a] = 0.5 * (H[a * 12 + b] + H[b * 12 + a]);

  // 特征分解，正的特征值各写一行：u 的两侧 (f, tau) 换成关于世界原点的力旋量 tau0 = tau + ref x f。
  double V[144], w[12];
  sym_eig(12, H, V, w);
  double lmax = 0;
  for (int k = 0; k < 12; ++k) lmax = fmax(lmax, w[k]);
  for (int k = 0; k < 12; ++k) {
    const size_t r = 12 * static_cast<size_t>(i) + k;
    double* wr = rows.w + 12 * r;
    const bool keep = lmax > 0 && w[k] > 1e-12 * lmax;
    rows.body[2 * r] = keep ? c.bodyA : 0;
    rows.body[2 * r + 1] = keep ? c.bodyB : 0;
    rows.D[r] = keep ? w[k] : 0.0;
    for (int s = 0; s < 2; ++s) {
      if (!keep || body[s] == 0) {
        for (int q = 0; q < 6; ++q) wr[6 * s + q] = 0;
        continue;
      }
      const V3 fu = v3(V[(6 * s) * 12 + k], V[(6 * s + 1) * 12 + k], V[(6 * s + 2) * 12 + k]);
      const V3 tu = v3(V[(6 * s + 3) * 12 + k], V[(6 * s + 4) * 12 + k], V[(6 * s + 5) * 12 + k]);
      v3store(fu, wr + 6 * s);
      v3store(tu + cross(ref[s], fu), wr + 6 * s + 3);
    }
  }
}

}  // namespace

void launch_contact_full(const DModel& m, const DFrames& F, const DContact* c, DCount n, DScalar kappa, double dhat,
                         double* e, double* g, DRows rows, double* CB) {
  if (n.grid() > 0) k_contact_full<<<blocks(n.grid()), kThreads>>>(m, F, c, n, kappa, dhat, e, g, rows, CB);
}

}  // namespace rcipc::gpu
