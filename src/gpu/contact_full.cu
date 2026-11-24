// GPU 版的完整接触 Hessian（阶段 14，笔记 notes/17_contact_hessian.md）。对应 CPU 版 contact.cc 里的
// squared_distance_hessian、contact_wrench_hessian、contact_rows_full，以及 contact_chain_curvature 里每个接触的那一份。
//
// 一个 warp 一个接触（中间量放在这个 warp 的共享内存里，WarpBuf）：
//   1. 顶点空间的梯度 f（3 nst）和 Hessian H_X（3 nst x 3 nst），同 contact.cc 的 vertex_derivatives
//      （vertex_scalars、vertex_vectors、hx_entry：H_X 的元素分给各线程）；
//   2. 投到两侧刚体的旋量上（参考点是各刚体的原点 xpos）：H12 = J^T H_X J 加顶点随转动走弧线的几何项（各线程分元素）；
//   3. 12 x 12 的 Jacobi 特征分解，按循环赛顺序每小轮 6 个互不相交的旋转一起做（warp_sym_eig12）；
//      正的特征值各写一行（rows 的第 12 i + k 行；用不到的行两侧刚体都是 0，D = 0）；
//   4. 梯度按两侧的力旋量加到 g；链上剩余项要的每刚体量 (G, F, P) 原子加到 CB（nbody x 21），
//      k_inertia_curvature（kin.cu）再把它们和惯性的那一份一起算、一起做正定检查。
// 能量和 k_contact_terms 算的一样，线搜索仍然用 k_contact_terms。
// 原来一个线程一个接触：选中一个接触的那次发射要 0.1–2.5 ms，现在约 0.1 ms（notes/11_gpu.md 第 16 节）。
#include "devmath.cuh"
#include "kernels.h"
#include "rows.cuh"

namespace rcipc::gpu {

namespace {

constexpr int kWarps = 1;  // 一个 warp 一个接触，每块一个 warp（双精度单元每个 SM 只有两个，接触少时分散到各个 SM 上）
constexpr unsigned kAll = 0xffffffffu;
inline int blocks(int n) { return (n + kWarps - 1) / kWarps; }

// 一个 warp 的共享内存。
struct WarpBuf {
  double A[144];     // H12（12 x 12），特征分解时原地变成对角
  double V[144];     // 先放 T = H_X Jv（N x 12），再放特征向量
  double HX[144];    // H_X（N x N，步长 N）
  double Jv[144];    // N x 12：顶点速度对两侧旋量，[I, -[rho]x]
  double f[12];      // 顶点空间的梯度
  double G[2][12], gy[12], gb[12], gcX[12];  // vertex_vectors 的输出
  double rot[6][3];  // 这一轮每对的 c、s、t
  double ref[2][3], Fs[2][3], tau[2][3], Q[2][9];
};

// 循环 Jacobi（n <= 2，vertex_scalars 里 F_{lam lam} 的伪逆用）：A（n x n，按行存）出来时对角是特征值，V 的列是特征向量。
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

// 循环赛顺序：第 r 轮（0..10）的第 j 对（0..5）。下标 0 不动，其余 11 个每轮转一格，第 j 个和第 11 - j 个配对：
// 每轮 6 对互不相交，11 轮正好是全部 66 对，各一次。
__device__ inline void rr_pair(int r, int j, int& p, int& q) {
  const int a = j == 0 ? 0 : 1 + (j - 1 + r) % 11, b = 1 + (10 - j + r) % 11;
  p = min(a, b);
  q = max(a, b);
}

// 一个 warp 做 12 x 12 的 Jacobi：A（共享内存，对称，按行存）出来时对角是特征值，V 的列是特征向量。
// 和上面的 sym_eig 是同一种旋转（A <- P^T A P，V <- V P，P 在 (p, q) 上是 [c s; -s c]），只是每轮 6 个一起做：
//   - 前 6 个线程各算一对的 c、s、t；
//   - A 按 6 x 6 个 2 x 2 的块分：块 (j, l) 只由第 j 对、第 l 对的旋转改，A' = R_j^T A_jl R_l；
//     对称，只做 j <= l 的 21 个块，同时写转置。对角块用 a_pp - t a_pq、a_qq + t a_pq、a_pq = 0（同 Numerical Recipes）；
//   - V 的每一行、每一对（72 份）各改两个数。
// 停的条件和 sym_eig 一样（off <= 1e-30 dg，最多 40 轮，每轮 11 小轮）。返回做了几轮。
__device__ int warp_sym_eig12(double* A, double* V, double (*rot)[3], int lane) {
  for (int e = lane; e < 144; e += 32) V[e] = e / 12 == e % 12 ? 1.0 : 0.0;
  __syncwarp();
  int sweep = 0;
  for (; sweep < 40; ++sweep) {
    double off = 0, dg = 0;
    for (int e = lane; e < 144; e += 32) {
      const int r = e / 12, c = e % 12;
      if (r < c) off += A[e] * A[e];
      if (r == c) dg += A[e] * A[e];
    }
    for (int o = 16; o > 0; o >>= 1) {
      off += __shfl_xor_sync(kAll, off, o);
      dg += __shfl_xor_sync(kAll, dg, o);
    }
    if (off <= 1e-30 * dg) break;  // 整个 warp 同一个判断
    for (int round = 0; round < 11; ++round) {
      if (lane < 6) {
        int p, q;
        rr_pair(round, lane, p, q);
        // 和 sym_eig 同一个旋转（t = sgn(theta) / (|theta| + sqrt(theta^2 + 1))，theta = d / (2 x)，d = a_qq - a_pp，
        // x = a_pq），写成两次 rsqrt、不用除法：rr = 1 / sqrt(d^2 + 4 x^2)，h = c^2 = (1 + |d| rr) / 2，
        // c = h / sqrt(h)，s = sgn(theta) |x| rr / sqrt(h)，t = s / c = s / sqrt(h)。原来的写法是 3 次除法、2 次开方
        // 串在一起，这张卡上双精度的除法、开方都是十几条指令的软件序列，一小轮的大半时间花在这里。
        const double x = A[p * 12 + q], d = A[q * 12 + q] - A[p * 12 + p], q2 = d * d + 4 * x * x;
        double c = 1, s = 0, t = 0;
        if (x != 0 && q2 > 0 && isfinite(q2)) {
          const double rr = rsqrt(q2), h = 0.5 * (1 + fabs(d) * rr), ic = rsqrt(h);
          c = h * ic;
          s = (d >= 0 ? x : -x) * rr * ic;
          t = s * ic;
        }
        rot[lane][0] = c;
        rot[lane][1] = s;
        rot[lane][2] = t;
      }
      __syncwarp();
      if (lane < 21) {
        int j = 0, rem = lane;  // 第 lane 个 j <= l 的块
        while (rem >= 6 - j) rem -= 6 - j++;
        const int l = j + rem;
        int p, q, r, s;
        rr_pair(round, j, p, q);
        rr_pair(round, l, r, s);
        if (j == l) {
          const double t = rot[j][2], apq = A[p * 12 + q];
          A[p * 12 + p] -= t * apq;
          A[q * 12 + q] += t * apq;
          A[p * 12 + q] = A[q * 12 + p] = 0;
        } else {
          const double cj = rot[j][0], sj = rot[j][1], cl = rot[l][0], sl = rot[l][1];
          const double apr = A[p * 12 + r], aps = A[p * 12 + s], aqr = A[q * 12 + r], aqs = A[q * 12 + s];
          // 先列（A R_l）：列 r' = cl 列 r - sl 列 s，列 s' = sl 列 r + cl 列 s
          const double bpr = cl * apr - sl * aps, bps = sl * apr + cl * aps;
          const double bqr = cl * aqr - sl * aqs, bqs = sl * aqr + cl * aqs;
          // 再行（R_j^T）：行 p' = cj 行 p - sj 行 q，行 q' = sj 行 p + cj 行 q
          A[p * 12 + r] = A[r * 12 + p] = cj * bpr - sj * bqr;
          A[p * 12 + s] = A[s * 12 + p] = cj * bps - sj * bqs;
          A[q * 12 + r] = A[r * 12 + q] = sj * bpr + cj * bqr;
          A[q * 12 + s] = A[s * 12 + q] = sj * bps + cj * bqs;
        }
      }
      for (int u = lane; u < 72; u += 32) {
        const int k = u / 6, j = u % 6;
        int p, q;
        rr_pair(round, j, p, q);
        const double c = rot[j][0], s = rot[j][1], vkp = V[k * 12 + p], vkq = V[k * 12 + q];
        V[k * 12 + p] = c * vkp - s * vkq;
        V[k * 12 + q] = s * vkp + c * vkq;
      }
      __syncwarp();
    }
  }
  return sweep;
}

// 顶点空间的梯度 f 和 Hessian H_X（contact.cc 的 vertex_derivatives、squared_distance_hessian），同样的公式，
// 按元素分给各线程：
//   1. 标量和 2 x 2 的伪逆（每个线程各算一份，vertex_scalars）；
//   2. 长度 N 的向量 gy、gb、gcX、G_k 和梯度 f（第 i 个线程算第 i 个分量，vertex_vectors，写到共享内存）；
//   3. H_X 的 N x N 个元素（hx_entry，各线程分）。
// 原来一个线程把 N x N 的几个矩阵一个一个元素算出来，边–边（N = 12，有 mollifier）一个接触约 120 μs。
struct VScal {
  int np;                 // 最近点参数的个数（contact.cc 的 squared_distance_hessian 里的 np）
  double dw[2][4];        // 第 k 个参数对各模板顶点的系数
  V3 Delta, E[2];
  double Finv[4];         // F_{lam lam} 的伪逆（np x np）
  double ell, alpha, cg, dba;  // dba = db alpha
  BV bv;
  bool mol;               // 边–边的 mollifier 起作用
  V3 gcu, gcv;
  double ua[3], va[3], uu, vv, uv, ddm;
};

__device__ VScal vertex_scalars(const DContact& c, const V3* X, double radii, double dhat) {
  VScal S;
  S.ell = c.d + radii;
  S.alpha = c.d / S.ell;
  S.bv = barrier(c.d * c.d, dhat * dhat);
  S.cg = S.bv.ddb * S.alpha * S.alpha + S.bv.db * radii / (2 * S.ell * S.ell * S.ell);
  S.dba = S.bv.db * S.alpha;
  S.np = 0;
  for (int k = 0; k < 2; ++k)
    for (int v = 0; v < 4; ++v) S.dw[k][v] = 0;
  S.Delta = v3(0, 0, 0);
  if (c.type != kGround) {
    for (int v = 0; v < c.nst; ++v) S.Delta = S.Delta + c.w[v] * X[v];
    if (c.type == kPointTriangle) {
      int active[3], na = 0;
      for (int i = 0; i < 3; ++i)
        if (-c.w[1 + i] > 0) active[na++] = i;
      for (int k = 1; k < na; ++k) {
        S.dw[S.np][1 + active[0]] = 1;
        S.dw[S.np][1 + active[k]] = -1;
        ++S.np;
      }
    } else {
      if (c.w[1] > 0 && c.w[1] < 1) {
        S.dw[S.np][0] = -1;
        S.dw[S.np][1] = 1;
        ++S.np;
      }
      if (-c.w[3] > 0 && -c.w[3] < 1) {
        S.dw[S.np][2] = 1;
        S.dw[S.np][3] = -1;
        ++S.np;
      }
    }
  }
  for (int k = 0; k < 4; ++k) S.Finv[k] = 0;
  if (S.np > 0) {
    for (int k = 0; k < S.np; ++k) {
      S.E[k] = v3(0, 0, 0);
      for (int v = 0; v < c.nst; ++v) S.E[k] = S.E[k] + S.dw[k][v] * X[v];
    }
    double Fll[4], V[4], w[2];
    for (int k = 0; k < S.np; ++k)
      for (int l = 0; l < S.np; ++l) Fll[k * S.np + l] = 2 * dot(S.E[k], S.E[l]);
    sym_eig(S.np, Fll, V, w);
    double lmax = 0;
    for (int k = 0; k < S.np; ++k) lmax = fmax(lmax, w[k]);
    for (int k = 0; k < S.np; ++k) {
      if (!(w[k] > 1e-10 * lmax)) continue;  // 伪逆：退化的方向不动（两条边平行、边长为 0）
      for (int a = 0; a < S.np; ++a)
        for (int b = 0; b < S.np; ++b) S.Finv[a * S.np + b] += V[a * S.np + k] * V[b * S.np + k] / w[k];
    }
  }
  S.mol = c.type == kEdgeEdge && !(c.moll == 1 && c.dmoll == 0);
  if (S.mol) {
    const V3 u = X[1] - X[0], v = X[3] - X[2];
    S.uv = dot(u, v);
    S.uu = norm2(u);
    S.vv = norm2(v);
    S.gcu = 2 * S.vv * u - 2 * S.uv * v;
    S.gcv = 2 * S.uu * v - 2 * S.uv * u;
    S.ua[0] = u.x, S.ua[1] = u.y, S.ua[2] = u.z;
    S.va[0] = v.x, S.va[1] = v.y, S.va[2] = v.z;
    const double eps = 1e-3 * S.uu * S.vv;
    S.ddm = c.dmoll != 0 && eps > 0 ? -2.0 / (eps * eps) : 0.0;
  }
  return S;
}

// 第 i 个分量（i < N）：gy、gb、gcX、G_k（k < np）和梯度 f（kappa 已乘上）。
__device__ void vertex_vectors(const DContact& c, const V3* X, const VScal& S, double kappa, int i, WarpBuf& B) {
  const int v = i / 3, a = i % 3;
  const double gy = (2 * S.ell * c.w[v]) * c.n[a];
  const double gb = S.bv.db * S.alpha * gy;
  B.gy[i] = gy;
  B.gb[i] = gb;
  const double Da[3] = {S.Delta.x, S.Delta.y, S.Delta.z};
  for (int k = 0; k < S.np; ++k) {
    const double Ea[3] = {S.E[k].x, S.E[k].y, S.E[k].z};
    B.G[k][i] = 2 * c.w[v] * Ea[a] + 2 * S.dw[k][v] * Da[a];
  }
  if (!S.mol) {
    B.f[i] = kappa * gb;
    return;
  }
  const double gu[3] = {S.gcu.x, S.gcu.y, S.gcu.z}, gv[3] = {S.gcv.x, S.gcv.y, S.gcv.z};
  const double gc = v == 0 ? -gu[a] : v == 1 ? gu[a] : v == 2 ? -gv[a] : gv[a];
  B.gcX[i] = gc;
  B.f[i] = kappa * (c.moll * gb + S.bv.b * c.dmoll * gc);
}

// mollifier 的 c_x = |u x v|^2 对 (u, v) 的二阶导数（6 x 6，contact.cc 的 C）的第 (A, B) 个元素。
__device__ inline double cuv_entry(const VScal& S, int A, int B) {
  if (A >= 3 && B < 3) {  // 下三角的块用转置
    const int t = A;
    A = B;
    B = t;
  }
  if (A < 3 && B < 3) return 2 * S.vv * (A == B) - 2 * S.va[A] * S.va[B];
  if (A >= 3 && B >= 3) return 2 * S.uu * (A == B) - 2 * S.ua[A - 3] * S.ua[B - 3];
  const int a = A, b = B - 3;
  return 4 * S.ua[a] * S.va[b] - 2 * S.va[a] * S.ua[b] - 2 * S.uv * (a == b);
}

// H_X 的第 (i, j) 个元素（kappa 已乘上）。B 里的向量要先写好。
__device__ double hx_entry(const DContact& c, const VScal& S, const WarpBuf& B, double kappa, int i, int j) {
  const int v = i / 3, a = i % 3, u = j / 3, b = j % 3;
  double hy;
  if (c.type == kGround) {
    hy = 2 * c.n[a] * c.n[b];
  } else {
    hy = a == b ? 2 * c.w[v] * c.w[u] : 0.0;
    double s = 0;
    for (int k = 0; k < S.np; ++k)
      for (int l = 0; l < S.np; ++l) s += B.G[k][i] * S.Finv[k * S.np + l] * B.G[l][j];
    hy -= s;
  }
  const double hb = S.cg * B.gy[i] * B.gy[j] + S.dba * hy;
  if (!S.mol) return kappa * hb;
  const double sgn[4] = {-1, 1, -1, 1};
  const int blk[4] = {0, 0, 3, 3};
  const double hc = sgn[v] * sgn[u] * cuv_entry(S, blk[v] + a, blk[u] + b);
  return kappa * (c.moll * hb + S.bv.b * (S.ddm * B.gcX[i] * B.gcX[j] + c.dmoll * hc) +
                  c.dmoll * (B.gcX[i] * B.gb[j] + B.gb[i] * B.gcX[j]));
}

// 一个接触的顶点空间的 f、H_X，和两侧的 Jv、ref、Fs、tau、Q（warp 一起做，结果在 B 里）。e 不为空时加能量。
__device__ void warp_contact_setup(const DModel& m, const DFrames& F, const DContact& c, double kappa, double dhat,
                                   WarpBuf& B, int lane, double* e) {
  const int N = 3 * c.nst;
  const int body[2] = {c.bodyA, c.bodyB};
  V3 X[4];
  for (int s = 0; s < c.nst; ++s) X[s] = v3load(F.X + 3 * c.v[s]);
  const double radii = c.type == kGround ? m.vradius[c.v[0]] : m.vradius[c.v[0]] + m.vradius[c.v[2]];
  for (int k = lane; k < 144; k += 32) B.Jv[k] = 0;
  // 顶点空间的 f、H_X（vertex_derivatives 的 warp 版）。
  const VScal S = vertex_scalars(c, X, radii, dhat);
  if (lane < N) vertex_vectors(c, X, S, kappa, lane, B);
  __syncwarp();
  for (int k = lane; k < N * N; k += 32) B.HX[k] = hx_entry(c, S, B, kappa, k / N, k % N);
  __syncwarp();

  if (lane == 0) {
    if (e) atomicAdd(e, S.mol ? kappa * c.moll * S.bv.b : kappa * S.bv.b);

    // 两侧：参考点、每个顶点的 rho、合力 Fs、关于参考点的力矩 tau、几何项的 Q = sum rho f^T。
    V3 ref[2], Fs[2], tau[2];
    double Q[2][9];
    for (int s = 0; s < 2; ++s) {
      ref[s] = body[s] > 0 ? v3load(F.xpos + 3 * body[s]) : v3(0, 0, 0);
      Fs[s] = tau[s] = v3(0, 0, 0);
      for (int k = 0; k < 9; ++k) Q[s][k] = 0;
    }
    double* Jv = B.Jv;
    for (int s = 0; s < c.nst; ++s) {
      const int vb = m.vbody[c.v[s]];
      if (vb == 0) continue;
      const int side = vb == c.bodyA ? 0 : 1;
      const V3 rho = X[s] - ref[side];
      const V3 fs = v3load(B.f + 3 * s);
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
    for (int s = 0; s < 2; ++s) {
      v3store(ref[s], B.ref[s]);
      v3store(Fs[s], B.Fs[s]);
      v3store(tau[s], B.tau[s]);
      for (int k = 0; k < 9; ++k) B.Q[s][k] = Q[s][k];
    }
  }
  __syncwarp();
}

// H12 = Jv^T H_X Jv 加几何项（warp_contact_setup 之后），放在 B.A；B.V 用作中间量。
__device__ void warp_contact_h12(const DContact& c, WarpBuf& B, int lane) {
  const int N = 3 * c.nst;
  const int body[2] = {c.bodyA, c.bodyB};
  // T = H_X Jv（N x 12，放在 V 里）。
  for (int k = lane; k < N * 12; k += 32) {
    const int r = k / 12, col = k % 12;
    double s = 0;
    for (int t = 0; t < N; ++t) s += B.HX[r * N + t] * B.Jv[t * 12 + col];
    B.V[k] = s;
  }
  __syncwarp();
  // H12 = Jv^T T 加几何项 (Q + Q^T) / 2 - tr(Q) I 到 omega 块，对称化（舍入）：各线程做上三角的一个元素，同时写转置。
  for (int k = lane; k < 78; k += 32) {
    int a = 0, rem = k;
    while (rem >= 12 - a) rem -= 12 - a++;
    const int b = a + rem;
    double hab = 0, hba = 0;
    for (int r = 0; r < N; ++r) {
      hab += B.Jv[r * 12 + a] * B.V[r * 12 + b];
      hba += B.Jv[r * 12 + b] * B.V[r * 12 + a];
    }
    double h = 0.5 * (hab + hba);
    const int s = a / 6;
    if (b / 6 == s && a % 6 >= 3 && b % 6 >= 3 && body[s] != 0) {
      const int aa = a % 6 - 3, bb = b % 6 - 3;
      const double* Q = B.Q[s];
      h += 0.5 * (Q[3 * aa + bb] + Q[3 * bb + aa]) - (aa == bb ? Q[0] + Q[4] + Q[8] : 0.0);
    }
    B.A[a * 12 + b] = B.A[b * 12 + a] = h;
  }
  __syncwarp();
}

__global__ void __launch_bounds__(32 * kWarps)
    k_contact_full(DModel m, DFrames F, const DContact* cs, DCount ncs, DScalar kp, double dhat, double* e, double* g,
                   DRows rows, double* CB) {
  // 第 i 个 warp 做第 i 个接触，行写到 12 i 起。
  __shared__ WarpBuf bufs[kWarps];
  const int lane = threadIdx.x % 32;
  const int i = blockIdx.x * kWarps + threadIdx.x / 32;
  if (i >= ncs.get()) return;  // 整个 warp 一起退
  WarpBuf& B = bufs[threadIdx.x / 32];
  const DContact c = cs[i];
  const int body[2] = {c.bodyA, c.bodyB};
  warp_contact_setup(m, F, c, kp.get(), dhat, B, lane, e);

  // 梯度：每侧的力旋量在链上各自由度上的值（和 sum_v f_v . J_v 相同），各线程分自由度。
  if (g) {
    for (int s = 0; s < 2; ++s) {
      if (body[s] == 0) continue;
      const V3 ref = v3load(B.ref[s]), Fs = v3load(B.Fs[s]), tau = v3load(B.tau[s]);
      for (int k = lane; k < m.chain_n[body[s]]; k += 32) {
        const int d = m.chain_dof[body[s] * m.CH + k];
        const V3 a = v3load(F.daxis + 3 * d);
        const double val = m.drot[d] ? dot(a, tau - cross(v3load(F.danchor + 3 * d) - ref, Fs)) : dot(a, Fs);
        atomicAdd(g + d, val);
      }
    }
  }
  // 链上剩余项要的每刚体量：G = P + xpos F^T（和惯性的 c f^T 同一个形式）、F、P（out 的 0..8、9..11、12..20）。
  if (CB) {
    for (int s = 0; s < 2; ++s) {
      if (body[s] == 0 || lane >= 21) continue;
      double* out = CB + 21 * static_cast<size_t>(body[s]);
      double v;
      if (lane < 9)
        v = B.Q[s][lane] + B.ref[s][lane / 3] * B.Fs[s][lane % 3];
      else if (lane < 12)
        v = B.Fs[s][lane - 9];
      else
        v = B.Q[s][lane - 12];
      atomicAdd(out + lane, v);
    }
  }
  if (!rows.D) return;
  warp_contact_h12(c, B, lane);

  // 特征分解，正的特征值各写一行（第 k 个线程写第 k 行）：u 的两侧 (f, tau) 换成关于世界原点的力旋量 tau0 = tau + ref x f。
  warp_sym_eig12(B.A, B.V, B.rot, lane);
  if (lane >= 12) return;
  const int k = lane;
  double lmax = 0;
  for (int t = 0; t < 12; ++t) lmax = fmax(lmax, B.A[t * 12 + t]);
  const double wk = B.A[k * 12 + k];
  const size_t r = 12 * static_cast<size_t>(i) + k;
  double* wr = rows.w + 12 * r;
  const bool keep = lmax > 0 && wk > 1e-12 * lmax;
  rows.body[2 * r] = keep ? c.bodyA : 0;
  rows.body[2 * r + 1] = keep ? c.bodyB : 0;
  rows.D[r] = keep ? wk : 0.0;
  for (int s = 0; s < 2; ++s) {
    if (!keep || body[s] == 0) {
      for (int q = 0; q < 6; ++q) wr[6 * s + q] = 0;
      continue;
    }
    const double* V = B.V;
    const V3 fu = v3(V[(6 * s) * 12 + k], V[(6 * s + 1) * 12 + k], V[(6 * s + 2) * 12 + k]);
    const V3 tu = v3(V[(6 * s + 3) * 12 + k], V[(6 * s + 4) * 12 + k], V[(6 * s + 5) * 12 + k]);
    v3store(fu, wr + 6 * s);
    v3store(tu + cross(v3load(B.ref[s]), fu), wr + 6 * s + 3);
  }
}

}  // namespace

void launch_contact_full(const DModel& m, const DFrames& F, const DContact* c, DCount n, DScalar kappa, double dhat,
                         double* e, double* g, DRows rows, double* CB) {
  if (n.grid() > 0) k_contact_full<<<blocks(n.grid()), 32 * kWarps>>>(m, F, c, n, kappa, dhat, e, g, rows, CB);
}

}  // namespace rcipc::gpu
