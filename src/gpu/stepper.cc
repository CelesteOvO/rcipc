// GPU 版的时间步：主机这一侧（阶段 11，笔记 notes/11_gpu.md 第 6 节）。
//
// 这个文件做两件事：
//   1. 把 Model 摊平成 DModel 的数组，上传到显存（Upload）；
//   2. 跑和 CPU 版 step()（src/step.cc）一行一行对应的 Newton 循环，每一项都调 kernels.h 里的内核。
// 需要读回主机的只有几个标量（能量、步长、收敛量、PCG 的迭代次数、计数），所以显存和主机之间来回的数据很少。
// 每次读回都要等 GPU 把前面的活做完，GPU 在主机发下一批内核之前是空着的；所以能一起读的尽量一起读
// （组装的能量、PCG 的迭代次数、收敛判断的位移一次读回；限位 CCD 的结果跟着第一段 CCD 读回；
// kappa 调整用的最小距离跟着线搜索的能量读回），往显存里写一个数也用小内核，不用同步的拷贝（笔记第 8 节）。
// 默认的写法更进一步：一个 Newton 迭代录成一张 CUDA graph，CCD 的分段和线搜索的回溯在设备上循环，
// 每个迭代只读回一次（newton_graph、record_graph，笔记第 12 节）；逐个发射的循环（step() 里）留着做对照和后备。
#include "rcipc/gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <cstddef>
#include <cstring>

#include "device_model.h"
#include "iter.h"
#include "kernels.h"
#include "rcipc/contact.h"
#include "rcipc/limits.h"

namespace rcipc::gpu {

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}

// 显存缓冲区的“版本”：任何一块重新分配、或者两组交换指针，都加一。录好的 CUDA graph 里写死了指针，
// 版本变了就要重录（笔记第 12 节）。
long g_buf_gen = 0;

// 一块显存。reserve 只增不减，增大时原来的内容不保留（这里的缓冲区都是每次重写的）。
template <class T>
class DBuf {
 public:
  DBuf() = default;
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  ~DBuf() {
    if (p_) cudaFree(p_);
  }
  void reserve(size_t n) {
    if (n <= n_ && p_) return;
    if (p_) cudaFree(p_);
    p_ = nullptr;
    check(cudaMalloc(&p_, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    n_ = std::max<size_t>(n, 1);
    ++g_buf_gen;
  }
  void upload(const T* src, size_t n) {
    reserve(n);
    if (n) check(cudaMemcpy(p_, src, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  void upload(const std::vector<T>& v) { upload(v.data(), v.size()); }
  void download(T* dst, size_t n) const {
    if (n) check(cudaMemcpy(dst, p_, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  }
  T* get() const { return p_; }
  size_t size() const { return n_; }
  void swap(DBuf& o) {
    std::swap(p_, o.p_);
    std::swap(n_, o.n_);
    ++g_buf_gen;
  }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

// 一组正运动学的结果。
struct FrameBufs {
  DBuf<double> xpos, xquat, daxis, danchor, X;
  void reserve(const DModel& m) {
    xpos.reserve(3 * m.nbody);
    xquat.reserve(4 * m.nbody);
    daxis.reserve(3 * std::max(m.nv, 1));
    danchor.reserve(3 * std::max(m.nv, 1));
    X.reserve(3 * std::max(m.nvert, 1));
  }
  DFrames view() const { return DFrames{xpos.get(), xquat.get(), daxis.get(), danchor.get(), X.get()}; }
  // 只交换指针，不拷贝（线搜索最后一次试探的 FK 直接拿来当新位置的 FK）。
  void swap(FrameBufs& o) {
    xpos.swap(o.xpos);
    xquat.swap(o.xquat);
    daxis.swap(o.daxis);
    danchor.swap(o.danchor);
    X.swap(o.X);
  }
};

double host_max(const VectorXd& x) { return x.size() ? x.cwiseAbs().maxCoeff() : 0.0; }

// 双精度转单精度，向下 / 向上取整（盒子要向外取整，单精度的盒子才一定包住双精度的）。
float f_down(double x) {
  float f = static_cast<float>(x);
  if (static_cast<double>(f) > x) f = std::nextafter(f, -std::numeric_limits<float>::infinity());
  return f;
}
float f_up(double x) {
  float f = static_cast<float>(x);
  if (static_cast<double>(f) < x) f = std::nextafter(f, std::numeric_limits<float>::infinity());
  return f;
}

// 一种图元（三角形或边）在刚体坐标系里的盒子和分组（narrow.cu）。
struct ChunkSet {
  std::vector<float> box;    // 每个图元 6 个：刚体坐标系里的盒子（含半径，向外取整）
  std::vector<int> order;    // 按组排好的图元编号
  std::vector<int> range;    // 每组 2 个：在 order 里的起点、个数
  std::vector<float> cbox;   // 每组 6 个：组里各个盒子的并
  std::vector<int> bch;      // 每个刚体 2 个：第一组、组数
  std::vector<float> bbox;   // 每个刚体 6 个：它全部图元的总盒子（没有图元时 lo > hi）
};

// 刚体 b 的图元是 [first[b], first[b] + count[b])，图元 i 的顶点是 verts[k i .. k i + k - 1]。
// 分组和上一个分支的一样：按盒子中心坐标跨度最大的那个轴，在中位数处一分为二，直到每组不超过 CH 个。
ChunkSet build_chunks(int nbody, const std::vector<int>& first, const std::vector<int>& count, const std::vector<int>& verts,
                      int k, const std::vector<double>& vlocal, const std::vector<double>& vradius, int CH = 32) {
  ChunkSet S;
  const int n = static_cast<int>(verts.size()) / k;
  S.box.resize(6 * static_cast<size_t>(n));
  std::vector<double> cen(3 * static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int j = 0; j < k; ++j) {
      const int v = verts[k * i + j];
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], vlocal[3 * v + a] - vradius[v]);
        hi[a] = std::max(hi[a], vlocal[3 * v + a] + vradius[v]);
      }
    }
    for (int a = 0; a < 3; ++a) {
      S.box[6 * i + a] = f_down(lo[a]);
      S.box[6 * i + 3 + a] = f_up(hi[a]);
      cen[3 * i + a] = 0.5 * (lo[a] + hi[a]);
    }
  }
  const float inf = std::numeric_limits<float>::infinity();
  const auto add_chunk = [&](const std::vector<int>& ids) {
    S.range.push_back(static_cast<int>(S.order.size()));
    S.range.push_back(static_cast<int>(ids.size()));
    float c[6] = {inf, inf, inf, -inf, -inf, -inf};
    for (int i : ids) {
      S.order.push_back(i);
      for (int a = 0; a < 3; ++a) {
        c[a] = std::min(c[a], S.box[6 * i + a]);
        c[3 + a] = std::max(c[3 + a], S.box[6 * i + 3 + a]);
      }
    }
    S.cbox.insert(S.cbox.end(), c, c + 6);
  };
  std::function<void(std::vector<int>)> split = [&](std::vector<int> ids) {
    if (static_cast<int>(ids.size()) <= CH) {
      add_chunk(ids);
      return;
    }
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int i : ids)
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], cen[3 * i + a]);
        hi[a] = std::max(hi[a], cen[3 * i + a]);
      }
    int ax = 0;
    for (int a = 1; a < 3; ++a)
      if (hi[a] - lo[a] > hi[ax] - lo[ax]) ax = a;
    std::stable_sort(ids.begin(), ids.end(), [&](int x, int y) { return cen[3 * x + ax] < cen[3 * y + ax]; });
    const size_t h = ids.size() / 2;
    split(std::vector<int>(ids.begin(), ids.begin() + h));
    split(std::vector<int>(ids.begin() + h, ids.end()));
  };
  S.bch.resize(2 * static_cast<size_t>(nbody));
  S.bbox.resize(6 * static_cast<size_t>(nbody));
  for (int b = 0; b < nbody; ++b) {
    const int c0 = static_cast<int>(S.range.size()) / 2;
    std::vector<int> ids(count[b]);
    for (int i = 0; i < count[b]; ++i) ids[i] = first[b] + i;
    if (!ids.empty()) split(ids);
    const int c1 = static_cast<int>(S.range.size()) / 2;
    S.bch[2 * b] = c0;
    S.bch[2 * b + 1] = c1 - c0;
    float bb[6] = {inf, inf, inf, -inf, -inf, -inf};
    for (int c = c0; c < c1; ++c)
      for (int a = 0; a < 3; ++a) {
        bb[a] = std::min(bb[a], S.cbox[6 * c + a]);
        bb[3 + a] = std::max(bb[3 + a], S.cbox[6 * c + 3 + a]);
      }
    for (int a = 0; a < 6; ++a) S.bbox[6 * b + a] = bb[a];
  }
  if (S.range.empty()) {  // 没有这种图元：放一个空组，显存里的数组不为空
    S.range = {0, 0};
    S.cbox = {inf, inf, inf, -inf, -inf, -inf};
  }
  if (S.order.empty()) S.order = {0};
  if (S.box.empty()) S.box = {inf, inf, inf, -inf, -inf, -inf};
  return S;
}

}  // namespace

bool available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct Stepper::Impl {
  Model model;
  DModel m;

  // 模型的数组。
  DBuf<int> i_parent, i_bjnt, i_btree, i_bsub_end, i_chain_dof, i_chain_n, i_bvert0, i_bnvert, i_bedge0, i_bnedge,
      i_bface0, i_bnface, i_jtype, i_jqadr, i_jdadr, i_jbody, i_jlimited, i_dbody, i_drot, i_dtree, i_tdof0, i_tndof,
      i_tbody0, i_tnbody, i_vbody, i_ev, i_ebody, i_fv, i_fbody;
  DBuf<double> d_bpos, d_bquat, d_mass, d_ipos, d_inertia, d_S, d_radius, d_jpos, d_jaxis, d_jref, d_jrange, d_vlocal,
      d_vradius, d_pn, d_poff;

  // 状态和工作区。
  FrameBufs F, F0, Fp, F1, Fa, Fb, Ft;
  DBuf<double> q, q0, qprev, p, qtrial, g, ge0, gb, Hk, P, scale, work, scal, tc, tA, A, boxes, rhs;
  DBuf<int> pairs, cnt;
  // 窄相剔除（narrow.cu）：刚体坐标系里的几何（上传时算一次）和每次用的工作区。
  DBuf<float> f_vlf, f_flbox, f_elbox, f_blbox_f, f_blbox_e, f_fch_box, f_ech_box;
  DBuf<int> i_fch_order, i_fch_range, i_bfch, i_ech_order, i_ech_range, i_bech;
  DBuf<float> n_pxf, n_ppad, n_pxm;
  DBuf<float> xfa, xfb;  // CCD 一段起点、终点的顶点坐标（单精度）
  DBuf<double> n_pT;
  DBuf<DWork> n_work;
  int work_cap = 0;
  DBuf<double> bdisp;      // CCD 每一段里每个刚体的最大位移
  DBuf<DCand> cand;        // 窄相的候选图元对（个数在 cnt[2]）
  int cand_cap = 0;
  int pair_cap = 0;
  DBuf<DContact> contacts;
  int contact_cap = 0, nc = 0;
  DBuf<DFriction> fric;
  int nf = 0;
  DBuf<int> row_body;
  DBuf<double> row_w, row_D;
  int row_cap = 0, nrow = 0;
  DBuf<double> pw_Wb;          // 预条件器：每个刚体的 6x6（21 个）
  DBuf<int> pw_cross, pw_n;    // 两侧在同一棵树里的行，和它们的个数
  int last_eval_n = 0;  // energy_at 最后一次检测到的接触个数（contacts 里放的就是这些）
  // 协作内核版的 PCG（pcg_coop.cu）：acc 是每次迭代的点积，pcg_out 是迭代次数和是否收敛。
  bool coop = false;
  bool timing = true;  // 分段计时（每段结束同步一次）
  bool reuse = true;   // CCD 的段落在候选的窗口里时不重新求候选（set_ccd_reuse，笔记第 13 节）
  double last_eval_dmin = 0;  // energy_at 最后一次检测到的接触里最小的距离
  DBuf<double> pcg_acc;

  // CUDA graph 版的 Newton 迭代（笔记第 12 节）。
  bool graphs = true;      // 用不用（set_graphs）
  bool graph_failed = false;  // 录图或者启动失败过一次，以后就不再试
  bool last_graph = false;    // 上一步是不是用图跑的（测试用）
  DBuf<DIter> d_it;
  DIter* h_it = nullptr;   // 锁页内存：迭代前写 kappa、nc、nf，迭代后整个读回
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  // 录图时写死的东西：缓冲区的版本，和 StepParams 里用到的常数。变了就重录。
  struct GraphKey {
    long gen = -1;
    double h = 0, dhat = 0, mu = 0, eps_v = 0, curv = 0, limit_k = 0, limit_dhat = 0, c = 0, tol = 0, pcg_tol = 0;
    int pcg_max = 0;
    bool tight = true, reuse = true;
    bool operator==(const GraphKey& o) const {
      return gen == o.gen && h == o.h && dhat == o.dhat && mu == o.mu && eps_v == o.eps_v && curv == o.curv &&
             limit_k == o.limit_k && limit_dhat == o.limit_dhat && c == o.c && tol == o.tol && pcg_tol == o.pcg_tol &&
             pcg_max == o.pcg_max && tight == o.tight && reuse == o.reuse;
    }
  };
  GraphKey graph_key;

  Timers timers;

  explicit Impl(const Model& model_in) : model(model_in) {
    upload();
    coop = pcg_coop_supported();
    d_it.reserve(1);
    check(cudaMallocHost(&h_it, sizeof(DIter)), "cudaMallocHost");
    *h_it = DIter();
  }
  ~Impl() {
    drop_graph();
    if (h_it) cudaFreeHost(h_it);
  }

  // ---------- 上传 ----------

  void upload() {
    const Model& M = model;
    const int nb = M.nbody();
    m.nbody = nb;
    m.nv = M.nv;
    m.nq = M.nq;
    m.njnt = static_cast<int>(M.joints.size());
    m.ntree = M.ntree;
    m.nvert = M.nvertex();
    m.nedge = M.nedge();
    m.nface = M.nface();
    m.nplane = static_cast<int>(M.planes.size());
    for (int i = 0; i < 3; ++i) m.gravity[i] = M.gravity[i];
    m.filter_parent = M.filter_parent ? 1 : 0;

    // 刚体。
    std::vector<int> parent(nb), bjnt(nb), btree(nb), sub_end(nb);
    std::vector<double> bpos(3 * nb), bquat(4 * nb), mass(nb), ipos(3 * nb), inertia(9 * nb), S(9 * nb), radius(nb);
    for (int b = 0; b < nb; ++b) {
      const Body& B = M.bodies[b];
      parent[b] = b == 0 ? 0 : B.parent;
      bjnt[b] = B.joint;
      btree[b] = M.body_tree[b];
      for (int i = 0; i < 3; ++i) {
        bpos[3 * b + i] = B.pos[i];
        ipos[3 * b + i] = B.ipos[i];
      }
      bquat[4 * b] = B.quat.w();
      bquat[4 * b + 1] = B.quat.x();
      bquat[4 * b + 2] = B.quat.y();
      bquat[4 * b + 3] = B.quat.z();
      mass[b] = B.mass;
      radius[b] = B.radius;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          inertia[9 * b + 3 * i + j] = B.inertia(i, j);
          S[9 * b + 3 * i + j] = B.S(i, j);
        }
    }
    // 子树的范围：MuJoCo 的刚体按深度优先编号，子树是连续的一段 [b, sub_end[b])。顺便检查这一点。
    for (int b = 0; b < nb; ++b) sub_end[b] = b + 1;
    for (int b = nb - 1; b >= 1; --b) sub_end[parent[b]] = std::max(sub_end[parent[b]], sub_end[b]);
    for (int b = 1; b < nb; ++b) {
      for (int k = b + 1; k < sub_end[b]; ++k) {
        int x = k;
        while (x != 0 && x != b) x = parent[x];
        if (x != b) throw std::runtime_error("gpu: bodies are not in depth-first order");
      }
    }
    // 每个刚体的链上的自由度。
    std::vector<std::vector<int>> chains(nb);
    int CH = 1;
    for (int b = 1; b < nb; ++b) {
      for (int k = b; k != 0; k = parent[k]) {
        const int j = bjnt[k];
        if (j < 0) continue;
        for (int d = M.joints[j].dadr; d < M.joints[j].dadr + M.joints[j].nv; ++d) chains[b].push_back(d);
      }
      CH = std::max(CH, static_cast<int>(chains[b].size()));
    }
    if (CH > 64) throw std::runtime_error("gpu: a chain has more than 64 dofs");
    m.CH = CH;
    std::vector<int> chain_dof(nb * CH, -1), chain_n(nb, 0);
    for (int b = 0; b < nb; ++b) {
      chain_n[b] = static_cast<int>(chains[b].size());
      for (int k = 0; k < chain_n[b]; ++k) chain_dof[b * CH + k] = chains[b][k];
    }
    // 图元的范围：同一个刚体的顶点、边、三角形是连续的（geom 按刚体排序，model.cc 按 geom 的顺序追加）。
    const auto ranges = [&](auto body_of, int n, std::vector<int>& first, std::vector<int>& count, const char* what) {
      first.assign(nb, 0);
      count.assign(nb, 0);
      for (int i = 0; i < n; ++i) {
        const int b = body_of(i);
        if (count[b] == 0) first[b] = i;
        else if (first[b] + count[b] != i) throw std::runtime_error(std::string("gpu: ") + what + " of a body are not contiguous");
        ++count[b];
      }
    };
    std::vector<int> bvert0, bnvert, bedge0, bnedge, bface0, bnface;
    ranges([&](int i) { return M.vertices[i].body; }, m.nvert, bvert0, bnvert, "vertices");
    ranges([&](int i) { return M.edges[i].body; }, m.nedge, bedge0, bnedge, "edges");
    ranges([&](int i) { return M.faces[i].body; }, m.nface, bface0, bnface, "faces");

    // 关节。
    const int nj = m.njnt;
    std::vector<int> jtype(nj), jqadr(nj), jdadr(nj), jbody(nj), jlimited(nj);
    std::vector<double> jpos(3 * nj), jaxis(3 * nj), jref(nj), jrange(2 * nj);
    for (int j = 0; j < nj; ++j) {
      const Joint& J = M.joints[j];
      jtype[j] = static_cast<int>(J.type);
      jqadr[j] = J.qadr;
      jdadr[j] = J.dadr;
      jbody[j] = J.body;
      jlimited[j] = J.limited ? 1 : 0;
      for (int i = 0; i < 3; ++i) {
        jpos[3 * j + i] = J.pos[i];
        jaxis[3 * j + i] = J.axis[i];
      }
      jref[j] = J.ref;
      jrange[2 * j] = J.range[0];
      jrange[2 * j + 1] = J.range[1];
    }
    // 自由度和树（检查每棵树的自由度、刚体都是连续的一段）。
    std::vector<int> dbody(M.nv), drot(M.nv), dtree(M.nv);
    std::vector<int> tdof0(std::max(M.ntree, 1), 0), tndof(std::max(M.ntree, 1), 0), tbody0(std::max(M.ntree, 1), 0),
        tnbody(std::max(M.ntree, 1), 0);
    for (int d = 0; d < M.nv; ++d) {
      dbody[d] = M.dof_body[d];
      drot[d] = M.dof_rot[d] ? 1 : 0;
      dtree[d] = M.dof_tree[d];
      const int t = dtree[d];
      if (tndof[t] == 0) tdof0[t] = d;
      else if (tdof0[t] + tndof[t] != d) throw std::runtime_error("gpu: dofs of a tree are not contiguous");
      ++tndof[t];
    }
    for (int b = 1; b < nb; ++b) {
      const int t = btree[b];
      if (tnbody[t] == 0) tbody0[t] = b;
      else if (tbody0[t] + tnbody[t] != b) throw std::runtime_error("gpu: bodies of a tree are not contiguous");
      ++tnbody[t];
    }
    m.TD = 1;
    m.TB = 1;
    for (int t = 0; t < M.ntree; ++t) {
      m.TD = std::max(m.TD, tndof[t]);
      m.TB = std::max(m.TB, tnbody[t]);
    }

    // 几何。
    std::vector<int> vbody(m.nvert), ev(2 * m.nedge), ebody(m.nedge), fv(3 * m.nface), fbody(m.nface);
    std::vector<double> vlocal(3 * m.nvert), vradius(m.nvert), pn(3 * m.nplane), poff(m.nplane);
    for (int v = 0; v < m.nvert; ++v) {
      vbody[v] = M.vertices[v].body;
      for (int i = 0; i < 3; ++i) vlocal[3 * v + i] = M.vertices[v].local[i];
      vradius[v] = M.vertices[v].radius;
    }
    for (int e = 0; e < m.nedge; ++e) {
      ev[2 * e] = M.edges[e].v0;
      ev[2 * e + 1] = M.edges[e].v1;
      ebody[e] = M.edges[e].body;
    }
    for (int f = 0; f < m.nface; ++f) {
      fv[3 * f] = M.faces[f].v0;
      fv[3 * f + 1] = M.faces[f].v1;
      fv[3 * f + 2] = M.faces[f].v2;
      fbody[f] = M.faces[f].body;
    }
    for (int pl = 0; pl < m.nplane; ++pl) {
      for (int i = 0; i < 3; ++i) pn[3 * pl + i] = M.planes[pl].normal[i];
      poff[pl] = M.planes[pl].offset;
    }

    // 刚体坐标系里的几何和分组（窄相的剔除用，narrow.cu）。
    std::vector<float> vlf(3 * static_cast<size_t>(m.nvert));
    for (size_t i = 0; i < vlf.size(); ++i) vlf[i] = static_cast<float>(vlocal[i]);
    const ChunkSet fc = build_chunks(nb, bface0, bnface, fv, 3, vlocal, vradius);
    const ChunkSet ec = build_chunks(nb, bedge0, bnedge, ev, 2, vlocal, vradius);

    // 上传，并把指针填进 DModel。
    const auto up_i = [](DBuf<int>& buf, const std::vector<int>& v) {
      buf.upload(v);
      return buf.get();
    };
    const auto up_d = [](DBuf<double>& buf, const std::vector<double>& v) {
      buf.upload(v);
      return buf.get();
    };
    m.parent = up_i(i_parent, parent);
    m.bpos = up_d(d_bpos, bpos);
    m.bquat = up_d(d_bquat, bquat);
    m.bjnt = up_i(i_bjnt, bjnt);
    m.mass = up_d(d_mass, mass);
    m.ipos = up_d(d_ipos, ipos);
    m.inertia = up_d(d_inertia, inertia);
    m.S = up_d(d_S, S);
    m.radius = up_d(d_radius, radius);
    m.btree = up_i(i_btree, btree);
    m.bsub_end = up_i(i_bsub_end, sub_end);
    m.chain_dof = up_i(i_chain_dof, chain_dof);
    m.chain_n = up_i(i_chain_n, chain_n);
    m.bvert0 = up_i(i_bvert0, bvert0);
    m.bnvert = up_i(i_bnvert, bnvert);
    m.bedge0 = up_i(i_bedge0, bedge0);
    m.bnedge = up_i(i_bnedge, bnedge);
    m.bface0 = up_i(i_bface0, bface0);
    m.bnface = up_i(i_bnface, bnface);
    m.jtype = up_i(i_jtype, jtype);
    m.jqadr = up_i(i_jqadr, jqadr);
    m.jdadr = up_i(i_jdadr, jdadr);
    m.jbody = up_i(i_jbody, jbody);
    m.jlimited = up_i(i_jlimited, jlimited);
    m.jpos = up_d(d_jpos, jpos);
    m.jaxis = up_d(d_jaxis, jaxis);
    m.jref = up_d(d_jref, jref);
    m.jrange = up_d(d_jrange, jrange);
    m.dbody = up_i(i_dbody, dbody);
    m.drot = up_i(i_drot, drot);
    m.dtree = up_i(i_dtree, dtree);
    m.tdof0 = up_i(i_tdof0, tdof0);
    m.tndof = up_i(i_tndof, tndof);
    m.tbody0 = up_i(i_tbody0, tbody0);
    m.tnbody = up_i(i_tnbody, tnbody);
    m.vbody = up_i(i_vbody, vbody);
    m.vlocal = up_d(d_vlocal, vlocal);
    m.vradius = up_d(d_vradius, vradius);
    m.ev = up_i(i_ev, ev);
    m.ebody = up_i(i_ebody, ebody);
    m.fv = up_i(i_fv, fv);
    m.fbody = up_i(i_fbody, fbody);
    m.pn = up_d(d_pn, pn);
    m.poff = up_d(d_poff, poff);
    const auto up_f = [](DBuf<float>& buf, const std::vector<float>& v) {
      buf.upload(v);
      return buf.get();
    };
    m.vlf = up_f(f_vlf, vlf);
    m.flbox = up_f(f_flbox, fc.box);
    m.elbox = up_f(f_elbox, ec.box);
    m.blbox_f = up_f(f_blbox_f, fc.bbox);
    m.blbox_e = up_f(f_blbox_e, ec.bbox);
    m.fch_order = up_i(i_fch_order, fc.order);
    m.fch_range = up_i(i_fch_range, fc.range);
    m.fch_box = up_f(f_fch_box, fc.cbox);
    m.bfch = up_i(i_bfch, fc.bch);
    m.ech_order = up_i(i_ech_order, ec.order);
    m.ech_range = up_i(i_ech_range, ec.range);
    m.ech_box = up_f(f_ech_box, ec.cbox);
    m.bech = up_i(i_bech, ec.bch);

    // 工作区。
    for (FrameBufs* f : {&F, &F0, &Fp, &F1, &Fa, &Fb, &Ft}) f->reserve(m);
    const int nv = std::max(m.nv, 1), nq = std::max(m.nq, 1);
    for (DBuf<double>* b : {&q, &q0, &qprev, &qtrial}) b->reserve(nq);
    for (DBuf<double>* b : {&p, &g, &ge0, &gb, &scale, &rhs}) b->reserve(nv);
    Hk.reserve(std::max(m.ntree, 1) * m.TD * m.TD);
    P.reserve(2 * std::max(m.ntree, 1) * m.TD * m.TD);  // 逆 + 工作区（launch_precond_build）
    work.reserve(4 * nv + 18 * nb);  // PCG：r、z、p、Ap，加上力旋量形式的行用的工作区（launch_matvec）
    scal.reserve(16);
    tc.reserve(3 * nb);
    tA.reserve(9 * nb);
    A.reserve(nb);
    boxes.reserve(6 * nb);
    grow_work(1 << 16);
    xfa.reserve(3 * static_cast<size_t>(std::max(m.nvert, 1)));
    xfb.reserve(3 * static_cast<size_t>(std::max(m.nvert, 1)));
    bdisp.reserve(nb);
    pw_Wb.reserve(21 * static_cast<size_t>(nb));
    pw_n.reserve(1);
    grow_cands(1 << 16);
    cnt.reserve(4);
    grow_pairs(1024);
    grow_contacts(1024);
    grow_rows(3 * 1024);
  }

  // 容量也写死在录好的图里，所以改了容量（哪怕没有重新分配）也要让版本加一。
  void grow_pairs(int n) {
    pair_cap = n;
    ++g_buf_gen;
    pairs.reserve(2 * static_cast<size_t>(n));
    n_pxf.reserve(24 * static_cast<size_t>(n));
    n_ppad.reserve(n);
    n_pxm.reserve(n);
    n_pT.reserve(n);
  }
  void grow_work(int n) {
    work_cap = n;
    ++g_buf_gen;
    n_work.reserve(n);
  }
  // 工作项的个数在 cnt[3]。
  DNarrow narrow() const {
    DNarrow N;
    N.pxf = n_pxf.get();
    N.pT = n_pT.get();
    N.ppad = n_ppad.get();
    N.pxm = n_pxm.get();
    N.work = n_work.get();
    N.work_n = cnt.get() + 3;
    N.work_cap = work_cap;
    return N;
  }
  void grow_contacts(int n) {
    // 只扩接触的缓冲区。摩擦接触在步首冻结以后整步都要用，不能在这里跟着重新分配（重新分配会丢掉内容）。
    contact_cap = n;
    contacts.reserve(n);
    ++g_buf_gen;
  }
  void grow_cands(int n) {
    cand_cap = n;
    ++g_buf_gen;
    cand.reserve(n);
  }
  DCands cands() const { return DCands{cand.get(), cnt.get() + 2, cand_cap}; }
  // 行数每次组装都在变，按 1.5 倍扩容：每次只扩到刚好够用时，接触一多就要重新分配，cudaFree 会让整个设备同步。
  void grow_rows(int n) {
    if (n <= row_cap) return;
    n = std::max(n, row_cap + row_cap / 2);
    row_cap = n;
    ++g_buf_gen;
    row_body.reserve(2 * static_cast<size_t>(n));
    row_w.reserve(12 * static_cast<size_t>(n));
    row_D.reserve(n);
    pw_cross.reserve(n);
  }
  DRows rows_view() const { return DRows{row_body.get(), row_w.get(), row_D.get()}; }
  DPrecondWork precond_work() const { return DPrecondWork{pw_Wb.get(), pw_cross.get(), pw_n.get(), row_cap}; }

  // ---------- 计时 ----------

  using Clock = std::chrono::steady_clock;
  double ms_since(Clock::time_point t0) const {
    if (!timing) return 0;
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }

  // ---------- 基本操作 ----------

  void set_scalar(int i, double v) { launch_fill(scal.get() + i, v); }  // 不等 GPU
  double get_scalar(int i) {
    double v = 0;
    check(cudaMemcpy(&v, scal.get() + i, sizeof(double), cudaMemcpyDeviceToHost), "get");
    return v;
  }

  bool pairs_on() const { return m.nedge > 0 || m.nface > 0; }

  // 接触检测（find_contacts，dmax = dhat）：结果放在 contacts 里，返回个数。
  // 刚体对、接触、候选、工作项的个数都留在显存里，最后一次读回（cnt[0..3]）；哪个放不下就扩容重来。
  int detect(const FrameBufs& Fr, double dmax) {
    launch_body_boxes(m, Fr.X.get(), boxes.get());
    int h[4];
    for (;;) {
      if (pairs_on()) launch_body_pairs(m, boxes.get(), dmax, nullptr, 0.0, pairs.get(), pair_cap, cnt.get());
      else cudaMemsetAsync(cnt.get(), 0, sizeof(int));
      launch_detect(m, Fr.view(), boxes.get(), pairs.get(), pairs_on() ? pair_cap : 0, dmax, narrow(), cands(),
                    contacts.get(), contact_cap, cnt.get(), h);
      if (h[0] > pair_cap) grow_pairs(2 * h[0]);
      else if (h[3] > work_cap) grow_work(2 * h[3]);
      else if (h[2] > cand_cap) grow_cands(2 * h[2]);
      else if (h[1] > contact_cap) grow_contacts(2 * h[1]);
      else break;
    }
    if (pairs_on()) {
      ++timers.pair_calls;
      timers.pair_sum += h[0];
    }
    return h[1];
  }

  void run_fk(const double* qd, const FrameBufs& Fr) { launch_fk(m, qd, Fr.view()); }

  // 接触里最小的距离；没有接触时 +无穷。
  double contact_dmin(int n) {
    if (n == 0) return std::numeric_limits<double>::infinity();
    set_scalar(4, std::numeric_limits<double>::infinity());
    launch_min_distance(contacts.get(), n, scal.get() + 4);
    return get_scalar(4);
  }

  // 步首（CPU 版 step 的“准备”）：倒推 q_{n-1}、惯性目标、步首的惯性梯度、滞后摩擦。
  void prepare(const StepParams& Pm, const VectorXd& q0h, const VectorXd& v0h, double kappa) {
    const auto t0 = Clock::now();
    const double h = Pm.h;
    q0.upload(q0h.data(), m.nq);
    rhs.upload(v0h.data(), m.nv);  // 借 rhs 放 v0
    launch_integrate(m, q0.get(), rhs.get(), -h, qprev.get());
    run_fk(q0.get(), F0);
    run_fk(qprev.get(), Fp);
    launch_targets(m, F0.view(), Fp.view(), h, tc.get(), tA.get());
    cudaMemsetAsync(ge0.get(), 0, sizeof(double) * m.nv);
    launch_inertia_gradient(m, F0.view(), tc.get(), tA.get(), h, ge0.get());
    nf = 0;
    if (Pm.mu > 0 && kappa > 0) {
      const int n0 = detect(F0, Pm.dhat);
      if (static_cast<size_t>(n0) > fric.size()) fric.reserve(std::max<size_t>(n0, fric.size() + fric.size() / 2));  // 冻结之前扩容
      nf = launch_friction_lag(m, F0.X.get(), contacts.get(), n0, kappa, Pm.dhat, fric.get(), cnt.get());
    }
    timers.prepare += ms_since(t0);
  }

  // 在当前点（F 和 contacts 里的 nc 个接触）组装 Newton 系统：g、Hk、行，返回能量 e0。
  void assemble(const StepParams& Pm, double kappa) {
    const auto t0 = Clock::now();
    const double h = Pm.h;
    cudaMemsetAsync(scal.get(), 0, sizeof(double));
    cudaMemsetAsync(g.get(), 0, sizeof(double) * m.nv);
    launch_inertia_hessian(m, F.view(), h, Hk.get());
    launch_inertia_energy(m, F.view(), tc.get(), tA.get(), h, scal.get());
    launch_inertia_gradient(m, F.view(), tc.get(), tA.get(), h, g.get());
    grow_rows(nc + 2 * nf);
    launch_contact_terms(m, F.view(), q.get(), contacts.get(), nc, kappa, Pm.dhat, Pm.curv_scale, scal.get(), g.get(),
                         Hk.get(), rows_view());
    launch_friction_terms(m, F.view(), fric.get(), nf, h, Pm.mu, Pm.eps_v, scal.get(), g.get(), rows_view(), nc);
    launch_limits(m, q.get(), Pm.limit_k, Pm.limit_dhat, scal.get(), g.get(), Hk.get());
    nrow = nc + 2 * nf;
    timers.contacts += nc;
    timers.rows += nrow;
    timers.assemble += ms_since(t0);
    // 能量在 scal[0]，不在这里读：和 PCG、收敛判断的结果一起读回（step 的第 2 步）。
  }

  // 线搜索用的能量：在 qd 处做 FK、检测接触，再把各项能量加起来（会覆盖 contacts）。
  double energy_at(const StepParams& Pm, const double* qd, double kappa) {
    ++timers.energy_evals;
    run_fk(qd, Ft);
    const int n = detect(Ft, Pm.dhat);
    cudaMemsetAsync(scal.get(), 0, sizeof(double));
    launch_inertia_energy(m, Ft.view(), tc.get(), tA.get(), Pm.h, scal.get());
    launch_contact_terms(m, Ft.view(), qd, contacts.get(), n, kappa, Pm.dhat, 0.0, scal.get(), nullptr, nullptr, DRows{});
    launch_friction_terms(m, Ft.view(), fric.get(), nf, Pm.h, Pm.mu, Pm.eps_v, scal.get(), nullptr, DRows{}, 0);
    launch_limits(m, qd, Pm.limit_k, Pm.limit_dhat, scal.get(), nullptr, nullptr);
    // 顺便算接触里最小的距离（kappa 调整要用接受的那个点的，而接受的点就是最后一次试探的点），和能量一起读回。
    set_scalar(4, std::numeric_limits<double>::infinity());
    launch_min_distance(contacts.get(), n, scal.get() + 4);
    double out[5];
    check(cudaMemcpy(out, scal.get(), sizeof(out), cudaMemcpyDeviceToHost), "energy read");
    last_eval_n = n;
    last_eval_dmin = out[4];
    return out[0];
  }

  // PCG 解 H x = rhs_d（rhs_d、x 都在显存）。
  // 迭代次数：主机循环版直接返回；协作内核版不等，返回 -1，次数在 scal[12]（defer = false 时这里读回）。
  int pcg_solve(const double* rhs_d, double* x, double tol, int max_iter, bool defer = false) {
    const auto t0 = Clock::now();
    launch_precond_build(m, F.view(), Hk.get(), rows_view(), nrow, P.get(), scale.get(), precond_work());
    bool conv = false;
    int it = -1;
    if (coop) {
      pcg_acc.reserve(2 * static_cast<size_t>(max_iter) + 4);
      if (run_pcg_coop(m, F.view(), Hk.get(), rows_view(), nrow, P.get(), rhs_d, x, tol, max_iter, work.get(),
                       pcg_acc.get(), scal.get() + 12) == 0) {
        timers.solve += ms_since(t0);
        if (defer) return -1;
        it = static_cast<int>(get_scalar(12));
        timers.pcg += it;
        return it;
      }
    }
    // 没用协作内核（关掉了，或者树太大放不进共享内存）
    it = run_pcg(m, F.view(), Hk.get(), rows_view(), nrow, P.get(), rhs_d, x, tol, max_iter, work.get(), scal.get() + 8,
                 &conv);
    timers.solve += ms_since(t0);
    timers.pcg += it;
    return it;
  }

  // CCD（curved_ccd + limit_ccd），从 F（q 处的 FK）出发沿 p。
  double ccd(const StepParams& Pm, const double* qd, const double* pd) {
    const auto t0 = Clock::now();
    launch_accel_bounds(m, qd, F.view(), pd, Pm.tight_bound, A.get());
    std::vector<double> Ah(m.nbody);
    A.download(Ah.data(), m.nbody);
    const double A_max = Ah.empty() ? 0.0 : *std::max_element(Ah.begin(), Ah.end());
    const bool pairs_on = m.nedge > 0 || m.nface > 0;
    double ds = A_max > 0 ? std::min(1.0, std::sqrt(8 * Pm.dhat / A_max)) : 1.0;
    double s0 = 0, result = -1, limit_t = 1.0;
    const FrameBufs* Fs = &F;  // 段起点的 FK
    // 候选的窗口 [w0, w1]（笔记第 13 节）：最近一次求候选的那一段。落在窗口里的段（缩短重做的、缩短以后接着往前走的）
    // 不再求刚体对、不再剔除，直接拿窗口的候选做复核和 ACCD。w1 < 0 表示还没有窗口。
    double w0 = 0, w1 = -1;
    // 限位的 CCD 和分段无关，先发出去；结果在 scal[3]，跟着每一段的结果一起读回。
    set_scalar(3, 1.0);
    launch_limit_ccd(m, qd, pd, Pm.c, scal.get() + 3);
    for (int guard = 0; guard < 10000 && s0 < 1.0; ++guard) {
      ++timers.ccd_segments;
      const double s1 = std::min(1.0, s0 + ds);
      const double len = s1 - s0;
      const bool fresh = !reuse || !(s0 >= w0 && s1 <= w1);
      if (fresh) {
        ++timers.ccd_windows;
        w0 = s0;
        w1 = s1;
      }
      launch_integrate(m, qd, pd, s1, qtrial.get());
      launch_fk(m, qtrial.get(), Fb.view());
      // scal[1] 截断位置，scal[2] 需要的段长，scal[9..11] 候选、刚体对、工作项的个数。这一段只在最后读回一次。
      const double* Xa = Fs->X.get();
      launch_ccd_init(scal.get() + 1, scal.get() + 2, scal.get() + 9, len);
      launch_ground_ccd(m, Xa, Fb.X.get(), A.get(), len, Pm.c, scal.get() + 1, scal.get() + 2);
      if (pairs_on) {
        launch_swept_boxes(m, Xa, Fb.X.get(), boxes.get());  // 复核用这一段的盒子；新窗口的剔除也用它
        if (fresh) {
          launch_body_pairs(m, boxes.get(), 0.0, A.get(), reuse ? kWindowPairLen * len : len, pairs.get(), pair_cap,
                            cnt.get());
          launch_body_disp(m, Xa, Fb.X.get(), bdisp.get());
          launch_to_float(Xa, 3 * m.nvert, xfa.get());
          launch_to_float(Fb.X.get(), 3 * m.nvert, xfb.get());
          DNarrow N = narrow();
          N.xa = xfa.get();
          N.xb = xfb.get();
          launch_ccd_cull(m, Fs->view(), boxes.get(), bdisp.get(), A.get(), len, Pm.c, pairs.get(), cnt.get(), pair_cap,
                          N, cands(), reuse);
        }
        launch_ccd_exact(m, Xa, Fb.X.get(), boxes.get(), A.get(), len, Pm.c, cnt.get(), pair_cap, narrow(), cands(),
                         scal.get() + 1, scal.get() + 2, scal.get() + 9);
      }
      double out[11];  // scal[1..11]
      check(cudaMemcpy(out, scal.get() + 1, 11 * sizeof(double), cudaMemcpyDeviceToHost), "ccd read");
      limit_t = out[2];
      if (out[9] > pair_cap || out[10] > work_cap || out[8] > cand_cap) {
        // 刚体对、工作列表或候选数组放不下：扩容，这一段重做（扩容丢了窗口的候选，重新求）
        if (out[9] > pair_cap) grow_pairs(static_cast<int>(2 * out[9]));
        if (out[10] > work_cap) grow_work(static_cast<int>(2 * out[10]));
        if (out[8] > cand_cap) grow_cands(static_cast<int>(2 * out[8]));
        w1 = -1;
        continue;
      }
      if (pairs_on) {
        ++timers.pair_calls;
        timers.pair_sum += static_cast<long>(out[9]);
      }
      const double t_min = out[0], ds_need = out[1];
      if (ds_need < len) {
        ds = std::min(0.5 * len, ds_need);
        continue;
      }
      if (t_min < 1.0) {
        result = s0 + t_min * len;
        break;
      }
      s0 = s1;
      Fa.swap(Fb);  // 这一段的终点是下一段的起点（Fb 下一段会被 FK 覆盖）
      Fs = &Fa;
    }
    if (result < 0) result = std::min(s0, 1.0);
    result = std::min(result, limit_t);
    timers.ccd += ms_since(t0);
    return result;
  }

  // kappa 的初值（initial_kappa）：g_b 是 kappa = 1 时的接触梯度，g_e 是步首的惯性梯度。
  double initial_kappa(const StepParams& Pm) {
    const KappaRange range = kappa_range(model, Pm.dhat);
    cudaMemsetAsync(gb.get(), 0, sizeof(double) * m.nv);
    launch_contact_terms(m, F.view(), q.get(), contacts.get(), nc, 1.0, Pm.dhat, 0.0, nullptr, gb.get(), nullptr,
                         DRows{});
    cudaMemsetAsync(scal.get() + 5, 0, 2 * sizeof(double));
    launch_dot(gb.get(), ge0.get(), m.nv, scal.get() + 5);
    launch_dot(gb.get(), gb.get(), m.nv, scal.get() + 6);
    const double gbge = get_scalar(5), gb2 = get_scalar(6);
    double kappa = range.ref;
    if (gb2 > 0) {
      const double balance = -gbge / gb2;
      if (balance > 0) kappa = balance;
    }
    return std::min(std::max(kappa, range.min), range.max);
  }

  // ---------- CUDA graph 版的 Newton 迭代（笔记第 12 节）----------
  //
  // 一个 Newton 迭代录成一张图：组装、预条件器、协作 PCG、收敛判断，然后两个条件节点（WHILE）：CCD 的分段循环、
  // 线搜索的回溯循环，最后接受试探点。循环里的判断是 iter.cu 的小内核，和下面 step() 里的那几行一一对应。
  // 图录好以后内核参数固定，会变的数都在显存里的 DIter（d_it）；主机每个迭代只发一次图、读回一次 DIter。
  // 任何缓冲区放不下时这次迭代作废（不接受试探点），主机扩容、重新检测当前点的接触，重录以后再发一次。

  void drop_graph() {
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
    graph_exec = nullptr;
    graph = nullptr;
    graph_key = GraphKey();
  }

  // f() 里发出去的操作捕获到图 g 里，接在 deps 后面；返回时 deps 是捕获出来的最后几个节点。
  template <class Fn>
  void capture_into(cudaGraph_t gr, std::vector<cudaGraphNode_t>& deps, Fn&& f) {
    check(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, gr, deps.empty() ? nullptr : deps.data(), nullptr,
                                        deps.size(), cudaStreamCaptureModeThreadLocal),
          "begin capture");
    try {
      f();
    } catch (...) {
      cudaGraph_t dummy = gr;
      cudaStreamEndCapture(cudaStreamPerThread, &dummy);
      throw;
    }
    cudaStreamCaptureStatus st;
    const cudaGraphNode_t* d = nullptr;
    size_t nd = 0;
    check(cudaStreamGetCaptureInfo(cudaStreamPerThread, &st, nullptr, nullptr, &d, &nd), "capture info");
    std::vector<cudaGraphNode_t> tail(d, d + nd);
    cudaGraph_t out = gr;
    check(cudaStreamEndCapture(cudaStreamPerThread, &out), "end capture");
    deps = std::move(tail);
  }

  // 条件节点接在 deps 后面（句柄 h；WHILE 是循环，IF 是 h 不为 0 时跑一次），返回循环体的图；deps 变成这个节点。
  cudaGraph_t add_cond(cudaGraph_t gr, std::vector<cudaGraphNode_t>& deps, cudaGraphConditionalHandle h,
                        cudaGraphConditionalNodeType type = cudaGraphCondTypeWhile) {
    cudaGraphNodeParams np = {};
    np.type = cudaGraphNodeTypeConditional;
    np.conditional.handle = h;
    np.conditional.type = type;
    np.conditional.size = 1;
    cudaGraphNode_t node;
    check(cudaGraphAddNode(&node, gr, deps.empty() ? nullptr : deps.data(), deps.size(), &np), "while node");
    deps = {node};
    return np.conditional.phGraph_out[0];
  }

  void copy_frames(const FrameBufs& from, FrameBufs& to) {
    cudaMemcpyAsync(to.xpos.get(), from.xpos.get(), sizeof(double) * 3 * m.nbody, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(to.xquat.get(), from.xquat.get(), sizeof(double) * 4 * m.nbody, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(to.daxis.get(), from.daxis.get(), sizeof(double) * 3 * m.nv, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(to.danchor.get(), from.danchor.get(), sizeof(double) * 3 * m.nv, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(to.X.get(), from.X.get(), sizeof(double) * 3 * m.nvert, cudaMemcpyDeviceToDevice);
  }

  // 录一个 Newton 迭代。
  void record_graph(const StepParams& Pm) {
    DIter* it = d_it.get();
    double* sc = scal.get();
    const double h = Pm.h;
    const DCount nc_d(&it->nc, contact_cap), nf_d(&it->nf, static_cast<int>(fric.size()));
    const DCount nrow_d(&it->nrow, row_cap);
    const DScalar kappa_d(0, &it->kappa), s_d(0, &it->s), len_d(0, &it->len);
    const bool fr = Pm.mu > 0;
    check(cudaGraphCreate(&graph, 0), "graph create");
    cudaGraphConditionalHandle hc, hl;
    check(cudaGraphConditionalHandleCreate(&hc, graph, 0, 0), "handle");
    check(cudaGraphConditionalHandleCreate(&hl, graph, 0, 0), "handle");
    std::vector<cudaGraphNode_t> deps;

    // 第 1、2 步：组装（assemble）、预条件器和 PCG（pcg_solve）、收敛判断；第 3 步的开头。
    capture_into(graph, deps, [&] {
      launch_it_begin(it);
      cudaMemsetAsync(sc, 0, sizeof(double));
      cudaMemsetAsync(g.get(), 0, sizeof(double) * m.nv);
      launch_inertia_hessian(m, F.view(), h, Hk.get());
      launch_inertia_energy(m, F.view(), tc.get(), tA.get(), h, sc);
      launch_inertia_gradient(m, F.view(), tc.get(), tA.get(), h, g.get());
      launch_contact_terms(m, F.view(), q.get(), contacts.get(), nc_d, kappa_d, Pm.dhat, Pm.curv_scale, sc, g.get(),
                           Hk.get(), rows_view());
      if (fr) launch_friction_terms(m, F.view(), fric.get(), nf_d, h, Pm.mu, Pm.eps_v, sc, g.get(), rows_view(), nc_d);
      launch_limits(m, q.get(), Pm.limit_k, Pm.limit_dhat, sc, g.get(), Hk.get());
      launch_precond_build(m, F.view(), Hk.get(), rows_view(), nrow_d, P.get(), scale.get(), precond_work());
      if (run_pcg_coop(m, F.view(), Hk.get(), rows_view(), nrow_d, P.get(), g.get(), p.get(), Pm.pcg_tol, Pm.pcg_max,
                       work.get(), pcg_acc.get(), sc + 12) != 0)
        throw std::runtime_error("cooperative PCG does not fit");
      launch_scale(p.get(), m.nv, -1.0);
      launch_integrate(m, q.get(), p.get(), 1.0, qtrial.get());
      launch_fk(m, qtrial.get(), F1.view());
      launch_fill(sc + 7, 0.0);
      launch_max_displacement(F.X.get(), F1.X.get(), m.nvert, sc + 7);
      launch_it_converge(it, sc, h, Pm.tol);
      launch_accel_bounds(m, q.get(), F.view(), p.get(), Pm.tight_bound, A.get());
      launch_ccd_setup(it, A.get(), m.nbody, Pm.dhat, sc, hc);
      launch_limit_ccd(m, q.get(), p.get(), Pm.c, sc + 3);
      copy_frames(F, Fa);  // 第一段从当前点开始
    });

    // 第 3 步：CCD 的分段（ccd()）。
    // 求候选（刚体对和剔除）放在一个 IF 条件节点里：k_ccd_seg_begin 判断这一段落不落在候选的窗口里，
    // 不落在里面才打开（和 ccd() 里的 fresh 一样）。
    cudaGraph_t body = add_cond(graph, deps, hc);
    cudaGraphConditionalHandle hf = 0;
    if (pairs_on()) check(cudaGraphConditionalHandleCreate(&hf, body, 0, 0), "handle");
    std::vector<cudaGraphNode_t> bd;
    capture_into(body, bd, [&] {
      launch_ccd_seg_begin(it, reuse, kWindowPairLen, hf);
      launch_integrate(m, q.get(), p.get(), s_d, qtrial.get());
      launch_fk(m, qtrial.get(), Fb.view());
      launch_ccd_init(sc + 1, sc + 2, sc + 9, len_d);
      launch_ground_ccd(m, Fa.X.get(), Fb.X.get(), A.get(), len_d, Pm.c, sc + 1, sc + 2);
      if (pairs_on()) launch_swept_boxes(m, Fa.X.get(), Fb.X.get(), boxes.get());
    });
    if (pairs_on()) {
      cudaGraph_t cull = add_cond(body, bd, hf, cudaGraphCondTypeIf);
      std::vector<cudaGraphNode_t> cd;
      const DScalar plen_d(0, reuse ? &it->plen : &it->len);
      capture_into(cull, cd, [&] {
        launch_body_pairs(m, boxes.get(), 0.0, A.get(), plen_d, pairs.get(), pair_cap, cnt.get());
        launch_body_disp(m, Fa.X.get(), Fb.X.get(), bdisp.get());
        launch_to_float(Fa.X.get(), 3 * m.nvert, xfa.get());
        launch_to_float(Fb.X.get(), 3 * m.nvert, xfb.get());
        DNarrow N = narrow();
        N.xa = xfa.get();
        N.xb = xfb.get();
        launch_ccd_cull(m, Fa.view(), boxes.get(), bdisp.get(), A.get(), len_d, Pm.c, pairs.get(), cnt.get(), pair_cap,
                        N, cands(), reuse);
      });
    }
    capture_into(body, bd, [&] {
      if (pairs_on()) {
        launch_ccd_exact(m, Fa.X.get(), Fb.X.get(), boxes.get(), A.get(), len_d, Pm.c, cnt.get(), pair_cap, narrow(),
                         cands(), sc + 1, sc + 2, sc + 9);
      } else {
        launch_fill(sc + 9, 0.0);
        launch_fill(sc + 10, 0.0);
        launch_fill(sc + 11, 0.0);
      }
      launch_ccd_seg_end(it, sc, pair_cap, work_cap, cand_cap, hc);
      launch_copy_frames_if(it, Fb.view(), Fa.view(), m.nbody, m.nv, m.nvert);
    });

    // 第 4 步：回溯线搜索（energy_at）。
    capture_into(graph, deps, [&] { launch_ls_setup(it, sc, hl); });
    body = add_cond(graph, deps, hl);
    bd.clear();
    capture_into(body, bd, [&] {
      launch_integrate(m, q.get(), p.get(), s_d, qtrial.get());
      launch_fk(m, qtrial.get(), Ft.view());
      launch_body_boxes(m, Ft.X.get(), boxes.get());
      if (pairs_on()) launch_body_pairs(m, boxes.get(), Pm.dhat, nullptr, 0.0, pairs.get(), pair_cap, cnt.get());
      else cudaMemsetAsync(cnt.get(), 0, sizeof(int));
      launch_detect_async(m, Ft.view(), boxes.get(), pairs.get(), pairs_on() ? pair_cap : 0, Pm.dhat, narrow(),
                          cands(), contacts.get(), contact_cap, cnt.get());
      const DCount n_d(cnt.get() + 1, contact_cap);
      cudaMemsetAsync(sc, 0, sizeof(double));
      launch_inertia_energy(m, Ft.view(), tc.get(), tA.get(), h, sc);
      launch_contact_terms(m, Ft.view(), qtrial.get(), contacts.get(), n_d, kappa_d, Pm.dhat, 0.0, sc, nullptr,
                           nullptr, DRows{});
      if (fr) launch_friction_terms(m, Ft.view(), fric.get(), nf_d, h, Pm.mu, Pm.eps_v, sc, nullptr, DRows{}, 0);
      launch_limits(m, qtrial.get(), Pm.limit_k, Pm.limit_dhat, sc, nullptr, nullptr);
      launch_fill(sc + 4, std::numeric_limits<double>::infinity());
      launch_min_distance(contacts.get(), n_d, sc + 4);
      launch_ls_end(it, sc, cnt.get(), pair_cap, contact_cap, cand_cap, work_cap, hl);
    });

    // 第 5 步：接受最后一次试探的点。
    capture_into(graph, deps, [&] {
      launch_it_accept(it, qtrial.get(), q.get(), m.nq, Ft.view(), F.view(), m.nbody, m.nv, m.nvert, cnt.get());
    });
    check(cudaGraphInstantiate(&graph_exec, graph, 0), "graph instantiate");
  }

  GraphKey key_for(const StepParams& Pm) const {
    GraphKey k;
    k.gen = g_buf_gen;
    k.h = Pm.h;
    k.dhat = Pm.dhat;
    k.mu = Pm.mu;
    k.eps_v = Pm.eps_v;
    k.curv = Pm.curv_scale;
    k.limit_k = Pm.limit_k;
    k.limit_dhat = Pm.limit_dhat;
    k.c = Pm.c;
    k.tol = Pm.tol;
    k.pcg_tol = Pm.pcg_tol;
    k.pcg_max = Pm.pcg_max;
    k.tight = Pm.tight_bound;
    k.reuse = reuse;
    return k;
  }

  // 图能不能用：录过而且没过期就直接用，过期了重录。录图失败时返回 false（以后都用逐个发射）。
  bool ensure_graph(const StepParams& Pm) {
    pcg_acc.reserve(2 * static_cast<size_t>(Pm.pcg_max) + 4);  // 录图时不能分配显存
    const GraphKey k = key_for(Pm);
    if (graph_exec && k == graph_key) return true;
    drop_graph();
    try {
      record_graph(Pm);
    } catch (const std::exception&) {
      drop_graph();
      cudaGetLastError();
      graph_failed = true;
      return false;
    }
    graph_key = key_for(Pm);  // 录图时不会分配显存，版本应该没变；保险起见再取一次
    return true;
  }

  bool use_graph(const StepParams& Pm, const std::vector<NewtonIterate>* trace) const {
    (void)Pm;
    return graphs && !graph_failed && coop && trace == nullptr && m.nvert > 0;
  }

  // 一次图版的 Newton 迭代：写 kappa、nc、nf，发图，读回 DIter。
  const DIter& run_graph_iteration(double kappa) {
    h_it->kappa = kappa;
    h_it->nc = nc;
    h_it->nf = nf;
    check(cudaMemcpyAsync(d_it.get(), h_it, offsetof(DIter, nrow), cudaMemcpyHostToDevice), "iter header");
    check(cudaGraphLaunch(graph_exec, cudaStreamPerThread), "graph launch");
    check(cudaMemcpyAsync(h_it, d_it.get(), sizeof(DIter), cudaMemcpyDeviceToHost), "iter read");
    check(cudaStreamSynchronize(cudaStreamPerThread), "iter sync");
    return *h_it;
  }

  // 图版的 Newton 循环（和下面 step() 里的循环一样，只是每个迭代的第 1–5 步是一次 run_graph_iteration）。
  // 第一次录图就失败时返回 false，step() 改用逐个发射的循环。
  bool newton_graph(const StepParams& Pm, SolverState* state, const KappaRange& krange, int& newton, int& pcg_total,
                    bool& semi_stopped) {
    double beta = 1.0;
    int redo = 0;
    for (int it = 1; it <= Pm.max_newton;) {
      newton = it;
      // 第 0 步：第一次出现接触时初始化 kappa（逐个发射，很少走到）。
      if (nc > 0 && state->kappa == 0) {
        const auto t0 = Clock::now();
        state->kappa = initial_kappa(Pm);
        state->dmin_prev = contact_dmin(nc);
        timers.assemble += ms_since(t0);
      }
      grow_rows(contact_cap + 2 * nf);  // 行数不超过 nc + 2 nf，nc 不超过 contact_cap：图里不用再扩容
      if (!ensure_graph(Pm)) {
        if (it == 1) return false;
        throw std::runtime_error("CUDA graph could not be recorded in the middle of a step");
      }
      const auto t0 = Clock::now();
      const DIter r = run_graph_iteration(state->kappa);
      timers.graph += ms_since(t0);
      if (r.status == kIterOverflow) {
        // 某个缓冲区放不下：这次迭代没有接受任何东西。扩容，重新检测当前点的接触（线搜索把它们覆盖了），重来。
        if (r.need[0] > pair_cap) grow_pairs(2 * r.need[0]);
        if (r.need[1] > contact_cap) grow_contacts(2 * r.need[1]);
        if (r.need[2] > cand_cap) grow_cands(2 * r.need[2]);
        if (r.need[3] > work_cap) grow_work(2 * r.need[3]);
        nc = detect(F, Pm.dhat);
        if (++redo > 50) throw std::runtime_error("CUDA graph iteration keeps overflowing");
        continue;
      }
      redo = 0;
      timers.contacts += nc;
      timers.rows += r.nrow;
      timers.pcg += r.pcg;
      timers.energy_evals += r.trials;
      timers.ccd_segments += r.segs;
      timers.ccd_windows += r.windows;
      pcg_total += r.pcg;
      if (r.status == kIterConverged) break;
      nc = r.nc;
      // 第 6 步：kappa 调整（接受的点上的最小距离）。
      if (Pm.adapt_kappa && state->kappa > 0) {
        const double dmin = r.dmin;
        if (dmin < 1e-3 * Pm.dhat && dmin < state->dmin_prev) {
          state->kappa = std::min(2 * state->kappa, krange.max);
        } else if (dmin > 1e-2 * Pm.dhat) {
          state->kappa = std::max(0.5 * state->kappa, krange.min);
        }
        state->dmin_prev = dmin;
      }
      // 第 7 步：半隐式提前停止。
      if (Pm.semi_implicit && it >= Pm.semi_kmin) {
        beta *= (1 - r.alpha_max);
        if (beta <= Pm.semi_beta_tol && r.alpha == r.alpha_max) {
          semi_stopped = true;
          break;
        }
      }
      ++it;
    }
    return true;
  }

  // ---------- 一个时间步：和 src/step.cc 的 step() 一行一行对应 ----------

  StepResult step(const StepParams& Pm, const VectorXd& q0h, const VectorXd& v0h, SolverState* state,
                  std::vector<NewtonIterate>* trace) {
    const auto t_step = Clock::now();
    const double h = Pm.h;
    SolverState local;
    if (state == nullptr) state = &local;
    const KappaRange krange = kappa_range(model, Pm.dhat);

    prepare(Pm, q0h, v0h, state->kappa);
    auto t_up = Clock::now();
    check(cudaMemcpy(q.get(), q0.get(), sizeof(double) * m.nq, cudaMemcpyDeviceToDevice), "q = q0");
    run_fk(q.get(), F);
    nc = detect(F, Pm.dhat);
    timers.update += ms_since(t_up);

    int newton = 0, pcg_total = 0;
    bool semi_stopped = false;
    double beta = 1.0;
    const bool graph_done =
        use_graph(Pm, trace) && newton_graph(Pm, state, krange, newton, pcg_total, semi_stopped);
    last_graph = graph_done;
    for (int it = 1; !graph_done && it <= Pm.max_newton; ++it) {
      newton = it;
      // 第 0 步：第一次出现接触时初始化 kappa。
      if (nc > 0 && state->kappa == 0) {
        const auto t0 = Clock::now();
        state->kappa = initial_kappa(Pm);
        state->dmin_prev = contact_dmin(nc);
        timers.assemble += ms_since(t0);
      }
      const double kappa = state->kappa;

      // 第 1 步：组装，PCG 解 H p = -g。
      const double dmin_here = trace ? contact_dmin(nc) : 0.0;
      assemble(Pm, kappa);
      int iters = pcg_solve(g.get(), p.get(), Pm.pcg_tol, Pm.pcg_max, true);
      launch_scale(p.get(), m.nv, -1.0);

      // 第 2 步：收敛判断（整步里顶点的最大位移 / h）。组装的能量 scal[0]、最大位移 scal[7]、
      // PCG 的迭代次数 scal[12] 到这里才一起读回：从组装到这里 GPU 不用停下来等主机。
      const auto t_conv = Clock::now();
      launch_integrate(m, q.get(), p.get(), 1.0, qtrial.get());
      launch_fk(m, qtrial.get(), F1.view());
      set_scalar(7, 0.0);
      if (m.nvert > 0) launch_max_displacement(F.X.get(), F1.X.get(), m.nvert, scal.get() + 7);
      double rb[13];
      check(cudaMemcpy(rb, scal.get(), sizeof(rb), cudaMemcpyDeviceToHost), "newton read");
      const double e0 = rb[0];
      if (iters < 0) {
        iters = static_cast<int>(rb[12]);
        timers.pcg += iters;
      }
      pcg_total += iters;
      double size;
      if (m.nvert > 0) {
        size = rb[7] / h;
      } else {
        VectorXd ph(m.nv);
        p.download(ph.data(), m.nv);
        size = host_max(ph) / h;
      }
      timers.converge += ms_since(t_conv);
      if (size < Pm.tol) break;

      // 第 3 步：CCD。
      const double alpha_max = ccd(Pm, q.get(), p.get());

      // 第 4 步：回溯线搜索。
      const auto t_ls = Clock::now();
      double alpha = alpha_max;
      for (;;) {
        launch_integrate(m, q.get(), p.get(), alpha, qtrial.get());
        if (!(energy_at(Pm, qtrial.get(), kappa) > e0 && alpha > 1e-12)) break;
        alpha *= 0.5;
      }
      timers.line_search += ms_since(t_ls);

      // 第 5 步：记录，更新。下一次迭代、kappa 调整都要新位置上的 FK 和接触。线搜索最后一次试探的正是
      // 接受的 alpha：qtrial = q + alpha p，Ft 是它的 FK，contacts 里是它的 last_eval_n 个接触。
      // 所以直接接过来（交换 F 和 Ft 的指针），不用再做一遍 FK 和检测。CPU 版重新算，结果一样。
      if (trace) trace->push_back({e0, size, alpha_max, alpha, dmin_here, iters});
      t_up = Clock::now();
      check(cudaMemcpyAsync(q.get(), qtrial.get(), sizeof(double) * m.nq, cudaMemcpyDeviceToDevice), "q update");
      F.swap(Ft);
      nc = last_eval_n;
      timers.update += ms_since(t_up);

      // 第 6 步：kappa 调整。
      if (Pm.adapt_kappa && state->kappa > 0) {
        const double dmin = last_eval_dmin;  // 接受的点上的最小距离（energy_at 顺便算的）
        if (dmin < 1e-3 * Pm.dhat && dmin < state->dmin_prev) {
          state->kappa = std::min(2 * state->kappa, krange.max);
        } else if (dmin > 1e-2 * Pm.dhat) {
          state->kappa = std::max(0.5 * state->kappa, krange.min);
        }
        state->dmin_prev = dmin;
      }

      // 第 7 步：半隐式提前停止。
      if (Pm.semi_implicit && it >= Pm.semi_kmin) {
        beta *= (1 - alpha_max);
        if (beta <= Pm.semi_beta_tol && alpha == alpha_max) {
          semi_stopped = true;
          break;
        }
      }
    }

    StepResult r;
    VectorXd qh(m.nq);
    q.download(qh.data(), m.nq);
    r.q = qh;
    r.v = differentiate(model, q0h, qh) / h;
    r.newton = newton;
    r.pcg = pcg_total;
    r.semi_stopped = semi_stopped;
    r.kappa = state->kappa;
    r.min_distance = contact_dmin(nc);
    r.iterate_min_distance = std::numeric_limits<double>::infinity();
    r.min_limit_distance = min_limit_distance(model, qh);
    timers.newton += newton;
    ++timers.steps;
    timers.total += ms_since(t_step);
    return r;
  }
};

Stepper::Stepper(const Model& model) : impl_(std::make_unique<Impl>(model)) {}
Stepper::~Stepper() = default;

StepResult Stepper::step(const StepParams& P, const VectorXd& q0, const VectorXd& v0, SolverState* state,
                         std::vector<NewtonIterate>* trace) {
  return impl_->step(P, q0, v0, state, trace);
}

const Stepper::Timers& Stepper::timers() const { return impl_->timers; }
bool Stepper::set_cooperative_pcg(bool on) {
  impl_->coop = on && pcg_coop_supported();
  return impl_->coop;
}
void Stepper::reset_timers() { impl_->timers = Timers(); }
void Stepper::set_graphs(bool on) { impl_->graphs = on; }
void Stepper::set_ccd_reuse(bool on) { impl_->reuse = on; }
bool Stepper::used_graphs() const { return impl_->last_graph; }
void Stepper::set_capacities(int pairs, int contacts, int candidates, int work) {
  impl_->grow_pairs(pairs);
  impl_->grow_contacts(contacts);
  impl_->grow_cands(candidates);
  impl_->grow_work(work);
}
void Stepper::set_timing(bool on) { impl_->timing = on; }

Frames Stepper::fk(const VectorXd& qh) {
  Impl& I = *impl_;
  I.q.upload(qh.data(), I.m.nq);
  I.run_fk(I.q.get(), I.F);
  Frames out;
  std::vector<double> xpos(3 * I.m.nbody), xquat(4 * I.m.nbody), ax(3 * I.m.nv), an(3 * I.m.nv);
  I.F.xpos.download(xpos.data(), xpos.size());
  I.F.xquat.download(xquat.data(), xquat.size());
  I.F.daxis.download(ax.data(), ax.size());
  I.F.danchor.download(an.data(), an.size());
  for (int b = 0; b < I.m.nbody; ++b) {
    out.xpos.emplace_back(xpos[3 * b], xpos[3 * b + 1], xpos[3 * b + 2]);
    out.xquat.emplace_back(xquat[4 * b], xquat[4 * b + 1], xquat[4 * b + 2], xquat[4 * b + 3]);
  }
  for (int d = 0; d < I.m.nv; ++d) {
    out.dof_axis.emplace_back(ax[3 * d], ax[3 * d + 1], ax[3 * d + 2]);
    out.dof_anchor.emplace_back(an[3 * d], an[3 * d + 1], an[3 * d + 2]);
  }
  return out;
}

Stepper::Terms Stepper::terms(const StepParams& P, const VectorXd& q0h, const VectorXd& v0h, double kappa,
                              const VectorXd& qh) {
  Impl& I = *impl_;
  I.prepare(P, q0h, v0h, kappa);
  I.q.upload(qh.data(), I.m.nq);
  I.run_fk(I.q.get(), I.F);
  I.nc = I.detect(I.F, P.dhat);
  Terms out;
  I.assemble(P, kappa);
  out.energy = I.get_scalar(0);
  out.ncontact = I.nc;
  out.nfriction = I.nf;
  const int nv = I.m.nv, TD = I.m.TD, RW = 2 * I.m.CH;
  out.g.resize(nv);
  I.g.download(out.g.data(), nv);
  // 稠密的 H：树块放回对角块，再加每一行的外积。
  std::vector<double> Hk(static_cast<size_t>(std::max(I.m.ntree, 1)) * TD * TD);
  I.Hk.download(Hk.data(), Hk.size());
  out.H = MatrixXd::Zero(nv, nv);
  for (int d = 0; d < nv; ++d) {
    const int t = I.model.dof_tree[d];
    int d0 = d;
    while (d0 > 0 && I.model.dof_tree[d0 - 1] == t) --d0;
    for (int e = 0; e < nv; ++e) {
      if (I.model.dof_tree[e] != t) continue;
      out.H(d, e) = Hk[static_cast<size_t>(t) * TD * TD + (d - d0) * TD + (e - d0)];
    }
  }
  // 行先在显存里展开成逐个自由度的值（launch_rows_expand），再下载。
  std::vector<int> idx(static_cast<size_t>(I.nrow) * RW);
  std::vector<double> val(static_cast<size_t>(I.nrow) * RW), D(I.nrow);
  DBuf<int> d_idx;
  DBuf<double> d_val;
  d_idx.reserve(idx.size());
  d_val.reserve(val.size());
  launch_rows_expand(I.m, I.F.view(), I.rows_view(), I.nrow, d_idx.get(), d_val.get());
  d_idx.download(idx.data(), idx.size());
  d_val.download(val.data(), val.size());
  I.row_D.download(D.data(), D.size());
  for (int r = 0; r < I.nrow; ++r) {
    VectorXd row = VectorXd::Zero(nv);
    for (int k = 0; k < RW && idx[r * RW + k] >= 0; ++k) row[idx[r * RW + k]] += val[r * RW + k];
    out.H += D[r] * row * row.transpose();
  }
  return out;
}

VectorXd Stepper::solve(const VectorXd& rhs_h, double tol, int* iterations) {
  Impl& I = *impl_;
  I.rhs.upload(rhs_h.data(), I.m.nv);
  const int it = I.pcg_solve(I.rhs.get(), I.p.get(), tol, 1000);
  if (iterations) *iterations = it;
  VectorXd x(I.m.nv);
  I.p.download(x.data(), I.m.nv);
  return x;
}

MatrixXd Stepper::precond_inverse() {
  Impl& I = *impl_;
  const int nv = I.m.nv, TD = I.m.TD;
  launch_precond_build(I.m, I.F.view(), I.Hk.get(), I.rows_view(), I.nrow, I.P.get(), I.scale.get(), I.precond_work());
  std::vector<double> P(static_cast<size_t>(std::max(I.m.ntree, 1)) * TD * TD);
  I.P.download(P.data(), P.size());
  MatrixXd out = MatrixXd::Zero(nv, nv);
  for (int d = 0; d < nv; ++d) {
    const int t = I.model.dof_tree[d];
    int d0 = d;
    while (d0 > 0 && I.model.dof_tree[d0 - 1] == t) --d0;
    for (int e = 0; e < nv; ++e)
      if (I.model.dof_tree[e] == t) out(d, e) = P[static_cast<size_t>(t) * TD * TD + (d - d0) * TD + (e - d0)];
  }
  return out;
}

double Stepper::ccd(const StepParams& P, const VectorXd& qh, const VectorXd& ph) {
  Impl& I = *impl_;
  I.q.upload(qh.data(), I.m.nq);
  I.p.upload(ph.data(), I.m.nv);
  I.run_fk(I.q.get(), I.F);
  return I.ccd(P, I.q.get(), I.p.get());
}

}  // namespace rcipc::gpu
