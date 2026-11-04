#pragma once

// GPU 版的数据布局（阶段 11，笔记 notes/11_gpu.md 第 1 节）。主机代码（.cc）和设备代码（.cu）共用，所以只有纯数组和整数。
//
// Model 里的结构体数组（vector<Body> 之类）在这里摊平成“每个字段一个数组”（structure of arrays），
// 相邻线程读相邻的元素，显存访问能合并。所有指针都指向显存。
namespace rcipc::gpu {

#ifdef __CUDACC__
#define RC_DM_HD __host__ __device__
#else
#define RC_DM_HD
#endif

// 一个标量参数：主机传的值，或者显存里的地址。CUDA graph 版的 Newton 迭代（stepper.cc、iter.cu）录好以后，
// 内核的参数就固定了；每次迭代都会变的数（步长、段长、kappa、接触数、行数）只能放在显存里，由内核自己去读。
// 从 double / int 隐式转换，所以逐个发射的调用照常传值。
struct DScalar {
  double v = 0;
  const double* p = nullptr;
  RC_DM_HD DScalar(double x = 0) : v(x) {}
  RC_DM_HD DScalar(double x, const double* ptr) : v(x), p(ptr) {}
  RC_DM_HD double get() const { return p ? *p : v; }
};
// 一个个数：主机传的值，或者显存里的地址（这时不超过 cap，网格按 cap 开）。
struct DCount {
  int v = 0;
  const int* p = nullptr;
  int cap = 0;
  RC_DM_HD DCount(int x = 0) : v(x) {}
  RC_DM_HD DCount(const int* ptr, int c) : v(0), p(ptr), cap(c) {}
#ifdef __CUDACC__
  __device__ int get() const { return p ? min(*p, cap) : v; }
#endif
  int grid() const { return p ? cap : v; }  // 发射时按多少个开网格
};

constexpr int kMaxDepth = 64;  // 链的最大深度（FK 和链上的遍历用的本地数组大小）

// 关节类型，和 JointType 的顺序一致。
enum JType { kFree = 0, kBall = 1, kSlide = 2, kHinge = 3 };
// 接触类型，和 ContactType 的顺序一致。
enum CType { kGround = 0, kPointTriangle = 1, kEdgeEdge = 2 };

struct DModel {
  int nbody = 0, nv = 0, nq = 0, njnt = 0, ntree = 0, nvert = 0, nedge = 0, nface = 0, nplane = 0;
  int TD = 0;  // 一棵树最多几个自由度（树块是 TD x TD）
  int TB = 0;  // 一棵树最多几个刚体
  int CH = 0;  // 一个刚体的链上最多几个自由度（一行最多 2 CH 个非零元）
  double gravity[3] = {0, 0, 0};
  int filter_parent = 1;

  // 刚体
  const int* parent = nullptr;       // nbody
  const double* bpos = nullptr;      // nbody x 3：在父刚体坐标系里的固定偏移
  const double* bquat = nullptr;     // nbody x 4
  const int* bjnt = nullptr;         // nbody：关节编号或 -1
  const double* mass = nullptr;      // nbody
  const double* ipos = nullptr;      // nbody x 3
  const double* inertia = nullptr;   // nbody x 9（刚体坐标系）
  const double* S = nullptr;         // nbody x 9
  const double* radius = nullptr;    // nbody
  const int* btree = nullptr;        // nbody：所属的树，世界是 -1
  const int* bsub_end = nullptr;     // nbody：子树是 [b, bsub_end[b])（MuJoCo 的刚体按深度优先排序）
  const int* chain_dof = nullptr;    // nbody x CH：刚体到根的链上的所有自由度
  const int* chain_n = nullptr;      // nbody
  const int* bvert0 = nullptr;       // nbody：这个刚体的顶点、边、面在全局数组里的起点和个数（同一个刚体的图元是连续的）
  const int* bnvert = nullptr;
  const int* bedge0 = nullptr;
  const int* bnedge = nullptr;
  const int* bface0 = nullptr;
  const int* bnface = nullptr;

  // 关节
  const int* jtype = nullptr;
  const int* jqadr = nullptr;
  const int* jdadr = nullptr;
  const int* jbody = nullptr;
  const int* jlimited = nullptr;
  const double* jpos = nullptr;    // njnt x 3
  const double* jaxis = nullptr;   // njnt x 3
  const double* jref = nullptr;
  const double* jrange = nullptr;  // njnt x 2

  // 自由度
  const int* dbody = nullptr;
  const int* drot = nullptr;
  const int* dtree = nullptr;

  // 树：自由度是 [tdof0, tdof0 + tndof)，刚体是 [tbody0, tbody0 + tnbody)
  const int* tdof0 = nullptr;
  const int* tndof = nullptr;
  const int* tbody0 = nullptr;
  const int* tnbody = nullptr;

  // 几何
  const int* vbody = nullptr;
  const double* vlocal = nullptr;   // nvert x 3
  const double* vradius = nullptr;
  const int* ev = nullptr;          // nedge x 2
  const int* ebody = nullptr;
  const int* fv = nullptr;          // nface x 3
  const int* fbody = nullptr;
  const double* pn = nullptr;       // nplane x 3
  const double* poff = nullptr;

  // 刚体坐标系里的几何（窄相的剔除用，narrow.cu；上传时算一次，以后不变）。单精度的盒子都向外取整。
  const float* vlf = nullptr;       // nvert x 3：顶点在自己刚体坐标系里的坐标（单精度）
  const float* flbox = nullptr;     // nface x 6：三角形在自己刚体坐标系里的盒子（含半径）
  const float* elbox = nullptr;     // nedge x 6：边的盒子
  const float* blbox_f = nullptr;   // nbody x 6：刚体全部三角形的总盒子（没有三角形时 lo > hi）
  const float* blbox_e = nullptr;   // nbody x 6：全部边的总盒子
  // 分组（chunk）：每个刚体的三角形按空间分成至多 32 个一组，组的盒子是组里各个盒子的并。
  const int* fch_order = nullptr;   // nface：按组排好的三角形编号
  const int* fch_range = nullptr;   // 每组 2 个：在 fch_order 里的起点、个数
  const float* fch_box = nullptr;   // 每组 6 个
  const int* bfch = nullptr;        // nbody x 2：刚体的第一组、组数
  const int* ech_order = nullptr;   // 边的分组，同上
  const int* ech_range = nullptr;
  const float* ech_box = nullptr;
  const int* bech = nullptr;
};

// 正运动学的结果（显存）。
struct DFrames {
  double* xpos = nullptr;     // nbody x 3
  double* xquat = nullptr;    // nbody x 4
  double* daxis = nullptr;    // nv x 3
  double* danchor = nullptr;  // nv x 3
  double* X = nullptr;        // nvert x 3：顶点的世界坐标
};

// 一个接触，和 Contact 的字段一一对应。
struct DContact {
  int type, nst;
  int v[4];
  double w[4];
  double n[3];
  double d;
  int bodyA, bodyB, plane;
  double moll, dmoll;
};

// 一个滞后的摩擦接触，和 FrictionContact 一一对应。
struct DFriction {
  DContact c;
  double lambda;
  double t1[3], t2[3];
  double X0[12];
};

// 窄相剔除的工作区（narrow.cu，调用者分配）：每对刚体的两个相对变换、阈值、单精度的余量，和工作列表。
struct DWork {
  int a, b;  // a = 3 x 刚体对的序号 + 查询的种类（0、1、2），b = 查询的顶点或边
};
struct DNarrow {
  float* pxf = nullptr;    // 每对 24 个：a 到 b、b 到 a 的变换（R 9 个，t 3 个）
  double* pT = nullptr;    // 每对的阈值
  float* ppad = nullptr;   // 每对的单精度余量
  float* pxm = nullptr;    // 每对的 xi（向上取整）加余量，CCD 的扫掠盒测试用
  DWork* work = nullptr;
  int* work_n = nullptr;   // 显存里的计数（可能超过 work_cap，超过时调用者扩容后重来）
  int work_cap = 0;
  // CCD 时段起点、终点的顶点世界坐标的单精度版（launch_to_float）；检测时为空。
  const float* xa = nullptr;
  const float* xb = nullptr;
};

// 窄相的候选图元对（narrow.cu）：剔除的内核写，精确判断的内核读。
//   点–三角形：a 是顶点、b 是三角形；边–边：a 是第一条边、b = ~（第二条边），所以 b < 0 就是边–边。
struct DCand {
  int a, b;
};
struct DCands {
  DCand* item = nullptr;
  int* count = nullptr;  // 显存里的计数（可能超过 cap，超过时调用者扩容后重来）
  int cap = 0;
};
#ifdef __CUDACC__
// 精确判断的内核用：显存里的候选个数（不超过 cap）。一个线程一个候选，网格跨步，不用回主机读个数。
__device__ inline int cand_count(const DCands& C) { return min(*C.count, C.cap); }
#endif

// CCD 的“尾巴”（notes/11_gpu.md 第 14 节）：ACCD 串行走了 split 步还没走完的对，由整个 warp 一起先试着证明
// 它返回 1，证不出来再接着走（k_ccd_exact）。
struct DTail {
  int split = 1000;  // 串行走几步以后交给 warp；不小于 kAccdMax 时不分
  // 累计：[0] 交给 warp 的对，[1] 证出返回 1 的，[2] 接着串行走的，[3] 单精度预筛跳过的（第 14.2 节）
  unsigned long long* stats = nullptr;
};

// Newton 系统的行（行形式，阶段 9）。GPU 上按“力旋量”存（rows.cuh）：每行两侧，每侧一个刚体和一个力旋量 (f, tau)。
// 这一行在自由度 d 上的值：d 在某一侧刚体的链上时，转动自由度 a_d . (tau - o_d x f)，滑动自由度 a_d . f；两侧相加。
struct DRows {
  int* body = nullptr;  // 每行 2 个：A 侧、B 侧的刚体（0 = 世界，这一侧没有贡献）
  double* w = nullptr;  // 每行 12 个：A 侧 f、tau，B 侧 f、tau
  double* D = nullptr;  // 每行的权重
};

// CUDA graph 版的一个 Newton 迭代（iter.cu、stepper.cc 的 Graph）：图录好以后内核参数都固定了，
// 每次迭代会变的数都放在这个结构里（显存），由设备上的小内核读写；迭代结束时主机把整个结构读回一次。
enum IterStatus { kIterRunning = 0, kIterConverged = 1, kIterDone = 2, kIterOverflow = 3 };
struct DIter {
  // 主机在发图之前写（异步拷贝，这三个放在最前面，一次拷完）：
  double kappa = 0;
  int nc = 0;       // 当前点的接触个数（接受试探点时由设备更新）
  int nf = 0;       // 摩擦接触的个数（步首冻结）
  // 设备写：
  int nrow = 0;     // nc + 2 nf
  int status = 0;   // IterStatus
  int pcg = 0;      // PCG 的迭代次数
  int segs = 0;     // CCD 的段数
  int windows = 0;  // 其中重新求候选（刚体对和剔除）的段数
  int trials = 0;   // 线搜索试探了几次
  int adv = 0;      // CCD：这一段通过了，下一段从它的终点开始（Fb 拷到 Fa）
  int cont = 0;     // 循环要不要继续（逐个发射时主机读它；录图时同时写进条件节点）
  int need[4] = {0, 0, 0, 0};  // 溢出时：刚体对、接触、候选、工作项各需要多少
  double e0 = 0, size = 0;     // 组装时的能量、收敛量（整步里顶点的最大位移 / h）
  double s = 0;                // launch_integrate 的参数（段终点，或试探的 alpha）
  double s0 = 0, ds = 0, len = 0, t_ccd = -1;  // CCD：段起点、下一段的长度、这一段的长度、截断位置
  double w0 = 0, w1 = -1, plen = 0;  // CCD：候选的窗口 [w0, w1]（w1 < 0 时没有）、求刚体对用的段长
  double alpha_max = 0, alpha = 0;
  double e_trial = 0, dmin = 0;  // 最后一次试探的能量、接触里最小的距离
};

// 预条件器的工作区（solver.cu 的 launch_precond_build，调用者分配）：
//   Wb：每个刚体一个 6x6 对称矩阵 sum D w w^T（只存上三角 21 个），w 是落在这个刚体上的那一侧力旋量（关于树根）；
//   cross：两侧在同一棵树里的行（它们还有两侧之间的交叉项），个数在显存里的 ncross，最多 cap 个（不少于行数）。
struct DPrecondWork {
  double* Wb = nullptr;
  int* cross = nullptr;
  int* ncross = nullptr;
  int cap = 0;
};

}  // namespace rcipc::gpu
