#pragma once

// GPU 版的数据布局（阶段 11，笔记 notes/11_gpu.md 第 1 节）。主机代码（.cc）和设备代码（.cu）共用，所以只有纯数组和整数。
//
// Model 里的结构体数组（vector<Body> 之类）在这里摊平成“每个字段一个数组”（structure of arrays），
// 相邻线程读相邻的元素，显存访问能合并。所有指针都指向显存。
namespace rcipc::gpu {

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

// 图元的缓冲区（launch_prim_boxes 写，窄相读，narrow.cuh）。
struct PrimBufs {
  float* fbox = nullptr;   // 每个图元的单精度盒子（向外取整），顶点、边、三角形依次排，每个 6 个 float
  double* dbox = nullptr;  // 同样的盒子的双精度版（可以为空；CCD 的精确复核用）
  float* xf = nullptr;     // 顶点坐标的单精度版（3 nvert 个；CCD 时是段起点的坐标）
};

// 窄相的候选图元对（narrow.cuh）：剔除的内核写，精确判断的内核读。
//   点–三角形：a 是顶点、b 是三角形；边–边：a 是第一条边、b = ~（第二条边），所以 b < 0 就是边–边。
struct DCand {
  int a, b;
};
struct DCands {
  DCand* item = nullptr;
  int* count = nullptr;  // 显存里的计数（可能超过 cap，超过时调用者扩容后重来）
  int cap = 0;
};

// Newton 系统的行（行形式，阶段 9）。GPU 上按“力旋量”存（rows.cuh）：每行两侧，每侧一个刚体和一个力旋量 (f, tau)。
// 这一行在自由度 d 上的值：d 在某一侧刚体的链上时，转动自由度 a_d . (tau - o_d x f)，滑动自由度 a_d . f；两侧相加。
struct DRows {
  int* body = nullptr;  // 每行 2 个：A 侧、B 侧的刚体（0 = 世界，这一侧没有贡献）
  double* w = nullptr;  // 每行 12 个：A 侧 f、tau，B 侧 f、tau
  double* D = nullptr;  // 每行的权重
};

}  // namespace rcipc::gpu
