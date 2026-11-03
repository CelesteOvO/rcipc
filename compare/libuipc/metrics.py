"""两边共用的指标（和旧分支 run_uipc_ragdoll.py 的 measure() 同一个公式）。

输入是每个刚体在世界里的 4x4 变换，坐标系是 ragdoll_scene 的刚体坐标系（局部 z 沿胶囊轴，原点在胶囊中心）：
  - libuipc：ABD 的仿射变换（左上 3x3 不一定是转动矩阵）；
  - RC-IPC：MuJoCo FK 的 xmat / xpos 再右乘固定的 R_b（make_mjcf.py 说明了为什么）。

指标（每个都是所有布娃娃里的最大或最小值）：
  drift_mm   关节漂移：同一个关节在父刚体和子刚体上的锚点之间的最大距离（mm）。约化坐标里结构上是 0。
  axis_deg   铰链轴偏差：父刚体和子刚体上的铰链轴之间的最大夹角（度）。
  orth       非刚性：max ||A^T A - I||_F，A 是左上 3x3。刚体是 0，仿射体不是。
  lowest_mm  离地：几何体最低点的 y（mm）。libuipc 用它模拟的网格顶点，RC-IPC 用它模拟的精确胶囊（端点 y - 半径）。
"""
import numpy as np

import ragdoll_scene as R

BODIES = R.body_rest()
JOINTS = R.joint_local(BODIES)


def measure(T, geometry):
    """T[b] 是形状 (n, 4, 4) 的数组：刚体 b 在 n 个布娃娃里的变换。geometry 是 "mesh" 或 "capsule"。"""
    drift, axerr, orth, lowest = 0.0, 0.0, 0.0, 1e9
    for bi, b in enumerate(BODIES):
        A = T[bi][:, :3, :3]
        e = np.einsum("kji,kjl->kil", A, A) - np.eye(3)
        orth = max(orth, float(np.sqrt((e ** 2).sum(axis=(1, 2))).max()))
        if geometry == "mesh":
            Vw = np.einsum("kij,vj->kvi", A, b["V"]) + T[bi][:, None, :3, 3]
            lowest = min(lowest, float(Vw[..., 1].min()))
        else:
            ends = np.array([[0.0, 0.0, -b["L"] / 2], [0.0, 0.0, b["L"] / 2]])
            Ew = np.einsum("kij,vj->kvi", A, ends) + T[bi][:, None, :3, 3]
            lowest = min(lowest, float(Ew[..., 1].min()) - b["r"])
    for j in JOINTS:
        Tp, Tc = T[j["parent"]], T[j["child"]]
        ap = np.einsum("kij,j->ki", Tp[:, :3, :3], j["a_p"]) + Tp[:, :3, 3]
        ac = np.einsum("kij,j->ki", Tc[:, :3, :3], j["a_c"]) + Tc[:, :3, 3]
        drift = max(drift, float(np.linalg.norm(ap - ac, axis=1).max()))
        if j["type"] == "hinge":
            xp = np.einsum("kij,j->ki", Tp[:, :3, :3], j["x_p"])
            xc = np.einsum("kij,j->ki", Tc[:, :3, :3], j["x_c"])
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            axerr = max(axerr, float(np.degrees(np.arccos(np.clip((xp * xc).sum(1), -1, 1))).max()))
    return dict(drift_mm=drift * 1e3, axis_deg=axerr, orth=orth, lowest_mm=lowest * 1e3)
