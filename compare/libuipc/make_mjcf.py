"""把 ragdoll_scene.py 的布娃娃堆写成 MJCF，给 RC-IPC 用（和 libuipc 对比，笔记 notes/compare_libuipc.md）。

用法（在仓库根目录）：
    python compare/libuipc/make_mjcf.py 8          # 写出 compare/libuipc/ragdolls_8.xml（精确胶囊）
    python compare/libuipc/make_mjcf.py 60 --mesh  # 写出 compare/libuipc/ragdolls_60_mesh.xml（和 libuipc 同一套胶囊网格）

--mesh：每个刚体用 ragdoll_scene.capsule_mesh 的三角网格（libuipc 模拟的就是它），而不是精确胶囊。
比速度时必须用它：精确胶囊每个只有 2 个顶点和 1 条边，接触检测的工作量比网格小两个数量级，那样比出来的加速比不公平。

和 libuipc 那边保持一致的地方：
  - 同一组胶囊（端点、半径）、同一组关节（锚点、铰链轴）、同一个初始姿态（ragdoll_scene.placements(n)，种子 0）；
  - y 轴朝上：重力 (0, -9.8, 0)，地面是 y = 0 的平面（geom 的 z 轴转到 +y）；
  - dt = 0.01，没有关节限位（libuipc 的球关节没有限位）；
  - 质量 = libuipc 用的胶囊网格的体积 x 密度 1000（网格是胶囊的内接多面体，比真胶囊轻几个百分点）；
  - 相邻刚体也检测接触（filterparent="disable"），和 libuipc 一样。

我们这边的刚体坐标系：原点在胶囊中心，静止姿态下和世界坐标轴对齐。所以子刚体的 pos 就是两个中心之差，
关节 q = 0 就是 ragdoll_scene 的静止姿态（T 姿态）。ragdoll_scene 的刚体坐标系（局部 z 沿胶囊轴）和我们的差一个
固定的转动 R_b = body_rest()[b]["T"][:3, :3]，metrics.py 用它换算。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ragdoll_scene as R  # noqa: E402


def mesh_volume(V, F):
    """闭合三角网格的体积（散度定理）。"""
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6)


def quat_from_matrix(M):
    """转动矩阵 -> MuJoCo 顺序 (w, x, y, z) 的单位四元数。"""
    w = np.sqrt(max(0.0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + M[0, 0] - M[1, 1] - M[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - M[0, 0] + M[1, 1] - M[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - M[0, 0] - M[1, 1] + M[2, 2])) / 2
    x = np.copysign(x, M[2, 1] - M[1, 2])
    y = np.copysign(y, M[0, 2] - M[2, 0])
    z = np.copysign(z, M[1, 0] - M[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def make(n, mesh=False):
    bodies = R.body_rest()
    centers = [(np.array(b[1]) + np.array(b[2])) / 2 for b in R.BODIES]
    masses = [mesh_volume(b["V"], b["F"]) * R.DENSITY for b in bodies]
    children = {i: [] for i in range(len(R.BODIES))}
    joint_of = {}
    for parent, child, jt, anchor, axis in R.JOINTS:
        ip, ic = R.BODY_INDEX[parent], R.BODY_INDEX[child]
        children[ip].append(ic)
        joint_of[ic] = (jt, np.array(anchor), axis)
    root = R.BODY_INDEX["pelvis"]

    def body_xml(i, k, indent, pos, quat=None):
        name, p0, p1, r = R.BODIES[i]
        c = centers[i]
        pad = "  " * indent
        q = f' quat="{fmt(quat)}"' if quat is not None else ""
        out = [f'{pad}<body name="{name}_{k}" pos="{fmt(pos)}"{q}>']
        if i == root:
            out.append(f"{pad}  <freejoint/>")
        else:
            jt, anchor, axis = joint_of[i]
            jpos = fmt(anchor - c)
            if jt == "ball":
                out.append(f'{pad}  <joint type="ball" pos="{jpos}"/>')
            else:
                out.append(f'{pad}  <joint type="hinge" pos="{jpos}" axis="{fmt(axis)}"/>')
        a, b = np.array(p0) - c, np.array(p1) - c
        if mesh:
            # 网格在 ragdoll_scene 的刚体坐标系里（局部 z 沿胶囊轴）；我们的刚体坐标系和世界轴对齐，差一个转动 R_b。
            qm = quat_from_matrix(bodies[i]["T"][:3, :3])
            out.append(f'{pad}  <geom type="mesh" mesh="{name}" quat="{fmt(qm)}" mass="{masses[i]:.9g}"/>')
        elif np.linalg.norm(b - a) < 1e-12:
            out.append(f'{pad}  <geom type="sphere" size="{r}" mass="{masses[i]:.9g}"/>')
        else:
            out.append(f'{pad}  <geom type="capsule" fromto="{fmt(a)} {fmt(b)}" size="{r}" mass="{masses[i]:.9g}"/>')
        for ch in children[i]:
            out += body_xml(ch, k, indent + 1, centers[ch] - c)
        out.append(f"{pad}</body>")
        return out

    lines = [
        "<!--",
        f"  libuipc 对比用的布娃娃堆：{n} 个布娃娃（compare/libuipc/make_mjcf.py 从 ragdoll_scene.py 生成，不要手改）。",
        "  y 轴朝上；没有关节限位；相邻刚体也检测接触。" + ("几何是 libuipc 用的胶囊网格。" if mesh else "几何是精确胶囊。"),
        "-->",
        f'<mujoco model="ragdolls_{n}{"_mesh" if mesh else ""}">',
        '  <compiler angle="radian"/>',
        f'  <option gravity="0 -9.8 0" timestep="{R.DT}">',
        '    <flag filterparent="disable"/>',
        "  </option>",
    ]
    if mesh:
        lines.append("  <asset>")
        for (name, *_), b in zip(R.BODIES, bodies):
            lines.append(f'    <mesh name="{name}" vertex="{fmt(b["V"].ravel())}" face="{" ".join(str(x) for x in b["F"].ravel())}"/>')
        lines.append("  </asset>")
    lines += [
        "  <worldbody>",
        '    <geom name="floor" type="plane" size="10 10 0.1" quat="0.70710678 -0.70710678 0 0"/>',
    ]
    for k, G in enumerate(R.placements(n)):
        T = G @ np.block([[np.eye(3), centers[root][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
        lines += body_xml(root, k, 2, T[:3, 3], quat_from_matrix(T[:3, :3]))
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    mesh = "--mesh" in sys.argv
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ragdolls_{n}{'_mesh' if mesh else ''}.xml")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(make(n, mesh))
    print("wrote", out)
