"""把 Rigid-IPC 的场景 JSON 转成 MJCF，给 scene_demo 跑同一个场景（笔记 notes/compare_rigid_ipc.md）。

用法（系统的 Python，要 numpy；加 --check 时还要 mujoco）：
    python compare/rigid_ipc/import_rigid_ipc.py <rigid-ipc>/fixtures/3D/mechanisms/gears/single-kinematic.json \
        compare/rigid_ipc/gears_single.xml --meshdir <rigid-ipc>/meshes [--kv 1e4] [--check]

Rigid-IPC 的刚体是最大坐标（6 个自由度），没有关节；它能表达的约束只有这几种，这里逐个对应到 MuJoCo 的关节：
- type = static，或 is_dof_fixed 全 true：固定不动，变成 worldbody 上的几何体；
- is_dof_fixed 只放开一个转动分量：绕世界坐标轴、过质心的铰链，变成 hinge（轴和位置写在刚体坐标系里）；
- 只放开一个平动分量：slide；
- 全放开：freejoint；
- 别的组合报错。
- type = kinematic 且给了 angular_velocity：Rigid-IPC 用增广拉格朗日（罚函数从 1e3 翻倍到 1e8）把位姿压到给定
  的轨迹上；这里变成那个 hinge 上的 velocity 驱动，kv 很大（默认 1e4），ctrl 用一个关键帧写成常数。
  带 kinematic_poses 的运动学体暂不支持。
- enabled = false 的刚体跳过（场景里的 pin 只用来和 IPC 原版对比）。

坐标系原样保留（Rigid-IPC 的场景多是 y 轴朝上），重力、步长照抄到 <option>。
rotation 是角度制的 xyz 欧拉角，R = Rz * Ry * Rx（read_rb_scene.cpp 第 199 行），绕网格原点转，再平移到 position。
每个刚体一个 mesh geom，density 照抄（Rigid-IPC 的默认是 1000）。两边都从网格算质量和质心，所以质量一致
（gear.obj 密度 1 时都是 1.266）。只用在固定几何体上的网格（比如平面）体积为 0，mesh 用 inertia="shell"。

摩擦系数 coefficient_friction、障碍距离 initial_barrier_activation_distance、时长 max_time 不进 MJCF，
打印成 scene_demo 的参数：mu=、dhat=、T=。
"""
import argparse
import json
import math
import os
import sys

import numpy as np


def rot_xyz_deg(rot):
    """Rigid-IPC 的 rotation（角度，xyz 欧拉角）-> 转动矩阵 R = Rz(c) Ry(b) Rx(a)。"""
    a, b, c = (math.radians(v) for v in rot)
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_from_matrix(M):
    """转动矩阵 -> MuJoCo 顺序 (w, x, y, z) 的单位四元数（Shepperd 的分支法，数值稳）。"""
    t = np.trace(M)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [s / 4, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(M)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + M[i, i] - M[j, j] - M[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (M[k, j] - M[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (M[j, i] + M[i, j]) / s
        q[1 + k] = (M[k, i] + M[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def read_obj(path):
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):  # 多边形按扇形切
                    F.append([idx[0], idx[k], idx[k + 1]])
    return np.array(V), np.array(F, dtype=int)


def mesh_volume_centroid(V, F):
    """闭合三角网格的体积和质心（原点出发的四面体求和）。体积为 0 时质心取顶点均值。"""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = vol6.sum() / 6
    if abs(vol) < 1e-15:
        return 0.0, V.mean(axis=0)
    centroid = ((a + b + c) * vol6[:, None]).sum(axis=0) / (4 * vol6.sum())
    return vol, centroid


def fmt(v):
    return " ".join(f"{x:.9g}" for x in np.asarray(v).ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--meshdir", required=True, help="Rigid-IPC 的 meshes 目录")
    ap.add_argument("--kv", type=float, default=1e4, help="运动学体换成 velocity 驱动时的增益")
    ap.add_argument("--check", action="store_true", help="用 mujoco 读一遍，打印每个刚体的质量")
    a = ap.parse_args()

    with open(a.scene) as f:
        S = json.load(f)
    if S.get("scene_type", "distance_barrier_rb_problem") != "distance_barrier_rb_problem":
        sys.exit(f"unsupported scene_type {S['scene_type']}")
    P = S["rigid_body_problem"]
    h = float(S.get("timestep", 0.01))
    gravity = P.get("gravity", [0.0, 0.0, 0.0])
    mu = float(P.get("coefficient_friction", 0.0))
    dhat = float(S.get("distance_barrier_constraint", {}).get("initial_barrier_activation_distance", 1e-3))
    T = float(S.get("max_time", -1))
    if T < 0 and int(S.get("max_iterations", -1)) > 0:
        T = int(S["max_iterations"]) * h
    time_stepper = P.get("time_stepper", "default")

    meshes = {}      # (file, scale, shell) -> mesh 名
    bodies_xml, world_xml, act_xml, ctrl = [], [], [], []
    notes = []
    for n, rb in enumerate(P["rigid_bodies"]):
        if not rb.get("enabled", True):
            continue
        mesh_file = rb.get("mesh", "")
        if not mesh_file:
            sys.exit(f"body {n}: inline vertices/faces are not supported, only mesh files")
        if mesh_file.lower().endswith(".obj"):
            V, F = read_obj(os.path.join(a.meshdir, mesh_file))
        else:
            sys.exit(f"body {n}: only .obj meshes are supported, got {mesh_file}")
        name = rb.get("name") or f"{os.path.splitext(os.path.basename(mesh_file))[0]}{n}"
        scale = rb.get("scale", 1.0)
        if "dimensions" in rb:
            sys.exit(f"body {n}: 'dimensions' is not supported")
        scale = np.array(scale if isinstance(scale, list) else [scale] * 3, dtype=float)
        density = float(rb.get("density", 1000.0))
        pos = np.array(rb.get("position", [0, 0, 0]), dtype=float)
        R = rot_xyz_deg(rb.get("rotation", [0, 0, 0]))
        quat = quat_from_matrix(R)
        btype = rb.get("type", "dynamic")
        fixed = rb.get("is_dof_fixed", False)
        fixed = [bool(fixed)] * 6 if isinstance(fixed, bool) else [bool(x) for x in fixed]
        if len(fixed) != 6:
            sys.exit(f"body {n}: is_dof_fixed must have 6 entries")
        for key in ("kinematic_poses", "force", "torque", "center_of_mass", "split_components"):
            if key in rb and rb[key] not in ([], None, False, [0, 0, 0]):
                sys.exit(f"body {n}: '{key}' is not supported")
        lin_v = np.array(rb.get("linear_velocity", [0, 0, 0]), dtype=float)
        ang_v = np.radians(np.array(rb.get("angular_velocity", [0, 0, 0]), dtype=float))

        static = btype == "static" or all(fixed)
        free_dofs = [i for i in range(6) if not fixed[i]]
        mkey = (mesh_file, tuple(scale), static)
        if mkey not in meshes:
            mname = os.path.splitext(os.path.basename(mesh_file))[0]
            if any(k[0] != mesh_file and v == mname for k, v in meshes.items()) or any(v == mname for v in meshes.values()):
                mname = f"{mname}_{len(meshes)}"
            meshes[mkey] = mname
        mname = meshes[mkey]
        geom = f'<geom name="{name}" type="mesh" mesh="{mname}" density="{density:.9g}"/>'

        vol, cm = mesh_volume_centroid(V, F)
        if static and abs(vol) < 1e-12 * max(1.0, float(np.abs(V).max())) ** 3:
            # 体积为 0 的固定网格（Rigid-IPC 的 plane.obj 这种一片四边形）：MuJoCo 要给它算凸包，会失败。
            # 平的就换成 MuJoCo 的无限平面（法向取三角形法向的均值），对地面来说和有限的一片没有区别。
            del meshes[mkey]
            a0, b0, c0 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            nrm = np.cross(b0 - a0, c0 - a0)
            nrm = (nrm * np.sign(nrm @ nrm[0])[:, None]).sum(axis=0)
            nrm /= np.linalg.norm(nrm)
            if np.abs((V - V[0]) @ nrm).max() > 1e-9 * max(1.0, float(np.abs(V).max())):
                sys.exit(f"body {n} ({name}): zero-volume static mesh that is not planar")
            n_world = R @ nrm
            # 平面 geom 的局部 z 轴是法向；取一个和法向垂直的 x 轴拼出姿态
            x_axis = np.cross([0.0, 1.0, 0.0], n_world) if abs(n_world[1]) < 0.9 else np.cross([1.0, 0.0, 0.0], n_world)
            x_axis /= np.linalg.norm(x_axis)
            Rp = np.column_stack([x_axis, np.cross(n_world, x_axis), n_world])
            ext = np.abs((V - cm) @ np.column_stack([x_axis, np.cross(nrm, x_axis)]) ).max(axis=0) * scale[:2]
            c_world = R @ (scale * cm) + pos
            world_xml.append(f'    <geom name="{name}" type="plane" pos="{fmt(c_world)}" quat="{fmt(quat_from_matrix(Rp))}" '
                             f'size="{fmt(ext)} 0.1"/>')
            notes.append(f"{name}: flat static mesh -> infinite plane, normal {fmt(n_world)}, half-extent {fmt(ext)}")
            continue
        if static:
            world_xml.append(f'    <geom name="{name}" type="mesh" mesh="{mname}" pos="{fmt(pos)}" quat="{fmt(quat)}"/>')
            continue

        cm_body = scale * cm  # 质心在刚体坐标系（缩放后、转动前）里的位置
        lines = [f'    <body name="{name}" pos="{fmt(pos)}" quat="{fmt(quat)}">']
        if len(free_dofs) == 6:
            lines.append('      <freejoint/>')
            if np.any(lin_v) or np.any(ang_v):
                notes.append(f"{name}: initial velocity dropped (not supported in MJCF import)")
        elif len(free_dofs) == 1:
            k = free_dofs[0]
            axis_world = np.zeros(3)
            axis_world[k % 3] = 1.0
            axis_body = R.T @ axis_world  # 轴在刚体坐标系里
            if k >= 3:
                lines.append(f'      <joint name="{name}_j" type="hinge" axis="{fmt(axis_body)}" pos="{fmt(cm_body)}"/>')
            else:
                lines.append(f'      <joint name="{name}_j" type="slide" axis="{fmt(axis_body)}"/>')
            if btype == "kinematic":
                if k < 3:
                    target = float(lin_v[k])
                else:
                    target = float(ang_v[k - 3])
                act_xml.append(f'    <velocity name="{name}_drive" joint="{name}_j" kv="{a.kv:.9g}"/>')
                ctrl.append(target)
                notes.append(f"{name}: kinematic body -> velocity drive, target {target:.6g} "
                             f"{'m/s' if k < 3 else 'rad/s'}, kv {a.kv:g}")
            elif np.any(lin_v) or np.any(ang_v):
                notes.append(f"{name}: initial velocity dropped (not supported in MJCF import)")
        else:
            sys.exit(f"body {n} ({name}): is_dof_fixed={fixed} has {len(free_dofs)} free dofs; "
                     "only 0, 1 (hinge/slide about a world axis) or 6 are supported")
        if btype == "kinematic" and len(free_dofs) != 1:
            sys.exit(f"body {n} ({name}): kinematic bodies need exactly one free dof")
        lines.append(f'      {geom}')
        lines.append('    </body>')
        bodies_xml.append("\n".join(lines))

    asset = []
    for (mesh_file, scale, static), mname in meshes.items():
        extra = ' inertia="shell"' if static else ''
        asset.append(f'    <mesh name="{mname}" file="{mesh_file}" scale="{fmt(scale)}"{extra}/>')

    meshdir = os.path.abspath(a.meshdir).replace("\\", "/")
    xml = [
        "<!--",
        f"  由 compare/rigid_ipc/import_rigid_ipc.py 从 Rigid-IPC 的 {os.path.relpath(a.scene, os.path.dirname(a.out)).replace(chr(92), '/')} 生成，",
        f"  scene_demo 参数：T={T:g} h={h:g} dhat={dhat:g} mu={mu:g}（笔记 notes/compare_rigid_ipc.md）。",
        "-->",
        f'<mujoco model="{os.path.splitext(os.path.basename(a.scene))[0]}">',
        f'  <compiler angle="radian" meshdir="{meshdir}"/>',
        f'  <option gravity="{fmt(gravity)}" timestep="{h:.9g}"/>',
        "  <default>",
        f'    <geom friction="{mu:.9g} 0 0"/>',
        "  </default>",
        "  <asset>",
        *asset,
        "  </asset>",
        "  <worldbody>",
        *world_xml,
        *bodies_xml,
        "  </worldbody>",
    ]
    if act_xml:
        xml += ["  <actuator>", *act_xml, "  </actuator>",
                "  <keyframe>", f'    <key name="drive" time="0" ctrl="{fmt(ctrl)}"/>', "  </keyframe>"]
    xml.append("</mujoco>")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(xml) + "\n")

    print(f"wrote {a.out}: {len(bodies_xml)} moving bodies, {len(world_xml)} static geoms, {len(meshes)} meshes")
    print(f"scene_demo args: T={T:g} h={h:g} dhat={dhat:g} mu={mu:g}   (Rigid-IPC time_stepper: {time_stepper})")
    for s in notes:
        print("  " + s)
    if a.check:
        import mujoco
        m = mujoco.MjModel.from_xml_path(a.out)
        for b in range(1, m.nbody):
            print(f"  body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)}: mass {m.body_mass[b]:.6g}, "
                  f"nq {m.nq} nv {m.nv}")


if __name__ == "__main__":
    main()
