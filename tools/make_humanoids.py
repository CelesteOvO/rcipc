"""生成 Unitree G1 人形机器人“多米诺”的 MJCF（笔记 notes/13_mechanisms.md 第 7 节）。

用法：
    python tools/make_humanoids.py <g1.xml 的路径> [models/local/g1_domino.xml] [--n 6] [--hullvert 64]
                                   [--spacing 0.7] [--tilt 0.3]

n 台 G1（MuJoCo Menagerie 的 unitree_g1/g1.xml，29 个限位铰链加浮动基座，每个铰链一个 position 驱动）
排成一列，面朝 +x，都摆成模型自带的 stand 姿势，驱动的目标就是这个姿势（kp = 500）。第一台绕脚尖往前倾 tilt 弧度，
倒下去撞第二台，依次倒下。

碰撞体：
- 模型的碰撞网格一共 21.9 万个三角形。MuJoCo 自己碰撞时用的是每个网格的凸包，但凸包也还有 5.6 万个三角形。
  这里让 MuJoCo 算凸包时最多取 hullvert 个顶点（MJCF 的 maxhullvert，qhull 的 TA 选项），
  把凸包换到刚体坐标系里，作为新的网格直接写进 MJCF（vertex、face 属性），碰撞 geom 改用它。
  所以生成的模型里，MuJoCo 和我们用的是同一套凸的碰撞体。
- 脚底的 8 个小球、肩上的 4 个圆柱照旧。显示用的网格留着（contype=0，我们的读模型会跳过）。

起始姿态：stand 关键帧的脚底比地面低 1.9 mm，这里把每台抬到最低的脚底球离地 1 mm。
生成器用 MuJoCo 的检测检查起始姿态（每个 geom 0.5 mm 的 margin，一对就是 1 mm）：除了脚底和地面，不能有接触。

G1 的模型文件不在这个仓库里，生成的 MJCF 写到 models/local/（被忽略），网格目录写成绝对路径。
需要 mujoco（MjSpec，3.2 以上）和 numpy。z 轴朝上。
"""
import argparse
import math
import os

import mujoco
import numpy as np

DT = 0.005  # 10 ms 时隐式欧拉的数值耗散让倒下明显变慢（笔记第 7 节）
FOOT_GAP = 0.001
COLORS = [(0.85, 0.35, 0.3), (0.3, 0.55, 0.85), (0.35, 0.7, 0.4), (0.85, 0.65, 0.25), (0.6, 0.4, 0.8),
          (0.3, 0.75, 0.75), (0.8, 0.45, 0.6), (0.55, 0.55, 0.55)]


def name_geoms(spec):
    """给没有名字的 geom 按顺序起名（col0、col1……），这样同一个文件读两遍，同一个名字是同一个 geom。"""
    for i, g in enumerate(spec.geoms):
        if not g.name:
            g.name = f"col{i}"


def hull_meshes(path, hullvert):
    """每个碰撞网格 geom 的凸包，换到刚体坐标系里：{geom 名: (顶点 n x 3, 三角形 m x 3)}。三角形朝外。"""
    spec = mujoco.MjSpec.from_file(path)
    name_geoms(spec)
    for mesh in spec.meshes:
        mesh.maxhullvert = hullvert
    m = spec.compile()
    out = {}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        mid = m.geom_dataid[g]
        ga = m.mesh_graphadr[mid]
        if ga < 0:
            raise SystemExit(f"mesh {mid} has no convex hull")
        # mesh_graph：numvert、numface、vert_edgeadr[numvert]、vert_globalid[numvert]、
        # edge_localid[numvert + 3 numface]、face_globalid[3 numface]（下标是网格自己的顶点编号）。
        nv, nf = m.mesh_graph[ga], m.mesh_graph[ga + 1]
        gid = m.mesh_graph[ga + 2 + nv: ga + 2 + 2 * nv]
        fo = ga + 2 + 3 * nv + 3 * nf
        faces_global = np.array(m.mesh_graph[fo: fo + 3 * nf]).reshape(nf, 3)
        local = {int(v): i for i, v in enumerate(gid)}
        faces = np.vectorize(local.get)(faces_global)
        v_mesh = m.mesh_vert[m.mesh_vertadr[mid] + np.array(gid)]
        # 网格坐标系 -> 刚体坐标系：编译时 MuJoCo 把网格挪到质心、转到主轴上，这个变换并进了 geom_pos、geom_quat。
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        X = m.geom_pos[g] + v_mesh @ R.reshape(3, 3).T
        c = X.mean(axis=0)
        for f in faces:  # 法向朝外（凸包上每个面都背着中心）
            a, b, d = X[f[0]], X[f[1]], X[f[2]]
            if np.dot(np.cross(b - a, d - a), a - c) < 0:
                f[1], f[2] = f[2], f[1]
        out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)] = (X, faces)
    return out


def g1_copy(path, hulls, tag):
    """读一份 G1（attach 会把刚体搬走，每台重新读），碰撞网格换成凸包。返回 spec。"""
    one = mujoco.MjSpec.from_file(path)
    name_geoms(one)
    for gname, (X, F) in hulls.items():
        mesh = one.add_mesh(name=f"hull_{gname}")
        mesh.uservert = X.ravel().tolist()
        mesh.userface = F.ravel().tolist()
        g = one.geom(gname)
        g.meshname = mesh.name
        g.pos = [0, 0, 0]
        g.quat = [1, 0, 0, 0]
    for mat in one.materials:  # 每台一个颜色，录像里分得清
        if mat.name == "metal":
            mat.rgba = [*COLORS[tag % len(COLORS)], 1]
    return one


def rot_y(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def stand_pose(path, hulls):
    """一台 G1 在 stand 姿势下：(各铰链的角度 {关节名: 角}, 控制量 {驱动名: u}, 基座的高度, 前脚尖在基座坐标系里的位置)。"""
    spec = mujoco.MjSpec()
    spec.meshdir = os.path.abspath(os.path.join(os.path.dirname(path), "assets"))
    spec.worldbody.add_frame().attach_body(g1_copy(path, hulls, 0).body("pelvis"), "", "")
    src = mujoco.MjSpec.from_file(path)
    m0 = src.compile()
    key = m0.key("stand")
    m = spec.compile()
    d = mujoco.MjData(m)
    angles = {}
    for j in range(m0.njnt):
        if m0.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            angles[mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)] = float(key.qpos[m0.jnt_qposadr[j]])
    ctrl = {mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_ACTUATOR, i): float(key.ctrl[i]) for i in range(m0.nu)}
    d.qpos[:] = 0
    d.qpos[3] = 1
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in angles:
            d.qpos[m.jnt_qposadr[j]] = angles[name]
    mujoco.mj_forward(m, d)
    # 脚底的球（在 ankle_roll 上的 sphere）：最低的球底离基座原点多远，前面两个球（x 最大）的中心。
    feet = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_contype[g]]
    bottom = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in feet)
    xs = sorted(feet, key=lambda g: d.geom_xpos[g][0])
    toe = np.mean([d.geom_xpos[g] for g in xs[-4:]], axis=0)  # 两只脚的前两个球
    return angles, ctrl, FOOT_GAP - bottom, toe


def build(args):
    hulls = hull_meshes(args.g1, args.hullvert)
    angles, ctrl, z0, toe = stand_pose(args.g1, hulls)
    spec = mujoco.MjSpec()
    spec.modelname = "g1_domino"
    spec.meshdir = os.path.abspath(os.path.join(os.path.dirname(args.g1), "assets"))
    spec.compiler.degree = False
    spec.option.timestep = DT
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.1])
    poses = []
    for k in range(args.n):
        pos = np.array([args.spacing * k, 0.0, z0])
        quat = np.array([1.0, 0, 0, 0])
        if k == 0 and args.tilt != 0:
            # 绕前脚尖的连线（平行于 y 轴）往前倾：基座的位置绕脚尖转，姿态左乘这个转动。脚尖离地的高度不变。
            p = pos + toe  # toe 是基座在原点时前脚尖的位置
            R = rot_y(args.tilt)
            pos = p + R @ (pos - p)
            mujoco.mju_mat2Quat(quat, R.ravel())
        poses.append((pos, quat))
        frame = spec.worldbody.add_frame()
        frame.attach_body(g1_copy(args.g1, hulls, k).body("pelvis"), f"g{k}_", "")
    m = spec.compile()
    qpos = np.array(m.qpos0)
    u = np.zeros(m.nu)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        k, base = int(name.split("_", 1)[0][1:]), name.split("_", 1)[1]
        a = m.jnt_qposadr[j]
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qpos[a: a + 3], qpos[a + 3: a + 7] = poses[k]
        else:
            qpos[a] = angles[base]
    for i in range(m.nu):
        u[i] = ctrl[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).split("_", 1)[1]]
    # attach 把每台自带的 stand 关键帧也搬了过来（只有那一台的姿态，其余是 qpos0），删掉，换成整个场景的一帧。
    for key in list(spec.keys):
        spec.delete(key)
    spec.add_key(name="stand", qpos=qpos.tolist(), ctrl=u.tolist())
    return spec, hulls


def check_start(spec):
    """起始姿态里，除了脚底和地面，不能有 1 mm 以内的接触（MuJoCo 把两个 geom 的 margin 相加）。"""
    m = spec.compile()
    m.geom_margin[:] = 0.5e-3
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    bad = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0:
            if c.dist < 0:
                bad.append(("world", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1 + b2), c.dist))
            continue
        bad.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2), c.dist))
    if bad:
        raise SystemExit(f"start pose has {len(bad)} contacts within 1 mm: {bad[:8]}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("g1")
    ap.add_argument("out", nargs="?", default="models/local/g1_domino.xml")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--hullvert", type=int, default=64)
    ap.add_argument("--spacing", type=float, default=0.7)
    ap.add_argument("--tilt", type=float, default=0.3)
    args = ap.parse_args()
    spec, hulls = build(args)
    m = check_start(spec)
    nface = sum(len(F) for _, F in hulls.values())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(spec.to_xml())
    print(f"wrote {args.out}: {args.n} G1, {m.nbody - 1} bodies, nv {m.nv}, nu {m.nu}; "
          f"{len(hulls)} hulls per robot with {nface} triangles (hullvert {args.hullvert})")


if __name__ == "__main__":
    main()
