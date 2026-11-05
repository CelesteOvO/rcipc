"""把一个 MJCF 场景导出成 JSON，给 run_uipc_scene.py 在 libuipc 里搭同一个场景（笔记 notes/15_drift.md）。

用法（系统的 Python，要 mujoco；libuipc 的 venv 里没有 mujoco）：
    python compare/libuipc/export_mjcf.py models/wrecking_ball.xml out.json

支持的范围（这次的场景用到的）：
- 参与碰撞的几何体只能是盒子；平面当地面。contype = conaffinity = 0 的几何体只用来显示，跳过。
- 关节：hinge（libuipc 的 AffineBodyRevoluteJoint）和 free（没有关节）。别的类型直接报错。
- 每个刚体是 libuipc 的一个仿射体：刚体上所有盒子合成一个三角网格，密度 = 刚体质量 / 盒子体积之和
  （同一个刚体上的盒子密度相同时，质量、质心、惯量都和 MuJoCo 的一致）。
- 世界上的每个盒子是一个固定的仿射体（is_fixed）。它的质量会进 libuipc 罚函数关节的刚度
  （刚度 = strength_ratio x 两边的质量之和），MuJoCo 的世界几何体没有质量，所以用 --world_density（默认钢）。
  连到世界的关节，另一方取离锚点最近的那个世界盒子。
输出的位姿都是 qpos0 时的世界坐标；libuipc 里每个仿射体的变换就是这个刚体在世界里的（仿射）位姿。
"""
import argparse
import json
import sys

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--world_density", type=float, default=7800.0)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    box, plane = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_PLANE

    def collides(g):
        return m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0

    def geom_in(g, frame_R, frame_p):
        """几何体 g 在某个坐标系（world 里的 R, p）里的位姿。"""
        Rg = d.geom_xmat[g].reshape(3, 3)
        return dict(pos=(frame_R.T @ (d.geom_xpos[g] - frame_p)).tolist(), R=(frame_R.T @ Rg).tolist(),
                    half=m.geom_size[g].tolist())

    bodies, world, planes = [], [], []
    mu = None
    for g in range(m.ngeom):
        if not collides(g):
            continue
        mu = mu if mu is not None else float(m.geom_friction[g][0])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        if m.geom_type[g] == plane:
            if m.geom_bodyid[g] != 0:
                sys.exit(f"{name}: a plane must belong to the world")
            n = d.geom_xmat[g].reshape(3, 3)[:, 2]
            planes.append(dict(name=name, normal=n.tolist(), height=float(n @ d.geom_xpos[g])))
        elif m.geom_type[g] != box:
            sys.exit(f"{name}: only boxes (and planes) are supported, got geom type {m.geom_type[g]}")
        elif m.geom_bodyid[g] == 0:
            b = geom_in(g, np.eye(3), np.zeros(3))
            b["name"] = name
            world.append(dict(name=name, boxes=[b], density=a.world_density))
    for b in range(1, m.nbody):
        R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
        boxes = [geom_in(g, R, p) for g in range(m.ngeom) if m.geom_bodyid[g] == b and collides(g)]
        if not boxes:
            sys.exit(f"body {b} has no colliding box")
        vol = sum(8 * np.prod(x["half"]) for x in boxes)
        X0 = np.eye(4)
        X0[:3, :3], X0[:3, 3] = R, p
        bodies.append(dict(id=b, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), parent=int(m.body_parentid[b]),
                           mass=float(m.body_mass[b]), density=float(m.body_mass[b] / vol), X0=X0.tolist(), boxes=boxes))
    joints = []
    for j in range(m.njnt):
        t = m.jnt_type[j]
        b = int(m.jnt_bodyid[j])
        if t == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if t != mujoco.mjtJoint.mjJNT_HINGE:
            sys.exit(f"joint {j}: only hinge and free joints are supported")
        anchor, axis = d.xanchor[j].copy(), d.xaxis[j].copy()
        jt = dict(type="hinge", child=b, parent=int(m.body_parentid[b]), anchor=anchor.tolist(), axis=axis.tolist(),
                  name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        if jt["parent"] == 0:
            if not world:
                sys.exit(f"joint {j} is attached to the world, but the world has no box to attach it to")
            dist = [np.linalg.norm(np.array(w["boxes"][0]["pos"]) - anchor) for w in world]
            jt["world_partner"] = int(np.argmin(dist))
        joints.append(jt)
    out = dict(model=a.model, timestep=float(m.opt.timestep), gravity=m.opt.gravity.tolist(), friction=mu,
               nq=int(m.nq), qpos0=m.qpos0.tolist(), bodies=bodies, world=world, planes=planes, joints=joints)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {a.out}: {len(bodies)} bodies, {len(world)} fixed world boxes, {len(planes)} planes, "
          f"{len(joints)} hinges, friction {mu}")


if __name__ == "__main__":
    main()
