"""生成四足机器人堆的 MJCF（阶段 12 之后的场景，笔记 notes/13_mechanisms.md 第 4 节）。

用法：
    python tools/make_robot_pile.py <go2.xml 的路径> [models/local/go2_pile.xml] [--n 24] [--seed 0]

把 MuJoCo Menagerie 的 Unitree Go2（unitree_go2/go2.xml，13 个刚体、12 个限位铰链加一个浮动基座）复制 n 份，
排成两层的方阵，从 0.5–1.1 m 高摔到几级台阶和几块石头上。每只随机转一个朝向，腿摆成站立的姿势再加一点随机
（关键帧 0 的 qpos，scene_demo 从它开始）。碰撞用模型自带的碰撞体（5 个盒子、12 个圆柱、4 个球），
显示用的网格留着（contype=0，我们的读模型会跳过，只给 tools/render.py 用）。

Go2 的模型文件不在这个仓库里，所以生成的 MJCF 写到 models/local/（被 .gitignore 忽略），网格目录写成绝对路径。
需要 mujoco（MjSpec，3.2 以上）和 numpy。
"""
import argparse
import math
import os

import mujoco
import numpy as np

STAND = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}  # 站立的姿势（弧度），都在 Go2 的限位里


def build(args, rng):
    """搭一次场景，返回 (spec, 编译好的模型, 关键帧的 qpos)。"""
    child = mujoco.MjSpec.from_file(args.go2)
    assets = os.path.abspath(os.path.join(os.path.dirname(args.go2), child.meshdir or "assets"))
    spec = mujoco.MjSpec()
    spec.modelname = "go2_pile"
    spec.meshdir = assets
    spec.compiler.degree = False
    spec.option.timestep = 0.01
    # 地面、三级台阶、几块石头（都是世界上的盒子）。
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    for k in range(3):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.8 + 0.35 * k, 0, 0.05 * (k + 1)],
                                size=[0.175, 1.8, 0.05 * (k + 1)])
    for k in range(6):
        yaw = rng.uniform(0, math.pi)
        q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[rng.uniform(-1.6, 0.2), rng.uniform(-1.4, 1.4), 0.04],
                                size=[0.08, 0.06, 0.04], quat=q)

    per_layer = (args.n + 1) // 2
    cols = math.ceil(math.sqrt(per_layer))
    for i in range(args.n):
        layer, j = divmod(i, per_layer)
        x = (j % cols - (cols - 1) / 2) * args.spacing + 0.5 * args.spacing * layer
        y = (j // cols - (cols - 1) / 2) * args.spacing
        z = 0.5 + 0.6 * layer + rng.uniform(0, 0.05)
        yaw, roll = rng.uniform(-math.pi, math.pi), rng.uniform(-0.3, 0.3)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, [roll, 0, yaw], "xyz")
        frame = spec.worldbody.add_frame(pos=[x, y, z], quat=quat)
        one = mujoco.MjSpec.from_file(args.go2)  # attach 会把刚体从原来的 spec 里搬走，每只重新读一份
        frame.attach_body(one.body("base"), f"r{i}_", "")

    m = spec.compile()
    key = np.array(m.qpos0)
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        for part, angle in STAND.items():
            if name.endswith(f"{part}_joint"):
                lo, hi = m.jnt_range[jid]
                key[m.jnt_qposadr[jid]] = float(np.clip(angle + rng.uniform(-0.15, 0.15), lo + 0.2, hi - 0.2))
    return spec, m, key


def touching(m, key, margin=0.01):
    """起始姿态里有没有 margin 以内的两个碰撞体（MuJoCo 自己的检测，排除了父子刚体）。"""
    m.geom_margin[:] = margin
    d = mujoco.MjData(m)
    d.qpos[:] = key
    mujoco.mj_forward(m, d)
    return d.ncon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("go2")
    ap.add_argument("out", nargs="?", default="models/local/go2_pile.xml")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spacing", type=float, default=0.9)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    # 随机的朝向可能让相邻的两只叠在一起：起始姿态里有任何 1 cm 以内的碰撞体就重新抽。
    for attempt in range(100):
        spec, m, key = build(args, rng)
        n = touching(m, key)
        if n == 0:
            break
    else:
        raise SystemExit("could not find a start pose without contacts; increase --spacing")
    spec.add_key(name="drop", qpos=key.tolist())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(spec.to_xml())
    print(f"wrote {args.out}: {args.n} Go2, {m.nbody - 1} bodies, nv {m.nv} (attempt {attempt + 1})")


if __name__ == "__main__":
    main()
