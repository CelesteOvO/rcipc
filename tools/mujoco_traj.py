"""用 MuJoCo 自己的仿真跑同一个 MJCF，写成和 scene_demo 一样格式的轨迹 CSV（笔记 notes/13_mechanisms.md 第 7 节）。

用法：
    python tools/mujoco_traj.py <model.xml> <out.csv> [--T 5] [--dt 0.001] [--every 0.01]

从第一个关键帧开始（qpos 和 ctrl），控制量保持关键帧里的值。积分器、求解器用 Menagerie 的 G1 的设置
（implicitfast、CG），时间步长 dt，接触参数全用 MuJoCo 的默认值。每 every 秒记一行：
    t, min_distance, min_limit, newton, pcg, qpos...
min_distance 是这一刻所有接触里最小的 dist（MuJoCo 的接触在 dist < margin = 0 时才有，所以是负的穿透深度；
没有接触时写 inf），min_limit、newton、pcg 写 0。tools/render.py 看到负的间隙会写成“最深穿透”。
最后打印整段里最深的穿透，分成“刚体和地面”“刚体和刚体”两类。
"""
import argparse

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--every", type=float, default=0.01)
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(args.model)
    m.opt.timestep = args.dt
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.solver = mujoco.mjtSolver.mjSOL_CG
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    every = max(1, round(args.every / args.dt))
    nstep = round(args.T / args.dt)
    worst_ground, worst_body = 0.0, 0.0
    rows = []
    for i in range(1, nstep + 1):
        mujoco.mj_step(m, d)
        for c in d.contact[: d.ncon]:
            ground = m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0
            if ground:
                worst_ground = min(worst_ground, c.dist)
            else:
                worst_body = min(worst_body, c.dist)
        if i % every == 0:
            dmin = min((c.dist for c in d.contact[: d.ncon]), default=np.inf)
            rows.append([i * args.dt, dmin, 0, 0, 0, *d.qpos])
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("t,min_distance,min_limit,newton,pcg,qpos...\n")
        for r in rows:
            f.write(",".join(repr(float(x)) for x in r) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows; deepest penetration body-ground {-worst_ground * 1e3:.2f} mm, "
          f"body-body {-worst_body * 1e3:.2f} mm")


if __name__ == "__main__":
    main()
