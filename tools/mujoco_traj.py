"""用 MuJoCo 自己的仿真跑同一个 MJCF，写成和 scene_demo 一样格式的轨迹 CSV（笔记 notes/13_mechanisms.md 第 7 节）。

用法：
    python tools/mujoco_traj.py <model.xml> <out.csv> [--T 5] [--dt 0.001] [--every 0.01]

从第一个关键帧开始（qpos 和 ctrl）。控制量和 scene_demo 一样按关键帧的时刻分段线性插值（drive.cc 的 ctrl_at，
取这一步结束的时刻）：只有一个关键帧时就是保持它的值。积分器、求解器用 Menagerie 的 G1 的设置
（implicitfast、CG），时间步长 dt，接触参数全用 MuJoCo 的默认值。每 every 秒记一行：
    t, min_distance, min_limit, newton, pcg, qpos...
min_distance 是这一刻所有接触里最小的 dist（MuJoCo 的接触在 dist < margin = 0 时才有，所以是负的穿透深度；
没有接触时写 inf），min_limit、newton、pcg 写 0。tools/render.py 看到负的间隙会写成“最深穿透”。
最后打印整段里最深的穿透，分成“刚体和世界”（地面、世界上的盒子）“刚体和刚体”两类。
--dt 可以给几个值（逗号分开），每个写一个文件：out.csv 里的 {dt} 换成步长的毫秒数。
--timeconst：接触的 solref 时间常数（所有 geom 一起改，阻尼比保持 1）。default 是 MuJoCo 的默认值 0.02 s，
2dt 是 MuJoCo 允许的最硬的值（refsafe 打开时，比 2 dt 小的时间常数会被夹到 2 dt），也可以直接给秒数。
out.csv 里的 {tc} 换成这个设置的名字。--bench：再从头跑一遍，只量 mj_step 的墙钟时间（单线程 CPU）。
"""
import argparse
import time

import mujoco
import numpy as np


def ctrl_at(keys, t):
    """keys：按时间排好的 (time, ctrl)。和 src/drive.cc 的 ctrl_at 一样：两端取端点的值，中间线性插值。"""
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    k = 1
    while keys[k][0] < t:
        k += 1
    t0, t1 = keys[k - 1][0], keys[k][0]
    w = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
    return (1 - w) * keys[k - 1][1] + w * keys[k][1]


def run(path, out, T, dt, every_s, timeconst="default", bench=False):
    m = mujoco.MjModel.from_xml_path(path)
    m.opt.timestep = dt
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.solver = mujoco.mjtSolver.mjSOL_CG
    if timeconst != "default":
        tc = 2 * dt if timeconst == "2dt" else float(timeconst)
        m.geom_solref[:, 0] = tc
        m.geom_solref[:, 1] = 1.0
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    keys = sorted((float(m.key_time[k]), m.key_ctrl[k].copy()) for k in range(m.nkey)) if m.nu > 0 else []
    every = max(1, round(every_s / dt))
    nstep = round(T / dt)
    worst_ground, worst_body = 0.0, 0.0
    rows = []
    for i in range(1, nstep + 1):
        if keys:
            d.ctrl[:] = ctrl_at(keys, i * dt)
        mujoco.mj_step(m, d)
        for c in d.contact[: d.ncon]:
            ground = m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0
            if ground:
                worst_ground = min(worst_ground, c.dist)
            else:
                worst_body = min(worst_body, c.dist)
        if i % every == 0:
            dmin = min((c.dist for c in d.contact[: d.ncon]), default=np.inf)
            rows.append([i * dt, dmin, 0, 0, 0, *d.qpos])
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("t,min_distance,min_limit,newton,pcg,qpos...\n")
        for r in rows:
            f.write(",".join(repr(float(x)) for x in r) + "\n")
    if bench:
        # 只量 mj_step 本身：重新从关键帧开始，不记录、不扫接触（上面那一遍在 Python 里扫接触，比仿真本身慢）。
        mujoco.mj_resetDataKeyframe(m, d, 0)
        t0 = time.perf_counter()
        for i in range(1, nstep + 1):
            if keys:
                d.ctrl[:] = ctrl_at(keys, i * dt)
            mujoco.mj_step(m, d)
        wall = time.perf_counter() - t0
        print(f"  mj_step alone: {wall:.2f} s for {nstep} steps ({wall / nstep * 1e3:.3f} ms/step)")
    print(f"wrote {out}: dt {dt * 1e3:g} ms, solref timeconst {timeconst}, {len(rows)} rows; deepest penetration body-world {-worst_ground * 1e3:.2f} mm, "
          f"body-body {-worst_body * 1e3:.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--dt", default="0.001")
    ap.add_argument("--every", type=float, default=0.01)
    ap.add_argument("--timeconst", default="default")
    ap.add_argument("--bench", action="store_true", help="再跑一遍只量 mj_step 的墙钟时间")
    args = ap.parse_args()
    for dt in [float(x) for x in str(args.dt).split(",")]:
        out = args.out.replace("{dt}", f"{dt * 1e3:g}").replace("{tc}", args.timeconst)
        run(args.model, out, args.T, dt, args.every, args.timeconst, args.bench)


if __name__ == "__main__":
    main()
