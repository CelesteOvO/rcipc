"""跑 Rigid-IPC 的 rigid_ipc_sim，把它的 sim.json 汇总成几行数字（笔记 notes/compare_rigid_ipc.md）。

用法：
    python compare/rigid_ipc/run_rigid_ipc.py <rigid-ipc>/fixtures/3D/mechanisms/gears/single-kinematic.json \
        --exe <rigid-ipc>/build/Release/rigid_ipc_sim.exe --out compare/rigid_ipc/results/gears_single \
        [--steps 100] [--nthreads 8] [--patch '{"timestep": 0.01}']
    python compare/rigid_ipc/run_rigid_ipc.py --summarize compare/rigid_ipc/results/gears_single/sim.json

sim.json 里有（SimState.cpp 第 430-447 行）：
- args：实际用到的全部参数（默认值补全以后的），对比时抄到笔记里；
- animation.state_sequence：每一步每个刚体的 position、rotation（转动向量）、速度；
- stats：step_timings（每步秒）、solver_iterations（每步走了几步 Newton，摩擦滞后迭代累加；判定收敛的那次不算）、
  num_contacts、step_minimum_distances（本应是步末最小距离，实际全是 -1，改用 state_sequence 里每步的 min_distance）。

统计只取 --skip 之后的步（默认跳过前 0 步）。时间都是 Rigid-IPC 自己量的，不含写文件。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np


def summarize(path, skip=0):
    with open(path) as f:
        R = json.load(f)
    st = R["stats"]
    t = np.array(st["step_timings"][skip:]) * 1e3
    it = np.array(st["solver_iterations"][skip:])
    nc = np.array(st["num_contacts"][skip:])
    # stats 里的 step_minimum_distances 全是 -1；animation.state_sequence 每步有 min_distance（没有激活的接触对时是 null）
    md = np.array([s["min_distance"] for s in R["animation"]["state_sequence"][1 + skip:] if s["min_distance"] is not None]
                  or [np.inf])
    args = R["args"]
    nb = len(args["rigid_body_problem"]["rigid_bodies"])
    print(f"{path}: {len(st['step_timings'])} steps, {nb} bodies, h {args['timestep']}, "
          f"dhat {args['distance_barrier_constraint']['initial_barrier_activation_distance']}, "
          f"trajectory {args['distance_barrier_constraint']['trajectory_type']}, "
          f"mu {args['rigid_body_problem']['coefficient_friction']}, time_stepper {args['rigid_body_problem']['time_stepper']}, "
          f"newton tol {args['newton_solver']['velocity_conv_tol']} ({args['newton_solver']['convergence_criteria']})")
    print(f"  ms/step: mean {t.mean():.1f}  median {np.median(t):.1f}  max {t.max():.1f}  total {t.sum() / 1e3:.1f} s")
    print(f"  newton steps/step: mean {it.mean():.2f}  max {it.max()}  (steps at >= 100: {(it >= 100).sum()})")
    print(f"  contacts/step: mean {nc.mean():.1f}  max {nc.max()}")
    print(f"  min distance at step ends: {md.min():.3e}")
    return dict(steps=int(len(st["step_timings"])), ms_mean=float(t.mean()), ms_median=float(np.median(t)),
                ms_max=float(t.max()), newton_mean=float(it.mean()), newton_max=int(it.max()),
                contacts_mean=float(nc.mean()), min_dist=float(md.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?")
    ap.add_argument("--exe", default=os.environ.get("RIGID_IPC_SIM", "D:/Project/rigid-ipc/build/Release/rigid_ipc_sim.exe"))
    ap.add_argument("--out", help="结果目录（写 sim.json 和 run.log）")
    ap.add_argument("--steps", type=int, default=-1, help="步数；默认按场景的 max_time / max_iterations")
    ap.add_argument("--nthreads", type=int, default=0, help="0 = Rigid-IPC 的默认（全部核）")
    ap.add_argument("--patch", help="JSON，合并进场景文件（Rigid-IPC 的 --patch）")
    ap.add_argument("--loglevel", default="info")
    ap.add_argument("--name", default="sim.json", help="结果文件名（Rigid-IPC 的 -f），同一个目录放几种设置")
    ap.add_argument("--summarize", help="只汇总一个已有的 sim.json")
    ap.add_argument("--skip", type=int, default=0)
    a = ap.parse_args()

    if a.summarize:
        summarize(a.summarize, a.skip)
        return
    if not a.scene or not a.out:
        sys.exit("need a scene and --out (or --summarize)")
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    cmd = [os.path.abspath(a.exe), "--ngui", os.path.abspath(a.scene), a.out, "--loglevel", a.loglevel, "-f", a.name]
    if a.steps > 0:
        cmd += ["--num-steps", str(a.steps)]
    if a.nthreads > 0:
        cmd += ["--nthreads", str(a.nthreads)]
    if a.patch:
        cmd += ["--patch", a.patch]
    print("running:", " ".join(cmd))
    t0 = time.time()
    log_name = "run" + a.name[3:-5] + ".log"  # sim.json -> run.log, sim_abs.json -> run_abs.log
    with open(os.path.join(a.out, log_name), "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(a.exe))
    wall = time.time() - t0
    print(f"exit {proc.returncode}, wall {wall:.1f} s (incl. start-up and writing sim.json)")
    if proc.returncode != 0:
        with open(os.path.join(a.out, log_name)) as log:
            print(log.read()[-3000:])
        sys.exit(proc.returncode)
    summarize(os.path.join(a.out, a.name), a.skip)


if __name__ == "__main__":
    main()
