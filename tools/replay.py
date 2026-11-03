"""在 MuJoCo 的窗口里回放 scene_demo 写出的轨迹（阶段 10）。

用法：
    python tools/replay.py models/ragdoll.xml ragdoll.csv [--speed 1.0] [--loop]
    python tools/replay.py models/ragdoll.xml ragdoll.csv --check     # 不开窗口，只检查文件能对上

CSV 的格式见 toys/scenes/main.cc：每行 t, min_distance, min_limit, newton, pcg, 然后是 nq 个 qpos。
只用 MuJoCo 做显示：这里不做任何物理计算。
"""
import argparse
import csv
import sys
import time

import mujoco
import numpy as np


def load(csv_path, nq):
    ts, qs = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 表头
        for row in reader:
            ts.append(float(row[0]))
            q = np.array([float(x) for x in row[5:]])
            if q.size != nq:
                raise ValueError(f"row has {q.size} qpos values, model has nq = {nq}")
            qs.append(q)
    return np.array(ts), np.array(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("csv")
    ap.add_argument("--speed", type=float, default=1.0, help="回放倍速")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--check", action="store_true", help="只检查文件，不开窗口")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    ts, qs = load(args.csv, m.nq)
    print(f"{len(ts)} frames, {ts[-1]:.3f} s, nq {m.nq}")
    if args.check:
        d.qpos[:] = qs[-1]
        mujoco.mj_forward(m, d)
        print("last frame forward kinematics ok")
        return

    from mujoco import viewer as mj_viewer  # 延迟导入：--check 时不需要窗口

    with mj_viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            start = time.time()
            for t, q in zip(ts, qs):
                if not viewer.is_running():
                    break
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                viewer.sync()
                wait = t / args.speed - (time.time() - start)
                if wait > 0:
                    time.sleep(wait)
            if not args.loop:
                break


if __name__ == "__main__":
    sys.exit(main())
