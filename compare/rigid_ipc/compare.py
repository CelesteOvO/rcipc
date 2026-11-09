"""把一个场景两边的结果并排打印成表（笔记 notes/compare_rigid_ipc.md 第 3 节）。

用法（在仓库根目录）：
    python compare/rigid_ipc/compare.py compare/rigid_ipc/results/gears_single [--gear-angles] [--final-y]

目录里要有 run_rigid_ipc.py 写的 sim.json（可以有几份：sim.json、sim_abs.json ...；Rigid-IPC 每 100 步还写
sim-chkpt00100.json 这样的检查点，跳过）和 measure_rcipc.py 写的 rcipc_cpu.json / rcipc_gpu.json。

Newton 计数两边不一样：Rigid-IPC 的 total_iter 是走了几步（newton_solver.cpp 第 215 行：第 k 次迭代算出的方向
够小就停，这次不算）；我们的 newton 把只检查、不移动的那次也算上（step.cc 第 124 行）。表里把我们的减 1 对齐。

最小距离：Rigid-IPC 的 stats.step_minimum_distances 存的全是 -1，用 animation.state_sequence 里每步的 min_distance
（没有激活的接触对时是 null，跳过）。

--gear-angles：齿轮场景专用。两边从动轮的转角随时间，看传动比是不是一样。Rigid-IPC 的 state_sequence 只有
enabled 的刚体，齿轮场景里两根销子 enabled=false，所以从动轮是下标 1；我们是 qpos 的第 2 个。两边都是绕世界
z 轴的铰链，转角 = 相对初始姿态绕 z 的角。
--track R:O：跟踪一个刚体（钟摆的臂、螺栓的螺杆）：y 位移和绕 --axis 的转角随时间，两边各一行。
--final-y：链条这类自由刚体场景，打印每个刚体最后一步的 y（重力方向）；Rigid-IPC 按 state_sequence 的顺序，
我们按 freejoint 的顺序（qpos 每 7 个一块，第 2 个是 y）。
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np


def rotvec_to_matrix(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-15:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rigid_ipc_z_angles(sim, body):
    """每一步 body 相对第 0 步绕世界 z 轴转过的角（弧度，累计解卷绕）。"""
    seq = sim["animation"]["state_sequence"]
    R0 = rotvec_to_matrix(seq[0]["rigid_bodies"][body]["rotation"])
    angles = []
    for s in seq:
        R = rotvec_to_matrix(s["rigid_bodies"][body]["rotation"]) @ R0.T
        angles.append(math.atan2(R[1, 0], R[0, 0]))
    return np.unwrap(np.array(angles))


def rigid_ipc_min_distance(sim, skip=0):
    md = [s["min_distance"] for s in sim["animation"]["state_sequence"][1 + skip:] if s["min_distance"] is not None]
    return (min(md), len(md)) if md else (float("inf"), 0)


def quat_to_matrix(q):
    """MuJoCo 顺序 (w, x, y, z) 的单位四元数 -> 转动矩阵。"""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angles_about_axis(Rs, axis):
    """一串转动矩阵相对第一个绕世界轴 axis 转过的角（弧度，累计解卷绕）：取一个垂直于轴的向量，看它转到哪。"""
    a = np.zeros(3)
    a["xyz".index(axis)] = 1.0
    u = np.array([0.0, 0.0, 1.0]) if axis != "z" else np.array([1.0, 0.0, 0.0])
    R0 = Rs[0]
    out = []
    for R in Rs:
        w = R @ R0.T @ u
        out.append(math.atan2(np.cross(u, w) @ a, u @ w))
    return np.unwrap(np.array(out))


def rcipc_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[float(x) for x in row] for row in reader if row and all(row)]
        return [r for r in rows if len(r) == len(rows[0])]  # 正在写的文件最后一行可能不完整


def sim_files(d):
    return [p for p in sorted(glob.glob(os.path.join(d, "sim*.json"))) if "chkpt" not in os.path.basename(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--gear-angles", action="store_true")
    ap.add_argument("--driven", type=int, default=1, help="Rigid-IPC state_sequence 里从动轮的下标")
    ap.add_argument("--final-y", action="store_true")
    ap.add_argument("--track", help="R:O——跟踪一个刚体：Rigid-IPC state_sequence 里的下标 R，我们 freejoint 的序号 O；"
                    "打印它在 t = 0.5, 1, 2, 3, 5, 10 s 的 y 位移（相对 t=0）和绕 --axis 转过的角")
    ap.add_argument("--axis", default="z", help="--track 的转轴（世界坐标，x/y/z）")
    ap.add_argument("--skip", type=int, default=0)
    a = ap.parse_args()

    rows = []
    sims = {}
    for path in sim_files(a.dir):
        with open(path) as f:
            S = json.load(f)
        sims[path] = S
        st = S["stats"]
        t = np.array(st["step_timings"][a.skip:]) * 1e3
        it = np.array(st["solver_iterations"][a.skip:])
        ns = S["args"]["newton_solver"]
        tol = f"{ns['velocity_conv_tol']:g}{' abs' if ns['is_velocity_conv_tol_abs'] else ' x bbox'}"
        md, n_md = rigid_ipc_min_distance(S, a.skip)
        rows.append((f"Rigid-IPC ({os.path.basename(path)[4:-5].strip('_') or 'default'})", "CPU", tol, len(t),
                     t.mean(), np.median(t), t.max(), it.mean(), it.max(), md, float(np.mean(st["num_contacts"][a.skip:])),
                     f"{n_md} steps with an active pair"))
    for path in sorted(glob.glob(os.path.join(a.dir, "rcipc_*.json"))):
        with open(path) as f:
            S = json.load(f)
        extra = " ".join(S["args"].get("extra", []))
        tol = next((e.split("=")[1] for e in S["args"].get("extra", []) if e.startswith("tol=")), "1e-6") + " abs"
        csv_path = path[:-5] + ".csv"
        med = mx = float("nan")
        if os.path.exists(csv_path):
            nw = np.array([r[3] for r in rcipc_rows(csv_path)][a.skip:])
            med, mx = float(np.median(nw)), float(nw.max())
        rows.append((f"RC-IPC {extra}".strip(), "GPU" if S["solver"].startswith("gpu") else "CPU", tol, S["steps"], S["ms_mean"], float("nan"),
                     float("nan"), S["newton_mean"] - 1, S["newton_max"] - 1, S["min_dist"], float("nan"),
                     f"newton incl. the check: mean {S['newton_mean']:.2f} median {med:g} max {mx:g}; "
                     f"{S['newton_capped']} steps at the cap; PCG/step {S['pcg_mean']:.1f}"))

    print(f"{'solver':<22} {'where':<5} {'newton tol (m/s)':<17} {'steps':>5} {'ms mean':>8} {'median':>7} {'max':>8} "
          f"{'newton steps':>12} {'max':>4} {'min dist':>10} {'contacts':>9}")
    for r in rows:
        print(f"{r[0]:<22} {r[1]:<5} {r[2]:<17} {r[3]:>5} {r[4]:>8.1f} {r[5]:>7.1f} {r[6]:>8.1f} {r[7]:>12.2f} {r[8]:>4} "
              f"{r[9]:>10.2e} {r[10]:>9.1f}   {r[11]}")
    print("(newton steps = Newton steps actually taken per time step, both sides; contacts = Rigid-IPC's num_contacts)")

    if a.gear_angles:
        print("\ndriven gear angle (rad) at t = 1, 2, 5, 10 s (positive = same sense as +z):")
        for path, S in sims.items():
            ang = rigid_ipc_z_angles(S, a.driven)
            h = S["args"]["timestep"]
            picks = [int(round(T / h)) for T in (1, 2, 5, 10) if int(round(T / h)) < len(ang)]
            print(f"  {os.path.basename(path):<14} " + "  ".join(f"{ang[k]:8.4f}" for k in picks))
        for path in sorted(glob.glob(os.path.join(a.dir, "rcipc_*.csv"))):
            R = rcipc_rows(path)
            t = np.array([r[0] for r in R])
            q = np.array([r[5 + 1] for r in R])
            picks = [int(np.argmin(np.abs(t - T))) for T in (1, 2, 5, 10) if T <= t[-1] + 1e-9]
            print(f"  {os.path.basename(path):<14} " + "  ".join(f"{q[k]:8.4f}" for k in picks))
        print("  (driver: -100 deg/s = -1.7453 rad/s; equal gears -> driven +1.7453 rad/s, 1.7453 rad after 1 s)")

    if a.track:
        ri, oi = (int(x) for x in a.track.split(":"))
        picks_T = (0.5, 1, 2, 3, 5, 10)
        print(f"\nbody {a.track}: y displacement since t=0 (m) and rotation about world {a.axis} (rad) at t = {picks_T} s:")
        for path, S in sims.items():
            seq = S["animation"]["state_sequence"]
            h = S["args"]["timestep"]
            ys = np.array([st["rigid_bodies"][ri]["position"][1] for st in seq]) - seq[0]["rigid_bodies"][ri]["position"][1]
            ang = angles_about_axis([rotvec_to_matrix(st["rigid_bodies"][ri]["rotation"]) for st in seq], a.axis)
            picks = [int(round(T / h)) for T in picks_T if int(round(T / h)) < len(seq)]
            print(f"  {os.path.basename(path):<14} dy  " + "  ".join(f"{ys[k]:8.4f}" for k in picks))
            print(f"  {'':<14} ang " + "  ".join(f"{ang[k]:8.4f}" for k in picks))
        for path in sorted(glob.glob(os.path.join(a.dir, "rcipc_*.csv"))):
            R = rcipc_rows(path)
            t = np.array([r[0] for r in R])
            q = np.array([r[5 + 7 * oi: 5 + 7 * oi + 7] for r in R])
            ys = q[:, 1] - q[0, 1]
            ang = angles_about_axis([quat_to_matrix(row[3:7]) for row in q], a.axis)
            picks = [int(np.argmin(np.abs(t - T))) for T in picks_T if T <= t[-1] + 1e-9]
            print(f"  {os.path.basename(path):<14} dy  " + "  ".join(f"{ys[k]:8.4f}" for k in picks))
            print(f"  {'':<14} ang " + "  ".join(f"{ang[k]:8.4f}" for k in picks))
        print("  (Rigid-IPC's position is the centre of mass, ours the body origin: dy differs by the rotation of that offset; "
              "our t=0 row is the first step, not the initial state)")

    if a.final_y:
        print("\nfinal y of each body (m):")
        for path, S in sims.items():
            ys = [b["position"][1] for b in S["animation"]["state_sequence"][-1]["rigid_bodies"]]
            print(f"  {os.path.basename(path):<14} " + " ".join(f"{y:7.4f}" for y in ys))
        for path in sorted(glob.glob(os.path.join(a.dir, "rcipc_*.csv"))):
            q = rcipc_rows(path)[-1][5:]
            ys = [q[7 * k + 1] for k in range(len(q) // 7)]
            print(f"  {os.path.basename(path):<14} " + " ".join(f"{y:7.4f}" for y in ys) + "   (free bodies only)")


if __name__ == "__main__":
    main()
