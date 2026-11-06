"""用 scene_demo 跑 import_rigid_ipc.py 导出的场景，把结尾的统计抓成和 run_rigid_ipc.py 同样的几行（笔记 notes/compare_rigid_ipc.md）。

用法（在仓库根目录）：
    python compare/rigid_ipc/measure_rcipc.py compare/rigid_ipc/gears_single.xml --out compare/rigid_ipc/results/gears_single \
        --T 10 --h 0.01 --dhat 0.001 --mu 0 [--gpu 0|1] [--exe build/Release/scene_demo.exe] [--extra tol=1e-6]

轨迹写到 <out>/rcipc_<cpu|gpu>.csv（scene_demo 的 out=，每行 t、步末最小距离、限位、Newton、PCG、qpos），
统计写到 <out>/rcipc_<cpu|gpu>.json。时间是 scene_demo 自己量的每步均值（含 GPU 的同步，不含写文件）。
"""
import argparse
import json
import os
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exe", default="build/Release/scene_demo.exe")
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--dhat", type=float, required=True)
    ap.add_argument("--mu", type=float, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[], help="scene_demo 的其他 key=value")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tag = "gpu" if a.gpu else "cpu"
    csv_path = os.path.abspath(os.path.join(a.out, f"rcipc_{tag}.csv"))
    cmd = [os.path.abspath(a.exe), f"model={os.path.abspath(a.model)}", f"T={a.T:g}", f"h={a.h:g}", f"dhat={a.dhat:g}", f"mu={a.mu:g}",
           f"gpu={a.gpu}", f"out={csv_path}", *a.extra]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path = os.path.join(a.out, f"rcipc_{tag}.log")
    with open(log_path, "w") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        sys.exit(proc.returncode)
    out = proc.stdout
    m_done = re.search(r"done: (\d+) steps in ([\d.]+) s \(([\d.]+) ms/step\)", out)
    m_dist = re.search(r"min distance at step ends ([\d.eE+-]+) m", out)
    m_newton = re.search(r"Newton per step max (\d+) mean ([\d.]+) \((\d+) hit the cap\); PCG per step mean ([\d.]+)", out)
    if not (m_done and m_dist and m_newton):
        print(out[-2000:])
        sys.exit("could not parse scene_demo output")
    S = dict(steps=int(m_done.group(1)), total_s=float(m_done.group(2)), ms_mean=float(m_done.group(3)),
             min_dist=float(m_dist.group(1)), newton_max=int(m_newton.group(1)), newton_mean=float(m_newton.group(2)),
             newton_capped=int(m_newton.group(3)), pcg_mean=float(m_newton.group(4)), solver=tag,
             args=dict(T=a.T, h=a.h, dhat=a.dhat, mu=a.mu, extra=a.extra))
    with open(os.path.join(a.out, f"rcipc_{tag}.json"), "w") as f:
        json.dump(S, f, indent=1)
    print(f"{a.model} ({tag}): {S['steps']} steps")
    print(f"  ms/step: mean {S['ms_mean']:.1f}  total {S['total_s']:.1f} s")
    print(f"  newton/step: mean {S['newton_mean']:.2f}  max {S['newton_max']}  (hit the cap: {S['newton_capped']})  "
          f"PCG/step mean {S['pcg_mean']:.1f}")
    print(f"  min distance at step ends: {S['min_dist']:.3e}")
    print(f"  log: {log_path}")


if __name__ == "__main__":
    main()
