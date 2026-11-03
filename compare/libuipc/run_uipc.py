"""libuipc（pyuipc 0.0.28，CUDA，ABD + 罚函数关节）跑布娃娃堆，记录和 measure_rcipc.py 一样的指标。

改自旧分支的 research/rcipc/run_uipc_ragdoll.py：场景搭法相同，多记了每步的 Newton 次数（从 libuipc 的计时器读），
指标用共享的 metrics.py。要在装了 pyuipc 的 venv 里运行：

    D:/Project/uipc-venv/Scripts/python.exe compare/libuipc/run_uipc.py 8 300 out.json --workspace <临时目录>
    ...                                                  --full        # 两边都完全收敛的那组设置

默认是 libuipc 的默认设置：半隐式停止（K_min 6、beta_tol 1e-3）、velocity_tol 0.05 m/s、d_hat 1 cm、eps_velocity 1 cm/s。
--full 关掉半隐式停止，velocity_tol 改成 --vel-tol（默认 1e-3 m/s）。
每步的墙钟时间也记下了，但计时器打开时有同步开销，而且这是 GPU，不和 CPU 的 RC-IPC 比。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402
import ragdoll_scene as R  # noqa: E402

from uipc import Engine, Logger, Scene, Timer, World, view  # noqa: E402
from uipc.constitution import AffineBodyConstitution, AffineBodyRevoluteJoint, AffineBodySphericalJoint  # noqa: E402
from uipc.geometry import ground, label_surface, linemesh, trimesh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("steps", type=int)
ap.add_argument("out")
ap.add_argument("--kappa", type=float, default=1e8, help="ABD 的刚度（仿射体抵抗变形的强度）")
ap.add_argument("--gamma", type=float, default=100.0, help="关节罚函数的强度")
ap.add_argument("--full", action="store_true", help="关掉半隐式停止，完全收敛")
ap.add_argument("--vel-tol", type=float, default=1e-3)
ap.add_argument("--workspace", required=True)
args = ap.parse_args()

Logger.set_level(Logger.Level.Warn)
os.makedirs(args.workspace, exist_ok=True)
engine = Engine("cuda", args.workspace)
world = World(engine)
config = Scene.default_config()
config["dt"] = R.DT
config["gravity"] = [[0.0], [-9.8], [0.0]]
if args.full:
    config["newton"]["semi_implicit"]["enable"] = 0
    config["newton"]["velocity_tol"] = args.vel_tol
scene = Scene(config)
scene.contact_tabular().default_model(R.MU, 1e9)

bodies = R.body_rest()
G = R.placements(args.n)
abd = AffineBodyConstitution()
obj = scene.objects().create("ragdolls")
slots = []
for b in bodies:
    sc = trimesh(b["V"].astype(np.float64), b["F"].astype(np.int32))
    abd.apply_to(sc, args.kappa, R.DENSITY)
    label_surface(sc)
    sc.instances().resize(args.n)
    tv = view(sc.transforms())
    for k in range(args.n):
        tv[k] = G[k] @ b["T"]
    slots.append(obj.geometries().create(sc)[0])
scene.objects().create("ground").geometries().create(ground(0.0))

joints = metrics.JOINTS


def world_pt(k, j):
    return (G[k] @ np.r_[j["anchor"], 1.0])[:3]


sph = AffineBodySphericalJoint()
pos, ls, li, rs, ri, st = [], [], [], [], [], []
for k in range(args.n):
    for j in joints:
        if j["type"] != "ball":
            continue
        pos.append(world_pt(k, j))
        ls.append(slots[j["parent"]]); li.append(k); rs.append(slots[j["child"]]); ri.append(k); st.append(args.gamma)
jm = sph.create_geometry(np.array(pos, dtype=np.float64), ls, np.array(li, dtype=np.int32), rs,
                         np.array(ri, dtype=np.int32), np.array(st, dtype=np.float64))
scene.objects().create("ball_joints").geometries().create(jm)

rev = AffineBodyRevoluteJoint()
Vh, Eh, ls, li, rs, ri, st = [], [], [], [], [], [], []
for k in range(args.n):
    for j in joints:
        if j["type"] != "hinge":
            continue
        a = world_pt(k, j)
        ax = G[k][:3, :3] @ j["axis"]
        Vh += [a - 0.05 * ax, a + 0.05 * ax]
        Eh.append([len(Vh) - 2, len(Vh) - 1])
        ls.append(slots[j["parent"]]); li.append(k); rs.append(slots[j["child"]]); ri.append(k); st.append(args.gamma)
hm = linemesh(np.array(Vh, dtype=np.float32), np.array(Eh, dtype=np.int32))
rev.apply_to(hm, ls, li, rs, ri, st)
scene.objects().create("hinge_joints").geometries().create(hm)

Timer.enable_all()
world.init(scene)
assert world.is_valid()


def transforms():
    return [np.array(view(s.geometry().transforms())) for s in slots]


def newton_count(report):
    """计时器报告里 "Newton Iteration" 节点的 count 就是这一步的 Newton 次数。"""
    node = json.loads(report) if isinstance(report, str) else report
    stack = [node]
    while stack:
        n = stack.pop()
        if n.get("name") == "Newton Iteration":
            return int(n["count"])
        stack += n.get("children", [])
    return -1


rec = dict(engine="libuipc", n=args.n, steps=args.steps, dt=R.DT, kappa_abd=args.kappa, gamma=args.gamma,
           full=args.full, config=json.loads(str(config)) if not isinstance(config, dict) else config,
           newton=[], step_s=[], samples=[dict(step=0, **metrics.measure(transforms(), "mesh"))])
for s in range(1, args.steps + 1):
    t0 = time.perf_counter()
    world.advance()
    world.retrieve()
    rec["step_s"].append(time.perf_counter() - t0)
    rec["newton"].append(newton_count(Timer.report_as_json()))
    if s % 10 == 0 or s == args.steps:
        m = metrics.measure(transforms(), "mesh")
        rec["samples"].append(dict(step=s, **m))
        print(f"step {s:4d}  newton {rec['newton'][-1]:3d}  drift {m['drift_mm']:.3f} mm  axis {m['axis_deg']:.3f} deg  "
              f"orth {m['orth']:.2e}  lowest {m['lowest_mm']:.2f} mm", flush=True)
with open(args.out, "w") as f:
    json.dump(rec, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("wrote", args.out)
