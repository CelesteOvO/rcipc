"""用 libuipc（pyuipc 0.0.28，CUDA，ABD + 罚函数关节）跑 export_mjcf.py 导出的场景（笔记 notes/15_drift.md）。

用法（在装了 pyuipc 的 venv 里）：
    D:/Project/uipc-venv/Scripts/python.exe compare/libuipc/run_uipc_scene.py scene.json out --T 4 --dt 0.01
        [--gamma 100] [--kappa 1e8] [--dhat 0.001] [--eps_v 0.01] [--full] [--vel_tol 1e-3] [--newton] [--every 1]

默认是 libuipc 的默认设置（半隐式停止、velocity_tol 0.05 m/s），只改了 d-hat（和我们一样 1 mm）。
- gamma：关节罚函数的 strength_ratio，刚度 = gamma x (两边刚体的质量之和)（libuipc 的说明）。默认 100。
- kappa：ABD 正交性能量的刚度（Pa）。libuipc 的例子用 100 MPa。
- --newton：打开 libuipc 的计时器，从里面读每步的 Newton 次数（有同步开销，量墙钟时不要开）。
输出：
- out.csv：scene_demo 的格式（t, min_distance, min_limit, newton, pcg, qpos...），间隙记 nan（这里不量），
  qpos 记 qpos0（libuipc 的状态不是关节坐标）。tools/render.py 用它对齐时间。
- out.npz：t（帧数）和 T（帧数 x MJCF 的刚体数 x 4 x 4），每个刚体在世界里的仿射变换（世界是单位矩阵）。
  tools/joint_check.py 用它量关节漂移和变形，tools/render.py --affine 用它画出变形以后的样子。
"""
import argparse
import json
import os
import time

import numpy as np
from uipc import Engine, Logger, Scene, Timer, World, builtin, view
from uipc.constitution import AffineBodyConstitution, AffineBodyRevoluteJoint
from uipc.geometry import ground, label_surface, linemesh, trimesh

CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
# 顶点编号 = 4*(x>0) + 2*(y>0) + (z>0)；每个面两个三角形，法向朝外
FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int32)


def mesh_of(boxes):
    """几个盒子（在同一个坐标系里）合成一个三角网格。"""
    V, F = [], []
    for b in boxes:
        R, p, h = np.array(b["R"]), np.array(b["pos"]), np.array(b["half"])
        F.append(FACES + 8 * len(V))
        V.extend(p + (CORNERS * h) @ R.T)
    return np.array(V, float), np.concatenate(F).astype(np.int32)


def newton_count(rep):
    stack = [rep]
    while stack:
        nd = stack.pop()
        if nd.get("name") == "Newton Iteration":
            return int(nd["count"])
        stack += nd.get("children", [])
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out", help="输出的前缀：写 out.csv 和 out.npz")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=100.0)
    ap.add_argument("--kappa", type=float, default=1e8)
    ap.add_argument("--dhat", type=float, default=0.001)
    ap.add_argument("--eps_v", type=float, default=0.01)
    ap.add_argument("--full", action="store_true", help="关掉半隐式停止，收敛到 --vel_tol")
    ap.add_argument("--vel_tol", type=float, default=1e-3)
    ap.add_argument("--axis_half", type=float, default=0.05, help="铰链轴两个端点到锚点的距离（m）")
    ap.add_argument("--newton", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="每几步记一帧")
    ap.add_argument("--workspace", default=None)
    a = ap.parse_args()
    S = json.load(open(a.scene))

    Logger.set_level(Logger.Level.Warn)
    ws = a.workspace or os.path.join(os.path.dirname(os.path.abspath(a.out)), "uipc_ws")
    os.makedirs(ws, exist_ok=True)
    engine = Engine("cuda", ws)
    world = World(engine)
    cfg = Scene.default_config()
    cfg["dt"] = a.dt
    cfg["gravity"] = [[x] for x in S["gravity"]]
    cfg["contact"]["d_hat"] = a.dhat
    cfg["contact"]["eps_velocity"] = a.eps_v
    if a.full:
        cfg["newton"]["semi_implicit"]["enable"] = 0
        cfg["newton"]["velocity_tol"] = a.vel_tol
    scene = Scene(cfg)
    scene.contact_tabular().default_model(S["friction"], 1e9)
    abd = AffineBodyConstitution()

    nbody = max(b["id"] for b in S["bodies"]) + 1
    slot_of = {}
    for b in S["bodies"]:
        V, F = mesh_of(b["boxes"])
        sc = trimesh(V, F)
        abd.apply_to(sc, a.kappa, b["density"])
        label_surface(sc)
        view(sc.transforms())[0] = np.array(b["X0"])
        slot_of[b["id"]] = scene.objects().create(b["name"]).geometries().create(sc)[0]
    # 世界上的东西（固定的盒子、地面）之间不检测接触（墙的底面贴着地面；我们这边世界和世界之间也不检测）。
    ct = scene.contact_tabular()
    static = ct.create("static")
    ct.insert(static, static, S["friction"], 1e9, False)
    fixed = []
    for w in S["world"]:
        V, F = mesh_of(w["boxes"])
        sc = trimesh(V, F)
        abd.apply_to(sc, a.kappa, w["density"])
        label_surface(sc)
        static.apply_to(sc)
        view(sc.instances().find(builtin.is_fixed))[0] = 1
        fixed.append(scene.objects().create("world_" + w["name"]).geometries().create(sc)[0])
    for pl in S["planes"]:
        g = ground(pl["height"], np.array(pl["normal"]))
        static.apply_to(g)
        scene.objects().create("ground_" + pl["name"]).geometries().create(g)

    Vh, Eh, ls, li, rs, ri, st = [], [], [], [], [], [], []
    for j in S["joints"]:
        p, ax = np.array(j["anchor"]), np.array(j["axis"])
        Vh += [p - a.axis_half * ax, p + a.axis_half * ax]
        Eh.append([len(Vh) - 2, len(Vh) - 1])
        ls.append(fixed[j["world_partner"]] if j["parent"] == 0 else slot_of[j["parent"]])
        rs.append(slot_of[j["child"]])
        li.append(0)
        ri.append(0)
        st.append(a.gamma)
    if Eh:
        hm = linemesh(np.array(Vh, np.float32), np.array(Eh, np.int32))
        AffineBodyRevoluteJoint().apply_to(hm, ls, np.array(li, np.int32), rs, np.array(ri, np.int32), np.array(st, float))
        scene.objects().create("hinges").geometries().create(hm)

    if a.newton:
        Timer.enable_all()
    world.init(scene)
    assert world.is_valid(), "libuipc rejected the scene"

    def transforms():
        T = np.tile(np.eye(4), (nbody, 1, 1))
        for b, s in slot_of.items():
            T[b] = np.array(view(s.geometry().transforms())[0])
        return T

    steps = int(round(a.T / a.dt))
    ts, Ts, newton = [0.0], [transforms()], [0]
    t0 = time.perf_counter()
    for k in range(1, steps + 1):
        world.advance()
        world.retrieve()
        nw = -1
        if a.newton:
            rep = Timer.report_as_json()
            nw = newton_count(json.loads(rep) if isinstance(rep, str) else rep)
        if k % a.every == 0 or k == steps:
            ts.append(k * a.dt)
            Ts.append(transforms())
            newton.append(nw)
        if k % max(1, int(round(0.5 / a.dt))) == 0:
            print(f"t {k * a.dt:5.2f}  newton {nw:3d}  {time.perf_counter() - t0:7.1f} s", flush=True)
    wall = time.perf_counter() - t0
    nws = [x for x in newton[1:] if x >= 0]
    print(f"done: {steps} steps in {wall:.1f} s ({1e3 * wall / steps:.1f} ms/step)"
          + (f"; Newton per step mean {np.mean(nws):.2f}, max {max(nws)}" if nws else ""))
    q0 = ",".join(f"{x:.17g}" for x in S["qpos0"])
    with open(a.out + ".csv", "w") as fh:
        fh.write("t,min_distance,min_limit,newton,pcg,qpos...\n")
        for t, nw in zip(ts, newton):
            fh.write(f"{t:.6f},nan,nan,{nw},-1,{q0}\n")
    np.savez_compressed(a.out + ".npz", t=np.array(ts), T=np.array(Ts), wall=wall, steps=steps,
                        args=json.dumps(vars(a)))
    print(f"wrote {a.out}.csv and {a.out}.npz")


if __name__ == "__main__":
    main()
