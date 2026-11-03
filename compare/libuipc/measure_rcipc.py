"""把 scene_demo 写出的 RC-IPC 轨迹换算成和 run_uipc.py 一样的指标（每 10 步一次），写成同样格式的 JSON。

用法（系统 Python，需要 mujoco 和 numpy；只用 MuJoCo 做正运动学，不做物理）：
    python compare/libuipc/measure_rcipc.py compare/libuipc/ragdolls_8.xml rcipc.csv rcipc.json

scene_demo 的 CSV 每行：t, min_distance, min_limit, newton, pcg, 然后是 nq 个 qpos。
"""
import csv
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402
import ragdoll_scene as R  # noqa: E402


def main(xml, csv_path, out):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    names = [b[0] for b in R.BODIES]
    n = sum(1 for i in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i).startswith("pelvis_"))
    # body_ids[b][k]：第 k 个布娃娃的刚体 b 在 MuJoCo 里的编号；Rb[b]：我们的刚体坐标系到 ragdoll_scene 坐标系的固定转动。
    body_ids = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}_{k}") for k in range(n)] for name in names]
    Rb = [metrics.BODIES[b]["T"][:3, :3] for b in range(len(names))]

    def transforms(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        T = []
        for b in range(len(names)):
            Tb = np.zeros((n, 4, 4))
            for k, i in enumerate(body_ids[b]):
                Tb[k, :3, :3] = d.xmat[i].reshape(3, 3) @ Rb[b]
                Tb[k, :3, 3] = d.xpos[i]
                Tb[k, 3, 3] = 1
            T.append(Tb)
        return T

    rec = dict(engine="rcipc", n=n, dt=float(m.opt.timestep), newton=[], pcg=[], min_distance=[], samples=[])
    rec["samples"].append(dict(step=0, **metrics.measure(transforms(m.qpos0), "capsule")))
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for s, row in enumerate(reader, start=1):
            rec["min_distance"].append(float(row[1]))
            rec["newton"].append(int(row[3]))
            rec["pcg"].append(int(row[4]))
            if s % 10 == 0:
                q = np.array([float(x) for x in row[5:]])
                rec["samples"].append(dict(step=s, **metrics.measure(transforms(q), "capsule")))
    rec["steps"] = len(rec["newton"])
    with open(out, "w") as f:
        json.dump(rec, f)
    print("wrote", out, "steps", rec["steps"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
