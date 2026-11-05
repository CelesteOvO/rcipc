"""量一段轨迹里的关节漂移和刚体变形，两边用同一个函数（笔记 notes/15_drift.md）。

用法：
    python tools/joint_check.py models/wrecking_ball.xml ours.csv           [--status ours.status] [--json out.json]
    python tools/joint_check.py models/wrecking_ball.xml uipc.csv --affine uipc.npz [--status ...] [--json ...]

输入：
- 我们（scene_demo 的 CSV）：每帧的 qpos，用 MuJoCo 的正运动学得到每个刚体的位姿（转动矩阵 + 位置）。
- libuipc（compare/libuipc/run_uipc_scene.py 的 npz）：每帧每个刚体的仿射变换 [A | p]，A 不一定是转动矩阵。
所以“漂移”和“变形”对我们来说只是检查（应该是舍入误差），对 libuipc 是真的误差。

每帧量：
- 关节漂移：每个铰链在父刚体上的锚点和在子刚体上的锚点之间的距离（父刚体是世界时，就是固定的锚点）。
- 铰链轴偏差：父、子刚体上铰链轴的夹角。
- 变形：每个刚体 A 的奇异值离 1 最远的那个（主应变），和 ||A^T A - I||_F（和 metrics.py 一样）。
- 链长：沿铰链走一遍的折线长度：转轴 → 第 0 节的近端锚点 → 第 0 节的远端锚点（下一个铰链在第 0 节上的位置）
  → 第 1 节的近端锚点 → ... → 摆锤上的锚点。等于各节钢条（铰链到铰链）的长度之和加上各关节拉开的距离。
  我们这边正好是名义长度；libuipc 的会变长。分开报：关节拉开多少、钢条伸长多少。
摆锤场景（有名字是 weight 的刚体和 wall 的几何体时）另外量：
- 摆锤有没有碰到墙：摆锤（按它的仿射变换算的盒子）和墙之间距离的下界（分离轴）小于 2 mm 就算碰到。
- 越过墙的次数：摆锤的质心从 x > 0 到 x < 0（或反过来）一次算一次。
--status：每帧一行“t<TAB>文字”，给 tools/render.py --status 写在画面左下角；碰到墙以后以 ! 开头（画成红色）。
"""
import argparse
import csv
import json
import sys

import mujoco
import numpy as np


def load_csv(path, nq):
    ts, qs = [], []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            ts.append(float(row[0]))
            qs.append(np.array([float(x) for x in row[5:]]))
            if qs[-1].size != nq:
                sys.exit(f"{path}: row has {qs[-1].size} qpos, model nq = {nq}")
    return np.array(ts), np.array(qs)


def box_gap(Ra, ca, ha, Rb, cb, hb):
    """两个（有向）盒子之间距离的下界：15 个分离轴上投影间隔的最大值。"""
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    axes += [np.cross(Ra[:, i], Rb[:, j]) for i in range(3) for j in range(3)]
    best = -np.inf
    c = cb - ca
    for u in axes:
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u = u / nu
        best = max(best, abs(c @ u) - np.abs(Ra.T @ u) @ ha - np.abs(Rb.T @ u) @ hb)
    return best


class Scene:
    def __init__(self, path):
        m = self.m = mujoco.MjModel.from_xml_path(path)
        d = self.d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        self.X0 = self.poses()  # qpos0 时每个刚体的位姿
        # 每个铰链：在父、子刚体坐标系里的锚点和轴
        self.joints = []
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            c = m.jnt_bodyid[j]
            p = m.body_parentid[c]
            a_w, x_w = d.xanchor[j].copy(), d.xaxis[j].copy()
            self.joints.append(dict(parent=p, child=c, a_p=self.local(p, a_w), a_c=self.local(c, a_w),
                                    x_p=np.linalg.solve(self.X0[p][:3, :3], x_w), x_c=np.linalg.solve(self.X0[c][:3, :3], x_w)))
        self.weight = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "weight")
        self.wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall")
        self.gweight = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "weight")
        # 链：从世界开始，按父子顺序排好的铰链（每个刚体至多一个铰链时就是一条链）
        self.chain = sorted(self.joints, key=lambda jt: self.depth(jt["child"]))
        self.nominal = self.chain_length(self.X0)[0] if self.chain else 0.0

    def depth(self, b):
        k = 0
        while b != 0:
            b = self.m.body_parentid[b]
            k += 1
        return k

    def poses(self):
        X = np.tile(np.eye(4), (self.m.nbody, 1, 1))
        X[:, :3, :3] = self.d.xmat.reshape(-1, 3, 3)
        X[:, :3, 3] = self.d.xpos
        return X

    def local(self, b, x):
        return np.linalg.solve(self.X0[b], np.r_[x, 1.0])[:3]

    @staticmethod
    def apply(X, x):
        return X[:3, :3] @ x + X[:3, 3]

    def chain_length(self, X):
        """沿链的折线长度，以及其中“关节拉开”的部分（返回 total, gaps, bars）。"""
        total = gaps = bars = 0.0
        prev = self.apply(X[self.chain[0]["parent"]], self.chain[0]["a_p"])
        for k, jt in enumerate(self.chain):
            pa = self.apply(X[jt["parent"]], jt["a_p"])
            ca = self.apply(X[jt["child"]], jt["a_c"])
            if k > 0:
                bars += np.linalg.norm(pa - prev)    # 父刚体上：上一个锚点到这一个锚点
            gaps += np.linalg.norm(ca - pa)
            prev = ca
        total = gaps + bars
        return total, gaps, bars

    def measure(self, X):
        out = dict(drift=0.0, axis_deg=0.0, strain=0.0, orth=0.0)
        for jt in self.joints:
            pa = self.apply(X[jt["parent"]], jt["a_p"])
            ca = self.apply(X[jt["child"]], jt["a_c"])
            out["drift"] = max(out["drift"], float(np.linalg.norm(pa - ca)))
            xp = X[jt["parent"]][:3, :3] @ jt["x_p"]
            xc = X[jt["child"]][:3, :3] @ jt["x_c"]
            cosang = xp @ xc / (np.linalg.norm(xp) * np.linalg.norm(xc))
            out["axis_deg"] = max(out["axis_deg"], float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
        for b in range(1, self.m.nbody):
            A = X[b][:3, :3]
            sv = np.linalg.svd(A, compute_uv=False)
            out["strain"] = max(out["strain"], float(np.abs(sv - 1).max()))
            out["orth"] = max(out["orth"], float(np.linalg.norm(A.T @ A - np.eye(3))))
        if self.chain:
            out["length"], out["gaps"], out["bars"] = self.chain_length(X)
        if self.weight >= 0 and self.wall >= 0:
            m, d = self.m, self.d
            Xw = X[self.weight]
            # 摆锤盒子：几何体在刚体坐标系里的位置和朝向（从 qpos0 的位姿反推），再用这一帧的仿射变换映到世界。
            gl = self.local(self.weight, self.X0_geom_pos)
            Rl = self.X0[self.weight][:3, :3].T @ self.X0_geom_mat
            M = Xw[:3, :3] @ Rl
            U, sv, Vt = np.linalg.svd(M)
            R = U @ Vt
            half = m.geom_size[self.gweight] * np.linalg.norm(M, axis=0)
            Rwall = d.geom_xmat[self.wall].reshape(3, 3)
            out["wall_gap"] = float(box_gap(R, self.apply(Xw, gl), half, Rwall, d.geom_xpos[self.wall].copy(),
                                            m.geom_size[self.wall]))
            out["weight_x"] = float(self.apply(Xw, gl)[0])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("csv")
    ap.add_argument("--affine", help="libuipc 的 npz（run_uipc_scene.py 写的）：用它的仿射变换，不用 CSV 里的 qpos")
    ap.add_argument("--status")
    ap.add_argument("--json")
    a = ap.parse_args()
    S = Scene(a.model)
    m, d = S.m, S.d
    if S.gweight >= 0:
        S.X0_geom_pos = d.geom_xpos[S.gweight].copy()
        S.X0_geom_mat = d.geom_xmat[S.gweight].reshape(3, 3).copy()
        mujoco.mj_kinematics(m, d)
    if a.affine:
        z = np.load(a.affine)
        ts, Xs = z["t"], z["T"]
    else:
        ts, qs = load_csv(a.csv, m.nq)
        Xs = []
        for q in qs:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            Xs.append(S.poses())
        Xs = np.array(Xs)
        d.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, d)
    rows, status = [], []
    touched_at, passes, side = None, 0, None
    for t, X in zip(ts, Xs):
        r = S.measure(X)
        r["t"] = float(t)
        if "wall_gap" in r:
            if touched_at is None and r["wall_gap"] < 0.002:
                touched_at = float(t)
            s = np.sign(r["weight_x"])
            if side is not None and s != 0 and s != side:
                passes += 1
            side = s if s != 0 else side
            r["passes"] = passes
        rows.append(r)
        if "length" in r:
            txt = (f"链长 {r['length'] * 1e3:.0f} mm（关节 +{r['gaps'] * 1e3:.0f}，"
                   f"钢条 {(r['bars'] - S.nominal) * 1e3:+.0f}）")
            if "wall_gap" in r:
                txt += f"越墙 {passes} 次" if touched_at is None else "撞墙"
            status.append((float(t), ("!" if touched_at is not None else "") + txt))
    keys = ["drift", "axis_deg", "strain", "orth"] + (["length", "gaps", "bars"] if S.chain else [])
    mx = {k: max(r[k] for r in rows) for k in keys}
    print(f"{a.csv}: {len(rows)} frames, t = {ts[0]:.3f}..{ts[-1]:.3f} s")
    print(f"  max joint drift {mx['drift'] * 1e3:.3f} mm, max hinge-axis deviation {mx['axis_deg']:.4f} deg, "
          f"max principal strain {mx['strain'] * 100:.4f} %, max ||A^T A - I|| {mx['orth']:.2e}")
    if S.chain:
        i = int(np.argmax([r["length"] for r in rows]))
        r = rows[i]
        print(f"  chain (nominal {S.nominal * 1e3:.1f} mm): longest {r['length'] * 1e3:.1f} mm at t = {r['t']:.2f} s "
              f"(joints opened {r['gaps'] * 1e3:.1f} mm, bars {(r['bars'] - S.nominal) * 1e3:+.1f} mm)")
    if S.weight >= 0 and S.wall >= 0:
        print(f"  weight: closest to the wall {min(r['wall_gap'] for r in rows) * 1e3:.1f} mm; "
              + (f"touched the wall at t = {touched_at:.2f} s" if touched_at is not None else "never touched the wall")
              + f"; passed over the wall {passes} times")
    if a.status:
        with open(a.status, "w", encoding="utf-8", newline="\n") as fh:
            for t, txt in status:
                fh.write(f"{t:.6f}\t{txt}\n")
    if a.json:
        json.dump(dict(max=mx, nominal=S.nominal, touched_at=touched_at, passes=passes, rows=rows), open(a.json, "w"))
    sys.exit(0 if touched_at is None else 1)


if __name__ == "__main__":
    main()
