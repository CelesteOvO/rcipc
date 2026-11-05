"""查一段轨迹里有没有“改变拓扑”的穿透（笔记 notes/14_visible.md）。

用法：
    python tools/topology.py chain   models/ring_chain.xml traj.csv
    python tools/topology.py blender models/blender.xml    traj.csv

CSV 是 scene_demo 或 tools/mujoco_traj.py 写的轨迹（t, min_distance, min_limit, newton, pcg, qpos...）。
只用 MuJoCo 算正运动学，不做物理计算。

chain：每个环（名字是 ring<i> 的刚体）的中心线是一个闭合的多边形（12 段胶囊的轴首尾相连）。
    两个闭合曲线的环绕数（linking number）是整数，只有当两条曲线相交时才会变：对于胶囊，就是两根线的轴
    碰到一起，穿透深度至少是线的直径。所以环绕数变了 <=> 有一对环“穿过去了”。
    算法：沿一个一般的方向投影，数两条曲线在投影里的交叉，Lk = (所有交叉的符号之和) / 2。
    符号：上面那段的方向 x 下面那段的方向，和视线方向同向为 +1。
    另外查第 0 个环还套在横杆上：横杆的轴线穿过第 0 个环围成的多边形。
blender：每个球（名字是 ball<i> 的刚体）的球心在不在罐子里：离转轴不超过罐子墙的外表面、
    不高于盖子的上表面（罐子的尺寸从名字是 wall0、lid 的几何体读）。

打印：第一次出事的时刻、最后一帧的状态。退出码 0 = 整段拓扑不变，1 = 变了。
--status 文件：每帧一行“t<TAB>文字”，给 tools/render.py --status 写在画面上（出事的行以 ! 开头）。
"""
import argparse
import csv
import sys

import mujoco
import numpy as np


def load(path, nq):
    ts, qs = [], []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            q = np.array([float(x) for x in row[5:]])
            if q.size != nq:
                sys.exit(f"{path}: row has {q.size} qpos, model nq = {nq}")
            ts.append(float(row[0]))
            qs.append(q)
    return np.array(ts), np.array(qs)


def bodies_named(m, prefix):
    out = []
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            out.append((int(name[len(prefix):]), b))
    return [b for _, b in sorted(out)]


def capsule_start(d, g, m):
    """胶囊轴的一个端点（fromto 的起点：geom 的 z 轴从起点指向终点）。"""
    z = d.geom_xmat[g].reshape(3, 3)[:, 2]
    return d.geom_xpos[g] - z * m.geom_size[g, 1]


def ring_polygons(m, d, rings):
    polys = []
    for b in rings:
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        polys.append(np.array([capsule_start(d, g, m) for g in gs]))
    return polys


def linking_number(P, Q, view):
    """两个闭合多边形的环绕数：沿 view 投影，数 P、Q 之间的交叉。"""
    e1 = np.cross(view, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(view, e1)
    def proj(X):
        return np.stack([X @ e1, X @ e2], axis=1), X @ view
    p2, pd = proj(P)
    q2, qd = proj(Q)
    nP, nQ = len(P), len(Q)
    total = 0
    for i in range(nP):
        a0, a1 = p2[i], p2[(i + 1) % nP]
        ua = a1 - a0
        for j in range(nQ):
            b0, b1 = q2[j], q2[(j + 1) % nQ]
            ub = b1 - b0
            den = ua[0] * ub[1] - ua[1] * ub[0]
            if abs(den) < 1e-18:
                continue
            w = b0 - a0
            s = (w[0] * ub[1] - w[1] * ub[0]) / den   # P 上的参数
            t = (w[0] * ua[1] - w[1] * ua[0]) / den   # Q 上的参数
            if not (0 <= s < 1 and 0 <= t < 1):
                continue
            # 交叉点在视线方向上的深度：大的离观察者近，在上面。
            dp = pd[i] + s * (pd[(i + 1) % nP] - pd[i])
            dq = qd[j] + t * (qd[(j + 1) % nQ] - qd[j])
            up, dn = (P[(i + 1) % nP] - P[i], Q[(j + 1) % nQ] - Q[j]) if dp > dq else \
                     (Q[(j + 1) % nQ] - Q[j], P[(i + 1) % nP] - P[i])
            total += 1 if np.dot(np.cross(up, dn), view) > 0 else -1
    return total / 2


def connected_to_first(Lk, n):
    """按环绕数非零的对连边，从第 0 个环能到达的环。"""
    link = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if Lk[i, j] != 0:
                link[i].add(j)
                link[j].add(i)
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in link[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def chain(m, ts, qs, status):
    d = mujoco.MjData(m)
    rings = bodies_named(m, "ring")
    bar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bar")
    view = np.array([0.3127, 0.5413, 0.7805])  # 一个“一般”的方向，避开坐标轴和环所在的平面
    view /= np.linalg.norm(view)
    n = len(rings)
    first, Lk0, changed = None, None, []
    for k, q in enumerate(qs):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        P = ring_polygons(m, d, rings)
        # 只算有可能缠在一起的对：中心距小于两个外接球半径之和。
        Lk = np.zeros((n, n))
        ctr = [p.mean(0) for p in P]
        rad = [np.max(np.linalg.norm(p - c, axis=1)) for p, c in zip(P, ctr)]
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(ctr[i] - ctr[j]) < rad[i] + rad[j]:
                    Lk[i, j] = round(linking_number(P[i], P[j], view))
        # 第 0 个环还套在横杆上：横杆的轴线和环的平面的交点在多边形里面（用到中心的距离近似，内切圆半径）。
        z = d.geom_xmat[bar].reshape(3, 3)[:, 2]
        c0 = ctr[0]
        nrm = np.cross(P[0][1] - P[0][0], P[0][2] - P[0][0])
        nrm /= np.linalg.norm(nrm)
        den = np.dot(z, nrm)
        on_bar = False
        if abs(den) > 1e-9:
            s = np.dot(c0 - d.geom_xpos[bar], nrm) / den
            hit = d.geom_xpos[bar] + s * z
            on_bar = abs(s) <= m.geom_size[bar, 1] and np.linalg.norm(hit - c0) < rad[0] * np.cos(np.pi / 12)
        if Lk0 is None:
            Lk0 = Lk.copy()
            adj = [abs(Lk0[i, i + 1]) for i in range(n - 1)]
            others = np.abs(Lk0).sum() - sum(adj)
            print(f"start: {n} rings, adjacent |Lk| = {sorted(set(adj))}, other pairs sum |Lk| = {others:g}, "
                  f"ring 0 on the bar: {on_bar}")
        diff = [(i, j, int(Lk0[i, j]), int(Lk[i, j])) for i in range(n) for j in range(i + 1, n) if Lk[i, j] != Lk0[i, j]]
        if (diff or not on_bar) and first is None:
            first = ts[k]
            print(f"t = {ts[k]:.3f} s: first change: " + ", ".join(f"rings {i}-{j} Lk {a} -> {b}" for i, j, a, b in diff)
                  + ("" if on_bar else "; ring 0 is off the bar"))
        changed = diff
        seen = connected_to_first(Lk, n)
        if not diff and on_bar:
            status.append((ts[k], f"拓扑没变：{n} 个环都还套在一起"))
        else:
            status.append((ts[k], f"!拓扑变了：{len(diff)} 对环互相穿过，{n - len(seen)} 个环已经和横杆断开"
                           + ("" if on_bar else "，第 0 个环掉下横杆")))
    print(f"end (t = {ts[-1]:.3f} s): {len(changed)} pairs changed linking; rings still connected to ring 0: "
          f"{len(seen)} of {n}; ring 0 on the bar: {on_bar}")
    if first is None:
        print("topology unchanged over the whole trajectory")
    return first is None


def blender(m, ts, qs, status):
    d = mujoco.MjData(m)
    balls = bodies_named(m, "ball")
    w0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall0")
    lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lid")
    r_out = np.linalg.norm(m.geom_pos[w0, :2]) + m.geom_size[w0, 1]   # 墙外表面到转轴的距离（12 边形的边心距）
    z_top = m.geom_pos[lid, 2] + m.geom_size[lid, 2]
    nb = len(balls)
    first, outside = None, []
    escaped_at = {}
    for k, q in enumerate(qs):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        outside = []
        for i, b in enumerate(balls):
            p = d.xpos[b]
            # 12 边形：把球心转到它所在的那一块墙的法向上，算到墙外表面的距离。
            phi = np.arctan2(p[1], p[0])
            sector = np.round(phi / (2 * np.pi / 12)) * (2 * np.pi / 12)
            radial = p[0] * np.cos(sector) + p[1] * np.sin(sector)
            if radial > r_out or p[2] > z_top or p[2] < 0:
                outside.append(i)
                escaped_at.setdefault(i, ts[k])
        status.append((ts[k], f"!{len(escaped_at)} 个球穿到了罐子外面" if escaped_at else f"{nb} 个球都在罐子里"))
        if outside and first is None:
            first = ts[k]
            print(f"t = {ts[k]:.3f} s: first ball outside the jar: {outside}")
    print(f"end (t = {ts[-1]:.3f} s): {len(outside)} of {nb} balls outside the jar; ever outside: {len(escaped_at)}"
          + (" (" + ", ".join(f"ball {i} at {t:.3f} s" for i, t in sorted(escaped_at.items(), key=lambda x: x[1])) + ")"
             if escaped_at else ""))
    if first is None:
        print("all balls stayed inside the jar")
    return first is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=["chain", "blender"])
    ap.add_argument("model")
    ap.add_argument("csv")
    ap.add_argument("--status", default=None)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(a.model)
    ts, qs = load(a.csv, m.nq)
    status = []
    ok = chain(m, ts, qs, status) if a.scene == "chain" else blender(m, ts, qs, status)
    if a.status:
        with open(a.status, "w", encoding="utf-8", newline="\n") as fh:
            for t, txt in status:
                fh.write(f"{t:.6f}\t{txt}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
