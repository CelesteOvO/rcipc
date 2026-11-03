# 复制自旧分支 lyf/research/rcipc 的 research/rcipc/scenes/ragdoll.py（G0 对比时 libuipc、MuJoCo 共用的场景定义），
# 没有改动。make_mjcf.py 用它生成我们的 MJCF，run_uipc.py 用它搭 libuipc 的场景，所以两边的几何、关节、初始姿态完全一样。
"""Ragdoll pile shared by every G0 baseline.

Mirrors UNBM Fig. 11 (60 ragdolls, 11 bodies and 10 joints each, dt = 1/100, mu = 0.2):
neck, waist, shoulders and hips are ball joints, elbows and knees hinges.  UNBM limits every joint
with cone-twist ranges; libuipc has no spherical limit, so G0 runs without joint limits on every
engine.  Bodies are capsules with gaps of >= 1.5 cm at the joints so no pair starts inside d_hat.

Rest pose: T-pose standing on y = 0, y up, facing +z, metres.
"""
import numpy as np

# name, p0, p1, radius (capsule axis end points in rest-pose world coordinates)
BODIES = [
    ("pelvis",    (-0.08, 0.95, 0.0), (0.08, 0.95, 0.0), 0.09),
    ("torso",     (0.0, 1.18, 0.0),   (0.0, 1.38, 0.0),  0.11),
    ("head",      (0.0, 1.65, 0.0),   (0.0, 1.65, 0.0),  0.10),
    ("uarm_r",    (0.20, 1.42, 0.0),  (0.42, 1.42, 0.0), 0.045),
    ("farm_r",    (0.52, 1.42, 0.0),  (0.72, 1.42, 0.0), 0.04),
    ("uarm_l",    (-0.20, 1.42, 0.0), (-0.42, 1.42, 0.0), 0.045),
    ("farm_l",    (-0.52, 1.42, 0.0), (-0.72, 1.42, 0.0), 0.04),
    ("thigh_r",   (0.10, 0.78, 0.0),  (0.10, 0.50, 0.0), 0.06),
    ("shin_r",    (0.10, 0.37, 0.0),  (0.10, 0.10, 0.0), 0.05),
    ("thigh_l",   (-0.10, 0.78, 0.0), (-0.10, 0.50, 0.0), 0.06),
    ("shin_l",    (-0.10, 0.37, 0.0), (-0.10, 0.10, 0.0), 0.05),
]
BODY_INDEX = {b[0]: i for i, b in enumerate(BODIES)}

# parent, child, type, anchor (rest world), axis (hinge only)
JOINTS = [
    ("pelvis", "torso",   "ball",  (0.0, 1.055, 0.0),     None),
    ("torso",  "head",    "ball",  (0.0, 1.52, 0.0),      None),
    ("torso",  "uarm_r",  "ball",  (0.135, 1.42, 0.0),    None),
    ("uarm_r", "farm_r",  "hinge", (0.4725, 1.42, 0.0),   (0.0, 1.0, 0.0)),
    ("torso",  "uarm_l",  "ball",  (-0.135, 1.42, 0.0),   None),
    ("uarm_l", "farm_l",  "hinge", (-0.4725, 1.42, 0.0),  (0.0, 1.0, 0.0)),
    ("pelvis", "thigh_r", "ball",  (0.10, 0.85, 0.0),     None),
    ("thigh_r", "shin_r", "hinge", (0.10, 0.435, 0.0),    (1.0, 0.0, 0.0)),
    ("pelvis", "thigh_l", "ball",  (-0.10, 0.85, 0.0),    None),
    ("thigh_l", "shin_l", "hinge", (-0.10, 0.435, 0.0),   (1.0, 0.0, 0.0)),
]

DENSITY = 1000.0
MU = 0.2
DT = 0.01


def capsule_mesh(L, r, nseg=8, nring=2):
    """Closed capsule along local z, centred at the origin, half length L/2; outward triangles."""
    verts = [(0.0, 0.0, -L / 2 - r)]
    rings = []
    zs = []
    for k in range(1, nring + 1):                      # bottom hemisphere up to the equator
        phi = -np.pi / 2 + k * (np.pi / 2) / nring
        rings.append(r * np.cos(phi))
        zs.append(-L / 2 + r * np.sin(phi))
    start = 0 if L > 1e-12 else 1                      # a sphere shares one equator
    for k in range(start, nring):                      # top hemisphere from the equator up
        phi = k * (np.pi / 2) / nring
        rings.append(r * np.cos(phi))
        zs.append(L / 2 + r * np.sin(phi))
    for rr, z in zip(rings, zs):
        for s in range(nseg):
            a = 2 * np.pi * s / nseg
            verts.append((rr * np.cos(a), rr * np.sin(a), z))
    verts.append((0.0, 0.0, L / 2 + r))
    V = np.array(verts)
    nr = len(rings)
    F = []
    ring0 = lambda i: 1 + i * nseg
    for s in range(nseg):                               # bottom fan
        F.append((0, ring0(0) + (s + 1) % nseg, ring0(0) + s))
    for i in range(nr - 1):
        for s in range(nseg):
            a, b = ring0(i) + s, ring0(i) + (s + 1) % nseg
            c, d = ring0(i + 1) + s, ring0(i + 1) + (s + 1) % nseg
            F.append((a, b, d))
            F.append((a, d, c))
    top = len(V) - 1
    for s in range(nseg):                               # top fan
        F.append((top, ring0(nr - 1) + s, ring0(nr - 1) + (s + 1) % nseg))
    F = np.array(F, dtype=np.int32)
    # orient outward
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    if vol < 0:
        F = F[:, [0, 2, 1]]
    return V, F


def _frame_from_axis(d):
    """Rotation whose local z is unit vector d."""
    z = d / np.linalg.norm(d)
    t = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(t, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def body_rest():
    """Per body: local mesh (V, F) and rest-pose transform (4x4) with the body origin at its centre."""
    out = []
    for name, p0, p1, r in BODIES:
        p0 = np.array(p0); p1 = np.array(p1)
        L = np.linalg.norm(p1 - p0)
        R = _frame_from_axis(p1 - p0) if L > 1e-12 else np.eye(3)
        V, F = capsule_mesh(L, r)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = (p0 + p1) / 2
        out.append(dict(name=name, V=V, F=F, T=T, L=L, r=r))
    return out


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def placements(n, seed=0, ncol=2, spacing=2.4, layer=0.35, y0=0.30):
    """Ragdoll world transforms: lying face down, random yaw, stacked in ncol x ncol columns."""
    rng = np.random.default_rng(seed)
    out = []
    cols = [(i, j) for i in range(ncol) for j in range(ncol)]
    for k in range(n):
        ci, cj = cols[k % len(cols)]
        lvl = k // len(cols)
        G = np.eye(4)
        # rest pose centre ~ (0, 0.9, 0): move it to the origin, lie down, yaw, place
        C = np.eye(4); C[:3, 3] = (0.0, -0.9, 0.0)
        Rlie = np.eye(4); Rlie[:3, :3] = _rot_y(rng.uniform(0, 2 * np.pi)) @ _rot_x(np.pi / 2)
        P = np.eye(4)
        P[:3, 3] = ((ci - (ncol - 1) / 2) * spacing, y0 + lvl * layer, (cj - (ncol - 1) / 2) * spacing)
        G = P @ Rlie @ C
        out.append(G)
    return out


def joint_local(bodies):
    """Joint anchors/axes in each body's local frame (from the rest pose)."""
    out = []
    for parent, child, jt, anchor, axis in JOINTS:
        ip, ic = BODY_INDEX[parent], BODY_INDEX[child]
        a = np.array(anchor)
        rec = dict(parent=ip, child=ic, type=jt, anchor=a)
        for key, ib in (("p", ip), ("c", ic)):
            T = bodies[ib]["T"]
            rec["a_" + key] = T[:3, :3].T @ (a - T[:3, 3])
            if axis is not None:
                rec["x_" + key] = T[:3, :3].T @ np.array(axis, dtype=float)
        if axis is not None:
            rec["axis"] = np.array(axis, dtype=float)
        out.append(rec)
    return out
