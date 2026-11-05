"""生成“摆锤越过矮墙”的 MJCF（笔记 notes/15_drift.md）。

用法：
    python tools/make_wrecking_ball.py [models/wrecking_ball.xml] [--n 10] [--pitch 0.1] [--bar 0.005] [--mass 30]
                                       [--clear 0.02] [--wall 0.3]

为什么要这个场景：libuipc 一类的方法（ABD + 罚函数关节）有关节漂移和刚体变形，但在布娃娃堆里只有
零点几毫米、百分之零点几，画面上看不出来。这两项误差都随“载荷 / 质量”变大：
- 罚函数关节的刚度是 strength_ratio x (两个刚体的质量之和)，和它要传的力无关。轻的连杆吊着重物，
  关节就被拉开：漂移 = F h^2 / (2 gamma (m_A + m_B))（两个轴点分担拉力）。
- ABD 的刚体是仿射体，正交性能量的刚度 kappa（libuipc 的例子用 100 MPa）比钢软三个数量级：
  细杆里的应力 sigma 让它伸长约 sigma / (7 kappa)。
所以这里是一个摆锤：30 kg 的钢块挂在 10 节 5 mm 粗的钢条连成的链上（每节 23 g），从水平位置释放。
摆到最低点时链上的拉力约 3 M g = 880 N。链的名义长度是 1 m，摆锤最低点比一堵矮墙的墙顶高 clear（2 cm）。
- 关节精确、刚体严格刚性时，摆锤从墙顶上 2 cm 处越过去，来回摆。
- 链只要被拉长 2 cm 以上，摆锤就会撞上墙。

链的样子像自行车链：相邻两节在 y 方向错开（一节在 +s，一节在 -s，中间隔 5 mm），两端各伸出铰链 1 cm，
所以看起来是连着的，但互相碰不到。每个铰链画一个孔（父刚体上，深色）和一根销子（子刚体上，橙色），
都只用来显示（contype = 0）：关节被拉开时，销子离开孔的中心，画面上能看到。

坐标：z 朝上，铰链轴沿 y，链在 qpos0 时沿 +x 水平伸出；转轴在 (0, 0, Z)，墙在 x = 0 的正下方。
另外关掉 MuJoCo 的父子碰撞过滤（filterparent），和 libuipc 一样检测所有刚体对。
生成以后用 MuJoCo 检查：起始姿态不同刚体之间的间隙，以及链竖直向下时摆锤底面和墙顶的距离。
"""
import argparse
import math
import sys

import mujoco
import numpy as np

HEADER = """<!--
  {comment}
  由 tools/make_wrecking_ball.py 生成（命令：{cmd}），笔记 notes/15_drift.md。
-->
"""


def f(x):
    return f"{x:.6g}"


def vec(v):
    return " ".join(f(x) for x in v)


def make(a):
    n, p, t, s, e = a.n, a.pitch, a.bar, a.layer, a.overlap
    w = a.size
    aw = 0.015                       # 最后一个铰链到摆锤近端面的距离
    L = n * p + aw + w               # 转轴到摆锤远端面
    Z = a.wall + a.clear + L         # 转轴高度：链竖直时摆锤底面比墙顶高 clear
    rho_w = a.mass / w ** 3
    pin_half = s + t / 2 + 0.002
    lines = [HEADER.format(comment=f"摆锤：{a.mass:g} kg 的钢块挂在 {n} 节 {t*1e3:g} mm 粗的钢条连成的链上，"
                                   f"从水平位置释放；链竖直时摆锤底面比矮墙的墙顶高 {a.clear*1e3:g} mm。",
                           cmd=" ".join(sys.argv[1:]) or "默认参数")]
    lines.append('<mujoco model="wrecking_ball">')
    lines.append('  <compiler angle="radian"/>')
    lines.append('  <option gravity="0 0 -9.81" timestep="0.01">')
    lines.append('    <flag filterparent="disable"/>')
    lines.append('  </option>')
    lines.append('  <default>')
    lines.append('    <geom friction="0.2 0.005 0.0001" density="7800"/>')
    lines.append('    <default class="visual"><geom contype="0" conaffinity="0" density="0" group="1"/></default>')
    # 每个铰链画两样东西（只用来显示）：父刚体上一个深色的“孔”（eye，半径 6.5 mm），子刚体上一根橙色的销子
    # （pin，半径 3 mm，更长）。关节没被拉开时销子在孔的正中；拉开 8 mm 时销子整个跑到孔外面。
    # tools/render.py --keep_rgba pin,eye 保留这两种颜色（别的几何体按树上色）。
    lines.append('    <default class="eye"><geom type="cylinder" size="0.0065" rgba="0.15 0.17 0.2 1" contype="0" '
                 'conaffinity="0" density="0" group="1"/></default>')
    lines.append('    <default class="pin"><geom type="cylinder" size="0.003" rgba="0.95 0.55 0.1 1" contype="0" '
                 'conaffinity="0" density="0" group="1"/></default>')
    lines.append('  </default>')
    lines.append('  <worldbody>')
    lines.append('    <geom name="floor" type="plane" size="0 0 1"/>')
    hw = a.wall / 2
    lines.append(f'    <geom name="wall" type="box" pos="0 0 {f(hw)}" size="0.03 0.25 {f(hw)}"/>')
    # 转轴的支架：在链的 -y 一侧，铰链 0 的轴穿过它（libuipc 里关节的另一方是这个固定刚体）。
    by = -(pin_half + 0.012)
    lines.append(f'    <geom name="bracket" type="box" pos="0 {f(by)} {f(Z)}" size="0.02 0.01 0.02"/>')
    eye_half = s + t / 2 + 0.0005
    lines.append(f'    <geom class="eye" name="eye0" fromto="0 {f(-eye_half)} {f(Z)} 0 {f(eye_half)} {f(Z)}"/>')
    # 只用来显示的架子（不参与碰撞），像秋千架：横梁在转轴上方 15 cm，两根立柱在 x = +-1.35 m，摆锤够不到。
    top = Z + 0.15
    lines.append(f'    <geom class="visual" name="hanger" type="box" pos="0 {f(by)} {f((Z + 0.02 + top) / 2)}" '
                 f'size="0.006 0.006 {f((top - Z - 0.02) / 2)}"/>')
    lines.append(f'    <geom class="visual" name="beam" type="box" pos="0 {f(by)} {f(top)}" size="1.38 0.015 0.015"/>')
    for sx in (-1, 1):
        lines.append(f'    <geom class="visual" name="post{"lr"[sx > 0]}" type="box" pos="{f(1.35 * sx)} {f(by)} '
                     f'{f((top + 0.015) / 2)}" size="0.015 0.015 {f((top + 0.015) / 2)}"/>')
    indent = "    "
    for i in range(n):
        y = s if i % 2 == 0 else -s
        pos = f"0 0 {f(Z)}" if i == 0 else f"{f(p)} 0 0"
        lines.append(f'{indent}<body name="link{i}" pos="{pos}">')
        lines.append(f'{indent}  <joint name="hinge{i}" type="hinge" axis="0 1 0"/>')
        x0 = -e
        x1 = p + e if i < n - 1 else p - 0.003   # 最后一节不伸过末端的铰链（摆锤会转）
        lines.append(f'{indent}  <geom name="bar{i}" type="box" pos="{f((x0 + x1) / 2)} {f(y)} 0" '
                     f'size="{f((x1 - x0) / 2)} {f(t / 2)} {f(t / 2)}"/>')
        lines.append(f'{indent}  <geom class="pin" name="pin{i}" fromto="0 {f(-pin_half)} 0 0 {f(pin_half)} 0"/>')
        lines.append(f'{indent}  <geom class="eye" name="eye{i + 1}" fromto="{f(p)} {f(-eye_half)} 0 {f(p)} {f(eye_half)} 0"/>')
        indent += "  "
    lines.append(f'{indent}<body name="weight" pos="{f(p)} 0 0">')
    lines.append(f'{indent}  <joint name="hinge{n}" type="hinge" axis="0 1 0"/>')
    lines.append(f'{indent}  <geom name="weight" type="box" pos="{f(aw + w / 2)} 0 0" size="{vec([w / 2] * 3)}" '
                 f'density="{f(rho_w)}"/>')
    lines.append(f'{indent}  <geom class="pin" name="pin{n}" fromto="0 {f(-pin_half)} 0 0 {f(pin_half)} 0"/>')
    lines.append(f'{indent}</body>')
    for i in range(n - 1, -1, -1):
        indent = indent[:-2]
        lines.append(f'{indent}</body>')
    lines.append('  </worldbody>')
    lines.append('</mujoco>')
    return "\n".join(lines) + "\n", Z


def box_gap(m, d, g1, g2):
    """两个盒子之间距离的下界：15 个分离轴（两边的面法向、边和边的叉积）上投影间隔的最大值。
    （MuJoCo 3.6 的 mj_geomDistance 对两个平行、在两个方向上重叠的盒子会返回 0，所以盒子对不用它。）"""
    R1, R2 = d.geom_xmat[g1].reshape(3, 3), d.geom_xmat[g2].reshape(3, 3)
    h1, h2 = m.geom_size[g1], m.geom_size[g2]
    c = d.geom_xpos[g2] - d.geom_xpos[g1]
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    axes += [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
    best = -np.inf
    for u in axes:
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u = u / nu
        r1 = np.abs(R1.T @ u) @ h1
        r2 = np.abs(R2.T @ u) @ h2
        best = max(best, abs(c @ u) - r1 - r2)
    return best


def check(path, gap_min, clear):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    worst = (np.inf, None)
    fromto = np.zeros(6)
    box = mujoco.mjtGeom.mjGEOM_BOX
    for i, g1 in enumerate(coll):
        for g2 in coll[i + 1:]:
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b1 == b2 or (b1 == 0 and b2 == 0):
                continue
            if m.geom_type[g1] == box and m.geom_type[g2] == box:
                dist = box_gap(m, d, g1, g2)
            else:
                dist = mujoco.mj_geomDistance(m, d, g1, g2, 1.0, fromto)
            if dist < worst[0]:
                worst = (dist, (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2)))
    links = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"link{i}") for i in range(m.nbody)]
    links = [b for b in links if b >= 0]
    mass_link = m.body_mass[links[0]]
    wb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "weight")
    print(f"{path}: {m.nbody - 1} bodies, nq {m.nq}; link {mass_link * 1e3:.1f} g, weight {m.body_mass[wb]:.2f} kg "
          f"(ratio {m.body_mass[wb] / mass_link:.0f}); closest pair at start {worst[0] * 1e3:.2f} mm {worst[1]}")
    if worst[0] < gap_min:
        sys.exit(f"start gap {worst[0]} < {gap_min}")
    # 链竖直向下：铰链 0 转 +pi/2（绕 y 把 +x 转到 -z），其余为 0。
    d.qpos[:] = 0
    d.qpos[0] = math.pi / 2
    mujoco.mj_kinematics(m, d)
    gw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "weight")
    gwall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    Rw = d.geom_xmat[gw].reshape(3, 3)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[gw]
    low = (d.geom_xpos[gw] + corners @ Rw.T)[:, 2].min()
    top = d.geom_xpos[gwall][2] + m.geom_size[gwall][2]
    print(f"chain hanging straight down: weight bottom {low:.4f} m, wall top {top:.4f} m, clearance {(low - top) * 1e3:.2f} mm")
    if abs(low - top - clear) > 1e-9:
        sys.exit("clearance mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="models/wrecking_ball.xml")
    ap.add_argument("--n", type=int, default=10, help="链的节数")
    ap.add_argument("--pitch", type=float, default=0.1, help="相邻铰链的距离（m）")
    ap.add_argument("--bar", type=float, default=0.005, help="钢条的截面边长（m）")
    ap.add_argument("--layer", type=float, default=0.005, help="钢条在 y 方向错开的距离（中心到 y = 0）")
    ap.add_argument("--overlap", type=float, default=0.01, help="钢条伸过铰链的长度")
    ap.add_argument("--size", type=float, default=0.16, help="摆锤（立方体）的边长")
    ap.add_argument("--mass", type=float, default=30.0, help="摆锤的质量（kg）")
    ap.add_argument("--wall", type=float, default=0.3, help="矮墙的高度")
    ap.add_argument("--clear", type=float, default=0.02, help="链竖直时摆锤底面到墙顶的距离")
    ap.add_argument("--gap_min", type=float, default=0.002)
    a = ap.parse_args()
    xml, Z = make(a)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(xml)
    print(f"wrote {a.out}; pivot height {Z:.4f} m")
    check(a.out, a.gap_min, a.clear)


if __name__ == "__main__":
    main()
