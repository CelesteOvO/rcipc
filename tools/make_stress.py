"""生成两个“穿透看得见”的场景的 MJCF（笔记 notes/14_visible.md）。

用法：
    python tools/make_stress.py chain   [models/ring_chain.xml] [--n 24] [--R 0.02] [--wire 0.003] [--pitch 0.03] [--H 0.5]
    python tools/make_stress.py blender [models/blender.xml]    [--balls 20] [--omega 100]

前面的场景（布娃娃、齿轮、G1）里，就算穿进去几毫米，画面上也看不出来：物体都是厚的，速度不快，
穿透以后会被推回来，结局和不穿透差不多。这两个场景反过来，任何一处穿透都会改变结局，而且改变的是拓扑，
回放时一眼能看出来，也能用程序精确地查（tools/topology.py）：

chain：一串互相套着的环（每个环是 12 段胶囊围成的正 12 边形，自由刚体），第 0 个环套在一根横杆上，
    横杆两端有挡球，环滑不下来。整串从水平伸直的姿态释放：甩下去，砸到地面上堆起来；1.6 s 起横杆在
    1 s 里升高 0.8 m，把整串从地面上拎起来。环的“线”只有 2 r = 6 mm 粗。只要有一处两根线互相穿过，
    两个环就脱开了（或者套上了本来没套的环），拎起来时整串断成两截。
    检查：每一对环的环绕数（linking number）。相邻的两个是 ±1，其余是 0，整段都不能变。

blender：一个封闭的罐子（12 块 4 mm 厚的墙围成正 12 边形，顶上一块 4 mm 的盖子，底是地面），
    底部一个 4 叶的转子（叶片 3 mm 厚、6 cm 高，绕竖直的铰链转），velocity 驱动在 0.2 s 里从 0
    加到 omega rad/s。罐子里从上面掉下 balls 个半径 12 mm 的球，被叶片打出去，撞墙、撞盖子，再被打。
    叶尖速度 omega x 0.26 m（100 rad/s 时 26 m/s），球被打出去的速度也是这个量级。
    只要有一次穿透（叶片穿过球，或者球穿过墙、盖子），球就会到罐子外面去。
    检查：每个球的球心是不是还在罐子里。

两个场景都只用 MuJoCo 原生的几何（胶囊、球、盒子、圆柱），所以 MuJoCo 和我们用的是完全相同的几何体，
tools/mujoco_traj.py 可以直接拿同一个文件来对比。

生成以后用 MuJoCo 的 mj_geomDistance 检查起始姿态：不同刚体的几何体之间至少隔开 gap_min（默认 2 mm）。
只用 numpy 和 MuJoCo 生成、检查文本，不做物理计算。
"""
import argparse
import math
import sys

import mujoco
import numpy as np

HEADER = """<!--
  {comment}
  由 tools/make_stress.py 生成（命令：{cmd}），笔记 notes/14_visible.md。
-->
"""


def f(x):
    return f"{x:.6g}"


def vec(v):
    return " ".join(f(x) for x in v)


# ---------------------------------------------------------------------------------------------------------------------
# 锁链

def ring_geoms(name, R, wire, K, plane):
    """一个环：正 K 边形（顶点在半径 R 上，第 0 个顶点在 +x），每条边一段胶囊。plane 是 'xz' 或 'xy'。"""
    out = []
    for k in range(K):
        a0, a1 = 2 * math.pi * k / K, 2 * math.pi * (k + 1) / K
        p0 = (R * math.cos(a0), R * math.sin(a0))
        p1 = (R * math.cos(a1), R * math.sin(a1))
        if plane == "xz":
            a, b = (p0[0], 0, p0[1]), (p1[0], 0, p1[1])
        else:
            a, b = (p0[0], p0[1], 0), (p1[0], p1[1], 0)
        out.append(f'      <geom name="{name}_{k}" type="capsule" fromto="{vec(a)} {vec(b)}" size="{f(wire)}"/>')
    return out


def make_chain(a):
    R, wire, p, n, H, K = a.R, a.wire, a.pitch, a.n, a.H, 12
    # 拉直时相邻两环的线心距是 2R - p，至少要 2 wire（否则一开始就穿透）。留出的间隙：
    gap = 2 * R - p - 2 * wire
    if gap <= 0:
        sys.exit(f"pitch {p} too large: adjacent rings would intersect when straight (gap {gap})")
    x0 = 0.010  # 第 0 个环的中心在横杆右边 1 cm（横杆从环的左半边穿过）
    bar_r, stop_r, half = 0.004, 0.02, 0.08
    lines = [HEADER.format(
        comment=f"锁链：{n} 个环（中心线半径 {R * 1e3:g} mm、线半径 {wire * 1e3:g} mm、节距 {p * 1e3:g} mm），"
                f"第 0 个套在高 {H} m 的横杆上；从水平伸直的姿态释放，1.6–2.6 s 横杆升高 0.8 m 把整串拎起来",
        cmd=" ".join(sys.argv[1:]))]
    lines.append('<mujoco model="ring_chain">')
    lines.append('  <compiler angle="radian"/>')
    lines.append('  <option gravity="0 0 -9.81" timestep="0.002"/>')
    lines.append('  <default>')
    lines.append('    <geom friction="0.3 0.005 0.0001" density="7800"/>')
    lines.append('  </default>')
    lines.append('  <worldbody>')
    lines.append('    <geom name="floor" type="plane" size="2 2 0.1"/>')
    lines.append(f'    <body name="bar" pos="0 0 {f(H)}">')
    lines.append('      <joint name="lift" type="slide" axis="0 0 1"/>')
    lines.append(f'      <geom name="bar" type="capsule" fromto="0 {f(-half)} 0 0 {f(half)} 0" size="{f(bar_r)}" mass="1"/>')
    lines.append(f'      <geom name="stop0" type="sphere" pos="0 {f(-half)} 0" size="{f(stop_r)}" mass="0.1"/>')
    lines.append(f'      <geom name="stop1" type="sphere" pos="0 {f(half)} 0" size="{f(stop_r)}" mass="0.1"/>')
    lines.append('    </body>')
    for i in range(n):
        plane = "xz" if i % 2 == 0 else "xy"
        lines.append(f'    <body name="ring{i}" pos="{f(x0 + i * p)} 0 {f(H)}">')
        lines.append('      <freejoint/>')
        lines += ring_geoms(f"ring{i}", R, wire, K, plane)
        lines.append('    </body>')
    lines.append('  </worldbody>')
    lines.append('  <actuator>')
    lines.append('    <position name="lift" joint="lift" kp="20000" dampratio="1"/>')
    lines.append('  </actuator>')
    lines.append('  <keyframe>')
    lines.append('    <key name="start" time="0" ctrl="0"/>')
    lines.append('    <key time="1.6" ctrl="0"/>')
    lines.append('    <key time="2.6" ctrl="0.8"/>')
    lines.append('  </keyframe>')
    lines.append('</mujoco>')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# 搅拌机

def make_blender(a):
    Rj, Hj, t_wall = 0.30, 0.35, 0.004       # 罐子：内半径、高、墙厚
    r_ball, nb = 0.012, a.balls
    hub_r, blade_in, blade_out, blade_t, blade_h, zr = 0.03, 0.03, 0.26, 0.003, 0.06, 0.036
    lines = [HEADER.format(
        comment=f"搅拌机：封闭的罐子（内半径 {Rj} m、高 {Hj} m、墙和盖子 {t_wall * 1e3:g} mm 厚），"
                f"4 叶转子（叶片 {blade_t * 1e3:g} mm 厚）0.2 s 里加速到 {a.omega:g} rad/s，"
                f"{nb} 个半径 {r_ball * 1e3:g} mm 的球从上面掉下去",
        cmd=" ".join(sys.argv[1:]))]
    lines.append('<mujoco model="blender">')
    lines.append('  <compiler angle="radian"/>')
    lines.append('  <option gravity="0 0 -9.81" timestep="0.002"/>')
    lines.append('  <default>')
    lines.append('    <geom friction="0.3 0.005 0.0001"/>')
    lines.append('  </default>')
    lines.append('  <worldbody>')
    lines.append('    <geom name="floor" type="plane" size="1 1 0.1"/>')
    # 墙：第 k 块的内表面在半径 Rj 处，法向沿 phi_k；盒子局部的 y 轴是厚度方向（朝外），x 轴沿切向。
    # 半宽取到外表面的多边形顶点再多 2 mm，相邻两块在角上重叠（世界上的几何体之间不检测），不留缝。
    nw = 12
    hw = (Rj + t_wall) * math.tan(math.pi / nw) + 0.002
    for k in range(nw):
        phi = 2 * math.pi * k / nw
        c = (Rj + t_wall / 2) * np.array([math.cos(phi), math.sin(phi)])
        lines.append(f'    <geom name="wall{k}" type="box" pos="{f(c[0])} {f(c[1])} {f(Hj / 2)}" '
                     f'euler="0 0 {f(phi - math.pi / 2)}" size="{f(hw)} {f(t_wall / 2)} {f(Hj / 2)}"/>')
    lines.append(f'    <geom name="lid" type="box" pos="0 0 {f(Hj + t_wall / 2)}" '
                 f'size="{f(Rj + 0.03)} {f(Rj + 0.03)} {f(t_wall / 2)}"/>')
    lines.append(f'    <body name="rotor" pos="0 0 {f(zr)}">')
    lines.append('      <joint name="spin" type="hinge" axis="0 0 1"/>')
    lines.append(f'      <geom name="hub" type="cylinder" size="{f(hub_r)} {f(blade_h / 2)}" density="7800"/>')
    for k in range(4):
        th = k * math.pi / 2
        rc = 0.5 * (blade_in + blade_out)
        lines.append(f'      <geom name="blade{k}" type="box" pos="{f(rc * math.cos(th))} {f(rc * math.sin(th))} 0" '
                     f'euler="0 0 {f(th)}" size="{f((blade_out - blade_in) / 2)} {f(blade_t / 2)} {f(blade_h / 2)}" '
                     f'density="7800"/>')
    lines.append('    </body>')
    # 球：螺旋地摆在转子上方，半径 0.07–0.24 m、高 0.12–0.32 m，彼此至少隔开 1 cm。
    rng = np.random.default_rng(1)
    pts = []
    while len(pts) < nb:
        rr = rng.uniform(0.07, 0.24)
        ang = rng.uniform(0, 2 * math.pi)
        z = rng.uniform(0.12, Hj - 0.03)
        p = np.array([rr * math.cos(ang), rr * math.sin(ang), z])
        if all(np.linalg.norm(p - q) > 2 * r_ball + 0.01 for q in pts):
            pts.append(p)
    for i, p in enumerate(pts):
        lines.append(f'    <body name="ball{i}" pos="{vec(p)}">')
        lines.append('      <freejoint/>')
        lines.append(f'      <geom name="ball{i}" type="sphere" size="{f(r_ball)}" density="1100"/>')
        lines.append('    </body>')
    lines.append('  </worldbody>')
    lines.append('  <actuator>')
    lines.append('    <velocity name="spin" joint="spin" kv="20"/>')
    lines.append('  </actuator>')
    lines.append('  <keyframe>')
    lines.append('    <key name="start" time="0" ctrl="0"/>')
    lines.append(f'    <key time="0.2" ctrl="{f(a.omega)}"/>')
    lines.append('  </keyframe>')
    lines.append('</mujoco>')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------

def check_start(path, gap_min):
    """起始姿态里，不同刚体（世界上的几何体算一个刚体）的几何体之间的最小距离；小于 gap_min 就报错。"""
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    fromto = np.zeros(6)
    worst, where = np.inf, None
    for g1 in range(m.ngeom):
        for g2 in range(g1 + 1, m.ngeom):
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b1 == b2 or (b1 == 0 and b2 == 0):
                continue
            dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.05, fromto)
            if dist < worst:
                worst, where = dist, (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                      mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2))
    total = sum(m.body_mass[1:])
    print(f"{path}: {m.nbody - 1} bodies, nv {m.nv}, {m.ngeom} geoms, total mass {total:.3f} kg; "
          f"closest start pair {where} at {worst * 1e3:.2f} mm")
    if worst < gap_min:
        sys.exit(f"start pose too tight: {worst * 1e3:.2f} mm < {gap_min * 1e3:.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=["chain", "blender"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--R", type=float, default=0.02)
    ap.add_argument("--wire", type=float, default=0.003)
    ap.add_argument("--pitch", type=float, default=0.03)
    ap.add_argument("--H", type=float, default=0.5)
    ap.add_argument("--balls", type=int, default=20)
    ap.add_argument("--omega", type=float, default=100)
    ap.add_argument("--gap_min", type=float, default=0.002)
    a = ap.parse_args()
    out = a.out or ("models/ring_chain.xml" if a.scene == "chain" else "models/blender.xml")
    text = make_chain(a) if a.scene == "chain" else make_blender(a)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    check_start(out, a.gap_min)


if __name__ == "__main__":
    main()
