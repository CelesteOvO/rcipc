"""生成齿轮组的 MJCF（阶段 12 之后的机构场景，笔记 notes/13_mechanisms.md 第 1 节）。

用法：
    python tools/make_gears.py [models/gears.xml] [--n 4] [--teeth 16] [--module 0.01]

一排 n 个一样的直齿轮（渐开线齿形，压力角 20°），每个都装在世界上的铰链（绕 z 轴）上，中心距等于两倍节圆半径。
第 0 个是主动轮：velocity 驱动，0.2 s 里从 0 加到 -100°/s；最后一个的铰链上有阻尼，当作负载，
所以转矩要经过每一对啮合的齿才能传到最后。相邻的齿轮错开半个齿，齿之间留有侧隙（节圆上约 1.5 mm），
开始时互不接触。重力沿转轴，不产生力矩。

齿轮的网格：齿廓一圈（逆时针）拉伸成厚度 t 的柱体，两个端面用中心点扇形三角化（齿廓对中心是星形的）。
三角形都朝外，MuJoCo 才能算出正的体积和惯量。只用 numpy 生成文本，不做物理计算。
"""
import argparse
import math

import numpy as np


def involute(alpha):
    return math.tan(alpha) - alpha


def gear_outline(teeth, module, backlash, flank_pts=6, pressure=math.radians(20)):
    """一个齿轮的齿廓（逆时针的二维点），节圆半径 r = module * teeth / 2。"""
    r = module * teeth / 2
    rb = r * math.cos(pressure)           # 基圆
    ra = r + module                       # 齿顶圆
    rf = r - 1.25 * module                # 齿根圆
    # 节圆上一个齿的半角（齿厚减掉侧隙），再换成基圆上的“起点角”：渐开线在半径 rho 处比基圆转过 inv(alpha)。
    half_pitch = math.pi / (2 * teeth) - backlash / (2 * r)
    psi0 = half_pitch + involute(pressure)
    def half_angle(rho):                  # 半径 rho 处齿的半角
        if rho <= rb:
            return psi0                   # 基圆以内用径向直线
        return psi0 - involute(math.acos(rb / rho))
    radii = np.linspace(max(rf, 0.0), ra, flank_pts)
    pts = []
    for k in range(teeth):
        c = 2 * math.pi * k / teeth
        # 右侧齿面（从齿根到齿顶），齿顶两点，左侧齿面（从齿顶到齿根）：绕齿的中心线逆时针走。
        right = [(rho, c - half_angle(rho)) for rho in radii]
        left = [(rho, c + half_angle(rho)) for rho in radii[::-1]]
        for rho, a in right + left:
            pts.append((rho * math.cos(a), rho * math.sin(a)))
        # 到下一个齿之间沿齿根圆补一个点
        mid = c + math.pi / teeth
        pts.append((rf * math.cos(mid), rf * math.sin(mid)))
    return np.array(pts)


def extrude(outline, thickness):
    """齿廓拉伸成柱体：返回顶点 (n, 3) 和三角形 (m, 3)，都朝外。"""
    n = len(outline)
    zt, zb = thickness / 2, -thickness / 2
    V = [(x, y, zb) for x, y in outline] + [(x, y, zt) for x, y in outline] + [(0, 0, zb), (0, 0, zt)]
    cb, ct = 2 * n, 2 * n + 1
    F = []
    for i in range(n):
        j = (i + 1) % n
        F.append((i, j, n + j))           # 侧面：外法向在前进方向的右边
        F.append((i, n + j, n + i))
        F.append((ct, n + i, n + j))      # 顶面朝 +z
        F.append((cb, j, i))              # 底面朝 -z
    return np.array(V), np.array(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="models/gears.xml")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--teeth", type=int, default=16)
    ap.add_argument("--module", type=float, default=0.01)
    ap.add_argument("--thickness", type=float, default=0.02)
    ap.add_argument("--backlash", type=float, default=0.003)
    ap.add_argument("--omega", type=float, default=-100.0, help="主动轮的目标角速度（度/秒）")
    ap.add_argument("--load", type=float, default=0.02, help="最后一个齿轮的阻尼（N m s/rad）")
    args = ap.parse_args()

    r = args.module * args.teeth / 2
    V, F = extrude(gear_outline(args.teeth, args.module, args.backlash), args.thickness)
    vert = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in V)
    face = " ".join(f"{a} {b} {c}" for a, b, c in F)
    omega = math.radians(args.omega)

    bodies = []
    for k in range(args.n):
        phase = math.pi / args.teeth if k % 2 else 0.0   # 相邻的齿轮错开半个齿
        damping = f' damping="{args.load}"' if k == args.n - 1 else ""
        bodies.append(f"""    <body name="gear{k}" pos="{2 * r * k:.6f} 0 0" euler="0 0 {phase:.9f}">
      <joint name="axle{k}" type="hinge" axis="0 0 1"{damping}/>
      <geom type="mesh" mesh="gear" density="7800" rgba="0.75 0.7 0.55 1"/>
    </body>""")

    xml = f"""<!--
  齿轮组（tools/make_gears.py 生成，不要手改）：{args.n} 个 {args.teeth} 齿的渐开线直齿轮，模数 {args.module} m，
  节圆半径 {r:.3f} m，厚 {args.thickness} m，节圆上的侧隙 {args.backlash} m。
  gear0 是主动轮：velocity 驱动（kv 5），0.2 s 里从 0 加到 {args.omega:g}°/s。最后一个齿轮的铰链阻尼 {args.load} N m s/rad 当负载。
  一样的齿轮相邻啮合，第 k 个的转角应该是主动轮的 (-1)^k 倍（差一个侧隙）。笔记 notes/13_mechanisms.md 第 1 节。
-->
<mujoco model="gears">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <mesh name="gear" vertex="{vert}" face="{face}"/>
  </asset>
  <worldbody>
{chr(10).join(bodies)}
  </worldbody>
  <actuator>
    <velocity name="drive" joint="axle0" kv="5"/>
  </actuator>
  <keyframe>
    <key time="0" ctrl="0"/>
    <key time="0.2" ctrl="{omega:.9f}"/>
  </keyframe>
</mujoco>
"""
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(xml)
    print(f"wrote {args.out}: {args.n} gears, {len(V)} vertices and {len(F)} triangles each")


if __name__ == "__main__":
    main()
