"""生成 Allegro 手“抓住再翻过来”的 MJCF（笔记 notes/13_mechanisms.md 第 5 节）。

用法：
    python tools/make_hand_grasp.py <right_hand.xml 的路径> [models/local/allegro_grasp.xml] [--n 3]

n 只右手（MuJoCo Menagerie 的 wonik_allegro/right_hand.xml，BSD 许可）排成一排，手心朝上，手心里各放一个物体
（球、圆柱、方块轮流）。每只手挂在一个 mount 刚体上，mount 绕世界 x 轴（手指的方向）转，转轴穿过物体的中心。
时间表（关键帧里的控制量）：
    0 – 0.2 s    手张开，物体落到手心里
    0.2 – 0.8 s  手指的目标从张开线性变到握拳
    0.8 – 1.8 s  mount 转 180°：手心朝下，物体只靠手指和摩擦挂住
    之后不动
手指的目标超过了碰到物体的位置，握力就是 kp (目标 - q)。Menagerie 的手指驱动 kp = 1，这里加到 3（旧分支也是）；
mount 的驱动 kp = 300、kv = 5。手指的关节阻尼 0.1 照模型里的。用 mu=0.8 跑。

起始姿态用 MuJoCo 自己的检测检查（1 mm 的余量，物体和手心之间留了 2 mm）：不能有接触。
手的模型文件不在这个仓库里，生成的 MJCF 写到 models/local/（被忽略），网格目录写成绝对路径。
需要 mujoco（MjSpec，3.2 以上）和 numpy。z 轴朝上。
"""
import argparse
import math
import os

import mujoco
import numpy as np

DT = 0.01
T_SETTLE, T_CLOSE, T_ROLL_END, T_END = 0.2, 0.8, 1.8, 2.5
K_FINGER, K_ROLL, KV_ROLL = 3.0, 300.0, 5.0
FINGER_OPEN, FINGER_CLOSED = (0.0, 0.0, 0.0, 0.0), (0.0, 1.2, 1.2, 1.2)  # 每个手指 j0..j3（弧度）
THUMB_OPEN, THUMB_CLOSED = (0.35, 0.0, 0.0, 0.0), (1.3, 0.4, 0.8, 0.8)    # 拇指 thj0 的下限是 0.263
HEIGHT, SPACING = 0.30, 0.30
# 物体在手掌坐标系里的位置：手掌的面在 x = 0.0111，手指沿 +z，手掌在 z ∈ [-0.095, 0]。物体离手心 2 mm。
PALM_FACE, PALM_GAP = 0.0111, 0.002
# 物体中心在手掌 (y, z) 上的位置。球放在旧分支的位置；圆柱和方块更宽，放在那里会压到拇指根部（手掌 +y 那边），
# 所以往 -y、往手指的方向挪一点。
OBJ_YZ = {"sphere": np.array([0.005, -0.025]), "cylinder": np.array([-0.008, -0.028]), "box": np.array([-0.008, -0.028])}
OBJECTS = [("sphere", 0.03), ("cylinder", (0.025, 0.03)), ("box", (0.022, 0.022, 0.022))]
OBJ_DENSITY = 400.0


def palm_frame():
    """手掌坐标系在世界里的姿态：手掌的 x（手心的法向）朝上，z（手指）朝 +x，y 朝 -y。返回 (frame 的四元数, R)。"""
    R = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    q_target = np.zeros(4)
    mujoco.mju_mat2Quat(q_target, R.ravel())
    q_body = np.array([0.0, 1.0, 0.0, 1.0]) / math.sqrt(2.0)  # right_hand.xml 里 palm 自己的 quat
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q_body)
    q_frame = np.zeros(4)
    mujoco.mju_mulQuat(q_frame, q_target, q_inv)
    return q_frame, R


def build(path, n):
    hand0 = mujoco.MjSpec.from_file(path)
    spec = mujoco.MjSpec()
    spec.modelname = "allegro_grasp"
    spec.meshdir = os.path.abspath(os.path.join(os.path.dirname(path), hand0.meshdir or "assets"))
    spec.compiler.degree = False
    spec.option.timestep = DT
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1])
    q_frame, R = palm_frame()
    for k in range(n):
        kind, size = OBJECTS[k % len(OBJECTS)]
        h = size if kind == "sphere" else (size[0] if kind == "cylinder" else size[2])  # 物体在竖直方向的半高
        centre = np.array([0.0, SPACING * k, HEIGHT])
        c_palm = np.array([PALM_FACE + PALM_GAP + h, *OBJ_YZ[kind]])  # 物体中心在手掌坐标系里
        mount = spec.worldbody.add_body(name=f"mount{k}", pos=centre)
        mount.add_joint(name=f"roll{k}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=1.0)
        mount.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0, mass=0.2)
        hand = mujoco.MjSpec.from_file(path)  # attach 会把刚体搬走，每只手重新读一份
        for a in hand.actuators:
            a.gainprm[0] = K_FINGER
            a.biasprm[1] = -K_FINGER
        frame = mount.add_frame(pos=-(R @ c_palm), quat=q_frame)
        frame.attach_body(hand.body("palm"), f"h{k}_", "")
        obj = spec.worldbody.add_body(name=f"obj{k}", pos=centre)
        obj.add_freejoint()
        g = obj.add_geom(density=OBJ_DENSITY, rgba=[0.85, 0.35, 0.3, 1])
        if kind == "sphere":
            g.type, g.size = mujoco.mjtGeom.mjGEOM_SPHERE, [size, 0, 0]
        elif kind == "cylinder":
            g.type, g.size = mujoco.mjtGeom.mjGEOM_CYLINDER, [size[0], size[1], 0]
            g.quat = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]  # 轴沿世界 y：横在手心上
        else:
            g.type, g.size = mujoco.mjtGeom.mjGEOM_BOX, list(size)
    for k in range(n):
        a = spec.add_actuator(name=f"roll{k}", target=f"roll{k}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=K_ROLL, kv=KV_ROLL)
    return spec


def finger_targets(m, a, roll):
    """张开（a = 0）到握拳（a = 1）之间的目标，和 mount 的转角，按驱动的顺序排好。"""
    u = np.zeros(m.nu)
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name.startswith("roll"):
            u[i] = roll
            continue
        jn = name.split("_", 1)[1]  # ffa0、tha3 ……
        idx = int(jn[3])
        if jn.startswith("th"):
            u[i] = (1 - a) * THUMB_OPEN[idx] + a * THUMB_CLOSED[idx]
        else:
            u[i] = (1 - a) * FINGER_OPEN[idx] + a * FINGER_CLOSED[idx]
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hand")
    ap.add_argument("out", nargs="?", default="models/local/allegro_grasp.xml")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    spec = build(args.hand, args.n)
    m = spec.compile()
    # 起始姿态：手张开（拇指 thj0 = 0.35），物体在手心上方 2 mm。
    q0 = np.array(m.qpos0)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.endswith("thj0"):
            q0[m.jnt_qposadr[j]] = THUMB_OPEN[0]
    for t, a, roll in [(0.0, 0, 0), (T_SETTLE, 0, 0), (T_CLOSE, 1, 0), (T_ROLL_END, 1, math.pi), (T_END, 1, math.pi)]:
        spec.add_key(time=t, qpos=q0.tolist(), ctrl=finger_targets(m, a, roll).tolist())
    m = spec.compile()
    m.geom_margin[:] = 0.5e-3  # MuJoCo 把两个 geom 的 margin 相加：一对是 1 mm
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    if d.ncon:
        pairs = {(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1]),
                  mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2])) for i in range(d.ncon)}
        dists = sorted(d.contact[i].dist for i in range(d.ncon))
        raise SystemExit(f"start pose has {d.ncon} contacts within 1 mm (closest {dists[0]:.4f} m): {sorted(pairs)[:6]}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(spec.to_xml())
    print(f"wrote {args.out}: {args.n} hands, {m.nbody - 1} bodies, nv {m.nv}, nu {m.nu}, {m.nexclude} excluded pairs")


if __name__ == "__main__":
    main()
