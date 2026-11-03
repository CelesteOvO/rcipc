"""从 models/ragdoll.xml 生成 models/ragdoll_pile.xml：三个布娃娃错开高度和朝向（阶段 10）。
用法：python tools/make_pile.py（在仓库根目录运行）。改了 ragdoll.xml 以后重新跑一次。"""
import io
import re

src = io.open("models/ragdoll.xml", encoding="utf-8").read()
torso = src[src.index('    <body name="torso"'):src.index("  </worldbody>")]


def instance(i, pos, quat):
    t = re.sub(r'name="([a-z_]+)"', lambda m: f'name="{m.group(1)}_{i}"', torso)
    t = re.sub(r'<body name="torso_%d" pos="[^"]*" quat="[^"]*">' % i,
               '<body name="torso_%d" pos="%s" quat="%s">' % (i, pos, quat), t)
    return t


out = '''<!--
  阶段 10 的布娃娃堆：三个 ragdoll.xml 的布娃娃（tools/make_pile.py 从 ragdoll.xml 复制，刚体和关节名加后缀），
  错开高度和朝向落下，叠在一起。三棵树、30 个刚体、75 个自由度。
  检验：每一步结束时所有候选对的距离都 > 0；关节都在限位里。
-->
<mujoco model="ragdoll_pile">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <default>
    <geom type="capsule"/>
    <joint type="hinge"/>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
''' + instance(1, "0 0 1.2", "0.96 0.2 0.15 0") + instance(2, "0.15 0.1 2.1", "0.7 0.1 0.7 0.1") + instance(3, "-0.1 -0.15 3.0", "0.5 0.5 0.5 -0.5") + '''  </worldbody>
</mujoco>
'''
io.open("models/ragdoll_pile.xml", "w", encoding="utf-8", newline="\n").write(out)
print("wrote models/ragdoll_pile.xml:", out.count("<freejoint/>"), "ragdolls")
