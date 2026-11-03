"""把 scene_demo 写出的轨迹渲染成 mp4（阶段 11 之后，给讲解和网页用）。

用法：
    python tools/render.py models/chain.xml chain.csv chain.mp4 [--speed 0.5] [--fps 30] [--w 960 --h 540]
                           [--az 30 --el 20 --dist 1.0] [--target x,y,z] [--title 文字]
    python tools/render.py a.xml a.csv out.mp4 --pair b.xml b.csv --title 左边 --title2 右边    # 两段并排

CSV 的格式见 toys/scenes/main.cc：每行 t, min_distance, min_limit, newton, pcg, 然后是 nq 个 qpos。
只用 MuJoCo 做显示，不做任何物理计算。
- 相机的“上”取重力的反方向：y 轴朝上的场景（libuipc 的布娃娃堆）、转了重力的斜面都能摆正
  （斜面的模型里地面是水平的、重力是斜的，按重力摆相机，看起来就是地面斜了）。
- 每棵运动树一个颜色；地面画成棋盘格；一盏带阴影的主光加一盏补光。
- 左上角写场景名，右下角写时间和这一步结束时的最小间隙（scene_demo 记的；GPU 版只看 d-hat 以内的接触）。
需要 mujoco、numpy、imageio（带 imageio-ffmpeg）、Pillow。
"""
import argparse
import colorsys
import csv
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_csv(path, nq):
    ts, gaps, qs = [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ts.append(float(row[0]))
            gaps.append(float(row[1]))
            q = np.array([float(x) for x in row[5:]])
            if q.size != nq:
                raise ValueError(f"{path}: row has {q.size} qpos values, model has nq = {nq}")
            qs.append(q)
    return np.array(ts), np.array(gaps), np.array(qs)


def font(size):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        p = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Shot:
    """一个场景：模型、轨迹、相机。frame(t) 返回 t 时刻的画面（h x w x 3）。"""

    def __init__(self, xml, csv_path, w, h, az, el, dist, target, title, range_m=None):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.ts, self.gaps, self.qs = load_csv(csv_path, self.m.nq)
        self.q0 = self.m.qpos0.copy()
        self.w, self.h, self.title = w, h, title
        m = self.m
        g = np.array(m.opt.gravity, float)
        self.up = -g / np.linalg.norm(g) if np.linalg.norm(g) > 0 else np.array([0.0, 0.0, 1.0])
        # 颜色：按树（根刚体）分色；世界上的静态几何灰蓝；平面另画。
        root = [m.body_rootid[b] for b in range(m.nbody)]
        roots = sorted({r for r in root if r != 0})
        for gi in range(m.ngeom):
            b = m.geom_bodyid[gi]
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
                m.geom_rgba[gi] = [0, 0, 0, 0]  # 地面由 floor() 画
            elif root[b] == 0:
                m.geom_rgba[gi] = [0.55, 0.60, 0.68, 1]
            else:
                k = roots.index(root[b])
                hue = (0.58 + 0.618034 * k) % 1.0
                m.geom_rgba[gi] = list(colorsys.hsv_to_rgb(hue, 0.50, 0.92)) + [1.0]
        self.planes = [gi for gi in range(m.ngeom) if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE]
        m.vis.headlight.ambient[:] = 0.25
        m.vis.headlight.diffuse[:] = 0.35
        m.vis.headlight.specular[:] = 0.05
        m.vis.quality.shadowsize = 4096
        m.vis.global_.offwidth = max(w, 640)
        m.vis.global_.offheight = max(h, 480)
        # 场景的范围：抽几帧，取所有非平面几何的中心。
        pts = []
        for i in np.linspace(0, len(self.qs) - 1, 8).astype(int):
            self.set_q(i)
            for gi in range(m.ngeom):
                if gi not in self.planes:
                    pts.append(self.d.geom_xpos[gi].copy())
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        self.diag = float(np.linalg.norm(hi - lo)) + 0.3
        if range_m:
            self.diag = max(self.diag, range_m)  # 地面、背景板的大小至少按相机距离算（几乎不动的场景范围太小）
        self.ctr = 0.5 * (lo + hi) if target is None else np.array(target, float)
        # 相机：绕“上”方向的方位角 az、仰角 el，距离 dist x 场景对角线。
        e1 = np.cross(self.up, [1.0, 0.0, 0.0] if abs(self.up[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.up, e1)
        a, e = np.radians(az), np.radians(el)
        self.dirv = np.cos(e) * (np.cos(a) * e1 + np.sin(a) * e2) + np.sin(e) * self.up
        self.cam_pos = self.ctr + self.dirv * (range_m if range_m else self.diag * dist)
        self.fwd = -self.dirv
        cu = self.up - self.fwd * self.fwd.dot(self.up)
        self.cam_up = cu / np.linalg.norm(cu)
        self.right = np.cross(self.fwd, self.cam_up)
        self.R = mujoco.Renderer(m, h, w, max_geom=40000)
        self.f_big, self.f_small = font(int(h * 0.055)), font(int(h * 0.042))

    def set_q(self, i):
        self.d.qpos[:] = self.qs[i]
        mujoco.mj_kinematics(self.m, self.d)

    def floor(self, scn):
        # 地面：平面所在的位置铺棋盘格（法向取平面的 z 轴）。
        for gi in self.planes:
            n = self.d.geom_xmat[gi].reshape(3, 3)[:, 2]
            p0 = self.d.geom_xpos[gi]  # 平面上的一点
            t1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            c = self.ctr - n * np.dot(self.ctr - p0, n)
            S = 2.5 * self.diag
            nt = 16
            t = S / nt
            rot = np.column_stack([t1, t2, n]).ravel()
            for i in range(nt):
                for j in range(nt):
                    if scn.ngeom >= scn.maxgeom:
                        return
                    pos = c + t1 * (-S / 2 + (i + 0.5) * t) + t2 * (-S / 2 + (j + 0.5) * t) - n * 0.002
                    shade = 0.80 if (i + j) % 2 else 0.72
                    gg = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_BOX, np.array([t / 2, t / 2, 0.002]), pos, rot,
                                        np.array([shade, shade, shade * 1.03, 1.0], np.float32))
                    scn.ngeom += 1

    def frame(self, tsim):
        i = int(np.clip(np.searchsorted(self.ts, tsim), 0, len(self.ts) - 1))
        self.set_q(i)
        R = self.R
        R.update_scene(self.d)
        scn = R.scene
        self.floor(scn)
        if scn.ngeom < scn.maxgeom:
            # 背景：视线后方一块自发光的板（MuJoCo 没有天空盒时背景是黑的）。
            rot = np.column_stack([self.right, self.cam_up, self.fwd]).ravel()
            gg = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_BOX, np.array([40 * self.diag, 40 * self.diag, 0.01]),
                                self.ctr + self.fwd * 4 * self.diag, rot, np.array([0.86, 0.89, 0.93, 1.0], np.float32))
            gg.emission = 1.0
            scn.ngeom += 1
        scn.nlight = 2
        key = self.fwd * 0.6 - self.up * 1.0 + self.right * 0.4
        fill = self.fwd * 0.9 - self.up * 0.3 - self.right * 0.6
        for k, (dr, dif, sh) in enumerate([(key, 0.75, 1), (fill, 0.30, 0)]):
            L = scn.lights[k]
            dr = dr / np.linalg.norm(dr)
            L.pos[:] = self.ctr - dr * 3 * self.diag
            L.dir[:] = dr
            L.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            L.diffuse[:] = dif
            L.ambient[:] = 0.10 if k == 0 else 0.0
            L.specular[:] = 0.25 if k == 0 else 0.0
            L.castshadow = sh
            L.headlight = 0
        for c in scn.camera:
            c.pos[:] = self.cam_pos
            c.forward[:] = self.fwd
            c.up[:] = self.cam_up
            c.frustum_near = 0.01 * self.diag
            c.frustum_far = 10 * self.diag
            c.frustum_top = c.frustum_near * np.tan(np.radians(22))
            c.frustum_bottom = -c.frustum_top
            c.frustum_width = c.frustum_top * self.w / self.h
            c.frustum_center = 0.0
        img = Image.fromarray(R.render())
        dr = ImageDraw.Draw(img)
        pad = int(self.h * 0.03)
        if self.title:
            dr.text((pad, pad), self.title, fill=(30, 34, 40), font=self.f_big)
        gap = self.gaps[i]
        gtxt = f"最小间隙 {gap * 1e6:.2f} μm" if np.isfinite(gap) and gap < 1 else "附近没有接触"
        txt = f"t = {self.ts[i]:.2f} s    {gtxt}"
        tw = dr.textlength(txt, font=self.f_small)
        dr.text((self.w - tw - pad, self.h - pad - self.f_small.size * 1.3), txt, fill=(30, 34, 40), font=self.f_small)
        return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--pair", nargs=2, metavar=("MODEL", "CSV"), help="右边再放一段")
    ap.add_argument("--speed", type=float, default=0.5, help="回放倍速（0.5 = 慢放一倍）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=544)
    ap.add_argument("--az", type=float, default=30)
    ap.add_argument("--el", type=float, default=22)
    ap.add_argument("--dist", type=float, default=1.0, help="相机距离，场景对角线的倍数")
    ap.add_argument("--range", type=float, default=None, help="相机距离（米），给了就不用 --dist")
    ap.add_argument("--target", default=None)
    ap.add_argument("--az2", type=float, default=None)
    ap.add_argument("--el2", type=float, default=None)
    ap.add_argument("--dist2", type=float, default=None)
    ap.add_argument("--range2", type=float, default=None)
    ap.add_argument("--target2", default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--title2", default="")
    a = ap.parse_args()
    target = [float(x) for x in a.target.split(",")] if a.target else None
    w = a.w // 2 if a.pair else a.w
    target2 = [float(x) for x in a.target2.split(",")] if a.target2 else target
    specs = [(a.model, a.csv, a.az, a.el, a.dist, a.title, target, a.range)]
    if a.pair:
        specs.append((a.pair[0], a.pair[1], a.az if a.az2 is None else a.az2, a.el if a.el2 is None else a.el2,
                      a.dist if a.dist2 is None else a.dist2, a.title2, target2, a.range2 or a.range))
    # 一次只开一个渲染器（同一个进程里同时开两个，后一个的 OpenGL 上下文会出问题，画面全黑）：
    # 并排时先把左边的帧全部渲染出来，关掉，再渲染右边的。
    frames, T = [], None
    for xml, csv_path, az, el, dist, title, tg, rng in specs:
        s = Shot(xml, csv_path, w, a.h, az, el, dist, tg, title, rng)
        if T is None:
            T = s.ts[-1]
        nframes = int(T / a.speed * a.fps) + 1
        frames.append([s.frame(k * a.speed / a.fps) for k in range(nframes)])
        s.R.close()
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=8,
                                ffmpeg_params=["-movflags", "+faststart"])
    for k in range(nframes):
        imgs = [f[k] for f in frames]
        img = np.concatenate(imgs, axis=1) if len(imgs) > 1 else imgs[0]
        if len(imgs) > 1:
            img = img.copy()
            img[:, w - 1:w + 1] = 255  # 中间一条分隔线
        writer.append_data(img)
    writer.close()
    print(f"wrote {a.out}: {nframes} frames, {T:.2f} s of simulation at {a.speed}x")


if __name__ == "__main__":
    main()
