"""画阶段 1 的落地仿真（ipc1d_demo drop 写出的 CSV，列为 t, x, v, newton）。

    python tools/plot_ipc1d.py ipc1d.csv [--dhat 1e-3] [--show]

三幅图，和书上的图 15 对应：整段轨迹；撞击附近的间隙（mm，虚线是 dhat）；每步的 Newton 次数。
图保存在 CSV 旁边，扩展名换成 .png；加 --show 会同时弹出窗口。
"""
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dhat", type=float, default=1e-3)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    d = np.genfromtxt(a.csv, delimiter=",", names=True)
    t, x, n = d["t"], d["x"], d["newton"]
    i = int(np.argmin(x))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(t, x)
    ax[0].set(xlabel="t (s)", ylabel="gap (m)", title="trajectory")

    w = (t > t[i] - 0.05) & (t < t[i] + 0.15)
    ax[1].plot(t[w], x[w] * 1e3, ".-", ms=3)
    ax[1].axhline(a.dhat * 1e3, ls="--", c="gray", lw=1, label="dhat")
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set(xlabel="t (s)", ylabel="gap (mm)", ylim=(-0.2 * a.dhat * 1e3, 3 * a.dhat * 1e3),
              title=f"impact: min {x[i] * 1e3:.4f} mm at {t[i]:.3f} s")
    ax[1].legend()

    ax[2].step(t, n, where="post")
    ax[2].set(xlabel="t (s)", ylabel="Newton iterations", title=f"max {int(n.max())}, mean {n.mean():.2f}")

    fig.tight_layout()
    out = pathlib.Path(a.csv).with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
