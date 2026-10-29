"""Plot a stage 1 drop written by ipc1d_demo (columns t, x, v, newton).

    python tools/plot_ipc1d.py ipc1d.csv [--dhat 1e-3] [--show]

Three panels, as in the book's figure 15: the whole trajectory, the gap around the impact in mm with
dhat dashed, and the Newton iterations per step. Saves the figure next to the CSV as .png.
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
