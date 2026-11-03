"""把 run_uipc.py 和 measure_rcipc.py 写出的 JSON 汇总成一张 Markdown 表（笔记 notes/compare_libuipc.md 里的表就是它打的）。

用法：python compare/libuipc/compare.py results/uipc_default.json results/rcipc_default.json [更多 JSON ...]
"""
import json
import sys

import numpy as np


def summarize(path):
    r = json.load(open(path))
    s = r["samples"]
    newton = np.array(r["newton"])
    row = dict(
        engine=r["engine"],
        steps=len(newton),
        newton_mean=newton.mean(),
        newton_median=float(np.median(newton)),
        newton_p95=float(np.percentile(newton, 95)),
        newton_max=int(newton.max()),
        drift=max(x["drift_mm"] for x in s),
        axis=max(x["axis_deg"] for x in s),
        orth=max(x["orth"] for x in s),
        lowest_min=min(x["lowest_mm"] for x in s[1:]),
        lowest_end=s[-1]["lowest_mm"],
    )
    row["pcg_mean"] = float(np.mean(r["pcg"])) if "pcg" in r else None
    row["min_distance"] = float(np.min(r["min_distance"])) if "min_distance" in r else None
    row["ms_median"] = float(np.median(r["step_s"])) * 1e3 if "step_s" in r else None
    return path, row


def main(paths):
    rows = [summarize(p) for p in paths]
    # 输出用英文（Windows 控制台默认不是 UTF-8）。笔记里的表是把它翻成中文的。
    print("| file | engine | steps | Newton/step mean / median / p95 / max | PCG/step | joint drift max (mm) | "
          "hinge axis max (deg) | non-rigidity max |A^T A - I| | lowest min (mm) | lowest end (mm) | min distance (mm) |")
    print("|" + "---|" * 12)
    for p, r in rows:
        pcg = f"{r['pcg_mean']:.1f}" if r["pcg_mean"] is not None else "-"
        dmin = f"{r['min_distance'] * 1e3:.4f}" if r["min_distance"] is not None else "-"
        print(f"| {p.split('/')[-1]} | {r['engine']} | {r['steps']} | {r['newton_mean']:.2f} / {r['newton_median']:.0f} / "
              f"{r['newton_p95']:.0f} / {r['newton_max']} | {pcg} | {r['drift']:.3g} | {r['axis']:.3g} | {r['orth']:.2e} | "
              f"{r['lowest_min']:.3f} | {r['lowest_end']:.3f} | {dmin} |")
    for p, r in rows:
        if r["ms_median"] is not None:
            print(f"{p}: median wall time {r['ms_median']:.1f} ms/step (GPU, timers on; not comparable with the CPU run)")


if __name__ == "__main__":
    main(sys.argv[1:])
