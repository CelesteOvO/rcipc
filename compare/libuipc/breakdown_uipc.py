"""把 run_uipc.py --timer-tree --info-log 的输出整理成“每步的时间花在哪里”。

    python compare/libuipc/breakdown_uipc.py timers.json info.log

timers.json 是逐步累加的计时器（节点路径 → [总时间, 总次数]），info.log 是 Info 级别的日志，
每次 Newton 迭代有一行 "Iterative linear solver iteration count: N"（check_interval = 5，所以 N 是 5 的倍数）。
计时器打开时每段都要同步，总时间比不开时多 3–7%；这里只看比例，墙钟用 --no-timers 的数字。
"""
import json
import re
import sys


flat = json.load(open(sys.argv[1], encoding="utf-8"))
steps = flat["GlobalTimer/Pipeline"][1]
newton = flat["GlobalTimer/Pipeline/Simulation/Newton Iteration"]


def ms(path):
    return 1e3 * flat[path][0] / steps


NI = "GlobalTimer/Pipeline/Simulation/Newton Iteration"
SG = NI + "/Solve Global Linear System"
LS = NI + "/Line Search"
rows = [
    ("整步（计时器）", "GlobalTimer/Pipeline"),
    ("Newton 迭代合计", NI),
    ("  组装线性系统", SG + "/Build Linear System"),
    ("    各子系统的 Hessian", SG + "/Build Linear System/Assemble Subsystems"),
    ("    预条件器", SG + "/Build Linear System/Assemble Preconditioner"),
    ("    转成 BCOO", SG + "/Build Linear System/Convert To BCOO"),
    ("  PCG", SG + "/Solve Linear System"),
    ("  线搜索合计", LS),
    ("    轨迹候选（CCD 的宽相）", LS + "/Detect Trajectory Candidates"),
    ("    CCD 的 TOI", LS + "/Filter CCD TOI"),
    ("    线搜索的试探", LS + "/Line Search Iteration"),
    ("    其中：重新筛接触候选", LS + "/Line Search Iteration/Filter Contact Candidates"),
    ("    其中：能量", LS + "/Line Search Iteration/Compute Energy"),
    ("    起点能量", LS + "/Compute Energy"),
    ("  接触候选（DCD）", NI + "/Detect DCD Candidates"),
    ("  接触的梯度和 Hessian", NI + "/Compute DyTopo Effect"),
]
print(f"{steps} 步，Newton {newton[1]} 次（{newton[1] / steps:.2f} 次/步）")
print(f"{'阶段':<28}{'ms/步':>10}{'ms/Newton':>12}")
for label, path in rows:
    if path in flat:
        print(f"{label:<28}{ms(path):>10.2f}{1e3 * flat[path][0] / newton[1]:>12.2f}")

pcg = [int(m.group(1)) for m in re.finditer(r"Iterative linear solver iteration count: (\d+)",
                                            open(sys.argv[2], encoding="utf-8", errors="replace").read())]
if pcg:
    print(f"PCG：{len(pcg)} 次求解，平均每次 {sum(pcg) / len(pcg):.1f} 次迭代，每步 {sum(pcg) / steps:.1f} 次")
