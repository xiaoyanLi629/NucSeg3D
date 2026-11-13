import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, numpy as np
R = json.load(open(f"{_NSP_SCRATCH}/nc/heldout/l1cmp/l1_cmp.json"))
SEL = ("bbbc050", "blastospim", "nucverse3d"); SS = (0, 1000, 2000, 3000, 4000)
def g(p, n, rule="r2"):
    k = f"{p}|{n}"
    return R.get(k, {}).get(rule)
for p in ("tFP", "tGT"):
    print(f"== {p}（pooled F1@0.5/0.7；括号内为逐体积平均 F1@0.5）")
    rows = [("L1_S1", "r2"), ("L1_2D", "r2"), ("base", "fix")]
    for n, rule in rows:
        x = g(p, n, rule)
        if not x: print(" ", n, "无"); continue
        per, mac = x
        print(f"  {n:6s} " + "  ".join(f"{s} {per[s][0]:.3f}/{per[s][1]:.3f}({per[s][2]:.3f})" for s in SEL) + f" | 宏 {mac[0]:.4f}/{mac[1]:.4f}({mac[2]:.4f})")
    for m in ("S1", "2D"):
        A = [g(p, f"{m}_s{s}") for s in SS]; A = [a for a in A if a]
        for s in SEL:
            v5 = [a[0][s][0] for a in A]; v7 = [a[0][s][1] for a in A]
        mac5 = [a[1][0] for a in A]; mac7 = [a[1][1] for a in A]
        print(f"  正式{m} " + "  ".join(f"{s} {np.mean([a[0][s][0] for a in A]):.3f}[{min(a[0][s][0] for a in A):.3f},{max(a[0][s][0] for a in A):.3f}]/{np.mean([a[0][s][1] for a in A]):.3f}" for s in SEL)
              + f" | 宏 {np.mean(mac5):.4f}[{min(mac5):.4f},{max(mac5):.4f}]/{np.mean(mac7):.4f}[{min(mac7):.4f},{max(mac7):.4f}]")
    d = [(g(p, f"S1_s{s}")[0][t][i] - g(p, f"2D_s{s}")[0][t][i]) for s in SS for t in SEL for i in (0,)]
    for s_ in SEL + ("宏",):
        f = (lambda a, i: a[1][i]) if s_ == "宏" else (lambda a, i, s_=s_: a[0][s_][i])
        dd = [[f(g(p, f"S1_s{s}"), i) - f(g(p, f"2D_s{s}"), i) for s in SS] for i in (0, 1)]
        l1 = [f(g(p, "L1_S1"), i) - f(g(p, "L1_2D"), i) for i in (0, 1)] if g(p, "L1_S1") else [np.nan] * 2
        lossS = [f(g(p, "L1_S1"), i) - np.mean([f(g(p, f"S1_s{s}"), i) for s in SS]) for i in (0, 1)] if g(p, "L1_S1") else [np.nan] * 2
        loss2 = [f(g(p, "L1_2D"), i) - np.mean([f(g(p, f"2D_s{s}"), i) for s in SS]) for i in (0, 1)]
        print(f"  差 {s_:10s} L1(S1−2D) {l1[0]:+.3f}/{l1[1]:+.3f}  正式(S1−2D) 均 {np.mean(dd[0]):+.3f}[{min(dd[0]):+.3f},{max(dd[0]):+.3f}]/{np.mean(dd[1]):+.3f}[{min(dd[1]):+.3f},{max(dd[1]):+.3f}]"
              f"  损失 S1 {lossS[0]:+.3f}/{lossS[1]:+.3f}  2D {loss2[0]:+.3f}/{loss2[1]:+.3f}")
# 完整流程逐体积：L1 S1 与 L1 2D
