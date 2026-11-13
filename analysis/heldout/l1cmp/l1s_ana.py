import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
V = json.load(open("l1s_vol.json")); SS = (0, 1000, 2000)
GROUPS = {"bbbc050": lambda k: k.startswith("bbbc050__"), "blastospim": lambda k: k.startswith("blastospim__"),
          "nucverse3d": lambda k: k.startswith("nucverse3d__"), "nv_肝脏": lambda k: k.startswith("nucverse3d__") and "Drosophila" not in k,
          "nv_果蝇": lambda k: k.startswith("nucverse3d__") and "Drosophila" in k}
def F(proto, n, g, i):
    rule = "fix" if n == "base" else "r2"; vs = [v[rule] for k, v in V[f"{proto}|{n}"].items() if GROUPS[g](k)]
    return 2 * sum(x["tp7" if i else "tp5"] for x in vs) / max(sum(x["n_gt"] + x["n_pred"] for x in vs), 1)
def G(proto, n, g, i):
    return np.mean([F(proto, n, s, i) for s in ("bbbc050", "blastospim", "nucverse3d")]) if g == "宏" else F(proto, n, g, i)
def verdict(d):
    m, sd = np.mean(d), np.std(d, ddof=1)
    return "确认落后" if m < 0 and -m > 2 * sd and -m >= 0.01 else ("确认领先" if m > 2 * sd and m >= 0.01 else "未确认")
fmt = lambda d: f"{np.mean(d):+.3f}±{np.std(d, ddof=1):.3f}（" + "、".join(f"{x:+.3f}" for x in d) + f"）{verdict(d)}"
res = {}
for proto, gs in (("tGT", ("宏", "bbbc050", "blastospim", "nucverse3d", "nv_肝脏", "nv_果蝇")), ("tFP", ("宏", "bbbc050", "blastospim", "nucverse3d"))):
    print(f"== {proto}")
    for g in gs:
        for i in (0, 1):
            a = {n: [G(proto, f"{n}_s{s}", g, i) for s in SS] for n in ("L1_S1", "L1_2D", "F_S1", "F_2D")}
            dL = [a["L1_S1"][j] - a["L1_2D"][j] for j in range(3)]; dF = [a["F_S1"][j] - a["F_2D"][j] for j in range(3)]
            lS = [a["L1_S1"][j] - a["F_S1"][j] for j in range(3)]; l2 = [a["L1_2D"][j] - a["F_2D"][j] for j in range(3)]
            res[(proto, g, i)] = dict(a=a, dL=dL, dF=dF)
            print(f" {g:9s} F1@{0.5 if i == 0 else 0.7} 绝对 L1S1 {np.mean(a['L1_S1']):.3f} L1二维 {np.mean(a['L1_2D']):.3f} 正式S1 {np.mean(a['F_S1']):.3f} 正式二维 {np.mean(a['F_2D']):.3f} 原版 {G(proto, 'base', g, i):.3f}")
            print(f"   L1(S1−二维) {fmt(dL)} | 正式(S1−二维) {fmt(dF)} | 优势变化 {fmt([x - y for x, y in zip(dL, dF)])}")
            print(f"   损失 S1 {fmt(lS)} | 损失二维 {fmt(l2)}")
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
gs = ("宏", "bbbc050", "blastospim", "nucverse3d")
for r, proto in enumerate(("tGT", "tFP")):
    for c in (0, 1):
        A = ax[r, c]
        for j, g in enumerate(gs):
            x = res[(proto, g, c)]
            for off, key, col, lab in ((-0.15, "dL", "#c0392b", "L1 留出版"), (0.15, "dF", "#2c7fb8", "正式版")):
                A.scatter([j + off] * 3, x[key], color=col, s=22, label=lab if j == 0 else None)
                A.hlines(np.mean(x[key]), j + off - 0.1, j + off + 0.1, color=col, lw=2)
        A.axhline(0, color="grey", lw=0.8); A.set_xticks(range(4)); A.set_xticklabels(["3 套宏平均"] + list(gs[1:]))
        A.set_title(f"{'gt 协议（真值直径）' if proto == 'tGT' else '完整流程（二维 r2a2 直径）'}  F1@{0.5 if c == 0 else 0.7}", fontsize=10)
        A.set_ylabel("S1 − 二维（同种子配对）"); A.grid(alpha=0.3, axis="y")
ax[0, 0].legend(fontsize=8)
fig.suptitle("L1 留出三套数据：S1 相对二维的优势（点 = 种子 0/1000/2000，横线 = 均值；规则 T，pooled F1）", fontsize=11)
fig.tight_layout(); fig.savefig(f"{_NSP_SCRATCH}/nc/heldout/L1_seeds.png", dpi=150); print("PLOT_OK")
