import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
R = json.load(open("l2_cmp.json")); SEL = ("bbbc050", "blastospim", "nucverse3d"); KS = (1, 2, 5, 10); SS = (0, 1000, 2000, 3000, 4000)
f = lambda n, d, i, rule="r2": R[n][rule][0][d][i]
fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
C = {"3D": "#1f6fb4", "2D": "#d9822b"}
lines = []
for a, d in zip(ax, SEL):
    for m, l1, ref in (("3D", "L1_S1", "S1"), ("2D", "L1_2D", "2D")):
        for i, ls in ((0, "-"), (1, "--")):
            y = [f(f"{m}_k0", d, i)] + [f(f"{m}_{d}_k{k}", d, i) for k in KS]
            a.plot([0] + list(KS), y, ls, marker="o", color=C[m], label=f"{m} F1@{0.5 if i == 0 else 0.7}")
            a.plot([0], [f(l1, d, i)], marker="x", ms=9, color=C[m], ls="none")
            r = np.mean([f(f"{ref}_s{s}", d, i) for s in SS]); a.axhline(r, color=C[m], lw=0.8, ls=ls, alpha=0.35)
        lines.append(d)
    for i in (0, 1):
        a.axhline(f("base", d, i, "fix"), color="grey", lw=0.8, ls="-" if i == 0 else "--", alpha=0.6)
    a.set_title(d); a.set_xscale("symlog", linthresh=1); a.set_xticks([0] + list(KS)); a.set_xticklabels(["0"] + [str(k) for k in KS])
    a.set_xlabel("放回的训练体积数 k"); a.grid(alpha=0.3)
ax[0].set_ylabel("pooled F1（规则 T，gt 协议）")
ax[0].legend(fontsize=8, loc="lower right")
fig.suptitle("L2 数据量曲线（种子 0，只作描述）：圆点 = 微调 1000 步（k=0 为不加数据的对照）；× = L1 原模型；淡横线 = 正式版五种子均值，灰线 = 原版", fontsize=9)
fig.tight_layout(); fig.savefig(f"{_NSP_SCRATCH}/nc/heldout/L2_curves.png", dpi=150)
print("PLOT_OK")
