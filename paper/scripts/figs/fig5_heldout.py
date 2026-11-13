"""Figure 5: datasets withheld from training (L1, L2).

Sources (read-only):
  a  $NUCSEG3D_SCRATCH/nc/heldout/l1cmp/l1s_vol.json  (per-volume counts; pooled F1 recomputed exactly as l1s_ana.py,
     rule T ('r2') for S1/2D, fixed threshold for the original; gt protocol 'tGT')
  b  $NUCSEG3D_SCRATCH/nc/heldout/l1cmp/l2_cmp.json   (rule T 'r2'; original 'fix'), as l2_plot.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json

import numpy as np
from matplotlib.lines import Line2D

import nc_style as S
from nc_style import plt

S.setup()
H = f"{_NSP_SCRATCH}/nc/heldout/l1cmp"
V = json.load(open(f"{H}/l1s_vol.json"))
L2 = json.load(open(f"{H}/l2_cmp.json"))
SS = (0, 1000, 2000)
SETS = ("bbbc050", "blastospim", "nucverse3d")


def F(proto, n, g, i):
    rule = "fix" if n == "base" else "r2"
    vs = [v[rule] for k, v in V[f"{proto}|{n}"].items() if k.startswith(g + "__")]
    return 2 * sum(x["tp7" if i else "tp5"] for x in vs) / max(sum(x["n_gt"] + x["n_pred"] for x in vs), 1)


def G(proto, n, g, i):
    return float(np.mean([F(proto, n, s, i) for s in SETS])) if g == "macro" else F(proto, n, g, i)


fig = plt.figure(figsize=(S.W2, 112 * S.MM))
outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], left=0.075, right=0.985, bottom=0.085, top=0.925, hspace=0.5)
prov = {"figure": "fig5_heldout", "panels": {}}

# ---------------- a: retrained (L1) vs production, gt protocol, per seed
ga = outer[0].subgridspec(1, 2, wspace=0.18)
groups = ("macro", "bbbc050", "blastospim", "nucverse3d")
gl = {"macro": "Macro (3 datasets)", "bbbc050": "BBBC050", "blastospim": "BlastoSPIM", "nucverse3d": "NucVerse3D"}
order = [("L1_S1", "S1", False), ("L1_2D", "2D", False), ("F_S1", "S1", True), ("F_2D", "2D", True)]
pa = {}
for i in (0, 1):
    ax = fig.add_subplot(ga[i])
    for gi, g in enumerate(groups):
        for oi, (n, m, prod) in enumerate(order):
            x0 = gi + (oi - 1.5) * 0.19
            ys = [G("tGT", f"{n}_s{s}", g, i) for s in SS]
            ax.scatter([x0] * 3, ys, s=7, marker=S.MK[m], color=S.C[m] if prod else "white", edgecolor=S.C[m], lw=0.6, zorder=3)
            ax.hlines(np.mean(ys), x0 - 0.075, x0 + 0.075, color=S.C[m], lw=1.0, zorder=4)
            pa[f"F1@{0.5 if i == 0 else 0.7}/{g}/{n}"] = [round(y, 4) for y in ys]
        b = G("tGT", "base", g, i)
        ax.hlines(b, gi - 0.42, gi + 0.42, color=S.C["base"], lw=0.8, ls=(0, (3, 1.5)))
        pa[f"F1@{0.5 if i == 0 else 0.7}/{g}/base"] = round(b, 4)
        if gi:
            ax.axvline(gi - 0.5, color="0.85", lw=0.4)
    NV = {g: len([k for k in V["tGT|L1_S1_s0"] if k.startswith(g + "__")]) for g in SETS}
    ax.set_xticks(range(4), [gl[g] + (f"\n({NV[g]} vol.)" if g in NV else f"\n({sum(NV.values())} vol.)") for g in groups])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel(f"Pooled F1@{'0.5' if i == 0 else '0.7'} (gt protocol)")
    ax.tick_params(axis="x", length=0)
    if i == 0:
        S.letter(fig, 0.005, ax.get_position().y1 + 0.04, "a")
h = [Line2D([], [], marker="o", ls="", mfc="white", mec=S.C["S1"], ms=3, label="3D, retrained without held-out data"),
     Line2D([], [], marker="s", ls="", mfc="white", mec=S.C["2D"], ms=3, label="2D, retrained without held-out data"),
     Line2D([], [], marker="o", ls="", color=S.C["S1"], ms=3, label="S1, production"),
     Line2D([], [], marker="s", ls="", color=S.C["2D"], ms=3, label="2D, production"),
     Line2D([], [], color=S.C["base"], ls=(0, (3, 1.5)), lw=0.8, label="Original Cellpose-SAM")]
fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.0), fontsize=5.3, handletextpad=0.3, columnspacing=1.2)
prov["panels"]["a"] = {"source": f"{H}/l1s_vol.json (tGT; rule T for S1/2D, fixed for original)",
                       "seeds": list(SS), "values": pa, "markers": "dots = seeds 0/1000/2000, bars = mean"}

# ---------------- b: L2 data-efficiency curves (seed 0)
KS = (1, 2, 5, 10)
SS5 = (0, 1000, 2000, 3000, 4000)
f = lambda n, d, i, rule="r2": L2[n][rule][0][d][i]  # noqa: E731
gb = outer[1].subgridspec(1, 3, wspace=0.28)
pb = {}
for j, d in enumerate(SETS):
    ax = fig.add_subplot(gb[j])
    for m, key, l1, ref in (("S1", "3D", "L1_S1", "S1"), ("2D", "2D", "L1_2D", "2D")):
        for i, ls in ((0, "-"), (1, (0, (3, 1.5)))):
            y = [f(f"{key}_k0", d, i)] + [f(f"{key}_{d}_k{k}", d, i) for k in KS]
            ax.plot([0] + list(KS), y, ls=ls, marker=S.MK[m], ms=2.4, color=S.C[m], lw=0.85,
                    mfc=S.C[m] if i == 0 else "white")
            ax.plot([0], [f(l1, d, i)], marker="x", ms=4, color=S.C[m], ls="none", mew=0.8)
            r = float(np.mean([f(f"{ref}_s{s}", d, i) for s in SS5]))
            ax.axhline(r, color=S.C[m], lw=2.2, alpha=0.18)
            pb[f"{d}/{m}/F1@{0.5 if i == 0 else 0.7}"] = {"k0_to_k10": [round(v, 4) for v in y], "L1": round(f(l1, d, i), 4),
                                                         "production_mean5": round(r, 4)}
    for i in (0, 1):
        ax.axhline(f("base", d, i, "fix"), color=S.C["base"], lw=0.6, ls="-" if i == 0 else (0, (3, 1.5)))
        pb[f"{d}/base/F1@{0.5 if i == 0 else 0.7}"] = round(f("base", d, i, "fix"), 4)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0] + list(KS), ["0"] + [str(k) for k in KS])
    ax.minorticks_off()
    ax.set_xlim(-0.25, 12)
    ax.set_title(f"{S.DS_NAME[d]} ({L2['S1_s0']['r2'][0][d][-1]} test vol.)", fontsize=6.5)
    ax.set_xlabel("Training volumes added back ($k$)")
    ax.grid(axis="y", lw=0.3, color="0.88")
    if j == 0:
        ax.set_ylabel("Pooled F1 (gt protocol)")
        S.letter(fig, 0.005, ax.get_position().y1 + 0.045, "b")
h = [Line2D([], [], color="0.3", marker="o", ms=2.4, label="F1@0.5"),
     Line2D([], [], color="0.3", marker="o", mfc="white", ms=2.4, ls=(0, (3, 1.5)), label="F1@0.7"),
     Line2D([], [], color="0.3", marker="x", ls="", ms=4, label="retrained model, no fine-tuning"),
     Line2D([], [], color="0.3", lw=2.2, alpha=0.25, label="production (mean of 5 seeds)"),
     Line2D([], [], color=S.C["base"], lw=0.6, label="original"),
     Line2D([], [], color=S.C["S1"], lw=1.2, label="3D"), Line2D([], [], color=S.C["2D"], lw=1.2, label="2D")]
fig.legend(handles=h, loc="center", ncol=7, bbox_to_anchor=(0.53, outer[1].get_position(fig).y1 + 0.075), fontsize=5.3,
           handletextpad=0.3, columnspacing=1.1)
prov["panels"]["b"] = {"source": f"{H}/l2_cmp.json (rule T 'r2'; original 'fix'); seed 0, 1,000 fine-tuning steps",
                       "values": pb}
S.save(fig, "fig5_heldout", prov)
for k in ("F1@0.5/macro", "F1@0.7/macro"):
    print(k, {n: round(np.mean(pa[f"{k}/{n}"]), 3) for n in ("L1_S1", "L1_2D", "F_S1", "F_2D")}, pa[f"{k}/base"])
print({k: v for k, v in pb.items() if "3D" in k or "S1" in k})
