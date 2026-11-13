"""FINAL supplementary mechanism figure (183 mm wide, NPG palette).
a  one nucleus: GT-derived 3D target and each model's angle (nc-viz V2-i2, nucleus 62, mouse embryo)
b  where along z the angle error sits (nc-viz V2-k2, single volume, descriptive)
c  F1 as a function of the IoU threshold (gt, 12 datasets, five seeds) + S1 - 2D with seed range   errors/results.json gt/S4/all
d  decoder output zeroed, by view group (gt, rule T, five seeds, mean +- s.d.)                       data_snapshot/m1_strat.json gt
"""
import hashlib
import json
import os
import sys

os.environ["NC_PALETTE"] = "npg"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import nc_style as S  # noqa: E402
from nc_style import plt  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "render3d"))
import figcheck  # noqa: E402
from cand_mv2 import crop  # noqa: E402

S.setup()
P = S.PALETTE
SNAP = os.path.abspath(os.path.join(HERE, "..", "..", "data_snapshot"))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "final"))
EJ = os.path.join(HERE, "localdata", "errors_results.json")
assert hashlib.md5(open(EJ, "rb").read()).hexdigest() == "01b398578a871f84d76efb5646f2ed58"
assert hashlib.md5(open(os.path.join(SNAP, "m1_strat.json"), "rb").read()).hexdigest() == "f4ba9ee63204fc169f58b4adadcd8e74"
E = json.load(open(EJ)); M1 = json.load(open(os.path.join(SNAP, "m1_strat.json")))["gt"]
M12 = [k for k in E["gt"]["S4"]["all"] if "mesospim" in k][0]
FG, MUT = P["text"], P["muted"]
W = 183.0
from PIL import Image as _I
_fa = _I.open(os.path.join(os.path.dirname(SNAP), "figures", "assets_viz", "V2-i3_nucleus62.png"))
ia, ib = crop("V2-i3_nucleus62.png"), crop("V2-k2_zprofile.png")
wa = 118 * ia.width / _fa.width; ha = wa * ia.height / ia.width          # native 118-mm scale
_fb = _I.open(os.path.join(os.path.dirname(SNAP), "figures", "assets_viz", "V2-k2_zprofile.png"))
wb = 66 * ib.width / _fb.width; hb = wb * ib.height / ib.width            # native 66-mm scale
H = max(ha, hb) + 12 + 58 + 8
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
prov = {"figure": "FigS_mech_final", "panels": {}}


def ax_mm(x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def letter(x, y, s, t):
    fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", va="top")
    fig.text((x + 4) / W, 1 - (y + 0.3) / H, t, fontsize=6.3, fontweight="bold", va="top", color=FG)


ax = ax_mm(0, 7, wa, ha); ax.imshow(ia); ax.axis("off")
letter(0, 1, "a", "One nucleus: GT-derived 3D flow target and each model's angle (mouse embryo, nucleus 62)")
ax = ax_mm(W - wb, 7, wb, hb); ax.imshow(ib); ax.axis("off")
letter(W - wb - 2, 1, "b", "Where along z the error sits")
y2 = max(ha, hb) + 14
# c
s4 = E["gt"]["S4"]["all"][M12]; g = np.round(0.5 + 0.05 * np.arange(10), 2)
axc = ax_mm(16, y2 + 6, 100, 32); axc2 = ax_mm(16, y2 + 41, 100, 16)
for m in ("S1", "2D", "base"):
    axc.plot(g, s4[m]["curve"], marker=S.MK[m], ms=2.2, color=S.C[m], lw=0.8, label=S.LAB[m])
axc.set_ylabel("macro F1 (12 datasets)"); plt.setp(axc.get_xticklabels(), visible=False); axc.legend(loc="lower left", fontsize=5.3)
d_seed = np.array(s4["S1"]["curve_seeds"]) - np.array(s4["2D"]["curve_seeds"])
axc2.fill_between(g, d_seed.min(0), d_seed.max(0), color=S.C["S1"], alpha=0.22, lw=0)
axc2.plot(g, s4["S1_minus_2D_curve"], marker="o", ms=2, color=S.C["S1"], lw=0.8)
axc2.axhline(0, color=MUT, lw=0.5, ls=(0, (3, 2))); axc2.set_ylabel("S1 − 2D"); axc2.set_xlabel("IoU threshold")
axc2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); axc2.set_yticks([0, 0.02, 0.04]); axc.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
for a_ in (axc, axc2):
    a_.set_xlim(0.48, 0.97)
letter(0, y2, "c", "F1 across IoU thresholds (gt protocol, five seeds)")
prov["panels"]["c"] = {"S1_minus_2D_curve": s4["S1_minus_2D_curve"], "band": "min-max over five paired seeds"}
fig.savefig(os.path.join(OUT, "FigS_mech.pdf")); fig.savefig(os.path.join(OUT, "FigS_mech.png"), dpi=400)
prov["overlap_issues"] = figcheck.check(fig); prov["size_mm"] = [W, round(H, 1)]
json.dump(prov, open(os.path.join(OUT, "FigS_mech.provenance.json"), "w"), indent=1, default=str)
print("saved FigS_mech", round(H), prov["overlap_issues"])
