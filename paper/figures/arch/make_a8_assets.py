"""A8 output tiles: the four raw S1 output maps (d_z, d_y, d_x, cell-probability logit) on the xz section through
GT nucleus 62 of mouse embryo E15_Ab2, identity view (0,1,2), from assets3/views_E15_Ab2_n62.npz (arch_views.py,
S1 seed 0, one 32-plane window). No new inference; colour mapping only."""
import json, os
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "..", "palette.json")))
OUT = os.path.join(HERE, "assets8")
d = np.load(os.path.join(HERE, "assets3", "views_E15_Ab2_n62.npz"))
a = d["(0,1,2)"]                       # (4, 28, 48): channel order of the network output (dz, dy, dx, logit), volume layout


def hx(c):
    c = c.lstrip("#"); return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float)


lo, mid, hi = (hx(c) for c in P["diverging"])


def div(v, lim):
    t = np.clip(v / lim, -1, 1)[..., None]
    return np.where(t < 0, mid + (lo - mid) * (-t), mid + (hi - mid) * t)


def seq(v):
    t = np.clip(v, 0, 1)[..., None]
    return hx("#EDECE9") + (hx("#3a3a3e") - hx("#EDECE9")) * t


log = {}
for k, nm in enumerate(["dz", "dy", "dx", "p"]):
    v = a[k].astype(float)
    nan = ~np.isfinite(v)
    v = np.nan_to_num(v)
    rgb = seq(1 / (1 + np.exp(-v))) if nm == "p" else div(v, 5.0)
    rgb[nan] = 255
    im = Image.fromarray(rgb.astype(np.uint8)).resize((48 * 8, 28 * 8), Image.NEAREST)
    im.save(os.path.join(OUT, f"out_{nm}.png"))
    log[nm] = dict(channel=k, min=float(v.min()), max=float(v.max()), nan_frac=float(nan.mean()),
                   mapping="palette diverging, clipped at +-5 (= 5 x unit flow)" if nm != "p" else "sigmoid(logit), #EDECE9 -> #3a3a3e")
from PIL import Image as _I
for nm in ("diff_heat", "diff_flow"):
    im = _I.open(os.path.join(HERE, "assets3", nm + ".png")).convert("RGBA")
    a_ = np.array(im); m = (a_[..., 3] > 10) & (a_[..., :3].sum(-1) < 740); ys, xs = np.where(m)
    box = (max(xs.min() - 8, 0), max(ys.min() - 8, 0), min(xs.max() + 8, im.size[0]), min(ys.max() + 8, im.size[1]))
    im.crop(box).save(os.path.join(OUT, nm + ".png")); log[nm] = dict(source="assets3/" + nm + ".png", crop_box_px=[int(v) for v in box])
json.dump(dict(source="assets3/views_E15_Ab2_n62.npz key (0,1,2) (arch_views.py, S1 seed 0 orx_t3d_r5s1_s0, raw outputs)",
               section="xz through GT nucleus 62, rows z (28, isotropic grid), cols x (48)", tiles=log),
          open(os.path.join(OUT, "assets8_log.json"), "w"), indent=1)
print(log)
