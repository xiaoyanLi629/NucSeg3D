"""Real-data rasters for Fig. 1 variant A8-d (writes assets8d/; reads only existing local data, nothing is invented).

(1) out_{dz,dy,dx,p}.png: S1 seed-0 raw outputs of the identity view (network z = volume z) on the xz section through GT nucleus 62
    of mouse_embryo E15_Ab2 (assets3/views_E15_Ab2_n62.npz key '(0,1,2)', computed on the server by arch_views.py: one 32-plane
    window, 256 x 256 laterally, isotropic 30-px grid). Channels [d_z, d_y, d_x, p] (net3d.py:295,299). Flows shown as raw/5
    (the network is trained on 5 x the unit flow) with the palette diverging ramp, clipped to [-1, 1]; p shown as sigmoid(logit)
    on a white-to-ink ramp. Nucleus-62 outline = alpha mask of assets3/view_z.png (make_a3_assets.py), sampled at cell centres.
(2) heat_{10,40,160,522}.png: the flows3d_gpu heat T inside nucleus 62 after 10, 40, 160 and 522 (= niter) iterations
    (assets3/diffusion.npz from diffusion_data.py), xz slice through the source voxel, log10 scale over three decades below
    each snapshot's maximum, masked to the nucleus.
Usage: python make_a8d_assets.py
"""
import json
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets8d")
os.makedirs(OUT, exist_ok=True)
PJ = json.load(open(os.path.join(HERE, "..", "palette.json")))
M = PJ["models"]
DIV = LinearSegmentedColormap.from_list("div", PJ["diverging"])
HEAT = LinearSegmentedColormap.from_list("heat", ["#ffffff", M["cellos"], M["S1"], M["S1_dark"]])
INKR = LinearSegmentedColormap.from_list("ink", ["#ffffff", PJ["text"]])
OUTL = np.array([int(PJ["gt_outline"][i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0])
LOG = {}


def outline(mask):
    m = mask.astype(bool)
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] != m[:-1, :]
    e[:-1, :] |= m[1:, :] != m[:-1, :]
    e[:, 1:] |= m[:, 1:] != m[:, :-1]
    e[:, :-1] |= m[:, 1:] != m[:, :-1]
    return e & m


def save(rgba, mask, name, up=10):
    big = np.kron(rgba, np.ones((up, up, 1)))
    if mask is not None:
        mb = np.kron(mask.astype(np.uint8), np.ones((up, up), np.uint8)).astype(bool)
        big[outline(mb)] = OUTL
    Image.fromarray((np.clip(big, 0, 1) * 255).astype(np.uint8), "RGBA").save(os.path.join(OUT, name))


# ---------------- (1) output maps
V = np.load(os.path.join(HERE, "assets3", "views_E15_Ab2_n62.npz"))
o = V["(0,1,2)"].astype(np.float32)                      # (4, 28, 48): d_z, d_y, d_x, p (raw)
a = np.asarray(Image.open(os.path.join(HERE, "assets3", "view_z.png")).convert("RGBA"))[..., 3]
nz, nx = o.shape[1:]
cy = ((np.arange(nz) + 0.5) * a.shape[0] / nz).astype(int)
cx = ((np.arange(nx) + 0.5) * a.shape[1] / nx).astype(int)
mk = a[cy][:, cx] > 128
fin = np.isfinite(o).all(0)
stats = {}
for i, nm in enumerate(("dz", "dy", "dx")):
    v = np.clip(np.nan_to_num(o[i]) / 5.0, -1, 1)
    rgba = DIV((v + 1) / 2)
    rgba[~fin] = [0.85, 0.85, 0.85, 1]
    save(rgba, mk, f"out_{nm}.png")
    stats[nm] = dict(min_raw=float(np.nanmin(o[i])), max_raw=float(np.nanmax(o[i])))
pp = 1 / (1 + np.exp(-np.nan_to_num(o[3])))
rgba = INKR(pp * 0.85)
save(rgba, mk, "out_p.png")
stats["p"] = dict(min_logit=float(np.nanmin(o[3])), max_logit=float(np.nanmax(o[3])))
LOG["outputs"] = dict(source="assets3/views_E15_Ab2_n62.npz '(0,1,2)' (arch_views.py, S1 seed 0, identity view)", shape=[nz, nx],
                      display="d/5 clipped to [-1,1], palette diverging; p = sigmoid(logit) x 0.85 on white-to-ink", stats=stats,
                      covered=float(fin.mean()), mask="view_z.png alpha (nucleus 62)")

# ---------------- (2) heat snapshots
D = np.load(os.path.join(HERE, "assets3", "diffusion.npz"))
L, k = D["L"], int(D["k62"])
cz, cy_, cx_ = (int(v) for v in D["centre"])
m = L[:, cy_, :] == k
for s in (10, 40, 160, 522):
    T = D[f"T{s}"][:, cy_, :].astype(np.float64)
    mx = T[m].max()
    v = np.clip((np.log10(np.maximum(T, 1e-30) / mx) + 3) / 3, 0, 1)
    rgba = HEAT(v)
    rgba[..., 3] = np.where(m, 1.0, 0.0)
    save(rgba, m, f"heat_{s}.png", up=8)
LOG["heat"] = dict(source="assets3/diffusion.npz (diffusion_data.py, flows3d_gpu update on GT nucleus 62, E15_Ab2, training grid)",
                   slice=f"xz at y = {cy_} (source voxel row)", scale="log10(T / max T in nucleus), 3 decades", snapshots=[10, 40, 160, 522],
                   niter=int(D["niter"]))
json.dump(LOG, open(os.path.join(OUT, "a8d_assets_log.json"), "w"), indent=1)
print(json.dumps(LOG, indent=1))

# ---------------- (3) one 8x8 patch of the tokenised plane (plane_2.png = the xy plane shown in the input stack)
P2 = np.asarray(Image.open(os.path.join(HERE, "assets", "plane_2.png")).convert("RGBA")).astype(np.float32)
g = P2[..., :3].mean(-1)
best, rc = -1, (0, 0)
for r in range(8, 24):                     # interior patches only
    for c in range(8, 24):
        blk = g[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8]
        sc = blk.std() + 0.2 * blk.mean()
        if sc > best:
            best, rc = sc, (r, c)
r, c = rc
patch = P2[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8].astype(np.uint8)
Image.fromarray(np.kron(patch, np.ones((24, 24, 1), np.uint8)), "RGBA").save(os.path.join(OUT, "patch8.png"))
LOG["patch"] = dict(source="assets/plane_2.png (256 x 256 xy plane of the input window)", patch_row_col=[r, c],
                    rule="interior 8x8 patch with the largest std + 0.2 mean of grey value", display="nearest x24")
json.dump(LOG, open(os.path.join(OUT, "a8d_assets_log.json"), "w"), indent=1)
print("patch", rc)
