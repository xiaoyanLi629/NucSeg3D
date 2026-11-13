"""Raster thumbnails for the whole-process Fig. 1 schematic (A2-a / A2-b), all from real data.

Source volume: mouse_embryo E15_Ab2 test volume as evaluated (paper/scripts/render3d/export/mouse_embryo.npz:
img, gt, vox = (1.0, 0.215, 0.215) um, d_gt = 36.69 px, z/xy = 4.65).
The 3D Cellpose training target is computed with the project's own function data3d.flows3d_gpu (copied verbatim from
the S1 run snapshot $NUCSEG3D_RUNS/07992b55-.../repo/src/data3d.py, run here on CPU), on a crop resampled the way the
training store is built (isotropic, median diameter 30 px; labels nearest-exact as in data3d.to_patch).
Outputs go to paper/figures/arch/assets/ and a JSON log of every choice to assets/assets_log.json.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
SNAP = os.environ.get("ARCH_SRC", os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))  # release: repository src/
sys.path.insert(0, SNAP)
from data3d import flows3d_gpu  # noqa: E402  (verbatim S1-snapshot code)

LOG = {}
V = np.load(os.path.join(PAPER, "scripts", "render3d", "export", "mouse_embryo.npz"))
img, gt, vox, d = V["img"].astype(np.float32) / 255.0, V["gt"], V["vox"], float(V["d"])
ratio = float(vox[0] / vox[1])
Z, Y, X = img.shape
LOG["volume"] = dict(ds="mouse_embryo", id="E15_Ab2", shape=[Z, Y, X], vox=vox.tolist(), ratio=ratio, d_gt=d)

# colours from paper/figures/palette.json
PJ = json.load(open(os.path.join(PAPER, "figures", "palette.json")))
PAL = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in PJ["qualitative_instances"]])
DZ = LinearSegmentedColormap.from_list("dz", PJ["diverging"])
OUTLINE = PJ["gt_outline"]


def save_gray(a, name, lo=None, hi=None):
    lo = np.percentile(a, 1) if lo is None else lo
    hi = np.percentile(a, 99.7) if hi is None else hi
    u = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    plt.imsave(os.path.join(OUT, name), u, cmap="gray", vmin=0, vmax=1)


def lab_rgb(lab, bg=(1, 1, 1), alpha=None):
    out = np.ones(lab.shape + (4,))
    out[..., :3] = bg
    m = lab > 0
    out[m, :3] = PAL[lab[m] % len(PAL)]
    if alpha is not None:
        out[..., 3] = np.where(m, 1.0, alpha)
    return out


# ------------------------------------------------------------------ choose planes deterministically
fgz = np.array([(gt[k] > 0).sum() for k in range(Z)])
zc = int(np.argmax(fgz))                      # most foreground-rich xy plane
y0 = x0 = 64; S = 256                          # 256 x 256 lateral window (network input size)
LOG["xy_plane"] = dict(z=zc, rule="xy plane with most GT foreground", window=[y0, y0 + S, x0, x0 + S])

save_gray(img[zc, y0:y0 + S, x0:x0 + S], "in_xy.png")
plt.imsave(os.path.join(OUT, "gt_xy.png"), lab_rgb(gt[zc, y0:y0 + S, x0:x0 + S]))
# ghost stack for the per-plane encoder: 5 planes around zc, step 3 (native planes)
ks = [zc - 6, zc - 3, zc, zc + 3, zc + 6]
lo, hi = np.percentile(img[zc], 1), np.percentile(img[zc], 99.7)
for i, k in enumerate(ks):
    save_gray(img[k, y0:y0 + S, x0:x0 + S], f"plane_{i}.png", lo, hi)
LOG["planes"] = ks

# ------------------------------------------------------------------ xz plane: native sampling vs isotropic grid
yc = y0 + S // 2
zr = np.where(fgz > 0)[0]
zs, ze = max(zc - 14, 0), min(zc + 14, Z)
xz = img[zs:ze, yc, x0 + 64:x0 + 192]
up = int(round(ratio))
save_gray(np.repeat(xz, up, axis=0), "xz_native.png")                    # each acquired plane drawn as a band
zi = ndi.zoom(xz, (ratio, 1.0), order=1)                                   # linear interpolation along z
save_gray(zi, "xz_iso.png")
LOG["xz"] = dict(y=yc, z_range=[zs, ze], native="nearest repeat x%d (display only)" % up,
                 iso="linear along z by z/xy = %.2f (eval_gt.zinterp is linear along z)" % ratio)

# ------------------------------------------------------------------ 3D Cellpose target on the training grid
# crop rule (deterministic, not chosen by any model result): within the slab zc +- 12 planes, the 112 x 112-px lateral
# window (step 8) that contains the most distinct GT nuclei; the xz display plane is the row crossing the most nuclei.
hz, h = 12, 56
slab = gt[max(zc - hz, 0):zc + hz]
best = (-1, 0, 0)
for yy in range(0, Y - 2 * h + 1, 8):
    for xx in range(0, X - 2 * h + 1, 8):
        n = len(np.unique(slab[:, yy:yy + 2 * h, xx:xx + 2 * h])) - 1
        if n > best[0]:
            best = (n, yy, xx)
nn_, cy, cx = best[0], best[1] + h, best[2] + h
cz = zc; nid = None
sl = (slice(max(cz - hz, 0), cz + hz), slice(cy - h, cy + h), slice(cx - h, cx + h))
lab = gt[sl]; im = img[sl]
s = 30.0 / d                                                               # median diameter -> 30 px
shape_iso = (int(round(lab.shape[0] * ratio * s)), int(round(lab.shape[1] * s)), int(round(lab.shape[2] * s)))
u, inv = np.unique(lab, return_inverse=True)
lt = torch.from_numpy(inv.reshape(lab.shape).astype(np.float32))[None, None]
lt = F.interpolate(lt, size=shape_iso, mode="nearest-exact")[0, 0].round().long()
it = F.interpolate(torch.from_numpy(im)[None, None], size=shape_iso, mode="trilinear", align_corners=False)[0, 0]
mu = flows3d_gpu(lt).numpy()                                               # (3, Z, Y, X) unit (dz, dy, dx)
L = lt.numpy(); I = it.numpy()
rows = [len(np.unique(L[:, r, :])) - 1 for r in range(L.shape[1])]
cc = np.array([L.shape[0] // 2, int(np.argmax(rows)), L.shape[2] // 2])
LOG["target"] = dict(n_nuclei_in_crop=nn_, rule="112x112 lateral window (step 8) with most distinct GT nuclei in slab zc+-12; xz row crossing most nuclei",
                     crop=[[a.start, a.stop] for a in sl], iso_shape=list(shape_iso), scale=s,
                     fn="data3d.flows3d_gpu (S1 snapshot), CPU, float64 diffusion", centre_iso=cc.tolist())


def quiver_plane(L2, a, b, comp_col, name, im2=None, step=5, outline=True, flat=False):
    """a, b: in-plane flow components (rows, cols); comp_col: component used for colour (dz)."""
    H, W = L2.shape
    fig = plt.figure(figsize=(W / 40, H / 40), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    if im2 is not None:
        ax.imshow(im2, cmap="gray", vmin=np.percentile(im2, 1), vmax=np.percentile(im2, 99.7), interpolation="bilinear")
    else:
        ax.imshow(lab_rgb(L2, alpha=0.0) * np.array([1, 1, 1, 0.30]), interpolation="nearest")
    if outline:
        for k in np.unique(L2[L2 > 0]):
            ax.contour((L2 == k).astype(float), [0.5], colors=[OUTLINE], linewidths=0.45)
    yy, xx = np.mgrid[step // 2:H:step, step // 2:W:step]
    m = L2[yy, xx] > 0
    A, B, C = a[yy, xx][m], b[yy, xx][m], comp_col[yy, xx][m]
    if flat:
        ax.quiver(xx[m], yy[m], B, A, color=PJ["text"], angles="xy", scale_units="xy", scale=0.26, width=0.009,
                  headwidth=3.2, headlength=3.4, headaxislength=3.0)
    else:
        ax.quiver(xx[m], yy[m], B, A, C, cmap=DZ, clim=(-1, 1), angles="xy", scale_units="xy", scale=0.26,
                  edgecolor=PJ["text"], linewidth=0.25, width=0.009, headwidth=3.2, headlength=3.4, headaxislength=3.0)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    fig.savefig(os.path.join(OUT, name), transparent=True)
    plt.close(fig)


# xz plane through the nucleus centre: rows = z (dz), cols = x (dx); colour = dz
yP = cc[1]
quiver_plane(L[:, yP, :], mu[0, :, yP, :], mu[2, :, yP, :], mu[0, :, yP, :], "target_xz.png")
# xy plane through the centre: rows = y, cols = x; colour = dz
zP = cc[0]
quiver_plane(L[zP], mu[1, zP], mu[2, zP], mu[0, zP], "target_xy.png")
# foreground target (cp channel = nucleus or not) on the xz plane
fgt = (L[:, yP, :] > 0).astype(float)
plt.imsave(os.path.join(OUT, "target_fg_xz.png"), fgt, cmap=LinearSegmentedColormap.from_list("f", [PJ["bg"], PJ["text"]]),
           vmin=0, vmax=1)
# matching image planes on the training grid
save_gray(I[:, yP, :], "train_img_xz.png"); save_gray(I[zP], "train_img_xy.png")
plt.imsave(os.path.join(OUT, "train_gt_xz.png"), lab_rgb(L[:, yP, :]))
# 2D Cellpose target on one plane (same diffusion restricted to the plane: Z = 1, so dz = 0)
mu2 = flows3d_gpu(lt[zP:zP + 1]).numpy()[:, 0]
quiver_plane(L[zP], mu2[1], mu2[2], mu2[0], "target2d_xy.png", flat=True)
LOG["target2d"] = "flows3d_gpu on the single plane z = %d of the iso crop (diffusion within the plane only, dz = 0)" % zP
LOG["dz_range_on_xz"] = [float(mu[0, :, yP, :].min()), float(mu[0, :, yP, :].max())]

# ------------------------------------------------------------------ sliding-window geometry (infer3d.run_volume)
def starts(Lz, w, ov):
    import math
    if Lz <= w:
        return [0]
    n = int(math.ceil((Lz - w) / (w * (1 - ov)))) + 1
    return sorted({int(round(v)) for v in np.linspace(0, Lz - w, n)})


Zu = int(round(Z * ratio))                     # eval_gt: z/xy >= 2 -> linear z interpolation to Zu planes, an = 1
Zp = int(Zu * 1.0 * (30.0 / d))                # infer3d.eval: int(Lz * an * rescale)
st = starts(Zp, 32, 0.25)
LOG["windows"] = dict(Zu=Zu, Z_network=Zp, starts=st, n=len(st), strides=sorted(set(np.diff(st).tolist())),
                      taper="sin^2(pi (k + 0.5) / 32) + 1e-3 (infer3d._taper)")
json.dump(LOG, open(os.path.join(OUT, "assets_log.json"), "w"), indent=1, default=int)
print(json.dumps(LOG, indent=1, default=int))
