"""Real-data textures and panes for Fig. 1 v3 (A3-a / A3-b). All from mouse_embryo E15_Ab2, seed 0.

Sources (all read-only, CPU):
  E   paper/scripts/render3d/export/mouse_embryo.npz         raw image, GT (eval grid, vox 1.0/0.215/0.215 um)
  C   <scratch>/viz/v2v4_crop.npz  (nc-viz viz_run.py on hostA, S1 snapshot code): on the resized isotropic grid
      (30/d, d = 36.69 px; 426 x 314 x 314) a 256-px-wide crop (x offset 58) around resized row y = 103 (native y = 126):
        crop_img (Z,3,256) input; s1_enc / c2d_enc (256,Z,5,32) encoder-neck features per plane; s1_dec (256,Z,5,32)
        3D-decoder last hidden features; s1_out (4,Z,3,256) S1 raw output (identity view); c2d_out (3,Z,3,256) 2D control
        raw output on xy planes only (dy, dx, cp).
  F   <scratch>/npz/{S1,2D}.npz  thr_eval saved dP (as evaluated: S1 3 views; 2D control Cellpose 3D = xy, xz, yz fused),
      native z planes (run 07992b55 / 81116913).
  T   <scratch>/viz/gtflow_E15_Ab2.npz  GT-derived 3D target (flows3d_gpu of S1's data3d.py on the isotropic grid).
Window shown on the tensor blocks: the network z window starting at resized plane 162 (one of the 18 windows of
infer3d._starts for this volume), i.e. 32 planes, which contains GT nucleus 62.
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_FIG_SCRATCH = _nsp_os.environ.get('NUCSEG3D_FIG_SCRATCH', '/path/to/figure_scratch')
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "assets3")
os.makedirs(OUT, exist_ok=True)
SCR = f"{_NSP_FIG_SCRATCH}"
PJ = json.load(open(os.path.join(PAPER, "figures", "palette.json")))
M = PJ["models"]
LOG = {}

E = np.load(os.path.join(PAPER, "scripts", "render3d", "export", "mouse_embryo.npz"))
img, gt, vox, d = E["img"].astype(np.float32) / 255.0, E["gt"], E["vox"], float(E["d"])
ratio = float(vox[0] / vox[1])
C = np.load(os.path.join(SCR, "viz", "v2v4_crop.npz"))
cx0, Zr = int(C["cx0"]), int(C["new_shape"][0])
s = 30.0 / d                       # resize factor (lateral); z: ratio * s
W0 = 162                           # window start (resized planes), one of infer3d._starts for Z' = 426
ZW = 32
LOG["window"] = dict(start=W0, planes=ZW, z_windows=json.load(open(os.path.join(SCR, "viz", "v2v4_crop.json")))["z_windows"])


def norm01(a, lo=1, hi=99.7):
    l, h = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - l) / max(h - l, 1e-6), 0, 1)


def save_rgb(a, name):
    plt.imsave(os.path.join(OUT, name), a)


GRAY = plt.get_cmap("gray")
# ------------------------------------------------------------------ input block faces (resized isotropic grid)
# native coordinates of the resized grid: z_n = z_r / (ratio*s), y_n = y_r / s, x_n = (x_r + cx0) / s
zr = np.arange(W0, W0 + ZW); yr = np.arange(0, 256); xr = np.arange(0, 256)


def sample(zz, yy, xx):
    return ndi.map_coordinates(img, [zz / (ratio * s), yy / s, (xx + cx0) / s], order=1, mode="nearest")


Zg, Yg, Xg = np.meshgrid([W0], yr, xr, indexing="ij")
front = sample(Zg, Yg, Xg)[0]                                         # xy plane at the first plane of the window
Zg, Xg = np.meshgrid(zr, xr, indexing="ij")
top_in = C["crop_img"][W0:W0 + ZW, 1, :].astype(np.float32)          # xz section at resized y = 103 (network input)
Zg, Yg = np.meshgrid(zr, yr, indexing="ij")
side = sample(Zg, Yg, np.full_like(Zg, 255))                          # yz section at the right edge
lo, hi = np.percentile(front, 1), np.percentile(front, 99.7)
save_rgb(GRAY(np.clip((front - lo) / (hi - lo), 0, 1)), "in_front.png")
save_rgb(GRAY(norm01(side.T)), "in_side.png")
t = top_in; save_rgb(GRAY(norm01(t)), "in_top.png")                  # rows = z (front -> back), cols = x
LOG["input_faces"] = "front = xy plane z_r=162 (linear interp. of native planes); top = crop_img xz section (resized y 103) planes 162-193; side = yz at x_r = 255"


# ------------------------------------------------------------------ feature faces: PC1 of the xz section, window planes
def pc1(F):
    """F: (C, Z, X) -> first principal component image (Z, X), sign chosen to correlate with GT foreground."""
    Cc, Zz, Xx = F.shape
    A = F.reshape(Cc, -1).T.astype(np.float64)
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    u, sv, vt = np.linalg.svd(A, full_matrices=False)
    p = (A @ vt[0]).reshape(Zz, Xx)
    return p, float(sv[0] ** 2 / (sv ** 2).sum())


# GT foreground on the same token grid for the sign convention
gz = np.clip(np.round(zr / (ratio * s)).astype(int), 0, gt.shape[0] - 1)
gx = np.clip(np.round((np.arange(32) * 8 + 4 + cx0) / s).astype(int), 0, gt.shape[2] - 1)
fg_tok = (gt[gz][:, 126][:, gx] > 0).astype(float)
ramps = {"enc": LinearSegmentedColormap.from_list("e", ["#ffffff", M["base"], PJ["models"]["stardist3d"]]),
         "dec": LinearSegmentedColormap.from_list("d", ["#ffffff", M["S1"], M["S1_dark"]])}
for key, nm in (("s1_enc", "enc"), ("s1_dec", "dec")):
    F = C[key][:, W0:W0 + ZW, 2, :].astype(np.float32)               # (256, 32 planes, 32 tokens)
    p, ev = pc1(F)
    if np.corrcoef(p.ravel(), fg_tok.ravel())[0, 1] < 0:
        p = -p
    save_rgb(ramps[nm](norm01(p, 2, 98)), f"feat_{nm}_top.png")
    # smoothness along z: mean |difference between adjacent planes| / std
    LOG[f"feat_{nm}"] = dict(pc1_var=ev, z_roughness=float(np.abs(np.diff(p, axis=0)).mean() / p.std()),
                             corr_fg=float(abs(np.corrcoef(p.ravel(), fg_tok.ravel())[0, 1])))
# output block top face: S1 raw output u_z on the xz section (identity view), window planes, 256 px
DIV = LinearSegmentedColormap.from_list("div", PJ["diverging"])
o = C["s1_out"][:, W0:W0 + ZW, 1, :].astype(np.float32)               # (4, 32, 256): dz, dy, dx, cp
nrm = np.sqrt((o[:3] ** 2).sum(0)) + 1e-6
uz = o[0] / nrm
cpm = 1 / (1 + np.exp(-o[3]))
rgb = DIV((uz + 1) / 2)[..., :3]
rgb = rgb * cpm[..., None] + (1 - cpm[..., None]) * 1.0              # fade background (p small) to white
save_rgb(np.clip(rgb, 0, 1), "out_top.png")
LOG["out_top"] = "S1 identity-view raw output, u_z = dz/|d| coloured with palette diverging, blended to white by sigmoid(cp)"

# ------------------------------------------------------------------ comparison panes on nucleus 62 (xz, native y = 126)
T = np.load(os.path.join(SCR, "viz", "gtflow_E15_Ab2.npz"))["t"].astype(np.float32)
Fd = {m: np.load(os.path.join(SCR, "npz", f"{m}.npz"))["dP"][:, :, 126, :].astype(np.float32) for m in ("2D", "S1")}
G = gt[:, 126, :]
nuc = 62
zz, xx = np.nonzero(G == nuc)
z0, z1 = zz.min() - 1, zz.max() + 2
x0, x1 = xx.min() - 6, xx.max() + 7
LOG["compare"] = dict(nucleus=nuc, y_native=126, z=[int(z0), int(z1)], x=[int(x0), int(x1)])


def pane(u_x, u_z, mask, name, xs, zs, lab_mask, step=(1, 3), outline_xy=None, uniform=True):
    """u_x, u_z on a grid with physical coords xs (um), zs (um). Arrows: unit in-plane direction, colour u_z (3D)."""
    W_um, H_um = (x1 - x0) * vox[2], (z1 - z0) * vox[0]
    fig = plt.figure(figsize=(W_um / 4.0, H_um / 4.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    # nucleus fill and outline (native GT)
    gx_um = (np.arange(x0, x1) + 0.5) * vox[2]; gz_um = (np.arange(z0, z1) + 0.5) * vox[0]
    m = (G[z0:z1, x0:x1] == nuc).astype(float)
    ax.imshow(np.where(m[..., None] > 0, np.array([0.93, 0.93, 0.95, 1.0]), np.array([1, 1, 1, 0.0])),
              extent=[x0 * vox[2], x1 * vox[2], z1 * vox[0], z0 * vox[0]], interpolation="nearest", aspect="auto")
    ax.contour(gx_um, gz_um, m, [0.5], colors=[PJ["gt_outline"]], linewidths=0.6)
    Xs, Zs = np.meshgrid(xs, zs)
    sel = mask
    U, V, Cz = u_x[sel], u_z[sel], u_z[sel]
    ax.quiver(Xs[sel], Zs[sel], U, V, Cz, cmap=DIV, clim=(-1, 1), angles="xy", scale_units="xy", scale=1 / 0.62,
              width=0.012, headwidth=3.4, headlength=3.6, headaxislength=3.2, edgecolor=PJ["text"], linewidth=0.3, pivot="mid")
    ax.set_xlim(x0 * vox[2], x1 * vox[2]); ax.set_ylim(z1 * vox[0], z0 * vox[0])
    fig.savefig(os.path.join(OUT, name), transparent=True)
    plt.close(fig)


def unit(*c):
    n = np.sqrt(sum(a ** 2 for a in c)) + 1e-9
    return [a / n for a in c]


# native-grid panes: one arrow per native plane (z), every 4th px in x
zi = np.arange(z0, z1); xi = np.arange(x0, x1, 4)
Zi, Xi = np.meshgrid(zi, xi, indexing="ij")
inside = G[Zi, Xi] == nuc
xs = (xi + 0.5) * vox[2]; zs = (zi + 0.5) * vox[0]
for key, arr, nm in (("2D", Fd["2D"], "cmp_2d_fused.png"), ("S1", Fd["S1"], "cmp_s1.png"),
                     ("T", T[:, :, 126, :], "cmp_target.png")):
    dz, dy, dx = arr[0][Zi, Xi], arr[1][Zi, Xi], arr[2][Zi, Xi]
    uz, uy, ux = unit(dz, dy, dx)
    pane(ux, uz, inside, nm, xs, zs, inside)
    LOG[f"cmp_{key}_mean_abs_uz"] = float(np.abs(uz[inside]).mean())
# 2D control, xy planes only: resized grid; one arrow per resized plane (0.263 um) would be dense -> every 4th plane
pr = vox[2] / s                                                       # resized px size (um)
zr_all = np.arange(Zr)
zsel = zr_all[(zr_all * pr >= z0 * vox[0]) & (zr_all * pr < z1 * vox[0])][::4]
xsel = np.arange(256)[((np.arange(256) + cx0) * pr >= x0 * vox[2]) & ((np.arange(256) + cx0) * pr < x1 * vox[2])][::4]
Zq, Xq = np.meshgrid(zsel, xsel, indexing="ij")
o2 = C["c2d_out"].astype(np.float32)                                   # (dy, dx, cp) on xy planes
dy2, dx2 = o2[0][Zq, 1, Xq], o2[1][Zq, 1, Xq]
uy2, ux2 = unit(dy2, dx2)
zn = np.clip(np.floor(Zq * pr / vox[0]).astype(int), 0, G.shape[0] - 1)
xn = np.clip(np.floor((Xq + cx0) * pr / vox[2]).astype(int), 0, G.shape[1] - 1)
ins2 = G[zn, xn] == nuc
pane(ux2, np.zeros_like(ux2), ins2, "cmp_2d_xyonly.png", (xsel + cx0 + 0.5) * pr, (zsel + 0.5) * pr, ins2)
LOG["cmp_2d_xyonly"] = "c2d_out (2D control on xy planes only, resized grid, every 4th plane and px), u_z = 0 by construction"
# S1 single (identity) view on the same resized grid, for the zero-init inset (step 0 = 2D control on xy planes)
o1 = C["s1_out"].astype(np.float32)
dz1, dx1 = o1[0][Zq, 1, Xq], o1[2][Zq, 1, Xq]
dy1 = o1[1][Zq, 1, Xq]
uz1, uy1, ux1 = unit(dz1, dy1, dx1)
pane(ux1, uz1, ins2, "cmp_s1_view1.png", (xsel + cx0 + 0.5) * pr, (zsel + 0.5) * pr, ins2)
LOG["cmp_s1_view1"] = "s1_out identity view (model z = volume z), same grid and sampling as cmp_2d_xyonly"
json.dump(LOG, open(os.path.join(OUT, "a3_assets_log.json"), "w"), indent=1, default=float)
print(json.dumps(LOG, indent=1, default=float))


# ------------------------------------------------------------------ A3-c: angle to the 3D target (FLOWERR definition)
# theta = degrees(arccos(d . t / (|d| |t|))) per voxel, region (a): GT foreground with |t| > 0.5 (flowerr_angles.py);
# shown on GT nucleus 62 of the xz section (native y = 126), native planes; 2D control and S1 as evaluated (seed 0).
ANG = LinearSegmentedColormap.from_list("ang", ["#ffffff", M["cellos"], M["S1"], M["S1_dark"]])
Tz = T[:, z0:z1, 126, x0:x1]
tn = np.sqrt((Tz ** 2).sum(0))
reg = (G[z0:z1, x0:x1] == nuc) & (tn > 0.5)
med = {}
for key, nm in (("2D", "ang_2d.png"), ("S1", "ang_s1.png")):
    D = Fd[key][:, z0:z1, x0:x1]
    c = (D * Tz).sum(0) / (np.sqrt((D ** 2).sum(0)) + 1e-6) / (tn + 1e-6)
    th = np.degrees(np.arccos(np.clip(c, -1, 1)))
    med[key] = float(np.median(th[reg]))
    W_um, H_um = (x1 - x0) * vox[2], (z1 - z0) * vox[0]
    fig = plt.figure(figsize=(W_um / 4.0, H_um / 4.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    rgba = ANG(np.clip(th / 90.0, 0, 1)); rgba[..., 3] = np.where(G[z0:z1, x0:x1] == nuc, 1.0, 0.0)
    ax.imshow(rgba, extent=[x0 * vox[2], x1 * vox[2], z1 * vox[0], z0 * vox[0]], interpolation="nearest", aspect="auto")
    m_ = (G[z0:z1, x0:x1] == nuc).astype(float)
    ax.contour((np.arange(x0, x1) + 0.5) * vox[2], (np.arange(z0, z1) + 0.5) * vox[0], m_, [0.5], colors=[PJ["gt_outline"]], linewidths=0.6)
    dz, dy, dx = Fd[key][0][Zi, Xi], Fd[key][1][Zi, Xi], Fd[key][2][Zi, Xi]
    uz, uy, ux = unit(dz, dy, dx)
    Xs, Zs = np.meshgrid(xs, zs)
    ax.quiver(Xs[inside], Zs[inside], ux[inside], uz[inside], color=PJ["text"], angles="xy", scale_units="xy", scale=1 / 0.62,
              width=0.010, headwidth=3.4, headlength=3.6, headaxislength=3.2, pivot="mid")
    ax.set_xlim(x0 * vox[2], x1 * vox[2]); ax.set_ylim(z1 * vox[0], z0 * vox[0])
    fig.savefig(os.path.join(OUT, nm), transparent=True); plt.close(fig)
LOG["angle_nucleus62"] = dict(definition="flowerr_angles.py: 3D angle, region GT fg & |t| > 0.5", median_deg=med,
                              n_vox=int(reg.sum()), ramp=["#ffffff", M["cellos"], M["S1"], M["S1_dark"]], range=[0, 90])
json.dump(LOG, open(os.path.join(OUT, "a3_assets_log.json"), "w"), indent=1, default=float)
print("angle medians nucleus 62:", med, "n", int(reg.sum()))


# ------------------------------------------------------------------ A4: per-view S1 outputs on nucleus 62 (server arch_views.py)
V3 = np.load(os.path.join(OUT, "views_E15_Ab2_n62.npz"))
zb, xb, rsv = V3["zb"], V3["xb"], float(V3["rs"])
zr_ = np.arange(zb[0], zb[1]); xr_ = np.arange(xb[0], xb[1])
zn_ = np.clip(np.floor((zr_ + 0.5) / (ratio * rsv)).astype(int), 0, gt.shape[0] - 1)
xn_ = np.clip(np.floor((xr_ + 0.5) / rsv).astype(int), 0, gt.shape[2] - 1)
mk = gt[zn_][:, 126][:, xn_] == nuc
views = {"z": V3["(0,1,2)"], "y": V3["(1,0,2)"], "x": V3["(2,0,1)"]}
allv = np.stack([views[k] for k in "zyx"]).astype(np.float32)
views["mean"] = np.nanmean(allv, 0) if np.isfinite(allv).all() else np.where(np.isfinite(allv).all(0), allv.mean(0), np.nan)
vstat = {}
for kname, o in views.items():
    o = o.astype(np.float32)
    n_ = np.sqrt((o[:3] ** 2).sum(0)) + 1e-6
    uz_ = o[0] / n_
    rgba = DIV((np.nan_to_num(uz_) + 1) / 2)
    rgba[..., 3] = np.where(mk & np.isfinite(uz_), 1.0, 0.0)
    rgba[mk & ~np.isfinite(uz_)] = [0.85, 0.85, 0.85, 1.0]
    fig = plt.figure(figsize=(xr_.size / 20, zr_.size / 20), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(rgba, interpolation="nearest", aspect="auto")
    ax.contour(mk.astype(float), [0.5], colors=[PJ["gt_outline"]], linewidths=0.6)
    fig.savefig(os.path.join(OUT, f"view_{kname}.png"), transparent=True); plt.close(fig)
    vstat[kname] = dict(mean_abs_uz=float(np.nanmean(np.abs(uz_[mk]))), covered=float(np.isfinite(uz_[mk]).mean()))
LOG["views_n62"] = dict(source="server /tmp/arch_views.py (copied to arch/arch_views.py): S1 seed 0, one 32-plane window per view centred on nucleus 62, raw outputs; mean = average of the three where all exist",
                        stats=vstat, zb=[int(v) for v in zb], xb=[int(v) for v in xb])
json.dump(LOG, open(os.path.join(OUT, "a3_assets_log.json"), "w"), indent=1, default=float)
print("views", vstat)
