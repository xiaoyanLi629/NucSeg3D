"""V2 detailed mechanism panels (V2-h ... V2-k), mouse_embryo E15_Ab2 (Fig. 2d volume), seed 0, 2D control (fused, as evaluated) vs S1.

Target: the FLOWERR GT-derived 3D flow target for this volume ($NUCSEG3D_SCRATCH/nc/flowerr/gtflow/mouse_embryo__E15_Ab2.npz,
flows3d_gpu from S1's data3d.py on the isotropic grid, values at native planes). Model flows: saved thr_eval dP (native planes).
Angle = arccos of the normalised dot product, degrees, at GT foreground with |t| > 0.5.
  V2-h: dense, high-resolution xz section (y = 126, x 119-375, z 16-82): angle map with fine quiver (every 4 px, every plane) on top.
  V2-i: zoom on GT nucleus 62 (anchor of the section rule): GT target arrows | 2D fused (angle map + arrows) | S1 (angle map + arrows),
        arrows every 2 px and every plane.
  V2-j: xy section (native z = 47, the plane of nucleus 62's centroid; y 0-256, x 119-375) next to the xz section; rows 2D fused, S1;
        dashed lines mark where the two sections cross.
  V2-k: per-nucleus z-profile for this volume: every GT nucleus spanning >= 5 native planes; relative z r = (z - z_first)/(z_last - z_first)
        of the nucleus (0 = first plane, 1 = last plane); 10 bins; per nucleus and bin the median angle; line = median over nuclei,
        band = interquartile range. Single-volume illustration (descriptive), not a FLOWERR result.
Arrows: in-plane projection of the unit flow, colour = u_z on the diverging map; angle maps: sequential map 0-90 deg."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_PAPER = _nsp_os.environ.get('NUCSEG3D_PAPER', _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", "..")))
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import vstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

V.setup()
PAL = V.palette()
DIV = V.diverging(); SEQ = V.sequential()
E = np.load(f"{_NSP_PAPER}/scripts/render3d/export/mouse_embryo.npz")
vox = E["vox"]; vz, vxy = float(vox[0]), float(vox[1])
gt = E["gt"]
T = np.load(f"{V.SCR}/viz/gtflow_E15_Ab2.npz")["t"].astype(np.float32)
F = {m: np.load(f"{V.SCR}/npz/{m}.npz")["dP"].astype(np.float32) for m in ("2D", "S1")}
LAB = {"2D": "2D control (as evaluated, 3 views fused)", "S1": "S1 (as evaluated)"}
tn = np.sqrt((T ** 2).sum(0)); valid = (gt > 0) & (tn > 0.5)


def ang(d):
    c = (d * T).sum(0) / (np.sqrt((d ** 2).sum(0)) + 1e-6) / (tn + 1e-6)
    return np.where(valid, np.degrees(np.arccos(np.clip(c, -1, 1))), np.nan)


A = {m: ang(F[m]) for m in F}
Y, X0, XW, Z0, Z1, ZXY = 126, 119, 256, 16, 82, 47
Y0, Y1 = 0, 256
PROV = dict(volume="mouse_embryo E15_Ab2", seed="s0", target=f"{_NSP_SCRATCH}/nc/flowerr/gtflow/mouse_embryo__E15_Ab2.npz (FLOWERR target)",
            flows="thr_eval masks npz dP (runs 07992b55 / 81116913)", palette=V.palette_source(), script=os.path.abspath(__file__),
            caveat="S1 is trained on this target type; the angle shows agreement with the 3D training target, not segmentation quality")


def xz(a):      # (.., Z, Y, X) -> section
    return a[..., Z0:Z1, Y, X0:X0 + XW]


def xy(a):
    return a[..., ZXY, Y0:Y1, X0:X0 + XW]


def outlines(ax, lab, xs, ys, color, lw):
    for i in np.unique(lab):
        if i:
            ax.contour(xs, ys, (lab == i).astype(float), levels=[0.5], colors=[color], linewidths=lw)


def quiv(ax, d, xs, ys, keep, step=(1, 4), L=None, comp=(0, 2), width=0.0018, color=None):
    """d: 3 x H x W section; comp: indices of (vertical, horizontal) components."""
    sv, sh = step
    dv, dh = d[comp[0]][::sv, ::sh], d[comp[1]][::sv, ::sh]
    n = np.sqrt((d ** 2).sum(0))[::sv, ::sh] + 1e-6
    k = keep[::sv, ::sh]
    Xg, Yg = np.meshgrid(xs[::sh], ys[::sv])
    L = L or 0.9 * sh * vxy
    kw = dict(angles="xy", scale_units="xy", scale=1, width=width, headwidth=3.0, headlength=3.2, headaxislength=2.8, pivot="mid")
    if color is None:
        ax.quiver(Xg[k], Yg[k], (dh / n * L)[k], (dv / n * L)[k], np.clip(d[0][::sv, ::sh] / n, -1, 1)[k], cmap=DIV, norm=Normalize(-1, 1), **kw)
    else:
        ax.quiver(Xg[k], Yg[k], (dh / n * L)[k], (dv / n * L)[k], color=color, **kw)


xs_x = (np.arange(XW) + 0.5) * vxy; zs = (np.arange(Z1 - Z0) + 0.5) * vz; ys = (np.arange(Y1 - Y0) + 0.5) * vxy


def angle_panel(ax, a, lab, xs_, ys_, ext, title, d=None, keep=None, step=(1, 4), comp=(0, 2), qcol=None):
    ax.imshow(a, cmap=SEQ, vmin=0, vmax=90, extent=ext, interpolation="nearest", aspect="auto")
    outlines(ax, lab, xs_, ys_, PAL["text"], 0.35)
    if d is not None:
        quiv(ax, d, xs_, ys_, keep, step=step, comp=comp, color=qcol)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=5.5, pad=2)


def cbar(fig, rect, cmap, norm, label, ticks):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.set_label(label, fontsize=5, labelpad=1)


def sbar(ax, x1, y1, um, color=None):
    ax.plot([x1 - um, x1], [y1, y1], color=color or PAL["text"], lw=1.0, solid_capstyle="butt")
    ax.text(x1 - um / 2, y1 - 0.6, f"{um} µm", ha="center", va="bottom", fontsize=5, color=color or PAL["text"])


ext_xz = [0, XW * vxy, (Z1 - Z0) * vz, 0]
gxz = xz(gt)
# ---- V2-h
W = 183 * V.MM; cw = W / 2; asp = ext_xz[2] / ext_xz[1]; H = cw * asp + 16 * V.MM
fig = plt.figure(figsize=(W, H))
for j, m in enumerate(("2D", "S1")):
    ax = fig.add_axes([j / 2 + 0.004, 11 * V.MM / H, 0.5 - 0.008, cw * asp / H])
    angle_panel(ax, xz(A[m]), gxz, xs_x, zs, ext_xz, LAB[m], d=xz(F[m]), keep=gxz > 0, step=(1, 4), qcol=PAL["text"])
sbar(ax, ext_xz[1] - 1.5, ext_xz[2] - 1.5, 10)
cbar(fig, [0.35, 4.5 * V.MM / H, 0.3, 1.6 * V.MM / H], SEQ, Normalize(0, 90), "angle to GT-derived 3D flow (°)", [0, 30, 60, 90])
V.save(fig, "V2-h_angle_quiver_xz_dense", dict(PROV, shows="dense xz section: angle map + flow arrows (every 4 px, every plane; dark)", section=dict(y=Y, x=[X0, X0 + XW], z=[Z0, Z1])))

# ---- V2-i zoom on nucleus 62
zl = gxz == 62; zr_, xr_ = np.nonzero(zl)
zx0, zx1 = max(xs_x[xr_.min()] - 2.5, 0), min(xs_x[xr_.max()] + 2.5, ext_xz[1]); zz0, zz1 = max(zs[zr_.min()] - 2.5, 0), min(zs[zr_.max()] + 2.5, ext_xz[2])
asp = (zz1 - zz0) / (zx1 - zx0); W = 183 * V.MM; cw = W / 3; H = cw * asp + 16 * V.MM
fig = plt.figure(figsize=(W, H))
cols = [("GT", "GT-derived 3D target"), ("2D", LAB["2D"]), ("S1", LAB["S1"])]
for j, (m, t) in enumerate(cols):
    ax = fig.add_axes([j / 3 + 0.004, 11 * V.MM / H, 1 / 3 - 0.008, cw * asp / H])
    if m == "GT":
        rgba = np.zeros((*gxz.shape, 4)); rgba[gxz > 0] = (0.92, 0.93, 0.95, 1)
        ax.imshow(rgba, extent=ext_xz, interpolation="nearest", aspect="auto"); outlines(ax, gxz, xs_x, zs, PAL["text"], 0.4)
        quiv(ax, xz(T), xs_x, zs, xz(valid), step=(1, 2), L=0.8, width=0.0035)
        ax.set_xlim(zx0, zx1); ax.set_ylim(zz1, zz0); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(t, fontsize=5.5, pad=2)
    else:
        angle_panel(ax, xz(A[m]), gxz, xs_x, zs, ext_xz, t, d=xz(F[m]), keep=gxz > 0, step=(1, 2), qcol=PAL["text"])
        ax.set_xlim(zx0, zx1); ax.set_ylim(zz1, zz0)
sbar(ax, zx1 - 0.6, zz1 - 0.6, 3)
cbar(fig, [0.12, 4.5 * V.MM / H, 0.3, 1.6 * V.MM / H], DIV, Normalize(-1, 1), "target arrows: $u_z$ (− up, + down)", [-1, 0, 1])
cbar(fig, [0.58, 4.5 * V.MM / H, 0.3, 1.6 * V.MM / H], SEQ, Normalize(0, 90), "angle to GT-derived 3D flow (°)", [0, 30, 60, 90])
V.save(fig, "V2-i_zoom_nucleus62_angle_quiver", dict(PROV, shows="zoom on GT nucleus 62: target arrows (u_z colour) | 2D fused angle map + dark arrows | S1 angle map + dark arrows; arrows every 2 px, every plane",
                                                    zoom_um=dict(x=[zx0, zx1], z=[zz0, zz1])))

# ---- V2-j xy + xz
ext_xy = [0, XW * vxy, (Y1 - Y0) * vxy, 0]
gxy = xy(gt)
W = 183 * V.MM; wxy, wxz = ext_xy[1], ext_xz[1]
rowh = (W / 2) * max(ext_xy[2] / ext_xy[1], ext_xz[2] / ext_xz[1])
H = 2 * rowh + 18 * V.MM
fig = plt.figure(figsize=(W, H))
for i, m in enumerate(("2D", "S1")):
    yb = (11 * V.MM + (1 - i) * rowh + 3 * V.MM * (1 - i)) / H
    ax = fig.add_axes([0.004, yb, 0.492, rowh / H])
    angle_panel(ax, xy(A[m]), gxy, xs_x, ys, ext_xy, f"{LAB[m]} · xy section (z = {ZXY})", d=xy(F[m]), keep=gxy > 0, step=(4, 4), comp=(1, 2), qcol=PAL["text"])
    ax.axhline(Y * vxy, color=PAL["muted"], lw=0.5, ls=(0, (3, 2)))
    ax2 = fig.add_axes([0.504, yb, 0.492, rowh / H])
    angle_panel(ax2, xz(A[m]), gxz, xs_x, zs, ext_xz, f"{LAB[m]} · xz section (y = {Y})", d=xz(F[m]), keep=gxz > 0, step=(1, 4), qcol=PAL["text"])
    ax2.axhline((ZXY - Z0 + 0.5) * vz, color=PAL["muted"], lw=0.5, ls=(0, (3, 2)))
sbar(ax2, ext_xz[1] - 1.5, ext_xz[2] - 1.5, 10)
cbar(fig, [0.35, 4.5 * V.MM / H, 0.3, 1.6 * V.MM / H], SEQ, Normalize(0, 90), "angle to GT-derived 3D flow (°)", [0, 30, 60, 90])
V.save(fig, "V2-j_xy_xz_angle", dict(PROV, shows="xy (z = 47, plane of nucleus 62's centroid) and xz (y = 126) sections side by side; angle map + arrows (xy: in-plane (dy, dx); xz: (dz, dx)); rows 2D fused, S1; dashed lines = the other section"))

# ---- V2-k per-nucleus z-profile (whole volume)
NB = 10
prof = {m: [] for m in F}
ids = [i for i in np.unique(gt) if i]
from scipy import ndimage as ndi  # noqa: E402
objs = ndi.find_objects(gt)
used = 0
for i in ids:
    sl = objs[i - 1]
    if sl is None or sl[0].stop - sl[0].start < 5:
        continue
    msk = (gt[sl] == i) & valid[sl]
    zz = np.nonzero(msk)[0]
    if zz.size < 50:
        continue
    r = (zz - zz.min()) / max(zz.max() - zz.min(), 1)
    b = np.minimum((r * NB).astype(int), NB - 1)
    used += 1
    for m in F:
        a = A[m][sl][msk]
        prof[m].append([np.median(a[b == k]) if (b == k).any() else np.nan for k in range(NB)])
fig = plt.figure(figsize=(V.W60 * 1.1, 45 * V.MM)); ax = fig.add_axes([0.16, 0.2, 0.8, 0.72])
xc = (np.arange(NB) + 0.5) / NB
for m, ck in (("2D", "C2D"), ("S1", "S1")):
    a = np.array(prof[m])
    med = np.nanmedian(a, 0); q1, q3 = np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0)
    ax.fill_between(xc, q1, q3, color=PAL[ck], alpha=0.18, lw=0)
    ax.plot(xc, med, color=PAL[ck], lw=1.2, marker="o", ms=2.2, label=LAB[m].split(" (")[0])
ax.set_xlabel("relative z within nucleus (0 = first plane, 1 = last plane)"); ax.set_ylabel("median angle to GT 3D flow (°)")
ax.set_xlim(0, 1); ax.set_ylim(0, None); ax.legend(loc="upper center", ncol=2, fontsize=5.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.99, 0.02, f"single-volume illustration · E15_Ab2 · {used} nuclei", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8, color=PAL["muted"])
V.save(fig, "V2-k_zprofile_angle", dict(PROV, shows="per-nucleus z-profile of the median angle (line = median over nuclei, band = IQR), whole volume E15_Ab2, seed 0",
                                       nuclei_used=used, rule="GT nuclei spanning >= 5 native planes with >= 50 valid voxels; 10 equal bins of relative z",
                                       label="single-volume illustration; descriptive, not a FLOWERR result"))
print("ok", used)
