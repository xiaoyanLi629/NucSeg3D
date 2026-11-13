"""V2 detail panels, polish pass for the mechanism figure (V2-h2, V2-i2, V2-k2). Same data and definitions as v2_detail.py
(E15_Ab2, seed 0; FLOWERR GT-derived 3D target; saved thr_eval flows; angle at GT foreground with |t| > 0.5).
Changes (nc-figures / team-lead request):
  V2-h2: crop rule -- the existing section window (y = 126, x 119-375 = 55 um, already narrower than 60 um; z 16-82); GT nuclei whose
         footprint in the section touches the left/right window edge or the top/bottom of the z window are dropped (angle, arrows and
         outlines masked); the z range is then trimmed to the remaining nuclei +-2 planes. Two columns (2D fused | S1), vertical colour
         bar on the right. Angle ramp white -> #F39B7F -> #E64B35 -> #7a1f12 (palette.json colours + a darker end), 0-90 deg.
  V2-i2: nucleus 62 only (angle, arrows and outline masked to GT 62), centred and enlarged; target arrows on the navy-white-red ramp.
  V2-k2: thinner IQR bands (alpha 0.10); the note moved under the x axis.
Text 6-6.5 pt in #1d1d1f; 900 dpi PNG with white background + SVG."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_PAPER = _nsp_os.environ.get('NUCSEG3D_PAPER', _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", "..")))
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(__file__))
import vstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

V.setup()
plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6})
PAL = V.palette(); TXT = "#1d1d1f"
SEQ = LinearSegmentedColormap.from_list("ang", ["#ffffff", "#F39B7F", "#E64B35", "#7a1f12"])
DIVA = V.diverging_arrows()
E = np.load(f"{_NSP_PAPER}/scripts/render3d/export/mouse_embryo.npz")
vox = E["vox"]; vz, vxy = float(vox[0]), float(vox[1]); gt = E["gt"]
T = np.load(f"{V.SCR}/viz/gtflow_E15_Ab2.npz")["t"].astype(np.float32)
F = {m: np.load(f"{V.SCR}/npz/{m}.npz")["dP"].astype(np.float32) for m in ("2D", "S1")}
LAB = {"2D": "2D control (3 views fused)", "S1": "S1"}
tn = np.sqrt((T ** 2).sum(0)); valid = (gt > 0) & (tn > 0.5)
A = {}
for m, d in F.items():
    c = (d * T).sum(0) / (np.sqrt((d ** 2).sum(0)) + 1e-6) / (tn + 1e-6)
    A[m] = np.where(valid, np.degrees(np.arccos(np.clip(c, -1, 1))), np.nan)
Y, X0, XW, Z0, Z1 = 126, 119, 256, 16, 82
sec = lambda a: a[..., Z0:Z1, Y, X0:X0 + XW]  # noqa: E731
g = sec(gt)
PROV = dict(volume="mouse_embryo E15_Ab2", seed="s0", target=f"{_NSP_SCRATCH}/nc/flowerr/gtflow/mouse_embryo__E15_Ab2.npz (FLOWERR target)",
            flows="thr_eval masks npz dP (runs 07992b55 / 81116913)", palette=V.palette_source(), angle_ramp=["#ffffff", "#F39B7F", "#E64B35", "#7a1f12"],
            script=os.path.abspath(__file__), caveat="S1 is trained on this target type; the angle shows agreement with the 3D training target, not segmentation quality")


def save(fig, name, extra):
    fig.savefig(os.path.join(V.OUT, name + ".png"), dpi=900, facecolor="white")
    fig.savefig(os.path.join(V.OUT, name + ".svg"), facecolor="white")
    w, h = fig.get_size_inches()
    json.dump(dict(PROV, asset=name, size_mm=[round(w / V.MM, 1), round(h / V.MM, 1)], dpi=900, **extra),
              open(os.path.join(V.OUT, name + ".provenance.json"), "w"), indent=1, default=float)
    plt.close(fig)


def outlines(ax, lab, xs, ys, keep, lw=0.4):
    for i in keep:
        ax.contour(xs, ys, (lab == i).astype(float), levels=[0.5], colors=[TXT], linewidths=lw)


def quiv(ax, d, xs, ys, keepmask, step=(1, 4), L=None, width=0.0022, cmap=None):
    sv, sh = step
    dv, dh = d[0][::sv, ::sh], d[2][::sv, ::sh]
    n = np.sqrt((d ** 2).sum(0))[::sv, ::sh] + 1e-6
    k = keepmask[::sv, ::sh]
    Xg, Yg = np.meshgrid(xs[::sh], ys[::sv]); L = L or 0.9 * sh * vxy
    kw = dict(angles="xy", scale_units="xy", scale=1, width=width, headwidth=3.0, headlength=3.2, headaxislength=2.8, pivot="mid")
    if cmap is None:
        ax.quiver(Xg[k], Yg[k], (dh / n * L)[k], (dv / n * L)[k], color=TXT, **kw)
    else:
        ax.quiver(Xg[k], Yg[k], (dh / n * L)[k], (dv / n * L)[k], np.clip(dv / n, -1, 1)[k], cmap=cmap, norm=Normalize(-1, 1), **kw)


# ---------------- V2-h2
ids = [int(i) for i in np.unique(g) if i]
edge = set(np.unique(np.r_[g[:, 0], g[:, -1], g[0, :], g[-1, :]]).tolist()) - {0}
keep = [i for i in ids if i not in edge]
km = np.isin(g, keep)
zr = np.nonzero(km.any(1))[0]; za, zb = max(zr.min() - 2, 0), min(zr.max() + 3, g.shape[0])
xr = np.nonzero(km.any(0))[0]; xa, xb = max(xr.min() - 6, 0), min(xr.max() + 7, g.shape[1])
xs = (np.arange(XW) + 0.5) * vxy; zs = (np.arange(Z1 - Z0) + 0.5) * vz
ext = [0, XW * vxy, (Z1 - Z0) * vz, 0]
xlim = (xa * vxy, xb * vxy); zlim = (zb * vz, za * vz)
asp = (zb - za) * vz / ((xb - xa) * vxy)
W = 183 * V.MM; cbw = 14 * V.MM; cw = (W - cbw) / 2; H = cw * asp + 6 * V.MM
fig = plt.figure(figsize=(W, H))
for j, m in enumerate(("2D", "S1")):
    ax = fig.add_axes([(j * cw) / W + 0.003, 0.0, cw / W - 0.006, (cw * asp) / H])
    a = np.where(km, sec(A[m]), np.nan)
    ax.imshow(a, cmap=SEQ, vmin=0, vmax=90, extent=ext, interpolation="nearest", aspect="auto")
    outlines(ax, g, xs, zs, keep, 0.4)
    quiv(ax, sec(F[m]), xs, zs, km, step=(1, 4))
    ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(LAB[m], fontsize=6.5, color=TXT, pad=2)
ax.plot([xlim[1] - 11.5, xlim[1] - 1.5], [zlim[0] - 1.2] * 2, color=TXT, lw=1.0, solid_capstyle="butt")
ax.text(xlim[1] - 6.5, zlim[0] - 1.8, "10 µm", ha="center", va="bottom", fontsize=6, color=TXT)
cax = fig.add_axes([(2 * cw + 3 * V.MM) / W, 0.12, 2.0 * V.MM / W, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 90), SEQ), cax=cax, orientation="vertical", ticks=[0, 30, 60, 90])
cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, length=1.5, pad=1, colors=TXT)
cb.set_label("angle to GT-derived\n3D flow (°)", fontsize=6, color=TXT, labelpad=2)
save(fig, "V2-h2_angle_quiver_xz", dict(shows="xz section (y = 126), angle to the GT-derived 3D flow with dark flow arrows (every 4 px, every plane), 2D fused | S1",
                                       crop_rule="section window x 119-375 (55 um), z 16-82; GT nuclei touching the window edge in this section dropped; z trimmed to kept nuclei +-2 planes, x to kept nuclei +-6 px",
                                       nuclei_kept=keep, nuclei_dropped_edge=sorted(edge), crop_px=dict(z=[Z0 + za, Z0 + zb], x=[X0 + xa, X0 + xb])))

# ---------------- V2-i2 nucleus 62 only
m62 = g == 62
zr_, xr_ = np.nonzero(m62)
zc, xc = (zr_.min() + zr_.max() + 1) / 2 * vz, (xr_.min() + xr_.max() + 1) / 2 * vxy
half_x = (xr_.max() - xr_.min() + 1) * vxy / 2 + 1.0; half_z = (zr_.max() - zr_.min() + 1) * vz / 2 + 1.0
asp = half_z / half_x
W = 150 * V.MM; cbw = 16 * V.MM; cw = (W - cbw) / 3; H = cw * asp + 6 * V.MM
fig = plt.figure(figsize=(W, H))
tg = np.where(m62[None], sec(T), 0)
for j, (m, t) in enumerate((("GT", "GT-derived 3D target"), ("2D", LAB["2D"]), ("S1", LAB["S1"]))):
    ax = fig.add_axes([(j * cw) / W + 0.004, 0.0, cw / W - 0.008, (cw * asp) / H])
    if m == "GT":
        rgba = np.zeros((*m62.shape, 4)); rgba[m62] = (0.94, 0.94, 0.95, 1)
        ax.imshow(rgba, extent=ext, interpolation="nearest", aspect="auto")
        quiv(ax, sec(T), xs, zs, m62 & sec(valid), step=(1, 2), L=0.8, width=0.006, cmap=DIVA)
    else:
        ax.imshow(np.where(m62, sec(A[m]), np.nan), cmap=SEQ, vmin=0, vmax=90, extent=ext, interpolation="nearest", aspect="auto")
        quiv(ax, sec(F[m]), xs, zs, m62, step=(1, 2), L=0.8, width=0.006)
    outlines(ax, g, xs, zs, [62], 0.6)
    ax.set_xlim(xc - half_x, xc + half_x); ax.set_ylim(zc + half_z, zc - half_z); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(t, fontsize=6.5, color=TXT, pad=2)
ax.plot([xc + half_x - 3.3, xc + half_x - 0.3], [zc + half_z - 0.25] * 2, color=TXT, lw=1.0, solid_capstyle="butt")
ax.text(xc + half_x - 1.8, zc + half_z - 0.45, "3 µm", ha="center", va="bottom", fontsize=6, color=TXT)
cax = fig.add_axes([(3 * cw + 3 * V.MM) / W, 0.12, 2.0 * V.MM / W, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 90), SEQ), cax=cax, orientation="vertical", ticks=[0, 30, 60, 90])
cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, length=1.5, pad=1, colors=TXT); cb.set_label("angle (°)", fontsize=6, color=TXT, labelpad=2)
save(fig, "V2-i2_nucleus62", dict(shows="GT nucleus 62 only (xz section y = 126): GT-derived 3D target arrows (colour u_z, navy up / red down) | 2D fused angle + arrows | S1 angle + arrows; arrows every 2 px, every plane",
                                  mask="angle, arrows and outline restricted to GT 62"))

# ---------------- V2-k2
NB = 10; prof = {m: [] for m in F}; used = 0
objs = ndi.find_objects(gt)
for i in [int(x) for x in np.unique(gt) if x]:
    sl = objs[i - 1]
    if sl is None or sl[0].stop - sl[0].start < 5:
        continue
    msk = (gt[sl] == i) & valid[sl]; zz = np.nonzero(msk)[0]
    if zz.size < 50:
        continue
    r = (zz - zz.min()) / max(zz.max() - zz.min(), 1); b = np.minimum((r * NB).astype(int), NB - 1); used += 1
    for m in F:
        a = A[m][sl][msk]
        prof[m].append([np.median(a[b == k]) if (b == k).any() else np.nan for k in range(NB)])
fig = plt.figure(figsize=(66 * V.MM, 50 * V.MM)); ax = fig.add_axes([0.16, 0.30, 0.80, 0.64])
xcn = (np.arange(NB) + 0.5) / NB
for m, ck in (("2D", "C2D"), ("S1", "S1")):
    a = np.array(prof[m]); med = np.nanmedian(a, 0); q1, q3 = np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0)
    ax.fill_between(xcn, q1, q3, color=PAL[ck], alpha=0.10, lw=0)
    ax.plot(xcn, med, color=PAL[ck], lw=1.1, marker="o", ms=2.0, label="2D control" if m == "2D" else "S1")
ax.set_xlabel("relative z within nucleus (0 = first plane, 1 = last plane)", color=TXT)
ax.set_ylabel("median angle to GT 3D flow (°)", color=TXT)
ax.set_xlim(0, 1); ax.set_ylim(0, None); ax.legend(loc="upper center", ncol=2, fontsize=6, frameon=False)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.tick_params(colors=TXT)
fig.text(0.56, 0.035, f"single volume, descriptive · E15_Ab2 · {used} nuclei", ha="center", va="bottom", fontsize=6, color="#4d4d52")
save(fig, "V2-k2_zprofile", dict(shows="per-nucleus z-profile of the median angle (line = median over nuclei, band = IQR, alpha 0.10)", nuclei_used=used,
                                 rule="GT nuclei spanning >= 5 native planes with >= 50 valid voxels; 10 equal bins of relative z", label="single volume, descriptive"))
print("ok", keep, sorted(edge), used)
