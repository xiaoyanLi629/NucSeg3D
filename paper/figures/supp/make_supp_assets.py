"""Raster assets for the R9 supplementary figures (no data, selection or number changes; display only).
Run with paper/scripts/render3d/.venv/bin/python (pyvista for the F5 re-render).

tiles/T_<ds>_<k>.png    Tgrid renders: grid_cache_viewer (R8, viewer palette) shot_set images, de-haloed, on the #EDECE9 pane
tiles/MM_<ds>_<k>.png   MM grid renders: grid_cache_viewer mm_* images, autocropped, de-haloed, on the pane
tiles/F5_<m>_k<k>.png   anisotropy renders: cand_f5.render with the viewer palette (same as R8-F5-a), on the pane
tiles/F5x_<m>_k<k>.png  anisotropy xz strips: cand_f5.xz (GT white, model outline in its colour)
tiles/O_<ds>_<view>_<m>.png  ortho tiles: same crops/outline rule as ortho_plot.tile, native-resolution tile (one-voxel outlines, as before),
                        nearest-upsampled 4x with z x round(4 z/xy) for physical aspect at physical aspect; GT white, model in its palette colour, overlap = mean
tiles/M_<k>.png + mech.json  FigS_mech a (nucleus 62 tiles without text) and b (z-profile numbers), code of v2_detail2/3.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_FIG_SCRATCH = _nsp_os.environ.get('NUCSEG3D_FIG_SCRATCH', '/path/to/figure_scratch')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
R3 = os.path.join(PAPER, "scripts", "render3d")
sys.path.insert(0, R3)
OUT = os.path.join(HERE, "tiles"); os.makedirs(OUT, exist_ok=True)
PVW = json.load(open(os.path.join(PAPER, "figures", "palette_viewer.json")))
PJ = json.load(open(os.path.join(PAPER, "figures", "palette.json")))
PANE = PVW["pane"]
LOG = {}

import rstyle as R  # noqa: E402

R.ERR.update(PVW["errors"]); R.QUAL[:] = PVW["qualitative_instances"]; R.LIGHT_BG = (PANE, PANE)


def dehalo(im):
    a = np.asarray(im.convert("RGBA")).copy(); a[..., 3][a[..., 3] < PVW["dehalo_alpha_below"]] = 0
    return Image.fromarray(a)


def which(arg):
    return len(sys.argv) == 1 or arg in sys.argv


# ---------------------------------------------------------------- 3D render grids (cached R8 renders)
CV = os.path.join(R3, "out", "grid_cache_viewer")
if which("T"):
    for ds in ("nis3d", "tapenade", "nisnet_zf", "mouse_embryo", "blastospim"):
        for k in ("gt", "2D", "S1"):
            im = dehalo(Image.open(os.path.join(CV, f"{ds}_gt-2D-S1_l_{k}.png")))
            R.tile(im, False, size=(800, 900), pad=0.03).save(os.path.join(OUT, f"T_{ds}_{k}.png"))
    LOG["T"] = "grid_cache_viewer/<ds>_gt-2D-S1_l_<k>.png -> dehalo (alpha < 40) -> rstyle.tile 800x900 pad 0.03 on #EDECE9 (as R8-T-grid-wide)"
if which("MM"):
    for ds in ("nis3d", "tapenade", "nisnet_zf", "mouse_embryo", "blastospim"):
        for k in ("gt", "base", "stardist3d", "useg_2d", "nnunet", "2D", "S1"):
            fn = os.path.join(CV, f"mm_{ds}_{k}_l.png")
            if os.path.exists(fn):
                R.tile(R.autocrop(dehalo(Image.open(fn))), False, size=(800, 800), pad=0.04).save(os.path.join(OUT, f"MM_{ds}_{k}.png"))
    LOG["MM"] = "grid_cache_viewer/mm_<ds>_<k>_l.png -> dehalo -> autocrop -> rstyle.tile 800x800 pad 0.04 on #EDECE9 (as R8-MM-grid)"

# ---------------------------------------------------------------- F5 anisotropy (re-render with the viewer palette, as supp_viewer.py)
if which("F5"):
    import cand_f5 as C5
    for m, col in (("2D", R.D2), ("S1", R.S1)):
        for k in (1, 2, 4, 8):
            Z = np.load(os.path.join(R3, "export", f"f5_{m}_k{k}.npz"))
            gt, pm, img, ratio = Z["gt"], Z["pred"], Z["img"], float(Z["ratio"])
            R.tile(dehalo(R.autocrop(C5.render(gt, pm, ratio, False))), False, size=(1000, 760), pad=0.03).save(os.path.join(OUT, f"F5_{m}_k{k}.png"))
            x = C5.xz(img, gt, pm, col, ratio)
            Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, f"F5x_{m}_k{k}.png"))
    LOG["F5"] = "cand_f5.render / cand_f5.xz on export/f5_<m>_k<k>.npz with rstyle.ERR = palette_viewer errors (same as R8-F5-a)"

# ---------------------------------------------------------------- ortho tiles
if which("O"):
    from matplotlib.colors import to_rgb
    from skimage.segmentation import find_boundaries
    OD = os.path.join(HERE, "data_ortho")
    SEL = json.load(open(os.path.join(OD, "selection.json")))
    EDGE = {"S1": PJ["models"]["S1"], "2D": PJ["models"]["2D"], "base": PJ["models"]["base"]}
    GTC = PJ.get("gt_outline_on_image", "#ffffff")
    U = 4
    dss = SEL["pool_sorted_by_ratio"] + ["mesospim"]
    shapes = {}
    for ds in dss:
        Cz = np.load(os.path.join(OD, "crops", f"{ds}.npz")); ratio = float(Cz["ratio"])
        for view in ("xy", "xz", "yz"):
            uz = U if view == "xy" else max(int(round(U * ratio)), 1)
            up = lambda a: np.repeat(np.repeat(a, uz, 0), U, 1)  # noqa: E731
            img0, gt0 = Cz[f"img_{view}"], Cz[f"gt_{view}"]
            bg = find_boundaries(gt0, mode="inner")
            for m in ("S1", "2D", "base"):
                bp = find_boundaries(Cz[f"{m}_{view}"], mode="inner")
                rgb = np.repeat(img0[..., None], 3, axis=2) * 0.85                        # exactly ortho_plot.tile at native resolution
                rgb[bg] = to_rgb(GTC); rgb[bp] = to_rgb(EDGE[m])
                rgb[bg & bp] = np.array(to_rgb(EDGE[m])) * 0.5 + np.array(to_rgb(GTC)) * 0.5
                Image.fromarray((np.clip(up(rgb), 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, f"O_{ds}_{view}_{m}.png"))
            img = up(img0)
            shapes[f"{ds}_{view}"] = list(img.shape)
    LOG["O"] = dict(source=f"figures/supp/data_ortho (copied read-only from server {_NSP_SCRATCH}/nc/figs/ortho)", upsample=U,
                    rule="ortho_plot.tile: image x 0.85, GT inner boundary white, model inner boundary in palette colour, overlap = mean; "
                         "identical tile at native resolution, then 4x nearest upsampling with z repeated round(4 z/xy) times for physical aspect", shapes=shapes)

# ---------------------------------------------------------------- FigS_mech a, b (code and data of assets_viz/scripts/v2_detail2/3.py)
if which("M"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from scipy import ndimage as ndi
    SCR = f"{_NSP_FIG_SCRATCH}"
    E = np.load(os.path.join(R3, "export", "mouse_embryo.npz"))
    vox = E["vox"]; vz, vxy = float(vox[0]), float(vox[1]); gt = E["gt"]
    T = np.load(f"{SCR}/viz/gtflow_E15_Ab2.npz")["t"].astype(np.float32)
    F = {m: np.load(f"{SCR}/npz/{m}.npz")["dP"].astype(np.float32) for m in ("2D", "S1")}
    tn = np.sqrt((T ** 2).sum(0)); valid = (gt > 0) & (tn > 0.5)
    A = {}
    for m, d in F.items():
        c = (d * T).sum(0) / (np.sqrt((d ** 2).sum(0)) + 1e-6) / (tn + 1e-6)
        A[m] = np.where(valid, np.degrees(np.arccos(np.clip(c, -1, 1))), np.nan)
    Y, X0, XW, Z0, Z1 = 126, 119, 256, 16, 82
    sec = lambda a: a[..., Z0:Z1, Y, X0:X0 + XW]  # noqa: E731
    g = sec(gt)
    SEQ = LinearSegmentedColormap.from_list("ang", ["#ffffff", "#F39B7F", "#E64B35", "#7a1f12"])
    d_ = PJ["diverging"]; DIVA = LinearSegmentedColormap.from_list("pal_div_a", [d_[0], "#b4b6bd", d_[-1]])
    TXT = "#1d1d1f"
    xs = (np.arange(XW) + 0.5) * vxy; zs = (np.arange(Z1 - Z0) + 0.5) * vz
    ext = [0, XW * vxy, (Z1 - Z0) * vz, 0]

    def quiv(ax, d, keepmask, step=(1, 2), L=0.8, width=0.008, cmap=None):
        sv, sh = step
        dv, dh = d[0][::sv, ::sh], d[2][::sv, ::sh]
        n = np.sqrt((d ** 2).sum(0))[::sv, ::sh] + 1e-6
        k = keepmask[::sv, ::sh]
        Xg, Yg = np.meshgrid(xs[::sh], zs[::sv])
        kw = dict(angles="xy", scale_units="xy", scale=1, width=width, headwidth=3.0, headlength=3.2, headaxislength=2.8, pivot="mid")
        if cmap is None:
            ax.quiver(Xg[k], Yg[k], (dh / n * L)[k], (dv / n * L)[k], color=TXT, **kw)
        else:
            ax.quiver(Xg[k], Yg[k], (dh / n * L)[k], (dv / n * L)[k], np.clip(dv / n, -1, 1)[k], cmap=cmap, norm=Normalize(-1, 1), **kw)

    m62 = g == 62
    zr_, xr_ = np.nonzero(m62)
    zc, xc = (zr_.min() + zr_.max() + 1) / 2 * vz, (xr_.min() + xr_.max() + 1) / 2 * vxy
    half_x = (xr_.max() - xr_.min() + 1) * vxy / 2 + 1.0; half_z = (zr_.max() - zr_.min() + 1) * vz / 2 + 1.0
    TWmm = 34.0; THmm = TWmm * half_z / half_x
    for m in ("GT", "2D", "S1"):
        fig = plt.figure(figsize=(TWmm / 25.4, THmm / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
        if m == "GT":
            rgba = np.zeros((*m62.shape, 4)); rgba[m62] = (0.94, 0.94, 0.95, 1)
            ax.imshow(rgba, extent=ext, interpolation="nearest", aspect="auto")
            quiv(ax, sec(T), m62 & sec(valid), cmap=DIVA)
        else:
            ax.imshow(np.where(m62, sec(A[m]), np.nan), cmap=SEQ, vmin=0, vmax=90, extent=ext, interpolation="nearest", aspect="auto")
            quiv(ax, sec(F[m]), m62)
        ax.contour(xs, zs, m62.astype(float), levels=[0.5], colors=[TXT], linewidths=0.6)
        ax.set_xlim(xc - half_x, xc + half_x); ax.set_ylim(zc + half_z, zc - half_z); ax.axis("off")
        fig.savefig(os.path.join(OUT, f"M_{m}.png"), dpi=900, transparent=True); plt.close(fig)
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
    zp = {}
    for m in F:
        a = np.array(prof[m])
        zp[m] = dict(med=np.nanmedian(a, 0).tolist(), q1=np.nanpercentile(a, 25, 0).tolist(), q3=np.nanpercentile(a, 75, 0).tolist())
    json.dump(dict(tile_mm=[TWmm, THmm], tile_um=[2 * half_x, 2 * half_z], zprofile=zp, nuclei_used=used,
                   source="code of figures/assets_viz/scripts/v2_detail2.py (V2-k2) and v2_detail3.py (V2-i3), same data paths"),
              open(os.path.join(OUT, "mech.json"), "w"), indent=1)
    LOG["M"] = dict(nuclei_used=used)

json.dump(LOG, open(os.path.join(OUT, "assets_log_" + "_".join(sys.argv[1:] or ["all"]) + ".json"), "w"), indent=1)
print(json.dumps({k: (v if isinstance(v, str) else "ok") for k, v in LOG.items()}, indent=1))
