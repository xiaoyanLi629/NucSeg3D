#!/usr/bin/env python
"""Where do ground-truth nucleus boundaries sit relative to image intensity?  (CPU only, read-only on data)

For each dataset: up to 4 non-test 3D volumes (mesospim: c1-c5, never `visual`; gonuclear: *_clear only),
each cropped in xy to the <=CROP x CROP window with the most GT foreground (full z kept), robust per-volume
normalisation (1st-99.8th percentile of the crop, no clipping), then up to MAX_NUC randomly sampled nuclei
(pooled over the volumes; nuclei touching the crop border or an NIS3D "ignore" region are excluded).

Per nucleus (all intensity ratios below are invariant to the affine normalisation):
  peak      = 95th percentile inside the GT mask (also p99 and max are stored)
  bg        = median of the background voxels (label 0, not ignore) in an ellipsoidal shell 2-4 nucleus radii
              from the centroid (radii = half the GT bbox extent per axis); noise = 1.4826 * MAD of that shell
  r_inner   = (mean I on inner boundary voxels - bg) / (peak - bg); inner boundary = GT voxels with a 6-neighbour
              in background; r_outer likewise for background voxels with a 6-neighbour in the GT
  r_face_*  = level at the GT surface itself: mean over boundary faces (GT voxel | background voxel, 6-conn.) of the
              average of the two voxels, split into faces normal to z and faces normal to y/x
  1-D lines through the centroid voxel along z, y, x:
     edge_*   = level at the two ends of the GT run (face midpoint), only ends that border background
     ratio_*  = half-max extent of the intensity profile / GT run length; half level = bg + 0.5 (peak - bg);
                the supra-half-max segment is taken from the outermost supra-threshold samples inside the GT run
                and extended outward while contiguous (tolerates internal texture, does not jump gaps); search
                is limited to +-1 GT run length and stops at other nuclei / ignore voxels. Each side is scored as
                (GT-run centre -> crossing) / (half run length); censored sides are dropped (counted in cens_*)
     off_*    = (blob centre - GT run centre) / GT run length (only when neither side is censored)
     grad_*   = normalisation-free: (GT-run centre -> steepest outward intensity drop) / (half run length)
  3-D iso-level blobs in the GT bbox padded by 1x its extent, restricted to the nucleus' own territory
  (background voxels whose nearest labelled voxel - anisotropic EDT - belongs to this nucleus):
     blob(l)  = hole-filled union of connected components of {I >= bg + l (peak - bg)} that overlap the GT
     per level l: volume ratio vr, IoU, RMS-extent ratios along z (zr) and in xy (xyr), bbox-span ratios (zsp, xysp)
     lvl_vol / lvl_z / lvl_xy = level at which blob volume / RMS z extent / RMS xy extent equals the GT's
Optional control (--down F): block-mean the image over FxF pixels in xy (partial-volume model), downsample the
labels by majority (>= half of the block) or `any` rule, then run the identical measurement.
Optional reverse control (--zdown F): the same block downsampling along z only (used to coarsen mesospim's z).

Usage: python gt_conv.py --out DIR --ds mesospim nucverse3d ... [--down ds=F ...] [--zdown ds=F ...]
       [--gt_rule majority|any]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import tifffile
from scipy import ndimage as ndi

STD = f"{_NSP_ROOT}/data/std"
LEVELS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ----------------------------------------------------------------------------------------------- volume selection
def select_volumes(ds, nvol=4, min_n=10):
    rows = [r for r in csv.DictReader(open(f"{STD}/manifest.csv"))
            if r["ds"] == ds and r["dim"] == "3d" and r["split"] == "train"]       # val volumes live in split=train
    if ds == "mesospim":                                                             # focus dataset: all of c1-c5
        return [r for r in rows if r["id"] in ("c1", "c2", "c3", "c4", "c5")]
    if ds == "gonuclear":                                                            # clear/noisy share labels
        rows = [r for r in rows if r["id"].endswith("_clear")]
    big = [r for r in rows if int(r["n_nuclei"]) >= min_n]
    if len(big) >= nvol:
        rows = big
    if len(rows) <= nvol:
        return rows
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, nvol)).astype(int))   # evenly spaced in manifest order
    return [rows[i] for i in idx]


def best_window(fg2d, wy, wx, stride=8):
    """Top-left corner of the wy x wx window with the largest sum of fg2d."""
    Y, X = fg2d.shape
    wy, wx = min(wy, Y), min(wx, X)
    if wy == Y and wx == X:
        return 0, 0
    S = np.zeros((Y + 1, X + 1), np.int64)
    S[1:, 1:] = fg2d.astype(np.int64).cumsum(0).cumsum(1)
    ys = np.unique(np.r_[np.arange(0, Y - wy + 1, stride), Y - wy])
    xs = np.unique(np.r_[np.arange(0, X - wx + 1, stride), X - wx])
    tot = (S[ys[:, None] + wy, xs[None, :] + wx] - S[ys[:, None], xs[None, :] + wx]
           - S[ys[:, None] + wy, xs[None, :]] + S[ys[:, None], xs[None, :]])
    iy, ix = np.unravel_index(int(np.argmax(tot)), tot.shape)
    return int(ys[iy]), int(xs[ix])


def downsample_labels(lab, f, rule, zchunk=8):
    """xy block downsampling of an instance label volume. Each FxF block gets the more frequent of its smallest and
    largest non-zero label (blocks with >2 labels are rare), kept if its count >= ceil(F*F/2) (majority) or >= 1 (any).
    Processed in z chunks to bound memory."""
    Z, Y, X = lab.shape
    need = (f * f + 1) // 2 if rule == "majority" else 1
    out = np.zeros((Z, Y // f, X // f), lab.dtype)
    big = np.iinfo(lab.dtype).max
    for z0 in range(0, Z, zchunk):
        sub = lab[z0:z0 + zchunk]
        b = sub.reshape(sub.shape[0], Y // f, f, X // f, f).transpose(0, 1, 3, 2, 4).reshape(sub.shape[0], Y // f, X // f, f * f)
        lmax = b.max(-1)
        lmin = np.where(b > 0, b, big).min(-1)
        lmin[lmin == big] = 0
        cmax = (b == lmax[..., None]).sum(-1)
        cmax[lmax == 0] = 0
        cmin = (b == lmin[..., None]).sum(-1)
        cmin[lmin == 0] = 0
        o = np.where(cmax >= cmin, lmax, lmin)
        o[np.maximum(cmax, cmin) < need] = 0
        out[z0:z0 + zchunk] = o
    return out


def zdownsample(img, lab, ign, f, rule):
    """z block downsampling (reverse control): block-mean of the image over F slices, labels by the same
    smallest/largest-label majority (>= ceil(F/2)) or `any` rule as in xy."""
    Z = img.shape[0] - img.shape[0] % f
    img = img[:Z].reshape(Z // f, f, *img.shape[1:]).mean(1, dtype=np.float64).astype(np.float32)
    b = np.moveaxis(lab[:Z].reshape(Z // f, f, *lab.shape[1:]), 1, -1)
    big = np.iinfo(lab.dtype).max
    lmax = b.max(-1)
    lmin = np.where(b > 0, b, big).min(-1)
    lmin[lmin == big] = 0
    cmax = (b == lmax[..., None]).sum(-1)
    cmax[lmax == 0] = 0
    cmin = (b == lmin[..., None]).sum(-1)
    cmin[lmin == 0] = 0
    out = np.where(cmax >= cmin, lmax, lmin)
    out[np.maximum(cmax, cmin) < ((f + 1) // 2 if rule == "majority" else 1)] = 0
    if ign is not None:
        ign = ign[:Z].reshape(Z // f, f, *ign.shape[1:]).any(1)
    return img, out, ign


def load_volume(r, crop, f=1, gt_rule="majority", zf=1):
    lab = tifffile.imread(os.path.join(STD, r["mask"]))
    if lab.dtype.kind not in "ui":
        lab = lab.astype(np.int32)
    Z, Y, X = lab.shape
    wy, wx = min(Y, crop * f), min(X, crop * f)
    wy -= wy % f
    wx -= wx % f
    y0, x0 = best_window((lab > 0).sum(0, dtype=np.int32), wy, wx)
    lab = np.ascontiguousarray(lab[:, y0:y0 + wy, x0:x0 + wx])
    img = tifffile.imread(os.path.join(STD, r["image"]))
    img = img[:, y0:y0 + wy, x0:x0 + wx].astype(np.float32)
    ip = os.path.join(STD, r["ds"], "ignore", r["id"] + ".tif")
    ign = tifffile.imread(ip).astype(bool)[:, y0:y0 + wy, x0:x0 + wx] if os.path.exists(ip) else None
    vox = (float(r["voxel_z"]), float(r["voxel_y"]), float(r["voxel_x"])) if r["voxel_z"] else (1.0, 1.0, 1.0)
    if f > 1:
        img = img.reshape(Z, wy // f, f, wx // f, f).mean((2, 4), dtype=np.float64).astype(np.float32)
        lab = downsample_labels(lab, f, gt_rule)
        if ign is not None:
            ign = ign.reshape(Z, wy // f, f, wx // f, f).any((2, 4))
        vox = (vox[0], vox[1] * f, vox[2] * f)
    if zf > 1:
        img, lab, ign = zdownsample(img, lab, ign, zf, gt_rule)
        vox = (vox[0] * zf, vox[1], vox[2])
    lo, hi = np.percentile(img, [1, 99.8])
    img = (img - lo) / max(float(hi - lo), 1e-6)                               # robust normalisation, no clipping
    bgmask = (lab == 0) if ign is None else ((lab == 0) & ~ign)
    info = dict(id=r["id"], shape_full=[Z, Y, X], y0=y0, x0=x0, wy=wy, wx=wx, down=f, zdown=zf, p1=float(lo), p998=float(hi),
                vox=list(vox), n_labels_crop=int(len(np.unique(lab)) - 1), fg_frac=float((lab > 0).mean()),
                bg_median_crop=float(np.median(img[bgmask])) if bgmask.any() else float("nan"))
    return img, lab, ign, vox, info


def eligible(lab, ign, min_vox=8):
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        if any(s.start == 0 or s.stop == n for s, n in zip(sl, lab.shape)):      # touches the crop border
            continue
        if any(s.stop - s.start < 2 for s in sl):
            continue
        if ign is not None:
            sl1 = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1)) for s, n in zip(sl, lab.shape))
            if ign[sl1].any():
                continue
        if int((lab[sl] == i).sum()) < min_vox:
            continue
        out.append((i, sl))
    return out


# ----------------------------------------------------------------------------------------------- per-nucleus
def line_metrics(p, cls, k, bg, c):
    """p: intensity along a line, cls: 1 GT, 0 background, -1 other nucleus, -2 ignore; k: index of the start voxel."""
    s = k
    while s > 0 and cls[s - 1] == 1:
        s -= 1
    e = k
    while e < len(p) - 1 and cls[e + 1] == 1:
        e += 1
    N = e - s + 1
    ends = []
    if s - 1 >= 0 and cls[s - 1] == 0:
        ends.append(0.5 * (p[s] + p[s - 1]))
    if e + 1 < len(p) and cls[e + 1] == 0:
        ends.append(0.5 * (p[e] + p[e + 1]))
    edge = (np.mean(ends) - bg) / c if ends else np.nan
    # search window: +-N beyond the run, through GT/background only
    wl = s
    while wl - 1 >= max(0, s - N) and cls[wl - 1] in (0, 1):
        wl -= 1
    wr = e
    while wr + 1 <= min(len(p) - 1, e + N) and cls[wr + 1] in (0, 1):
        wr += 1
    return dict(run=N, edge=edge), (s, e, wl, wr)


def fwhm(p, s, e, wl, wr, L):
    """Half-level extent of the profile relative to the GT run [s, e]. Each side is scored separately as
    (distance GT-run centre -> level crossing) / (half GT run length); a side is censored (dropped) when the profile is
    still >= L where the search window ends (other nucleus, ignore voxel, line end, or +-1 run length).
    Returns (mean ratio over uncensored sides, centre offset / run length [both sides needed], #censored sides,
    flag_below: 1 if the profile never reaches L inside the GT run)."""
    inside = np.nonzero(p[s:e + 1] >= L)[0]
    if inside.size == 0:
        return np.nan, np.nan, 0, 1
    a, b = s + int(inside[0]), s + int(inside[-1])
    j = a
    while j - 1 >= wl and p[j - 1] >= L:
        j -= 1
    cl = j - 1 < wl
    xl = np.nan if cl else j - (p[j] - L) / (p[j] - p[j - 1])
    j = b
    while j + 1 <= wr and p[j + 1] >= L:
        j += 1
    cr = j + 1 > wr
    xr = np.nan if cr else j + (p[j] - L) / (p[j] - p[j + 1])
    N, cg = e - s + 1, 0.5 * (s + e)
    sides = [(cg - xl) / (0.5 * N), (xr - cg) / (0.5 * N)]
    ratio = float(np.nanmean(sides)) if not (cl and cr) else np.nan
    off = (0.5 * (xl + xr) - cg) / N if not (cl or cr) else np.nan
    return ratio, off, int(cl) + int(cr), 0


def grad_edge(p, s, e, wl, wr):
    """Normalisation-free edge position: on each side of the profile maximum inside the GT run, the steepest
    outward drop of the outer-hull envelope (running max taken from the window end inward, which fills internal
    dips such as nucleoli), located with parabolic refinement. Returns (mean over sides of
    (GT-run centre -> steepest drop) / (half run length), #sides whose steepest drop is the last step of the window)."""
    m = s + int(np.argmax(p[s:e + 1]))
    N, cg = e - s + 1, 0.5 * (s + e)
    vals, last = [], 0
    for side in (1, -1):
        seg = p[m:wr + 1] if side == 1 else p[wl:m + 1][::-1]
        if seg.size < 3:
            continue
        env = np.maximum.accumulate(seg[::-1])[::-1]
        d = env[:-1] - env[1:]
        k = int(np.argmax(d))
        if d[k] <= 0:
            continue
        delta = 0.0
        if 0 < k < d.size - 1:
            den = d[k - 1] - 2 * d[k] + d[k + 1]
            if den != 0:
                delta = float(np.clip(0.5 * (d[k - 1] - d[k + 1]) / den, -0.5, 0.5))
        pos = m + side * (k + 0.5 + delta)
        last += int(k == d.size - 1)
        vals.append(abs(pos - cg) / (0.5 * N))
    return (float(np.mean(vals)) if vals else np.nan), last


def fill_holes(m):
    """Hole filling by labelling the complement (faster than ndi.binary_fill_holes on large boxes)."""
    lc, n = ndi.label(~m)
    if n == 0:
        return m
    border = np.unique(np.concatenate([lc[0].ravel(), lc[-1].ravel(), lc[:, 0].ravel(), lc[:, -1].ravel(),
                                       lc[:, :, 0].ravel(), lc[:, :, -1].ravel()]))
    keep = np.zeros(n + 1, bool)
    keep[border] = True
    keep[0] = True
    return m | ~keep[lc]


def matched_level(levels, ratios):
    """Level at which a non-increasing blob/GT ratio crosses 1 (linear interpolation). Returns (level, flag):
    flag -1: ratio already <= 1 at the lowest level (true level is lower), +1: still > 1 at the highest level."""
    r = np.asarray(ratios, float)
    if r[0] <= 1:
        return levels[0], -1
    k = np.nonzero(r <= 1)[0]
    if k.size == 0:
        return levels[-1], 1
    k = int(k[0])
    l0, l1, r0, r1 = levels[k - 1], levels[k], r[k - 1], r[k]
    return l0 + (r0 - 1) / (r0 - r1) * (l1 - l0), 0


def per_nucleus(i, sl, img, lab, ign, vox, bg_crop):
    shp = lab.shape
    ext = np.array([s.stop - s.start for s in sl])
    padb = np.ceil(1.5 * ext).astype(int) + 2
    B = tuple(slice(max(0, s.start - p), min(n, s.stop + p)) for s, p, n in zip(sl, padb, shp))
    I = img[B]
    Lb = lab[B]
    G = Lb == i
    O = (Lb > 0) & ~G
    IG = ign[B] if ign is not None else np.zeros(G.shape, bool)
    BG = (Lb == 0) & ~IG
    gb = tuple(slice(s.start - b.start, s.stop - b.start) for s, b in zip(sl, B))   # GT bbox in box coords
    idx = np.nonzero(G)
    cen = np.array([a.mean() for a in idx])
    rz, rxy = ext[0] / 2.0, (ext[1] + ext[2]) / 4.0
    d2 = (((np.arange(G.shape[0]) - cen[0]) / rz) ** 2)[:, None, None] \
        + (((np.arange(G.shape[1]) - cen[1]) / rxy) ** 2)[None, :, None] \
        + (((np.arange(G.shape[2]) - cen[2]) / rxy) ** 2)[None, None, :]
    shell = BG & (d2 >= 4) & (d2 <= 16)
    vin = I[G]
    rec = dict(label=int(i), n_vox=int(vin.size), ext_z=int(ext[0]), ext_y=int(ext[1]), ext_x=int(ext[2]),
               cz=float(cen[0] + B[0].start), cy=float(cen[1] + B[1].start), cx=float(cen[2] + B[2].start))
    rec["d_eq_xy"] = float(2 * np.sqrt(G.sum((1, 2)).max() / np.pi))
    p95, p99 = np.percentile(vin, [95, 99])
    rec.update(peak95=float(p95), peak99=float(p99), peakmax=float(vin.max()), med_in=float(np.median(vin)))
    sv = I[shell]
    rec["bg_n"] = int(sv.size)
    if sv.size >= 20:
        bg = float(np.median(sv))
        rec.update(bg=bg, bg_fallback=0, noise=float(1.4826 * np.median(np.abs(sv - bg))))
    else:
        bg = bg_crop
        rec.update(bg=bg, bg_fallback=1, noise=np.nan)
    c = rec["peak95"] - bg
    rec["contrast"] = c
    rec["snr"] = c / rec["noise"] if rec["noise"] and np.isfinite(rec["noise"]) and rec["noise"] > 0 else np.nan
    if c <= 0:
        rec["valid"] = 0
        return rec
    rec["valid"] = 1
    # ---- boundary voxels and faces (in the GT bbox padded by 1)
    S1 = tuple(slice(max(0, g.start - 1), min(n, g.stop + 1)) for g, n in zip(gb, G.shape))
    g1, b1, o1, i1 = G[S1], BG[S1], O[S1], I[S1]
    inner = np.zeros(g1.shape, bool)
    outer = np.zeros(g1.shape, bool)
    fv = {0: [], 1: [], 2: []}
    touch, nfaces = 0, 0
    for ax in range(3):
        sa = [slice(None)] * 3
        sb = [slice(None)] * 3
        sa[ax] = slice(0, -1)
        sb[ax] = slice(1, None)
        sa, sb = tuple(sa), tuple(sb)
        m1 = g1[sa] & b1[sb]          # GT at a, background at b
        m2 = g1[sb] & b1[sa]
        fv[ax].append(0.5 * (i1[sa][m1] + i1[sb][m1]))
        fv[ax].append(0.5 * (i1[sb][m2] + i1[sa][m2]))
        inner[sa] |= m1
        inner[sb] |= m2
        outer[sb] |= m1
        outer[sa] |= m2
        t = int((g1[sa] & o1[sb]).sum() + (g1[sb] & o1[sa]).sum())
        touch += t
        nfaces += t + int(m1.sum() + m2.sum())
    fz = np.concatenate(fv[0])
    fxy = np.concatenate(fv[1] + fv[2])
    fall = np.concatenate([fz, fxy])
    rec["touch_frac"] = touch / nfaces if nfaces else np.nan
    rec["n_face_z"], rec["n_face_xy"] = int(fz.size), int(fxy.size)
    rec["I_inner"] = float(i1[inner].mean()) if inner.any() else np.nan
    rec["I_outer"] = float(i1[outer].mean()) if outer.any() else np.nan
    rec["I_face"] = float(fall.mean()) if fall.size else np.nan
    rec["I_face_z"] = float(fz.mean()) if fz.size else np.nan
    rec["I_face_xy"] = float(fxy.mean()) if fxy.size else np.nan
    for k in ("inner", "outer", "face", "face_z", "face_xy"):
        rec["r_" + k] = (rec["I_" + k] - bg) / c
    # ---- 1-D lines through the centroid voxel
    v0 = np.round(cen).astype(int)
    if not G[tuple(v0)]:
        dd = ((idx[0] - cen[0]) / rz) ** 2 + ((idx[1] - cen[1]) / rxy) ** 2 + ((idx[2] - cen[2]) / rxy) ** 2
        k = int(np.argmin(dd))
        v0 = np.array([idx[0][k], idx[1][k], idx[2][k]])
        rec["centroid_outside"] = 1
    else:
        rec["centroid_outside"] = 0
    cls = np.zeros(G.shape, np.int8)
    cls[G] = 1
    cls[O] = -1
    cls[IG & ~G & ~O] = -2
    lines = {"z": (I[:, v0[1], v0[2]], cls[:, v0[1], v0[2]], v0[0]),
             "y": (I[v0[0], :, v0[2]], cls[v0[0], :, v0[2]], v0[1]),
             "x": (I[v0[0], v0[1], :], cls[v0[0], v0[1], :], v0[2])}
    cmax = rec["peakmax"] - bg
    for a, (p, cl, k) in lines.items():
        lm, (s, e, wl, wr) = line_metrics(p, cl, k, bg, c)
        rec["run_" + a] = lm["run"]
        rec["edge_" + a] = lm["edge"]
        rr, off, cens, below = fwhm(p, s, e, wl, wr, bg + 0.5 * c)
        rec["ratio_" + a], rec["off_" + a], rec["cens_" + a], rec["below_" + a] = rr, off, cens, below
        rr2, _, cens2, _ = fwhm(p, s, e, wl, wr, bg + 0.5 * cmax)                   # sensitivity: peak = max
        rec["ratio_" + a + "_pmax"], rec["cens_" + a + "_pmax"] = rr2, cens2
        rec["grad_" + a], rec["gradlast_" + a] = grad_edge(p, s, e, wl, wr)
    # ---- 3-D iso-level blobs in the nucleus' territory (GT bbox padded by 1x extent)
    IB = tuple(slice(max(0, g.start - p), min(n, g.stop + p)) for g, p, n in zip(gb, ext, G.shape))
    Li, Gi, Ii, IGi = Lb[IB], G[IB], I[IB], IG[IB]
    if (Li > 0).sum() > Gi.sum():
        _, inds = ndi.distance_transform_edt(Li == 0, sampling=vox, return_indices=True)
        near = Li[inds[0], inds[1], inds[2]]
        terr = (near == i) & ~IGi
        del inds
    else:
        terr = ~IGi
    gspan = [int(Gi.any(axis=tuple(j for j in range(3) if j != ax)).sum()) for ax in range(3)]
    gidx = np.nonzero(Gi)
    gvz, gvxy = gidx[0].var() + 1 / 12, gidx[1].var() + gidx[2].var() + 2 / 12     # +1/12: voxel's own extent
    vg = int(Gi.sum())
    vr, iou, zr, xyr, zsp, xysp, cens3 = [], [], [], [], [], [], []
    for l in LEVELS:
        Sm = (Ii >= bg + l * c) & terr
        ls, n = ndi.label(Sm)
        ids = np.unique(ls[Gi & Sm])
        ids = ids[ids > 0]
        if ids.size == 0:
            for L_ in (vr, iou, zr, xyr, zsp, xysp):
                L_.append(0.0)
            cens3.append(0)
            continue
        keep = np.zeros(n + 1, bool)
        keep[ids] = True
        blob = fill_holes(keep[ls])
        vb = int(blob.sum())
        inter = int((blob & Gi).sum())
        bidx = np.nonzero(blob)
        vr.append(vb / vg)
        iou.append(inter / (vb + vg - inter))
        # RMS (second-moment) extent ratios, blob / GT
        zr.append(float(np.sqrt((bidx[0].var() + 1 / 12) / gvz)))
        xyr.append(float(np.sqrt((bidx[1].var() + bidx[2].var() + 2 / 12) / gvxy)))
        sp = [int(blob.any(axis=tuple(j for j in range(3) if j != ax)).sum()) for ax in range(3)]
        zsp.append(sp[0] / gspan[0])
        xysp.append(0.5 * (sp[1] / gspan[1] + sp[2] / gspan[2]))
        cens3.append(int(blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any()
                         or blob[:, :, 0].any() or blob[:, :, -1].any()))
    for j, l in enumerate(LEVELS):
        t = f"{l:.2f}"
        rec["vr_" + t], rec["iou_" + t], rec["zr_" + t], rec["xyr_" + t] = vr[j], iou[j], zr[j], xyr[j]
        rec["zsp_" + t], rec["xysp_" + t], rec["c3_" + t] = zsp[j], xysp[j], cens3[j]
    for nm, rat in (("vol", vr), ("z", zr), ("xy", xyr), ("zsp", zsp), ("xysp", xysp)):
        rec["lvl_" + nm], rec["lvlflag_" + nm] = matched_level(LEVELS, rat)
    jb = int(np.argmax(iou))
    rec["best_iou"], rec["best_lvl"] = iou[jb], LEVELS[jb]
    return rec


# ----------------------------------------------------------------------------------------------- driver
def run_dataset(args):
    ds, a = args
    t0 = time.time()
    f = a["down"].get(ds, 1)
    zf = a["zdown"].get(ds, 1)
    vols = select_volumes(ds, a["nvol"])
    data, vinfo, pool = [], [], []
    for vi, r in enumerate(vols):
        img, lab, ign, vox, info = load_volume(r, a["crop"], f, a["gt_rule"], zf)
        el = eligible(lab, ign)
        info["n_eligible"] = len(el)
        data.append((img, lab, ign, vox, info))
        vinfo.append(info)
        pool += [(vi, i, sl) for i, sl in el]
    rng = np.random.default_rng(a["seed"])
    pick = rng.choice(len(pool), size=min(a["max_nuc"], len(pool)), replace=False) if pool else []
    pick = sorted(int(j) for j in pick)
    recs = []
    for j in pick:
        vi, i, sl = pool[j]
        img, lab, ign, vox, info = data[vi]
        rec = dict(ds=ds, vol=info["id"], down=f)
        rec.update(per_nucleus(i, sl, img, lab, ign, vox, info["bg_median_crop"]))
        recs.append(rec)
    for info in vinfo:
        info["n_sampled"] = sum(1 for r in recs if r["vol"] == info["id"])
    os.makedirs(a["out"], exist_ok=True)
    keys = []
    for r in recs:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(f"{a['out']}/nuclei_{ds}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    json.dump(dict(ds=ds, down=f, zdown=zf, gt_rule=a["gt_rule"], seed=a["seed"], crop=a["crop"], volumes=vinfo,
                   n_pool=len(pool), n_sampled=len(recs), seconds=round(time.time() - t0, 1)),
              open(f"{a['out']}/volumes_{ds}.json", "w"), indent=1)
    return f"{ds}: down={f} zdown={zf} vols={[v['id'] for v in vinfo]} pool={len(pool)} sampled={len(recs)} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", nargs="+", required=True)
    ap.add_argument("--nvol", type=int, default=4)
    ap.add_argument("--max_nuc", type=int, default=300)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--down", nargs="*", default=[], help="ds=F xy block-downsampling factors (control)")
    ap.add_argument("--zdown", nargs="*", default=[], help="ds=F z block-downsampling factors (reverse control)")
    ap.add_argument("--gt_rule", default="majority", choices=["majority", "any"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = dict(out=a.out, nvol=a.nvol, max_nuc=a.max_nuc, crop=a.crop, seed=a.seed, gt_rule=a.gt_rule,
               down={k: int(v) for k, v in (d.split("=") for d in a.down)},
               zdown={k: int(v) for k, v in (d.split("=") for d in a.zdown)})
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), levels=LEVELS), open(f"{a.out}/run_config.json", "w"), indent=1)
    with Pool(a.workers) as P:
        for msg in P.imap_unordered(run_dataset, [(d, cfg) for d in a.ds]):
            print(msg, flush=True)
