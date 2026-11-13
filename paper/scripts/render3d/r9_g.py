"""Fig. 3g, round 9 (team lead 2026-11-10): denser panel g for the median-rule volume mouse_embryo E15_Ab2 (seed 0), dark background.
Registered before computing: AGENTS.md "FIG3G 事后描述" (server repo cellposeft_orx, commit f5e0cb9): post hoc, single volume, descriptive.
Masks: evaluated final masks (render3d/export/mouse_embryo.npz: rule T, fragments < 0.15 V removed; exported from the thr_eval npz of
runs 07992b55 (S1) / 81116913 (2D control)). Matching: best IoU >= 0.5 (unique), both models.
Elements:
  xz plane (as R7: y = 148 of the R5 crop) with three numbered boxes;
  three zoomed xz callouts of single nuclei: among GT nuclei that intersect the shown xz crop and are matched by both models, sort by
    dIoU = IoU(S1) - IoU(2D) and take the nucleus nearest to the 10th, 50th and 90th percentile of dIoU (nearest rank; ties -> lower id),
    labelled with both IoUs;
  xy plane (as R7: z = 39);
  inset: for all GT nuclei of the volume matched by both models, GT surface voxels split by surface orientation (normal from the gradient
    of the Gaussian-smoothed GT mask in physical units; axial cap if the angle between normal and z <= 45 deg, otherwise lateral side),
    per-nucleus mean distance (um) from GT surface voxels to the matched prediction's surface (EDT sampled with the voxel size,
    one-sided GT -> prediction); paired dots 2D control vs S1, medians marked; labelled 'single volume, descriptive'. No tests."""
import json
import os

import numpy as np
from scipy import ndimage as ndi
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, ConnectionPatch
from skimage.segmentation import find_boundaries

import r5_panels as P5
import rstyle as R

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "out", "r9_cache"); os.makedirs(CACHE, exist_ok=True)
Z = P5.Z
GT, S1F, D2F = Z["gt"].astype(np.int32), Z["S1"].astype(np.int32), Z["2D"].astype(np.int32)
VOX = P5.VOX; RAT = P5.RAT
CAP_DEG = 45.0


def best_match(gt, pm):
    """GT id -> (pred id, IoU) for IoU >= 0.5 (unique when > 0.5)."""
    m = (gt > 0) & (pm > 0)
    gsz = np.bincount(gt.ravel()); psz = np.bincount(pm.ravel(), minlength=int(pm.max()) + 1)
    key = gt[m].astype(np.int64) * (int(pm.max()) + 1) + pm[m]
    u, c = np.unique(key, return_counts=True)
    g, p = u // (int(pm.max()) + 1), u % (int(pm.max()) + 1)
    iou = c / (gsz[g] + psz[p] - c)
    out = {}
    for gi, pi, v in zip(g, p, iou):
        if v >= 0.5 and v > out.get(int(gi), (0, 0.0))[1]:
            out[int(gi)] = (int(pi), float(v))
    return out


def orientation_distances():
    fn = os.path.join(CACHE, "orient.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    MS, M2 = best_match(GT, S1F), best_match(GT, D2F)
    both = sorted(set(MS) & set(M2))
    objs = ndi.find_objects(GT)
    st6 = ndi.generate_binary_structure(3, 1)
    res = []
    for g in both:
        sl = objs[g - 1]
        pad = 4
        sl = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, GT.shape))
        A = GT[sl] == g
        surfA = A & ~ndi.binary_erosion(A, st6, border_value=0)
        sm = ndi.gaussian_filter(A.astype(np.float32), sigma=(0.6 / VOX[0], 0.3 / VOX[1], 0.3 / VOX[2]))   # 0.6 um in z, 0.3 um in xy
        gz, gy, gx = np.gradient(sm, *VOX)
        nrm = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2) + 1e-12
        cosz = np.abs(gz) / nrm
        cap = surfA & (cosz >= np.cos(np.deg2rad(CAP_DEG)))
        side = surfA & ~cap
        row = dict(gt=g, n_cap=int(cap.sum()), n_side=int(side.sum()), iou_S1=MS[g][1], iou_2D=M2[g][1])
        for name, pm, M in (("S1", S1F, MS), ("2D", D2F, M2)):
            B = pm[sl] == M[g][0]
            surfB = B & ~ndi.binary_erosion(B, st6, border_value=0)
            dB = ndi.distance_transform_edt(~surfB, sampling=VOX)
            row[f"cap_{name}"] = float(dB[cap].mean()) if cap.sum() >= 10 else None
            row[f"side_{name}"] = float(dB[side].mean()) if side.sum() >= 10 else None
        res.append(row)
    json.dump(res, open(fn, "w"), indent=1)
    return res


def callout_picks():
    """Nuclei in the shown xz crop (y = ymid of the R5 crop), matched by both; nearest to the 10/50/90th dIoU percentiles."""
    y = P5.ymid + P5.y0
    sec = GT[P5.z0:P5.z0 + P5.nz, y, P5.x0:P5.x0 + P5.s]
    ids = [int(i) for i in np.unique(sec) if i]
    MS, M2 = best_match(GT, S1F), best_match(GT, D2F)
    cand = [(g, MS[g][1] - M2[g][1], MS[g][1], M2[g][1]) for g in ids if g in MS and g in M2]
    d = np.array([c[1] for c in cand])
    picks = []
    for q in (10, 50, 90):
        target = np.percentile(d, q)
        best = min(cand, key=lambda c: (abs(c[1] - target), c[0]))
        picks.append(dict(percentile=q, target_dIoU=float(target), gt=best[0], dIoU=best[1], iou_S1=best[2], iou_2D=best[3]))
    return picks, dict(n_candidates=len(cand), in_section=ids, excluded_unmatched=[g for g in ids if not (g in MS and g in M2)],
                       all=[dict(gt=c[0], dIoU=c[1], iou_S1=c[2], iou_2D=c[3]) for c in sorted(cand, key=lambda c: c[1])])


def _rgb_xz(gid_box=None):
    from scipy.ndimage import zoom
    y = P5.ymid
    img = P5.img[:, y]; labs = [(P5.gt[:, y], P5.GTC), (P5.d2[:, y], R.D2), (P5.s1[:, y], R.S1)]
    zf = (RAT, 1)
    im = zoom(img, zf, order=1)
    rgb = np.repeat(P5._lut(im, P5.LO, P5.HI)[..., None], 3, -1)
    for L, c in labs:
        rgb[find_boundaries(zoom(L, zf, order=0), mode="inner")] = to_rgb(c)
    return rgb


def _rgb_xz_full():
    """Whole-volume xz plane at the same y (callouts are cut from it so that nuclei at the crop edge stay complete)."""
    from scipy.ndimage import zoom
    yy = P5.ymid + P5.y0
    img = P5.IMG[:, yy]; labs = [(GT[:, yy], P5.GTC), (D2F[:, yy], R.D2), (S1F[:, yy], R.S1)]
    im = zoom(img, (RAT, 1), order=1)
    rgb = np.repeat(P5._lut(im, P5.LO, P5.HI)[..., None], 3, -1)
    for L, c in labs:
        rgb[find_boundaries(zoom(L, (RAT, 1), order=0), mode="inner")] = to_rgb(c)
    return rgb


def draw_g(fig, W, H, x, y, w, h, dark=True):
    """Draw panel g into the rectangle (mm from top-left). Returns provenance dict."""
    import matplotlib.pyplot as plt
    # dark=False (round 10): white page, dark only inside the image tiles; text, inset and legend in light-page colours
    tc, mc = (P5.DARK_TC, P5.DARK_MC) if dark else (R.FG, R.MUTED)
    D2T, S1T = (R.D2, R.S1) if dark else (R.PALETTE["models"]["2D_dark"], R.PALETTE["models"]["S1_dark"])
    GTL = P5.GTC if dark else "#8a8a8e"
    if dark:
        fig.add_artist(Rectangle((x / W, 1 - (y + h) / H), w / W, h / H, transform=fig.transFigure, fc=R.DARK_BG[0], ec="none", zorder=-5))
    up = lambda a: np.repeat(np.repeat(a, 4, 0), 4, 1)  # noqa: E731  (4x nearest, embedded unresampled)
    mm = lambda xx, yy, ww, hh: fig.add_axes(P5._mm(fig, W, H, xx, yy, ww, hh))  # noqa: E731
    top, bot = 4.0, 3.0
    side = h - top - bot
    s = P5.s
    # xz plane
    rgb = _rgb_xz()
    axl = mm(x + 1.5, y + top, side, side)
    axl.imshow(up(rgb), interpolation="none", extent=(-0.5, s - 0.5, rgb.shape[0] - 0.5, -0.5))
    picks, pinfo = callout_picks()
    # callout boxes on the xz plane
    ysec = P5.ymid
    for k, pk in enumerate(picks):
        sec = GT[:, ysec + P5.y0] == pk["gt"]          # footprint in the whole-volume plane (complete even at the crop edge)
        zz, xx = np.nonzero(sec)
        m = 6
        x1, x2 = xx.min() - P5.x0 - m, xx.max() - P5.x0 + m
        z1, z2 = (zz.min() - P5.z0 - 0.5) * RAT - m, (zz.max() - P5.z0 + 0.5) * RAT + m
        pk["box_px"] = [int(x1), int(x2), float(z1), float(z2)]
        axl.add_patch(Rectangle((x1, z1), x2 - x1, z2 - z1, fill=False, ec="#ffd166", lw=0.7, clip_on=True))
        axl.text(x1 + 2, z1 - 3, str(k + 1), color="#ffd166", fontsize=5.6, fontweight="bold", va="bottom")
    # callouts (cut from the whole-volume xz plane at the same y, so nuclei touching the crop edge stay complete)
    full = _rgb_xz_full()
    axl.set_xlim(-0.5, s - 0.5); axl.set_ylim(rgb.shape[0] - 0.5, -0.5); axl.set_facecolor("black")
    cw = 17.0; gap = 2.6; cx0 = x + 1.5 + side + 4.0
    caxes = []
    for k, pk in enumerate(picks):
        x1, x2, z1, z2 = pk["box_px"]
        X1, X2, Zz1, Zz2 = x1 + P5.x0, x2 + P5.x0, z1 + P5.z0 * RAT, z2 + P5.z0 * RAT
        sz = max(X2 - X1, Zz2 - Zz1)
        xc_, zc_ = (X1 + X2) / 2, (Zz1 + Zz2) / 2
        ax = mm(cx0 + k * (cw + gap), y + top + 2.4, cw, cw)
        ax.imshow(up(full), interpolation="none", extent=(-0.5, full.shape[1] - 0.5, full.shape[0] - 0.5, -0.5))
        ax.set_xlim(xc_ - sz / 2, xc_ + sz / 2); ax.set_ylim(zc_ + sz / 2, zc_ - sz / 2); ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#ffd166"); sp.set_linewidth(0.6)
        ax.set_title(f"{k + 1} ({pk['percentile']}th pct ΔIoU)", fontsize=5.6, color=tc, pad=1.5)
        L = 2.0 / P5.UM_PX
        ax.plot([xc_ - sz / 2 + sz * 0.06, xc_ - sz / 2 + sz * 0.06 + L], [zc_ + sz / 2 - sz * 0.07] * 2, color="#ffffff", lw=0.9, solid_capstyle="butt")
        cxm = (cx0 + k * (cw + gap) + cw / 2) / W
        fig.text(cxm, 1 - (y + top + 2.4 + cw + 0.9) / H, f"2D  IoU {pk['iou_2D']:.2f}", ha="center", va="top", fontsize=5.6, color=D2T)
        fig.text(cxm, 1 - (y + top + 2.4 + cw + 3.2) / H, f"S1  IoU {pk['iou_S1']:.2f}", ha="center", va="top", fontsize=5.6, color=S1T)
        caxes.append(ax)
    fig.text((cx0 + (3 * cw + 2 * gap) / 2) / W, 1 - (y + top + 2.4 + cw + 6.0) / H, "callout bar 2 µm; xz rows interpolated to physical aspect",
             fontsize=5.0, color=mc, va="top", ha="center")
    # xy plane
    xx0 = cx0 + 3 * cw + 2 * gap + 4.0
    axr = mm(xx0, y + top, side, side)
    labs = [(P5.gt[P5.kmid], P5.GTC), (P5.d2[P5.kmid], R.D2), (P5.s1[P5.kmid], R.S1)]
    axr.imshow(up(P5._rgb(P5.img[P5.kmid], labs)), interpolation="none", extent=(-0.5, s - 0.5, s - 0.5, -0.5))
    for a_, ttl in ((axl, f"xz plane (y = {P5.ymid + P5.y0})"), (axr, f"xy plane (z = {P5.kmid + P5.z0})")):
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_color(tc); sp.set_linewidth(0.6)
        a_.set_title(ttl, fontsize=6.0, fontweight="bold", color=tc, pad=2)
    P5._scalebar(axr, 10 / P5.UM_PX, "10 µm", "#ffffff", s, s)
    P5._scalebar(axl, 10 / P5.UM_PX, "10 µm", "#ffffff", rgb.shape[0], s)
    # inset: orientation-split surface distance
    od = orientation_distances()
    ix = xx0 + side + 5.0; iw = x + w - ix - 1.5
    ax = mm(ix + 5.5, y + top + 1.0, iw - 6.0, side - 9.0)
    ax.set_facecolor("none")
    for gi, reg in enumerate(("cap", "side")):
        a2 = np.array([r[f"{reg}_2D"] for r in od if r[f"{reg}_2D"] is not None and r[f"{reg}_S1"] is not None])
        a1 = np.array([r[f"{reg}_S1"] for r in od if r[f"{reg}_2D"] is not None and r[f"{reg}_S1"] is not None])
        x2_, x1_ = gi * 1.0 - 0.17, gi * 1.0 + 0.17
        for v2, v1 in zip(a2, a1):
            ax.plot([x2_, x1_], [v2, v1], color="#6b6f7a" if dark else "#c8c8cc", lw=0.3, zorder=1)
        ax.scatter(np.full(len(a2), x2_), a2, s=2.0, color=R.D2, lw=0, zorder=2)
        ax.scatter(np.full(len(a1), x1_), a1, s=2.0, color=R.S1, lw=0, zorder=2)
        for xv, arr, c in ((x2_, a2, R.D2), (x1_, a1, R.S1)):
            ax.plot([xv - 0.12, xv + 0.12], [np.median(arr)] * 2, color=c, lw=1.2, zorder=3)
        pk_ = dict(n=int(len(a1)), median_2D=float(np.median(a2)), median_S1=float(np.median(a1)))
        od_summary = od_summary if gi else {}
        od_summary[reg] = pk_
    ax.set_yscale("log")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["axial caps", "lateral sides"], fontsize=5.6, color=tc)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("mean surface distance (µm)", fontsize=5.6, color=tc, labelpad=1)
    ax.tick_params(colors=tc, labelsize=5.4, length=1.5, pad=1)
    for k_, sp in ax.spines.items():
        sp.set_color(tc if k_ in ("left", "bottom") else "none"); sp.set_linewidth(0.5)
    ax.set_title("GT → prediction boundary", fontsize=5.6, color=tc, pad=2)
    fig.text((ix + iw / 2 + 2.0) / W, 1 - (y + h - 0.5) / H, f"single volume, descriptive\nn = {od_summary['cap']['n']} nuclei matched by both",
             fontsize=5.0, color=mc, ha="center", va="bottom")
    # legend + source line
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color=GTL, lw=1.2, label=R.LAB["gt"] + ("" if dark else " (white)")), Line2D([], [], color=R.D2, lw=1.2, label=R.LAB["2D"]),
           Line2D([], [], color=R.S1, lw=1.2, label=R.LAB["S1"])]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(((x + w / 2) if dark else (cx0 + (3 * cw + 2 * gap) / 2)) / W, 1 - (y + 0.4) / H), ncol=3, fontsize=5.6, frameon=False,
               labelcolor=tc, handlelength=1.4, columnspacing=1.0 if dark else 0.7, borderaxespad=0)
    fig.text((x + 1.5 + side / 2) / W, 1 - (y + h - 0.5) / H, f"{P5.SOURCE['dataset']} {P5.SOURCE['volume']} (median rule), seed 0",
             fontsize=5.0, color=mc, ha="center", va="bottom")
    return {"panel": "g (round 9): xz plane + three dIoU-percentile callouts + xy plane + orientation-split surface distance inset",
            "registration": "AGENTS.md FIG3G 事后描述, cellposeft_orx commit f5e0cb9 (post hoc, single volume, descriptive)",
            "callout_rule": "GT nuclei intersecting the shown xz crop and matched by both models (best IoU >= 0.5); dIoU = IoU(S1) - IoU(2D); nucleus nearest the 10th/50th/90th percentile of dIoU (ties -> lower id)",
            "callouts": [{k: v for k, v in p.items() if k != "box_px"} for p in picks], "callout_pool": pinfo,
            "inset_rule": f"all GT nuclei matched by both; GT surface voxels; normal from gradient of Gaussian-smoothed GT mask (physical units); cap if angle to z <= {CAP_DEG} deg; per-nucleus mean one-sided distance GT surface -> matched prediction surface (EDT, sampling = voxel size um); >= 10 voxels per region",
            "inset_summary": od_summary, "inset_per_nucleus": "scripts/render3d/out/r9_cache/orient.json",
            "xz_pulled_y": int(P5.ymid + P5.y0), "xy_pulled_z": int(P5.kmid + P5.z0), "outlines": {"GT": P5.GTC, "2D": R.D2, "S1": R.S1},
            "rasters": "4x nearest upsampling, embedded unresampled (interpolation none)", **P5.SOURCE}
