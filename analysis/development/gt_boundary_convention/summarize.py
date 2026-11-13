#!/usr/bin/env python
"""Per-dataset summary of the GT boundary-convention metrics written by gt_conv.py (nuclei_<ds>.csv).

Writes <dir>/summary.json (medians, IQRs, per-volume medians, SNR>=10 subset, elasticity, mesospim outlier test)
and <dir>/summary_table.csv (one row per dataset), and prints compact tables.
Usage: python summarize.py <results dir> [--focus mesospim]
"""
import argparse
import csv
import glob
import json
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# (key, short label) - metrics summarised per dataset
METRICS = [
    ("run_z", "GT diameter z (vox, centroid run)"),
    ("run_xy", "GT diameter xy (vox, mean of y/x centroid runs)"),
    ("snr", "SNR = (p95 - bg) / noise"),
    ("peak_flat", "(p95 - bg)/(max - bg): ~1 plateau, lower = peaked"),
    ("r_inner", "r on inner boundary voxels (literal)"),
    ("r_outer", "r on outer boundary voxels"),
    ("r_face", "r at GT surface (face midpoints, all)"),
    ("r_face_z", "r at GT surface, z-normal faces"),
    ("r_face_xy", "r at GT surface, xy-normal faces"),
    ("edge_z", "1-D centroid line: level at GT end, z"),
    ("edge_xy", "1-D centroid line: level at GT end, xy"),
    ("ratio_z", "half-max extent / GT extent, z (1-D)"),
    ("ratio_xy", "half-max extent / GT extent, xy (1-D)"),
    ("ratio_z_pmax", "same, peak = max, z"),
    ("ratio_xy_pmax", "same, peak = max, xy"),
    ("grad_z", "steepest-drop position / GT half-extent, z"),
    ("grad_xy", "steepest-drop position / GT half-extent, xy"),
    ("off_z", "(blob centre - GT centre)/GT extent, z"),
    ("lvl_vol", "3-D iso level matching GT volume"),
    ("lvl_z", "3-D iso level matching GT RMS z extent"),
    ("lvl_xy", "3-D iso level matching GT RMS xy extent"),
    ("vr50", "3-D half-max blob volume / GT volume"),
    ("zr50", "3-D half-max blob RMS z extent / GT"),
    ("xyr50", "3-D half-max blob RMS xy extent / GT"),
    ("iou50", "IoU(3-D half-max blob, GT)"),
    ("best_iou", "best IoU over iso levels"),
    ("best_lvl", "iso level of best IoU"),
    ("r_face_z_pmax", "r_face_z with peak = max"),
    ("r_face_xy_pmax", "r_face_xy with peak = max"),
    ("r_face_z_bg0", "r_face_z with bg = crop 1st percentile"),
    ("r_face_xy_bg0", "r_face_xy with bg = crop 1st percentile"),
    ("touch_frac", "fraction of GT surface touching other nuclei"),
]
FLAGS = ["cens_z", "cens_xy", "below_z", "below_xy", "bg_fallback", "lvlflag_vol_low", "lvlflag_z_low", "lvlflag_xy_low"]


def load(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/nuclei_*.csv")):
        R = list(csv.DictReader(open(f)))
        if not R:
            continue
        C = {}
        for k in R[0]:
            if k in ("ds", "vol"):
                C[k] = np.array([r[k] for r in R])
            else:
                C[k] = np.array([float(r[k]) if r[k] not in ("", "nan", "None") else np.nan for r in R])
        out[R[0]["ds"]] = C
    return out


def nanmean2(a, b):
    m = np.vstack([a, b])
    with np.errstate(all="ignore"):
        return np.nanmean(m, 0)


def derive(C):
    v = C["valid"] == 1
    C = {k: x[v] for k, x in C.items()}
    C["run_xy"] = 0.5 * (C["run_y"] + C["run_x"])
    for k in ("edge", "ratio", "ratio_{}_pmax", "grad", "below"):
        if "{}" in k:
            C[k.format("xy")] = nanmean2(C[k.format("y")], C[k.format("x")])
        else:
            C[k + "_xy"] = nanmean2(C[k + "_y"], C[k + "_x"])
    C["cens_xy"] = 0.5 * (C["cens_y"] + C["cens_x"])
    c95, cmax = C["peak95"] - C["bg"], C["peakmax"] - C["bg"]
    C["peak_flat"] = c95 / cmax
    C["r_face_z_pmax"] = (C["I_face_z"] - C["bg"]) / cmax
    C["r_face_xy_pmax"] = (C["I_face_xy"] - C["bg"]) / cmax
    C["r_face_z_bg0"] = C["I_face_z"] / C["peak95"]
    C["r_face_xy_bg0"] = C["I_face_xy"] / C["peak95"]
    for nm in ("vr", "zr", "xyr", "iou"):
        C[nm + "50"] = C[nm + "_0.50"]
    for nm in ("vol", "z", "xy"):
        C[f"lvlflag_{nm}_low"] = (C["lvlflag_" + nm] == -1).astype(float)
    return C


def q(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(med=np.nan, q25=np.nan, q75=np.nan, n=0)
    return dict(med=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)), n=int(x.size))


def elasticity(C):
    """Within-volume slope of log(I_face - bg) on log(peak95 - bg): ~1 if the boundary scales with the nucleus'
    own contrast (relative convention), ~0 if it sits at a fixed absolute intensity (global threshold)."""
    b, c = C["I_face"] - C["bg"], C["peak95"] - C["bg"]
    ok = (b > 0) & (c > 0)
    x, y, vol = np.log(c[ok]), np.log(b[ok]), C["vol"][ok]
    xd, yd = x.copy(), y.copy()
    for v in np.unique(vol):
        m = vol == v
        xd[m] -= x[m].mean()
        yd[m] -= y[m].mean()
    slope = float((xd * yd).sum() / (xd * xd).sum()) if (xd * xd).sum() > 0 else np.nan
    rho = stats.spearmanr(C["r_face"], c, nan_policy="omit")[0]
    # absolute boundary level (bg-subtracted) spread vs contrast spread, within volume
    return dict(slope_logb_logc=slope, spearman_rface_contrast=float(rho),
                sd_log_b=float(yd.std()), sd_log_c=float(xd.std()), n=int(ok.sum()))


def summarize(D, focus="mesospim"):
    S = {}
    for ds, C0 in D.items():
        C = derive(C0)
        s = dict(n_nuclei=int(C["valid"].size), n_vol=int(len(np.unique(C["vol"]))), vols=sorted(np.unique(C["vol"]).tolist()))
        s["metrics"] = {k: q(C[k]) for k, _ in METRICS}
        s["flags"] = {k: float(np.nanmean(C[k])) for k in FLAGS if k in C}
        s["per_volume"] = {v: {k: q(C[k][C["vol"] == v])["med"] for k, _ in METRICS} for v in s["vols"]}
        hi = C["snr"] >= 10
        s["snr_ge10"] = dict(n=int(hi.sum()), metrics={k: q(C[k][hi]) for k, _ in METRICS})
        s["elasticity"] = elasticity(C)
        S[ds] = s
    if focus in S:
        out = {}
        for k, _ in METRICS:
            ref = np.array([S[d]["metrics"][k]["med"] for d in S if d != focus])
            ref = ref[np.isfinite(ref)]
            m = S[focus]["metrics"][k]["med"]
            mad = 1.4826 * np.median(np.abs(ref - np.median(ref))) if ref.size else np.nan
            out[k] = dict(focus=m, ref_min=float(ref.min()), ref_max=float(ref.max()), ref_median=float(np.median(ref)),
                          robust_z=float((m - np.median(ref)) / mad) if mad > 0 else np.nan,
                          rank_from_low=int((ref < m).sum()) + 1, n_ref=int(ref.size),
                          outside_ref_range=bool(m < ref.min() or m > ref.max()))
        S["_outlier_test_" + focus] = out
    return S


def fmt(m, k, iqr=True):
    x = m[k]
    if not np.isfinite(x["med"]):
        return "  -  "
    return f"{x['med']:.2f} ({x['q25']:.2f}-{x['q75']:.2f})" if iqr else f"{x['med']:.2f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--focus", default="mesospim")
    a = ap.parse_args()
    D = load(a.dir)
    S = summarize(D, a.focus)
    json.dump(S, open(f"{a.dir}/summary.json", "w"), indent=1)
    dss = sorted([d for d in S if not d.startswith("_")], key=lambda d: (d == a.focus, d))
    with open(f"{a.dir}/summary_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ds", "n_nuclei", "n_vol"] + [f"{k}_{s}" for k, _ in METRICS for s in ("med", "q25", "q75")]
                   + [f"flag_{k}" for k in FLAGS] + ["elast_slope", "elast_rho"])
        for d in dss:
            s = S[d]
            w.writerow([d, s["n_nuclei"], s["n_vol"]]
                       + [f"{s['metrics'][k][t]:.4f}" for k, _ in METRICS for t in ("med", "q25", "q75")]
                       + [f"{s['flags'].get(k, np.nan):.3f}" for k in FLAGS]
                       + [f"{s['elasticity']['slope_logb_logc']:.3f}", f"{s['elasticity']['spearman_rface_contrast']:.3f}"])
    groups = [["run_z", "run_xy", "snr", "peak_flat", "touch_frac"],
              ["r_inner", "r_face_z", "r_face_xy", "edge_z", "edge_xy"],
              ["ratio_z", "ratio_xy", "grad_z", "grad_xy", "off_z"],
              ["lvl_vol", "lvl_z", "lvl_xy", "iou50", "best_iou", "best_lvl"],
              ["ratio_z_pmax", "ratio_xy_pmax", "r_face_z_pmax", "r_face_xy_pmax", "r_face_z_bg0", "r_face_xy_bg0"]]
    for g in groups:
        print("\n" + "ds".ljust(13) + "n".rjust(5) + "".join(k.rjust(19) for k in g))
        for d in dss:
            s = S[d]
            print(d.ljust(13) + str(s["n_nuclei"]).rjust(5) + "".join(fmt(s["metrics"], k).rjust(19) for k in g))
    print("\nflags (fraction of nuclei)")
    print("ds".ljust(13) + "".join(k.rjust(16) for k in FLAGS) + "elast_slope".rjust(13) + "rho(r,c)".rjust(10))
    for d in dss:
        s = S[d]
        print(d.ljust(13) + "".join(f"{s['flags'].get(k, np.nan):.2f}".rjust(16) for k in FLAGS)
              + f"{s['elasticity']['slope_logb_logc']:.2f}".rjust(13) + f"{s['elasticity']['spearman_rface_contrast']:.2f}".rjust(10))
    ot = S.get("_outlier_test_" + a.focus)
    if ot:
        print(f"\n{a.focus} vs the other {ot['run_z']['n_ref']} datasets (dataset medians)")
        for k, lab in METRICS:
            o = ot[k]
            print(f"  {k:15s} {o['focus']:7.3f}  ref median {o['ref_median']:6.3f} range [{o['ref_min']:6.3f}, {o['ref_max']:6.3f}]"
                  f"  rank {o['rank_from_low']:2d}/{o['n_ref'] + 1}  robust z {o['robust_z']:6.2f}  {'OUTSIDE' if o['outside_ref_range'] else ''}")
