"""直接边界指标（口径见 BOUNDARY.md，事后分析，只报告）。
第一步（逐模型 × 体积）：匈牙利匹配 → IoU ≥ 0.5 的匹配核对 → 每对的 ASSD、HD95（像素口径；µm = 像素 × vox_xy）、IoU；写 boundary/per/<tag>/<key>.npz。
第二步：按套 pooled → 宏平均（13 套 / 12 套 / µm 套 / C1），S1−二维五种子配对差，共同匹配敏感性分析；写 boundary/results.json 与 report.md。
用法：python boundary.py [--workers 16] [--agg_only]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import argparse
import glob
import json
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import common as C  # noqa: E402

OUT = f"{C.OUT}/boundary"
BL = f"{_NSP_SHARED}/baselines/results/eval"
BASELINES = {"nnunet": (f"{BL}/nnunet_test_gt", "pred_v4", 0.25, 4.0), "stardist": (f"{BL}/stardistB_test_gt", "pred_v2", 0.15, 2.0)}
NO_UM = {"bbbc050", "nisnet_zf", "nis3d", "purdue"}
ST6 = ndi.generate_binary_structure(3, 1)


def tags():
    runs = json.load(open(f"{C.OUT}/runs.json"))
    t = sorted(k for k in runs if k.startswith("gt_"))
    return t + ["gt_nnunet_s0", "gt_stardist_s0"], runs


def surf(m):
    return m & ~ndi.binary_erosion(m, ST6, border_value=0)


def pair_metrics(gt, pr, gl, pl, sg, sp, samp):
    sl = tuple(slice(max(min(a.start, b.start) - 1, 0), min(max(a.stop, b.stop) + 1, n)) for a, b, n in zip(sg, sp, gt.shape))
    A = gt[sl] == gl; B = pr[sl] == pl
    SA, SB = surf(A), surf(B)
    dB = ndi.distance_transform_edt(~SB, sampling=samp); dA = ndi.distance_transform_edt(~SA, sampling=samp)
    ab, ba = dB[SA], dA[SB]
    assd = (ab.sum() + ba.sum()) / max(len(ab) + len(ba), 1)
    hd95 = max(np.percentile(ab, 95), np.percentile(ba, 95))
    return assd, hd95


def job(arg):
    tg, key, src = arg
    fo = f"{OUT}/per/{tg}/{key}.npz"
    if os.path.exists(fo):
        return "skip"
    try:
        z = np.load(src["gtnpz"])
        gt = z["gt"]; vox = np.asarray(z["vox"], float); d = float(z["d"])
        if src["kind"] == "cache":
            pr = np.load(src["path"])["mask"]
        else:
            b = np.load(src["path"]); pr = C.drop_small(b[src["field"]].astype(np.int32), float(b["d"]), b["vox"], frac=src["frac"])
            assert np.array_equal(b["gt"], gt)
        gids, pids, gsz, psz, pg, pp, inter = C.overlaps(gt, pr)
        iou = inter / (gsz[pg] + psz[pp] - inter) if len(pg) else np.zeros(0)
        mg, mp, mv = C.hungarian(gids, pids, pg, pp, iou)
        k = mv >= 0.5
        mg, mp, mv = mg[k], mp[k], mv[k]
        tp7 = int((mv >= 0.7).sum())
        og, op = ndi.find_objects(gt), ndi.find_objects(pr)
        samp = vox / vox[1]
        res = np.zeros((len(mg), 2))
        for i, (a, b) in enumerate(zip(mg, mp)):
            ga, pb = int(gids[a]), int(pids[b])
            res[i] = pair_metrics(gt, pr, ga, pb, og[ga - 1], op[pb - 1], samp)
        os.makedirs(os.path.dirname(fo), exist_ok=True)
        np.savez_compressed(fo + f".{os.getpid()}.tmp.npz", gt_id=gids[mg].astype(np.int64), iou=mv, assd_px=res[:, 0], hd95_px=res[:, 1],
                            vox_xy=vox[1], d=d, tp5=len(mg), tp7=tp7, n_gt=len(gids), n_pred=len(pids))
        os.replace(fo + f".{os.getpid()}.tmp.npz", fo)
        return "ok"
    except Exception:
        return f"ERR {tg} {key} " + traceback.format_exc()


def build_jobs():
    T, runs = tags()
    gtsrc = runs["gt_S1_s0"]
    jobs = []
    keys = sorted(f[:-4] for f in os.listdir(f"{gtsrc}/masks") if f.split("__")[0] not in C.DROP)
    for tg in T:
        for k in keys:
            src = dict(gtnpz=f"{gtsrc}/masks/{k}.npz")
            nm = tg.split("_")[1]
            if nm in BASELINES:
                d, fld, fr, _ = BASELINES[nm]
                src.update(kind="bl", path=f"{d}/masks/{k}.npz", field=fld, frac=fr)
            else:
                src.update(kind="cache", path=f"{C.OUT}/cache/{tg}/{k}.npz")
            jobs.append((tg, k, src))
    jobs.sort(key=lambda j: -os.path.getsize(j[2]["gtnpz"]))
    return jobs, T, keys


# ---------------------------------------------------------------- 汇总
METRICS = ("assd_px", "hd95_px", "assd_rel", "hd95_rel", "assd_um", "hd95_um")


def load(tg, k):
    z = np.load(f"{OUT}/per/{tg}/{'__'.join(k)}.npz")
    out = {f: z[f] for f in ("gt_id", "iou", "assd_px", "hd95_px")}
    d, vxy = float(z["d"]), float(z["vox_xy"])
    out.update(assd_rel=z["assd_px"] / d, hd95_rel=z["hd95_px"] / d, assd_um=z["assd_px"] * vxy, hd95_um=z["hd95_px"] * vxy)
    out.update(tp5=int(z["tp5"]), tp7=int(z["tp7"]))
    return out


def ds_stats(P, keys, ds, common=None):
    """P: {key: per-volume dict}；common: {key: set(gt_id)} 时只取共同匹配的核。返回该套 pooled 的指标。"""
    cols = {f: [] for f in METRICS + ("iou",)}
    for k in keys:
        x = P[k]
        sel = np.ones(len(x["gt_id"]), bool) if common is None else np.isin(x["gt_id"], list(common[k]))
        for f in cols:
            cols[f].append(x[f][sel])
    cols = {f: np.concatenate(v) for f, v in cols.items()}
    n = len(cols["iou"])
    r = {"n_pairs": n, "iou_median": float(np.median(cols["iou"])) if n else None}
    for f in METRICS:
        if f.endswith("_um") and ds in NO_UM:
            continue
        r[f + "_mean"] = float(cols[f].mean()) if n else None
        r[f + "_median"] = float(np.median(cols[f])) if n else None
    return r


def macro(per_ds, dl, field):
    v = [per_ds[d][field] for d in dl if per_ds[d].get(field) is not None]
    return float(np.mean(v)) if v else None


def aggregate():
    T, runs = tags()
    keys_all = sorted(tuple(os.path.basename(f)[:-4].split("__", 1)) for f in glob.glob(f"{OUT}/per/gt_S1_s0/*.npz"))
    P = {tg: {k: load(tg, k) for k in keys_all} for tg in T}
    # 一致性：tp5/tp7 对已存计数
    chk = {}
    for tg in T:
        nm = tg.split("_")[1]
        bad = []
        if nm in BASELINES:
            d, _, fr, th = BASELINES[nm]
            rows = {(x["ds"], x["id"]): x for x in json.load(open(f"{d}/rows.json")) if x["thr"] == th and x["frac"] == fr}
            for k in keys_all:
                if (P[tg][k]["tp5"], P[tg][k]["tp7"]) != (rows[k]["tp5"], rows[k]["tp7"]):
                    bad.append(["__".join(k), [P[tg][k]["tp5"], P[tg][k]["tp7"]], [rows[k]["tp5"], rows[k]["tp7"]]])
        else:
            for k in keys_all:
                c = json.load(open(f"{C.OUT}/per/{tg}/{'__'.join(k)}.json"))["counts"]
                if (P[tg][k]["tp5"], P[tg][k]["tp7"]) != (c["tp5"], c["tp7"]):
                    bad.append(["__".join(k), [P[tg][k]["tp5"], P[tg][k]["tp7"]], [c["tp5"], c["tp7"]]])
        chk[tg] = dict(n=len(keys_all), mismatched=bad)
    subs = C.subsets()
    seeds = [s for s in C.SEEDS if f"gt_S1_{s}" in P and f"gt_2D_{s}" in P]
    others = {"base": "gt_base_s0", "nnunet": "gt_nnunet_s0", "stardist": "gt_stardist_s0"}
    res = {"seeds": seeds, "consistency": chk, "no_um_sets": sorted(NO_UM), "subsets": {}}
    flds = ["iou_median"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "median")]
    for sname in ("all", "C1"):
        sub = subs[sname]
        kk = [k for k in keys_all if sub is None or k in sub]
        dss = sorted({k[0] for k in kk})
        groups = {"13 套" if sname == "all" else f"C1 {len(dss)} 套": dss,
                  "12 套（去 mesospim）" if sname == "all" else f"C1 去 mesospim": [d for d in dss if d != "mesospim"]}
        S = {}
        for mode in ("all_matched", "common"):
            ent = {}
            # 逐模型逐种子的每套统计
            def per_ds_for(tg, common=None):
                return {d: ds_stats(P[tg], [k for k in kk if k[0] == d], d, common) for d in dss}

            def common_ids(t1, t2):
                return {k: set(P[t1][k]["gt_id"].tolist()) & set(P[t2][k]["gt_id"].tolist()) for k in kk}

            for gname, dl in groups.items():
                g = {}
                um_sets = [d for d in dl if d not in NO_UM]
                g["um_sets"] = um_sets
                # S1 对二维
                s1v, s2v, diffs = [], [], {f: [] for f in flds}
                for s in seeds:
                    cm = common_ids(f"gt_S1_{s}", f"gt_2D_{s}") if mode == "common" else None
                    a, b = per_ds_for(f"gt_S1_{s}", cm), per_ds_for(f"gt_2D_{s}", cm)
                    ma = {f: macro(a, dl, f) for f in flds}; mb = {f: macro(b, dl, f) for f in flds}
                    s1v.append(ma); s2v.append(mb)
                    for f in flds:
                        if ma[f] is not None and mb[f] is not None:
                            diffs[f].append(ma[f] - mb[f])
                    if mode == "all_matched" and gname.startswith(("13", "C1 ")) and "per_ds" not in g:
                        pass
                g["S1"] = {f: float(np.mean([x[f] for x in s1v if x[f] is not None])) for f in flds}
                g["2D"] = {f: float(np.mean([x[f] for x in s2v if x[f] is not None])) for f in flds}
                cmp = {}
                for f in flds:
                    v = diffs[f]
                    mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
                    good = mu > 0 if f == "iou_median" else mu < 0
                    cmp[f] = dict(seeds=v, mean=mu, sd=sd, confirmed=bool(abs(mu) > 2 * sd and good))
                g["S1_minus_2D"] = cmp
                for on, otg in others.items():
                    vals = []
                    for s in seeds:
                        cm = common_ids(f"gt_S1_{s}", otg) if mode == "common" else None
                        a, b = per_ds_for(f"gt_S1_{s}", cm), per_ds_for(otg, cm)
                        vals.append(({f: macro(a, dl, f) for f in flds}, {f: macro(b, dl, f) for f in flds}))
                    g[on] = {f: float(np.mean([y[f] for _, y in vals])) for f in flds}
                    g[f"S1_minus_{on}"] = {f: float(np.mean([x[f] - y[f] for x, y in vals])) for f in flds}
                if mode == "all_matched":
                    g["per_ds"] = {}
                    for d in dl:
                        row = {}
                        for nm, tl in (("S1", [f"gt_S1_{s}" for s in seeds]), ("2D", [f"gt_2D_{s}" for s in seeds]), ("base", ["gt_base_s0"]),
                                       ("nnunet", ["gt_nnunet_s0"]), ("stardist", ["gt_stardist_s0"])):
                            st = [ds_stats(P[t], [k for k in kk if k[0] == d], d) for t in tl]
                            row[nm] = {f: (float(np.mean([x[f] for x in st])) if st[0].get(f) is not None else None) for f in ("n_pairs", "iou_median", "assd_rel_mean", "hd95_rel_mean", "assd_um_mean", "hd95_um_mean")}
                        g["per_ds"][d] = row
                ent[gname] = g
            S[mode] = ent
        res["subsets"][sname] = S
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
    report(res)


def report(R):
    L = []; w = L.append
    w("# 直接边界指标（审稿 S3，事后分析，只报告）\n")
    w(f"口径见 ../BOUNDARY.md。种子 {', '.join(R['seeds'])}；没有物理体素、不报 µm 的套：{', '.join(R['no_um_sets'])}。"
      "ASSD、HD95 是每套匹配核对（IoU ≥ 0.5）的均值，IoU 是中位数，都是按套 pooled 后再宏平均；rel = 像素口径除以真值核直径中位数。\n")
    w("## 核对（tp5、tp7 对已存计数）\n")
    for tg, v in R["consistency"].items():
        w(f"- {tg}：{v['n'] - len(v['mismatched'])}/{v['n']} 个体积相同" + (f"，不同：{v['mismatched'][:3]}" if v["mismatched"] else ""))
    show = ("iou_median", "assd_rel_mean", "hd95_rel_mean", "assd_px_mean", "hd95_px_mean", "assd_um_mean", "hd95_um_mean", "assd_um_median", "hd95_um_median")
    for sname, S in R["subsets"].items():
        for mode, ent in S.items():
            for gname, g in ent.items():
                w(f"\n## {sname} · {'全部匹配核对' if mode == 'all_matched' else '共同匹配（敏感性）'} · {gname}（µm 套：{', '.join(g['um_sets'])}）\n")
                w("| 指标 | S1 | 二维 | 原版 | nnU-Net | StarDist | S1−二维（SD；确认） | S1−原版 | S1−nnU-Net | S1−StarDist |\n|---|---|---|---|---|---|---|---|---|---|")
                for f in show:
                    c = g["S1_minus_2D"][f]
                    w(f"| {f} | {g['S1'][f]:.4f} | {g['2D'][f]:.4f} | {g['base'][f]:.4f} | {g['nnunet'][f]:.4f} | {g['stardist'][f]:.4f} | "
                      f"{c['mean']:+.4f}（{c['sd']:.4f}；{'是' if c['confirmed'] else '否'}） | {g['S1_minus_base'][f]:+.4f} | {g['S1_minus_nnunet'][f]:+.4f} | {g['S1_minus_stardist'][f]:+.4f} |")
                if "per_ds" in g and gname.startswith("13"):
                    w("\n逐套（S1 / 二维 / 原版 / nnU-Net / StarDist）：IoU 中位数；ASSD rel；HD95 µm\n")
                    w("| 套 | 匹配对数 S1 | IoU 中位数 | ASSD rel | HD95 µm |\n|---|---|---|---|---|")
                    for d, row in g["per_ds"].items():
                        j = lambda f: " / ".join("—" if row[m][f] is None else f"{row[m][f]:.3f}" for m in ("S1", "2D", "base", "nnunet", "stardist"))
                        w(f"| {d} | {row['S1']['n_pairs']:.0f} | {j('iou_median')} | {j('assd_rel_mean')} | {j('hd95_um_mean')} |")
    open(f"{OUT}/report.md", "w").write("\n".join(L))
    print("写入", f"{OUT}/report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--agg_only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not a.agg_only:
        jobs, T, keys = build_jobs()
        if a.limit:
            jobs = jobs[-a.limit:]
        print(len(T), "个模型标签", len(keys), "个体积", len(jobs), "个作业", flush=True)
        with Pool(a.workers, maxtasksperchild=8) as p:
            for i, r in enumerate(p.imap_unordered(job, jobs)):
                if r.startswith("ERR"):
                    print(r, flush=True)
                if i % 200 == 0:
                    print(i, flush=True)
        if a.limit:
            return
    aggregate()


if __name__ == "__main__":
    main()
