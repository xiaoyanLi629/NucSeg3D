"""汇总：A1（误差分解 + z 向切碎 + 分层）、A2（种子一致度 → 错核）、S4（F1–IoU 曲线）、C3（估直径的代价）、C4（规则 T 对事后最优阈值），
外加复现核对（逐体积计数对 rows.json、宏平均对 test_r6 / s5_stats 的已存结果）。输出 results.json 与 PNG 图。
种子：S1 与二维取两者都已完成的种子（同种子配对），原版只有一个模型；五种子齐后重跑即为正式版。
用法：python aggregate.py"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = C.OUT
R6 = f"{_NSP_SCRATCH}/r6"
BINS = {
    "relvol": ([0, 0.5, 0.8, 1.25, 2.0, np.inf], ["<0.5", "0.5–0.8", "0.8–1.25", "1.25–2", "≥2"]),
    "depth": ([0, 0.2, 0.4, 0.6, 0.8, 1.0001], ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"]),
    "zxy": ([0, 1.5, 3, 6, np.inf], ["<1.5", "1.5–3", "3–6", "≥6"]),
}
MIN_BIN = 20          # 分层的按数据集宏平均：某套在该层至少 20 个真值核才计入


def load_per():
    per = defaultdict(dict)
    for f in glob.glob(f"{OUT}/per/*/*.json"):
        x = json.load(open(f))
        per[x["tag"]][(x["ds"], x["id"])] = x
    return per


def seeds_of(per, proto):
    av = {nm: {t.split("_")[2] for t in per if t.startswith(f"{proto}_{nm}_")} for nm in ("S1", "2D")}
    full = {nm: [s for s in C.SEEDS if s in av[nm] and len(per[f"{proto}_{nm}_{s}"]) == len(per[f"{proto}_base_s0"])] for nm in av}
    matched = [s for s in C.SEEDS if s in full["S1"] and s in full["2D"]]
    return matched, full


def dss_of(keys):
    return sorted({k[0] for k in keys})


def macro_tags(dss, subset_name):
    if subset_name == "all":
        return {"13 套": dss, "12 套（去 mesospim）": [d for d in dss if d != "mesospim"]}
    return {f"{subset_name} {len(dss)} 套": dss, f"{subset_name} 去 mesospim {sum(d != 'mesospim' for d in dss)} 套": [d for d in dss if d != "mesospim"]}


def f1_from(tp, n):
    return 2 * tp / max(n, 1)


def ds_score(recs, keys, proto, getter):
    """getter(rec) -> (tp, n)；gt 协议 pooled，完整流程逐体积 F1 平均（与 test_r6.score 相同）。"""
    if proto == "fp":
        return float(np.mean([f1_from(*getter(recs[k])) for k in keys]))
    tp = sum(getter(recs[k])[0] for k in keys); n = sum(getter(recs[k])[1] for k in keys)
    return f1_from(tp, n)


def nsum(x):
    return x["counts"]["n_gt"] + x["counts"]["n_pred"]


# ---------------------------------------------------------------- S4 与复现核对
def s4(per, proto, seeds, keys, subs):
    res = {}
    for sname, sub in subs.items():
        kk = [k for k in keys if sub is None or k in sub]
        dss = dss_of(kk)
        out = {}
        for mtag, dl in macro_tags(dss, sname).items():
            ent = {}
            for nm, ss in (("S1", seeds), ("2D", seeds), ("base", ["s0"])):
                curves = []
                for s in ss:
                    R = per[f"{proto}_{nm}_{s}"]
                    c = []
                    for ti in range(len(C.IOU_GRID)):
                        c.append(float(np.mean([ds_score(R, [k for k in kk if k[0] == d], proto,
                                                          lambda x: (x["counts"]["tp_grid"][ti], nsum(x))) for d in dl])))
                    curves.append(c)
                curves = np.array(curves)
                auc = [float(np.trapz(c, C.IOU_GRID) / 0.45) for c in curves]
                ent[nm] = dict(curve=curves.mean(0).tolist(), curve_seeds=curves.tolist(), auc_norm=float(np.mean(auc)), auc_seeds=auc)
            d2 = [a - b for a, b in zip(ent["S1"]["auc_seeds"], ent["2D"]["auc_seeds"])]
            ent["S1_minus_2D_auc"] = dict(seeds=d2, mean=float(np.mean(d2)), sd=float(np.std(d2, ddof=1)) if len(d2) > 1 else None)
            ent["S1_minus_base_auc"] = float(ent["S1"]["auc_norm"] - ent["base"]["auc_norm"])
            ent["S1_minus_2D_curve"] = (np.array(ent["S1"]["curve_seeds"]) - np.array(ent["2D"]["curve_seeds"])).mean(0).tolist()
            out[mtag] = ent
        res[sname] = out
    return res


def consistency(per, proto, seeds, keys):
    """逐体积：我算的计数与 rows.json (t, 0.15) 逐项相同的比例；宏平均：与 test_r6_{proto}_basefixed.json 比较（种子 0/1000/2000）。"""
    out = {"per_volume": {}, "macro_vs_test_r6": {}}
    for tg, R in per.items():
        if not tg.startswith(proto + "_"):
            continue
        n = len(R); bad = [k for k, x in R.items() if x["diffs"]]
        f1bad = [k for k, x in R.items() if any(f in x["diffs"] for f in ("tp5", "tp7", "n_gt", "n_pred"))]
        rec = [k for k, x in R.items() if x["recomputed"]]
        out["per_volume"][tg] = dict(n=n, all_fields_equal=n - len(bad), f1_equal=n - len(f1bad), recomputed=len(rec),
                                     recomputed_iso=sum(R[k]["iso"] for k in rec),
                                     f1_diff_keys=["__".join(k) for k in f1bad],
                                     other_diff=[dict(key="__".join(k), diffs=R[k]["diffs"]) for k in bad][:20])
    ref = json.load(open(f"{R6}/test_r6_{proto}_basefixed.json"))
    s3 = [s for s in ("s0", "s1000", "s2000") if s in seeds]
    if len(s3) == 3:
        dss = dss_of(keys)
        for lab, ki in (("F1@0.5", 0), ("F1@0.7", 1)):
            for mtag, mref in (("13 套", "13 套"), ("12 套（去 mesospim）", "去 mesospim 12 套")):
                dl = dss if mtag == "13 套" else [d for d in dss if d != "mesospim"]
                mine = {}
                for nm, ss, rk in (("S1", s3, "S1_seeds"), ("2D", s3, "two_d_seeds"), ("base", ["s0"], "base")):
                    v = [float(np.mean([ds_score(per[f"{proto}_{nm}_{s}"], [k for k in keys if k[0] == d], proto,
                                                 lambda x: (x["counts"]["tp_grid"][0 if ki == 0 else 4], nsum(x))) for d in dl])) for s in ss]
                    r = ref["macro"][lab][mref][rk]
                    r = r if isinstance(r, list) else [r]
                    mine[nm] = dict(mine=v, ref=r, max_abs_diff=float(np.max(np.abs(np.array(v) - np.array(r)))))
                out["macro_vs_test_r6"][f"{lab} {mtag}"] = mine
    return out


# ---------------------------------------------------------------- A1
def a1_counts(per, proto, seeds, keys, subs):
    comp = ("miss", "merge", "split", "boundary", "false")
    res = {}
    for sname, sub in subs.items():
        kk = [k for k in keys if sub is None or k in sub]
        dss = dss_of(kk)
        out = {}
        for mtag, dl in macro_tags(dss, sname).items():
            ent = {}
            for nm, ss in (("S1", seeds), ("2D", seeds), ("base", ["s0"])):
                seedv = []
                for s in ss:
                    R = per[f"{proto}_{nm}_{s}"]
                    gl = json.load(open(f"{OUT}/_gtlevel_{proto}_{nm}_{s}.json")) if os.path.exists(f"{OUT}/_gtlevel_{proto}_{nm}_{s}.json") else {}
                    per_ds = []
                    for d in dl:
                        ks = [k for k in kk if k[0] == d]
                        S = defaultdict(float)
                        for k in ks:
                            for f, v in R[k]["counts"].items():
                                if f != "tp_grid":
                                    S[f] += v
                            g = gl.get("__".join(k), {})
                            S["g_zmerge"] += g.get("zmerge", 0); S["g_xymerge"] += g.get("xymerge", 0)
                        den = max(S["n_gt"] + S["n_pred"], 1); ng = max(S["n_gt"], 1); nerr = max(S["n_gt"] - S["hit"], 1)
                        per_ds.append(dict(
                            one_minus_f1=1 - 2 * S["tp5"] / den,
                            decomp=dict(miss=S["miss"] / den, merge=(S["merge"] + S["fpc_merge"]) / den, split=(S["split"] + S["fpc_split"]) / den,
                                        boundary=(S["boundary"] + S["fpc_boundary"]) / den, false=S["fpc_false"] / den),
                            gt_rate=dict(miss=S["miss"] / ng, merge=S["merge"] / ng, split=S["split"] / ng, boundary=S["boundary"] / ng,
                                         zsplit=S["zsplit_err"] / ng, zsplit_any=S["zsplit_any"] / ng,
                                         zmerge=S["g_zmerge"] / ng, xymerge=S["g_xymerge"] / ng),
                            zsplit_share_of_gt_errors=S["zsplit_err"] / nerr,
                            zmerge_pair_share=S["merge_z"] / max(S["merge_z"] + S["merge_xy"], 1)))
                    mac = dict(one_minus_f1=float(np.mean([x["one_minus_f1"] for x in per_ds])),
                               decomp={c: float(np.mean([x["decomp"][c] for x in per_ds])) for c in comp},
                               gt_rate={c: float(np.mean([x["gt_rate"][c] for x in per_ds])) for c in per_ds[0]["gt_rate"]},
                               zsplit_share_of_gt_errors=float(np.mean([x["zsplit_share_of_gt_errors"] for x in per_ds])),
                               zmerge_pair_share=float(np.mean([x["zmerge_pair_share"] for x in per_ds])),
                               per_ds={d: x for d, x in zip(dl, per_ds)})
                    seedv.append(mac)
                avg = lambda f: float(np.mean([f(v) for v in seedv]))
                ent[nm] = dict(one_minus_f1=avg(lambda v: v["one_minus_f1"]),
                               decomp={c: avg(lambda v: v["decomp"][c]) for c in comp},
                               gt_rate={c: avg(lambda v: v["gt_rate"][c]) for c in seedv[0]["gt_rate"]},
                               zsplit_share_of_gt_errors=avg(lambda v: v["zsplit_share_of_gt_errors"]),
                               zmerge_pair_share=avg(lambda v: v["zmerge_pair_share"]),
                               per_ds={d: {c: float(np.mean([v["per_ds"][d]["gt_rate"][c] for v in seedv])) for c in seedv[0]["gt_rate"]}
                                       | {"d_" + c: float(np.mean([v["per_ds"][d]["decomp"][c] for v in seedv])) for c in comp}
                                       for d in dl},
                               seeds=[dict(decomp=v["decomp"], gt_rate=v["gt_rate"]) for v in seedv])
            diff = {}
            for c in list(ent["S1"]["decomp"]) + ["gt:" + c for c in ent["S1"]["gt_rate"]]:
                g = (lambda v, c=c: v["gt_rate"][c[3:]]) if c.startswith("gt:") else (lambda v, c=c: v["decomp"][c])
                ds_ = [g(a) - g(b) for a, b in zip(ent["S1"]["seeds"], ent["2D"]["seeds"])]
                diff[c] = dict(S1_minus_2D=float(np.mean(ds_)), sd=float(np.std(ds_, ddof=1)) if len(ds_) > 1 else None,
                               S1_minus_base=float(np.mean([g(a) for a in ent["S1"]["seeds"]]) - g(ent["base"]["seeds"][0])))
            ent["diff"] = diff
            out[mtag] = ent
        res[sname] = out
    return res


def gt_tables(per, proto, tags, keys):
    """逐真值核大表：协变量 + 每个标签的结果。同时写 _gtlevel_*.json（逐体积 z 向 / xy 向合并的真值核数）。"""
    cov = {}
    for k in keys:
        z = np.load(f"{OUT}/gtcov/{proto}/{'__'.join(k)}.npz")
        cov[k] = {f: z[f] for f in z.files}
    T = {}
    for tg in tags:
        gl = {}
        cols = defaultdict(list)
        for k in keys:
            z = np.load(f"{OUT}/cache/{tg}/{'__'.join(k)}.npz")
            assert np.array_equal(z["g_id"], cov[k]["g_id"]), (tg, k)
            cols["cls"].append(z["g_cls"]); cols["zsplit"].append(z["g_zsplit"]); cols["zmerge"].append(z["g_zmerge"])
            gl["__".join(k)] = dict(zmerge=int(z["g_zmerge"].sum()), xymerge=int(z["g_xymerge"].sum()))
        json.dump(gl, open(f"{OUT}/_gtlevel_{tg}.json", "w"))
        T[tg] = {c: np.concatenate(v) for c, v in cols.items()}
    base = dict(ds=np.concatenate([[k[0]] * len(cov[k]["g_id"]) for k in keys]),
                key=np.concatenate([["__".join(k)] * len(cov[k]["g_id"]) for k in keys]),
                **{f: np.concatenate([cov[k][f] for k in keys]) for f in ("vol_um3", "relvol", "depth", "crowd")},
                zxy=np.concatenate([[per[tags[0]][k]["ratio"]] * len(cov[k]["g_id"]) for k in keys]))
    return base, T


def strata(base, T, proto, seeds, subs, crowd_edges):
    bins = dict(BINS, crowd=(crowd_edges, [f"Q{i + 1}" for i in range(len(crowd_edges) - 1)]))
    outc = dict(err=lambda t: t["cls"] != 0, zsplit=lambda t: t["zsplit"] & (t["cls"] != 0), zmerge=lambda t: t["zmerge"],
                miss=lambda t: t["cls"] == 1, merge=lambda t: t["cls"] == 2, split=lambda t: t["cls"] == 3, boundary=lambda t: t["cls"] == 4)
    res = {}
    for sname, sub in subs.items():
        keep = np.array([(base["ds"][i], base["key"][i].split("__", 1)[1]) in sub for i in range(len(base["ds"]))]) if sub else np.ones(len(base["ds"]), bool)
        keep &= base["ds"] != "mesospim"
        out = {}
        for var, (edges, labs) in bins.items():
            x = base[var]
            bi = np.digitize(x, edges[1:-1]) if var != "crowd" else np.digitize(x, edges[1:-1])
            bi = np.where(np.isnan(x), -1, bi)
            ent = dict(labels=labs, edges=[float(e) if np.isfinite(e) else None for e in edges],
                       n=[int(((bi == b) & keep).sum()) for b in range(len(labs))])
            for nm, ss in (("S1", seeds), ("2D", seeds), ("base", ["s0"])):
                for oc, fn in outc.items():
                    pooled, macro = [], []
                    for b in range(len(labs)):
                        sel = keep & (bi == b)
                        vals = [fn(T[f"{proto}_{nm}_{s}"])[sel].mean() if sel.any() else np.nan for s in ss]
                        pooled.append(float(np.mean(vals)))
                        dsv = []
                        for d in np.unique(base["ds"][sel]):
                            sd = sel & (base["ds"] == d)
                            if sd.sum() >= MIN_BIN:
                                dsv.append(np.mean([fn(T[f"{proto}_{nm}_{s}"])[sd].mean() for s in ss]))
                        macro.append(float(np.mean(dsv)) if dsv else None)
                    ent.setdefault(nm, {})[oc] = dict(pooled=pooled, macro=macro)
            out[var] = ent
        res[sname] = out
    return res


# ---------------------------------------------------------------- A2
def auroc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); n1 = y.sum(); n0 = (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at(score, y, frac, rng):
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return None
    o = np.lexsort((rng.random(len(score)), -np.asarray(score)))
    k = int(math.ceil(frac * len(score)))
    return float(y[o[:k]].sum() / y.sum())


def a2(proto, subs):
    res = {}
    for nm in ("S1", "2D"):
        fs = sorted(glob.glob(f"{OUT}/a2/{proto}_{nm}/*.npz"))
        if not fs:
            continue
        D = []
        for f in fs:
            z = np.load(f)
            key = os.path.basename(f)[:-4]; ds, vid = key.split("__", 1)
            if ds in C.DROP:
                continue
            D.append((ds, vid, 1 - z["agree"], z["err"], str(z["seeds"])))
        seeds = sorted({x[4] for x in D})
        ent = {"seeds": seeds, "provisional": not any(len(s.split("_")) == 5 for s in seeds)}
        for sname, sub in subs.items():
            dd = [x for x in D if (sub is None or (x[0], x[1]) in sub) and x[0] != "mesospim"]
            sc = np.concatenate([x[2] for x in dd]); y = np.concatenate([x[3] for x in dd])
            rng = np.random.default_rng(0)
            per_ds = {}
            for d in sorted({x[0] for x in dd}):
                s_ = np.concatenate([x[2] for x in dd if x[0] == d]); y_ = np.concatenate([x[3] for x in dd if x[0] == d])
                per_ds[d] = dict(n=int(len(y_)), err_rate=float(y_.mean()) if len(y_) else None, auroc=auroc(s_, y_),
                                 recall10=recall_at(s_, y_, 0.1, rng), recall20=recall_at(s_, y_, 0.2, rng))
            mac = lambda f: float(np.mean([v[f] for v in per_ds.values() if v[f] is not None]))
            # 曲线：复查比例 → 召回
            o = np.lexsort((rng.random(len(sc)), -sc))
            cum = np.cumsum(y[o]) / max(y.sum(), 1)
            fr = np.linspace(0, 1, 51)
            rc = [float(cum[max(int(math.ceil(f * len(sc))) - 1, 0)]) if f > 0 else 0.0 for f in fr]
            ent[sname] = dict(n_pred=int(len(y)), err_rate=float(y.mean()), auroc_pooled=auroc(sc, y),
                              recall10_pooled=recall_at(sc, y, 0.1, rng), recall20_pooled=recall_at(sc, y, 0.2, rng),
                              auroc_macro=mac("auroc"), recall10_macro=mac("recall10"), recall20_macro=mac("recall20"),
                              review_curve=dict(frac=fr.tolist(), recall=rc), per_ds=per_ds)
        res[nm] = ent
    return res


# ---------------------------------------------------------------- C3
def c3(per, seeds, subs):
    res = {}
    fpk = sorted(set.intersection(*[set(per[f"fp_{nm}_{s}"]) for nm in ("S1", "2D") for s in seeds]))
    for nm in ("S1", "2D"):
        rows = []
        for k in fpk:
            v = []
            for s in seeds:
                a = per[f"fp_{nm}_{s}"][k]; b = per[f"gt_{nm}_{s}"].get(k)
                if b is None:
                    continue
                v.append(dict(f5_fp=f1_from(a["counts"]["tp5"], nsum(a)), f5_gt=f1_from(b["counts"]["tp5"], nsum(b)),
                              f7_fp=f1_from(a["counts"]["tp7"], nsum(a)), f7_gt=f1_from(b["counts"]["tp7"], nsum(b)),
                              ratio=a["d_used"] / a["d_gt"], t_fp=a["t"], t_gt=b["t"], dgt_fp=a["d_gt"], dgt_gt=b["d_gt"]))
            if len(v) != len(seeds):
                continue
            rows.append(dict(ds=k[0], id=k[1], **{f: float(np.mean([x[f] for x in v])) for f in v[0]},
                             ratio_seeds=[x["ratio"] for x in v]))
        ent = {"n_vol": len(rows), "per_vol": rows}
        for sname, sub in subs.items():
            rr = [r for r in rows if sub is None or (r["ds"], r["id"]) in sub]
            dss = sorted({r["ds"] for r in rr})
            out = {}
            for mtag, dl in macro_tags(dss, sname).items():
                g = lambda f: float(np.mean([np.mean([r[f] for r in rr if r["ds"] == d]) for d in dl]))
                out[mtag] = dict(F1_05_gtdiam=g("f5_gt"), F1_05_estdiam=g("f5_fp"), cost_05=g("f5_gt") - g("f5_fp"),
                                 F1_07_gtdiam=g("f7_gt"), F1_07_estdiam=g("f7_fp"), cost_07=g("f7_gt") - g("f7_fp"))
            lr = np.array([math.log2(r["ratio"]) for r in rr])
            from scipy.stats import spearmanr
            cost = np.array([r["f5_gt"] - r["f5_fp"] for r in rr])
            out["diam_error"] = dict(median_ratio=float(2 ** np.median(lr)), median_abs_pct=float(np.median(np.abs(2 ** lr - 1)) * 100),
                                     within10=float(np.mean(np.abs(2 ** lr - 1) <= 0.10)), within25=float(np.mean(np.abs(2 ** lr - 1) <= 0.25)),
                                     q05_q95_ratio=[float(2 ** np.percentile(lr, 5)), float(2 ** np.percentile(lr, 95))],
                                     spearman_abslog_vs_cost05=[float(x) for x in spearmanr(np.abs(lr), cost)] if len(rr) > 3 else None,
                                     per_ds_median_ratio={d: float(2 ** np.median([math.log2(r["ratio"]) for r in rr if r["ds"] == d])) for d in dss})
            ent[sname] = out
        res[nm] = ent
    res["same_gt_check"] = dict(max_dgt_mismatch=float(max(abs(r["dgt_fp"] - r["dgt_gt"]) for nm in ("S1", "2D") for r in res[nm]["per_vol"])))
    return res


# ---------------------------------------------------------------- C4
def c4(per, runs, proto, seeds, keys, subs, base_t):
    res = {}
    for nm, ss in (("S1", seeds), ("2D", seeds), ("base", ["s0"])):
        recs = []
        for s in ss:
            tg = f"{proto}_{nm}_{s}"
            by = C.load_rows(runs[tg])
            R = {}
            for k in keys:
                vr = by[k]
                t = per[tg][k]["t"] if nm != "base" else base_t[proto][k]
                g = lambda th, fr: (vr[(th, fr)]["tp5"], vr[(th, fr)]["tp7"], vr[(th, fr)]["n_gt"] + vr[(th, fr)]["n_pred"])
                thrs = sorted({th for th, _ in vr})
                f5 = {th: f1_from(g(th, C.FRAC)[0], g(th, C.FRAC)[2]) for th in thrs}
                f7 = {th: f1_from(g(th, C.FRAC)[1], g(th, C.FRAC)[2]) for th in thrs}
                o5 = max(thrs, key=lambda th: (f5[th], th == -2.0)); o7 = max(thrs, key=lambda th: (f7[th], th == -2.0))
                allset = list(vr)
                o5b = max(allset, key=lambda q: f1_from(g(*q)[0], g(*q)[2])); o7b = max(allset, key=lambda q: f1_from(g(*q)[1], g(*q)[2]))
                R[k] = dict(t=t, o5=o5, o7=o7,
                            fixed=(g(-2.0, C.FRAC)[0], g(-2.0, C.FRAC)[1], g(-2.0, C.FRAC)[2]), ruleT=g(t, C.FRAC),
                            orc5=g(o5, C.FRAC), orc7=g(o7, C.FRAC), orcb5=g(*o5b), orcb7=g(*o7b),
                            same5=bool(f5[t] == f5[o5]), same7=bool(f7[t] == f7[o7]))
            recs.append(R)
        ent = {}
        for sname, sub in subs.items():
            kk = [k for k in keys if sub is None or k in sub]
            dss = dss_of(kk)
            out = {}
            for mtag, dl in macro_tags(dss, sname).items():
                sc = {}
                for lab, col, idx in (("fixed_cp-2", "fixed", None), ("rule_T", "ruleT", None), ("oracle_thr", "orc", None), ("oracle_thr_frac", "orcb", None)):
                    for mi, ml in ((0, "F1@0.5"), (1, "F1@0.7")):
                        c = col + ("5" if mi == 0 else "7") if col in ("orc", "orcb") else col
                        v = [float(np.mean([ds_score(R, [k for k in kk if k[0] == d], proto, lambda x, c=c, mi=mi: (x[c][mi], x[c][2])) for d in dl])) for R in recs]
                        sc[f"{lab} {ml}"] = float(np.mean(v))
                sc["gap_oracle_minus_ruleT F1@0.5"] = sc["oracle_thr F1@0.5"] - sc["rule_T F1@0.5"]
                sc["gap_oracle_minus_ruleT F1@0.7"] = sc["oracle_thr F1@0.7"] - sc["rule_T F1@0.7"]
                sc["gap_oracle_minus_fixed F1@0.5"] = sc["oracle_thr F1@0.5"] - sc["fixed_cp-2 F1@0.5"]
                sc["gap_oracle_minus_fixed F1@0.7"] = sc["oracle_thr F1@0.7"] - sc["fixed_cp-2 F1@0.7"]
                out[mtag] = sc
            kk_ = [k for k in kk if k[0] != "mesospim"]
            out["share_ruleT_equals_oracle_F1@0.5"] = float(np.mean([R[k]["same5"] for R in recs for k in kk_]))
            out["share_ruleT_equals_oracle_F1@0.7"] = float(np.mean([R[k]["same7"] for R in recs for k in kk_]))
            out["oracle_thr_hist_F1@0.5"] = {str(t): int(sum(R[k]["o5"] == t for R in recs for k in kk_)) for t in (0.0, -1.0, -2.0, -3.0, -4.0, -6.0)}
            out["ruleT_thr_hist"] = {str(t): int(sum(R[k]["t"] == t for R in recs for k in kk_)) for t in (0.0, -1.0, -2.0, -3.0, -4.0, -6.0)}
            gaps = {}
            for d in dss:
                ks = [k for k in kk if k[0] == d]
                gaps[d] = float(np.mean([ds_score(R, ks, proto, lambda x: (x["orc5"][0], x["orc5"][2])) - ds_score(R, ks, proto, lambda x: (x["ruleT"][0], x["ruleT"][2])) for R in recs]))
            out["per_ds_gap_F1@0.5"] = gaps
            ent[sname] = out
        res[nm] = ent
    return res


def base_rule_t(runs):
    """原版 + 规则 T 的阈值（C4 用；主对照的原版是固定 cp−2）。缓存到 _base_ruleT.json。"""
    f = f"{OUT}/_base_ruleT.json"
    if os.path.exists(f):
        return {p: {tuple(k.split("__", 1)): v for k, v in d.items()} for p, d in json.load(open(f)).items()}
    from multiprocessing import Pool
    out = {}
    for proto in ("gt", "fp"):
        d = runs[f"{proto}_base_s0"]
        ks = [fn[:-4] for fn in sorted(os.listdir(f"{d}/masks")) if fn.split("__")[0] not in C.DROP]
        with Pool(16) as p:
            ts = p.map(_bt, [(d, k) for k in ks])
        out[proto] = dict(zip(ks, ts))
    json.dump(out, open(f, "w"))
    return {p: {tuple(k.split("__", 1)): v for k, v in d.items()} for p, d in out.items()}


def _bt(a):
    d, k = a
    ds, vid = k.split("__", 1)
    cp = np.load(f"{d}/masks/{k}.npz")["cp"].astype(np.float32)
    return C.rule_t(cp, C.load_rows(d)[(ds, vid)])[0]


# ---------------------------------------------------------------- 主程序
def main():
    runs = json.load(open(f"{OUT}/runs.json"))
    per = load_per()
    subs = C.subsets()
    R = {"generated": __import__("time").strftime("%Y-%m-%d %H:%M"), "runs": runs}
    base_t = base_rule_t(runs)
    crowd_edges = None
    for proto in ("gt", "fp"):
        seeds, full = seeds_of(per, proto)
        tags = [f"{proto}_{nm}_{s}" for nm in ("S1", "2D") for s in seeds] + [f"{proto}_base_s0"]
        keys = sorted(set.intersection(*[set(per[t]) for t in tags]))
        print(proto, "种子", seeds, "（完整的：", full, "）", len(keys), "个体积", flush=True)
        P = dict(seeds=seeds, seeds_complete=full, n_vol=len(keys))
        P["consistency"] = consistency(per, proto, seeds, keys)
        P["S4"] = s4(per, proto, seeds, keys, subs)
        base, T = gt_tables(per, proto, tags, keys)
        if crowd_edges is None:       # 用 gt 协议全部真值核（去 mesospim）的五分位数定拥挤度的分层边界（只看数据，不看结果）
            x = base["crowd"][(base["ds"] != "mesospim") & ~np.isnan(base["crowd"])]
            crowd_edges = [0.0] + [float(np.percentile(x, q)) for q in (20, 40, 60, 80)] + [np.inf]
        P["A1"] = a1_counts(per, proto, seeds, keys, subs)
        P["A1_strata"] = strata(base, T, proto, seeds, subs, crowd_edges)
        P["A1_strata_crowd_edges"] = [float(e) if np.isfinite(e) else None for e in crowd_edges]
        P["A2"] = a2(proto, subs)
        P["C4"] = c4(per, runs, proto, seeds, keys, subs, base_t)
        R[proto] = P
    R["C3"] = c3(per, R["fp"]["seeds"], subs)
    json.dump(R, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("写入", f"{OUT}/results.json")


if __name__ == "__main__":
    main()
