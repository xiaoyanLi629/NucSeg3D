"""M1 按视图分层重算（第二轮审稿 M1；只用已有 M1 结果，CPU，只报告）。
分层：单视图 3 套（cellos、parhyale、mesospim，视图规则下走单视图）、三视图 10 套（其余）、三视图 9 套（再去掉 … 实际三视图里没有 mesospim，所以另给“单视图去 mesospim 2 套”）与 12 套（全部去 mesospim）。
每个种子先在组内对数据集求宏平均（完整流程每套逐体积平均、gt 协议每套 pooled，与 test_r6.score 相同，规则 T），
再给 S1−置零版、置零版−二维、S1−二维 的五种子配对差均值 ± SD（ddof=1）与确认判定（均值 > 2×SD 且 ≥ 0.01；负向时同样报告 |均值| > 2×SD 是否成立）。
误差分解按组宏平均（五种子平均）：split、merge、miss、boundary、false。输出 m1/m1_strat.json 与打印。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
from s5_stats import dirs  # noqa: E402
from test_r6 import recs, score  # noqa: E402
RES = f"{_NSP_ROOT}/results"
ST = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]
V1 = ("cellos", "parhyale", "mesospim")


def zero_dirs(mode):
    pre = "tFP" if mode == "fp" else "tGT"
    return [glob.glob(RES + "/orx_" + ST["M1_" + pre + "_" + s]["run"] + "*_gx_thr_eval")[0] for s in SEEDS]


def errs(d, keys):
    rc = recs(d); by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(d + "/rows.json"))}
    out = {}
    for ds in sorted({k[0] for k in keys}):
        L = [by[(*k, rc[k]["t"], 0.15)] for k in keys if k[0] == ds]; den = max(sum(x["n_gt"] + x["n_pred"] for x in L), 1)
        g = lambda c: sum(x[c] for x in L)
        out[ds] = dict(split=(g("split") + g("fpc_split")) / den, merge=(g("merge") + g("fpc_merge")) / den, miss=g("miss") / den,
                       boundary=(g("boundary") + g("fpc_boundary")) / den, false=g("fpc_false") / den)
    return out


res = {}
for mode in ("fp", "gt"):
    D = dirs("test", mode, SEEDS); M = {"S1": D["S1"], "Z": zero_dirs(mode), "2D": D["2D"]}
    R = {nm: [recs(d) for d in L] for nm, L in M.items()}
    common = set.intersection(*[set(r) for L in R.values() for r in L])
    dss = sorted({k[0] for k in common})
    groups = {"单视图3套": [d for d in dss if d in V1], "单视图去mesospim2套": [d for d in ("cellos", "parhyale") if d in dss],
              "三视图10套": [d for d in dss if d not in V1], "12套(去mesospim)": [d for d in dss if d != "mesospim"], "13套": dss}
    E = {nm: [errs(d, common) for d in L] for nm, L in M.items()}
    out = {}
    for k, lab in (("tp5", "F1@0.5"), ("tp7", "F1@0.7")):
        for g, gd in groups.items():
            per = {nm: [float(np.mean([score(r, sorted(x for x in common if x[0] == ds), mode, k) for ds in gd])) for r in L] for nm, L in R.items()}
            ent = {nm: (float(np.mean(v)), float(np.std(v, ddof=1))) for nm, v in per.items()}
            for a, b in (("S1", "Z"), ("Z", "2D"), ("S1", "2D")):
                dd = [x - y for x, y in zip(per[a], per[b])]; mu, sd = float(np.mean(dd)), float(np.std(dd, ddof=1))
                ent[f"{a}-{b}"] = dict(mean=mu, sd=sd, seeds=dd, confirmed=bool(mu > 2 * sd and mu >= 0.01), sig_any=bool(abs(mu) > 2 * sd and abs(mu) >= 0.01))
            if k == "tp5":
                ent["err"] = {nm: {q: float(np.mean([np.mean([e[ds][q] for ds in gd]) for e in E[nm]])) for q in ("split", "merge", "miss", "boundary", "false")} for nm in E}
            out[f"{lab}|{g}"] = ent
            s = "  ".join(f"{p} {ent[p]['mean']:+.3f}±{ent[p]['sd']:.3f}{'✓' if ent[p]['confirmed'] else ('(负,显著)' if ent[p]['sig_any'] else '')}" for p in ("S1-Z", "Z-2D", "S1-2D"))
            print(f"{mode} {lab} {g}: S1 {ent['S1'][0]:.3f} Z {ent['Z'][0]:.3f} 2D {ent['2D'][0]:.3f} | {s}" +
                  (("  | split S1/Z/2D " + "/".join(f"{ent['err'][n]['split']:.3f}" for n in ("S1", "Z", "2D"))) if k == "tp5" else ""), flush=True)
    for ds in V1:
        if ds in dss:
            kk = sorted(x for x in common if x[0] == ds)
            v = {nm: float(np.mean([score(r, kk, mode, "tp5") for r in L])) for nm, L in R.items()}
            print(f"  {mode} {ds} F1@0.5 S1 {v['S1']:.3f} Z {v['Z']:.3f} 2D {v['2D']:.3f}  S1−Z {v['S1'] - v['Z']:+.3f}")
    res[mode] = out
json.dump(res, open(f"{_NSP_SCRATCH}/nc/gpu/m1/m1_strat.json", "w"), ensure_ascii=False, indent=1)
