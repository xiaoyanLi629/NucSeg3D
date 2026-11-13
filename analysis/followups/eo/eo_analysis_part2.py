"""（并行运行用：eo_analysis.py 第 2、3 部分的逐字拷贝，内容未改；主脚本仍在跑第 1 部分）EO 对照分析（第三轮预审 SF2；口径按 DESIGN_EO.md 第 5 节预先写定，md5 fb9507e7…，AGENTS.md 331ec39 登记；只报告，CPU）。
1）复用 r6/s5_stats.main（不改它），参数与正式五种子表相同（test fp|gt，五种子，--base_fixed），把它读的两列换成：
    A 解码器的贡献：“S1”= S1，“二维”= EO（同种子）；B 编码器再训练的贡献：“S1”= EO，“二维”= 二维（同种子）。全部 13 套与 C1 子集。
2）按视图分层（单视图 3 套 cellos、parhyale、mesospim；三视图 10 套；另给 12 套与 13 套），S1−EO、EO−二维、S1−二维 的五种子配对均值 ± SD 与确认判定
   （均值 > 2×SD 且 ≥ 0.01），以及分层误差分解（切碎、合并、漏检、边界、误检、z 向合并占比；五种子平均）。同 m1/m1_strat.py 的写法。
3）验证集参照（不检验）：EO、S1、二维的五种子验证集 gt 协议 12/13 套宏平均（规则 T）。
种子 1000 的测试集 gt 用重跑 EO_tGT_s1000_r2（9d487c18）；原运行 d22bc567 在 18/227 被 SIGTERM 终止，不用。
输出 $NUCSEG3D_SCRATCH/nc/gpu/eo/eo_A_{fp,gt}[_C1].json、eo_B_*.json、eo_strat.json、eo_val.json、eo_analysis.log。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, os, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import s5_stats as S5  # noqa: E402
from test_r6 import recs, score  # noqa: E402
RES = f"{_NSP_ROOT}/results"; O = f"{_NSP_SCRATCH}/nc/gpu/eo"
ST = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]; V1 = ("cellos", "parhyale", "mesospim")
orig_dirs = S5.dirs
KEY = {("gt", s): ("EO_tGT_s1000_r2" if s == "s1000" else f"EO_tGT_{s}") for s in SEEDS}
KEY.update({("fp", s): f"EO_tFP_{s}" for s in SEEDS}); KEY.update({("val", s): f"EO_v_{s}" for s in SEEDS})


def eo_dirs(mode):
    return [glob.glob(f"{RES}/orx_{ST[KEY[(mode, s)]]['run']}_thr_eval")[0] for s in SEEDS]


def make(which):
    def dirs(split, mode, seeds):
        assert seeds == SEEDS
        D = orig_dirs(split, mode, seeds); E = eo_dirs("val" if split == "val" else mode)
        return {"S1": D["S1"], "2D": E, "base": D["base"]} if which == "A" else {"S1": E, "2D": D["2D"], "base": D["base"]}
    return dirs


def errs(d, keys):
    rc = recs(d); by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(d + "/rows.json"))}
    out = {}
    for ds in sorted({k[0] for k in keys}):
        L = [by[(*k, rc[k]["t"], 0.15)] for k in keys if k[0] == ds]; den = max(sum(x["n_gt"] + x["n_pred"] for x in L), 1)
        g = lambda c: sum(x[c] for x in L)
        out[ds] = dict(split=(g("split") + g("fpc_split")) / den, merge=(g("merge") + g("fpc_merge")) / den, miss=g("miss") / den,
                       boundary=(g("boundary") + g("fpc_boundary")) / den, false=g("fpc_false") / den,
                       zmerge=g("merge_z") / max(g("merge_z") + g("merge_xy"), 1))
    return out


strat = {}
for mode in ("fp", "gt"):
    D = orig_dirs("test", mode, SEEDS); M = {"S1": D["S1"], "EO": eo_dirs(mode), "2D": D["2D"]}
    R = {nm: [recs(d) for d in L] for nm, L in M.items()}
    common = set.intersection(*[set(r) for L in R.values() for r in L]); dss = sorted({k[0] for k in common})
    groups = {"单视图3套": [d for d in dss if d in V1], "三视图10套": [d for d in dss if d not in V1], "12套(去mesospim)": [d for d in dss if d != "mesospim"], "13套": dss}
    E = {nm: [errs(d, common) for d in L] for nm, L in M.items()}
    out = {}
    for k, lab in (("tp5", "F1@0.5"), ("tp7", "F1@0.7")):
        for g, gd in groups.items():
            per = {nm: [float(np.mean([score(r, sorted(x for x in common if x[0] == ds), mode, k) for ds in gd])) for r in L] for nm, L in R.items()}
            ent = {nm: (float(np.mean(v)), float(np.std(v, ddof=1))) for nm, v in per.items()}
            for a, b in (("S1", "EO"), ("EO", "2D"), ("S1", "2D")):
                dd = [x - y for x, y in zip(per[a], per[b])]; mu, sd = float(np.mean(dd)), float(np.std(dd, ddof=1))
                ent[f"{a}-{b}"] = dict(mean=mu, sd=sd, seeds=dd, confirmed=bool(mu > 2 * sd and mu >= 0.01), neg_sig=bool(-mu > 2 * sd and -mu >= 0.01))
            if k == "tp5":
                ent["err"] = {nm: {q: float(np.mean([np.mean([e[ds][q] for ds in gd]) for e in E[nm]])) for q in ("split", "merge", "miss", "boundary", "false", "zmerge")} for nm in E}
            out[f"{lab}|{g}"] = ent
            s = "  ".join(f"{p} {ent[p]['mean']:+.4f}±{ent[p]['sd']:.4f}{' ✓' if ent[p]['confirmed'] else (' (负，|均值|>2SD)' if ent[p]['neg_sig'] else '')}" for p in ("S1-EO", "EO-2D", "S1-2D"))
            print(f"{mode} {lab} {g}: S1 {ent['S1'][0]:.4f} EO {ent['EO'][0]:.4f} 2D {ent['2D'][0]:.4f} | {s}" +
                  (("  | 切碎 S1/EO/2D " + "/".join(f"{ent['err'][n]['split']:.3f}" for n in ("S1", "EO", "2D")) +
                    "  z向合并占比 " + "/".join(f"{ent['err'][n]['zmerge']:.2f}" for n in ("S1", "EO", "2D"))) if k == "tp5" else ""), flush=True)
    strat[mode] = out
json.dump(strat, open(f"{O}/eo_strat_p2.json", "w"), ensure_ascii=False, indent=1)
D = orig_dirs("val", "gt", SEEDS); M = {"S1": D["S1"], "EO": eo_dirs("val"), "2D": D["2D"]}
R = {nm: [recs(d) for d in L] for nm, L in M.items()}; common = set.intersection(*[set(r) for L in R.values() for r in L]); dss = sorted({k[0] for k in common})
val = {}
for k, lab in (("tp5", "F1@0.5"), ("tp7", "F1@0.7")):
    for g, gd in (("12套", [d for d in dss if d != "mesospim"]), ("13套", dss), ("三视图", [d for d in dss if d not in V1])):
        per = {nm: [float(np.mean([score(r, sorted(x for x in common if x[0] == ds), "gt", k) for ds in gd])) for r in L] for nm, L in R.items()}
        val[f"{lab}|{g}"] = {nm: (float(np.mean(v)), float(np.std(v, ddof=1)), v) for nm, v in per.items()}
        print(f"val {lab} {g}: " + "  ".join(f"{nm} {np.mean(v):.4f}±{np.std(v, ddof=1):.4f}" for nm, v in per.items()), flush=True)
json.dump(val, open(f"{O}/eo_val_p2.json", "w"), ensure_ascii=False, indent=1)
print("EO_PART2_DONE", flush=True)
