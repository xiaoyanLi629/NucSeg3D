"""M1 消融统计（只报告；口径见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN_M1.md 第 3 节）。直接复用 r6/s5_stats.py 的 main（不改它）：
把 s5_stats.dirs 换成本文件的版本，让它的“S1 / 二维”两列分别是
  A 解码器的贡献：S1 对 置零版（同种子同权重）；B 编码器再训练的贡献：置零版 对 二维（同种子）。
其余（规则 T、gt pooled / 完整流程逐体积平均、12/13 套、五种子配对差均值与 SD、确认规则、五种子共用体积索引的配对自助 10000 次、三档判定、
单体积套逐核 McNemar、--subset 干净子集）全部是 s5_stats 原样。“原版”一列照旧是原版，不用于 M1 结论。
另算误差分解（规则 T 下每套合并后宏平均，五种子平均）：miss、merge、split、boundary、false 与 z 向合并占比。
输出 $NUCSEG3D_SCRATCH/nc/gpu/m1/m1_{A,B}_{fp,gt}[_C1].json、m1_err.json，并打印 s5_stats 的原样输出。用法：python m1_stats.py"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, json, os, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import s5_stats as S5  # noqa: E402
from test_r6 import recs  # noqa: E402
RES = f"{_NSP_ROOT}/results"
O = f"{_NSP_SCRATCH}/nc/gpu/m1"; os.makedirs(O, exist_ok=True)
ST = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]
orig_dirs = S5.dirs


def zero_dirs(mode):
    pre = "tFP" if mode == "fp" else "tGT"
    return [glob.glob(RES + "/orx_" + ST["M1_" + pre + "_" + s]["run"] + "*_gx_thr_eval")[0] for s in SEEDS]


def make(which):
    def dirs(split, mode, seeds):
        assert split == "test" and seeds == SEEDS
        D = orig_dirs(split, mode, seeds); Z = zero_dirs(mode)
        return {"S1": D["S1"], "2D": Z, "base": D["base"]} if which == "A" else {"S1": Z, "2D": D["2D"], "base": D["base"]}
    return dirs


for which, lab in (("A", "解码器的贡献：S1 − 置零版（输出里的“S1”= S1，“二维”= 置零版）"), ("B", "编码器再训练的贡献：置零版 − 二维（输出里的“S1”= 置零版，“二维”= 二维）")):
    for mode in ("fp", "gt"):
        for sub in (None, f"{_NSP_SHARED}/cellposeft_orx/audit/clean_test.json"):
            S5.dirs = make(which)
            sys.argv = ["s5_stats.py", "test", mode, "--seeds", *SEEDS, "--out", f"{O}/m1_{which}_{mode}{'_C1' if sub else ''}.json"] + (["--subset", sub] if sub else [])
            print(f"\n######## {lab}｜{'完整流程' if mode == 'fp' else 'gt 协议'}{'｜干净子集 C1' if sub else ''}", flush=True)
            S5.main()
S5.dirs = orig_dirs
# 误差分解
ERR = ("miss", "merge", "split", "boundary", "false")
def err_of(d):
    rc = recs(d); by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(d + "/rows.json"))}
    rows = [by[(*k, v["t"], 0.15)] for k, v in rc.items()]
    out = {}
    for ds in sorted({x["ds"] for x in rows}):
        L = [x for x in rows if x["ds"] == ds]; den = max(sum(x["n_gt"] + x["n_pred"] for x in L), 1)
        g = lambda c: sum(x[c] for x in L)
        out[ds] = dict(miss=g("miss") / den, merge=(g("merge") + g("fpc_merge")) / den, split=(g("split") + g("fpc_split")) / den,
                       boundary=(g("boundary") + g("fpc_boundary")) / den, false=g("fpc_false") / den,
                       zmerge=g("merge_z") / max(g("merge_z") + g("merge_xy"), 1))
    return {q: float(np.mean([v[q] for v in out.values()])) for q in ERR + ("zmerge",)}
E = {}
for mode in ("fp", "gt"):
    D = orig_dirs("test", mode, SEEDS)
    for nm, L in (("S1", D["S1"]), ("置零版", zero_dirs(mode)), ("二维", D["2D"])):
        es = [err_of(d) for d in L]
        E[f"{mode}_{nm}"] = {q: float(np.mean([e[q] for e in es])) for q in es[0]}
        print(f"误差分解 {mode} {nm}（五种子平均，13 套宏平均）：" + "，".join(f"{q} {E[f'{mode}_{nm}'][q]:.3f}" for q in E[f"{mode}_{nm}"]), flush=True)
json.dump(E, open(f"{O}/m1_err.json", "w"), ensure_ascii=False, indent=1)
