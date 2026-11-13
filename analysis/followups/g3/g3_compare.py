"""nc G3 测试集比较（CPU，只报告）。口径见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN_G3.md 第 2 节；汇总口径与投稿统计相同（r6/test_r6.score：
gt 协议每套 pooled，完整流程每套逐体积 F1 平均；再对数据集宏平均，13 套与 12 套（去 mesospim））。
模型：u-Segment3D(原版)、u-Segment3D(二维 s0)（各自验证集选中的变体，体积下限 0.15）；u-Segment3D 默认（官方默认 v0，事后敏感性分析，运行 G3dg_*/G3df_*）；直接推理：原版固定 cp−2、二维 s0 规则 T；
S1 规则 T 五种子（r6/s5_stats.dirs 取目录）均值 ± SD。所有模型取共同的体积。
配对差：共识重建 − 对应直接推理（原版对原版、二维对二维），逐套与宏平均，按体积在各套内分层自助 10000 次（种子 0）的 95% 区间。
误差类型：eval_gt 的分解（1−F1@0.5 = miss + merge + split + boundary + false，各项除以 n_gt+n_pred），每套合并后宏平均。
输出 $NUCSEG3D_SCRATCH/nc/gpu/g3/g3_compare.json 与 g3_compare.md。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, json, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
from s5_stats import dirs  # noqa: E402
from test_r6 import recs, score  # noqa: E402
R = f"{_NSP_ROOT}/results/"
O = f"{_NSP_SCRATCH}/nc/gpu/g3"
ST = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SEL = json.load(open(f"{_NSP_SHARED}/nc_gpu/G3/selected.json"))
NB = 10000
ERR = ("miss", "merge", "split", "boundary", "false")


def rows_of(d):
    return {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(d + "/rows.json"))}


def err_rec(x):
    return dict(miss=x["miss"], merge=x["merge"] + x["fpc_merge"], split=x["split"] + x["fpc_split"], boundary=x["boundary"] + x["fpc_boundary"], false=x["fpc_false"])


def direct(d, fixed):
    rc = recs(d, fixed=fixed); by = rows_of(d)
    return {k: dict(tp5=v["tp5"], tp7=v["tp7"], n=v["n"], **err_rec(by[(*k, v["t"], 0.15)])) for k, v in rc.items()}


def useg(key):
    d = glob.glob(R + "orx_" + ST[key]["run"] + "*")[0]
    return {(x["ds"], x["id"]): dict(tp5=x["tp5"], tp7=x["tp7"], n=x["n_gt"] + x["n_pred"], **err_rec(x), variant=x["variant"])
            for x in json.load(open(d + "/rows.json")) if x["thr"] == -2.0 and x["frac"] == 0.15}


def per_ds(Rm, keys, mode, k):
    out = {}
    for ds in sorted({a for a, _ in keys}):
        kk = [x for x in keys if x[0] == ds]
        out[ds] = score(Rm, kk, mode, k)
    return out


def errs(Rm, keys):
    out = {}
    for ds in sorted({a for a, _ in keys}):
        kk = [x for x in keys if x[0] == ds]; den = max(sum(Rm[x]["n"] for x in kk), 1)
        out[ds] = {e: sum(Rm[x][e] for x in kk) / den for e in ERR}
    return out


res = {"selected": {k: dict(variant=v["variant"], val_f1_05=v["f1_05"], val_f1_07=v["f1_07"]) for k, v in SEL.items()}}
md = []
for mode, tag in (("gt", "gt 协议"), ("fp", "完整流程")):
    print("MODE", mode, flush=True)
    D = dirs("test", mode, ["s0", "s1000", "s2000", "s3000", "s4000"])
    M = {"U_base": useg(f"G3t{'g' if mode == 'gt' else 'f'}_base"), "U_d2": useg(f"G3t{'g' if mode == 'gt' else 'f'}_d2"),
         "D_base": useg(f"G3d{'g' if mode == 'gt' else 'f'}_base"), "D_d2": useg(f"G3d{'g' if mode == 'gt' else 'f'}_d2"),
         "base": direct(D["base"][0], True), "d2": direct(D["2D"][0], False)}
    S1 = [direct(d, False) for d in D["S1"]]
    keys = sorted(set.intersection(*[set(v) for v in M.values()], *[set(s) for s in S1]))
    SC = lambda nm, kk, k: float(np.mean([score(s, kk, mode, k) for s in S1])) if nm == "S1" else score(M[nm], kk, mode, k)
    dss = sorted({a for a, _ in keys}); d12 = [d for d in dss if d != "mesospim"]
    out = {"n_vol": len(keys), "n_ds": len(dss), "macro": {}, "per_ds": {}, "pairs": {}, "err": {}}
    for k in ("tp5", "tp7"):
        for nm, Rm in M.items():
            p = per_ds(Rm, keys, mode, k); out["per_ds"].setdefault(nm, {})[k] = p
            out["macro"].setdefault(nm, {})[k] = (float(np.mean([p[d] for d in dss])), float(np.mean([p[d] for d in d12])))
        ps = [per_ds(s, keys, mode, k) for s in S1]
        m13 = [np.mean([p[d] for d in dss]) for p in ps]; m12 = [np.mean([p[d] for d in d12]) for p in ps]
        out["macro"].setdefault("S1", {})[k] = (float(np.mean(m13)), float(np.std(m13, ddof=1)), float(np.mean(m12)), float(np.std(m12, ddof=1)))
        out["per_ds"].setdefault("S1", {})[k] = {d: float(np.mean([p[d] for p in ps])) for d in dss}
    rng = np.random.default_rng(0); idx = {d: np.array([i for i, x in enumerate(keys) if x[0] == d]) for d in dss}
    BI = {d: rng.integers(0, len(idx[d]), (NB, len(idx[d]))) for d in dss}            # 每套一组重抽样下标，所有配对共用
    def arr(Rm, k):
        return np.array([Rm[x][k] for x in keys], float), np.array([Rm[x]["n"] for x in keys], float)
    def boot_ds(Rm, k, d):                                                             # (NB,) 该套重抽样后的分数，口径同 test_r6.score
        tp, n = arr(Rm, k); t, m = tp[idx[d]][BI[d]], n[idx[d]][BI[d]]
        return 2 * t.sum(1) / np.maximum(m.sum(1), 1) if mode == "gt" else (2 * t / np.maximum(m, 1)).mean(1)
    def boot(nm, k, d):
        return np.mean([boot_ds(s, k, d) for s in S1], 0) if nm == "S1" else boot_ds(M[nm], k, d)
    for a, b in (("U_base", "base"), ("U_d2", "d2"), ("D_base", "base"), ("D_d2", "d2"), ("D_base", "U_base"), ("D_d2", "U_d2"), ("D_base", "S1"), ("D_d2", "S1")):
        pr = {}
        for k in ("tp5", "tp7"):
            pa, pb = out["per_ds"][a][k], out["per_ds"][b][k]
            bd = {d: boot(a, k, d) - boot(b, k, d) for d in dss}
            mb13 = np.mean([bd[d] for d in dss], 0); mb12 = np.mean([bd[d] for d in d12], 0)
            pr[k] = dict(macro13=(float(np.mean([pa[d] - pb[d] for d in dss])), *map(float, np.percentile(mb13, [2.5, 97.5]))),
                         macro12=(float(np.mean([pa[d] - pb[d] for d in d12])), *map(float, np.percentile(mb12, [2.5, 97.5]))),
                         per_ds={d: (pa[d] - pb[d], *map(float, np.percentile(bd[d], [2.5, 97.5]))) for d in dss})
        out["pairs"][f"{a}-{b}"] = pr
    for nm, Rm in list(M.items()) + [("S1_s0", S1[0])]:
        e = errs(Rm, keys); out["err"][nm] = {q: float(np.mean([e[d][q] for d in dss])) for q in ERR}
        out["err_ds_" + nm] = e
    res[mode] = out
    M5 = out["macro"]
    md.append(f"\n### {tag}（{len(keys)} 个体积，{len(dss)} 套）\n\n| 方法 | F1@0.5 13 套 | F1@0.5 12 套 | F1@0.7 13 套 | F1@0.7 12 套 |\n|---|---|---|---|---|")
    md.append(f"| S1 规则 T（5 种子） | {M5['S1']['tp5'][0]:.4f} ± {M5['S1']['tp5'][1]:.4f} | {M5['S1']['tp5'][2]:.4f} ± {M5['S1']['tp5'][3]:.4f} | {M5['S1']['tp7'][0]:.4f} ± {M5['S1']['tp7'][1]:.4f} | {M5['S1']['tp7'][2]:.4f} ± {M5['S1']['tp7'][3]:.4f} |")
    for nm, lab in (("d2", "二维 s0 规则 T（直接）"), ("U_d2", "u-Segment3D(二维 s0)，选中设置"), ("D_d2", "u-Segment3D(二维 s0)，官方默认（事后）"),
                    ("base", "原版固定 cp−2（直接）"), ("U_base", "u-Segment3D(原版)，选中设置"), ("D_base", "u-Segment3D(原版)，官方默认（事后）")):
        md.append(f"| {lab} | {M5[nm]['tp5'][0]:.4f} | {M5[nm]['tp5'][1]:.4f} | {M5[nm]['tp7'][0]:.4f} | {M5[nm]['tp7'][1]:.4f} |")
    for pn, pr in out["pairs"].items():
        md.append(f"\n{pn}：F1@0.5 13 套 {pr['tp5']['macro13'][0]:+.4f} [{pr['tp5']['macro13'][1]:+.4f}, {pr['tp5']['macro13'][2]:+.4f}]，12 套 {pr['tp5']['macro12'][0]:+.4f} [{pr['tp5']['macro12'][1]:+.4f}, {pr['tp5']['macro12'][2]:+.4f}]；"
                  f"F1@0.7 13 套 {pr['tp7']['macro13'][0]:+.4f} [{pr['tp7']['macro13'][1]:+.4f}, {pr['tp7']['macro13'][2]:+.4f}]，12 套 {pr['tp7']['macro12'][0]:+.4f} [{pr['tp7']['macro12'][1]:+.4f}, {pr['tp7']['macro12'][2]:+.4f}]")
        md.append("逐套 F1@0.5：" + "，".join(f"{d} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for d, v in pr["tp5"]["per_ds"].items()))
    md.append("\n误差分解（13 套宏平均，1−F1@0.5 = miss + merge + split + boundary + false）：")
    for nm, e in out["err"].items():
        md.append(f"- {nm}：" + "，".join(f"{q} {e[q]:.3f}" for q in ERR))
json.dump(res, open(f"{O}/g3_compare.json", "w"), indent=1, ensure_ascii=False)
head = "# G3 u-Segment3D 测试集比较（只报告）\n\n选中设置（验证集 88 个体积 gt 协议 13 套 pooled F1@0.5 / F1@0.7）：" + "；".join(f"{k} v{v['variant']} {v['val_f1_05']:.4f}/{v['val_f1_07']:.4f}" for k, v in res["selected"].items())
open(f"{O}/g3_compare.md", "w").write(head + "\n" + "\n".join(md) + "\n")
print(head); print("\n".join(md))
