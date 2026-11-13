"""源体素敏感性分析（第二轮审稿 M6；按 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN_VOX.md 预先写定的口径，只报告，CPU）。
新体素：VOX_* 运行（bbbc050 z/xy 2.5、nisnet_zf 3.97）；原体素：已有 tGT_* 运行的同一批 60 个体积（s5_stats.dirs 取目录）。
S1 五种子与二维种子 0 用规则 T（test_r6.recs，体积下限 0.15），原版用固定 cp−2（recs(fixed=True)）。每套 pooled F1@0.5/0.7。
差值：新 − 旧（每个模型）；S1 − 二维（种子 0 对种子 0，以及 S1 五种子均值对二维）、S1 − 原版，在两种体素下各算一次，并给“新 − 旧”的变化。
区间：按体积在各套内配对重抽样 10000 次（种子 0），S1 五种子均值在每次重抽样里重算。另报规则 T 改阈值的体积数、合并比例（被归为合并的真值核 / 真值核数）与 z 向合并占比。
输出 $NUCSEG3D_SCRATCH/nc/gpu/vox/vox_result.json、vox_report.html。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, os, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
from s5_stats import dirs  # noqa: E402
from test_r6 import recs  # noqa: E402
RES = f"{_NSP_ROOT}/results"
O = f"{_NSP_SCRATCH}/nc/gpu/vox"; os.makedirs(O, exist_ok=True)
ST = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]; DSS = ("bbbc050", "nisnet_zf"); NB = 10000


def rd(key):
    return glob.glob(f"{RES}/orx_{ST[key]['run']}*_thr_eval")[0]


def load(d, fixed):
    rc = recs(d, fixed=fixed); by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(d + "/rows.json"))}
    out = {}
    for k, v in rc.items():
        if k[0] in DSS:
            x = by[(*k, v["t"], 0.15)]
            out[k] = dict(tp5=x["tp5"], tp7=x["tp7"], n=x["n_gt"] + x["n_pred"], t=v["t"], merge=x["merge"], n_gt=x["n_gt"],
                          mz=x["merge_z"], mxy=x["merge_xy"], ratio=x["ratio"], iso=x["iso"], d=x.get("d_used", x["d_gt"]))
    return out


D = dirs("test", "gt", SEEDS)
OLD = {"S1": [load(d, False) for d in D["S1"]], "2D": [load(D["2D"][0], False)], "base": [load(D["base"][0], True)]}
NEW = {"S1": [load(rd(f"VOX_S1_{s}"), False) for s in SEEDS], "2D": [load(rd("VOX_2D_s0"), False)], "base": [load(rd("VOX_base"), True)]}
keys = sorted(set.intersection(*[set(r) for M in (OLD, NEW) for L in M.values() for r in L]))
assert len(keys) == 60, len(keys)
idx = {ds: np.array([i for i, k in enumerate(keys) if k[0] == ds]) for ds in DSS}
rng = np.random.default_rng(0); BI = {ds: rng.integers(0, len(idx[ds]), (NB, len(idx[ds]))) for ds in DSS}


def arrs(R, k):
    return np.array([R[x][k] for x in keys], float), np.array([R[x]["n"] for x in keys], float)


def f1(R, k, ds, boot=False):
    tp, n = arrs(R, k); i = idx[ds]
    if not boot:
        return 2 * tp[i].sum() / max(n[i].sum(), 1)
    t, m = tp[i][BI[ds]], n[i][BI[ds]]
    return 2 * t.sum(1) / np.maximum(m.sum(1), 1)


def model(M, nm, k, ds, boot=False, seed0=False):
    L = M[nm][:1] if seed0 else M[nm]
    return np.mean([f1(R, k, ds, boot) for R in L], 0)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


res = {"n_volumes": len(keys), "per_ds": {}}
for ds in DSS:
    e = {"n": int(len(idx[ds])), "ratio_old": sorted({round(OLD["S1"][0][k]["ratio"], 2) for k in keys if k[0] == ds}),
         "ratio_new": sorted({round(NEW["S1"][0][k]["ratio"], 2) for k in keys if k[0] == ds}),
         "iso_old": sorted({OLD["S1"][0][k]["iso"] for k in keys if k[0] == ds}), "iso_new": sorted({NEW["S1"][0][k]["iso"] for k in keys if k[0] == ds}),
         "d_old_med": float(np.median([OLD["S1"][0][k]["d"] for k in keys if k[0] == ds])), "d_new_med": float(np.median([NEW["S1"][0][k]["d"] for k in keys if k[0] == ds]))}
    for k, lab in (("tp5", "F1@0.5"), ("tp7", "F1@0.7")):
        m = {}
        for nm in ("S1", "2D", "base"):
            o = [f1(R, k, ds) for R in OLD[nm]]; n = [f1(R, k, ds) for R in NEW[nm]]
            dd = model(NEW, nm, k, ds, True) - model(OLD, nm, k, ds, True)
            m[nm] = dict(old=float(np.mean(o)), new=float(np.mean(n)), old_sd=float(np.std(o, ddof=1)) if len(o) > 1 else None,
                         new_sd=float(np.std(n, ddof=1)) if len(n) > 1 else None, delta=float(np.mean(n) - np.mean(o)),
                         delta_seeds=[b - a for a, b in zip(o, n)], delta_ci=ci(dd))
        pairs = {}
        for a, b, s0, nmp in (("S1", "2D", True, "S1s0−2D"), ("S1", "2D", False, "S1均值−2D"), ("S1", "base", False, "S1均值−原版")):
            out = {}
            for tag, M in (("old", OLD), ("new", NEW)):
                pt = float(model(M, a, k, ds, seed0=s0) - model(M, b, k, ds))
                bs = model(M, a, k, ds, True, seed0=s0) - model(M, b, k, ds, True)
                out[tag] = dict(diff=pt, ci=ci(bs))
            ch = (model(NEW, a, k, ds, True, seed0=s0) - model(NEW, b, k, ds, True)) - (model(OLD, a, k, ds, True, seed0=s0) - model(OLD, b, k, ds, True))
            out["change"] = dict(diff=out["new"]["diff"] - out["old"]["diff"], ci=ci(ch))
            pairs[nmp] = out
        e[lab] = dict(models=m, pairs=pairs)
    ex = {}
    for tag, M in (("old", OLD), ("new", NEW)):
        for nm in ("S1", "2D", "base"):
            L = M[nm]; kk = [x for x in keys if x[0] == ds]
            ex[f"{tag}_{nm}"] = dict(thr_changed=float(np.mean([sum(R[x]["t"] != -2.0 for x in kk) for R in L])),
                                    merge=float(np.mean([sum(R[x]["merge"] for x in kk) / max(sum(R[x]["n_gt"] for x in kk), 1) for R in L])),
                                    zmerge=float(np.mean([sum(R[x]["mz"] for x in kk) / max(sum(R[x]["mz"] + R[x]["mxy"] for x in kk), 1) for R in L])))
    e["extra"] = ex
    res["per_ds"][ds] = e
json.dump(res, open(f"{O}/vox_result.json", "w"), ensure_ascii=False, indent=1)
# 打印与 HTML
L = []
for ds, e in res["per_ds"].items():
    L.append(f"== {ds}（{e['n']} 个体积；z/xy {e['ratio_old']}→{e['ratio_new']}；插值 {e['iso_old']}→{e['iso_new']}；直径中位数 {e['d_old_med']:.1f}→{e['d_new_med']:.1f}）")
    for lab in ("F1@0.5", "F1@0.7"):
        m = e[lab]["models"]
        L.append(f"  {lab} " + "；".join(f"{nm} {m[nm]['old']:.3f}→{m[nm]['new']:.3f}（{m[nm]['delta']:+.3f} [{m[nm]['delta_ci'][0]:+.3f}, {m[nm]['delta_ci'][1]:+.3f}]）" for nm in m))
        for p, v in e[lab]["pairs"].items():
            L.append(f"    {p}：旧 {v['old']['diff']:+.3f} [{v['old']['ci'][0]:+.3f}, {v['old']['ci'][1]:+.3f}]，新 {v['new']['diff']:+.3f} [{v['new']['ci'][0]:+.3f}, {v['new']['ci'][1]:+.3f}]，变化 {v['change']['diff']:+.3f} [{v['change']['ci'][0]:+.3f}, {v['change']['ci'][1]:+.3f}]")
    L.append("  规则T改阈值/合并比例/z向合并占比：" + "；".join(f"{k} {v['thr_changed']:.1f}/{v['merge']:.3f}/{v['zmerge']:.2f}" for k, v in e["extra"].items()))
txt = "\n".join(L); print(txt)
CSS = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--bd:#ddd;--hd:#f4f4f6}@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#e8e8ea;--mut:#a0a0a6;--bd:#3a3a3e;--hd:#222226}}:root[data-theme="dark"]{--bg:#161618;--fg:#e8e8ea;--mut:#a0a0a6;--bd:#3a3a3e;--hd:#222226}
body{background:var(--bg);color:var(--fg);font-family:ui-sans-serif,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:960px;margin:2em auto;padding:0 1em;line-height:1.7}
.tw{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.92em}td,th{border:1px solid var(--bd);padding:.3em .6em;text-align:right}th{background:var(--hd)}td:first-child,th:first-child{text-align:left}p.m{color:var(--mut)}"""
H = ["<!doctype html><html lang='zh'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>源体素敏感性分析</title><style>" + CSS + "</style><body>",
     "<h1>源体素敏感性分析（第二轮审稿 M6）</h1><p class='m'>投稿前事后补加，只推理、只报告；口径按 DESIGN_VOX.md 预先写定。BBBC050 测试集 44 个、NISNet3D zebrafish 测试集 16 个体积，gt 协议；S1 五种子与二维种子 0 用规则 T，原版固定 cp−2；区间为各套内按体积配对重抽样 10000 次的 95% 区间。</p>"]
for ds, e in res["per_ds"].items():
    H.append(f"<h2>{ds}</h2><p>{e['n']} 个体积；z/xy 由 {e['ratio_old']} 改为 {e['ratio_new']}；各向同性插值由 {e['iso_old']} 变为 {e['iso_new']}；真值直径中位数由 {e['d_old_med']:.1f} 变为 {e['d_new_med']:.1f} 像素。</p>")
    for lab in ("F1@0.5", "F1@0.7"):
        m = e[lab]["models"]
        H.append(f"<h3>{lab}</h3><div class='tw'><table><tr><th>模型</th><th>原体素</th><th>源体素</th><th>新 − 旧</th><th>95% 区间</th></tr>")
        for nm, lab2 in (("S1", "S1（五种子均值）"), ("2D", "二维种子 0"), ("base", "原版")):
            H.append(f"<tr><td>{lab2}</td><td>{m[nm]['old']:.3f}</td><td>{m[nm]['new']:.3f}</td><td>{m[nm]['delta']:+.3f}</td><td>[{m[nm]['delta_ci'][0]:+.3f}, {m[nm]['delta_ci'][1]:+.3f}]</td></tr>")
        H.append("</table></div><div class='tw'><table><tr><th>模型间差</th><th>原体素</th><th>源体素</th><th>变化（新 − 旧）</th></tr>")
        for p, v in e[lab]["pairs"].items():
            H.append(f"<tr><td>{p}</td><td>{v['old']['diff']:+.3f} [{v['old']['ci'][0]:+.3f}, {v['old']['ci'][1]:+.3f}]</td><td>{v['new']['diff']:+.3f} [{v['new']['ci'][0]:+.3f}, {v['new']['ci'][1]:+.3f}]</td><td>{v['change']['diff']:+.3f} [{v['change']['ci'][0]:+.3f}, {v['change']['ci'][1]:+.3f}]</td></tr>")
        H.append("</table></div>")
    H.append("<div class='tw'><table><tr><th>运行</th><th>规则 T 改阈值的体积数</th><th>合并比例</th><th>z 向合并占比</th></tr>" +
             "".join(f"<tr><td>{k}</td><td>{v['thr_changed']:.1f}</td><td>{v['merge']:.3f}</td><td>{v['zmerge']:.2f}</td></tr>" for k, v in e["extra"].items()) + "</table></div>")
H.append(f"<p class='m'>脚本 {_NSP_SCRATCH}/nc/gpu/vox/vox_analysis.py；数据 vox_result.json。</p></body></html>")
open(f"{O}/vox_report.html", "w").write("\n".join(H))
