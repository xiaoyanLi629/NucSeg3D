"""FLOWERR step 3 (DESIGN_FLOWERR.md): statistics, results.json, report.html, per_volume.csv.

Primary: Delta = median theta(S1) - median theta(2D) per volume and seed (region a primary; b, c secondary); per-volume Delta = mean over
the five seeds (seeds with a missing region are dropped for that volume; a volume missing in all seeds is dropped); dataset mean over
volumes; macro mean over datasets. Sets: "12" = all datasets except mesospim (primary), "13" = all datasets (mesospim enters as a fixed
value; single-volume datasets are not resampled), "C1" = the paper's clean-test subset (errors/common.subsets()["C1"]) without mesospim.
95 % CI: 10,000 bootstrap resamples of volumes with replacement within each dataset, percentile, numpy default_rng(20261109).
Five-seed confirmation: macro Delta per seed (region a, set 12); confirmed iff mean < 0, |mean| > 2 x SD (ddof=1) and |mean| >= 1 deg.
Secondary: G_flow = -Delta (seed mean, region a), G_F1 = F1@0.5(S1) - F1@0.5(2D) (seed mean; tp5, n_gt, n_pred from
errors/per/gt_{S1,2D}_{seed}/<key>.json 'counts', F1 = 2 tp5 / (n_gt + n_pred)); pooled Spearman rho over volumes of set 12 (primary)
and set 13; p from 10,000 permutations of G_F1 within datasets (two-sided, (1 + #|rho*| >= |rho|) / (1 + B)); per-dataset rho for
datasets with >= 5 volumes (descriptive); the same with F1@0.7 (tp7). Original Cellpose-SAM (gt_base_s0): descriptive medians only."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

OUT = f"{_NSP_SCRATCH}/nc/flowerr"
SEEDS = ("s0", "s1000", "s2000", "s3000", "s4000")
REG = ("a", "b", "c")
B = 10000
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import common as C  # noqa: E402

runs = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))
keys = sorted(f[:-4] for f in os.listdir(f"{runs['gt_S1_s0']}/masks") if f.endswith(".npz"))
ds_of = {k: k.split("__")[0] for k in keys}
C1 = {f"{a}__{b}" for a, b in C.subsets()["C1"]}


def per(tag, k):
    return json.load(open(f"{OUT}/per/{tag}/{k}.json"))


P = {(m, s, k): per(f"gt_{m}_{s}", k) for m in ("S1", "2D") for s in SEEDS for k in keys}
PB = {k: per("gt_base_s0", k) for k in keys}


def f1(m, s, k, tp):
    c = json.load(open(f"{_NSP_SCRATCH}/nc/errors/per/gt_{m}_{s}/{k}.json"))["counts"]
    return 2 * c[tp] / max(c["n_gt"] + c["n_pred"], 1)


# per-volume, per-seed Delta
D = {r: {} for r in REG}
for r in REG:
    for k in keys:
        v = []
        for s in SEEDS:
            a, b = P[("S1", s, k)][r], P[("2D", s, k)][r]
            v.append(np.nan if (a is None or b is None) else a["median"] - b["median"])
        D[r][k] = np.array(v)


def setkeys(name):
    if name == "12":
        return [k for k in keys if ds_of[k] != "mesospim"]
    if name == "13":
        return list(keys)
    if name == "C1":
        return [k for k in keys if k in C1 and ds_of[k] != "mesospim"]


def macro(vals_by_ds):
    return float(np.mean([np.mean(v) for v in vals_by_ds.values()]))


def by_ds(r, ks, seed_idx=None):
    out = {}
    for k in ks:
        x = D[r][k] if seed_idx is None else D[r][k][[seed_idx]]
        if np.all(np.isnan(x)):
            continue
        out.setdefault(ds_of[k], []).append(float(np.nanmean(x)))
    return {d: np.array(v) for d, v in out.items()}


def boot(vbd, rng):
    names = sorted(vbd); est = np.empty(B)
    for i in range(B):
        m = []
        for d in names:
            v = vbd[d]
            m.append(v.mean() if len(v) == 1 else v[rng.integers(0, len(v), len(v))].mean())
        est[i] = np.mean(m)
    return np.percentile(est, [2.5, 97.5]).tolist()


rng = np.random.default_rng(20261109)
res = dict(design="DESIGN_FLOWERR.md md5 413b141fe6edeb395dd05a4faa1ebcc6 (AGENTS.md 82d7398)", primary=dict())
for setn in ("12", "13", "C1"):
    ks = setkeys(setn)
    for r in REG:
        vbd = by_ds(r, ks)
        ent = dict(macro_delta=macro(vbd), ci95=boot(vbd, rng), n_datasets=len(vbd), n_volumes=int(sum(len(v) for v in vbd.values())),
                   per_dataset={d: dict(mean=float(v.mean()), n=len(v)) for d, v in sorted(vbd.items())})
        ent["per_seed"] = [macro(by_ds(r, ks, i)) for i in range(len(SEEDS))]
        res["primary"][f"set{setn}_region_{r}"] = ent
ps = np.array(res["primary"]["set12_region_a"]["per_seed"])
res["confirmation"] = dict(per_seed=ps.tolist(), mean=float(ps.mean()), sd=float(ps.std(ddof=1)),
                           confirmed=bool(ps.mean() < 0 and abs(ps.mean()) > 2 * ps.std(ddof=1) and abs(ps.mean()) >= 1.0),
                           rule="mean < 0 and |mean| > 2 SD (ddof 1) and |mean| >= 1 deg; set 12, region a")
# descriptive model medians (macro over datasets of per-volume seed-mean medians)
desc = {}
for setn in ("12", "13"):
    ks = setkeys(setn)
    for r in REG:
        for m in ("S1", "2D", "base"):
            vb = {}
            for k in ks:
                if m == "base":
                    x = [PB[k][r]["median"]] if PB[k][r] else []
                else:
                    x = [P[(m, s, k)][r]["median"] for s in SEEDS if P[(m, s, k)][r]]
                if x:
                    vb.setdefault(ds_of[k], []).append(np.mean(x))
            desc[f"set{setn}_region_{r}_{m}"] = float(np.mean([np.mean(v) for v in vb.values()]))
res["descriptive_macro_median_theta"] = desc
# secondary: Spearman
G = {}
for k in keys:
    if np.all(np.isnan(D["a"][k])):
        continue
    G[k] = dict(gflow=-float(np.nanmean(D["a"][k])),
                gf1_05=float(np.mean([f1("S1", s, k, "tp5") - f1("2D", s, k, "tp5") for s in SEEDS])),
                gf1_07=float(np.mean([f1("S1", s, k, "tp7") - f1("2D", s, k, "tp7") for s in SEEDS])))


def perm_spearman(ks, fld, rng):
    x = np.array([G[k]["gflow"] for k in ks]); y = np.array([G[k][fld] for k in ks]); ds = np.array([ds_of[k] for k in ks])
    rho = spearmanr(x, y).correlation
    groups = [np.flatnonzero(ds == d) for d in np.unique(ds)]
    cnt = 0
    for _ in range(B):
        yp = y.copy()
        for g in groups:
            yp[g] = y[g][rng.permutation(len(g))]
        cnt += abs(spearmanr(x, yp).correlation) >= abs(rho) - 1e-12
    within = {}
    for d in np.unique(ds):
        g = ds == d
        if g.sum() >= 5:
            within[str(d)] = dict(rho=float(spearmanr(x[g], y[g]).correlation), n=int(g.sum()))
    return dict(rho=float(rho), p_perm=(1 + cnt) / (1 + B), n=len(ks), within_dataset=within)


res["spearman"] = {}
for setn in ("12", "13"):
    ks = [k for k in setkeys(setn) if k in G]
    for fld in ("gf1_05", "gf1_07"):
        res["spearman"][f"set{setn}_{fld}"] = perm_spearman(ks, fld, rng)
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
with open(f"{OUT}/per_volume.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["key", "dataset", "in_C1"] + [f"delta_{r}_{s}" for r in REG for s in SEEDS] +
               [f"median_{m}_{s}_a" for m in ("S1", "2D") for s in SEEDS] + ["median_base_s0_a", "gflow", "gf1_05", "gf1_07"])
    for k in keys:
        row = [k, ds_of[k], int(k in C1)] + [("" if np.isnan(x) else round(float(x), 4)) for r in REG for x in D[r][k]]
        row += [(round(P[(m, s, k)]["a"]["median"], 4) if P[(m, s, k)]["a"] else "") for m in ("S1", "2D") for s in SEEDS]
        row += [round(PB[k]["a"]["median"], 4) if PB[k]["a"] else ""]
        row += [round(G[k][f], 5) if k in G else "" for f in ("gflow", "gf1_05", "gf1_07")]
        w.writerow(row)

# ---------------------------------------------------------------- report.html (Chinese)
STYLE = """<style>:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--bd:#ddd;--hd:#f4f4f6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161618;--fg:#e8e8ea;--mut:#a0a0a6;--bd:#3a3a3e;--hd:#222226}}
:root[data-theme="dark"]{--bg:#161618;--fg:#e8e8ea;--mut:#a0a0a6;--bd:#3a3a3e;--hd:#222226}
body{background:var(--bg);color:var(--fg);font-family:ui-sans-serif,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:960px;margin:2em auto;padding:0 1em;line-height:1.7;overflow-wrap:anywhere}
h2{border-bottom:1px solid var(--bd);padding-bottom:.2em;margin-top:2em}.tw{overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.92em;width:100%}td,th{border:1px solid var(--bd);padding:.3em .6em;text-align:left}th{background:var(--hd)}p.m{color:var(--mut)}</style>"""


def f(x, n=2):
    return f"{x:.{n}f}"


rows = ""
for setn, lab in (("12", "12 套（主，去 mesospim）"), ("13", "13 套"), ("C1", "C1 子集（去 mesospim）")):
    for r, rl in (("a", "(a) 全部前景（主）"), ("b", "(b) z 端带"), ("c", "(c) 内部")):
        e = res["primary"][f"set{setn}_region_{r}"]
        rows += (f"<tr><td>{lab}</td><td>{rl}</td><td>{f(e['macro_delta'])}</td><td>[{f(e['ci95'][0])}, {f(e['ci95'][1])}]</td>"
                 f"<td>{', '.join(f(x) for x in e['per_seed'])}</td><td>{e['n_datasets']} / {e['n_volumes']}</td></tr>")
cf = res["confirmation"]
sp = res["spearman"]
srows = "".join(f"<tr><td>{k}</td><td>{f(v['rho'], 3)}</td><td>{f(v['p_perm'], 4)}</td><td>{v['n']}</td></tr>" for k, v in sp.items())
drows = "".join(f"<tr><td>{k}</td><td>{f(v)}</td></tr>" for k, v in desc.items())
a12 = res["primary"]["set12_region_a"]
doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FLOWERR 结果</title>{STYLE}</head><body>
<h1>FLOWERR：模型流场与真值三维流场目标的一致程度</h1>
<p class="m">事后补加、只报告的分析；设计见 DESIGN_FLOWERR.md（md5 413b141fe6edeb395dd05a4faa1ebcc6，AGENTS.md 提交 82d7398），本报告由 flowerr_stats.py 自动生成。</p>
<h2>1. 结论</h2>
<p>在主分析（12 套、区域 a）中，S1 与二维对照的中位夹角之差的数据集宏平均为 {f(a12['macro_delta'])}°，95% 自助区间为 [{f(a12['ci95'][0])}, {f(a12['ci95'][1])}]°，负值表示 S1 的流场更接近真值三维目标。五个种子的宏平均分别为 {', '.join(f(x) for x in cf['per_seed'])}°，按预先规定的规则（均值小于 0、绝对值大于两倍种子间标准差且不小于 1°），这一差别{'得到确认' if cf['confirmed'] else '没有得到确认'}。解读时必须记住：S1 直接以这种三维流场为训练目标，二维对照以逐层二维流场训练，所以 S1 更接近目标有一部分是构造使然；这里度量的是“与三维训练目标的一致程度”，不是分割质量。</p>
<p>次要分析把逐体积的流场增益与逐体积的 F1 增益做秩相关。12 套上 F1@0.5 的 Spearman ρ 为 {f(sp['set12_gf1_05']['rho'], 3)}（数据集内置换检验 p = {f(sp['set12_gf1_05']['p_perm'], 4)}），F1@0.7 为 {f(sp['set12_gf1_07']['rho'], 3)}（p = {f(sp['set12_gf1_07']['p_perm'], 4)}）。这只是相关，不能推出流场改进导致了分割改进。</p>
<h2>2. 主比较</h2><div class="tw"><table><tr><th>集合</th><th>区域</th><th>宏平均 Δ（°）</th><th>95% 区间</th><th>各种子宏平均</th><th>套数 / 体积数</th></tr>{rows}</table></div>
<h2>3. 秩相关（次要）</h2><div class="tw"><table><tr><th>集合与指标</th><th>Spearman ρ</th><th>置换 p</th><th>体积数</th></tr>{srows}</table></div>
<h2>4. 描述：各模型中位夹角的宏平均（°）</h2><div class="tw"><table><tr><th>集合、区域、模型</th><th>值</th></tr>{drows}</table></div>
<h2>5. 方法要点与局限</h2>
<p>真值目标用 S1 训练代码里的 flows3d_gpu 原样计算，网格是把真值沿 z 按最近原生层上采样到各向同性后的网格，只取原生层上的值；这与训练时直径缩放到 30 像素的 iso30 存储尺度不同，但比较的是单位方向，受尺度影响很小。模型流场是评测时保存的 float16 输出，已经从各向同性网格取回原生层。区域 b 是沿 z 列到本核 z 边界距离小于 0.2 倍等效直径的体素，区域 c 是其余前景；某体积某区域少于 50 个体素时记为缺失。mesospim 的 z/xy 小于 1，它的“各向同性”网格实际上是沿 z 下采样，只进入 13 套估计。原版 Cellpose-SAM 只作描述，不进任何检验。按设计，无论结果如何都照实报告，看到结果后不改区域、阈值或数据集。</p>
<p class="m">文件：results.json、per_volume.csv、per/&lt;tag&gt;/&lt;key&gt;.json、gtflow/&lt;key&gt;.npz、selfcheck.json；脚本 md5 见 SCRIPTS_MD5.txt。</p>
</body></html>"""
open(f"{OUT}/report.html", "w").write(doc)
print(json.dumps(dict(primary={k: (round(v["macro_delta"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in res["primary"].items()},
                      confirmation=cf, spearman={k: (round(v["rho"], 3), v["p_perm"]) for k, v in sp.items()}), indent=1))
