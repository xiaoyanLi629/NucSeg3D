"""投稿 Nature Communications 的统计分析 S1、S2、S3、S5、S6、S7（口径：cpsam_finetune/投稿分析方案.html 第 4.1 节，预先登记于 2026-11-05 17:40）。
测试集只报告。比较：S1 + 规则 T 对 二维 + 规则 T（vs2D）；S1 + 规则 T 对 原版固定 cp−2（vsBase）。
两种协议 gt（227 个体积）与 fp（完整流程 45 个体积）；三个子集 all / C1 / C2；F1@0.5 与 F1@0.7。
体积级 F1 = 2·tp/(n_gt+n_pred)（规则 T 选阈值、体积下限 0.15，读数与 s5_stats.py --base_fixed 完全相同），先按种子平均再统计。
  S1 线性混合效应：体积级差值 ~ 1 + (1 | 数据集)，REML（statsmodels MixedLM），截距与 95% Wald 区间；敏感性：按数据集整簇自助 10000 次、去 mesospim。
  S2 随机效应荟萃：每套效应 = 体积级平均差，SE = SD/√n（ddof=1）；n < 2 的套只画不合并；DerSimonian–Laird τ²，Hartung–Knapp 区间，I²；森林图。
  S3 非劣效（TOST 下侧）：只对 S2 点估计为负的套，界值 −0.02（F1@0.5），配对 t 单侧 + 体积自助区间；三档结论。
  S5 逐套配对 Wilcoxon 符号秩，Holm 校正，族 = 同一对照、同一指标（同一协议、同一子集）下的全部数据集。
  S6 元回归（探索性）：S2 的每套效应逐个对 z/xy、核密度、核直径、信噪比做单协变量随机效应元回归（矩估计 τ²，Knapp–Hartung 区间）。
  S7 Bland–Altman：核数、核体积中位数、最近邻距离中位数（readouts.py 的缓存），三个模型都做。
S4 由另一个代理做。
用法：python nc_stats.py --replot（只按已有 JSON 重画森林图、重写摘要）
      python nc_stats.py [--seeds s0 s1000 s2000 ...] [--B 10000] [--workers 64] [--tag 三种子临时版]
先跑 covariates.py 与 readouts.py（同样的 --seeds）生成缓存。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys
import warnings
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats'))
import ncio  # noqa: E402

HERE = ncio.HERE
FIG = f"{HERE}/fig"
COMPS = (("vs2D", "2D", "S1 − 二维"), ("vsBase", "base", "S1 − 原版"))
MARGIN = -0.02
ALL_SEEDS = ("s0", "s1000", "s2000", "s3000", "s4000")


def complete_seeds():
    ok = []
    for s in ALL_SEEDS:
        try:
            for mode in ncio.MODES:
                ncio.s5_stats.dirs("test", mode, [s])
            ok.append(s)
        except (KeyError, AssertionError):
            pass
    return ok


# ---------------------------------------------------------------- S1 混合效应
def fit_mixed(y, g):
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sm.MixedLM(np.asarray(y, float), np.ones((len(y), 1)), groups=np.asarray(g))
        r = m.fit(reml=True)
    ci = np.asarray(r.conf_int())[0]
    return dict(beta=float(r.fe_params[0]), se=float(r.bse_fe[0]), ci=[float(ci[0]), float(ci[1])],
                tau2=float(np.asarray(r.cov_re)[0, 0]), sigma2=float(r.scale), converged=bool(r.converged),
                p=float(r.pvalues[0]) if np.isfinite(r.pvalues[0]) else None)


def _boot_chunk(a):
    y, g, seed, nb = a
    rng = np.random.default_rng(seed)
    dss = sorted(set(g))
    idx = {d: np.nonzero(np.asarray(g) == d)[0] for d in dss}
    out = []
    for _ in range(nb):
        pick = rng.integers(0, len(dss), len(dss))
        yy, gg = [], []
        for j, c in enumerate(pick):
            ii = idx[dss[c]]
            yy.append(y[ii]); gg += [j] * len(ii)
        yy = np.concatenate(yy)
        try:
            if len(set(gg)) < 2 or np.allclose(yy, yy[0]):
                out.append(float(np.mean(yy)) if len(set(gg)) >= 1 else np.nan)   # 退化样本：只有一个簇或差值全相同，截距即均值
            else:
                out.append(fit_mixed(yy, gg)["beta"])
        except Exception:
            out.append(np.nan)
    return out


def s1(rows, comp_key, k, pool, B, seed):
    y = np.array([r[f"S1_{k}"] - r[f"{comp_key}_{k}"] for r in rows]); g = [r["ds"] for r in rows]
    res = {"n_vol": len(y), "n_ds": len(set(g)), "main": fit_mixed(y, g)}
    keep = [i for i, d in enumerate(g) if d != "mesospim"]
    res["no_mesospim"] = fit_mixed(y[keep], [g[i] for i in keep]) if len(keep) < len(y) else "子集中没有 mesospim，与主分析相同"
    nchunk = 20
    jobs = [(y, g, seed * 1000 + c, B // nchunk + (1 if c < B % nchunk else 0)) for c in range(nchunk)]
    bs = np.array([v for L in pool.map(_boot_chunk, jobs) for v in L], float)
    fin = bs[np.isfinite(bs)]
    res["cluster_boot"] = dict(B=int(B), n_fail=int((~np.isfinite(bs)).sum()),
                               ci=[float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))], median=float(np.median(fin)))
    return res


# ---------------------------------------------------------------- S2 随机效应荟萃
def per_ds_effects(rows, comp_key, k):
    out = {}
    for ds in sorted({r["ds"] for r in rows}):
        d = np.array([r[f"S1_{k}"] - r[f"{comp_key}_{k}"] for r in rows if r["ds"] == ds])
        n = len(d)
        sd = float(np.std(d, ddof=1)) if n >= 2 else None
        out[ds] = dict(n=n, effect=float(d.mean()), sd=sd, se=(sd / np.sqrt(n)) if n >= 2 else None, diffs=d.tolist())
    return out


def dl_hk(y, v):
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    w = 1 / v
    mu_f = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - mu_f) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c) if k > 1 else 0.0
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    q_hk = float(np.sum(ws * (y - mu) ** 2) / (k - 1)) if k > 1 else np.nan
    se_hk = float(np.sqrt(q_hk / np.sum(ws)))
    tq = stats.t.ppf(0.975, k - 1)
    se_dl = float(np.sqrt(1 / np.sum(ws)))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    pred = [mu - stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se_hk ** 2), mu + stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se_hk ** 2)] if k > 2 else None
    p = float(2 * stats.t.sf(abs(mu / se_hk), k - 1)) if se_hk > 0 else None
    return dict(k=k, mu=mu, ci_hk=[float(mu - tq * se_hk), float(mu + tq * se_hk)], se_hk=se_hk, p_hk=p,
                ci_dl_wald=[float(mu - 1.96 * se_dl), float(mu + 1.96 * se_dl)], tau2=float(tau2), Q=Q, I2=float(I2),
                pred_interval=[float(x) for x in pred] if pred else None, weights={})


def s2(eff):
    use = [ds for ds, e in eff.items() if e["n"] >= 2]
    v = []
    zero_se = []
    for ds in use:
        se = eff[ds]["se"]
        if se == 0:
            zero_se.append(ds); se = 1e-6        # 差值全相同：SE 为 0 时取 1e-6，避免权重无穷（若出现会在报告里注明）
        v.append(se ** 2)
    r = dl_hk([eff[ds]["effect"] for ds in use], v)
    ws = 1 / (np.array(v) + r["tau2"])
    r["weights"] = {ds: float(w / ws.sum()) for ds, w in zip(use, ws)}
    r["pooled_ds"] = use
    r["plotted_only"] = [ds for ds, e in eff.items() if e["n"] < 2]
    r["zero_se_ds"] = zero_se
    return r


# ---------------------------------------------------------------- S3 非劣效
def s3(eff, rng, B):
    out = {}
    for ds, e in eff.items():
        if e["effect"] >= 0:
            continue
        d = np.array(e["diffs"]); n = len(d)
        if n < 2:
            out[ds] = dict(n=n, effect=e["effect"], verdict="证据不足", note="只有 1 个体积，无法做配对 t 检验或自助")
            continue
        se = d.std(ddof=1) / np.sqrt(n)
        if se == 0:
            t_, p = (np.inf if d.mean() > MARGIN else -np.inf), (0.0 if d.mean() > MARGIN else 1.0)
        else:
            t_ = (d.mean() - MARGIN) / se; p = float(stats.t.sf(t_, n - 1))
        tlo, thi = d.mean() - stats.t.ppf(0.95, n - 1) * se, d.mean() + stats.t.ppf(0.95, n - 1) * se
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
        b90 = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
        b95 = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        if p < 0.05:
            verdict = "非劣效成立"
        elif thi < MARGIN:
            verdict = "非劣效不成立"
        else:
            verdict = "证据不足"
        out[ds] = dict(n=n, effect=float(d.mean()), t=float(t_), df=n - 1, p_one_sided=float(p), t_ci90=[float(tlo), float(thi)],
                       boot_ci90=b90, boot_ci95=b95, verdict=verdict,
                       boot_agrees=bool((b90[0] > MARGIN) == (p < 0.05)))
    return out


# ---------------------------------------------------------------- S5 Wilcoxon + Holm
def holm(ps):
    keys = [k for k, p in ps.items() if p is not None]
    order = sorted(keys, key=lambda k: ps[k])
    m = len(order); adj = {}; run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        adj[k] = run
    return adj


def s5(eff):
    out, ps = {}, {}
    for ds, e in eff.items():
        d = np.array(e["diffs"])
        if len(d) < 2:
            out[ds] = dict(n=len(d), p=None, note="单体积，不检验"); ps[ds] = None
            continue
        if np.all(d == 0):
            out[ds] = dict(n=len(d), stat=None, p=1.0, median=0.0, note="差值全为 0"); ps[ds] = 1.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        out[ds] = dict(n=len(d), n_nonzero=int((d != 0).sum()), stat=float(w.statistic), p=float(w.pvalue), median=float(np.median(d)),
                       mean=float(d.mean()))
        ps[ds] = float(w.pvalue)
    adj = holm(ps)
    for ds in out:
        out[ds]["p_holm"] = adj.get(ds)
        out[ds]["sig_holm_005"] = (adj[ds] < 0.05) if ds in adj else None
    return out


# ---------------------------------------------------------------- S6 元回归
def meta_reg(y, v, x):
    y, v, x = (np.asarray(a, float) for a in (y, v, x))
    k = len(y); X = np.column_stack([np.ones(k), x]); p = 2
    W = np.diag(1 / v)
    XtWX_inv = np.linalg.inv(X.T @ W @ X)
    b_fe = XtWX_inv @ X.T @ W @ y
    QE = float(np.sum((y - X @ b_fe) ** 2 / v))
    P = W - W @ X @ XtWX_inv @ X.T @ W
    tau2 = max(0.0, (QE - (k - p)) / np.trace(P))
    Ws = np.diag(1 / (v + tau2))
    A = np.linalg.inv(X.T @ Ws @ X)
    b = A @ X.T @ Ws @ y
    e = y - X @ b
    s2 = float(np.sum(np.diag(Ws) * e ** 2) / (k - p))
    se = np.sqrt(np.diag(A) * s2)
    tq = stats.t.ppf(0.975, k - p)
    return dict(k=k, slope=float(b[1]), se_kh=float(se[1]), ci=[float(b[1] - tq * se[1]), float(b[1] + tq * se[1])],
                p=float(2 * stats.t.sf(abs(b[1] / se[1]), k - p)), intercept=float(b[0]), tau2_res=float(tau2), QE=QE)


COVS = (("zxy", "z/xy 体素比"), ("density", "核密度（前景体素占比）"), ("diam_px", "核直径（xy 像素）"), ("snr", "信噪比"))


def s6(eff, s2r, rows, cov):
    use = s2r["pooled_ds"]
    if len(use) < 4:
        return {"note": f"合并的数据集只有 {len(use)} 套，不做元回归"}
    out = {"ds_values": {}, "undefined_volumes": {}}
    for ds in use:
        vv = [cov[f"{r['ds']}__{r['id']}"] for r in rows if r["ds"] == ds]
        out["ds_values"][ds] = {c: (float(np.median([x[c] for x in vv])) if all(x[c] is not None for x in vv) else None) for c, _ in COVS}
        for c, _ in COVS:
            nu = sum(x[c] is None for x in vv)
            if nu:
                out["undefined_volumes"].setdefault(c, {})[ds] = f"{nu}/{len(vv)}"
    y = [eff[ds]["effect"] for ds in use]
    v = [max(eff[ds]["se"], 1e-6) ** 2 for ds in use]
    for c, _ in COVS:
        if any(out["ds_values"][ds][c] is None for ds in use):
            # 口径无法执行：信噪比定义的分母（核外 MAD）为 0 时没有定义；按要求不改方法，只报告哪些体积无定义
            out[c] = {"not_executable": True, "note": "部分体积核外强度 MAD = 0，预先登记的定义无法计算", "undefined": out["undefined_volumes"].get(c)}
            continue
        out[c] = meta_reg(y, v, [out["ds_values"][ds][c] for ds in use])
    return out


# ---------------------------------------------------------------- S7 Bland–Altman
def ba(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    pct = 100 * d / ((a + b) / 2)
    def summ(x):
        m, s = float(np.mean(x)), float(np.std(x, ddof=1))
        return dict(bias=m, sd=s, loa=[m - 1.96 * s, m + 1.96 * s], n=len(x))
    return dict(raw=summ(d), pct=summ(pct))


def s7(rows, M, readouts):
    out = {}
    for nm in ("S1", "2D", "base"):
        o = {}
        for q in ("n", "vol_med", "nn_med"):
            P, G, per = [], [], {}
            for r in rows:
                x = (r["ds"], r["id"])
                g = readouts.load_gt_ro(*x)[q]
                vals = [readouts.load_ro(R[x]["npz"], R[x]["t"])[q] for R in M[nm]]
                vals = [v for v in vals if v is not None]
                if g is None or not vals or (q != "n" and g == 0):
                    continue
                p = float(np.mean(vals))
                if q != "n" and p == 0:
                    continue
                P.append(p); G.append(g); per.setdefault(r["ds"], []).append((p, g))
            if len(P) < 2:
                o[q] = {"note": "可用体积不足 2 个"}
                continue
            o[q] = ba(P, G)
            o[q]["per_ds_pct_bias"] = {ds: float(np.mean([100 * (p - g) / ((p + g) / 2) for p, g in L])) for ds, L in per.items()}
        out[nm] = o
    return out


# ---------------------------------------------------------------- 森林图
def forest(effs, s2rs, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, len(effs), figsize=(5.2 * len(effs), 0.38 * (max(len(e) for e in effs.values()) + 4) + 1), squeeze=False)
    for ax, (lab, eff) in zip(axes[0], effs.items()):
        r = s2rs[lab]
        dss = sorted(eff)
        for i, ds in enumerate(dss):
            e = eff[ds]; yv = len(dss) - i
            if e["n"] >= 2:
                lo, hi = e["effect"] - 1.96 * e["se"], e["effect"] + 1.96 * e["se"]
                ax.plot([lo, hi], [yv, yv], color="#444", lw=1)
                ax.plot(e["effect"], yv, "s", color="#1f5fa8", ms=4 + 10 * r["weights"].get(ds, 0))
            else:
                ax.plot(e["effect"], yv, "o", mfc="white", mec="#888", ms=5)
            ax.text(1.02, yv, f"n={e['n']}", transform=ax.get_yaxis_transform(), fontsize=7, va="center")
        mu, (lo, hi) = r["mu"], r["ci_hk"]
        ax.fill([lo, mu, hi, mu], [-0.5, -0.1, -0.5, -0.9], color="#c0392b")
        ax.axvline(0, color="#999", lw=0.8, ls="--")
        ax.set_yticks(list(range(len(dss), 0, -1)) + [-0.5])
        ax.set_yticklabels(dss + ["合并（HK）"], fontsize=8)
        ax.set_title(f"{lab}\n合并 {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]，I²={100 * r['I2']:.0f}%，k={r['k']}", fontsize=9)
        ax.set_xlabel("体积级平均差（95% 区间 = ±1.96 SE；空心点为单体积，不合并）", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- 一致性核对
def consistency(M_by_mode, seeds):
    if list(seeds) != ["s0", "s1000", "s2000"]:
        # 非三种子：核对对象是用同样种子独立跑出的 s5_stats.py 输出（ref/ 下，--base_fixed，--out 指到本目录，不写 r6/）
        n = len(seeds); R0 = f"{HERE}/ref/s5_test_{{m}}_{n}seeds_basefixed{{s}}.json"
        refs = {(m, sub): R0.format(m=m, s=sfx) for m in ("gt", "fp") for sub, sfx in (("all", ""), ("C1", "_clean_test"), ("C2", "_clean_test_plus_b"))}
        if not all(os.path.exists(f) for f in refs.values()):
            return {"note": f"ref/ 下缺少 {n} 种子的 s5_stats.py 输出，未核对"}
        return _check(M_by_mode, refs)
    refs = {("gt", "all"): f"{_NSP_SCRATCH}/r6/test_r6_gt_basefixed.json", ("fp", "all"): f"{_NSP_SCRATCH}/r6/test_r6_fp_basefixed.json",
            ("gt", "C1"): f"{_NSP_SCRATCH}/r6/s5_test_gt_3seeds_basefixed_clean_test.json",
            ("fp", "C1"): f"{_NSP_SCRATCH}/r6/s5_test_fp_3seeds_basefixed_clean_test.json",
            ("gt", "C2"): f"{_NSP_SCRATCH}/r6/s5_test_gt_3seeds_basefixed_clean_test_plus_b.json",
            ("fp", "C2"): f"{_NSP_SCRATCH}/r6/s5_test_fp_3seeds_basefixed_clean_test_plus_b.json"}
    return _check(M_by_mode, refs)


def _check(M_by_mode, refs):
    out, worst = {}, 0.0
    for (mode, sub), f in refs.items():
        ref = json.load(open(f, encoding="utf-8"))
        M = M_by_mode[mode]
        keys = ncio.common_keys(M, ncio.subset_keys(sub))
        for k, lab in ncio.METRICS:
            for tag, ent in ref["macro"][lab].items():
                keep = (lambda ds: ds != "mesospim") if "mesospim" in tag else (lambda ds: True)
                mine, dss = ncio.macro_seeds(M, keys, mode, k, keep)
                pairs = list(zip(mine["S1"], ent["S1_seeds"])) + list(zip(mine["2D"], ent["two_d_seeds"])) + [(mine["base"][0], ent["base"])]
                dmax = max(abs(a - b) for a, b in pairs)
                worst = max(worst, dmax)
                out[f"{mode}/{sub}/{lab}/{tag}"] = dict(max_abs_diff=dmax, n_vol=len(keys), mine_S1=mine["S1"], ref_S1=ent["S1_seeds"])
            # 逐套点估计也核对
            for ds, e in ref["per_ds"][lab].items():
                kk = [x for x in keys if x[0] == ds]
                v = {nm: float(np.mean([ncio.score(R, kk, mode, k) for R in L])) for nm, L in M.items()}
                d = max(abs(v["S1"] - e["S1"]), abs(v["2D"] - e["two_d"]), abs(v["base"] - e["base"]))
                worst = max(worst, d)
                out[f"{mode}/{sub}/{lab}/{ds}"] = dict(max_abs_diff=d)
    return {"items": out, "worst_abs_diff": worst, "n_items": len(out), "identical": worst < 1e-12, "refs": {f"{m}/{b}": f for (m, b), f in refs.items()}}


# ---------------------------------------------------------------- 主程序
def replot(path):
    res = json.load(open(path, encoding="utf-8"))
    for mode, D in res["results"].items():
        for sub, R in D.items():
            for _, lab in ncio.METRICS:
                effs = {clab: R[lab][cn]["per_ds"] for cn, _, clab in COMPS}
                s2s = {clab: R[lab][cn]["S2"] for cn, _, clab in COMPS}
                forest(effs, s2s, f"{res['tag']}｜{'gt 协议' if mode == 'gt' else '完整流程'}｜{sub}｜{lab}",
                       f"{FIG}/forest_{mode}_{sub}_{lab.replace('@', '').replace('.', '')}.png")
    import nc_report
    nc_report.write(res, f"{HERE}/nc_stats_report.md")


def main():
    if "--replot" in sys.argv:
        return replot(f"{HERE}/nc_stats_results.json")
    seeds = sys.argv[sys.argv.index("--seeds") + 1:] if "--seeds" in sys.argv else None
    seeds = [s for s in seeds if s.startswith("s") and s[1:].isdigit()] if seeds else complete_seeds()
    B = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 10000
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 64
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else (f"{len(seeds)}种子" + ("临时版" if len(seeds) < 5 else "正式版"))
    os.makedirs(FIG, exist_ok=True)
    import readouts
    cov = json.load(open(f"{ncio.CACHE}/covariates.json", encoding="utf-8"))
    print(f"种子 {seeds}（{tag}），B={B}", flush=True)
    M_by_mode = {mode: ncio.load_mode(mode, seeds)[0] for mode in ncio.MODES}
    res = {"tag": tag, "seeds": seeds, "B": B, "dirs": {m: ncio.load_mode(m, seeds)[1] for m in ncio.MODES},
           "consistency": consistency(M_by_mode, seeds), "results": {}}
    c = res["consistency"]
    print("一致性核对：", c.get("note") or f"{c['n_items']} 项，最大绝对差 {c['worst_abs_diff']:.3g}，{'完全一致' if c['identical'] else '有差异'}", flush=True)
    from multiprocessing import Pool
    rng = np.random.default_rng(20261105)
    with Pool(W) as pool:
        for mode in ncio.MODES:
            M = M_by_mode[mode]
            for sub in ncio.SUBSETS:
                keys = ncio.common_keys(M, ncio.subset_keys(sub))
                rows = ncio.volume_table(M, keys)
                R = res["results"].setdefault(mode, {}).setdefault(sub, {"n_vol": len(rows), "n_ds": len({r["ds"] for r in rows}),
                                                                            "volumes": rows})
                for k, lab in ncio.METRICS:
                    effs, s2s = {}, {}
                    for ci, (cname, ckey, clab) in enumerate(COMPS):
                        eff = per_ds_effects(rows, ckey, k)
                        ent = {"S1": s1(rows, ckey, k, pool, B, seed=zlib.crc32(f"{mode}|{sub}|{k}|{cname}".encode()) % 10 ** 6),
                               "per_ds": eff, "S2": s2(eff), "S5": s5(eff)}
                        ent["S3"] = s3(eff, rng, B) if k == "tp5" else "界值只预先登记了 F1@0.5，F1@0.7 不做"
                        ent["S6"] = s6(eff, ent["S2"], rows, cov)
                        R.setdefault(lab, {})[cname] = ent
                        effs[clab] = eff; s2s[clab] = ent["S2"]
                        m1 = ent["S1"]["main"]
                        print(f"[{mode} {sub} {lab} {cname}] S1 β={m1['beta']:+.4f} [{m1['ci'][0]:+.4f},{m1['ci'][1]:+.4f}] "
                              f"簇自助 [{ent['S1']['cluster_boot']['ci'][0]:+.4f},{ent['S1']['cluster_boot']['ci'][1]:+.4f}]  "
                              f"S2 μ={ent['S2']['mu']:+.4f} HK [{ent['S2']['ci_hk'][0]:+.4f},{ent['S2']['ci_hk'][1]:+.4f}] I²={ent['S2']['I2']:.2f}", flush=True)
                    forest(effs, s2s, f"{tag}｜{'gt 协议' if mode == 'gt' else '完整流程'}｜{sub}｜{lab}",
                           f"{FIG}/forest_{mode}_{sub}_{lab.replace('@', '').replace('.', '')}.png")
                R["S7"] = s7(rows, M, readouts)
    mm = {"n_masks": 0, "n_recomputed": 0, "n_mismatch": 0, "mismatch": []}
    for mode, M in M_by_mode.items():
        for nm, L in M.items():
            for R in L:
                for x, r in R.items():
                    o = readouts.load_ro(r["npz"], r["t"]); mm["n_masks"] += 1; mm["n_recomputed"] += r["t"] != -2.0
                    if not o["n_match"]:
                        mm["n_mismatch"] += 1; mm["mismatch"].append([r["npz"], r["t"], o["n"], o["n_pred_rec"]])
    res["s7_mask_check"] = mm
    out = f"{HERE}/nc_stats_results.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("写入", out)
    import nc_report
    nc_report.write(res, f"{HERE}/nc_stats_report.md")
    print("写入", f"{HERE}/nc_stats_report.md")


if __name__ == "__main__":
    main()
