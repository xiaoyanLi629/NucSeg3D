"""SF4（口径 round3/SF4_DESIGN.md，md5 5e24768e…）：S1 − 原版（固定 cp−2）在同一数据集内干净（d）对相关（a/b/c）体积的对比，
以及 D ~ related + (1|数据集) 混合模型；S1 − 二维作阴性对照。gt 协议为主，fp 只作描述。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import sys
import warnings
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats/round3'))
from r3common import CAT, R3, SEEDS, dump, ncio, np, vf1  # noqa: E402
from scipy import stats  # noqa: E402

COMP = (("vsBase", "base"), ("vs2D", "2D"))


def D(M, x, k, ck, seed=None):
    """体积级差值；seed=None 时 S1（及二维）用五种子平均，seed=i 时用第 i 个种子（原版单模型）。"""
    if seed is None:
        a = np.mean([vf1(R[x], k) for R in M["S1"]]); b = np.mean([vf1(R[x], k) for R in M[ck]])
    else:
        a = vf1(M["S1"][seed][x], k); b = vf1(M[ck][seed if ck == "2D" else 0][x], k)
    return float(a - b)


def mixed(y, X, g, names):
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.MixedLM(np.asarray(y, float), np.asarray(X, float), groups=np.asarray(g)).fit(reml=True)
    ci = np.asarray(r.conf_int())
    return {nm: dict(coef=float(r.fe_params[i]), se=float(r.bse_fe[i]), ci=[float(ci[i, 0]), float(ci[i, 1])], p=float(r.pvalues[i]))
            for i, nm in enumerate(names)} | {"tau2": float(np.asarray(r.cov_re)[0, 0]), "sigma2": float(r.scale), "converged": bool(r.converged)}


def within(M, keys, ds, rel_cats, k, ck, rng, B=10000):
    kd = [x for x in keys if x[0] == ds and CAT[x] == "d"]; kr = [x for x in keys if x[0] == ds and CAT[x] in rel_cats]
    dd = np.array([D(M, x, k, ck) for x in kd]); dr = np.array([D(M, x, k, ck) for x in kr])
    out = dict(n_clean=len(dd), n_related=len(dr), mean_clean=float(dd.mean()), mean_related=float(dr.mean()), delta=float(dr.mean() - dd.mean()))
    seed_delta = [float(np.mean([D(M, x, k, ck, i) for x in kr]) - np.mean([D(M, x, k, ck, i) for x in kd])) for i in range(len(SEEDS))]
    mu, sd = float(np.mean(seed_delta)), float(np.std(seed_delta, ddof=1))
    out.update(seed_delta=seed_delta, seed_mean=mu, seed_sd=sd, seed_confirm=bool(abs(mu) > 2 * sd and abs(mu) >= 0.01))
    if len(dr) >= 2 and len(dd) >= 2:
        bs = dr[rng.integers(0, len(dr), (B, len(dr)))].mean(1) - dd[rng.integers(0, len(dd), (B, len(dd)))].mean(1)
        out.update(boot_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   welch_p=float(stats.ttest_ind(dr, dd, equal_var=False).pvalue), mwu_p=float(stats.mannwhitneyu(dr, dd).pvalue))
    elif len(dr) == 1:
        allv = np.concatenate([dd, dr])
        rank = int((allv > dr[0]).sum()) + 1
        out.update(related_value=float(dr[0]), rank_desc=rank, n_total=len(allv), empirical_quantile=float((dd < dr[0]).mean()))
    return out


def main():
    out = {"design": "round3/SF4_DESIGN.md", "seeds": SEEDS, "gt": {}, "fp": {}}
    rng = np.random.default_rng(0)
    M, _ = ncio.load_mode("gt", SEEDS)
    keys = ncio.common_keys(M)
    for k, lab in ncio.METRICS:
        for cn, ck in COMP:
            e = {}
            e["blastospim"] = within(M, keys, "blastospim", ("a", "b", "c"), k, ck, rng)
            e["mouse_embryo"] = within(M, keys, "mouse_embryo", ("a", "b", "c"), k, ck, rng)
            y = [D(M, x, k, ck) for x in keys]; g = [x[0] for x in keys]
            rel = [0.0 if CAT[x] == "d" else 1.0 for x in keys]
            e["mixed_related"] = mixed(y, np.column_stack([np.ones(len(y)), rel]), g, ["intercept", "related"])
            X = np.column_stack([np.ones(len(y))] + [[1.0 if CAT[x] == c else 0.0 for x in keys] for c in "abc"])
            e["mixed_category"] = mixed(y, X, g, ["intercept", "cat_a", "cat_b", "cat_c"])
            out["gt"][f"{lab}|{cn}"] = e
            b, m, mr = e["blastospim"], e["mouse_embryo"], e["mixed_related"]["related"]
            print(f"gt {lab} {cn}: blastospim 相关 {b['mean_related']:+.4f} 干净 {b['mean_clean']:+.4f} Δ {b['delta']:+.4f} [{b['boot_ci'][0]:+.4f},{b['boot_ci'][1]:+.4f}] "
                  f"Welch p={b['welch_p']:.3g} MWU p={b['mwu_p']:.3g} 种子 {b['seed_mean']:+.4f}±{b['seed_sd']:.4f}{'（确认）' if b['seed_confirm'] else ''} | "
                  f"mouse_embryo E7_Ab1 {m['related_value']:+.4f} 对干净均值 {m['mean_clean']:+.4f}（Δ {m['delta']:+.4f}，13 个中第 {m['rank_desc']}） | "
                  f"混合模型 related {mr['coef']:+.4f} [{mr['ci'][0]:+.4f},{mr['ci'][1]:+.4f}] p={mr['p']:.3g}", flush=True)
    Mf, _ = ncio.load_mode("fp", SEEDS)
    kf = [x for x in ncio.common_keys(Mf) if x[0] == "blastospim"]
    for k, lab in ncio.METRICS:
        out["fp"][lab] = [dict(id=x[1], cat=CAT[x], vsBase=D(Mf, x, k, "base"), vs2D=D(Mf, x, k, "2D")) for x in kf]
        print(f"fp {lab} blastospim 逐体积：" + "；".join(f"{r['id']}({r['cat']}) 原版 {r['vsBase']:+.3f} 二维 {r['vs2D']:+.3f}" for r in out["fp"][lab]), flush=True)
    dump(out, f"{R3}/sf4/results.json")


if __name__ == "__main__":
    main()
