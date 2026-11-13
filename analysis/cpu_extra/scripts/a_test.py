"""分析 A：形态学读数偏差的正式检验（DESIGN_CPUX 第 1 节）。相对误差 e = (预测 − 真值)/真值，成对完整；
主统计量 宏平均 |每套 e 均值|，次统计量 宏平均 每套 mean|e|；逐种子差 S1 − 二维；判定 = |种子均值| > 2×SD 且套内分层自助 95% 区间不含 0。
六个判定（三项读数 × 12/13 套）不做多重比较校正（登记 b8932d9），结果全部报告。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
import sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/cpu_extra/scripts'))
from cpux_common import B, OUT, models, np, ro, ro_gt  # noqa: E402

Q = ("n", "vol_med", "nn_med")


def main():
    M, _ = models()
    keys = sorted(set.intersection(*[set(R) for L in M.values() for R in L]))
    E, excl, per_vol = {}, {}, {}
    for q in Q:
        rows = []
        for x in keys:
            g = ro_gt(*x)
            gv = g[q]
            if q == "n" and not (gv and gv >= 1):
                excl.setdefault(q, []).append(x); continue
            if q == "nn_med" and (gv is None or g["n"] < 2):
                excl.setdefault(q, []).append(x); continue
            if q == "vol_med" and not gv:
                excl.setdefault(q, []).append(x); continue
            vals = {}
            bad = False
            for nm, L in M.items():
                vs = []
                for R in L:
                    o = ro(R[x]["npz"], R[x]["t"])
                    if o[q] is None or (q == "nn_med" and o["n"] < 2) or (q == "vol_med" and o["n"] < 1):
                        bad = True; break
                    vs.append((o[q] - gv) / gv)
                if bad:
                    break
                vals[nm] = vs
            if bad:
                excl.setdefault(q, []).append(x); continue
            rows.append((x, vals))
        E[q] = rows
        for x, vals in rows:
            pv = per_vol.setdefault(x, {})
            for nm, vs in vals.items():
                pv[f"{q}_rel_{nm}"] = float(np.mean(vs))
    res = {"no_multiplicity_correction": "六个判定（三项读数 × 12/13 套）不做多重比较校正，结果全部报告（登记 b8932d9）", "readouts": {}}
    rng = np.random.default_rng(0)
    for q in Q:
        rows = E[q]
        dss_all = sorted({x[0] for x, _ in rows})
        ex = {}
        for x in excl.get(q, []):
            ex[x[0]] = ex.get(x[0], 0) + 1
        R = {"n_vol": len(rows), "excluded_by_ds": ex, "sets": {}}
        arr = {ds: {nm: np.array([v[nm] for x, v in rows if x[0] == ds]).T for nm in M} for ds in dss_all}   # [种子, 体积]
        idx = {ds: rng.integers(0, arr[ds]["S1"].shape[1], (B, arr[ds]["S1"].shape[1])) for ds in dss_all}
        for tag, dss in (("12", [d for d in dss_all if d != "mesospim"]), ("13", dss_all)):
            ent = {}
            for stat in ("abs_bias", "mean_abs_err"):
                def f(a, ix=None):
                    """a: [种子, 体积] → [种子] 或 [种子, B] 的每套统计量。"""
                    x = a if ix is None else a[:, ix]
                    return np.abs(x.mean(-1)) if stat == "abs_bias" else np.abs(x).mean(-1)
                per_seed = {nm: np.mean([f(arr[ds][nm]) for ds in dss], 0) for nm in M}      # [种子]
                diff = per_seed["S1"] - per_seed["2D"]
                mu, sd = float(diff.mean()), float(diff.std(ddof=1))
                bs = np.mean([f(arr[ds]["S1"], idx[ds]).mean(0) - f(arr[ds]["2D"], idx[ds]).mean(0) for ds in dss], 0)   # [B]
                ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                seedok = abs(mu) > 2 * sd; ciok = ci[0] > 0 or ci[1] < 0
                bsb = np.mean([f(arr[ds]["S1"], idx[ds]).mean(0) - f(arr[ds]["base"], idx[ds]).mean(0) for ds in dss], 0)
                ent[stat] = dict(S1=float(per_seed["S1"].mean()), two_d=float(per_seed["2D"].mean()), base=float(per_seed["base"][0]),
                                 diff_seeds=diff.tolist(), diff_mean=mu, diff_sd=sd, boot_ci=ci, seed_consistent=bool(seedok), ci_excludes_0=bool(ciok),
                                 confirmed=bool(seedok and ciok), direction=("S1 更小" if mu < 0 else "二维更小"),
                                 desc_vs_base=dict(diff=float(per_seed["S1"].mean() - per_seed["base"][0]),
                                                   boot_ci=[float(np.percentile(bsb, 2.5)), float(np.percentile(bsb, 97.5))]))
            ent["per_ds_bias"] = {ds: {nm: float(arr[ds][nm].mean()) for nm in M} for ds in dss}
            ent["pooled_bias_rel"] = {nm: float(np.mean(np.concatenate([arr[ds][nm].mean(0) for ds in dss]))) for nm in M}
            R["sets"][tag] = ent
            a = ent["abs_bias"]; b = ent["mean_abs_err"]
            print(f"{q} {tag} 套：|偏差| S1 {a['S1']:.4f} 二维 {a['two_d']:.4f} 原版 {a['base']:.4f}；差 {a['diff_mean']:+.4f}±{a['diff_sd']:.4f} "
                  f"[{a['boot_ci'][0]:+.4f},{a['boot_ci'][1]:+.4f}] {'确认' if a['confirmed'] else '未确认'}（种子{'一致' if a['seed_consistent'] else '不一致'}，区间{'不含' if a['ci_excludes_0'] else '含'} 0）"
                  f" | mean|e| S1 {b['S1']:.4f} 二维 {b['two_d']:.4f} 差 {b['diff_mean']:+.4f}±{b['diff_sd']:.4f} [{b['boot_ci'][0]:+.4f},{b['boot_ci'][1]:+.4f}] {'确认' if b['confirmed'] else '未确认'}", flush=True)
        res["readouts"][q] = R
    json.dump(res, open(f"{OUT}/a_results.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"||".join(k): v for k, v in per_vol.items()}, open(f"{OUT}/a_per_volume.json", "w"))


if __name__ == "__main__":
    main()
