"""O6（口径 round3/O6_DESIGN.md，md5 d4328b29…）：规则 T 常数敏感性，只用验证集（gt 协议，88 个体积，13 套），S1 与二维各五种子。
a ∈ {0.6,0.7,0.8}，b ∈ {0.75,0.85,0.95}，f ∈ {0,0.15,0.25}；规则与 test_r6.recs 相同，只把常数换成参数。f = 0.05 不做（主代理确认跳过）。
自检：登记格 (0.7,0.85,0.15) 逐体积与 test_r6.recs 相同，逐种子宏平均与 s5_stats.py val gt（五种子，ref/）相同。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob
import json
import math
import os
import sys
from multiprocessing import Pool
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats/round3'))
from r3common import R3, SEEDS, dump, ncio, np  # noqa: E402
import test_r6  # noqa: E402

A = (0.6, 0.7, 0.8); BB = (0.75, 0.85, 0.95); F = (0.0, 0.15, 0.25)


def load_dir(d):
    rows = json.load(open(d + "/rows.json"))
    by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True)
    out = {}
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        if ds in test_r6.DROP:
            continue
        cp = np.load(f)["cp"].astype(np.float32)
        r0 = by[(ds, vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
        out["||".join((ds, vid))] = dict(m=m, rec={f"{t}|{fr}": [by[(ds, vid, t, fr)][k] for k in ("tp5", "tp7", "n_gt", "n_pred")]
                                                   for t in thrs for fr in F})
    ref = test_r6.recs(d)                      # 登记格自检
    bad = 0
    for x, r in ref.items():
        e = out["||".join(x)]
        t = pick(e["m"], 0.7, 0.85)
        tp5, tp7, ng, npd = e["rec"][f"{t}|0.15"]
        bad += (t != r["t"]) or (tp5 != r["tp5"]) or (tp7 != r["tp7"]) or (ng + npd != r["n"])
    return d, out, bad


def pick(m, a, b):
    thrs = sorted((float(t) for t in m), reverse=True)
    m = {float(t): v for t, v in m.items()}
    return -2.0 if m[-2.0] >= a else next((t for t in thrs if t < -2 and m[t] >= b), min(thrs))


def macro(V, a, b, f, keep):
    """一个运行目录在某一格的 13/12 套宏平均（每套 pooled）与改阈值的体积数。"""
    per = {}
    nchg = 0
    for key, e in V.items():
        ds = key.split("||")[0]
        if not keep(ds):
            continue
        t = pick(e["m"], a, b); nchg += t != -2.0
        tp5, tp7, ng, npd = e["rec"][f"{t}|{f}"]
        p = per.setdefault(ds, [0, 0, 0]); p[0] += tp5; p[1] += tp7; p[2] += ng + npd
    f5 = float(np.mean([2 * p[0] / max(p[2], 1) for p in per.values()])); f7 = float(np.mean([2 * p[1] / max(p[2], 1) for p in per.values()]))
    return f5, f7, nchg


def main():
    D = ncio.s5_stats.dirs("val", "gt", SEEDS)
    with Pool(10) as pool:
        L = pool.map(load_dir, D["S1"] + D["2D"])
    data = {d: v for d, v, _ in L}
    selfcheck_vol = {os.path.basename(d): bad for d, _, bad in L}
    print("逐体积自检（登记格与 test_r6.recs 不一致的体积数）：", selfcheck_vol, flush=True)
    out = {"design": "round3/O6_DESIGN.md", "seeds": SEEDS, "selfcheck_volume_mismatch": selfcheck_vol, "grid": {}}
    for tag, keep in (("13", lambda s: True), ("12", lambda s: s != "mesospim")):
        for f in F:
            for a in A:
                for b in BB:
                    res = {nm: [macro(data[d], a, b, f, keep) for d in D[nm]] for nm in ("S1", "2D")}
                    ent = {}
                    for i, lab in ((0, "F1@0.5"), (1, "F1@0.7")):
                        s1 = np.array([r[i] for r in res["S1"]]); t2 = np.array([r[i] for r in res["2D"]]); diff = s1 - t2
                        mu, sd = float(diff.mean()), float(diff.std(ddof=1))
                        ent[lab] = dict(S1_seeds=s1.tolist(), two_d_seeds=t2.tolist(), S1=float(s1.mean()), S1_sd=float(s1.std(ddof=1)),
                                        two_d=float(t2.mean()), two_d_sd=float(t2.std(ddof=1)), diff=mu, diff_sd=sd, confirm=bool(mu > 2 * sd and mu >= 0.01))
                    ent["n_changed"] = {nm: [r[2] for r in res[nm]] for nm in res}
                    out["grid"][f"{tag}|a={a}|b={b}|f={f}"] = ent
    refp = f"{R3}/o6/ref/s5_val_gt_5seeds_basefixed.json"
    if os.path.exists(refp):
        ref = json.load(open(refp, encoding="utf-8"))["macro"]; worst = 0.0
        for tag, rt in (("13", "13 套"), ("12", "去 mesospim 12 套")):
            g = out["grid"][f"{tag}|a=0.7|b=0.85|f=0.15"]
            for lab in ("F1@0.5", "F1@0.7"):
                worst = max([worst] + [abs(x - y) for x, y in zip(g[lab]["S1_seeds"], ref[lab][rt]["S1_seeds"])]
                            + [abs(x - y) for x, y in zip(g[lab]["two_d_seeds"], ref[lab][rt]["two_d_seeds"])])
        out["selfcheck_macro_vs_s5_stats"] = worst
        print("登记格逐种子宏平均与 s5_stats.py val（五种子）最大绝对差：", worst, flush=True)
    else:
        out["selfcheck_macro_vs_s5_stats"] = "ref 未生成"
    for tag in ("12", "13"):
        for lab in ("F1@0.5", "F1@0.7"):
            cells = {k: v for k, v in out["grid"].items() if k.startswith(tag + "|")}
            ds_ = [v[lab]["diff"] for v in cells.values()]
            nc = [k for k, v in cells.items() if not v[lab]["confirm"]]
            r = cells[f"{tag}|a=0.7|b=0.85|f=0.15"][lab]
            print(f"{tag} 套 {lab}：登记格 S1 {r['S1']:.4f} 二维 {r['two_d']:.4f} 差 {r['diff']:+.4f}±{r['diff_sd']:.4f}；27 格差值 {min(ds_):+.4f} 至 {max(ds_):+.4f}；"
                  f"S1 {min(v[lab]['S1'] for v in cells.values()):.4f}–{max(v[lab]['S1'] for v in cells.values()):.4f}，"
                  f"二维 {min(v[lab]['two_d'] for v in cells.values()):.4f}–{max(v[lab]['two_d'] for v in cells.values()):.4f}；未确认 {len(nc)} 格 {nc}", flush=True)
    dump(out, f"{R3}/o6/results.json")
    heat(out)


def heat(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"] = False
    for tag in ("12", "13"):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for i, lab in enumerate(("F1@0.5", "F1@0.7")):
            for j, f in enumerate(F):
                Z = np.array([[out["grid"][f"{tag}|a={a}|b={b}|f={f}"][lab]["diff"] for b in BB] for a in A])
                ax = axes[i, j]; im = ax.imshow(Z, cmap="viridis")
                for p in range(3):
                    for q in range(3):
                        c = out["grid"][f"{tag}|a={A[p]}|b={BB[q]}|f={f}"][lab]["confirm"]
                        ax.text(q, p, f"{Z[p, q]:+.3f}{'' if c else '*'}", ha="center", va="center", color="w", fontsize=9)
                ax.set_xticks(range(3)); ax.set_xticklabels(BB); ax.set_yticks(range(3)); ax.set_yticklabels(A)
                ax.set_xlabel("b"); ax.set_ylabel("a"); ax.set_title(f"S1 − 二维 {lab}，f={f}", fontsize=9); fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f"O6 验证集规则 T 常数敏感性（{tag} 套，五种子；* 为未通过确认；登记格 a=0.7、b=0.85、f=0.15）", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{R3}/o6/heat_{tag}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
