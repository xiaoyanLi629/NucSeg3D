"""O5（口径 round3/O5_DESIGN.md，md5 9ddcd198…）：逐个留出数据集的宏平均，S1 − 二维（同种子配对）与 S1 − 原版（固定 cp−2），五种子；
宏平均口径同 s5_stats（gt 每套 pooled，fp 每套逐体积平均）；10000 次套内分层体积自助（所有模型共用同一组索引，随机种子 0）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats/round3'))
from r3common import R3, SEEDS, dump, ncio, np  # noqa: E402

B = 10000


def arrays(M, keys, mode, k):
    """每个数据集、每个模型（S1 五种子、二维五种子、原版）的逐体积 tp 与 n 数组，供快速自助。"""
    out = {}
    for ds in sorted({x[0] for x in keys}):
        ks = [x for x in keys if x[0] == ds]
        out[ds] = {nm: np.array([[[R[x][k], R[x]["n"]] for x in ks] for R in L], float) for nm, L in M.items()}  # [种子, 体积, 2]
    return out


def score(a, mode, idx=None):
    """a: [种子, 体积, 2]；idx: [B, n] 或 None。返回 [种子] 或 [B, 种子]。"""
    if idx is None:
        tp, n = a[:, :, 0], a[:, :, 1]
        return (2 * tp / np.maximum(n, 1)).mean(1) if mode == "fp" else 2 * tp.sum(1) / np.maximum(n.sum(1), 1)
    tp, n = a[:, idx, 0], a[:, idx, 1]                           # [种子, B, n]
    s = (2 * tp / np.maximum(n, 1)).mean(2) if mode == "fp" else 2 * tp.sum(2) / np.maximum(n.sum(2), 1)
    return s.T                                                   # [B, 种子]


def main():
    out = {"design": "round3/O5_DESIGN.md", "seeds": SEEDS, "B": B, "results": {}}
    for mode in ("gt", "fp"):
        M, _ = ncio.load_mode(mode, SEEDS)
        keys = ncio.common_keys(M)
        for k, lab in ncio.METRICS:
            A = arrays(M, keys, mode, k)
            dss = sorted(A)
            rng = np.random.default_rng(0)
            pt = {ds: {nm: score(A[ds][nm], mode) for nm in M} for ds in dss}
            bs = {}
            for ds in dss:
                n = A[ds]["S1"].shape[1]; idx = rng.integers(0, n, (B, n))
                bs[ds] = {nm: score(A[ds][nm], mode, idx) for nm in M}
            for base_tag, base_set in (("12", [d for d in dss if d != "mesospim"]), ("13", dss)):
                for drop in [None] + base_set:
                    S = [d for d in base_set if d != drop]
                    s1 = np.mean([pt[d]["S1"] for d in S], 0); t2 = np.mean([pt[d]["2D"] for d in S], 0); bb = float(np.mean([pt[d]["base"][0] for d in S]))
                    b1 = np.mean([bs[d]["S1"] for d in S], 0); b2 = np.mean([bs[d]["2D"] for d in S], 0); bbb = np.mean([bs[d]["base"][:, 0] for d in S], 0)
                    ent = {}
                    for cn, diff, bdist in (("vs2D", s1 - t2, b1.mean(1) - b2.mean(1)), ("vsBase", s1 - bb, b1.mean(1) - bbb)):
                        mu, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
                        ent[cn] = dict(seed_diffs=diff.tolist(), mean=mu, sd=sd, confirm=bool(mu > 2 * sd and mu >= 0.01),
                                       boot_ci=[float(np.percentile(bdist, 2.5)), float(np.percentile(bdist, 97.5))])
                    out["results"][f"{mode}|{lab}|from{base_tag}|drop={drop or '无'}"] = dict(n_sets=len(S), **ent)
            for base_tag in ("12", "13"):
                rows = {kk.split("drop=")[1]: v for kk, v in out["results"].items() if kk.startswith(f"{mode}|{lab}|from{base_tag}|")}
                for cn in ("vs2D", "vsBase"):
                    allc = all(v[cn]["confirm"] and v[cn]["boot_ci"][0] > 0 for d, v in rows.items() if d != "无")
                    mn = min(((d, v[cn]["mean"]) for d, v in rows.items() if d != "无"), key=lambda t: t[1])
                    me = rows.get("mouse_embryo", {}).get(cn)
                    print(f"{mode} {lab} 从 {base_tag} 套留一 {cn}: 全部 {rows['无'][cn]['mean']:+.4f}±{rows['无'][cn]['sd']:.4f} "
                          f"[{rows['无'][cn]['boot_ci'][0]:+.4f},{rows['无'][cn]['boot_ci'][1]:+.4f}]；最小 留出 {mn[0]} {mn[1]:+.4f}；"
                          f"留出 mouse_embryo {me['mean']:+.4f}±{me['sd']:.4f} [{me['boot_ci'][0]:+.4f},{me['boot_ci'][1]:+.4f}] {'确认' if me['confirm'] else '未确认'}；"
                          f"全部留出集合都确认且区间 > 0：{'是' if allc else '否'}"
                          + ("" if allc else "；失效：" + "、".join(d for d, v in rows.items() if d != "无" and not (v[cn]["confirm"] and v[cn]["boot_ci"][0] > 0))), flush=True)
    dump(out, f"{R3}/o5/results.json")


if __name__ == "__main__":
    main()
