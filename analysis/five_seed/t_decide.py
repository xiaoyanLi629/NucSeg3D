"""候选 T 的预定判定（AGENTS.md 1257af7）：对若干个 thr_eval 结果目录（每个种子一个），逐体积按规则 R2 选阈值，
按数据集 pooled 再宏平均，报告 12 套（去 mesospim）与 13 套的固定 cp−2 与 R2，以及配对差均值；判定 (1)：12 套两个阈值的配对差均值都 ≥ −0.002。
用法：python t_decide.py 名=目录[,目录,...] ...（目录为 results/ 下的 thr_eval 目录名或绝对路径）"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import glob, json, math, os, statistics as st, sys
import numpy as np
R = f"{_NSP_ROOT}/results/"


def rule(d):
    d = d if d.startswith("/") else R + d
    rows = json.load(open(d + "/rows.json"))
    by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True); grid = [t for t in thrs if t <= -2]
    recs = []
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        cp = np.load(f)["cp"].astype(np.float32)
        r0 = by[(ds, vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in grid}
        t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in grid if t < -2 and m[t] >= 0.85), min(grid))
        recs.append(dict(ds=ds, id=vid, m2=m[-2.0], t=t2, fix=by[(ds, vid, -2.0, 0.15)], r2=by[(ds, vid, t2, 0.15)]))
    n_rows = len({(x["ds"], x["id"]) for x in rows}); assert len(recs) == n_rows, (d, len(recs), n_rows)
    pooled = lambda L, nm, k: 2 * sum(r[nm][k] for r in L) / max(sum(r[nm]["n_gt"] + r[nm]["n_pred"] for r in L), 1)
    per = {s: {nm: (pooled([r for r in recs if r["ds"] == s], nm, "tp5"), pooled([r for r in recs if r["ds"] == s], nm, "tp7")) for nm in ("fix", "r2")}
           for s in sorted({r["ds"] for r in recs})}
    ch = {s: sum(r["t"] != -2.0 for r in recs if r["ds"] == s) for s in per}
    return per, ch, len(recs)


def mac(per, nm, i, ex=()):
    ks = [s for s in per if s not in ex]; return sum(per[s][nm][i] for s in ks) / len(ks)


for arg in sys.argv[1:]:
    nm, ds = arg.split("="); res = [rule(d) for d in ds.split(",")]
    print(f"== {nm}（{len(res)} 个种子，体积 {[r[2] for r in res]}）")
    for tag, ex in (("12 套（去 mesospim）", ("mesospim",)), ("13 套", ())):
        for i, k in ((0, "F1@0.5"), (1, "F1@0.7")):
            fx = [mac(p, "fix", i, ex) for p, _, _ in res]; r2 = [mac(p, "r2", i, ex) for p, _, _ in res]
            dif = [b - a for a, b in zip(fx, r2)]
            print(f"  {tag} {k}：固定 −2 {' '.join('%.4f' % x for x in fx)} | R2 {' '.join('%.4f' % x for x in r2)} | 差 {' '.join('%+.4f' % x for x in dif)} 均值 {st.mean(dif):+.4f}")
    for i, (p, ch, n) in enumerate(res):
        changed = {s: c for s, c in ch.items() if c}
        print(f"  种子 {i}：改了阈值的体积 {sum(ch.values())}/{n}，按数据集 {changed}")
        for s in changed:
            print(f"     {s:12s} 固定 −2 {p[s]['fix'][0]:.3f}/{p[s]['fix'][1]:.3f} → R2 {p[s]['r2'][0]:.3f}/{p[s]['r2'][1]:.3f}")
    d12 = [[mac(p, "r2", i, ("mesospim",)) - mac(p, "fix", i, ("mesospim",)) for p, _, _ in res] for i in (0, 1)]
    ok = all(st.mean(x) >= -0.002 for x in d12)
    print(f"  判定 (1)：12 套配对差均值 F1@0.5 {st.mean(d12[0]):+.4f}、F1@0.7 {st.mean(d12[1]):+.4f} → {'满足' if ok else '不满足'}（要求都 ≥ −0.002）")
