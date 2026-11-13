"""对任意 eval_gt 结果目录（须带 --save_masks，--thrs 含 −2 及更低的阈值，如 --thrs -2 -3 -4 -6）按“尺度一致”规则逐体积选
cellprob 阈值，与固定 cp−2 比较（体积下限 0.15）。规则同 sizeThr.py：V = π/6·d³·vxy/vz（d 为该次评测用的直径：gt 协议是真值直径，
--diam_from 时是 r2a2 的 d_used），m(t) = #(cellprob > t) / 预测核数(t, 不删小块)；
  R2：m(−2)/V ≥ 0.7 时仍用 −2，否则取 m(t)/V ≥ 0.85 的最高阈值（都不到则取最低阈值）。
输出每套数据 pooled F1@0.5/F1@0.7（固定 −2 与 R2）、宏平均（全部 / 去 mesospim），以及每个体积的 m(−2)/V 与选出的阈值。
用法：python sizeThr_dir.py <结果目录> [...]"""
import glob
import json
import math
import os
import sys

import numpy as np


def run(d):
    rows = json.load(open(d + "/rows.json"))
    by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True)
    assert -2.0 in thrs and min(thrs) < -2, f"{d}：需要 −2 及更低的阈值，现有 {thrs}"
    recs = []
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        cp = np.load(f)["cp"].astype(np.float32)
        r0 = by[(ds, vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
        t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
        recs.append(dict(ds=ds, id=vid, m2=m[-2.0], t=t2, fix=by[(ds, vid, -2.0, 0.15)], r2=by[(ds, vid, t2, 0.15)]))
    pooled = lambda L, nm, k: 2 * sum(r[nm][k] for r in L) / max(sum(r[nm]["n_gt"] + r[nm]["n_pred"] for r in L), 1)
    dss = sorted({r["ds"] for r in recs})
    print(f"== {d}（{len(recs)} 个体积）")
    per = {}
    for s in dss:
        L = [r for r in recs if r["ds"] == s]
        per[s] = {nm: (pooled(L, nm, "tp5"), pooled(L, nm, "tp7")) for nm in ("fix", "r2")}
        ch = sum(r["t"] != -2.0 for r in L)
        print(f"  {s:12s} 固定 −2 {per[s]['fix'][0]:.3f}/{per[s]['fix'][1]:.3f}  R2 {per[s]['r2'][0]:.3f}/{per[s]['r2'][1]:.3f}  "
              f"改阈值的体积 {ch}/{len(L)}  m(−2)/V 最小 {min(r['m2'] for r in L):.2f}")
    for tag, keep in (("全部", lambda s: True), ("去 mesospim", lambda s: s != "mesospim")):
        ks = [s for s in dss if keep(s)]
        if ks:
            f = {nm: (np.mean([per[s][nm][0] for s in ks]), np.mean([per[s][nm][1] for s in ks])) for nm in ("fix", "r2")}
            print(f"  宏平均（{tag}，{len(ks)} 套）固定 −2 {f['fix'][0]:.4f}/{f['fix'][1]:.4f}  R2 {f['r2'][0]:.4f}/{f['r2'][1]:.4f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d.rstrip("/"))
