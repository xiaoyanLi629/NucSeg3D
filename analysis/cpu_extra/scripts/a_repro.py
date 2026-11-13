"""第一步：用 inputs/ro 重现论文 S7 的 gt 协议全部 227 个体积的核数偏差（S1 +0.8%、二维 +3.2%、原版 −6.2%），口径与 nc_stats.s7 逐行相同。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
import sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/cpu_extra/scripts'))
from cpux_common import OUT, models, np, ro, ro_gt  # noqa: E402

M, _ = models()
keys = sorted(set.intersection(*[set(R) for L in M.values() for R in L]))
out = {}
for nm in ("S1", "2D", "base"):
    for q in ("n", "vol_med", "nn_med"):
        P, G = [], []
        for x in keys:
            g = ro_gt(*x)[q]
            vals = [v for v in (ro(R[x]["npz"], R[x]["t"])[q] for R in M[nm]) if v is not None]
            if g is None or not vals or (q != "n" and g == 0):
                continue
            p = float(np.mean(vals))
            if q != "n" and p == 0:
                continue
            P.append(p); G.append(g)
        P, G = np.array(P), np.array(G)
        pct = 100 * (P - G) / ((P + G) / 2)
        out[f"{nm}|{q}"] = dict(n=len(P), bias_pct=float(pct.mean()))
print(len(keys), "个体积")
for k, v in out.items():
    print(k, f"{v['bias_pct']:+.2f}%", v["n"])
exp = {"S1|n": 0.8, "2D|n": 3.2, "base|n": -6.2, "S1|vol_med": 1.1, "2D|vol_med": 6.7, "base|vol_med": -22.0, "S1|nn_med": -0.3, "2D|nn_med": -0.7, "base|nn_med": -4.3}
ok = all(round(out[k]["bias_pct"], 1) == v for k, v in exp.items())
import os
os.makedirs(OUT, exist_ok=True)
json.dump(dict(values=out, expected=exp, reproduced=ok), open(f"{OUT}/a_repro.json", "w"), indent=1)
print("重现论文数字：", "全部一致" if ok else "不一致，停止")
sys.exit(0 if ok else 3)
