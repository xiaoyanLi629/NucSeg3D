"""L1 种子 0 比较（只报告）：三套留出数据，规则 T（R2，同 sizeThr_dir.py）与固定 cp−2，体积下限 0.15。
每套 pooled F1@0.5/0.7（对象合并再算）与逐体积平均；3 套宏平均。只读已有结果目录。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, math, os, sys
import numpy as np
st = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json")); R = f"{_NSP_ROOT}/results"
SEL = ("bbbc050", "blastospim", "nucverse3d")
def rdir(k): return glob.glob(R + '/orx_' + st[k]['run'] + '_thr_eval')[0]
def recs(d, keep=None):
    rows = json.load(open(d + "/rows.json")); by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True); out = {}
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        if ds not in SEL or (keep is not None and (ds, vid) not in keep): continue
        cp = np.load(f)["cp"].astype(np.float32); r0 = by[(ds, vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
        t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
        out[(ds, vid)] = dict(fix=by[(ds, vid, -2.0, 0.15)], r2=by[(ds, vid, t2, 0.15)], t=t2, d=dd)
    return out
def score(rc, nm):
    per = {}
    for s in SEL:
        L = [v[nm] for (ds, _), v in rc.items() if ds == s]
        if not L: continue
        den = max(sum(x["n_gt"] + x["n_pred"] for x in L), 1)
        pv = lambda k: np.mean([2 * x[k] / max(x["n_gt"] + x["n_pred"], 1) for x in L])
        per[s] = (2 * sum(x["tp5"] for x in L) / den, 2 * sum(x["tp7"] for x in L) / den, pv("tp5"), pv("tp7"), len(L))
    mac = tuple(np.mean([per[s][i] for s in per]) for i in range(4))
    return per, mac
res = {}
for proto in ("tFP", "tGT"):
    l1 = rdir(f"L1_{proto}_2D"); keep = set(recs(l1).keys())
    runs = {"L1_S1": f"L1_{proto}_S1", "L1_2D": f"L1_{proto}_2D", "base": f"{proto}_base"}
    for s in (0, 1000, 2000, 3000, 4000):
        p = "" if s < 3000 else "S5_"
        runs[f"S1_s{s}"] = f"{p}{proto}_S1_s{s}"; runs[f"2D_s{s}"] = f"{p}{proto}_2D_s{s}"
    for name, k in runs.items():
        d = rdir(k)
        if not os.path.exists(d + "/rows.json"): print(proto, name, "未完成"); continue
        rc = recs(d, keep)
        assert set(rc) == keep, (proto, name, len(rc), len(keep))
        res[(proto, name)] = dict(fix=score(rc, "fix"), r2=score(rc, "r2"), nchg={s: sum(v["t"] != -2 for (ds, _), v in rc.items() if ds == s) for s in SEL},
                                  d={f"{a}__{b}": v["d"] for (a, b), v in rc.items()})
        print(proto, name, len(rc), "R2 宏", np.round(res[(proto, name)]["r2"][1], 4), "fix 宏", np.round(res[(proto, name)]["fix"][1], 4), flush=True)
json.dump({f"{a}|{b}": v for (a, b), v in res.items()}, open(os.path.dirname(os.path.abspath(__file__)) + "/l1_cmp.json", "w"), default=float)
