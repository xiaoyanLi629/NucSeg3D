"""L3 测试集最终结果（只报告，AGENTS.md 095228b 口径）：完整流程 45 个体积、规则 T、逐体积 F1 → 每套平均 → 宏平均（与 s5final/test_fp 相同，复用 test_r6.recs/score）。
对照：同种子 S1 完整流程（二维 r2a2 直径，s5final/test_fp.json 的 dirs.S1）。另报三遍的变化与估计直径/真值直径之比。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed')); import test_r6 as T
st = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json")); R = f"{_NSP_ROOT}/results"
SEEDS = (0, 1000, 2000, 3000, 4000)
F = json.load(open(f"{_NSP_SCRATCH}/r6/s5final/test_fp.json"))["dirs"]["S1"]
import glob, os
def rdir(k):
    c = glob.glob(f"{R}/orx_{st[k][chr(114)+chr(117)+chr(110)]}_*"); c = [x for x in c if os.path.exists(x + "/rows.json")]; assert len(c) == 1, (k, c); return c[0]
def rowsd(d): return {(x["ds"], x["id"]): x for x in json.load(open(d + "/rows.json")) if x["thr"] == -2.0 and x["frac"] == 0.15}
out = dict(seeds=SEEDS, per_seed={}, ratio={})
for s in SEEDS:
    dirs = {"P1": rdir(f"L3t_P1_s{s}"), "P2": rdir(f"L3t_P2_s{s}"), "P3": rdir(f"L3t_P3_s{s}"), "S1_2Ddiam": F[SEEDS.index(s)]}
    assert f"_s{s}" in json.load(open(dirs["S1_2Ddiam"] + "/summary.json"))["model"] or s == 0, dirs["S1_2Ddiam"]
    rec = {n: T.recs(d, fixed=(n == "P1")) for n, d in dirs.items()}
    fix = {n: T.recs(d, fixed=True) for n, d in dirs.items() if n != "P1"}; fix["P1"] = rec["P1"]
    keys = sorted(rec["P3"]); assert all(sorted(r) == keys for r in rec.values()) and len(keys) == 45, [len(r) for r in rec.values()]
    dss = sorted({k[0] for k in keys})
    res = {}
    for nm, RR in (("T", rec), ("fix", fix)):
        for n, r in RR.items():
            per = {ds: [T.score(r, [k for k in keys if k[0] == ds], "fp", q) for q in ("tp5", "tp7")] for ds in dss}
            m13 = [float(np.mean([per[ds][i] for ds in dss])) for i in (0, 1)]
            m12 = [float(np.mean([per[ds][i] for ds in dss if ds != "mesospim"])) for i in (0, 1)]
            res[f"{n}|{nm}"] = dict(per=per, m13=m13, m12=m12)
    rw = {n: rowsd(d) for n, d in dirs.items()}
    out["ratio"][s] = {f"{k[0]}__{k[1]}": {n: rw[n][k].get("d_used", rw[n][k]["d_gt"]) / rw[n][k]["d_gt"] for n in dirs} for k in keys}
    out["ratio"][s]["_nthr"] = {n: sum(v["t"] != -2 for v in rec[n].values()) for n in ("P2", "P3", "S1_2Ddiam")}
    out["per_seed"][s] = res; print("种子", s, "P3 T", np.round(res["P3|T"]["m12"], 4), np.round(res["P3|T"]["m13"], 4), "S1(二维直径) T", np.round(res["S1_2Ddiam|T"]["m12"], 4), np.round(res["S1_2Ddiam|T"]["m13"], 4), flush=True)
json.dump(out, open("l3_final.json", "w"), ensure_ascii=False, default=float)
print("L3F_DONE")
