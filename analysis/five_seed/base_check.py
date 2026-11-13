"""对照检查：第六轮测试集完整流程里，二维与原版在 eval_gt 下“固定 cp−2”与“规则 T”的逐体积宏平均，与第五轮 orx_r5merged_* 目录（r2a2 流程）逐体积对比。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import test_r6 as T
RES = f"{_NSP_ROOT}/results"
def old_rows(name):
    rows = json.load(open(f"{RES}/{name}/rows.json"))
    return {(r["ds"], r["id"]): r for r in rows if r["thr"] == -2.0 and r["frac"] == 0.15 and r["ds"] not in T.DROP}
for key, old in (("tFP_base", "orx_r5merged_base_test"), ("tFP_2D_s0", "orx_r5merged_2d_s0_test")):
    d = T.run_dir(key); R = T.recs(d)
    rows = json.load(open(d + "/rows.json")); fix = {(r["ds"], r["id"]): r for r in rows if r["thr"] == -2.0 and r["frac"] == 0.15}
    O = old_rows(old); ks = sorted(set(R) & set(O))
    f = lambda r, k: 2 * r[k] / max(r["n_gt"] + r["n_pred"], 1)
    by = {}
    for x in ks:
        by.setdefault(x[0], []).append((f(fix[x], "tp5"), R[x], O[x]["f1"], O[x].get("f1"), fix[x]["d_used"] if "d_used" in fix[x] else None, O[x].get("d_used")))
    print(f"== {key}：{len(ks)} 个体积（与 {old} 共有）")
    mac = {"新 固定": [], "新 R2": [], "旧 固定": []}
    for ds, L in sorted(by.items()):
        a = np.mean([l[0] for l in L]); b = np.mean([2 * l[1]["tp5"] / max(l[1]["n"], 1) for l in L]); c = np.mean([l[2] for l in L])
        ch = sum(l[1]["t"] != -2.0 for l in L)
        mac["新 固定"].append(a); mac["新 R2"].append(b); mac["旧 固定"].append(c)
        dif = max(abs(l[0] - l[2]) for l in L)
        print(f"  {ds:12s} n={len(L)} 新固定 {a:.3f}  新R2 {b:.3f}（改 {ch}）  旧固定 {c:.3f}  逐体积最大差 {dif:.3f}  d_used 新/旧 {L[0][4]}/{L[0][5]}")
    print("  宏平均 F1@0.5：" + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in mac.items()))
