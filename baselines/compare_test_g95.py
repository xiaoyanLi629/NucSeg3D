"""G2 测试集并列表（PREREV PR1 副本：BL 再加 nnU-Net 1f g95（单折、网格外扩后的验证集选择 cp5_vol0.25），输出写 results/eval/prerev/stats_g95/，其余逐字同 compare_test.py）（只报告）：基线（验证集选定的一格）与 S1 五种子、二维五种子（s0/s1000/s2000/s3000/s4000，目录由 r6/s5_stats.dirs 给出）、原版，在同一批体积上按 eval_gt 口径汇总。
fp = 完整流程（逐体积 F1 → 每套平均 → 宏平均）；gt = gt 协议（每套 pooled → 宏平均）。13 套（去 bbbc032）与 12 套（再去 mesospim）。
S1/二维/原版用 test_r6.recs 读记录：规则 T（主流程），以及固定 cp−2（--fixed，敏感性）；体积下限都是 0.15。
用法：python compare_test.py fp|gt [--fixed]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
import sys

import numpy as np

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import test_r6 as T  # noqa: E402
import s5_stats as S5  # noqa: E402
from recs_cache import crecs  # noqa: E402
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]

R = f"{_NSP_SHARED}/baselines/results/eval"
SEL = json.load(open(f"{R}/selection_val.json")); SEL5 = json.load(open(f"{R}/selection_val_5f.json"))
BL = {"nnU-Net": ("nnunet", SEL["nnU-Net"]), "StarDist": ("stardistB", SEL["StarDist"]), "nnU-Net 5f": ("nnunet5f", SEL5["nnU-Net 5f"]),
      "nnU-Net 1f g95": ("prerev/nnunet_g95", {"key": "cp5_vol0.25", "frac": 0.25})}   # PREREV PR1: single fold, new validation selection
OUT5 = f"{R}/prerev/stats_g95"


def bl_recs(d, thr, frac):
    out = {}
    for x in json.load(open(f"{d}/rows.json")):
        if x["thr"] == thr and x["frac"] == frac and x["ds"] not in T.DROP:
            out[(x["ds"], x["id"])] = dict(tp5=x["tp5"], tp7=x["tp7"], n=x["n_gt"] + x["n_pred"])
    return out


def main():
    mode = sys.argv[1]; fixed = "--fixed" in sys.argv
    D = S5.dirs("test", mode, SEEDS)
    M = {"S1": [crecs(d, fixed=fixed) for d in D["S1"]], "二维": [crecs(d, fixed=fixed) for d in D["2D"]],
         "原版": [crecs(D["base"][0], fixed=fixed)]}
    for nm, (tag, s) in BL.items():
        thr = float(s["key"].split("_vol")[0][2:]); frac = s["frac"]
        M[nm] = [bl_recs(f"{R}/{tag}_test_{mode}", thr, frac)]
    common = set.intersection(*[set(r) for L in M.values() for r in L])
    dss = sorted({d for d, _ in common})
    print(f"== 测试集 {mode}（{'完整流程，逐体积再平均' if mode == 'fp' else 'gt 协议，pooled'}），{len(common)} 个体积 / {len(dss)} 套；"
          f"S1/二维/原版 {'固定 cp−2' if fixed else '规则 T'}，体积下限 0.15；基线为验证集选定格；S1、二维为五种子均值（宏平均行另给种子间 SD）")
    out = {}
    for k, lab in (("tp5", "F1@0.5"), ("tp7", "F1@0.7")):
        per = {nm: {} for nm in M}
        for ds in dss:
            keys = sorted(x for x in common if x[0] == ds)
            for nm, L in M.items():
                per[nm][ds] = float(np.mean([T.score(r, keys, mode, k) for r in L]))
        print(f"\n-- {lab}    " + "  ".join(f"{nm:>8s}" for nm in M))
        for ds in dss:
            print(f"  {ds:12s} " + "  ".join(f"{per[nm][ds]:8.3f}" for nm in M))
        m13, m12, sd13, sd12 = {}, {}, {}, {}
        for nm, L in M.items():                                      # 每个种子先算宏平均，再求均值与 SD
            a13, a12 = [], []
            for r in L:
                v = {ds: T.score(r, sorted(x for x in common if x[0] == ds), mode, k) for ds in dss}
                a13.append(np.mean(list(v.values()))); a12.append(np.mean([x for d, x in v.items() if d != "mesospim"]))
            m13[nm], m12[nm] = float(np.mean(a13)), float(np.mean(a12))
            sd13[nm], sd12[nm] = (float(np.std(a13, ddof=1)), float(np.std(a12, ddof=1))) if len(L) > 1 else (0.0, 0.0)
        fmt = lambda m, sd, nm: f"{m[nm]:.4f}" + (f"±{sd[nm]:.4f}" if len(M[nm]) > 1 else "")
        print("  宏平均13套   " + "  ".join(f"{nm} {fmt(m13, sd13, nm)}" for nm in M))
        print("  宏平均12套   " + "  ".join(f"{nm} {fmt(m12, sd12, nm)}" for nm in M))
        out[lab] = dict(per_ds=per, macro13=m13, macro12=m12, sd13=sd13, sd12=sd12)
    json.dump(dict(mode=mode, fixed=fixed, n_volumes=len(common), datasets=dss, **out),
              open(f"{OUT5}/compare_test_{mode}{'_fixed' if fixed else ''}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
