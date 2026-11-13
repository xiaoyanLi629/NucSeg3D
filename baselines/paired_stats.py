"""G2 配对统计（只报告）：S1（规则 T，五种子）对 nnU-Net、StarDist-3D（单模型，验证集选定格），两种协议 × {13 套, 12 套, C1} × F1@0.5/0.7。
(a) 种子级：每个 S1 种子的宏平均减基线宏平均，给五种子均值 ± SD；确认规则（AGENTS.md）：|均值| > 2×SD 且 |均值| ≥ 0.01。
(b) 体积级：S1 取五种子的体积级均值（fp：逐体积 F1 平均；gt：TP 与 n 平均后 pooled），按数据集分层、套内按体积有放回重采样
    10000 次（随机种子 0），给宏平均差的 95% 区间；逐套差的区间给三档判定（下界 > 0 胜；≥ −0.02 不劣；否则劣；1 个体积的套只给点估计）。
C1 = audit/clean_test.json 的 clean 名单（没有体积剩下的数据集不进宏平均）。
用法：python paired_stats.py fp|gt"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
import sys

import numpy as np

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'baselines')); sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import s5_stats as S5  # noqa: E402
from compare_test import BL, R, SEEDS, bl_recs  # noqa: E402
from recs_cache import crecs  # noqa: E402

NB = 10000
CLEAN = {tuple(x) for x in json.load(open(f"{_NSP_SHARED}/cellposeft_orx/audit/clean_test.json"))["clean"]}


def call(lo):
    return "胜" if lo > 0 else ("不劣" if lo >= -0.02 else "劣")


def main():
    mode = sys.argv[1]
    D = S5.dirs("test", mode, SEEDS)
    S1 = [crecs(d) for d in D["S1"]]
    out = {"mode": mode, "n_boot": NB}
    for nm, (tag, s) in BL.items():
        thr = float(s["key"].split("_vol")[0][2:])
        Bv = bl_recs(f"{R}/{tag}_test_{mode}", thr, s["frac"])
        common = set(Bv).intersection(*[set(r) for r in S1])
        for sub in ("13", "12", "C1"):
            keys = sorted(k for k in common if (sub != "12" or k[0] != "mesospim") and (sub != "C1" or k in CLEAN))
            dss = sorted({k[0] for k in keys})
            per = {ds: [k for k in keys if k[0] == ds] for ds in dss}
            for met in ("tp5", "tp7"):
                def f1(rec, ks):
                    if mode == "fp":
                        return float(np.mean([2 * rec[k][met] / max(rec[k]["n"], 1) for k in ks]))
                    return 2 * sum(rec[k][met] for k in ks) / max(sum(rec[k]["n"] for k in ks), 1)
                mb = np.mean([f1(Bv, per[ds]) for ds in dss])
                dseed = [float(np.mean([f1(r, per[ds]) for ds in dss]) - mb) for r in S1]
                mu, sd = float(np.mean(dseed)), float(np.std(dseed, ddof=1))
                conf = ("S1 优（确认）" if mu > 0 else "基线优（确认）") if (abs(mu) > 2 * sd and abs(mu) >= 0.01) else "未确认"
                # 体积级数组：每个数据集的 S1（五种子均值）与基线
                arr = {}
                for ds in dss:
                    ks = per[ds]
                    if mode == "fp":
                        a = np.array([np.mean([2 * r[k][met] / max(r[k]["n"], 1) for r in S1]) for k in ks])
                        b = np.array([2 * Bv[k][met] / max(Bv[k]["n"], 1) for k in ks])
                        arr[ds] = (a, b, None, None)
                    else:
                        at = np.array([np.mean([r[k][met] for r in S1]) for k in ks]); an = np.array([np.mean([r[k]["n"] for r in S1]) for k in ks])
                        bt = np.array([Bv[k][met] for k in ks], float); bn = np.array([Bv[k]["n"] for k in ks], float)
                        arr[ds] = (at, bt, an, bn)
                rng = np.random.default_rng(0)
                bs = np.zeros((NB, len(dss)))
                for j, ds in enumerate(dss):
                    a, b, an, bn = arr[ds]; n = len(a)
                    idx = rng.integers(0, n, (NB, n))
                    if mode == "fp":
                        bs[:, j] = a[idx].mean(1) - b[idx].mean(1)
                    else:
                        bs[:, j] = 2 * a[idx].sum(1) / np.maximum(an[idx].sum(1), 1) - 2 * b[idx].sum(1) / np.maximum(bn[idx].sum(1), 1)
                pt = {}
                for j, ds in enumerate(dss):
                    a, b, an, bn = arr[ds]
                    d0 = float(a.mean() - b.mean()) if mode == "fp" else float(2 * a.sum() / max(an.sum(), 1) - 2 * b.sum() / max(bn.sum(), 1))
                    if len(a) >= 2:
                        lo, hi = np.percentile(bs[:, j], [2.5, 97.5]); pt[ds] = dict(n=len(a), d=d0, ci=[float(lo), float(hi)], call=call(lo))
                    else:
                        pt[ds] = dict(n=1, d=d0, call="单体积")
                d0 = float(np.mean([pt[ds]["d"] for ds in dss]))
                lo, hi = (float(x) for x in np.percentile(bs.mean(1), [2.5, 97.5]))
                key = f"{nm}|{sub}|{'F1@0.5' if met == 'tp5' else 'F1@0.7'}"
                out[key] = dict(n_vol=len(keys), n_ds=len(dss), seed_diffs=dseed, mean=mu, sd=sd, confirm=conf, boot_d=d0, boot_ci=[lo, hi], per_ds=pt)
                calls = {c: [ds for ds in dss if pt[ds]["call"] == c] for c in ("胜", "不劣", "劣", "单体积")}
                print(f"{mode} {key:24s} n={len(keys):3d}/{len(dss):2d}套  种子级 {mu:+.4f}±{sd:.4f} {conf:8s}  体积自助 {d0:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                      f"胜{len(calls['胜'])} 不劣{len(calls['不劣'])} 劣{len(calls['劣'])}{'(' + ','.join(calls['劣']) + ')' if calls['劣'] else ''}"
                      f"{' 单体积:' + ','.join(f'{ds}{pt[ds][chr(100)]:+.3f}' for ds in calls['单体积']) if calls['单体积'] else ''}", flush=True)
    json.dump(out, open(f"{R}/paired_stats_{mode}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
