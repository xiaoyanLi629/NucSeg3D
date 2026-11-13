"""DESIGN_NNUNET5F 第 4 节：五折版对单折版（两者都是单个模型）：体积级配对自助（套内分层、按体积有放回 10000 次、随机种子 0），
宏平均差（五折减单折）与 95% 区间，逐套三档判定（下界 > 0 胜；≥ −0.02 不劣；否则劣；单体积套只给点估计）；
两种协议 × {13, 12, C1} × F1@0.5/0.7；各用自己验证集选定的一格（单折 cp4_vol0.25，五折 cp4_vol0.15）。自助与汇总代码同 paired_stats.py。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
import sys

import numpy as np

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'baselines')); sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
from compare_test_5f import BL, R, bl_recs, OUT5  # noqa: E402
from paired_stats import CLEAN, NB, call  # noqa: E402


def main():
    out = {"n_boot": NB}
    for mode in ("gt", "fp"):
        rec = {}
        for nm in ("nnU-Net 5f", "nnU-Net"):
            tag, s = BL[nm]
            rec[nm] = bl_recs(f"{R}/{tag}_test_{mode}", float(s["key"].split("_vol")[0][2:]), s["frac"])
        A, B = rec["nnU-Net 5f"], rec["nnU-Net"]
        common = set(A) & set(B)
        for sub in ("13", "12", "C1"):
            keys = sorted(k for k in common if (sub != "12" or k[0] != "mesospim") and (sub != "C1" or k in CLEAN))
            dss = sorted({k[0] for k in keys}); per = {d: [k for k in keys if k[0] == d] for d in dss}
            for met in ("tp5", "tp7"):
                def f1(r, ks):
                    if mode == "fp":
                        return float(np.mean([2 * r[k][met] / max(r[k]["n"], 1) for k in ks]))
                    return 2 * sum(r[k][met] for k in ks) / max(sum(r[k]["n"] for k in ks), 1)
                d0 = float(np.mean([f1(A, per[d]) for d in dss]) - np.mean([f1(B, per[d]) for d in dss]))
                rng = np.random.default_rng(0); bs = np.zeros((NB, len(dss))); pt = {}
                for j, d in enumerate(dss):
                    ks = per[d]; n = len(ks)
                    if mode == "fp":
                        a = np.array([2 * A[k][met] / max(A[k]["n"], 1) for k in ks]); b = np.array([2 * B[k][met] / max(B[k]["n"], 1) for k in ks])
                        idx = rng.integers(0, n, (NB, n)); bs[:, j] = a[idx].mean(1) - b[idx].mean(1); dd = float(a.mean() - b.mean())
                    else:
                        at, an = np.array([A[k][met] for k in ks], float), np.array([A[k]["n"] for k in ks], float)
                        bt, bn = np.array([B[k][met] for k in ks], float), np.array([B[k]["n"] for k in ks], float)
                        idx = rng.integers(0, n, (NB, n))
                        bs[:, j] = 2 * at[idx].sum(1) / np.maximum(an[idx].sum(1), 1) - 2 * bt[idx].sum(1) / np.maximum(bn[idx].sum(1), 1)
                        dd = float(2 * at.sum() / max(an.sum(), 1) - 2 * bt.sum() / max(bn.sum(), 1))
                    if n >= 2:
                        lo, hi = np.percentile(bs[:, j], [2.5, 97.5]); pt[d] = dict(n=n, d=dd, ci=[float(lo), float(hi)], call=call(lo))
                    else:
                        pt[d] = dict(n=1, d=dd, call="单体积")
                lo, hi = (float(x) for x in np.percentile(bs.mean(1), [2.5, 97.5]))
                key = f"{mode}|{sub}|{'F1@0.5' if met == 'tp5' else 'F1@0.7'}"
                out[key] = dict(n_vol=len(keys), n_ds=len(dss), diff=d0, boot_ci=[lo, hi], per_ds=pt)
                c = {x: [d for d in dss if pt[d]["call"] == x] for x in ("胜", "不劣", "劣", "单体积")}
                print(f"5f−1f {key:16s} n={len(keys):3d}/{len(dss):2d}  {d0:+.4f} [{lo:+.4f}, {hi:+.4f}]  胜{len(c['胜'])} 不劣{len(c['不劣'])} 劣{len(c['劣'])}"
                      f"{'(' + ','.join(c['劣']) + ')' if c['劣'] else ''}", flush=True)
    json.dump(out, open(f"{OUT5}/fivefold_vs_single.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
