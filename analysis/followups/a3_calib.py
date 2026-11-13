"""nc A3：cellprob 是否校准（CPU）。设计见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN.md 第 A3 节。
不重新推理：已有的测试集 gt 协议运行都带 --save_masks，masks/<ds>__<id>.npz 里存了网络输出 cp（float16，已缩回原网格、原层位）与真值。
概率 p = sigmoid(cp / 3)：S1 的 cp = 3 × logit（视图平均后），二维与原版 run_3D 的 cp = 三个切片方向 logit 之和，
除以 3 即三个方向的平均 logit；三者同一口径。真实前景 = 真值 > 0；未定义区域（nis3d 的 ignore）从载入的原始掩码重新取，不计入。
分箱：p 等宽 15 箱；ECE = Σ_b (n_b/N)·|前景比例_b − 平均 p_b|；每套把体素合在一起算（pooled），再对数据集取平均；另给逐体积 ECE 的中位数、
Brier 分数与可靠性曲线（每箱的平均 p 与前景比例）。
主分析 = 预先登记的“测试集每套 2 个体积”（eval_gt.pick_cases(k=2) 的均匀取法，25 个）；补充 = 全部 227 个测试体积。
用法：python a3_calib.py --out <目录> [--runs s1=<结果目录> d2=<...> base=<...>] [--per_ds 2|0] [--max_vols n]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = f"{_NSP_ROOT}/results"
RUNS = dict(s1=f"{R}/orx_07992b55-0982-4116-b065-b34859c9a80b_thr_eval", d2=f"{R}/orx_81116913-06c5-47a3-9dcc-ab7d64615e68_thr_eval",
            base=f"{R}/orx_b5973303-f479-41fe-be14-f80d7c2fbb1a_thr_eval")
NBIN = 15


def stats(p, y):
    b = np.minimum((p * NBIN).astype(np.int64), NBIN - 1)
    n = np.bincount(b, minlength=NBIN).astype(np.float64)
    sp = np.bincount(b, weights=p, minlength=NBIN); sy = np.bincount(b, weights=y, minlength=NBIN)
    return n, sp, sy, float(((p - y) ** 2).sum())


def ece(n, sp, sy):
    N = max(n.sum(), 1)
    return float(np.sum(np.abs(sy - sp)) / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--runs", nargs="*")
    ap.add_argument("--per_ds", type=int, default=2, help="2 = 预先登记的主分析；0 = 全部测试体积")
    ap.add_argument("--max_vols", type=int, default=0, help="冒烟测试用")
    ap.add_argument("--src", default=os.path.join(HERE, "src"), help="仓库 src 目录（取 pick_cases 与 load_ignore）")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    from eval_gt import pick_cases
    from eval_cpsam import load_case
    runs = dict(x.split("=", 1) for x in a.runs) if a.runs else RUNS
    cases = pick_cases("test", os.path.join(a.src, "..", "val_split.json"), None, None, a.per_ds)
    if a.max_vols:
        cases = cases[:a.max_vols]
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for tag, d in runs.items():
        per_vol, per_ds = {}, {}
        for r in cases:
            k = f"{r['ds']}__{r['id']}"
            f = f"{d}/masks/{k}.npz"
            if not os.path.exists(f):
                print("缺", tag, k, flush=True); continue
            z = np.load(f)
            cp = z["cp"].astype(np.float32); y = (z["gt"] > 0)
            valid = np.ones(y.shape, bool)
            if r["ds"] == "nis3d":                                       # 只有 nis3d 有未定义区域
                _, _, _, ign = load_case(r, 384)
                if ign is not None:
                    assert ign.shape == y.shape, (ign.shape, y.shape)
                    valid = ~ign
            p = 1.0 / (1.0 + np.exp(-cp[valid] / 3.0)); yy = y[valid].astype(np.float64)
            n, sp, sy, br = stats(p.astype(np.float64), yy)
            per_vol[k] = dict(ds=r["ds"], ece=ece(n, sp, sy), brier=br / max(n.sum(), 1), n=n.tolist(), sp=sp.tolist(), sy=sy.tolist(), brier_sum=br)
        for ds in sorted({v["ds"] for v in per_vol.values()}):
            L = [v for v in per_vol.values() if v["ds"] == ds]
            n, sp, sy = (np.sum([np.asarray(v[q]) for v in L], 0) for q in ("n", "sp", "sy"))
            per_ds[ds] = dict(ece=ece(n, sp, sy), brier=sum(v["brier_sum"] for v in L) / max(n.sum(), 1), nvol=len(L),
                              conf=(sp / np.maximum(n, 1)).round(4).tolist(), frac=(sy / np.maximum(n, 1)).round(4).tolist(), n=n.astype(int).tolist())
        res[tag] = dict(run=d, per_ds=per_ds, macro_ece=float(np.mean([v["ece"] for v in per_ds.values()])),
                        macro_ece_no_meso=float(np.mean([v["ece"] for k, v in per_ds.items() if k != "mesospim"])),
                        median_vol_ece=float(np.median([v["ece"] for v in per_vol.values()])),
                        macro_brier=float(np.mean([v["brier"] for v in per_ds.values()])),
                        per_vol={k: dict(ds=v["ds"], ece=v["ece"], brier=v["brier"]) for k, v in per_vol.items()})
        print(f"SUMMARY A3 {tag}: {len(per_vol)} 个体积 {len(per_ds)} 套  宏平均 ECE {res[tag]['macro_ece']:.4f}（去 mesospim "
              f"{res[tag]['macro_ece_no_meso']:.4f}）  逐体积中位 {res[tag]['median_vol_ece']:.4f}  Brier {res[tag]['macro_brier']:.4f}", flush=True)
    tag = f"per_ds{a.per_ds}" if a.per_ds else "all"
    json.dump(dict(bins=NBIN, cases=len(cases), results=res), open(f"{a.out}/a3_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
