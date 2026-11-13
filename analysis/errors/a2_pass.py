"""A2 第二遍（逐协议 × 模型 × 体积）：以种子 0 的最终预测核为基准，求它与其余每个种子最佳匹配的 IoU，取平均作一致度；
错核 = 与真值的最佳 IoU < 0.5（第一遍存下的 p_best）。只用 vol_pass.py 的 cache/（规则 T 后的最终掩膜）。
种子数取当时已有的全部种子（至少 2 个），记录用了哪些种子；五种子齐后重跑即为正式版。
用法：python a2_pass.py [--workers 16]"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def best_iou_against(a, b):
    """a 中每个标签与 b 中任一标签的最大 IoU（按 a 的升序标签返回）。"""
    ids = np.unique(a); ids = ids[ids > 0]
    out = np.zeros(len(ids))
    if not len(ids) or b.max() == 0:
        return ids, out
    asz = np.bincount(a.ravel()); bsz = np.bincount(b.ravel())
    m = (a > 0) & (b > 0)
    nb = int(b.max()) + 1
    u, c = np.unique(a[m].astype(np.int64) * nb + b[m], return_counts=True)
    ga, gb = u // nb, u % nb
    iou = c / (asz[ga] + bsz[gb] - c)
    idx = np.searchsorted(ids, ga)
    np.maximum.at(out, idx, iou)
    return ids, out


def job(arg):
    proto, nm, seeds, key = arg
    od = f"{C.OUT}/a2/{proto}_{nm}"
    fo = f"{od}/{key}.npz"
    sig = "_".join(seeds)
    if os.path.exists(fo) and str(np.load(fo)["seeds"]) == sig:
        return "skip"
    z0 = np.load(f"{C.OUT}/cache/{proto}_{nm}_{seeds[0]}/{key}.npz")
    m0 = z0["mask"]
    ag = []
    for s in seeds[1:]:
        mk = np.load(f"{C.OUT}/cache/{proto}_{nm}_{s}/{key}.npz")["mask"]
        ids, bi = best_iou_against(m0, mk)
        assert np.array_equal(ids, z0["p_id"]), key
        ag.append(bi)
    agree = np.mean(ag, 0) if ag else np.zeros(0)
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(fo + ".tmp.npz", p_id=z0["p_id"], agree=agree, per_seed=np.array(ag), err=z0["p_best"] < 0.5,
                        p_best=z0["p_best"], seeds=sig)
    os.replace(fo + ".tmp.npz", fo)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    jobs = []
    for proto in ("gt", "fp"):
        for nm in ("S1", "2D"):
            seeds = [s for s in C.SEEDS if os.path.isdir(f"{C.OUT}/cache/{proto}_{nm}_{s}")]
            if len(seeds) < 2 or seeds[0] != "s0":
                continue
            keys = set.intersection(*[{os.path.basename(f)[:-4] for f in glob.glob(f"{C.OUT}/cache/{proto}_{nm}_{s}/*.npz")} for s in seeds])
            print(proto, nm, "种子", seeds, "体积", len(keys), flush=True)
            jobs += [(proto, nm, tuple(seeds), k) for k in sorted(keys)]
    with Pool(a.workers) as p:
        r = p.map(job, jobs, chunksize=4)
    print({x: r.count(x) for x in set(r)})


if __name__ == "__main__":
    main()
