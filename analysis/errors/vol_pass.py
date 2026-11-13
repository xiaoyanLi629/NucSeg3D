"""第一遍（逐运行 × 逐体积）：规则 T 选阈值 → 最终掩膜（阈值 t 的预测删去 < 0.15 预期体积的小块）→ 误差分类、IoU 网格 TP、
z 向切碎、逐核记录；并把每个计数与 rows.json 在 (t, 0.15) 上的记录逐项核对。结果可续跑（已存在的跳过）。
用法：python vol_pass.py [--workers 16] [--only_tags gt_S1_s0 ...] [--keys ds__id ...] [--test]"""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def ign_for(ds, vid):
    if ds != "nis3d":
        return None
    sys.path.insert(0, C.SRC)
    from eval_gt import pick_cases
    from eval_cpsam import load_case
    r = [x for x in pick_cases("test", None) if x["ds"] == ds and x["id"] == vid][0]
    _, _, _, ign = load_case(r, 384)
    return ign


def job(arg):
    tg, d, key, fixed = arg
    pj = f"{C.OUT}/per/{tg}/{key}.json"
    if os.path.exists(pj):
        return pj, "skip"
    try:
        t0 = time.time()
        ds, vid = key.split("__", 1)
        vrows = C.load_rows(d)[(ds, vid)]
        z = np.load(f"{d}/masks/{key}.npz")
        cp = z["cp"].astype(np.float32)
        t, m = C.rule_t(cp, vrows, fixed=fixed)
        vox = z["vox"]; dd = float(z["d"]); gt = z["gt"]
        if t == -2.0:
            pm = z["pred"]; recomputed = False
        else:
            pm = C.recompute(z, vrows, t, ign=ign_for(ds, vid)); recomputed = True
        pm = C.drop_small(pm, dd, vox)
        res, G, P = C.analyse(gt, pm, vox)
        ref = vrows[(t, C.FRAC)]
        diffs = {k: [res[k], ref[k]] for k in C.INT_FIELDS if res[k] != ref[k]}
        os.makedirs(f"{C.OUT}/cache/{tg}", exist_ok=True)
        np.savez_compressed(f"{C.OUT}/cache/{tg}/{key}.npz", mask=pm.astype(np.int32), g_id=G["id"], g_cls=G["cls"], g_zsplit=G["zsplit"],
                            g_zmerge=G["zmerge"], g_xymerge=G["xymerge"], g_best=G["best"], p_id=P["id"], p_best=P["best"])
        proto = tg.split("_")[0]
        gp = f"{C.OUT}/gtcov/{proto}/{key}.npz"
        if not os.path.exists(gp):
            os.makedirs(os.path.dirname(gp), exist_ok=True)
            cv = C.gt_covariates(gt, vox, G["id"])
            np.savez_compressed(gp + f".{os.getpid()}.tmp.npz", g_id=G["id"], **cv)
            os.replace(gp + f".{os.getpid()}.tmp.npz", gp)
        r0 = vrows[(-2.0, 0.0)]
        out = dict(tag=tg, ds=ds, id=vid, t=t, m2=m[-2.0], recomputed=recomputed, iso=r0["iso"], ratio=r0["ratio"], d_gt=r0["d_gt"],
                   d_used=r0.get("d_used", r0["d_gt"]), shape=list(gt.shape), counts=res, ref={k: ref[k] for k in C.INT_FIELDS},
                   diffs=diffs, f1_ref={str(k): [2 * x["tp5"] / max(x["n_gt"] + x["n_pred"], 1), 2 * x["tp7"] / max(x["n_gt"] + x["n_pred"], 1)]
                                         for k, x in vrows.items()},
                   sec=round(time.time() - t0, 1))
        os.makedirs(os.path.dirname(pj), exist_ok=True)
        json.dump(out, open(pj + ".part", "w")); os.replace(pj + ".part", pj)
        return pj, f"ok {out['sec']}s t={t} diffs={diffs}"
    except Exception:
        return pj, "ERR " + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only_tags", nargs="*")
    ap.add_argument("--keys", nargs="*")
    a = ap.parse_args()
    runs = C.discover()
    jobs = []
    for (proto, nm, s), d in sorted(runs.items()):
        tg = C.tag(proto, nm, s)
        if a.only_tags and tg not in a.only_tags:
            continue
        for f in sorted(os.listdir(f"{d}/masks")):
            key = f[:-4]
            if key.split("__")[0] in C.DROP or (a.keys and key not in a.keys):
                continue
            jobs.append((tg, d, key, nm == "base", os.path.getsize(f"{d}/masks/{f}")))
    jobs.sort(key=lambda x: -x[4])
    jobs = [j[:4] for j in jobs]
    print(f"{len(runs)} 个运行，{len(jobs)} 个作业", flush=True)
    json.dump({C.tag(*k): v for k, v in runs.items()}, open(f"{C.OUT}/runs.json", "w"), indent=1)
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for i, (pj, msg) in enumerate(pool.imap_unordered(job, jobs)):
            if msg != "skip":
                print(i, os.path.relpath(pj, C.OUT), msg[:2000], flush=True)


if __name__ == "__main__":
    main()
