"""FLOWERR step 2 (DESIGN_FLOWERR.md): per-voxel angle between each model's evaluated flow and the GT-derived target.

Tags (11): gt_S1_{s0..s4000}, gt_2D_{s0..s4000}, gt_base_s0 (descriptive only), result dirs from $NUCSEG3D_SCRATCH/nc/errors/runs.json.
For each tag x volume: dP (3,Z,Y,X) from the thr_eval masks npz (eval_gt's isotropic-grid output subsampled to native planes),
target t from gtflow/<key>.npz; theta = degrees(arccos(clip(d.t / (|d||t|), -1, 1))) (|d| + 1e-6 in the denominator) at
region (a) = GT foreground with |t| > 0.5, (b) = band, (c) = (a) minus band. No cellprob restriction. GT of each npz is asserted equal
to the gt_S1_s0 GT. A region with < 50 voxels is stored as missing (null). Per region: median, mean, fraction theta > 45 deg, n.
Output: per/<tag>/<key>.json (skip if present).
--selfcheck: E15_Ab2, section y = 126, x 119-375, native z 16-82, seed 0, GT foreground with |t| > 0.5; S1 and 2D medians from this
pipeline, and the xy-view median from $NUCSEG3D_SCRATCH/nc/figs/viz/v2v4_crop.npz (viz_run.py) against the same target;
expected 36 / 17 / 9 deg within rounding (|diff| < 0.5 deg after rounding to integer); exits 1 otherwise.
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 python flowerr_angles.py [--workers 4] | --selfcheck"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

OUT = f"{_NSP_SCRATCH}/nc/flowerr"
TAGS = [f"gt_{m}_{s}" for m in ("S1", "2D") for s in ("s0", "s1000", "s2000", "s3000", "s4000")] + ["gt_base_s0"]
MINVOX = 50


def angles(dP, t, msk):
    d = dP[:, msk].astype(np.float32); tt = t[:, msk].astype(np.float32)
    c = (d * tt).sum(0) / (np.sqrt((d ** 2).sum(0)) + 1e-6) / (np.sqrt((tt ** 2).sum(0)) + 1e-6)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def summ(th):
    if th.size < MINVOX:
        return None
    return dict(median=float(np.median(th)), mean=float(th.mean()), frac_gt45=float((th > 45).mean()), n=int(th.size))


def job(args):
    tag, key, path, gtref = args
    fo = f"{OUT}/per/{tag}/{key}.json"
    if os.path.exists(fo):
        return tag, key, "skip"
    t0 = time.time()
    z = np.load(path); dP = z["dP"]; gt = z["gt"]
    g0 = np.load(gtref)["gt"]
    assert np.array_equal(gt, g0), (tag, key)
    T = np.load(f"{OUT}/gtflow/{key}.npz"); t = T["t"]; band = T["band"]
    tn = np.sqrt((t.astype(np.float32) ** 2).sum(0))
    A = (gt > 0) & (tn > 0.5)
    res = dict(tag=tag, key=key, a=summ(angles(dP, t, A)), b=summ(angles(dP, t, band & A)), c=summ(angles(dP, t, A & ~band)),
               sec=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    json.dump(res, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    return tag, key, "ok"


def selfcheck(runs):
    from scipy import ndimage as ndi
    key = "mouse_embryo__E15_Ab2"
    T = np.load(f"{OUT}/gtflow/{key}.npz"); t = T["t"].astype(np.float32)
    Y, X0, XW, Z0, Z1 = 126, 119, 256, 16, 82
    sl = (slice(Z0, Z1), Y, slice(X0, X0 + XW))
    ts = t[(slice(None),) + sl]
    out = {}
    for m in ("S1", "2D"):
        z = np.load(f"{runs['gt_' + m + '_s0']}/masks/{key}.npz")
        gt = z["gt"][sl]; dP = z["dP"][(slice(None),) + sl]
        msk = (gt > 0) & (np.sqrt((ts ** 2).sum(0)) > 0.5)
        out[m] = float(np.median(angles(dP, ts, msk))); n = int(msk.sum())
    c = np.load(f"{_NSP_SCRATCH}/nc/figs/viz/v2v4_crop.npz")
    ns = c["new_shape"]; row = int(c["row"]) - int(c["rows"][0]); cx0 = int(c["cx0"]); Zu = int(c["Zu"])
    o = c["c2d_out"][:, :, row].astype(np.float32)
    zr = (np.round(np.linspace(0, Zu - 1, 112)).astype(int) * ns[0] / float(Zu))[Z0:Z1]
    xr = (np.arange(X0, X0 + XW) + 0.5) * ns[2] / 384.0 - 0.5 - cx0
    zz, xx = np.meshgrid(zr, xr, indexing="ij")
    samp = lambda a: ndi.map_coordinates(a, [zz, xx], order=1, mode="nearest")  # noqa: E731
    dxy = np.stack([np.zeros_like(zz), samp(o[0]), samp(o[1])])
    out["xy"] = float(np.median(angles(dxy, ts, msk)))
    exp = {"xy": 36, "2D": 17, "S1": 9}
    ok = all(abs(round(out[k]) - exp[k]) < 0.5 for k in exp)
    res = dict(selfcheck=out, expected=exp, n_voxels=n, passed=ok)
    json.dump(res, open(f"{OUT}/selfcheck.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    runs = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))
    if a.selfcheck:
        selfcheck(runs)
    gtsrc = runs["gt_S1_s0"]
    keys = sorted(f[:-4] for f in os.listdir(f"{gtsrc}/masks") if f.endswith(".npz"))
    if a.only:
        keys = [k for k in keys if k in a.only]
    jobs = [(tg, k, f"{runs[tg]}/masks/{k}.npz", f"{gtsrc}/masks/{k}.npz") for k in keys for tg in TAGS]
    jobs.sort(key=lambda j: -os.path.getsize(j[2]))
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
