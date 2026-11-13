"""FLOWERR step 1 (DESIGN_FLOWERR.md, registered AGENTS.md 82d7398): GT-derived 3D flow target per test volume.

For each gt-protocol test volume (227; GT and vox from the gt_S1_s0 thr_eval masks npz, ignore regions already 0):
  GT labels are upsampled along z by nearest native plane to Zu = round(Z * z/xy) (zi = errors/common.zi_of, the eval_gt mapping),
  cropped to the foreground bounding box (diffusion is per nucleus, so cropping does not change the result), and passed to
  flows3d_gpu copied verbatim from $NUCSEG3D_RUNS/07992b55-0982-4116-b065-b34859c9a80b/repo/src/data3d.py (the function starting at line 29), on CPU,
  float64 (its default), max_pass 3, max_iter 4000. Only the target at the native planes zi is kept.
Regions: (a) GT foreground with |t| > 0.5 at native planes; (b) z-cap band: voxels whose distance along z, along their own (y,x)
  column and within their own nucleus, to that nucleus's z boundary is < 0.2 x the nucleus's equivalent-sphere diameter;
  distance = (number of consecutive same-label voxels between the voxel and the boundary, i.e. min(n_above, n_below), + 0.5) planes
  x vox_z; equivalent diameter = (6 V / pi)^(1/3), V = voxel count x prod(vox). All lengths in xy-pixel units (vox / vox[1]), which
  equals the um definition for datasets with um voxels and is the manifest-ratio definition for bbbc050/nisnet_zf/nis3d/purdue.
  (c) interior = (a) minus band.
Output: gtflow/<key>.npz with t (3,Z,Y,X) float16 at native planes, band (Z,Y,X) bool, n_a/n_b/n_c, Zu, ratio, sec.
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 python flowerr_target.py [--workers 4] [--threads 2] [--only key ...]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

OUT = f"{_NSP_SCRATCH}/nc/flowerr"
SNAP = f"{_NSP_RUNS}/07992b55-0982-4116-b065-b34859c9a80b/repo/src/data3d.py"
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))


def load_flows3d():
    import torch
    import torch.nn.functional as F  # noqa: F401
    src = open(SNAP).read().splitlines()
    # the function as it stands in the snapshot: from line 29 ("def flows3d_gpu") up to (excluding) the next top-level line;
    # lines 29-85 of the design cover this function plus the following section header/stub, which must not be exec'd
    end = next(i for i in range(29, len(src)) if src[i] and not src[i][0].isspace())
    code = "\n".join(src[28:end])
    assert code.startswith("def flows3d_gpu(lab, max_pass=3, max_iter=4000"), code[:80]
    assert code.rstrip().splitlines()[-1].strip().startswith("return mu_all.view(3"), code[-120:]
    ns = {"torch": torch, "F": F, "np": np}
    exec(code, ns)
    return ns["flows3d_gpu"]


def mem_ok(need_gb=40):
    try:
        mx = open('/sys/fs/cgroup/memory.max').read().strip(); mx = int(mx) if mx != 'max' else 10 ** 13
        st = dict(l.split() for l in open('/sys/fs/cgroup/memory.stat'))
        return (mx - int(st['anon']) - int(st.get('shmem', 0))) / 1e9 >= need_gb
    except Exception:
        return True


def band_mask(gt, vox_rel):
    """z-cap band per the design (see module docstring)."""
    Z = gt.shape[0]
    up = np.zeros(gt.shape, np.int32); dn = np.zeros(gt.shape, np.int32)
    for z in range(1, Z):
        same = (gt[z] > 0) & (gt[z] == gt[z - 1])
        up[z] = np.where(same, up[z - 1] + 1, 0)
    for z in range(Z - 2, -1, -1):
        same = (gt[z] > 0) & (gt[z] == gt[z + 1])
        dn[z] = np.where(same, dn[z + 1] + 1, 0)
    dist = (np.minimum(up, dn) + 0.5) * vox_rel[0]
    cnt = np.bincount(gt.ravel()).astype(np.float64)
    deq = (6 * cnt * np.prod(vox_rel) / np.pi) ** (1 / 3)
    return (gt > 0) & (dist < 0.2 * deq[gt])


def job(args):
    key, gtnpz, threads = args
    fo = f"{OUT}/gtflow/{key}.npz"
    if os.path.exists(fo):
        return key, "skip", 0.0
    import torch
    torch.set_num_threads(threads)
    while not mem_ok():
        time.sleep(300)
    import common as C
    t0 = time.time()
    z = np.load(gtnpz); gt = z["gt"].astype(np.int64); vox = np.asarray(z["vox"], float)
    ratio = vox[0] / vox[1]; Zn = gt.shape[0]
    Zu, zi = C.zi_of(Zn, ratio)
    q = np.arange(Zu)
    nat = np.clip(np.round(q * (Zn - 1) / max(Zu - 1, 1)).astype(int), 0, Zn - 1)
    zz, yy, xx = np.nonzero(gt)
    t = np.zeros((3, *gt.shape), np.float32)
    if zz.size:
        y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        q0, q1 = zi[zz.min()], zi[zz.max()] + 1
        iso = gt[nat[q0:q1], y0:y1, x0:x1]
        with torch.no_grad():                                           # the snapshot's @torch.no_grad() decorator is on line 28
            mu = load_flows3d()(torch.from_numpy(np.ascontiguousarray(iso))).numpy()
        for n in range(Zn):
            k = zi[n] - q0
            if 0 <= k < mu.shape[1]:
                t[:, n, y0:y1, x0:x1] = mu[:, k]
    tn = np.sqrt((t ** 2).sum(0))
    A = (gt > 0) & (tn > 0.5)
    band = band_mask(gt.astype(np.int64), vox / vox[1]) & A
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    tmp = fo + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, t=t.astype(np.float16), band=band, n_a=int(A.sum()), n_b=int(band.sum()), n_c=int((A & ~band).sum()),
                        Zu=Zu, ratio=ratio, sec=time.time() - t0)
    os.replace(tmp, fo)
    return key, "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    runs = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))
    src = runs["gt_S1_s0"]
    keys = sorted(f[:-4] for f in os.listdir(f"{src}/masks") if f.endswith(".npz"))
    if a.only:
        keys = [k for k in keys if k in a.only]
    jobs = [(k, f"{src}/masks/{k}.npz", a.threads) for k in keys]
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    if a.workers <= 1:
        for j in jobs:
            print(*job(j), flush=True)
    else:
        from multiprocessing import Pool
        with Pool(a.workers) as p:
            for r in p.imap_unordered(job, jobs):
                print(*r, flush=True)


if __name__ == "__main__":
    main()
