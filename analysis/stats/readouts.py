"""S7 的逐掩膜生物学读数（CPU）：核数、核体积中位数、最近邻距离中位数。
最终掩膜与 test_r6.hits() 完全相同：阈值 t = 规则 T 选出的阈值（原版固定 cp−2），t = −2 时用存下的 pred，否则由存下的 dP/cp 用
cellpose.dynamics.compute_masks 重算；再删去体积 < 0.15·预期体积的小块。核数与评测记录的 n_pred 逐个核对。
体积 = 体素数 × prod(vox)，距离按 vox 缩放的质心（vox 来自清单，部分数据集 xy 记为 1 像素，因此单位按数据集不同，Bland–Altman 主口径用百分比差）。
真值读数取同一 npz 里的 gt。结果逐个缓存为 cache/ro/<运行目录名>/<键>__t<t>.json。
用法：python readouts.py [--seeds s0 s1000 ...] [--workers 48]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats'))
import ncio  # noqa: E402

RO = f"{ncio.CACHE}/ro"


def readout(lab, vox):
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    n = int(len(ids))
    out = {"n": n, "vol_med": float(np.median(cnt) * float(np.prod(vox))) if n else None, "nn_med": None}
    if n >= 2:
        c = np.array(ndi.center_of_mass(np.ones_like(lab, np.uint8), lab, ids)).reshape(-1, 3) * np.asarray(vox, float)
        dd, _ = cKDTree(c).query(c, k=2)
        out["nn_med"] = float(np.median(dd[:, 1]))
    return out


def final_mask(z, t, frac=0.15):
    gt = z["gt"]; d = float(z["d"]); vox = z["vox"]
    if t == -2.0:
        pm = z["pred"]
    else:
        from cellpose import dynamics
        dP, cp = z["dP"].astype(np.float32), z["cp"].astype(np.float32)
        pm = dynamics.compute_masks(dP, cp, cellprob_threshold=t, flow_threshold=0.4, min_size=15, max_size_fraction=0.4,
                                    niter=int(200 / (30.0 / d)), do_3D=True)
        pm = np.asarray(pm).squeeze()
        assert pm.shape == gt.shape, (pm.shape, gt.shape)
    ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(pm.ravel())
    small = np.nonzero(cnt < frac * ev)[0]
    return np.where(np.isin(pm, small[small > 0]), 0, pm)


def job(a):
    npz, t, want_gt, n_pred = a
    run = os.path.basename(os.path.dirname(os.path.dirname(npz)))
    key = os.path.basename(npz)[:-4]
    p = f"{RO}/{run}/{key}__t{t:g}.json"
    pg = f"{RO}/gt/{key}.json"
    if os.path.exists(p) and (not want_gt or os.path.exists(pg)):
        return p
    z = np.load(npz)
    vox = np.asarray(z["vox"], float)
    pm = final_mask(z, t)
    r = readout(pm, vox)
    r.update(npz=npz, t=t, n_pred_rec=n_pred, n_match=bool(r["n"] == n_pred))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(r, open(p, "w"))
    if want_gt and not os.path.exists(pg):
        g = readout(z["gt"], vox)
        g.update(vox=vox.tolist(), d=float(z["d"]))
        os.makedirs(os.path.dirname(pg), exist_ok=True)
        json.dump(g, open(pg, "w"))
    return p


def jobs_for(seeds):
    J, seen_gt = [], set()
    for mode in ncio.MODES:
        M, _ = ncio.load_mode(mode, seeds)
        for nm, L in M.items():
            for R in L:
                for x, r in R.items():
                    key = f"{x[0]}__{x[1]}"
                    J.append((r["npz"], float(r["t"]), key not in seen_gt, int(r["n_pred"])))
                    seen_gt.add(key)
    return J


def load_ro(npz, t):
    run = os.path.basename(os.path.dirname(os.path.dirname(npz)))
    return json.load(open(f"{RO}/{run}/{os.path.basename(npz)[:-4]}__t{float(t):g}.json"))


def load_gt_ro(ds, vid):
    return json.load(open(f"{RO}/gt/{ds}__{vid}.json"))


def main():
    seeds = sys.argv[sys.argv.index("--seeds") + 1:] if "--seeds" in sys.argv else None
    if seeds:
        seeds = [s for s in seeds if s.startswith("s") and s[1:].isdigit()]
    else:
        import nc_stats
        seeds = nc_stats.complete_seeds()
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 48
    J = jobs_for(seeds)
    # 重算 t ≠ −2 的放前面（慢）
    J.sort(key=lambda a: a[1] == -2.0)
    print(f"种子 {seeds}：{len(J)} 个掩膜，其中 t ≠ −2 的 {sum(a[1] != -2.0 for a in J)} 个", flush=True)
    from multiprocessing import Pool
    done = 0
    with Pool(w) as pool:
        for _ in pool.imap_unordered(job, J, chunksize=1):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(J)}", flush=True)
    bad = [a for a in J if not load_ro(a[0], a[1])["n_match"]]
    print(f"完成；核数与评测记录 n_pred 不一致的掩膜 {len(bad)} 个", flush=True)
    for a in bad[:20]:
        r = load_ro(a[0], a[1]); print("  不一致", a[0], a[1], r["n"], r["n_pred_rec"])


if __name__ == "__main__":
    main()
