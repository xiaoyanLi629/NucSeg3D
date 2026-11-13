"""分析 B 的逐真值核计算（DESIGN_CPUX 第 2 节）。
val：只算验证集真值核的局部信噪比 → bins.json（看测试预测之前）。
test：测试集每个真值核的相对深度、局部信噪比，以及 S1、二维各五种子的 g_best 与边界 IoU（只对 IoU ≥ 0.5 的匹配）。
用法：python b_nuc.py val | test"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/cpu_extra/scripts'))
from cpux_common import INP, NPROC, OUT, RUNMAP, SEEDS, SRC, np  # noqa: E402

TAGS = [f"gt_{nm}_{s}" for nm in ("S1", "2D") for s in SEEDS]


def case(ds, vid, split):
    sys.path.insert(0, SRC)
    from stddata import load_manifest
    from eval_cpsam import load_case, eq_diam_xy_px
    man = case.man.get(split)
    if man is None:
        man = case.man[split] = {(r["ds"], r["id"]): r for r in load_manifest(split=split, dim="3d")}
    r = man[(ds, vid)]
    img, gt, box, ign = load_case(r, 384)
    gt = gt.astype(np.int32)
    if ign is not None:
        gt = gt.copy(); gt[ign] = 0
    vox = r["voxel"] or (1.0, 1.0, 1.0)
    return np.asarray(img, np.float32), gt, ign, vox, float(eq_diam_xy_px(gt, vox))


case.man = {}


def sub(arr, lo, hi, pad_val=0):
    """按 [lo, hi) 取子块，越界部分用 pad_val 补。"""
    lo = np.asarray(lo); hi = np.asarray(hi)
    cl, ch = np.maximum(lo, 0), np.minimum(hi, arr.shape)
    s = arr[tuple(slice(a, b) for a, b in zip(cl, ch))]
    return np.pad(s, [(a - b, c - d) for a, b, c, d in zip(cl, lo, hi, ch)], constant_values=pad_val)


def per_nucleus(img, gt, ign, vox, D):
    from scipy import ndimage as ndi
    ratio = float(vox[0] / vox[1]); samp = (ratio, 1.0, 1.0)
    ids = np.unique(gt); ids = ids[ids > 0]
    sl = ndi.find_objects(gt)
    cen = np.array(ndi.center_of_mass(np.ones_like(gt, np.uint8), gt, ids)).reshape(-1, 3)
    depth = cen[:, 0] / max(gt.shape[0] - 1, 1)
    s = max(2.0, 0.25 * D)
    pz, pxy = int(np.ceil(s / ratio)) + 1, int(np.ceil(s)) + 1
    snr = np.full(len(ids), np.nan); nshell = np.zeros(len(ids), int)
    ignm = ign if ign is not None else np.zeros(gt.shape, bool)
    for k, i in enumerate(ids):
        z = sl[i - 1]
        lo = [z[0].start - pz, z[1].start - pxy, z[2].start - pxy]; hi = [z[0].stop + pz, z[1].stop + pxy, z[2].stop + pxy]
        g = sub(gt, lo, hi, -1); im = sub(img, lo, hi, 0); ig = sub(ignm, lo, hi, True)
        obj = g == i
        dist = ndi.distance_transform_edt(~obj, sampling=samp)
        shell = (dist > 0) & (dist <= s) & (g == 0) & (~ig)
        nshell[k] = int(shell.sum())
        if nshell[k] >= 20:
            sh = im[shell]; sd = float(sh.std())
            if sd > 0:
                snr[k] = (float(np.median(im[obj])) - float(np.median(sh))) / sd
    return ids, depth, snr, nshell


def biou(gt, pm, gi, pj, vox, d, bg_=None, bp_=None):
    """bg_/bp_：真值核与预测核的外接框（find_objects 的切片）；只为提速，结果与在整幅上求外接框相同。"""
    from scipy import ndimage as ndi
    samp = (float(vox[0] / vox[1]), 1.0, 1.0)
    lo = [min(a.start, b.start) - 1 for a, b in zip(bg_, bp_)]; hi = [max(a.stop, b.stop) + 1 for a, b in zip(bg_, bp_)]
    g = sub(gt, lo, hi, 0) == gi; p = sub(pm, lo, hi, 0) == pj
    bg = g & (ndi.distance_transform_edt(g, sampling=samp) <= d)
    bp = p & (ndi.distance_transform_edt(p, sampling=samp) <= d)
    iou = float((g & p).sum() / (g | p).sum())
    return float((bg & bp).sum() / max((bg | bp).sum(), 1)), iou


def job_val(key):
    ds, vid = key.split("__", 1)
    img, gt, ign, vox, D = case(ds, vid, "train")
    ids, depth, snr, nshell = per_nucleus(img, gt, ign, vox, D)
    return key, dict(snr=[None if np.isnan(x) else float(x) for x in snr], n=len(ids))


def job_test(key):
    p = f"{OUT}/bvol/{key}.npz"
    if os.path.exists(p):
        return key, "skip"
    ds, vid = key.split("__", 1)
    img, gt, ign, vox, D = case(ds, vid, "test")
    ids, depth, snr, nshell = per_nucleus(img, gt, ign, vox, D)
    d = max(1.0, 0.1 * D)
    pos = {int(i): k for k, i in enumerate(ids)}
    rec = dict(ids=ids, depth=depth, snr=snr, nshell=nshell, D=np.float64(D), d_band=np.float64(d), vox=np.asarray(vox, float))
    maxdiff = 0.0
    for tg in TAGS:
        z = np.load(f"{INP}/errcache/{tg}/{key}.npz")
        assert set(z["g_id"].tolist()) == set(ids.tolist()), (tg, key)
        best = np.zeros(len(ids)); best[[pos[int(i)] for i in z["g_id"]]] = z["g_best"]
        pm = z["mask"]
        bi = np.full(len(ids), np.nan)
        from scipy import ndimage as ndi
        gsl = ndi.find_objects(gt); psl = ndi.find_objects(pm); psz_all = np.bincount(pm.ravel()); gsz_all = np.bincount(gt.ravel())
        for k in np.nonzero(best >= 0.5)[0]:
            gi = int(ids[k]); sg = gsl[gi - 1]
            pl, cnt = np.unique(pm[sg][gt[sg] == gi], return_counts=True)
            cnt = cnt[pl > 0]; pl = pl[pl > 0]
            psz = psz_all[pl]; gsz = int(gsz_all[gi])
            j = int(pl[np.argmax(cnt / (gsz + psz - cnt))])
            bi[k], iou = biou(gt, pm, gi, j, vox, d, sg, psl[j - 1])
            maxdiff = max(maxdiff, abs(iou - best[k]))
        rec[f"best_{tg}"] = best; rec[f"biou_{tg}"] = bi
    rec["iou_check_maxdiff"] = np.float64(maxdiff)
    os.makedirs(f"{OUT}/bvol", exist_ok=True)
    np.savez_compressed(p + ".tmp.npz", **rec); os.replace(p + ".tmp.npz", p)
    return key, maxdiff


def main():
    mode = sys.argv[1]
    if mode == "val":
        vdir = RUNMAP["val_gt_S1_s0"]
        keys = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{vdir}/masks/*.npz") if not os.path.basename(f).startswith("bbbc032__"))
        with Pool(NPROC) as pool:
            res = dict(pool.imap_unordered(job_val, keys))
        by = {}
        for k, v in res.items():
            by.setdefault(k.split("__")[0], []).extend(x for x in v["snr"] if x is not None)
        allv = np.concatenate([np.array(v) for v in by.values()])
        pooled = [float(np.percentile(allv, 100 / 3)), float(np.percentile(allv, 200 / 3))]
        bins = {"depth_edges": [1 / 3, 2 / 3], "snr_pooled": pooled, "snr_per_ds": {}, "n_val_vol": len(keys),
                "n_val_nuclei_defined": {ds: len(v) for ds, v in by.items()},
                "n_val_nuclei_undefined": {ds: sum(sum(x is None for x in res[k]["snr"]) for k in res if k.split("__")[0] == ds) for ds in by}}
        for ds, v in by.items():
            bins["snr_per_ds"][ds] = ([float(np.percentile(v, 100 / 3)), float(np.percentile(v, 200 / 3))], "own") if len(v) >= 30 else (pooled, "pooled_fallback")
        os.makedirs(OUT, exist_ok=True)
        json.dump(bins, open(f"{OUT}/bins.json", "w"), ensure_ascii=False, indent=1)
        print(json.dumps(bins, ensure_ascii=False, indent=1))
    else:
        assert os.path.exists(f"{OUT}/bins.json"), "先算 bins"
        D = RUNMAP["test_gt"]
        keys = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{INP}/errcache/gt_S1_s0/*.npz"))
        with Pool(NPROC) as pool:
            n = 0
            for k, r in pool.imap_unordered(job_test, keys, chunksize=1):
                n += 1
                if n % 20 == 0:
                    print(n, "/", len(keys), flush=True)
        print("完成", len(keys))


if __name__ == "__main__":
    main()
