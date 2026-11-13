"""r9 泄漏审计：16 套数据 train / val / test 之间的图像级重复与重叠。
只读 data/std 的图像、标签与清单（以及 raw/ 下核实坐标用的原始整幅图），不读任何模型输出或评测结果，不用 GPU；
单进程、逐体积逐层流式读取，读完一个文件就丢掉它的页缓存。val = val_split.json 里的体积（train 划分的一部分），train = 其余 train 划分。
子命令（结果都写在 $NUCSEG3D_SCRATCH/r9/leak/）：
  fp        每个体积一个指纹：块平均网格 hi（≤32×128×128，z-score 后存 float16）、z 向最大投影 mip（原分辨率，z-score 后存 float16）、
            逐层均值 zprof、像素 md5；每套数据一个 fp/<ds>.npz 与 fp/<ds>.json，已有的跳过。
  pairs     每套数据内两两比较：fp3（hi 重采样到 16×32×32）与 mip64（mip 重采样到 64×64）的相关，同尺寸的再比 hi 网格本身；
            列出最高相关 ≥ 0.90 的对（≥ 0.95 另标），以及每个 test 体积在 train/val 里、每个 val 体积在 train 里最像的体积。
  contain   形状不同、可能是裁块：小体积在大体积里做归一化互相关模板匹配（skimage match_template），先降采样粗搜再原分辨率复核。
  coords    带坐标的编号：nisnet_zf 把裁块放回原始整幅图里定位，核实坐标含义并算块之间的重叠；purdue 按坐标假设算重叠，
            并对同一原始体积的子体积两两做带掩码的归一化互相关（能发现任意平移下的部分重叠）。
  seq       blastospim、parhyale、bbbc050 的延时序列：每个 test 体积与同序列 train/val 体积的时间距离和 hi 相关。
  nis3d     nis3d 的 xL75 与 xR25 是否来自同一原始体积、是否有重叠列。
  labels    同一图像的多份标注：核数、逐核匹配（IoU ≥ 0.5）的 F1 与匹配核的平均 IoU。
用法：python leak_audit.py fp|pairs|contain|coords|seq|nis3d|labels [--ds 数据集 ...]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

STD = f"{_NSP_ROOT}/data/std"
RAW = f"{_NSP_ROOT}/data/raw"
OUT = f"{_NSP_SCRATCH}/r9/leak"
ALL_DS = ["bbbc032", "bbbc050", "blastospim", "celegans", "cellos", "crypt", "gonuclear", "mesospim", "mouse_embryo",
          "nis3d", "nisnet_zf", "nucverse3d", "parhyale", "purdue", "ratbrain", "tapenade"]


# ---------------------------------------------------------------- 清单与读图
def manifest():
    rows = list(csv.DictReader(open(f"{STD}/manifest.csv")))
    val = json.load(open(f"{STD}/val_split.json"))
    vs = {(d, i) for d, ids in val.items() for i in ids}
    for r in rows:
        r["part"] = "test" if r["split"] == "test" else ("val" if (r["ds"], r["id"]) in vs else "train")
    return rows


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_vol(path, dtype=np.float32):
    """整读一个体积（只用于小体积或少数复核），读完丢页缓存。"""
    import tifffile
    a = tifffile.imread(path)
    drop_cache(path)
    return a.astype(dtype) if dtype is not None else a


def edges(n, nb):
    nb = max(1, min(n, nb))
    return np.linspace(0, n, nb + 1).round().astype(int)


def block2d(a, ey, ex):
    s = np.add.reduceat(np.add.reduceat(a, ey[:-1], axis=0), ex[:-1], axis=1)
    return s / (np.diff(ey)[:, None] * np.diff(ex)[None, :])


def zs(a):
    a = np.asarray(a, np.float32)
    return (a - a.mean()) / (a.std() + 1e-9)


def fp_volume(path):
    import tifffile
    md = hashlib.md5()
    with tifffile.TiffFile(path) as t:
        shp = t.series[0].shape
        pages = [t.pages[0]] if len(shp) == 2 else t.pages
        Z, Y, X = (1, *shp) if len(shp) == 2 else shp
        assert len(pages) == Z, (path, len(pages), shp)
        ez, ey, ex = edges(Z, 32), edges(Y, 128), edges(X, 128)
        zb = np.searchsorted(ez, np.arange(Z), side="right") - 1
        hi = np.zeros((len(ez) - 1, len(ey) - 1, len(ex) - 1), np.float64)
        mip = None; zprof = np.zeros(Z, np.float64); s1 = s2 = 0.0; lo, hi_v = np.inf, -np.inf
        dt = None
        for z in range(Z):
            a = pages[z].asarray()
            dt = str(a.dtype)
            md.update(np.ascontiguousarray(a).tobytes())
            f = a.astype(np.float32)
            hi[zb[z]] += block2d(f, ey, ex)
            mip = f.copy() if mip is None else np.maximum(mip, f)
            zprof[z] = f.mean(); s1 += float(f.sum(dtype=np.float64)); s2 += float((f.astype(np.float64) ** 2).sum())
            lo = min(lo, float(f.min())); hi_v = max(hi_v, float(f.max()))
    drop_cache(path)
    hi /= np.diff(ez)[:, None, None]
    n = Z * Y * X
    return dict(shape=[Z, Y, X], dtype=dt, md5=md.hexdigest(), mean=s1 / n, std=float(np.sqrt(max(s2 / n - (s1 / n) ** 2, 0))),
                min=lo, max=hi_v), zs(hi).astype(np.float16), zs(mip).astype(np.float16), zprof.astype(np.float32)


def cmd_fp(a):
    rows = manifest()
    os.makedirs(f"{OUT}/fp", exist_ok=True)
    order = sorted(set(r["ds"] for r in rows), key=lambda d: sum(1 for r in rows if r["ds"] == d) * (50 if d == "blastospim" else 1))
    for ds in (a.ds or order):
        if os.path.exists(f"{OUT}/fp/{ds}.npz") and not a.force:
            print("已有", ds, flush=True); continue
        R = [r for r in rows if r["ds"] == ds]
        meta, arr = [], {}
        t0 = time.time()
        for k, r in enumerate(R):
            m, hi, mip, zp = fp_volume(f"{STD}/{r['image']}")
            m.update(ds=ds, id=r["id"], part=r["part"], split=r["split"], voxel=[r["voxel_z"], r["voxel_y"], r["voxel_x"]],
                     n_nuclei=int(r["n_nuclei"]), source=r["source"], note=r["note"])
            meta.append(m)
            arr[f"{r['id']}|hi"] = hi; arr[f"{r['id']}|mip"] = mip; arr[f"{r['id']}|zprof"] = zp
            if (k + 1) % 50 == 0:
                print(f"  {ds} {k + 1}/{len(R)} {time.time() - t0:.0f}s", flush=True)
        np.savez(f"{OUT}/fp/{ds}.npz.part.npz", **arr)
        os.replace(f"{OUT}/fp/{ds}.npz.part.npz", f"{OUT}/fp/{ds}.npz")
        json.dump(meta, open(f"{OUT}/fp/{ds}.json", "w"), indent=0, ensure_ascii=False)
        print(f"指纹 {ds}：{len(R)} 个体积，{time.time() - t0:.0f} 秒", flush=True)


def load_fp(ds):
    meta = json.load(open(f"{OUT}/fp/{ds}.json"))
    z = np.load(f"{OUT}/fp/{ds}.npz")
    return meta, z


# ---------------------------------------------------------------- 两两比较
def resample(a, shape):
    from scipy.ndimage import zoom
    a = np.asarray(a, np.float32)
    return zoom(a, [s / n for s, n in zip(shape, a.shape)], order=1, prefilter=False)


def cmd_pairs(a):
    os.makedirs(f"{OUT}/pairs", exist_ok=True)
    summ = {}
    for ds in (a.ds or ALL_DS):
        meta, z = load_fp(ds)
        n = len(meta)
        ids = [m["id"] for m in meta]
        F3 = np.stack([zs(resample(z[f"{i}|hi"], (16, 32, 32) if meta[k]["shape"][0] > 1 else (1, 32, 32))).ravel()
                       for k, i in enumerate(ids)])
        M64 = np.stack([zs(resample(z[f"{i}|mip"], (64, 64))).ravel() for i in ids])
        C3 = F3 @ F3.T / F3.shape[1]; Cm = M64 @ M64.T / M64.shape[1]
        Ch = np.full((n, n), np.nan, np.float32)
        byshape = defaultdict(list)
        for k, m in enumerate(meta):
            byshape[tuple(m["shape"])].append(k)
        for shp, K in byshape.items():
            if len(K) > 1:
                H = np.stack([np.asarray(z[f"{ids[k]}|hi"], np.float32).ravel() for k in K])
                C = H @ H.T / H.shape[1]
                Ch[np.ix_(K, K)] = C
        md5 = defaultdict(list)
        for k, m in enumerate(meta):
            md5[m["md5"]].append(k)
        exact = [[ids[k] for k in v] for v in md5.values() if len(v) > 1]
        best = np.fmax(np.fmax(C3, Cm), Ch)
        P = []
        for i in range(n):
            for j in range(i + 1, n):
                if best[i, j] >= 0.90:
                    P.append(dict(a=ids[i], pa=meta[i]["part"], b=ids[j], pb=meta[j]["part"], same_shape=bool(meta[i]["shape"] == meta[j]["shape"]),
                                  c_hi=None if np.isnan(Ch[i, j]) else round(float(Ch[i, j]), 4), c_fp3=round(float(C3[i, j]), 4),
                                  c_mip=round(float(Cm[i, j]), 4), best=round(float(best[i, j]), 4)))
        P.sort(key=lambda p: -p["best"])

        def nearest(k, allowed):
            cand = [j for j in range(n) if j != k and meta[j]["part"] in allowed]
            if not cand:
                return None
            out = {}
            for nm, C in (("hi", Ch), ("fp3", C3), ("mip", Cm)):
                v = [(C[k, j], j) for j in cand if not np.isnan(C[k, j])]
                if v:
                    c, j = max(v)
                    out[nm] = dict(id=ids[j], part=meta[j]["part"], c=round(float(c), 4))
            return out
        near_test = {ids[k]: nearest(k, ("train", "val")) for k in range(n) if meta[k]["part"] == "test"}
        near_val = {ids[k]: nearest(k, ("train",)) for k in range(n) if meta[k]["part"] == "val"}
        cats = Counter()
        for p in P:
            key = "-".join(sorted([p["pa"], p["pb"]]))
            cats[(key, "≥0.95" if p["best"] >= 0.95 else "0.90–0.95")] += 1
        # 参照分布：要查的划分组合（test 对 train/val、val 对 train）里全部对的相关分位数
        iu = [(i, j) for i in range(n) for j in range(i + 1, n) if cross(meta[i]["part"], meta[j]["part"])]
        calib = {}
        for nm, C in (("fp3", C3), ("mip", Cm), ("hi_same_shape", Ch)):
            v = np.array([C[i, j] for i, j in iu], np.float64)
            v = v[np.isfinite(v)]
            if len(v):
                calib[nm] = dict(n=int(len(v)), q50=round(float(np.quantile(v, 0.5)), 3), q90=round(float(np.quantile(v, 0.9)), 3),
                                 q99=round(float(np.quantile(v, 0.99)), 3), max=round(float(v.max()), 3))
        res = dict(ds=ds, n=n, parts=dict(Counter(m["part"] for m in meta)), exact_md5=exact, pairs=P, calib=calib,
                   counts={f"{k[0]} {k[1]}": v for k, v in sorted(cats.items())}, nearest_test=near_test, nearest_val=near_val)
        json.dump(res, open(f"{OUT}/pairs/{ds}.json", "w"), indent=1, ensure_ascii=False)
        summ[ds] = dict(n=n, parts=res["parts"], exact=len(exact), counts=res["counts"])
        print(f"== {ds}：{n} 个体积 {res['parts']}；像素 md5 完全相同的组 {len(exact)}；≥0.90 的对 {len(P)}：{res['counts']}", flush=True)
        for p in [p for p in P if "test" in (p["pa"], p["pb"]) and p["pa"] != p["pb"]][:12]:
            print(f"    test 跨划分 {p['best']:.3f} {p['a']}（{p['pa']}）~ {p['b']}（{p['pb']}） hi {p['c_hi']} fp3 {p['c_fp3']} mip {p['c_mip']}")
    json.dump(summ, open(f"{OUT}/pairs/_summary.json", "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- 原分辨率复核（同尺寸的对）
def corr_stream(pa, pb):
    """两个同尺寸体积在原分辨率下的皮尔逊相关与是否逐像素相同；两边各一次一层流式读。"""
    import tifffile
    n = 0; sa = sb = saa = sbb = sab = 0.0; equal = True
    with tifffile.TiffFile(pa) as ta, tifffile.TiffFile(pb) as tb:
        assert len(ta.pages) == len(tb.pages)
        for z in range(len(ta.pages)):
            x = ta.pages[z].asarray(); y = tb.pages[z].asarray()
            if equal and (x.shape != y.shape or not np.array_equal(x, y)):
                equal = False
            x = x.astype(np.float64).ravel(); y = y.astype(np.float64).ravel()
            n += x.size; sa += x.sum(); sb += y.sum(); saa += x @ x; sbb += y @ y; sab += x @ y
    drop_cache(pa); drop_cache(pb)
    cov = sab / n - (sa / n) * (sb / n); va = saa / n - (sa / n) ** 2; vb = sbb / n - (sb / n) ** 2
    return float(cov / np.sqrt(max(va * vb, 1e-30))), equal


def cmd_verify(a):
    """对 pairs 里跨划分、同尺寸、指纹最高相关 ≥ 0.95 的对（再加上每个 test 体积在 train/val 里 hi 最像的那个），原分辨率复核。"""
    os.makedirs(f"{OUT}/verify", exist_ok=True)
    rows = {(r["ds"], r["id"]): r for r in manifest()}
    for ds in (a.ds or ALL_DS):
        P = json.load(open(f"{OUT}/pairs/{ds}.json"))
        cand = {}
        for p in P["pairs"]:
            if p["same_shape"] and cross(p["pa"], p["pb"]) and p["best"] >= 0.95:
                cand[(p["a"], p["b"])] = dict(p)
        for t, nb in P["nearest_test"].items():
            if nb and nb.get("hi") and nb["hi"]["c"] >= 0.80:
                key = tuple(sorted([t, nb["hi"]["id"]]))
                cand.setdefault(key, dict(a=key[0], b=key[1], c_hi=nb["hi"]["c"], from_nearest_test=True))
        L = sorted(cand.values(), key=lambda p: -(p.get("c_hi") or 0))[:a.max_verify]
        res = []
        for p in L:
            ra, rb = rows[(ds, p["a"])], rows[(ds, p["b"])]
            c, eq = corr_stream(f"{STD}/{ra['image']}", f"{STD}/{rb['image']}")
            res.append(dict(p, pa=ra["part"], pb=rb["part"], corr_full=round(c, 5), pixels_identical=eq, same_mask_file=ra["mask"] == rb["mask"]))
        json.dump(res, open(f"{OUT}/verify/{ds}.json", "w"), indent=1, ensure_ascii=False)
        n99 = sum(r["corr_full"] >= 0.99 for r in res); n95 = sum(r["corr_full"] >= 0.95 for r in res)
        print(f"== {ds}：复核 {len(res)} 对（候选 {len(cand)}），原分辨率相关 ≥0.99 的 {n99} 对、≥0.95 的 {n95} 对，逐像素相同 "
              f"{sum(r['pixels_identical'] for r in res)} 对", flush=True)
        for r in res[:8]:
            print(f"    {r['corr_full']:.4f} {r['a']}（{r['pa']}）~ {r['b']}（{r['pb']}） hi {r.get('c_hi')} 同像素 {r['pixels_identical']}", flush=True)


# ---------------------------------------------------------------- 模板匹配（裁块）
def ncc_match(big, small):
    """small 在 big 里的归一化互相关最大值与位置（skimage match_template，pad_input=False，所以 small 完全在 big 里面）。"""
    from skimage.feature import match_template
    if any(s > b for s, b in zip(small.shape, big.shape)):
        return None
    if small.std() < 1e-6 or big.std() < 1e-6:
        return None
    r = match_template(big.astype(np.float32), small.astype(np.float32), pad_input=False)
    k = np.unravel_index(int(np.argmax(r)), r.shape)
    return float(r[k]), [int(x) for x in k]


def down(a, f):
    """各轴按整数因子 f（标量或元组）块平均降采样。"""
    f = (f,) * a.ndim if np.isscalar(f) else f
    sl = tuple(slice(0, (n // k) * k) for n, k in zip(a.shape, f))
    a = a[sl]
    shp = [x for n, k in zip(a.shape, f) for x in (n // k, k)]
    return a.reshape(shp).mean(axis=tuple(range(1, 2 * a.ndim, 2)))


def cross(pa, pb):
    """要查的划分组合：test 对 train/val，val 对 train。"""
    return ("test" in (pa, pb) and pa != pb) or ({pa, pb} == {"train", "val"})


def contain_pairs(meta):
    """三维包含：小体积在每个轴上都能放进大体积，体素一致，形状不同，划分组合要查。返回 (小, 大) 下标。"""
    out = []
    for i, j in itertools.permutations(range(len(meta)), 2):
        a, b = meta[i], meta[j]
        if a["voxel"] == b["voxel"] and a["shape"] != b["shape"] and cross(a["part"], b["part"]) \
                and all(s <= t for s, t in zip(a["shape"], b["shape"])):
            out.append((i, j))
    return out


def masked_best(A, B, ratio):
    """A、B 任意平移、重叠至少 ratio（占较小者）时的最大归一化互相关（skimage 的带掩码互相关，Padfield 2012）。"""
    from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
    cc = cross_correlate_masked(A, B, np.ones(A.shape, bool), np.ones(B.shape, bool), mode="full", axes=tuple(range(A.ndim)),
                                overlap_ratio=ratio)
    cc = np.where(np.isfinite(cc), np.real(cc), -1.0)
    k = np.unravel_index(int(np.argmax(cc)), cc.shape)
    return float(cc[k]), [int(x) for x in k]


# 三维包含：(粗搜降采样因子, 复核因子)；1 表示原分辨率。粗搜 NCC ≥ 0.6 的才复核。
C3D = {"tapenade": (2, 1), "mesospim": (1, 1), "bbbc050": (1, 1), "cellos": (2, 1), "nucverse3d": ((1, 4, 4), (1, 2, 2)),
       "celegans": ((2, 2, 2), 1), "mouse_embryo": ((2, 4, 4), (1, 2, 2))}
CMIP = {"bbbc050": 1, "cellos": 1, "celegans": 2, "mouse_embryo": 4, "gonuclear": 4, "nucverse3d": 4, "mesospim": 1, "tapenade": 1, "nis3d": 4}


def cmd_contain(a):
    """A：三维包含（match_template，小体积完全在大体积里）；B：z 向最大投影上的任意平移部分重叠（带掩码互相关，重叠 ≥ 25%）。"""
    os.makedirs(f"{OUT}/contain", exist_ok=True)
    rows = {(r["ds"], r["id"]): r for r in manifest()}
    for ds in (a.ds or sorted(set(C3D) | set(CMIP))):
        meta, z = load_fp(ds)
        ids = [m["id"] for m in meta]
        t0 = time.time()
        if ds in C3D:
            fc, ff = C3D[ds]
            pairs = sorted(contain_pairs(meta), key=lambda p: (p[1], p[0]))
            cache, nbytes = {}, [0]

            def vol(k):
                if k not in cache:
                    if nbytes[0] > 3 * 2**30:                          # 缓存不超过约 3 GB
                        cache.clear(); nbytes[0] = 0
                    cache[k] = read_vol(f"{STD}/{rows[(ds, ids[k])]['image']}")
                    nbytes[0] += cache[k].nbytes
                return cache[k]
            sc = lambda v, f: v if f == 1 else down(v, f)
            res = []
            print(f"== {ds} 三维包含：{len(pairs)} 个候选对，粗搜因子 {fc}，复核因子 {ff}", flush=True)
            for n_, (i, j) in enumerate(pairs):
                sm, bg = vol(i), vol(j)
                c = ncc_match(sc(bg, fc), sc(sm, fc))
                if c is None:
                    continue
                rec = dict(small=ids[i], ps=meta[i]["part"], big=ids[j], pb=meta[j]["part"], ncc_coarse=round(c[0], 4), pos_coarse=c[1],
                           coarse_factor=fc)
                if fc == ff:
                    rec.update(ncc_fine=rec["ncc_coarse"], pos_fine=c[1], fine_factor=ff)
                elif c[0] >= 0.6:
                    cf = ncc_match(sc(bg, ff), sc(sm, ff))
                    rec.update(ncc_fine=round(cf[0], 4), pos_fine=cf[1], fine_factor=ff)
                res.append(rec)
                if (n_ + 1) % 500 == 0:
                    print(f"   {n_ + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
            res.sort(key=lambda r: -(r.get("ncc_fine") or r["ncc_coarse"]))
            json.dump(res, open(f"{OUT}/contain/{ds}_3d.json", "w"), indent=1, ensure_ascii=False)
            print(f"   完成 {len(res)} 对，{time.time() - t0:.0f} 秒；最高：" + "；".join(
                f"{r['small']}({r['ps']})⊂{r['big']}({r['pb']}) {r.get('ncc_fine') or r['ncc_coarse']}" for r in res[:6]), flush=True)
            cache.clear()
        if ds in CMIP:
            f = CMIP[ds]
            mips = {k: down(np.asarray(z[f"{ids[k]}|mip"], np.float32), f) for k in range(len(meta))}
            res = []
            for i, j in itertools.combinations(range(len(meta)), 2):
                A, B = meta[i], meta[j]
                if not cross(A["part"], B["part"]) or A["voxel"][1:] != B["voxel"][1:]:
                    continue
                c, k = masked_best(mips[i], mips[j], 0.25)
                res.append(dict(a=ids[i], pa=A["part"], b=ids[j], pb=B["part"], same_shape=A["shape"] == B["shape"],
                                mip_overlap_ncc=round(c, 4), shift_index=k))
            res.sort(key=lambda r: -r["mip_overlap_ncc"])
            json.dump(dict(note=f"z 向最大投影、xy 各 1/{f} 降采样、任意平移且重叠 ≥ 25% 时的最大归一化互相关", pairs=res),
                      open(f"{OUT}/contain/{ds}_mip.json", "w"), indent=1, ensure_ascii=False)
            hi_ = [r for r in res if r["mip_overlap_ncc"] >= 0.8]
            print(f"== {ds} 投影部分重叠：{len(res)} 对，≥0.8 的 {len(hi_)} 对，{time.time() - t0:.0f} 秒；最高：" + "；".join(
                f"{r['a']}({r['pa']})~{r['b']}({r['pb']}) {r['mip_overlap_ncc']}" for r in res[:6]), flush=True)


# ---------------------------------------------------------------- 坐标
def cmd_coords(a):
    os.makedirs(f"{OUT}/coords", exist_ok=True)
    import tifffile
    from skimage.feature import match_template
    rows = manifest()
    out = {}
    if not a.ds or "nisnet_zf" in a.ds:
        # 1) 把每个裁块的中间层放回原始整幅图的同一层里定位（原图在 raw/nisnet_zf/EmbryoN/，层数与裁块相同）；
        #    同时试原分辨率与 2 倍降采样两种假设，取归一化互相关高的
        full = {"Embryo1": f"{RAW}/nisnet_zf/Embryo1/Sytox_Orangeonly_4hpf.tif", "Embryo2": f"{RAW}/nisnet_zf/Embryo2/channel2.tif",
                "Embryo3": f"{RAW}/nisnet_zf/Embryo3/elifeData/C2-wtstk1.tif"}
        info, loc = {}, {}
        for emb, p in full.items():
            with tifffile.TiffFile(p) as t:
                shp = t.series[0].shape
                info[emb] = dict(path=p, shape=list(shp))
                R = [r for r in rows if r["ds"] == "nisnet_zf" and r["id"].startswith(emb + "_")]
                if not R:
                    continue
                Zt = int(R[0]["z"])
                if len(shp) != 3 or Zt != shp[0]:
                    info[emb]["note"] = f"裁块层数 {Zt} 与原图 {shp} 不一致，没有定位"
                    continue
                zc = Zt // 2
                plane = t.pages[zc].asarray().astype(np.float32)
                plane2 = down(plane, 2)
                for n_, r in enumerate(R):
                    tile = read_vol(f"{STD}/{r['image']}")
                    tp = tile[zc]
                    r1 = match_template(plane, tp, pad_input=False); k1 = np.unravel_index(int(np.argmax(r1)), r1.shape)
                    r2 = match_template(plane2, tp, pad_input=False); k2 = np.unravel_index(int(np.argmax(r2)), r2.shape)
                    scale, k, c = (1, k1, float(r1[k1])) if r1[k1] >= r2[k2] else (2, (2 * k2[0], 2 * k2[1]), float(r2[k2]))
                    y0, x0 = [int(v) for v in r["id"].split("_")[1:3]]
                    same = None
                    if n_ < 6 and scale == 1 and c > 0.99:                # 前几个块：每 16 层逐像素核对一次
                        blk = np.stack([t.pages[zz].asarray()[k[0]:k[0] + tile.shape[1], k[1]:k[1] + tile.shape[2]] for zz in range(0, Zt, 16)])
                        same = bool(np.array_equal(blk.astype(np.float32), tile[::16]))
                    loc[r["id"]] = dict(part=r["part"], name_y=y0, name_x=x0, scale=scale, found_y=int(k[0]), found_x=int(k[1]),
                                        ncc=round(c, 4), ncc_other_scale=round(float(min(r1[k1], r2[k2])), 4),
                                        pixels_identical_every16z=same, footprint=[tile.shape[1] * scale, tile.shape[2] * scale])
            drop_cache(p)
            print(f"nisnet_zf {emb}：原图 {shp}，定位 {sum(1 for i in loc if i.startswith(emb + '_'))} 个裁块", flush=True)
        # 2) 用定位到的位置算块之间的重叠（同一胚胎内、不同划分的块）
        ov = []
        for (i, A), (j, B) in itertools.combinations(sorted(loc.items()), 2):
            if i.split("_")[0] != j.split("_")[0] or A["part"] == B["part"]:
                continue
            h, w = A["footprint"]
            dy, dx = abs(A["found_y"] - B["found_y"]), abs(A["found_x"] - B["found_x"])
            if dy < h and dx < w:
                ov.append(dict(a=i, pa=A["part"], b=j, pb=B["part"], overlap_frac=round((h - dy) * (w - dx) / (h * w), 3)))
        rel = Counter((v["found_y"] - 2 * v["name_y"], v["found_x"] - 2 * v["name_x"], v["scale"]) for v in loc.values())
        out["nisnet_zf"] = dict(info=info, located=loc, overlaps_cross_split=ov,
                                found_minus_2name_counts={str(k): c for k, c in rel.most_common(10)})
        print(f"nisnet_zf：不同划分之间重叠的块对 {len(ov)}；定位位置与 2×编号之差的分布 {rel.most_common(5)}", flush=True)
    if not a.ds or "purdue" in a.ds:
        from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
        R = [r for r in rows if r["ds"] == "purdue"]
        grp = defaultdict(list)
        for r in R:
            grp[re.search(r"ground_truth_and_synthetic/([^/]+)/", r["source"]).group(1)].append(r)
        res = []
        for g, L in sorted(grp.items()):
            vols = {r["id"]: read_vol(f"{STD}/{r['image']}") for r in L}
            for A, B in itertools.combinations(L, 2):
                va, vb = vols[A["id"]], vols[B["id"]]
                ca = [int(x) for x in re.findall(r"\d+", A["id"].replace(g, ""))]
                cb = [int(x) for x in re.findall(r"\d+", B["id"].replace(g, ""))]
                # 坐标假设：编号里的数是子体积在原体积里的位置（readme 原话：“the location of the subvolume in the original volume”），
                # 不知道是角点还是中心、各是哪个轴；这里按“前两个数是 xy 平面上的位置、块大小 128”算 xy 是否可能重叠
                d = [abs(x - y) for x, y in zip(ca[:2], cb[:2])]
                xy_overlap_possible = len(d) == 2 and all(v < 128 for v in d)
                cc = cross_correlate_masked(va, vb, np.ones_like(va, bool), np.ones_like(vb, bool), mode="full", axes=(0, 1, 2),
                                            overlap_ratio=0.05)
                k = np.unravel_index(int(np.nanargmax(cc)), cc.shape)
                shift = [int(k[ax]) - (vb.shape[ax] - 1) for ax in range(3)]
                ovf = float(np.prod([max(0, min(va.shape[ax], vb.shape[ax] + shift[ax]) - max(0, shift[ax])) for ax in range(3)])
                            / min(va.size, vb.size))
                res.append(dict(group=g, a=A["id"], pa=A["part"], b=B["id"], pb=B["part"], coords_a=ca, coords_b=cb,
                                xy_overlap_possible_by_name=xy_overlap_possible, masked_ncc=round(float(cc[k]), 4), shift_zyx=shift,
                                overlap_frac_at_best=round(ovf, 3)))
            print(f"purdue {g}：{len(L)} 个子体积，{len(L) * (len(L) - 1) // 2} 对", flush=True)
        res.sort(key=lambda r: -r["masked_ncc"])
        out["purdue"] = res
    json.dump(out, open(f"{OUT}/coords/coords{'_' + '_'.join(a.ds) if a.ds else ''}.json", "w"), indent=1, ensure_ascii=False)
    if "purdue" in out:
        for r in out["purdue"][:10]:
            print("   ", json.dumps(r, ensure_ascii=False))


# ---------------------------------------------------------------- 延时序列
def tkey(ds, i):
    """(序列键, 时间)。blastospim：BlastoSPIM1 的键是编号（F11、H5 等），时间是帧号；BlastoSPIM2 的编号 Blast_NNN 每个只有一帧，
    键统一记为 BlastoSPIM2:Blast，“时间”是 NNN（采集顺序号，不是真正的时间）。"""
    if ds == "blastospim":
        m = re.match(r"^(BlastoSPIM\d)_(?:train|val|test)(?:_[a-z]+SNR)?_(.+)_(\d+)_image_\d+$", i)
        return m.group(1) + ":" + m.group(2), int(m.group(3))
    if ds == "parhyale":
        return "parhyale", int(i[2:])
    if ds == "bbbc050":
        m = re.match(r"^(train|test)_(Emb\d+)_t(\d+)$", i)
        return m.group(2) + ("" if m.group(1) == "train" else "(test)"), int(m.group(3))
    raise KeyError(ds)


def cmd_seq(a):
    os.makedirs(f"{OUT}/seq", exist_ok=True)
    out = {}
    for ds in (a.ds or ["blastospim", "parhyale", "bbbc050"]):
        meta, z = load_fp(ds)
        ids = [m["id"] for m in meta]
        H = {i: np.asarray(z[f"{i}|hi"], np.float32).ravel() for i in ids}
        info = {i: (tkey(ds, i), tuple(m["shape"]), m["part"]) for i, m in zip(ids, meta)}
        corr = lambda i, j: float(H[i] @ H[j] / H[i].size) if info[i][1] == info[j][1] else None
        # 组内相邻帧（同编号、时间最近）的相关分布，按划分组合分开
        adj = defaultdict(list)
        byk = defaultdict(list)
        for i in ids:
            byk[info[i][0][0]].append(i)
        for k, L in byk.items():
            L = sorted(L, key=lambda i: info[i][0][1])
            for x, y in zip(L, L[1:]):
                c = corr(x, y)
                if c is not None:
                    adj["-".join(sorted([info[x][2], info[y][2]]))].append((info[y][0][1] - info[x][0][1], c))
        adj_s = {k: dict(n=len(v), dt_median=float(np.median([d for d, _ in v])), c_min=round(min(c for _, c in v), 4),
                         c_median=round(float(np.median([c for _, c in v])), 4), c_max=round(max(c for _, c in v), 4)) for k, v in adj.items()}
        # 每个 test 体积：同编号里时间最近的 train/val 帧，以及同尺寸序列里相关最高的 train/val 帧
        rec = {}
        for i in ids:
            if info[i][2] != "test":
                continue
            (key, t), shp, _ = info[i]
            same = [j for j in byk[key] if info[j][2] in ("train", "val")]
            near = min(same, key=lambda j: abs(info[j][0][1] - t)) if same else None
            ser = [j for j in ids if info[j][1] == shp and info[j][2] in ("train", "val")]
            bestj = max(ser, key=lambda j: corr(i, j)) if ser else None
            rec[i] = dict(key=key, t=t, n_trainval_same_key=len(same),
                          nearest_same_key=None if near is None else dict(id=near, part=info[near][2], dt=info[near][0][1] - t, c_hi=corr(i, near)),
                          best_same_shape=None if bestj is None else dict(id=bestj, part=info[bestj][2], key=info[bestj][0][0],
                                                                           dt=info[bestj][0][1] - t if info[bestj][0][0] == key else None,
                                                                           c_hi=round(corr(i, bestj), 4)))
        out[ds] = dict(adjacent=adj_s, test=rec)
        if ds == "blastospim":
            # BlastoSPIM2 里 xy 尺寸是 BlastoSPIM1 转置的体积：hi 网格转置（及 4 种翻转）后与 BlastoSPIM1 同尺寸体积比
            tr = {}
            for i, m in zip(ids, meta):
                if not i.startswith("BlastoSPIM2"):
                    continue
                Zs, Ys, Xs = m["shape"]
                cand = [j for j, mj in zip(ids, meta) if j.startswith("BlastoSPIM1") and mj["shape"] == [Zs, Xs, Ys]]
                if not cand:
                    continue
                h = np.asarray(z[f"{i}|hi"], np.float32).transpose(0, 2, 1)
                best = max((float(np.ravel(v) @ H[j] / H[j].size), j) for j in cand
                           for v in (h, h[:, ::-1], h[:, :, ::-1], h[:, ::-1, ::-1]))
                tr[i] = dict(part=m["part"], best_b1=best[1], best_b1_part=info[best[1]][2], c=round(best[0], 4))
            out[ds]["b2_transposed_vs_b1"] = tr
            print(f"   BlastoSPIM2 转置与 BlastoSPIM1 比：{len(tr)} 个，最大相关 {max([v['c'] for v in tr.values()], default=None)}", flush=True)
        n_same = sum(1 for v in rec.values() if v["n_trainval_same_key"] > 0)
        print(f"== {ds}：test {len(rec)} 个，其中与 train/val 同编号（同一胚胎/序列）的 {n_same} 个；相邻帧相关：{json.dumps(adj_s, ensure_ascii=False)}", flush=True)
    json.dump(out, open(f"{OUT}/seq/seq.json", "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- nis3d 的左 75% 与右 25%
def cmd_nis3d(a):
    import tifffile
    rows = [r for r in manifest() if r["ds"] == "nis3d"]
    out = []
    for r in rows:
        if not r["id"].endswith("_xR25"):
            continue
        base = r["id"][:-5]
        L = next(x for x in rows if x["id"] == base + "_xL75")
        with tifffile.TiffFile(f"{RAW}/{r['source'][4:]}") as t:
            raw_shape = list(t.series[0].shape)
        A = read_vol(f"{STD}/{L['image']}"); B = read_vol(f"{STD}/{r['image']}")
        # 若两块在 x 上重叠 k 列，左块最后 k 列应与右块最前 k 列相同（相关 ≈ 1）；逐个 k 算相关，取最大
        cs = [(float(np.corrcoef(A[:, :, -k:].ravel(), B[:, :, :k].ravel())[0, 1]), k) for k in range(1, 33)]
        c_ov, k_ov = max(cs)
        # 参照：边界两侧紧邻的两列（同一体积切开时相关较高）与两块最远两端的列
        c_edge = float(np.corrcoef(A[:, :, -1].ravel(), B[:, :, 0].ravel())[0, 1])
        c_far = float(np.corrcoef(A[:, :, 0].ravel(), B[:, :, -1].ravel())[0, 1])
        out.append(dict(volume=base, L=L["id"], L_part=L["part"], R=r["id"], R_part=r["part"], L_shape=list(A.shape), R_shape=list(B.shape),
                        raw_shape=raw_shape, widths_sum=A.shape[2] + B.shape[2], best_overlap_corr=round(c_ov, 4), best_overlap_k=k_ov,
                        corr_boundary_cols=round(c_edge, 4), corr_far_cols=round(c_far, 4)))
        del A, B
        print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    os.makedirs(f"{OUT}/nis3d", exist_ok=True)
    json.dump(out, open(f"{OUT}/nis3d/nis3d.json", "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- 多份标注
def match_labels(la, lb):
    a = la.ravel().astype(np.int64); b = lb.ravel().astype(np.int64)
    na = np.bincount(a); nb = np.bincount(b)
    m = (a > 0) & (b > 0)
    key = a[m] * (int(b.max()) + 1) + b[m]
    u, c = np.unique(key, return_counts=True)
    ia, ib = u // (int(b.max()) + 1), u % (int(b.max()) + 1)
    iou = c / (na[ia] + nb[ib] - c)
    order = np.argsort(-iou)
    used_a, used_b, tp, ious = set(), set(), 0, []
    for k in order:
        if iou[k] < 0.5:
            break
        if ia[k] in used_a or ib[k] in used_b:
            continue
        used_a.add(ia[k]); used_b.add(ib[k]); tp += 1; ious.append(float(iou[k]))
    A = int((na[1:] > 0).sum()); B = int((nb[1:] > 0).sum())
    return dict(n_a=A, n_b=B, matched=tp, f1=round(2 * tp / max(A + B, 1), 4), mean_iou_matched=round(float(np.mean(ious)), 4) if ious else None,
                fg_iou=round(float(((a > 0) & (b > 0)).sum() / max(((a > 0) | (b > 0)).sum(), 1)), 4))


def cmd_labels(a):
    """输入 [{ds, a, b, pos?}]：a、b 同尺寸时整体比较；给了 pos（a 在 b 里的 z,y,x 起点）时把 b 在该位置裁出与 a 同尺寸的一块再比。
    比较图像（逐像素相同？相关）与标签（核数、逐核匹配 F1、匹配核平均 IoU、前景 IoU）。只比在两边都完整落在块内的核时，
    裁块边缘被切开的核会被算作不匹配，所以裁块情况下的 F1 是下限。"""
    rows = {(r["ds"], r["id"]): r for r in manifest()}
    pairs = json.load(open(a.pairs_json)) if a.pairs_json else []
    out = []
    for p in pairs:
        ra, rb = rows[(p["ds"], p["a"])], rows[(p["ds"], p["b"])]
        ia = read_vol(f"{STD}/{ra['image']}"); ib = read_vol(f"{STD}/{rb['image']}")
        la = read_vol(f"{STD}/{ra['mask']}", None); lb = read_vol(f"{STD}/{rb['mask']}", None)
        if p.get("pos") is not None:
            z0, y0, x0 = p["pos"]
            sl = (slice(z0, z0 + ia.shape[0]), slice(y0, y0 + ia.shape[1]), slice(x0, x0 + ia.shape[2]))
            ib, lb = ib[sl], lb[sl]
        rec = dict(p, pa=ra["part"], pb=rb["part"], shape_a=list(ia.shape), shape_b_used=list(ib.shape), same_mask_file=ra["mask"] == rb["mask"])
        if ia.shape == ib.shape:
            rec.update(img_identical=bool(np.array_equal(ia, ib)), img_corr=round(float(np.corrcoef(ia.ravel(), ib.ravel())[0, 1]), 5),
                       img_max_abs_diff=float(np.abs(ia - ib).max()))
            rec.update(match_labels(la, lb))
        del ia, ib, la, lb
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(f"{OUT}/labels", exist_ok=True)
    json.dump(out, open(f"{OUT}/labels/labels.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fp", "pairs", "verify", "contain", "coords", "seq", "nis3d", "labels"])
    ap.add_argument("--ds", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max_verify", type=int, default=150, help="verify：每套至多复核多少对")
    ap.add_argument("--pairs_json", default=None, help="labels：要比较标注的体积对 [{ds, a, b}, ...]")
    a = ap.parse_args()
    dict(fp=cmd_fp, pairs=cmd_pairs, verify=cmd_verify, contain=cmd_contain, coords=cmd_coords, seq=cmd_seq, nis3d=cmd_nis3d,
         labels=cmd_labels)[a.cmd](a)
