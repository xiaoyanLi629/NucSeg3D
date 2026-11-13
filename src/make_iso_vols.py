"""生成真三维模型用的各向同性体积库 iso30（共享约定 true3d_spec.md 第 1 节）。

对每个三维训练体积（manifest 的 train 去掉 val_split.json 里的验证体积、去掉二维数据 crypt 和 mesospim）
和每个验证体积（val_split.json，共 88 个），按该体积真值细胞核的 xy 等效球直径 d（xy 像素，与 eval_cpsam.eq_diam_xy_px 相同）
缩放：factor_xy = 30 / d，factor_z = factor_xy × (vz / vxy)，缩放后体素各向同性、细胞核中位直径约 30 像素。
  图像：线性插值（与 ndi.zoom(order=1) 逐体素相同，实现上先逐层做 xy 二维缩放、再沿 z 线性插值，内存只占一份中间结果），
        再按 1–99 百分位归一化到 0..1（不截断，百分位在缩放前的原图上算），存 float16 .npy。
  标签：最近邻，用 make_slices_flow3d.zoom_index 的下标规则（与 ndi.zoom(order=0) 完全一致，与图像网格逐体素对齐），
        存 .npy；编号最大值 < 65535 时存 uint16，否则 uint32（index.json 里 lab_dtype 记录；读的时候按 np.load 的类型即可）。
  忽略区（NIS3D 的未定义区域）：另存 uint8 .npy（index.json 的 ign 字段），没有时为 null。
  缩放后体素数超过 --crop_vox（默认 1e8，主要是 BlastoSPIM）的体积，只存前景外接框外扩 16 个各向同性像素的部分。
输出目录（默认 $NUCSEG3D_SCRATCH/cpft_iso30）：<ds>/<name>_img.npy、<name>_lab.npy、[<name>_ign.npy]，以及 index.json：
  [{ds, name, split: "train"|"val", shape, factors: [fz, fy, fx], diam_xy_px, anis, n_nuclei, img, lab, ign, lab_dtype,
    voxel, crop0 (原图坐标下裁剪起点 [z,y,x]), orig_shape, p1, p99, diam_iso (缩放后标签实测的等效球直径，应≈30)}]
  img/lab/ign 是相对输出目录的路径。
用法: python make_iso_vols.py [--out DIR] [--workers 8] [--only ds ...] [--ids id ...] [--limit N] [--tar]
  --tar  建完后打包到共享盘 $NUCSEG3D_ROOT/iso30/iso30.tar（包内顶层目录名为输出目录名），供其他机器解包
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse
import json
import os
import re
import time

import numpy as np
from scipy import ndimage as ndi

from stddata import STD, load, load_ignore, load_manifest

TARGET_D = 30.0
MARGIN = 16
SPLIT_JSON = f"{STD}/val_split.json"
TAR = f"{_NSP_ROOT}/iso30/iso30.tar"


def zoom_index(n_in, f):
    """ndi.zoom(order=0) 沿一个轴按比例 f 取样时用到的输入下标（与 make_slices_flow3d.zoom_index 相同）。"""
    return ndi.zoom(np.arange(n_in, dtype=np.float64), f, order=0, mode="nearest").round().astype(int)


def label_counts(lab):
    """各编号的体素数（不含背景）；编号很大时改用 np.unique，避免 bincount 开巨大数组。"""
    mx = int(lab.max())
    if mx < 50_000_000:
        cnt = np.bincount(np.asarray(lab).ravel(), minlength=mx + 1)[1:]
    else:
        _, cnt = np.unique(lab[lab > 0], return_counts=True)
    return cnt[cnt > 0]


def eq_diam_xy_px(lab, voxel):
    cnt = label_counts(lab)
    if not len(cnt):
        return None
    return float(np.median((6 * cnt * np.prod(voxel) / np.pi) ** (1 / 3)) / voxel[1])


def cases(only=None, ids=None):
    val = json.load(open(SPLIT_JSON))
    vset = {(d, i) for d, v in val.items() for i in v}
    out = []
    for r in load_manifest(split="train", dim="3d"):
        if only and r["ds"] not in only:
            continue
        if ids and r["id"] not in ids:
            continue
        isval = (r["ds"], r["id"]) in vset
        if not isval and r["ds"] == "mesospim":
            continue
        out.append(dict(r, part="val" if isval else "train"))
    return out


def zoom_linear_to(img, f3, out):
    """img (Z,Y,X) float32 按 f3 线性缩放，写入 out（形状须等于 ndi.zoom 的输出形状）。
    三线性插值可分离：先每层做 xy 缩放，再沿 z 做一维线性插值（坐标 i·(Z−1)/(Zo−1)，与 ndi.zoom grid_mode=False 一致）。"""
    Z = img.shape[0]
    Zo, Yo, Xo = out.shape
    tmp = np.empty((Z, Yo, Xo), np.float32)
    for z in range(Z):
        tmp[z] = ndi.zoom(img[z], f3[1:], order=1) if f3[1] != 1 or f3[2] != 1 else img[z]
    zc = np.arange(Zo) * ((Z - 1) / (Zo - 1) if Zo > 1 else 0.0)
    i0 = np.clip(np.floor(zc).astype(int), 0, max(Z - 2, 0))
    t = (zc - i0).astype(np.float32)
    i1 = np.minimum(i0 + 1, Z - 1)
    for a in range(0, Zo, 16):
        b = min(a + 16, Zo)
        tt = t[a:b, None, None]
        out[a:b] = (1 - tt) * tmp[i0[a:b]] + tt * tmp[i1[a:b]]
    del tmp


def work(args):
    r, out_dir, crop_vox = args
    t0 = time.time()
    try:
        img, lab = load(r)
        ign = load_ignore(r)
        vox = r["voxel"] or (1.0, 1.0, 1.0)
        d = eq_diam_xy_px(lab, vox)
        if d is None:
            return dict(ds=r["ds"], name=r["id"], skip="没有细胞核")
        fxy = TARGET_D / d
        f3 = (fxy * vox[0] / vox[1], fxy, fxy)
        Z, Y, X = lab.shape
        c0, c1 = [0, 0, 0], [Z, Y, X]
        if Z * Y * X * np.prod(f3) > crop_vox:                       # 只留前景外接框外扩 MARGIN 个各向同性像素
            sl = ndi.find_objects((lab > 0).astype(np.uint8))[0]
            for a, (s, n, f) in enumerate(zip(sl, (Z, Y, X), f3)):
                m = int(np.ceil(MARGIN / f))
                c0[a], c1[a] = max(s.start - m, 0), min(s.stop + m, n)
        crop = tuple(slice(a, b) for a, b in zip(c0, c1))
        img, lab = img[crop], lab[crop]
        ign = ign[crop] if ign is not None else None
        # 归一化（1–99 百分位，抽样计算）
        x = img.ravel()
        step = max(1, x.size // 20_000_000)
        p1, p99 = (float(v) for v in np.percentile(x[::step], [1, 99]))
        img = (img.astype(np.float32) - p1) / max(p99 - p1, 1e-6)
        iz, iy, ix = (zoom_index(n, f) for n, f in zip(lab.shape, f3))
        shape = (len(iz), len(iy), len(ix))
        tag = re.sub("[^A-Za-z0-9_.-]", "_", r["id"])
        os.makedirs(f"{out_dir}/{r['ds']}", exist_ok=True)
        rel = {k: f"{r['ds']}/{tag}_{k}.npy" for k in ("img", "lab", "ign")}
        mx = int(lab.max())
        ldt = np.uint16 if mx < 65535 else np.uint32
        oi = np.lib.format.open_memmap(f"{out_dir}/{rel['img']}.part", "w+", np.float16, shape)
        zoom_linear_to(img, f3, oi)
        oi.flush(); del oi, img
        ol = np.lib.format.open_memmap(f"{out_dir}/{rel['lab']}.part", "w+", ldt, shape)
        lab = lab.astype(ldt)
        for a in range(0, shape[0], 32):
            ol[a:a + 32] = lab[np.ix_(iz[a:a + 32], iy, ix)]
        ol.flush(); del ol
        if ign is not None and ign.any():
            og = np.lib.format.open_memmap(f"{out_dir}/{rel['ign']}.part", "w+", np.uint8, shape)
            for a in range(0, shape[0], 32):
                og[a:a + 32] = ign[np.ix_(iz[a:a + 32], iy, ix)]
            og.flush(); del og
        else:
            rel["ign"] = None
        for k, p in rel.items():
            if p:
                os.replace(f"{out_dir}/{p}.part", f"{out_dir}/{p}")
        # 核对：缩放后标签的等效球直径（各向同性，单位像素）与细胞核数
        labm = np.load(f"{out_dir}/{rel['lab']}", mmap_mode="r")
        cnt = label_counts(labm)
        d_iso = float(np.median((6 * cnt / np.pi) ** (1 / 3))) if len(cnt) else float("nan")
        return dict(ds=r["ds"], name=tag, id=r["id"], split=r["part"], shape=list(shape), factors=[round(f, 6) for f in f3],
                    diam_xy_px=round(d, 3), anis=round(vox[0] / vox[1], 4), n_nuclei=int(len(cnt)),
                    img=rel["img"], lab=rel["lab"], ign=rel["ign"], lab_dtype=np.dtype(ldt).name, voxel=list(vox),
                    crop0=c0, orig_shape=[Z, Y, X], p1=p1, p99=p99, diam_iso=round(d_iso, 2), seconds=round(time.time() - t0, 1))
    except Exception as e:
        import traceback
        return dict(ds=r["ds"], name=r["id"], error=repr(e)[:300], tb=traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{_NSP_SCRATCH}/cpft_iso30")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop_vox", type=float, default=1e8)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int, default=0, help="每套数据至多几个体积（冒烟测试用）")
    ap.add_argument("--tar", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cs = cases(a.only, a.ids)
    if a.limit:
        keep, seen = [], {}
        for r in cs:
            seen[r["ds"]] = seen.get(r["ds"], 0) + 1
            if seen[r["ds"]] <= a.limit:
                keep.append(r)
        cs = keep
    # 大体积先做（缩放后体素数估计用 manifest 的形状），避免最后只剩一个大体积单独跑
    cs.sort(key=lambda r: -r["z"] * r["y"] * r["x"] * (r["voxel"] or (1, 1, 1))[0])
    print(f"iso30：{len(cs)} 个体积（train {sum(r['part'] == 'train' for r in cs)}，val {sum(r['part'] == 'val' for r in cs)}），"
          f"{a.workers} 个进程 -> {a.out}", flush=True)
    t0 = time.time()
    rows = []
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=20) as p:
        for k, x in enumerate(p.imap_unordered(work, [(r, a.out, a.crop_vox) for r in cs], chunksize=1)):
            rows.append(x)
            if "error" in x:
                print("  ERR", x["ds"], x["name"], x["error"], x["tb"], flush=True)
            elif k % 50 == 0 or len(cs) < 50:
                print(f"  {k + 1}/{len(cs)} {x['ds']:12s} {x['name'][-40:]:40s} {x.get('shape')} d={x.get('diam_xy_px')} "
                      f"-> {x.get('diam_iso')}  {x.get('seconds')}s  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    errs = [x for x in rows if "error" in x]
    ok = sorted([x for x in rows if "error" not in x and "skip" not in x], key=lambda x: (x["ds"], x["name"]))
    json.dump(ok, open(f"{a.out}/index.json", "w"), indent=0)
    nv = sum(np.prod(x["shape"]) for x in ok)
    print(f"ISO30_DONE {len(ok)} 个体积，{nv / 1e9:.2f} G 体素，出错 {len(errs)} 个，跳过 {sum('skip' in x for x in rows)} 个，"
          f"{(time.time() - t0) / 60:.1f} 分钟", flush=True)
    by = {}
    for x in ok:
        by.setdefault(x["ds"], []).append(x)
    for ds, xs in sorted(by.items()):
        print(f"  {ds:12s} n={len(xs):4d} diam_iso 中位 {np.median([x['diam_iso'] for x in xs]):5.1f} "
              f"[{min(x['diam_iso'] for x in xs):.1f}, {max(x['diam_iso'] for x in xs):.1f}]  {sum(np.prod(x['shape']) for x in xs) / 1e9:.2f} G 体素", flush=True)
    if errs:
        raise SystemExit("有体积出错，不打包")
    if a.tar:
        import subprocess
        os.makedirs(os.path.dirname(TAR), exist_ok=True)
        parent, nm = os.path.split(os.path.abspath(a.out))
        subprocess.run(["tar", "cf", TAR + ".part", "-C", parent, nm], check=True)
        os.replace(TAR + ".part", TAR)
        print("打包", TAR, os.path.getsize(TAR) // 2**30, "GB", flush=True)


if __name__ == "__main__":
    main()
