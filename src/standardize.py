"""把各套原始数据整理成统一的标准格式，一次性处理掉已知的格式问题。

输出 $NUCSEG3D_ROOT/data/std/：
  <ds>/images/<id>.tif   3D 为 (z, y, x)，2D 为 (y, x)；整数图像保持 uint8/uint16，浮点统一为 float32
  <ds>/masks/<id>.tif    同尺寸实例标签，0 为背景，1..N 连续编号（uint16，超过 65535 用 uint32）
  <ds>/conf/<id>.tif     仅 NIS3D：每个核的标注置信度
  manifest.csv           每行一个样本：ds,id,split,dim,z,y,x,voxel_z,voxel_y,voxel_x,n_nuclei,img_dtype,image,mask,source,note
TIFF 写入 ImageJ 元数据（spacing = z 体素，resolution = 1/xy 体素，单位 micron），用 Fiji 打开即为正确物理尺寸。
已处理的问题：Cellos 多页 tif 只读到首层、Hoechst 掩码文件名小写；NIS3D MusMusculus_2 命名不同；
Tapenade square.tif 为 17 个独立块叠在一起（拆开）；BlastoSPIM 的 npy 与 Go-Nuclear 的 h5 转 tif；
NIS3D 按 x 方向切成 train（左 75%）与 test（右 25%）两个文件；隐窝支架损坏的一张、Purdue 缺标签的一个样本剔除。
划分（split）在这里一次定死，后续脚本直接读 manifest。
用法: python standardize.py [--workers 24] [--only ds ...]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse
import csv
import os
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import tifffile

from cpft_data import samples, read, voxel_of

STD = f"{_NSP_ROOT}/data/std"
FIELDS = ["ds", "id", "split", "dim", "z", "y", "x", "voxel_z", "voxel_y", "voxel_x", "n_nuclei",
          "img_dtype", "image", "mask", "source", "note"]


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def relabel(lab):
    ids = np.unique(lab); ids = ids[ids > 0]
    dt = np.uint16 if len(ids) < 65535 else np.uint32
    lut = np.zeros(int(lab.max()) + 1, dt); lut[ids] = np.arange(1, len(ids) + 1, dtype=dt)
    return lut[lab], len(ids)


def norm_img(img):
    img = np.squeeze(img)
    if img.dtype in (np.uint8, np.uint16):
        return img
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.uint16) if img.max() < 65536 else img.astype(np.float32)
    return img.astype(np.float32)


def write(path, arr, voxel, compress=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(compression="zlib", compressionargs={"level": 1}) if compress else {}
    if arr.ndim == 3 and voxel:
        tifffile.imwrite(path, arr, imagej=arr.dtype != np.uint32 and arr.dtype != np.float64, resolution=(1 / voxel[2], 1 / voxel[1]),
                         metadata={"spacing": voxel[0], "unit": "micron", "axes": "ZYX"}, **kw)
    else:
        tifffile.imwrite(path, arr, **kw)


def split_rule(s, order):
    if s["split"] in ("train", "val"):
        return "train"
    if s["split"] == "test":
        return "test"
    if s["ds"] == "cellos":
        return "test" if order["cellos"].index(s["id"]) % 6 == 0 else "train"
    if s["ds"] == "gonuclear":
        return "test" if s["id"].startswith("1170") else "train"
    if s["ds"] == "purdue":
        return "test" if order["purdue"].index(s["id"]) % 5 == 0 else "train"
    if s["ds"] == "nis3d":
        return "split_x"
    return "train"


def work(job):
    s, split = job
    rows, ds = [], s["ds"]
    try:
        img = norm_img(read(s["img"]))
        lab = np.squeeze(read(s["lab"]))
        if img.shape != lab.shape:
            return [dict(ds=ds, id=s["id"], error=f"shape mismatch {img.shape} vs {lab.shape}")]
        vox = voxel_of(s)
        vox = tuple(float(v) for v in vox) if vox else None
        dim = "2d" if img.ndim == 2 else "3d"
        parts = [(sanitize(s["id"]), split, img, lab, "")]
        if split == "split_x":
            cut = int(img.shape[-1] * 0.75)
            parts = [(sanitize(s["id"]) + "_xL75", "train", img[..., :cut], lab[..., :cut], "NIS3D 体积 x 方向左 75%"),
                     (sanitize(s["id"]) + "_xR25", "test", img[..., cut:], lab[..., cut:], "NIS3D 体积 x 方向右 25%")]
        if ds == "gonuclear":       # clear / noisy 两个图像版本共用一份标签：标签按体积号存一次
            vol = sanitize(s["id"]).rsplit("_", 1)[0]
            parts = [(sanitize(s["id"]), split, img, lab, f"标签共用 masks/{vol}.tif")]
        for pid, sp, im, lb, note in parts:
            lb2, n = relabel(lb)
            if n == 0:
                continue
            ip = f"{STD}/{ds}/images/{pid}.tif"
            mid = pid.rsplit("_", 1)[0] if ds == "gonuclear" else pid
            mp = f"{STD}/{ds}/masks/{mid}.tif"
            write(ip, im, vox)
            if ds != "gonuclear" or pid.endswith("_clear"):     # Go-Nuclear 的标签只由 clear 版本写一次，避免并发写同一文件
                write(mp, lb2, vox)
            if ds == "nis3d" and s.get("conf") and os.path.exists(s["conf"]):
                c = np.squeeze(tifffile.imread(s["conf"]))
                if split == "split_x":
                    cut = int(c.shape[-1] * 0.75)
                    c = c[..., :cut] if pid.endswith("xL75") else c[..., cut:]
                write(f"{STD}/{ds}/conf/{pid}.tif", c, vox)
            shp = im.shape if im.ndim == 3 else (1,) + im.shape
            rows.append(dict(ds=ds, id=pid, split=sp, dim=dim, z=shp[0], y=shp[1], x=shp[2],
                             voxel_z=vox[0] if vox else "", voxel_y=vox[1] if vox else "", voxel_x=vox[2] if vox else "",
                             n_nuclei=n, img_dtype=str(im.dtype), image=os.path.relpath(ip, STD), mask=os.path.relpath(mp, STD),
                             source=s["img"].replace(f"{_NSP_ROOT}/data/raw/", "raw/"), note=note))
    except Exception as e:
        rows = [dict(ds=ds, id=s["id"], error=repr(e)[:200])]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    ALL = samples()
    order = defaultdict(list)
    for x in ALL:
        order[x["ds"]].append(x["id"])
    order = {k: sorted(v) for k, v in order.items()}
    S = [s for s in ALL if not a.only or s["ds"] in a.only]
    jobs = [(s, split_rule(s, order)) for s in S]
    # 大体积放前面，避免最后剩几个大任务拖尾
    jobs.sort(key=lambda j: {"blastospim": 0, "nis3d": 0, "gonuclear": 0}.get(j[0]["ds"], 1))
    rows, errs = [], []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
            for x in r:
                (errs if "error" in x else rows).append(x)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    os.makedirs(STD, exist_ok=True)
    mf = f"{STD}/manifest{'_' + '_'.join(a.only) if a.only else ''}.csv"
    rows.sort(key=lambda r: (r["ds"], r["split"], r["id"]))
    with open(mf, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    for e in errs:
        print("  剔除/出错:", e)
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        agg[r["ds"]][r["split"]][0] += 1; agg[r["ds"]][r["split"]][1] += r["n_nuclei"]
    for ds, d in sorted(agg.items()):
        print(f"  {ds:11s} " + "  ".join(f"{sp}: {n} 个样本 / {k} 个核" for sp, (n, k) in sorted(d.items())))
    print("写入", mf)


if __name__ == "__main__":
    main()
