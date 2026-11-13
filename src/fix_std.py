"""修正标准化数据中两处已核实的问题（在 standardize.py 之后运行，就地更新 data/std 与 manifest.csv）。

1. NIS3D：Drosophila_1/2、MusMusculus_1/2 的 Info.txt 写 1×1×1 µm，但真值细胞核在 z 方向只有 xy 方向的 1/2.5–1/4，
   说明真实 z 间距更大。按“细胞核大致为球形”，用整个体积（train+test 两部分）真值的 xy/z 尺寸比估计有效 z 间距，
   写回 manifest 与 TIFF 元数据。Zebrafish_1（0.43×0.43×2.5）与 Zebrafish_2（尺寸比约 1.2）保持原值。
2. Purdue：多数子体积的标签只有 3–6 个编号，每个覆盖一大片细胞核，是语义标签而非实例。
   对编号数 ≤ 10 的体积按 3D 连通域（6 邻接）重新拆分为实例；相互贴在一起的核会被合成一个。
用法: python fix_std.py
"""
import csv
import os

import numpy as np
import tifffile
from scipy import ndimage as ndi

from stddata import STD

FIELDS = ["ds", "id", "split", "dim", "z", "y", "x", "voxel_z", "voxel_y", "voxel_x", "n_nuclei",
          "img_dtype", "image", "mask", "source", "note"]


def extents(lab):
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        c = np.argwhere(lab[sl] == k)
        if len(c) > 20:
            out.append(2 * np.sqrt(5) * c.std(0))
    return np.median(np.array(out), 0)


def rewrite(path, arr, voxel):
    tifffile.imwrite(path, arr, imagej=arr.dtype in (np.uint8, np.uint16, np.float32), resolution=(1 / voxel[2], 1 / voxel[1]),
                     metadata={"spacing": voxel[0], "unit": "micron", "axes": "ZYX"},
                     compression="zlib", compressionargs={"level": 1})


rows = list(csv.DictReader(open(f"{STD}/manifest.csv")))
# 1) NIS3D 有效 z 间距
fix_names = ("Drosophila_1", "Drosophila_2", "MusMusculus_1", "MusMusculus_2")
for name in fix_names:
    parts = [r for r in rows if r["ds"] == "nis3d" and r["id"].startswith(name)]
    labs = [tifffile.imread(os.path.join(STD, r["mask"])).astype(np.int32) for r in parts]
    e = np.median(np.array([extents(l) for l in labs]), 0)
    vz = round(float((e[1] + e[2]) / 2 / e[0]), 2)          # xy 体素为 1 µm
    print(f"  NIS3D {name}: 真值细胞核尺寸 (z,y,x) 像素 = {np.round(e, 1)}  → 有效 z 间距 {vz} µm")
    for r in parts:
        vox = (vz, 1.0, 1.0)
        for key in ("image", "mask"):
            p = os.path.join(STD, r[key]); rewrite(p, tifffile.imread(p), vox)
        r["voxel_z"] = vz
        r["note"] = (r["note"] + "；" if r["note"] else "") + f"Info.txt 写 1×1×1 µm，按真值核形状估计有效 z 间距为 {vz} µm"
# 2) Purdue 语义标签 → 连通域实例
for r in rows:
    if r["ds"] != "purdue" or int(r["n_nuclei"]) > 10:
        continue
    p = os.path.join(STD, r["mask"])
    lab = tifffile.imread(p)
    cc, n = ndi.label(lab > 0)
    sizes = np.bincount(cc.ravel()); small = np.where(sizes < 30)[0]
    cc[np.isin(cc, small[small > 0])] = 0
    ids = np.unique(cc); ids = ids[ids > 0]
    lut = np.zeros(int(cc.max()) + 1, np.uint16); lut[ids] = np.arange(1, len(ids) + 1)
    cc = lut[cc]
    tifffile.imwrite(p, cc, compression="zlib", compressionargs={"level": 1})
    print(f"  Purdue {r['id'][:40]}: 编号 {r['n_nuclei']} → 连通域实例 {len(ids)}")
    r["n_nuclei"] = len(ids)
    r["note"] = (r["note"] + "；" if r["note"] else "") + "原标签为语义标签，已按 3D 连通域拆成实例（贴在一起的核会合成一个）"
with open(f"{STD}/manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("manifest.csv 已更新")
