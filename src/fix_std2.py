"""修正 NIS3D 的“未定义区域”（在 fix_std.py 之后运行，就地更新 data/std/nis3d 与 manifest.csv）。

NIS3D 的 MusMusculus_1/2 与 Zebrafish_2 在图像模糊处用一个巨大的标签把整片区域盖住（Info.txt 称 undefined mask，
置信度图中为 1），它不是细胞核。体积超过该样本细胞核中位体积 100 倍的标签即为此区域（真实细胞核最大约 11 倍）。
处理：从实例标签中去掉，单独存为 data/std/nis3d/ignore/<id>.tif（uint8，1=未定义）；标签重新连续编号。
训练切片时把未定义区域的图像抹成背景；评测时去掉大部分落在未定义区域里的预测（见 stddata.load_ignore）。
用法: python fix_std2.py
"""
import csv
import os

import numpy as np
import tifffile

from stddata import STD

FIELDS = ["ds", "id", "split", "dim", "z", "y", "x", "voxel_z", "voxel_y", "voxel_x", "n_nuclei",
          "img_dtype", "image", "mask", "source", "note"]


def rewrite(path, arr, voxel):
    tifffile.imwrite(path, arr, imagej=arr.dtype in (np.uint8, np.uint16, np.float32), resolution=(1 / voxel[2], 1 / voxel[1]),
                     metadata={"spacing": voxel[0], "unit": "micron", "axes": "ZYX"},
                     compression="zlib", compressionargs={"level": 1})

rows = list(csv.DictReader(open(f"{STD}/manifest.csv")))
os.makedirs(f"{STD}/nis3d/ignore", exist_ok=True)
for r in rows:
    if r["ds"] != "nis3d":
        continue
    p = os.path.join(STD, r["mask"]); lab = tifffile.imread(p)
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    big = ids[cnt > 100 * np.median(cnt)]
    if not len(big):
        continue
    ign = np.isin(lab, big)
    lab[ign] = 0
    keep = np.unique(lab); keep = keep[keep > 0]
    lut = np.zeros(int(lab.max()) + 1, lab.dtype); lut[keep] = np.arange(1, len(keep) + 1)
    vox = (float(r["voxel_z"]), float(r["voxel_y"]), float(r["voxel_x"]))
    rewrite(p, lut[lab], vox)
    tifffile.imwrite(f"{STD}/nis3d/ignore/{r['id']}.tif", ign.astype(np.uint8), compression="zlib", compressionargs={"level": 1})
    print(f"  {r['id']}: 去掉未定义区域 {len(big)} 个标签、{ign.mean():.1%} 的体素；细胞核 {r['n_nuclei']} → {len(keep)}")
    r["n_nuclei"] = len(keep)
    r["note"] = (r["note"] + "；" if r["note"] else "") + "未定义区域已移出标签，见 ignore/"
w = csv.DictWriter(open(f"{STD}/manifest.csv", "w", newline=""), fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
