"""大鼠脑 4 个块中 #19 是 512×512（xy 约放大 4 倍），其余是 128×128；fix_std3 对整套只估了一个 z/xy（1.68），
这里改为逐样本估计：#19 为 6.58，其余三块 1.54–1.60 取 1.58。用法: python fix_std5.py"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import csv, os
import numpy as np, tifffile

STD = f"{_NSP_ROOT}/data/std"
FIELDS = ["ds", "id", "split", "dim", "z", "y", "x", "voxel_z", "voxel_y", "voxel_x", "n_nuclei", "img_dtype", "image", "mask", "source", "note"]


def main():
    rows = list(csv.DictReader(open(f"{STD}/manifest.csv")))
    for r in rows:
        if r["ds"] != "ratbrain":
            continue
        vz = 6.58 if "19_019" in r["id"] else 1.58
        for key in ("image", "mask"):
            p = os.path.join(STD, r[key]); a = tifffile.imread(p)
            tifffile.imwrite(p, a, imagej=a.dtype in (np.uint8, np.uint16, np.float32), resolution=(1.0, 1.0),
                             metadata={"spacing": vz, "unit": "pixel", "axes": "ZYX"}, compression="zlib", compressionargs={"level": 1})
        r["voxel_z"] = vz
        r["note"] = f"体素未知，按该样本真值核形状估计 z/xy = {vz}（#19 为 512×512，xy 约放大 4 倍，故逐样本估计）"
    with open(f"{STD}/manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print("大鼠脑已改为逐样本体素")


if __name__ == "__main__":
    main()
