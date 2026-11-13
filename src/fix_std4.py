"""NISNet3D 三个胚胎的 z 间距不同（层数 348 / 188 / 122，按真值核形状估计 z/xy = 1.12 / 2.57 / 2.18），
fix_std3.py 按整套数据只估了一个值（2.29），这里改为逐胚胎；同时去掉 fix_std3 误跑两次留下的重复备注。
用法: python fix_std4.py"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import csv, os
import numpy as np, tifffile

STD = f"{_NSP_ROOT}/data/std"
FIELDS = ["ds", "id", "split", "dim", "z", "y", "x", "voxel_z", "voxel_y", "voxel_x", "n_nuclei", "img_dtype", "image", "mask", "source", "note"]
PER_EMBRYO = {"Embryo1": 1.12, "Embryo2": 2.57, "Embryo3": 2.18}


def main():
    rows = list(csv.DictReader(open(f"{STD}/manifest.csv")))
    for r in rows:
        if r["note"]:
            parts = []
            for p in r["note"].split("；"):
                if p not in parts:
                    parts.append(p)
            r["note"] = "；".join(parts)
        if r["ds"] != "nisnet_zf":
            continue
        vz = PER_EMBRYO[r["id"].split("_")[0]]
        for key in ("image", "mask"):
            p = os.path.join(STD, r[key]); a = tifffile.imread(p)
            tifffile.imwrite(p, a, imagej=a.dtype in (np.uint8, np.uint16, np.float32), resolution=(1.0, 1.0),
                             metadata={"spacing": vz, "unit": "pixel", "axes": "ZYX"}, compression="zlib", compressionargs={"level": 1})
        r["voxel_z"] = vz
        r["note"] = f"体素未知，按该胚胎真值核形状估计 z/xy = {vz}（xy 记为 1 像素；三个胚胎 z 间距不同，逐胚胎估计）"
    with open(f"{STD}/manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print("NISNet3D 已改为逐胚胎体素", PER_EMBRYO)


if __name__ == "__main__":
    main()
