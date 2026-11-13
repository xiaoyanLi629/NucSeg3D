"""标准化数据集的读取接口（数据由 standardize.py 生成，说明见 data/std/README.md）。

    from stddata import load_manifest, load
    rows = load_manifest(ds=["cellos", "tapenade"], split="train", dim="3d")
    for r in rows:
        img, mask = load(r)                  # numpy 数组，3D 为 (z, y, x)
        voxel = r["voxel"]                   # (vz, vy, vx) µm，未知时为 None
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import csv
import os

import numpy as np
import tifffile

STD = f"{_NSP_ROOT}/data/std"


def load_manifest(ds=None, split=None, dim=None, path=None):
    rows = []
    for r in csv.DictReader(open(path or f"{STD}/manifest.csv")):
        if ds and r["ds"] not in ([ds] if isinstance(ds, str) else ds):
            continue
        if split and r["split"] != split:
            continue
        if dim and r["dim"] != dim:
            continue
        for k in ("z", "y", "x", "n_nuclei"):
            r[k] = int(r[k])
        r["voxel"] = tuple(float(r[k]) for k in ("voxel_z", "voxel_y", "voxel_x")) if r["voxel_z"] else None
        rows.append(r)
    return rows


def load(r, image=True, mask=True):
    """返回 (image, mask)；image 为 float32，mask 为整数实例标签。"""
    img = tifffile.imread(os.path.join(STD, r["image"])).astype(np.float32) if image else None
    lab = tifffile.imread(os.path.join(STD, r["mask"])) if mask else None
    return img, lab


def load_conf(r):
    """NIS3D 的逐核标注置信度（其他数据集返回 None）。"""
    p = os.path.join(STD, r["ds"], "conf", r["id"] + ".tif")
    return tifffile.imread(p) if os.path.exists(p) else None


def load_ignore(r):
    """NIS3D 的“未定义区域”掩码（bool，与图像同尺寸；没有时返回 None）。该区域内既无标注也不应计入评测。"""
    p = os.path.join(STD, r["ds"], "ignore", r["id"] + ".tif")
    return tifffile.imread(p).astype(bool) if os.path.exists(p) else None
