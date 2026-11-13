"""第三批：体素未知的数据按“细胞核大致为球形”估计有效 z 间距（xy 记为 1），写回 manifest 与 TIFF 元数据。
已知体素的数据只打印估计值作对照，不改。用法: python fix_std3.py [--dry]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import csv, os, sys
import numpy as np, tifffile
from scipy import ndimage as ndi
STD = f"{_NSP_ROOT}/data/std"
FIELDS = ["ds", "id", "split", "dim", "z", "y", "x", "voxel_z", "voxel_y", "voxel_x", "n_nuclei", "img_dtype", "image", "mask", "source", "note"]
UNKNOWN = ("nisnet_zf", "bbbc050", "ratbrain", "mesospim")
CHECK = ("mouse_embryo", "parhyale", "bbbc032")
def extents(lab):
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        c = np.argwhere(lab[sl] == k)
        if len(c) > 20: out.append(2 * np.sqrt(5) * c.std(0))
    return np.array(out)
rows = list(csv.DictReader(open(f"{STD}/manifest.csv")))
est = {}
for ds in UNKNOWN + CHECK:
    rr = [r for r in rows if r["ds"] == ds]
    rr = [rr[int(i)] for i in np.linspace(0, len(rr) - 1, min(15, len(rr)))]
    E = np.concatenate([extents(tifffile.imread(os.path.join(STD, r["mask"])).astype(np.int32)) for r in rr])
    e = np.median(E, 0); ratio = float((e[1] + e[2]) / 2 / e[0])
    est[ds] = ratio
    known = rr[0]["voxel_z"] and float(rr[0]["voxel_z"]) / float(rr[0]["voxel_y"])
    print(f"{ds:13s} 核 {len(E):5d}  中位尺寸(z,y,x) 像素 {np.round(e, 1)}  估计 z/xy = {ratio:.2f}" + (f"  已知 {known:.2f}" if known else ""))
if "--dry" in sys.argv: sys.exit()
for r in rows:
    if r["ds"] in UNKNOWN:
        vz = round(est[r["ds"]], 2); vox = (vz, 1.0, 1.0)
        for key in ("image", "mask"):
            p = os.path.join(STD, r[key]); a = tifffile.imread(p)
            tifffile.imwrite(p, a, imagej=a.dtype in (np.uint8, np.uint16, np.float32), resolution=(1.0, 1.0),
                             metadata={"spacing": vz, "unit": "pixel", "axes": "ZYX"}, compression="zlib", compressionargs={"level": 1})
        r["voxel_z"], r["voxel_y"], r["voxel_x"] = vz, 1.0, 1.0
        r["note"] = (r["note"] + "；" if r["note"] else "") + f"体素未知，按真值核形状估计 z/xy = {vz}（xy 记为 1 像素）"
with open(f"{STD}/manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("已写回")
