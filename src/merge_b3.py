"""把第三批标准化结果并入主 manifest：剔除 NISNet3D 的坏块与疑似未标完的块，备份原 manifest。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import csv, os, shutil
STD = f"{_NSP_ROOT}/data/std"
FIELDS = ["ds", "id", "split", "dim", "z", "y", "x", "voxel_z", "voxel_y", "voxel_x", "n_nuclei", "img_dtype", "image", "mask", "source", "note"]
new = list(csv.DictReader(open(f"{STD}/manifest_nisnet_zf_mouse_embryo_bbbc050_bbbc032_parhyale_ratbrain_mesospim.csv")))
keep, drop = [], []
for r in new:
    n = int(r["n_nuclei"])
    if r["ds"] == "nisnet_zf" and (n > 100000 or n < 20):
        drop.append((r["id"], n)); continue
    keep.append(r)
for i, n in drop:
    print("剔除", i, n)
    for sub in ("images", "masks"):
        p = f"{STD}/nisnet_zf/{sub}/{i}.tif"
        if os.path.exists(p): os.remove(p)
old = list(csv.DictReader(open(f"{STD}/manifest.csv")))
names = {r["ds"] for r in keep}
old = [r for r in old if r["ds"] not in names]
shutil.copy(f"{STD}/manifest.csv", f"{STD}/manifest_before_b3.csv")
rows = sorted(old + keep, key=lambda r: (r["ds"], r["split"], r["id"]))
with open(f"{STD}/manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("manifest 行数", len(rows), "新增", len(keep))
