"""把清单里 mesospim 六个体积的体素改为源数据值 5×10.52×10.52 µm（z, y, x），先备份；README 追加说明。只执行一次。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import csv, os, shutil, time
D = f"{_NSP_ROOT}/data/std"; M = f"{D}/manifest.csv"; B = f"{D}/manifest_before_1102.csv"
assert not os.path.exists(B), "已经改过（备份已存在）"
shutil.copy2(M, B)
rows = list(csv.DictReader(open(B, encoding="utf-8"))); f = rows[0].keys()
n = 0
for r in rows:
    if r["ds"] == "mesospim":
        r["voxel_z"], r["voxel_y"], r["voxel_x"] = "5.0", "10.52", "10.52"
        r["note"] = "体素取源数据：IDR idr0066 experimentC（mesoSPIM 0.63×，10.52×10.52×5 µm）；原按真值核形状估的 z/xy 0.16 弃用（2026-11-02）"
        n += 1
assert n == 6, n
with open(M, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(f)); w.writeheader(); w.writerows(rows)
open(f"{D}/README.md", "a", encoding="utf-8").write(
    "\n## 2026-11-02 mesoSPIM 体素改为源数据值\n\nmesospim 六个体积（c1–c5、visual）的体素由按真值核形状估的 0.16/1/1 改为源数据 5×10.52×10.52 µm（z, y, x；z/xy 0.475）。"
    "依据：IDR idr0066 experimentC 注释表（github.com/IDR/idr0066-voigt-mesospim，experimentC/idr0066-experimentC-annotation.csv）写明 "
    "“taken using the mesoSPIM (zoom 0.63x) … 10.52 x 10.52 x 5 um”，CellSeg3D（Achard 等，eLife 2025）的标注体积取自该脑、在 0.63× 采集。"
    "原清单备份为 manifest_before_1102.csv。改动之前的评测读的是旧值。\n")
print(f"改了 {n} 行，备份 {B}，{time.ctime()}")
