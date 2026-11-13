"""在 BP 第 14 页用的 3DCellScope 演示图（PDAC 类器官，109×512×512）上跑 Cellpose 模型。
参数与 NucSeg3D 横评一致：diameter=31.25 px（由原版 DeepStar3D 输出估计），anisotropy=3.28。
用法: python pdac_run.py <模型路径或名称> <输出标签>
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import sys, time
import numpy as np, tifffile
from cellpose import models
img = tifffile.imread(f"{_NSP_SHARED}/data/ds3d_ref/images/PDAC.tif").astype(np.float32)
m = models.CellposeModel(gpu=True, pretrained_model=sys.argv[1])
t0 = time.time()
lab, _, _ = m.eval(img, do_3D=True, z_axis=0, diameter=31.251275489877024, anisotropy=3.2820084548239077, flow3D_smooth=1, batch_size=32)
tifffile.imwrite(f"{_NSP_ROOT}/results/pdac/{sys.argv[2]}.tif", lab.astype(np.uint16), compression="zlib")
print(sys.argv[2], "细胞核", int(lab.max()), f"{time.time()-t0:.0f}s", flush=True)
