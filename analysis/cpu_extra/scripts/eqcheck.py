"""提速版 b_nuc.job_test 与原版已算出的逐体积结果逐数组比对（只为核对实现等价）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob
import sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/cpu_extra/scripts'))
import b_nuc
b_nuc.OUT = f"{_NSP_SHARED}/nc_cpu_extra/out_eqcheck"
for pre in ("mouse_embryo", "tapenade", "nis3d", "bbbc050"):
    k = sorted(glob.glob(f"{_NSP_SHARED}/nc_cpu_extra/out/bvol/{pre}__*.npz"))[0].split("/")[-1][:-4]
    b_nuc.job_test(k)
    a = np.load(f"{_NSP_SHARED}/nc_cpu_extra/out/bvol/{k}.npz"); b = np.load(f"{b_nuc.OUT}/bvol/{k}.npz")
    print(k, all(np.array_equal(a[x], b[x], equal_nan=True) for x in a.files), flush=True)
