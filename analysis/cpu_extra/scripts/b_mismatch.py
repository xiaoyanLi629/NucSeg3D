"""列出 g_best 推出的 tp5/tp7 与评测记录不一致的体积，并标出规则 T 阈值与 z/xy（看是否都是 vol_pass 已知的 z 插值近似）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, json, os, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/cpu_extra/scripts'))
from cpux_common import OUT, SEEDS, models
M, _ = models()
rows = []
for f in sorted(glob.glob(f"{OUT}/bvol/*.npz")):
    k = os.path.basename(f)[:-4]; x = tuple(k.split("__", 1)); v = np.load(f)
    ratio = float(v["vox"][0] / v["vox"][1])
    for nm in ("S1", "2D"):
        for i, s in enumerate(SEEDS):
            b = v[f"best_gt_{nm}_{s}"]; r = M[nm][i][x]
            d5 = int((b >= 0.5).sum()) - r["tp5"]; d7 = int((b >= 0.7).sum()) - r["tp7"]
            if d5 or d7:
                rows.append(dict(key=k, model=f"{nm}_{s}", t=r["t"], ratio=ratio, d_tp5=d5, d_tp7=d7, tp5=r["tp5"], tp7=r["tp7"]))
json.dump(rows, open(f"{OUT}/b_mismatch.json", "w"), indent=1)
print(len(rows), "个（体积×模型种子）有差；全部 t≠−2 且 z/xy≥2：", all(r["t"] != -2.0 and r["ratio"] >= 2 for r in rows))
print("最大 |Δtp5| / tp5：", max(abs(r["d_tp5"]) / max(r["tp5"], 1) for r in rows), " 最大 |Δtp7| / tp7：", max(abs(r["d_tp7"]) / max(r["tp7"], 1) for r in rows))
for r in rows[:8]:
    print(r)
