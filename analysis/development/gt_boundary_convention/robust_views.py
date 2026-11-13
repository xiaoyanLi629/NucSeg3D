#!/usr/bin/env python
"""Print robustness views from a summary.json: SNR>=10 subset medians and per-volume median ranges."""
import json
import sys

import numpy as np

S = json.load(open(sys.argv[1]))
K = ["r_face_z", "r_face_xy", "ratio_z", "ratio_xy", "grad_z", "grad_xy", "zr50", "xyr50", "vr50", "iou50",
     "lvl_vol", "best_lvl", "best_iou"]
dss = sorted(k for k in S if not k.startswith("_"))
print("SNR>=10 subset medians")
print("ds".ljust(13) + "n".rjust(5) + "".join(k.rjust(10) for k in K))
for d in dss:
    s = S[d]["snr_ge10"]
    vals = [(f"{s['metrics'][k]['med']:.2f}" if s["n"] else "-") for k in K]
    print(d.ljust(13) + str(s["n"]).rjust(5) + "".join(v.rjust(10) for v in vals))
print("\nper-volume medians, min..max across volumes")
K2 = ["r_face_z", "r_face_xy", "ratio_z", "ratio_xy", "grad_z", "grad_xy", "iou50"]
print("ds".ljust(13) + "".join(k.rjust(12) for k in K2))
for d in dss:
    pv = S[d]["per_volume"]
    cells = [f"{np.nanmin([pv[v][k] for v in pv]):.2f}..{np.nanmax([pv[v][k] for v in pv]):.2f}" for k in K2]
    print(d.ljust(13) + "".join(c.rjust(12) for c in cells))
print()
K3 = ["run_z", "run_xy", "r_face_z", "r_face_xy", "ratio_z", "ratio_xy", "grad_z", "grad_xy", "iou50", "lvl_vol"]
for d in sys.argv[2:]:
    pv = S[d]["per_volume"]
    for v in pv:
        print(d, v[:40].ljust(42), " ".join(f"{k}={pv[v][k]:.2f}" for k in K3))
