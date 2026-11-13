#!/usr/bin/env python
"""Reference medians/ranges of additional metrics for native vs. downsampled controls, next to mesospim."""
import json

import numpy as np

R = {n: json.load(open(f"{d}/summary.json")) for n, d in
     [("native", "results_main"), ("majority", "results_ctrl_majority"), ("any", "results_ctrl_any")]}
refs = sorted(d for d in R["native"] if not d.startswith("_") and d != "mesospim")
m = R["native"]["mesospim"]["metrics"]
for k in ["r_inner", "r_outer", "r_face", "edge_xy", "edge_z", "snr", "run_z", "lvl_xy", "lvl_z", "lvl_vol", "best_lvl", "best_iou"]:
    line = k.ljust(10)
    for n in R:
        v = np.array([R[n][d]["metrics"][k]["med"] for d in refs])
        line += f"  {n}: {np.median(v):.2f} [{v.min():.2f},{v.max():.2f}]"
    print(line + f"  | mesospim {m[k]['med']:.2f}")
