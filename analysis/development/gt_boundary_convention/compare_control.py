#!/usr/bin/env python
"""Compare native reference datasets, their xy-downsampled versions (majority / any GT rule) and mesospim.
Reads results_main, results_ctrl_majority, results_ctrl_any (summary.json); writes compare_control.json / .csv."""
import csv
import json

import numpy as np

KEYS = ["run_xy", "run_z", "r_face_xy", "r_face_z", "edge_xy", "edge_z", "ratio_xy", "ratio_z", "grad_xy", "grad_z",
        "xyr50", "zr50", "vr50", "iou50", "lvl_xy", "lvl_z", "lvl_vol", "best_iou", "best_lvl"]
RUNS = [("native", "results_main"), ("majority", "results_ctrl_majority"), ("any", "results_ctrl_any")]
S = {name: json.load(open(f"{d}/summary.json")) for name, d in RUNS}
fac = json.load(open("control_factors.json"))["factors"]
refs = sorted(d for d in S["native"] if not d.startswith("_") and d != "mesospim")
meso = S["native"]["mesospim"]["metrics"]
out = dict(factors=fac, per_dataset={}, ref_median={}, ref_range={}, mesospim={k: meso[k] for k in KEYS})
for d in refs:
    out["per_dataset"][d] = {name: {k: S[name][d]["metrics"][k]["med"] for k in KEYS} for name, _ in RUNS if d in S[name]}
    out["per_dataset"][d]["n"] = {name: S[name][d]["n_nuclei"] for name, _ in RUNS if d in S[name]}
for name, _ in RUNS:
    out["ref_median"][name], out["ref_range"][name] = {}, {}
    for k in KEYS:
        v = np.array([S[name][d]["metrics"][k]["med"] for d in refs if d in S[name]])
        out["ref_median"][name][k] = float(np.median(v))
        out["ref_range"][name][k] = [float(v.min()), float(v.max())]
json.dump(out, open("compare_control.json", "w"), indent=1)
with open("compare_control.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ds", "factor", "run"] + KEYS)
    for d in refs:
        for name, _ in RUNS:
            if name in out["per_dataset"][d]:
                w.writerow([d, fac[d] if name != "native" else 1, name] + [f"{out['per_dataset'][d][name][k]:.4f}" for k in KEYS])
    for name, _ in RUNS:
        w.writerow(["REF_MEDIAN", "", name] + [f"{out['ref_median'][name][k]:.4f}" for k in KEYS])
    w.writerow(["mesospim", 1, "native"] + [f"{meso[k]['med']:.4f}" for k in KEYS])
show = ["run_xy", "r_face_xy", "edge_xy", "ratio_xy", "grad_xy", "xyr50", "r_face_z", "ratio_z", "grad_z", "zr50", "iou50", "best_lvl"]
print("ds".ljust(13) + "f".rjust(3) + "".join(k.rjust(11) for k in show))
for d in refs:
    for name, _ in RUNS:
        if name in out["per_dataset"][d]:
            r = out["per_dataset"][d][name]
            print((d if name == "native" else "  " + name).ljust(13) + str(fac[d] if name != "native" else 1).rjust(3)
                  + "".join(f"{r[k]:.2f}".rjust(11) for k in show))
for name, _ in RUNS:
    print(("REF med " + name).ljust(16) + "".join(f"{out['ref_median'][name][k]:.2f}".rjust(11) for k in show))
    print(("REF min " + name).ljust(16) + "".join(f"{out['ref_range'][name][k][0]:.2f}".rjust(11) for k in show))
    print(("REF max " + name).ljust(16) + "".join(f"{out['ref_range'][name][k][1]:.2f}".rjust(11) for k in show))
print("mesospim".ljust(16) + "".join(f"{meso[k]['med']:.2f}".rjust(11) for k in show))
