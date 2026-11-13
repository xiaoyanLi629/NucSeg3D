#!/usr/bin/env python
"""Markdown tables for the report (median (q25-q75) per dataset), the control comparison and the 4- vs 12-volume
check. Writes report_tables.txt next to the results."""
import json
import os

import numpy as np

S = json.load(open("results_main/summary.json"))
dss = sorted([d for d in S if not d.startswith("_") and d != "mesospim"]) + ["mesospim"]


def cell(d, k, nd=2, iqr=True, src=S):
    m = src[d]["metrics"][k]
    if not np.isfinite(m["med"]):
        return "-"
    f = f"{{:.{nd}f}}"
    return (f.format(m["med"]) + " (" + f.format(m["q25"]) + "–" + f.format(m["q75"]) + ")") if iqr else f.format(m["med"])


lines = []
T1 = [("run_z", 0), ("run_xy", 1), ("snr", 0), ("r_inner", 2), ("r_face_z", 2), ("r_face_xy", 2), ("edge_z", 2), ("edge_xy", 2)]
T2 = [("ratio_z", 2), ("ratio_xy", 2), ("grad_z", 2), ("grad_xy", 2), ("zr50", 2), ("xyr50", 2), ("iou50", 2), ("best_lvl", 2)]
for title, T in (("Table 1", T1), ("Table 2", T2)):
    lines.append(f"\n### {title}\n")
    lines.append("| ds | n (vol) | " + " | ".join(k for k, _ in T) + " |")
    lines.append("|" + "---|" * (len(T) + 2))
    for d in dss:
        lines.append(f"| {d} | {S[d]['n_nuclei']} ({S[d]['n_vol']}) | " + " | ".join(cell(d, k, nd) for k, nd in T) + " |")
    ref = [d for d in dss if d != "mesospim"]
    row = []
    for k, nd in T:
        v = np.array([S[d]["metrics"][k]["med"] for d in ref])
        row.append(f"{np.median(v):.{nd}f} [{v.min():.{nd}f}, {v.max():.{nd}f}]")
    lines.append("| 13 ref: median [min, max] of dataset medians | | " + " | ".join(row) + " |")

# control
C = json.load(open("compare_control.json"))
K = ["r_inner", "r_face_xy", "edge_xy", "ratio_xy", "grad_xy", "xyr50", "r_face_z", "ratio_z", "grad_z", "zr50", "iou50", "best_lvl"]
R = {n: json.load(open(f"{d}/summary.json")) for n, d in
     [("native", "results_main"), ("majority", "results_ctrl_majority"), ("any", "results_ctrl_any")]}
refs = sorted(d for d in R["native"] if not d.startswith("_") and d != "mesospim")
lines.append("\n### Control (13 reference datasets: median [min, max] of dataset medians)\n")
lines.append("| set | run_xy | " + " | ".join(K) + " |")
lines.append("|" + "---|" * (len(K) + 2))
for n in R:
    cells = []
    for k in ["run_xy"] + K:
        v = np.array([R[n][d]["metrics"][k]["med"] for d in refs])
        cells.append(f"{np.median(v):.2f} [{v.min():.2f}, {v.max():.2f}]")
    lines.append(f"| {n} | " + " | ".join(cells) + " |")
lines.append("| mesospim | " + " | ".join(f"{S['mesospim']['metrics'][k]['med']:.2f}" for k in ["run_xy"] + K) + " |")
lines.append("\nfactors: " + json.dumps(C["factors"]))

# 4 vs 12 volumes
if os.path.exists("results_nvol12/summary.json"):
    N = json.load(open("results_nvol12/summary.json"))
    K4 = ["r_face_z", "r_face_xy", "ratio_z", "ratio_xy", "grad_z", "grad_xy", "zr50", "xyr50", "iou50", "best_lvl"]
    lines.append("\n### 4 volumes (main) vs 12 volumes: dataset medians\n")
    lines.append("| ds | n main / n12 | " + " | ".join(K4) + " |")
    lines.append("|" + "---|" * (len(K4) + 2))
    for d in sorted(k for k in N if not k.startswith("_")):
        lines.append(f"| {d} | {S[d]['n_nuclei']} / {N[d]['n_nuclei']} | "
                     + " | ".join(f"{S[d]['metrics'][k]['med']:.2f} / {N[d]['metrics'][k]['med']:.2f}" for k in K4) + " |")
    # reference range when the 12-volume medians replace the 4-volume ones where available
    lines.append("\nreference [min, max] using 12-volume medians where available:")
    for k in K4:
        v = np.array([(N[d] if d in N else S[d])["metrics"][k]["med"] for d in dss if d != "mesospim"])
        lines.append(f"- {k}: [{v.min():.2f}, {v.max():.2f}], median {np.median(v):.2f}; mesospim {S['mesospim']['metrics'][k]['med']:.2f}")
open("report_tables.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
