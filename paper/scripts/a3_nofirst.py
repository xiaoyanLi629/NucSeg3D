"""ECE without the first bin (p < 1/15), recomputed from per-bin counts stored in the A3 outputs.
Pre-registered as a supplementary readout in $NUCSEG3D_SCRATCH/nc/gpu/DESIGN.md section 5.
Usage: python3 -I a3_nofirst.py a3_per_ds2.json a3_all.json"""
import json, sys
import numpy as np
for f in sys.argv[1:]:
    d = json.load(open(f))
    for m, v in d["results"].items():
        eces = []
        for ds, x in v["per_ds"].items():
            n = np.asarray(x["n"], float)[1:]; c = np.asarray(x["conf"], float)[1:]; fr = np.asarray(x["frac"], float)[1:]
            ok = n > 0
            eces.append(float((n[ok] * np.abs(fr[ok] - c[ok])).sum() / n[ok].sum()))
        print(f"{f} {m}: macro ECE {v['macro_ece']:.4f}  macro ECE without first bin {np.mean(eces):.4f}  Brier {v['macro_brier']:.4f}  ({len(eces)} datasets)")
