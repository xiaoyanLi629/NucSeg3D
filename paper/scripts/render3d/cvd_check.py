"""Colour-vision-deficiency check of palette.json: minimum pairwise CAM02-UCS distance within each colour group under normal,
deuteranopia, protanopia and tritanopia (colorspacious, severity 100). Writes figures/palette_cvd.json."""
import itertools, json, os
import numpy as np
from colorspacious import cspace_convert
P = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "figures", "palette.json")))
def rgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
groups = {"models": {k: v for k, v in P["models"].items() if k in ("S1", "2D", "base", "nnunet", "stardist3d", "useg", "tools")},
          "errors": P["errors"], "T_panel": {"S1": P["models"]["S1"], "2D": P["models"]["2D"], "base": P["models"]["base"], "GT": P["gt_outline"]}}
out = {}
for g, cols in groups.items():
    out[g] = {}
    for cvd in ("normal", "deuteranomaly", "protanomaly", "tritanomaly"):
        lab = {}
        for k, h in cols.items():
            c = rgb(h)
            if cvd != "normal":
                c = np.clip(cspace_convert(c, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"), 0, 1)
            lab[k] = cspace_convert(c, "sRGB1", "CAM02-UCS")
        d = sorted((float(np.linalg.norm(lab[a] - lab[b])), a, b) for a, b in itertools.combinations(lab, 2))
        out[g][cvd] = {"min": round(d[0][0], 1), "closest": d[0][1:], "pairs_below_15": [(round(x, 1), a, b) for x, a, b in d if x < 15]}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "figures", "palette_cvd.json"), "w"), indent=1)
for g in out:
    print(g, {c: (v["min"], v["closest"]) for c, v in out[g].items()})
