"""Tight crops of existing Fig. 1 assets for A7 (no pixel values changed; only empty transparent margins removed)."""
import json, os
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets7")
crops = {
    "cmp_2d_xyonly": ("assets3/cmp_2d_xyonly.png", (89, 67, 619, 457)),
    "cmp_s1_view1": ("assets3/cmp_s1_view1.png", (89, 67, 619, 457)),
    "ang_2d": ("assets3/ang_2d.png", (89, 67, 619, 457)),
    "ang_s1": ("assets3/ang_s1.png", (89, 67, 619, 457)),
    "diff_source": ("assets3/diff_source.png", (167, 62, 726, 745)),
    "diff_heat": ("assets3/diff_heat.png", (167, 62, 726, 745)),
}
for v in ("z", "y", "x", "mean"):
    crops[f"view_{v}"] = (f"assets3/view_{v}.png", (115, 65, 384, 264))
log = {}
for k, (src, box) in crops.items():
    im = Image.open(os.path.join(HERE, src))
    im.crop(box).save(os.path.join(OUT, k + ".png"))
    log[k] = dict(source=src, crop_box_px=box, size=im.crop(box).size)
json.dump(log, open(os.path.join(OUT, "assets7_log.json"), "w"), indent=1)
print(log)
