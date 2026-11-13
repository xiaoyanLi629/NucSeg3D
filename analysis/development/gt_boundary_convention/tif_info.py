"""Print TIFF shape/dtype/compression for the largest non-test 3D volume of each dataset (no pixel data read)."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import collections
import csv
import os

import tifffile

STD = f"{_NSP_ROOT}/data/std"
rows = [r for r in csv.DictReader(open(f"{STD}/manifest.csv")) if r["dim"] == "3d" and r["split"] != "test"]
by = collections.defaultdict(list)
for r in rows:
    by[r["ds"]].append(r)
for ds, L in sorted(by.items()):
    r = max(L, key=lambda r: int(r["z"]) * int(r["y"]) * int(r["x"]))
    out = [ds, r["id"]]
    for k in ("image", "mask"):
        p = os.path.join(STD, r[k])
        with tifffile.TiffFile(p) as t:
            s = t.series[0]
            pg = t.pages[0]
            out.append(f"{k}:{s.shape} {s.dtype} comp={pg.compression.name} {os.path.getsize(p)/1e6:.0f}MB")
    ign = os.path.join(STD, ds, "ignore")
    out.append(f"ignore_dir={os.path.isdir(ign)}")
    print(" | ".join(out))
