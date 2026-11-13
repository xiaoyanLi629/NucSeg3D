"""List non-test 3D volumes per dataset (read-only helper for choosing volumes)."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import collections
import csv
import json

STD = f"{_NSP_ROOT}/data/std"
rows = list(csv.DictReader(open(f"{STD}/manifest.csv")))
val = json.load(open(f"{STD}/val_split.json"))
print("val keys:", {k: len(v) for k, v in val.items()})
by = collections.defaultdict(list)
for r in rows:
    if r["dim"] != "3d" or r["split"] == "test":
        continue
    by[r["ds"]].append(r)
for ds, L in sorted(by.items()):
    vs = set(val.get(ds, []))
    shapes = collections.Counter((r["z"], r["y"], r["x"]) for r in L)
    vox = collections.Counter((r["voxel_z"], r["voxel_y"], r["voxel_x"]) for r in L)
    nn = sorted(int(r["n_nuclei"]) for r in L)
    nval = sum(r["id"] in vs for r in L)
    print(f"== {ds}: n={len(L)} nval={nval} nuclei min/med/max={nn[0]}/{nn[len(nn)//2]}/{nn[-1]}  dtypes={set(r['img_dtype'] for r in L)}")
    print("   shapes:", shapes.most_common(5))
    print("   vox:", vox.most_common(5))
    print("   ids:", [r["id"] for r in L][:8], "...")
    for n in sorted(set(r["note"] for r in L))[:4]:
        print("   note:", n[:300])
