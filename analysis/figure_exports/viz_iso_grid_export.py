"""nc-viz ISO_grid (selection_rules.json ISO_grid): draw one pair per dataset (datasets with >= 3 candidates) from the 72 'merged off,
separated on' pairs with numpy default_rng(20261110), and export small crops (GT, off/on dP and cp, final masks) for rendering.
Read-only on results; CPU; writes $NUCSEG3D_SCRATCH/nc/figs/viz/iso_off/grid/."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/figure_exports'))
import viz_iso_ext as X  # noqa: E402

ext = json.load(open(f"{_NSP_SCRATCH}/nc/figs/viz/iso_off/iso_ext.json"))
cand = ext["candidates"]
by = {}
for c in cand:
    by.setdefault(c["key"].split("__")[0], []).append(c)
rng = np.random.default_rng(20261110)
picks = []
for ds in sorted(by):
    L = sorted(by[ds], key=lambda c: (c["key"], c["a"], c["b"]))
    if len(L) < 3:
        continue
    picks.append(L[int(rng.integers(0, len(L)))])
print(json.dumps([(p["key"], p["a"], p["b"]) for p in picks]))
O = f"{_NSP_SCRATCH}/nc/figs/viz/iso_off/grid"; os.makedirs(O, exist_ok=True)
for p in picks:
    key, A, B = p["key"], p["a"], p["b"]
    zo = np.load(f"{X.OFF}/masks/{key}.npz"); zn = np.load(f"{X.ON}/masks/{key}.npz")
    gt = zo["gt"].astype(np.int32); vox = np.asarray(zo["vox"], float); d = float(zo["d"])
    off_final, t = X.off_mask(key, zo)
    on_final = np.load(f"{_NSP_SCRATCH}/nc/errors/cache/gt_S1_s0/{key}.npz")["mask"]
    pair = (gt == A) | (gt == B); zz, yy, xx = np.nonzero(pair)
    pz, pxy = 2, int(np.ceil(d / 2))
    b0 = [max(zz.min() - pz, 0), max(yy.min() - pxy, 0), max(xx.min() - pxy, 0)]
    b1 = [min(zz.max() + pz + 1, gt.shape[0]), min(yy.max() + pxy + 1, gt.shape[1]), min(xx.max() + pxy + 1, gt.shape[2])]
    sl = tuple(slice(a, b) for a, b in zip(b0, b1))
    # on: keep native planes over the whole z range (needed for z re-interpolation inside the crop) -> crop only y, x
    np.savez_compressed(f"{O}/{key}__{A}_{B}.npz", gt=gt[sl], vox=vox, d=d, b0=np.array(b0), b1=np.array(b1), Zn=gt.shape[0], A=A, B=B,
                        dP_off=zo["dP"][(slice(None),) + sl].astype(np.float16), cp_off=zo["cp"][sl].astype(np.float16),
                        dP_on=zn["dP"][(slice(None),) + sl].astype(np.float16), cp_on=zn["cp"][sl].astype(np.float16),
                        final_off=off_final[sl].astype(np.int32), final_on=on_final[sl].astype(np.int32), t_off=t)
json.dump(dict(rule="selection_rules.json ISO_grid", seed=20261110, n_by_dataset={k: len(v) for k, v in sorted(by.items())},
               picks=picks), open(f"{O}/picks.json", "w"), indent=1)
print("done")
