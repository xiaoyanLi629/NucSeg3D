"""Multi-model export for the 3D strip (server, CPU, read-only). Rule: render3d/MM_RULE.md.
For each dataset's median-rule volume writes export/mm_<ds>.npz with the evaluated final masks of nnU-Net, StarDist-3D,
u-Segment3D (2D control) and u-Segment3D (original), plus their per-volume F1@0.5 / F1@0.7 read from the runs' rows.
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob
import json
import os
import sys

import numpy as np

SEL = json.load(open(f"{_NSP_SCRATCH}/nc/figs/ortho/selection.json"))["volumes"]
BL = f"{_NSP_SHARED}/baselines/results/eval"
RES = f"{_NSP_ROOT}/results"
O = f"{_NSP_SCRATCH}/nc/figs/render3d/export"
MODELS = {
    "nnunet": dict(dir=f"{BL}/nnunet_test_gt", key="pred_v4", thr=4.0, frac=0.25),
    "stardist3d": dict(dir=f"{BL}/stardistB_test_gt", key="pred_v2", thr=2.0, frac=0.15),
    "useg_2d": dict(dir=glob.glob(f"{RES}/orx_99e7b2ae*_useg_eval")[0], key="pred", variant=10, thr=-2.0, frac=0.15),
    "useg_base": dict(dir=glob.glob(f"{RES}/orx_4f914272*_useg_eval")[0], key="pred", variant=8, thr=-2.0, frac=0.15),
}


def drop_small(m, d, vox, frac):
    ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(m.ravel())
    small = np.where(cnt < frac * ev)[0]; small = small[small > 0]
    m = m.copy(); m[np.isin(m, small)] = 0
    return m


log = {}
for ds in sys.argv[1:]:
    vid = SEL[ds]["id"]
    out, info = {}, {}
    for nm, c in MODELS.items():
        fn = f"{c['dir']}/masks/{ds}__{vid}" + (f"__v{c['variant']}" if "variant" in c else "") + ".npz"
        if not os.path.exists(fn):
            info[nm] = "n/a (no saved mask)"
            continue
        z = np.load(fn)
        m = drop_small(z[c["key"]].astype(np.int32), float(z["d"]), z["vox"], c["frac"])
        out[nm] = m
        rows = json.load(open(f"{c['dir']}/vols/{ds}__{vid}.json"))
        r = [x for x in rows if x["thr"] == c["thr"] and x["frac"] == c["frac"] and x.get("variant", c.get("variant")) == c.get("variant")]
        info[nm] = dict(f1_05=r[0]["f1"], f1_07=r[0]["f1_7"], n_pred_row=r[0]["n_pred"], n_pred_mask=int((np.unique(m) > 0).sum()), file=fn) if r else "row missing"
    np.savez_compressed(f"{O}/mm_{ds}.npz", **out)
    log[ds] = dict(id=vid, models=info)
    print(ds, vid, json.dumps({k: (v if isinstance(v, str) else {kk: v[kk] for kk in ('f1_05', 'n_pred_row', 'n_pred_mask')}) for k, v in info.items()}), flush=True)
old = json.load(open(f"{O}/mm_log.json")) if os.path.exists(f"{O}/mm_log.json") else {}
old.update(log)
json.dump(old, open(f"{O}/mm_log.json", "w"), indent=1)
