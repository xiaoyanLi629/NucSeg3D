"""R6 Fig. 6 export (server, CPU, read-only). Rule: paper/scripts/render3d/R6_FIG6_RULE.md (md5 d0ca6e98...).
For each dataset: the median-rule volume (figs/ortho/selection.json); final masks gt / base (fixed -2) / nnunet (pred_v4, 0.25) /
2D / S1 (rule T, seed 0; errors cache); per-volume F1@0.5/F1@0.7 from stored counts. Writes r6f6_<ds>.npz and r6f6_log.json."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json, os, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import common as C
O = f"{_NSP_SCRATCH}/nc/figs/r6fig6"
SEL = json.load(open(f"{_NSP_SCRATCH}/nc/figs/ortho/selection.json"))["volumes"]
RUNS = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))
NN = f"{_NSP_SHARED}/baselines/results/eval/nnunet_test_gt"
log = json.load(open(f"{O}/r6f6_log.json")) if os.path.exists(f"{O}/r6f6_log.json") else {}
for ds in sys.argv[1:]:
    vid = SEL[ds]["id"]; key = f"{ds}__{vid}"
    z = np.load(RUNS["gt_S1_s0"] + "/masks/" + key + ".npz")
    gt, vox, d = z["gt"], z["vox"], float(z["d"])
    out = {"gt": gt}; info = {}
    for nm in ("base", "2D", "S1"):
        tg = f"gt_{nm}_s0"
        out[nm] = np.load(f"{C.OUT}/cache/{tg}/{key}.npz")["mask"]
        c = json.load(open(f"{C.OUT}/per/{tg}/{key}.json"))
        n = c["counts"]["n_gt"] + c["counts"]["n_pred"]
        info[nm] = dict(f1_05=2 * c["counts"]["tp5"] / n, f1_07=2 * c["counts"]["tp7"] / n, tp5=c["counts"]["tp5"], tp7=c["counts"]["tp7"],
                        n_gt=c["counts"]["n_gt"], n_pred=c["counts"]["n_pred"], t=c["t"], source=f"errors/per/{tg}/{key}.json")
        assert int((np.unique(out[nm]) > 0).sum()) == c["counts"]["n_pred"], (ds, nm)
        if nm == "S1":
            assert abs(info[nm]["f1_05"] - SEL[ds]["f1"]["S1"]) < 1e-12, "selection mismatch"
    b = np.load(f"{NN}/masks/{key}.npz"); assert np.array_equal(b["gt"], gt)
    out["nnunet"] = C.drop_small(b["pred_v4"].astype(np.int32), float(b["d"]), b["vox"], frac=0.25)
    r = [x for x in json.load(open(f"{NN}/vols/{key}.json")) if x["thr"] == 4.0 and x["frac"] == 0.25][0]
    assert int((np.unique(out["nnunet"]) > 0).sum()) == r["n_pred"]
    info["nnunet"] = dict(f1_05=r["f1"], f1_07=r["f1_7"], tp5=r["tp5"], tp7=r["tp7"], n_gt=r["n_gt"], n_pred=r["n_pred"], source=f"{NN}/vols/{key}.json (thr 4, frac 0.25)")
    np.savez_compressed(f"{O}/r6f6_{ds}.npz", vox=vox, d=d, ratio=float(vox[0] / vox[1]), **{k: v.astype(np.int32) for k, v in out.items()})
    log[ds] = dict(id=vid, shape=list(gt.shape), d=d, vox=list(map(float, vox)), models=info)
    print(ds, vid, gt.shape, {k: (round(v["f1_05"], 3), round(v["f1_07"], 3)) for k, v in info.items()}, flush=True)
json.dump(log, open(f"{O}/r6f6_log.json", "w"), indent=1)
