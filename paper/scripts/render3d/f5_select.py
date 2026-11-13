import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, numpy as np
R = f"{_NSP_ROOT}/results"
S = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
def per_vol(key):
    d = glob.glob(f"{R}/orx_{S[key]['run']}*")[0] + "/noise_0"
    rt = {x["ds"] + "__" + x["id"]: x["t"] for x in json.load(open(f"{d}/ruleT.json"))}
    out = {}
    for f in glob.glob(f"{d}/vols/*.json"):
        rows = json.load(open(f)); k = rows[0]["ds"] + "__" + rows[0]["id"]
        t = rt[k]
        r = [x for x in rows if x["thr"] == float(t) and x["frac"] == 0.15][0]
        out[k] = (r["f1"], r["n_gt"], float(t))
    return out, list(rt.items())[:1]
s1, ex = per_vol("G4_S1_s0_z8"); d2, _ = per_vol("G4_2D_s0_z8")
print("ruleT example", ex)
keys = sorted(set(s1) & set(d2)); diff = {k: s1[k][0] - d2[k][0] for k in keys}
med = float(np.median(list(diff.values())))
pick = min(keys, key=lambda k: (abs(diff[k] - med), -s1[k][1], k))
print(json.dumps({"n": len(keys), "median": med, "pick": pick, "diff": diff[pick], "thr_S1": s1[pick][2], "thr_2D": d2[pick][2]}))
for k in keys: print(k, round(diff[k], 4), s1[k][1])
