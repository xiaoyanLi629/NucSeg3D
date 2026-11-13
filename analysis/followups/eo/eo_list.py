import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import sys, glob, json, os
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import s5_stats as S5
RES = f"{_NSP_ROOT}/results"; ST = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]
KEY = {("gt", s): ("EO_tGT_s1000_r2" if s == "s1000" else "EO_tGT_" + s) for s in SEEDS}
KEY.update({("fp", s): "EO_tFP_" + s for s in SEEDS}); KEY.update({("val", s): "EO_v_" + s for s in SEEDS})
out = []
for split, mode in (("test", "gt"), ("val", "gt"), ("test", "fp")):
    D = S5.dirs(split, mode, SEEDS); m = "val" if split == "val" else mode
    out += [(m, "S1", d) for d in D["S1"]] + [(m, "2D", d) for d in D["2D"]]
    out += [(m, "EO", glob.glob(RES + "/orx_" + ST[KEY[(m, s)]]["run"] + "_thr_eval")[0]) for s in SEEDS]
open(f"{_NSP_SCRATCH}/nc/gpu/eo/eo_prefetch_list.txt", "w").write("\n".join(f"{m}\t{n}\t{d}" for m, n, d in out) + "\n")
print(len(out), len(set(d for _, _, d in out)))
