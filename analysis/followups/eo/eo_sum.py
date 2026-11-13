import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, os
os.chdir(f"{_NSP_SCRATCH}/nc/gpu/eo")
for w in ("A", "B"):
    for m in ("fp", "gt"):
        for c in ("", "_C1"):
            f = f"eo_{w}_{m}{c}.json"
            if not os.path.exists(f):
                print(f, "missing"); continue
            d = json.load(open(f))
            for lab in ("F1@0.5", "F1@0.7"):
                mm = d["macro"][lab]
                print(f, lab, " | ".join(f"{t}: {v['diff_mean']:+.4f}+-{v['diff_sd']:.4f}{' OK' if v['confirmed'] else ''} (S1col {v['S1']:.4f} 2Dcol {v['two_d']:.4f})" for t, v in mm.items()))
