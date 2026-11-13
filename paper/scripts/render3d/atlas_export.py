"""F4 atlas export (server, CPU): GT of each dataset's median-rule volume (ortho/selection.json), laterally subsampled x2
(x3 if > 300 px), as uint16, plus voxel ratio and species/organ label from the manifest. Writes export/atlas.npz."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import numpy as np
SEL = json.load(open(f"{_NSP_SCRATCH}/nc/figs/ortho/selection.json"))["volumes"]
out, meta = {}, {}
for ds, v in SEL.items():
    z = np.load(v["npz"]["S1"])
    g = z["gt"]
    f = 1  # full resolution (F4 final)
    out[ds] = g[:, ::f, ::f].astype(np.uint16)
    meta[ds] = dict(id=v["id"], ratio=v["ratio"], f=f, shape=list(g.shape), n_gt=int(len(np.unique(g)) - 1))
np.savez_compressed(f"{_NSP_SCRATCH}/nc/figs/render3d/export/atlas_full.npz", meta=json.dumps(meta), **out)
print(json.dumps(meta))
