"""nc-arch Fig. 1 v4 data (server, CPU only, read-only on results; writes /tmp/nc_arch/).
(1) S1 seed 0 on mouse_embryo E15_Ab2, the three views of the pipeline (infer3d.VIEW_PERMS), each on ONE 32-plane window
    (256 x 256 laterally) centred on GT nucleus 62, same preprocessing as viz_run.py / eval_gt (1/99.8 % scaling, linear z
    interpolation, normalize_img norm3D, resize 30/d). Output: per-view raw outputs on the xz section at resized y = 103, mapped back to
    volume layout with infer3d.from_view, over the region the three windows share.
(2) rule T m(t) for E15_Ab2 (S1 seed 0, run 07992b55 thr_eval) exactly as sizeThr_dir.py.
(3) diameter scan rows (eval_val joint/r2a2) for mouse_embryo volumes in run 4a362b0f (2D control seed 0, test split).
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 python arch_views.py"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
import os, sys, json, time, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
torch.set_num_threads(16)
SRC = _nsp_os.path.join(_NSP_REPO, 'src')
sys.path.insert(0, SRC)
from eval_cpsam import load_case, eq_diam_xy_px
from eval_val import zinterp
from stddata import load_manifest
from infer3d import VIEW_PERMS, from_view
from net3d import build_from_dir
from cellpose import models, transforms
OUT = "/tmp/nc_arch"; os.makedirs(OUT, exist_ok=True)
res = {}
# ---------------- (2) rule T
R = f"{_NSP_ROOT}/results/orx_07992b55-0982-4116-b065-b34859c9a80b_thr_eval"
rows = json.load(open(R + "/rows.json"))
by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
thrs = sorted({x["thr"] for x in rows}, reverse=True)
cp = np.load(R + "/masks/mouse_embryo__E15_Ab2.npz")["cp"].astype(np.float32)
r0 = by[("mouse_embryo", "E15_Ab2", -2.0, 0.0)]
dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
m = {t: float((cp > t).sum()) / max(by[("mouse_embryo", "E15_Ab2", t, 0.0)]["n_pred"], 1) / V for t in thrs}
t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
res["ruleT"] = dict(m={str(k): v for k, v in m.items()}, n_pred={str(t): by[("mouse_embryo", "E15_Ab2", t, 0.0)]["n_pred"] for t in thrs},
                    n_fg={str(t): float((cp > t).sum()) for t in thrs}, V=V, d=dd, ratio=r0["ratio"], chosen=t2, n_gt=r0["n_gt"],
                    f1_by_thr={str(t): by[("mouse_embryo", "E15_Ab2", t, 0.15)]["f1"] for t in thrs})
# ---------------- (3) diameter scan
RA = json.load(open(f"{_NSP_ROOT}/results/orx_4a362b0f-6b0e-476e-879d-a5080632bbdd_test/rows_all.json"))
RR = json.load(open(f"{_NSP_ROOT}/results/orx_4a362b0f-6b0e-476e-879d-a5080632bbdd_test/rows.json"))
res["diam"] = {k: [x for x in RA if x["ds"] == "mouse_embryo" and x["id"] == k] for k in ("E15_Ab1", "E18_Ab1", "E20_Ab2", "E9_Ab3")}
res["diam_sel"] = {k: [dict(scale=x.get("scale"), sel_thr=x.get("sel_thr"), d_used=x["d_used"], d_gt=x["d_gt"]) for x in RR
                       if x["ds"] == "mouse_embryo" and x["id"] == k][:1] for k in res["diam"]}
json.dump(res, open(OUT + "/ruleT_diam.json", "w"), indent=1, default=float)
print("ruleT", res["ruleT"]["m"], "chosen", t2, flush=True)
# ---------------- (1) three views
r = [x for x in load_manifest(split="test", dim="3d") if x["ds"] == "mouse_embryo" and x["id"] == "E15_Ab2"][0]
img, gt, box, ign = load_case(r, 384)
vox = r["voxel"]; ratio = vox[0] / vox[1]; d = eq_diam_xy_px(gt, vox)
Zu = int(round(img.shape[0] * ratio))
im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
img_in = zinterp(im, Zu)
x4 = transforms.normalize_img(img_in[..., None], **{**models.normalize_default, "normalize": True, "norm3D": True})
rs = 30.0 / d
new_shape = (int(Zu * rs), int(384 * rs), int(384 * rs))
xi = np.ascontiguousarray(transforms.resize_image_3d(x4, new_shape, no_channels=False)[..., 0], dtype=np.float32)
g = (gt[:, 126, :] == 62)
zz, xx = np.nonzero(g)
c = np.array([zz.mean() * ratio * rs, 126 * new_shape[1] / 384.0, xx.mean() * new_shape[2] / 384.0])
print("resized", xi.shape, "nucleus 62 centre (resized)", c, flush=True)
net, meta = build_from_dir(f"{_NSP_ROOT}/models/models/orx_t3d_r5s1_s0", torch.device("cpu"))
YR = int(round(c[1]))
# section box (resized coords) around the nucleus
zb = (int(c[0]) - 14, int(c[0]) + 14); xb = (int(c[2]) - 24, int(c[2]) + 24)
outs = {}
t0 = time.time()
for p in VIEW_PERMS:
    shp = [xi.shape[k] for k in p]
    cv = [c[k] for k in p]
    st = []
    for k, (L, w) in enumerate(zip(shp, (32, 256, 256))):
        s0 = int(round(cv[k] - w / 2)); s0 = min(max(s0, 0), L - w); st.append(s0)
    v = np.transpose(xi, p)[st[0]:st[0] + 32, st[1]:st[1] + 256, st[2]:st[2] + 256]
    with torch.no_grad():
        y = net(torch.from_numpy(np.ascontiguousarray(v))[None, None])[0].numpy()     # (4, 32, 256, 256) view layout
    full = np.full((4,) + tuple(xi.shape), np.nan, np.float32)
    yv = from_view(y, p)                                                           # volume layout of the slab
    sl = [None] * 3
    for k in range(3):
        sl[p[k]] = slice(st[k], st[k] + (32, 256, 256)[k])
    full[(slice(None),) + tuple(sl)] = yv
    outs[str(p)] = full[:, zb[0]:zb[1], YR, xb[0]:xb[1]].copy()
    print("view", p, "start", st, f"{time.time() - t0:.0f}s", "covered", float(np.isfinite(outs[str(p)][0]).mean()), flush=True)
np.savez_compressed(OUT + "/views_E15_Ab2_n62.npz", **{k.replace(" ", ""): v for k, v in outs.items()}, zb=zb, xb=xb, YR=YR,
                    centre=c, new_shape=np.array(new_shape), d=d, ratio=ratio, rs=rs)
json.dump(dict(views=list(map(str, VIEW_PERMS)), zb=zb, xb=xb, YR=YR, centre=c.tolist(), new_shape=new_shape, d=d,
               note="one 32-plane window per view centred on GT nucleus 62; raw outputs (5x unit flow, logit), not the full sin^2-blended pipeline"),
          open(OUT + "/views_E15_Ab2_n62.json", "w"), indent=1)
print("done", flush=True)
