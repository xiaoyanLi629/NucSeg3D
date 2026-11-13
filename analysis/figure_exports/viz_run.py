"""nc-viz V2/V4: run S1 (identity view only) and the 2D control's network (xy planes only) on a crop of mouse_embryo E15_Ab2,
on CPU, with the pipeline's exact preprocessing (eval_gt.run_volume -> Model3D.eval: 1/99.8 % scaling, z-interpolation to isotropic,
normalize_img norm3D on the whole volume, resize by 30/d). Hooks: S1 encoder neck output (per plane) and the input of the 3D decoder's
last 1x1 conv (= gelu(GroupNorm(x)), the decoder's last hidden features). Read-only on results; writes $NUCSEG3D_SCRATCH/nc/figs/viz/.
Usage: CUDA_VISIBLE_DEVICES="" python viz_run.py [--bench]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import os, sys, json, time, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(8)
SRC = _nsp_os.path.join(_NSP_REPO, 'src')
sys.path.insert(0, SRC)
from eval_cpsam import load_case, eq_diam_xy_px
from eval_val import zinterp
from stddata import load_manifest
from infer3d import _starts, _taper
from net3d import build_from_dir, load_cpsam2d
from cellpose import models, transforms

OUT = f"{_NSP_SCRATCH}/nc/figs/viz"; os.makedirs(OUT, exist_ok=True)
BENCH = "--bench" in sys.argv
r = [x for x in load_manifest(split="test", dim="3d") if x["ds"] == "mouse_embryo" and x["id"] == "E15_Ab2"][0]
img, gt, box, ign = load_case(r, 384)
vox = r["voxel"]; ratio = vox[0] / vox[1]; d = eq_diam_xy_px(gt, vox)
Zu = int(round(img.shape[0] * ratio))
im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
img_in = zinterp(im, Zu)
x4 = transforms.normalize_img(img_in[..., None], **{**models.normalize_default, "normalize": True, "norm3D": True})
rs = 30.0 / d
new_shape = (int(Zu * rs), int(384 * rs), int(384 * rs))
xi = transforms.resize_image_3d(x4, new_shape, no_channels=False)[..., 0]
print("d", d, "Zu", Zu, "resized", xi.shape, flush=True)
# crop: xz slice at native y=126 (rule in selection_rules.json), native x 119:375
Y0N, X0N = 126, 119
yr = Y0N * new_shape[1] / 384.0
cy0 = int(min(max(round(yr) - 128, 0), new_shape[1] - 256)); cx0 = int(min(max(round(X0N * new_shape[2] / 384.0 + 128 - 128) - 0, 0), new_shape[2] - 256))
cx0 = int(min(max(round((X0N + 128) * new_shape[2] / 384.0) - 128, 0), new_shape[2] - 256))
crop = np.ascontiguousarray(xi[:, cy0:cy0 + 256, cx0:cx0 + 256], dtype=np.float32)
row = int(round(yr)) - cy0; trow = row // 8
print("crop", crop.shape, "cy0", cy0, "cx0", cx0, "row", row, "token row", trow, flush=True)
ROWS = slice(max(row - 16, 0), row + 17); TROWS = slice(max(trow - 2, 0), trow + 3)

dev = torch.device("cpu")
net, meta = build_from_dir(f"{_NSP_ROOT}/models/models/orx_t3d_r5s1_s0", dev)
cap = {}
cap["enc"] = []
net.net.encoder.neck.register_forward_hook(lambda m, i, o: cap["enc"].append(o.detach()))
net.dec.out.register_forward_pre_hook(lambda m, i: cap.__setitem__("dec", i[0].detach()))
Z = crop.shape[0]; wz = 32
sz = _starts(Z, wz, 0.25)
if BENCH:
    sz = sz[:1]
tz = _taper(wz, len(sz) > 1)
acc = np.zeros((4, Z, ROWS.stop - ROWS.start, 256), np.float32)
accE = np.zeros((256, Z, TROWS.stop - TROWS.start, 32), np.float32)
accD = np.zeros((256, Z, TROWS.stop - TROWS.start, 32), np.float32)
ws = np.zeros(Z, np.float32)
vt = torch.from_numpy(crop)
t0 = time.time()
with torch.no_grad():
    for z0 in sz:
        cap["enc"] = []
        y = net(vt[z0:z0 + wz][None, None])[0].numpy()            # (4,32,256,256) raw (5x unit flow, logit)
        e = torch.cat(cap["enc"]).numpy()                                    # (32,256,32,32) per plane
        dd = cap["dec"][0].numpy()                                  # (256,32,32,32)
        w = tz[:, None, None]
        acc[:, z0:z0 + wz] += y[:, :, ROWS] * w
        accE[:, z0:z0 + wz] += np.transpose(e, (1, 0, 2, 3))[:, :, TROWS] * w
        accD[:, z0:z0 + wz] += dd[:, :, TROWS] * w
        ws[z0:z0 + wz] += tz
        print("S1 window", z0, f"{time.time() - t0:.0f}s", flush=True)
if BENCH:
    print("bench S1 per window", time.time() - t0); sys.exit()
den = np.maximum(ws, 1e-6)[None, :, None, None]
acc /= den; accE /= den; accD /= den
del net
# 2D control: same crop, each z plane through the 2D CPSAM network (the xy view of Cellpose run_3D), plus its encoder neck features
n2 = load_cpsam2d("orx_v5_2d_s0").to(dev).eval()
cap2 = {}
n2.encoder.neck.register_forward_hook(lambda m, i, o: cap2.__setitem__("enc", o.detach()))
y2 = np.zeros((3, Z, ROWS.stop - ROWS.start, 256), np.float32); e2 = np.zeros((256, Z, TROWS.stop - TROWS.start, 32), np.float32)
t0 = time.time()
with torch.no_grad():
    for z0 in range(0, Z, 16):
        b = vt[z0:z0 + 16][:, None]
        b3 = torch.cat([b, torch.zeros_like(b), torch.zeros_like(b)], 1) if n2.encoder.patch_embed.proj.in_channels == 3 else b
        o = n2(b3)[0].numpy()                                        # (n,3,256,256)
        y2[:, z0:z0 + 16] = np.transpose(o, (1, 0, 2, 3))[:, :, ROWS]
        e2[:, z0:z0 + 16] = np.transpose(cap2["enc"].numpy(), (1, 0, 2, 3))[:, :, TROWS]
        print("2D planes", z0, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(f"{OUT}/v2v4_crop.npz", s1_out=acc.astype(np.float16), s1_enc=accE.astype(np.float16), s1_dec=accD.astype(np.float16),
                    c2d_out=y2.astype(np.float16), c2d_enc=e2.astype(np.float16), crop_img=crop[:, ROWS].astype(np.float16),
                    rows=np.array([ROWS.start, ROWS.stop]), trows=np.array([TROWS.start, TROWS.stop]), row=row, trow=trow,
                    cy0=cy0, cx0=cx0, new_shape=np.array(new_shape), d=d, Zu=Zu, vox=np.array(vox))
json.dump(dict(script=os.path.abspath(__file__), src=SRC, d=d, Zu=Zu, new_shape=new_shape, cy0=cy0, cx0=cx0, row=row, trow=trow,
               z_windows=sz, note="S1 identity view only (pipeline used --views 3); 2D net on xy planes only (pipeline fuses 3 views). "
               "Crop 256x256 in resized space; full z. Outputs are raw network outputs (5x unit flow, logit) before x2/x3 scaling and smoothing."),
          open(f"{OUT}/v2v4_crop.json", "w"), indent=1)
print("done", flush=True)
