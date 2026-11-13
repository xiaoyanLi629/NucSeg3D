"""nc-viz V3: flow-following trajectories for the pre-specified GT nucleus pair (paper/figures/assets_viz/selection_rules.json, V3_pair).
Uses the saved pipeline outputs (thr_eval masks npz: dP, cellprob at native planes) for S1 seed 0 and the 2D control seed 0,
linearly interpolated back to the isotropic grid exactly as $NUCSEG3D_SCRATCH/nc/errors/common.recompute does, then Cellpose 4.2
dynamics.steps_interp (copied, with positions recorded every few iterations) on dP*(cp>-2)/5, niter = int(200/(30/d)) as in eval_gt.
Final masks are the evaluated ones (errors/cache/gt_{S1,2D}_s0). CPU only, read-only; writes $NUCSEG3D_SCRATCH/nc/figs/viz/v3_pair.npz."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from scipy import ndimage as ndi
torch.set_num_threads(8)
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import common as C

cand = json.load(open(sys.argv[4] if len(sys.argv) > 4 else f"{_NSP_SCRATCH}/nc/figs/viz/v3_candidates.json"))
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
if len(sys.argv) > 2:
    sel = [c for c in cand if c["key"] == sys.argv[1] and c["gt_id"] == int(sys.argv[2])][0]
else:
    sel = cand[0]
key, tid = sel["key"], sel["gt_id"]
runs = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))
Z = {m: np.load(f"{runs['gt_' + m + '_s0']}/masks/{key}.npz") for m in ("S1", "2D")}
gt = Z["S1"]["gt"].astype(np.int32); vox = np.asarray(Z["S1"]["vox"], float); d = float(Z["S1"]["d"])
assert np.array_equal(Z["2D"]["gt"], gt)
M = {m: np.load(f"{_NSP_SCRATCH}/nc/errors/cache/gt_{m}_s0/{key}.npz")["mask"] for m in ("S1", "2D")}
ratio = vox[0] / vox[1]
# partner
if "partner" in sel:
    partner = int(sel["partner"]); prule = "pair given by the candidate file (2D z-merge pair, revision 3)"
elif sel["cls2D"] == "merge":
    p2 = M["2D"]; ov = p2[gt == tid]; j = np.bincount(ov[ov > 0]).argmax()
    cov = {g: ((gt == g) & (p2 == j)).sum() / (gt == g).sum() for g in np.unique(gt[p2 == j]) if g not in (0, tid)}
    partner = max(cov, key=cov.get); prule = f"other GT nucleus covered most by 2D prediction {j} (cov {cov[partner]:.2f})"
else:
    t = gt == tid; contact = {}
    for ax in range(3):
        for sgn in (1, -1):
            nb = np.roll(gt, sgn, axis=ax); sl = [slice(None)] * 3; sl[ax] = slice(0, 1) if sgn == 1 else slice(-1, None)
            nb[tuple(sl)] = 0
            v = nb[t]; v = v[(v > 0) & (v != tid)]
            for g, c in zip(*np.unique(v, return_counts=True)):
                contact[int(g)] = contact.get(int(g), 0) + int(c)
    if contact:
        partner = max(contact, key=contact.get); prule = f"GT neighbour with the largest 6-connected contact ({contact[partner]} faces)"
    else:   # no touching neighbour: nearest GT nucleus by physical distance to the target's voxels
        dt = ndi.distance_transform_edt(~t, sampling=vox)
        ids_ = np.unique(gt); ids_ = ids_[(ids_ > 0) & (ids_ != tid)]
        dmin = ndi.minimum(dt, gt, ids_)
        partner = int(ids_[int(np.argmin(dmin))]); prule = f"no touching neighbour; nearest GT nucleus ({float(np.min(dmin)):.2f} um gap)"
print(key, tid, "partner", partner, prule, flush=True)
# crop box around the pair, padded by d/2 in xy and d/2/ratio planes in z
pair = (gt == tid) | (gt == partner)
zz, yy, xx = np.nonzero(pair)
pz, pxy = int(np.ceil(d / 2 / ratio)) + 2, int(np.ceil(d / 2))
b0 = [max(zz.min() - pz, 0), max(yy.min() - pxy, 0), max(xx.min() - pxy, 0)]
b1 = [min(zz.max() + pz + 1, gt.shape[0]), min(yy.max() + pxy + 1, gt.shape[1]), min(xx.max() + pxy + 1, gt.shape[2])]
sl = tuple(slice(a, b) for a, b in zip(b0, b1))
Zu, zi = C.zi_of(gt.shape[0], ratio)
iz0, iz1 = zi[b0[0]], zi[b1[0] - 1] + 1                        # isotropic planes covered by the crop
q = np.arange(iz0, iz1, dtype=np.float64)
j = np.clip(np.searchsorted(zi, q, side="right") - 1, 0, len(zi) - 2)
w = ((q - zi[j]) / np.maximum(zi[j + 1] - zi[j], 1)).astype(np.float32)[:, None, None]
niter = int(200 / (30.0 / d))
out = dict(key=key, tid=tid, partner=partner, vox=vox, d=d, ratio=ratio, b0=np.array(b0), b1=np.array(b1), iz0=iz0, zi=zi,
           gt=gt[sl], niter=niter)
# seeds: native-plane voxels of the pair, mapped to isotropic z
sz, sy, sx = np.nonzero(pair[sl])
lab = gt[sl][sz, sy, sx]
iso_z = zi[sz + b0[0]] - iz0
for m in ("S1", "2D"):
    dP = Z[m]["dP"][:, :, sl[1], sl[2]].astype(np.float32); cp = Z[m]["cp"][:, sl[1], sl[2]].astype(np.float32)
    cpi = cp[j] * (1 - w) + cp[j + 1] * w
    dPi = dP[:, j] * (1 - w[None]) + dP[:, j + 1] * w[None]
    eff = dPi * (cpi > -2) / 5.0
    shape = eff.shape[1:]
    keep = cpi[iso_z, sy, sx] > -2                               # only pixels the pipeline runs dynamics on
    inds = (iso_z[keep].astype(np.float32), sy[keep].astype(np.float32), sx[keep].astype(np.float32))
    pt = torch.zeros((1, 1, 1, len(inds[0]), 3)); im = torch.zeros((1, 3, *shape))
    for n in range(3):
        pt[0, 0, 0, :, 2 - n] = torch.from_numpy(inds[n]); im[0, 2 - n] = torch.from_numpy(eff[n])
    shp = np.array(shape)[::-1].astype(float) - 1
    for k in range(3):
        im[:, k] *= 2. / shp[k]; pt[..., k] /= shp[k]
    pt = pt * 2 - 1
    rec = []
    every = max(niter // 60, 1)

    def unnorm(p):
        u = (p.clone() + 1) * 0.5
        for k in range(3):
            u[..., k] *= shp[k]
        return u[0, 0, 0][:, [2, 1, 0]].numpy().copy()             # (n, 3) as (z_iso, y, x) crop coords
    rec.append(unnorm(pt))
    with torch.no_grad():
        for t in range(niter):
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
            for k in range(3):
                pt[..., k] += dPt[:, k]
                torch.clamp_(pt[..., k], -1., 1.)
            if (t + 1) % every == 0 or t == niter - 1:
                rec.append(unnorm(pt))
    traj = np.stack(rec, 1).astype(np.float32)                    # (n, steps, 3)
    out[f"traj_{m}"] = traj; out[f"lab_{m}"] = lab[keep]; out[f"seed_{m}"] = np.stack([sz[keep], sy[keep], sx[keep]], 1)
    out[f"mask_{m}"] = M[m][sl]
    out[f"nseed_dropped_{m}"] = int((~keep).sum())
    print(m, "traj", traj.shape, "dropped (cp<=-2)", int((~keep).sum()), flush=True)
np.savez_compressed(f"{_NSP_SCRATCH}/nc/figs/viz/v3_pair{TAG}.npz", **out)
json.dump(dict(sel=sel, partner=int(partner), partner_rule=prule, box_native=[b0, b1], iso_planes=[int(iz0), int(iz1)], niter=niter,
               record_every=every, script=os.path.abspath(__file__)), open(f"{_NSP_SCRATCH}/nc/figs/viz/v3_pair{TAG}.json", "w"), indent=1, default=int)
print("done")
