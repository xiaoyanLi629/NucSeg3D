"""nc-viz ISO_pair extension (selection_rules.json ISO_pair_ext; approved by team lead 2026-11-10). Descriptive; S1 seed 0; gt protocol.
Volumes: the 144 test volumes with z/xy >= 2 (those in the G6 'off' run orx_d10cd4b9).
On  = formal S1 seed-0 final masks ($NUCSEG3D_SCRATCH/nc/errors/cache/gt_S1_s0/<key>.npz 'mask': rule T, fragments < 0.15 x expected removed).
Off = G6 masks npz (gx_eval --gx_iso off): rule-T threshold from its ruleT.json; t = -2 -> saved pred; otherwise recomputed from the saved
      native-grid dP/cp with Cellpose _compute_masks on the native grid (the grid 'off' follows on; float16 rounding only;
      5 volumes; drop_ignored is not re-applied to these, which only affects predictions in unannotated regions); fragments < 0.15 x
      expected volume removed.
Pairs: z-stacked GT pairs = touching (>= 1 shared 6-connected face) with centroid offset (um) mainly along z (errors/common.is_z).
States: merged (one prediction covers > 50 % of both), separated (both best IoU >= 0.5 with different predictions), other.
Consistency check: merge fraction of GT nuclei from the rows.json taxonomy at each volume's rule-T threshold and frac 0.15 (both runs),
compared in direction with the paper (2.2 % on vs 3.3 % off, g6_test.json T_err_on/off).
Output: $NUCSEG3D_SCRATCH/nc/figs/viz/iso_off/iso_ext.json. Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 python viz_iso_ext.py [--workers 4]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import common as C  # noqa: E402

OFF = [d for d in os.listdir(f"{_NSP_ROOT}/results") if d.startswith("orx_d10cd4b9")][0]
OFF = f"{_NSP_ROOT}/results/{OFF}"
ON = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))["gt_S1_s0"]
RT = {f"{x['ds']}__{x['id']}": x["t"] for x in json.load(open(f"{OFF}/ruleT.json"))}


def drop_small(m, d, vox):
    ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    c = np.bincount(m.ravel()); s = np.nonzero(c < 0.15 * ev)[0]; s = s[s > 0]
    return np.where(np.isin(m, s), 0, m)


def off_mask(key, z):
    t = RT[key]; d = float(z["d"]); vox = np.asarray(z["vox"], float)
    if t == -2.0:
        pm = z["pred"].astype(np.int32)
    else:
        import torch
        from cellpose import models

        class Dev:
            device = torch.device("cpu")
        cp = z["cp"].astype(np.float32); dP = z["dP"].astype(np.float32)
        pm = np.asarray(models.CellposeModel._compute_masks(Dev(), (*cp.shape, 1), dP, cp, flow_threshold=0.4, cellprob_threshold=t,
                                                            min_size=15, max_size_fraction=0.4, niter=int(200 / (30.0 / d)), do_3D=True)).squeeze().astype(np.int32)
    return drop_small(pm, d, vox), t


def zpairs(gt, vox):
    ids = np.unique(gt); ids = ids[ids > 0]
    if ids.size < 2:
        return [], {}
    cen = dict(zip(ids.tolist(), (np.array(ndi.center_of_mass(np.ones_like(gt, np.uint8), gt, ids)).reshape(-1, 3) * vox).tolist()))
    pairs = set()
    for ax in range(3):
        a = np.take(gt, range(gt.shape[ax] - 1), axis=ax); b = np.take(gt, range(1, gt.shape[ax]), axis=ax)
        m = (a > 0) & (b > 0) & (a != b)
        u = np.unique(np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], 1), axis=0)
        pairs |= {tuple(map(int, x)) for x in u}
    return sorted(p for p in pairs if C.is_z(np.array(cen[p[1]]) - np.array(cen[p[0]]))), cen


def states(gt, pm, pairs):
    gids, pids, gsz, psz, pg, pp, inter = C.overlaps(gt, pm)
    cov = {}; iou = {}
    for g, p, c in zip(pg, pp, inter):
        cov[(int(g), int(p))] = c / gsz[g]; iou[(int(g), int(p))] = c / (gsz[g] + psz[p] - c)
    byg = {}
    for (g, p) in cov:
        byg.setdefault(g, []).append(p)
    out = []
    for a, b in pairs:
        common = set(byg.get(a, [])) & set(byg.get(b, []))
        if any(cov[(a, p)] > 0.5 and cov[(b, p)] > 0.5 for p in common):
            out.append("merged"); continue
        def best(g):
            L = [(iou[(g, p)], p) for p in byg.get(g, [])]
            return max(L) if L else (0.0, None)
        (ia, pa), (ib, pb) = best(a), best(b)
        out.append("separated" if ia >= 0.5 and ib >= 0.5 and pa != pb else "other")
    return out


def job(key):
    zo = np.load(f"{OFF}/masks/{key}.npz"); gt = zo["gt"].astype(np.int32); vox = np.asarray(zo["vox"], float)
    on = np.load(f"{_NSP_SCRATCH}/nc/errors/cache/gt_S1_s0/{key}.npz")["mask"].astype(np.int32)
    gon = np.load(f"{ON}/masks/{key}.npz")["gt"]
    assert np.array_equal(gon, gt), key
    off, t = off_mask(key, zo)
    pairs, _ = zpairs(gt, vox)
    so, sn = states(gt, off, pairs), states(gt, on, pairs)
    cnt = np.bincount(gt.ravel())
    return [dict(key=key, a=a, b=b, nvox_a=int(cnt[a]), nvox_b=int(cnt[b]), off=x, on=y, t_off=t) for (a, b), x, y in zip(pairs, so, sn)]


def tax_merge(run, keys, rt):
    rows = json.load(open(f"{run}/rows.json")); n = m = 0
    for r in rows:
        k = f"{r['ds']}__{r['id']}"
        if k in keys and r["thr"] == rt[k] and abs(r["frac"] - 0.15) < 1e-9:
            n += r["n_gt"]; m += r["merge"]
    return m, n


if __name__ == "__main__":
    keys = sorted(RT)
    from multiprocessing import Pool
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    with Pool(W) as p:
        allrows = [r for L in p.imap_unordered(job, keys) for r in L]
    trans = {}
    for r in allrows:
        trans[f"{r['off']}->{r['on']}"] = trans.get(f"{r['off']}->{r['on']}", 0) + 1
    cand = sorted([r for r in allrows if r["off"] == "merged" and r["on"] == "separated"],
                  key=lambda r: ((r["nvox_a"] + r["nvox_b"]) / 2, r["key"], r["a"], r["b"]))
    pick = cand[(len(cand) - 1) // 2] if cand else None
    rt_on = {}
    # rule-T threshold of the formal on run per volume = the threshold of the cache mask (common.rule_t); recover from errors/per
    for k in keys:
        rt_on[k] = json.load(open(f"{_NSP_SCRATCH}/nc/errors/per/gt_S1_s0/{k}.json"))["t"]
    m_on, n_on = tax_merge(ON, set(keys), rt_on); m_off, n_off = tax_merge(OFF, set(keys), RT)
    res = dict(label="descriptive; S1 seed 0; 144 test volumes with z/xy >= 2; gt protocol", n_volumes=len(keys),
               n_zstacked_pairs=len(allrows), transitions=trans,
               summary=dict(merged_off_separated_on=trans.get("merged->separated", 0), separated_off_merged_on=trans.get("separated->merged", 0),
                            merged_both=trans.get("merged->merged", 0), separated_both=trans.get("separated->separated", 0)),
               candidates=cand, pick=pick,
               merge_check=dict(on=dict(merge=m_on, n_gt=n_on, frac=m_on / max(n_on, 1)), off=dict(merge=m_off, n_gt=n_off, frac=m_off / max(n_off, 1)),
                                paper=dict(on=0.0217, off=0.0334, n_gt=17173, source=f"{_NSP_SCRATCH}/nc/gpu/g6/g6_test.json T_err_on/off")),
               rows=allrows)
    json.dump(res, open(f"{_NSP_SCRATCH}/nc/figs/viz/iso_off/iso_ext.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_volumes", "n_zstacked_pairs", "transitions", "summary", "pick", "merge_check")}, indent=1))
