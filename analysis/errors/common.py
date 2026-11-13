"""NC 误差与 AI 分析（CPU）的公用部分：运行发现、规则 T、最终掩膜、稀疏版误差分类、逐核协变量。

口径来源：cpsam_finetune/投稿分析方案.html 第 4 节（A1、A2、S4、C3、C4），评测代码 eval_gt.py（taxonomy）、
test_r6.py（recs 的规则 T、hits 的最终掩膜）。只读 $NUCSEG3D_SCRATCH/r6 与 $NUCSEG3D_ROOT/results，不改它们。
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import functools
import glob
import json
import math
import os
import sys

import numpy as np

RES = f"{_NSP_ROOT}/results"
STATE = f"{_NSP_SCRATCH}/r6/sched_state.json"
SRC = _nsp_os.path.join(_NSP_REPO, 'src')          # 与原 thr_eval 运行相同的 eval_gt / eval_cpsam 快照（load_case、drop_ignored）
OUT = f"{_NSP_SCRATCH}/nc/errors"
AUDIT = f"{_NSP_SHARED}/cellposeft_orx/audit"
DROP = {"bbbc032"}
SEEDS = ("s0", "s1000", "s2000", "s3000", "s4000")
MODELS = ("S1", "2D", "base")
FRAC = 0.15
IOU_GRID = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))     # 0.50 … 0.95
CLS = ("hit", "miss", "merge", "split", "boundary")
INT_FIELDS = ("n_gt", "n_pred", "tp5", "tp7", "hit", "miss", "merge", "split", "boundary",
              "fpc_false", "fpc_merge", "fpc_split", "fpc_boundary", "merge_z", "merge_xy")


# ---------------------------------------------------------------- 运行发现
def discover():
    """返回 {(proto, model, seed): 结果目录}；proto ∈ {gt, fp}；只收已完成（有 summary.json）的运行。"""
    st = json.load(open(STATE, encoding="utf-8"))
    out = {}
    for proto, pre in (("gt", "tGT"), ("fp", "tFP")):
        for nm in ("S1", "2D"):
            for s in SEEDS:
                for k in (f"{pre}_{nm}_{s}", f"S5_{pre}_{nm}_{s}"):
                    if k in st and st[k].get("run"):
                        d = f"{RES}/orx_{st[k]['run']}_thr_eval"
                        if os.path.exists(f"{d}/summary.json") and os.path.exists(f"{d}/rows.json"):
                            out[(proto, nm, s)] = d
        k = f"{pre}_base"
        d = f"{RES}/orx_{st[k]['run']}_thr_eval"
        if os.path.exists(f"{d}/summary.json"):
            out[(proto, "base", "s0")] = d
    return out


def tag(proto, nm, s):
    return f"{proto}_{nm}_{s}"


@functools.lru_cache(maxsize=4)
def load_rows(d):
    rows = json.load(open(d + "/rows.json"))
    by = {}
    for x in rows:
        by.setdefault((x["ds"], x["id"]), {})[(x["thr"], x["frac"])] = x
    return by


def subsets():
    c1 = json.load(open(f"{AUDIT}/clean_test.json", encoding="utf-8"))["clean"]
    c2 = json.load(open(f"{AUDIT}/clean_test_plus_b.json", encoding="utf-8"))["keep"]
    return {"all": None, "C1": {tuple(x) for x in c1}, "C2": {tuple(x) for x in c2}}


# ---------------------------------------------------------------- 规则 T（与 test_r6.recs 逐字相同的判定）
def rule_t(cp, vrows, fixed=False):
    thrs = sorted({t for t, _ in vrows}, reverse=True)
    r0 = vrows[(-2.0, 0.0)]
    dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
    m = {t: float((cp > t).sum()) / max(vrows[(t, 0.0)]["n_pred"], 1) / V for t in thrs}
    t2 = -2.0 if (fixed or m[-2.0] >= 0.7) else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
    return t2, m


# ---------------------------------------------------------------- 最终掩膜
def zi_of(z0, ratio):
    Zu = int(round(z0 * ratio))
    return Zu, np.round(np.linspace(0, Zu - 1, z0)).astype(int)


def recompute(z, vrows, t, ign=None):
    """阈值 t ≠ −2 时由存下的 dP、cp（float16，原层位）重算掩膜。
    z/xy < 2：存档就是跟流网格，只差 float16 舍入（r8 复现核对：近乎逐位一致）。
    z/xy ≥ 2：存档只有原层位，沿 z 线性插值回各向同性网格再跟流、取回原层位——近似，与 rows.json 的差另行核对。"""
    import torch
    from cellpose import models
    torch.set_num_threads(int(os.environ.get("NC_TORCH_THREADS", "2")))

    class Dev:
        device = torch.device("cpu")

    d = float(z["d"]); r0 = vrows[(-2.0, 0.0)]
    dP = z["dP"].astype(np.float32); cp = z["cp"].astype(np.float32)
    zi = None
    if r0["iso"]:
        from scipy.ndimage import map_coordinates  # noqa: F401
        Zu, zi = zi_of(cp.shape[0], r0["ratio"])
        q = np.arange(Zu, dtype=np.float64)
        j = np.clip(np.searchsorted(zi, q, side="right") - 1, 0, len(zi) - 2)
        w = ((q - zi[j]) / np.maximum(zi[j + 1] - zi[j], 1)).astype(np.float32)[:, None, None]
        cp = cp[j] * (1 - w) + cp[j + 1] * w
        dP = dP[:, j] * (1 - w[None]) + dP[:, j + 1] * w[None]
    m = models.CellposeModel._compute_masks(Dev(), (*cp.shape, 1), dP, cp, flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                                            max_size_fraction=0.4, niter=int(200 / (30.0 / d)), do_3D=True)
    m = np.asarray(m).squeeze()
    if zi is not None:
        m = m[zi]
    m = m.astype(np.int32)
    if ign is not None:
        sys.path.insert(0, SRC)
        from eval_cpsam import drop_ignored
        m = drop_ignored(m, ign)
    return m


def drop_small(m, d, vox, frac=FRAC):
    ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(m.ravel())
    small = np.where(cnt < frac * ev)[0]; small = small[small > 0]
    if small.size:
        m = m.copy(); m[np.isin(m, small)] = 0
    return m


# ---------------------------------------------------------------- 稀疏版 taxonomy（与 eval_gt.taxonomy 规则相同）
def overlaps(gt, pred):
    gids = np.unique(gt); gids = gids[gids > 0]
    pids = np.unique(pred); pids = pids[pids > 0]
    gsz = np.bincount(gt.ravel(), minlength=int(gt.max()) + 1)
    psz = np.bincount(pred.ravel(), minlength=int(pred.max()) + 1)
    msk = (gt > 0) & (pred > 0)
    g, p = gt[msk].astype(np.int64), pred[msk].astype(np.int64)
    key = g * (int(pred.max()) + 1) + p
    u, c = np.unique(key, return_counts=True)
    pg, pp = u // (int(pred.max()) + 1), u % (int(pred.max()) + 1)
    return gids, pids, gsz, psz, pg, pp, c.astype(np.float64)


def hungarian(gids, pids, pg, pp, iou):
    """按重叠图的连通分量分别做 linear_sum_assignment（与稠密矩阵上的匈牙利匹配最优值相同；IoU 为 0 的配对不影响计数）。"""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(pg) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    gi = {v: i for i, v in enumerate(gids)}; pi = {v: i for i, v in enumerate(pids)}
    a = np.array([gi[v] for v in pg]); b = np.array([pi[v] for v in pp])
    ng, npr = len(gids), len(pids)
    A = coo_matrix((np.ones(len(a)), (a, ng + b)), shape=(ng + npr, ng + npr))
    _, lab = connected_components(A, directed=False)
    rows_g, rows_p, vals = [], [], []
    comp = lab[a]
    order = np.argsort(comp, kind="stable")
    a, b, w, comp = a[order], b[order], iou[order], comp[order]
    cuts = np.flatnonzero(np.diff(comp)) + 1
    for sa, sb, sw in zip(np.split(a, cuts), np.split(b, cuts), np.split(w, cuts)):
        ug, ia = np.unique(sa, return_inverse=True); up, ib = np.unique(sb, return_inverse=True)
        M = np.zeros((len(ug), len(up))); M[ia, ib] = sw
        r, c = linear_sum_assignment(-M)
        rows_g += list(ug[r]); rows_p += list(up[c]); vals += list(M[r, c])
    return np.array(rows_g), np.array(rows_p), np.array(vals)


def is_z(dv):
    return abs(dv[0]) > np.hypot(dv[1], dv[2])


def analyse(gt, pred, vox):
    """返回 (计数 dict，逐真值核 dict，逐预测核 dict)。计数字段与 eval_gt.taxonomy 相同，外加 IoU 网格上的 TP 与 z 向切碎。"""
    from scipy import ndimage as ndi
    gids, pids, gsz, psz, pg, pp, inter = overlaps(gt, pred)
    ng, npr = len(gids), len(pids)
    res = dict(n_gt=ng, n_pred=npr, tp5=0, tp7=0, **{k: 0 for k in CLS},
               fpc_false=0, fpc_merge=0, fpc_split=0, fpc_boundary=0, merge_z=0, merge_xy=0,
               tp_grid=[0] * len(IOU_GRID), zsplit=0, zsplit_err=0, zsplit_any=0)
    G = dict(id=gids, cls=np.zeros(ng, np.int8), zsplit=np.zeros(ng, bool), zmerge=np.zeros(ng, bool), xymerge=np.zeros(ng, bool),
             best=np.zeros(ng))
    P = dict(id=pids, best=np.zeros(npr))
    if ng == 0:
        res["fpc_false"] = npr; return res, G, P
    if npr == 0:
        res["miss"] = ng; G["cls"][:] = 1; return res, G, P
    iou = inter / (gsz[pg] + psz[pp] - inter)
    cov = inter / gsz[pg]
    gi = np.zeros(int(gt.max()) + 1, np.int64); gi[gids] = np.arange(ng)
    pi = np.zeros(int(pred.max()) + 1, np.int64); pi[pids] = np.arange(npr)
    a, b = gi[pg], pi[pp]
    # 匈牙利匹配
    mg, mp, mv = hungarian(gids, pids, pg, pp, iou)
    res["tp5"] = int((mv >= 0.5).sum()); res["tp7"] = int((mv >= 0.7).sum())
    res["tp_grid"] = [int((mv >= t - 1e-12).sum()) for t in IOU_GRID]
    tp_p = {int(j) for j, v in zip(mp, mv) if v >= 0.5}
    best = np.zeros(ng); np.maximum.at(best, a, iou)
    bcov = np.zeros(ng); np.maximum.at(bcov, a, cov)
    pbest = np.zeros(npr); np.maximum.at(pbest, b, iou)
    G["best"] = best; P["best"] = pbest
    # 每个真值核：行（按预测下标升序，与 np.argmax 的取法一致）
    order = np.lexsort((b, a))
    a_s, b_s, cov_s, int_s = a[order], b[order], cov[order], inter[order]
    starts = np.searchsorted(a_s, np.arange(ng)); ends = np.searchsorted(a_s, np.arange(ng), side="right")
    # 每个预测核上各真值核的 cov（列）
    col = {}
    for i, j, c in zip(a, b, cov):
        col.setdefault(int(j), []).append((int(i), float(c)))
    cen = np.array(ndi.center_of_mass(np.ones_like(gt, np.uint8), gt, gids)).reshape(-1, 3) * np.asarray(vox)
    cls = []
    for i in range(ng):
        s, e = starts[i], ends[i]
        if best[i] >= 0.5:
            c_ = "hit"
        elif best[i] < 0.1 and bcov[i] < 0.15:
            c_ = "miss"
        else:
            if e > s:
                k = s + int(np.argmax(cov_s[s:e])); j = int(b_s[k]); cij = cov_s[k]
                oth = [c for (g_, c) in col.get(j, []) if g_ != i]
                om = max(oth) if oth else 0.0
            else:
                om, cij = 0.0, 0.0
            if om > 0.5 and cij > 0.3:
                c_ = "merge"
            elif (cov_s[s:e] >= 0.15).sum() >= 2:
                c_ = "split"
            else:
                c_ = "boundary"
        cls.append(c_); res[c_] += 1
    G["cls"] = np.array([CLS.index(c) for c in cls], np.int8)
    # 预测核一侧
    ob = np.argsort(b, kind="stable"); bs_ = b[ob]
    cutb = np.flatnonzero(np.diff(bs_)) + 1
    bix = {int(b[s[0]]): s for s in np.split(ob, cutb) if s.size}
    for j in range(npr):
        L = col.get(j, [])
        g = sorted(i for i, c in L if c > 0.5)
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                if is_z(cen[g[y]] - cen[g[x]]):
                    res["merge_z"] += 1; G["zmerge"][g[x]] = G["zmerge"][g[y]] = True
                else:
                    res["merge_xy"] += 1; G["xymerge"][g[x]] = G["xymerge"][g[y]] = True
        if j in tp_p:
            continue
        osum, oarg = 0.0, -1
        sel = bix.get(j, np.zeros(0, int))
        if sel.size:
            osum = inter[sel].sum()
            # np.argmax(o) 在稠密列上取最小真值下标的最大值
            ii = a[sel]; vv = inter[sel]; mx = vv.max(); oarg = int(ii[vv == mx].min())
        if osum < 0.1 * psz[pids[j]]:
            res["fpc_false"] += 1
        elif len(g) >= 2:
            res["fpc_merge"] += 1
        else:
            cb = cls[oarg]
            res["fpc_split" if cb in ("split", "hit") else ("fpc_merge" if cb == "merge" else "fpc_boundary")] += 1
    # z 向切碎：真值核被 ≥2 个预测核分走（每个覆盖 ≥15%，与 split 判据相同），且其中某两个预测核的 xy 投影 IoU > 0.5
    objs = ndi.find_objects(pred)
    proj_cache = {}

    def proj(lbl):
        if lbl not in proj_cache:
            sl = objs[lbl - 1]
            proj_cache[lbl] = (sl[1].start, sl[2].start, (pred[sl] == lbl).any(0))
        return proj_cache[lbl]

    def piou(l1, l2):
        y1, x1, m1 = proj(l1); y2, x2, m2 = proj(l2)
        y0, x0 = min(y1, y2), min(x1, x2)
        Y = max(y1 + m1.shape[0], y2 + m2.shape[0]) - y0; X = max(x1 + m1.shape[1], x2 + m2.shape[1]) - x0
        A_ = np.zeros((Y, X), bool); B_ = np.zeros((Y, X), bool)
        A_[y1 - y0:y1 - y0 + m1.shape[0], x1 - x0:x1 - x0 + m1.shape[1]] = m1
        B_[y2 - y0:y2 - y0 + m2.shape[0], x2 - x0:x2 - x0 + m2.shape[1]] = m2
        u = (A_ | B_).sum()
        return (A_ & B_).sum() / u if u else 0.0

    for i in range(ng):
        s, e = starts[i], ends[i]
        ps = [int(pids[b_s[k]]) for k in range(s, e) if cov_s[k] >= 0.15]
        if len(ps) < 2:
            continue
        hitz = any(piou(ps[x], ps[y]) > 0.5 for x in range(len(ps)) for y in range(x + 1, len(ps)))
        if hitz:
            G["zsplit"][i] = True
            res["zsplit_any"] += 1
            if cls[i] != "hit":
                res["zsplit_err"] += 1
            if cls[i] == "split":
                res["zsplit"] += 1
    return res, G, P


def gt_covariates(gt, vox, gids):
    """逐真值核协变量：体积（µm³）、相对体积（÷ 本体积真值核体积中位数）、相对深度（质心 z ÷ (Z−1)，0 = 第一层）、
    局部拥挤度（到 10 个最近邻真值核质心的平均距离 ÷ 本体积真值核等效球直径中位数；越小越挤；核数 ≤ 10 时用全部其余核）。"""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    n = len(gids)
    vv = float(np.prod(vox))
    sz = np.bincount(gt.ravel(), minlength=int(gt.max()) + 1)[gids].astype(np.float64) * vv
    cen = np.array(ndi.center_of_mass(np.ones_like(gt, np.uint8), gt, gids)).reshape(-1, 3)
    depth = cen[:, 0] / max(gt.shape[0] - 1, 1)
    eqd = (6 * sz / np.pi) ** (1 / 3)
    medd = float(np.median(eqd)) if n else np.nan
    crowd = np.full(n, np.nan)
    if n >= 2:
        k = min(10, n - 1)
        dist, _ = cKDTree(cen * np.asarray(vox)).query(cen * np.asarray(vox), k=k + 1)
        crowd = dist[:, 1:].mean(1) / medd
    return dict(vol_um3=sz, relvol=sz / np.median(sz) if n else sz, depth=depth, crowd=crowd)
