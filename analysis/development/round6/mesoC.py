"""第六轮诊断 C：mesoSPIM 开发体积 c1–c5（源数据体素 5/10.52/10.52，gt 协议、单视图）上二维、A1、C、X1、S1（另附 X3、X2）
各种子的差距在 z 还是 xy；种子 2000 的三维模型为什么特别差。只读第五轮 meso_diag 结果（summary/rows/shape_diag 与
masks/*.npz 里 cellprob −2 的预测、真值与网络 cellprob），CPU。
逐个真值核 g、取重叠最多的预测 p：
  iou      三维 IoU；
  iou_xy   两者共有的层里逐层交并之和的比（“z 范围对上以后”的 IoU，只看 xy）；
  iou_zo   “xy 截面换成真值、只保留预测的 z 范围”的 IoU：Σ_{共有层} A_g / (Σ_{g 的层} A_g + Σ_{p 独有层} A_p)（只看 z）；
  nz       预测层数 / 真值层数；dxy 最大截面等效直径比；
  cov      真值体素里 cellprob > −2 的比例（网络前景覆盖）；cpc 真值核质心处的 cellprob；
反事实命中率：hit = iou ≥ 0.5；hit_xyfix = iou_zo ≥ 0.5（xy 全对时）；hit_zfix = iou_xy ≥ 0.5（z 全对时）。
输出 $NUCSEG3D_SCRATCH/r6/diag/mesoC.json 与打印的表。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob
import json
import multiprocessing as mp
import os

import numpy as np
from scipy import ndimage as ndi

R = f"{_NSP_ROOT}/results/"
VAR = {  # (模型, 种子) -> meso_diag 子目录
    ("2D", "0"): "0d0c694b*/2d_s0", ("A1", "0"): "0d0c694b*/a1_s0", ("C", "0"): "0d0c694b*/C", ("X1", "0"): "4dc44a35*/X1",
    ("S1", "0"): "bba3badf*/S1", ("X3", "0"): "bc6d10a5*/X3", ("X2", "0"): "3ed9690f*/X2",
    ("2D", "1000"): "58715d57*/2d_s1000_phys", ("A1", "1000"): "58715d57*/a1_s1000_phys", ("C", "1000"): "0e7c7c0f*/C_1000",
    ("X1", "1000"): "5d00c86d*/X1_1000", ("S1", "1000"): "f333910e*/S1_1000", ("X3", "1000"): "2a0c8e11*/X3_1000",
    ("2D", "2000"): "58715d57*/2d_s2000_phys", ("A1", "2000"): "58715d57*/a1_s2000_phys", ("C", "2000"): "836164dc*/C_2000",
    ("X1", "2000"): "7f58c6a8*/X1_2000", ("S1", "2000"): "9b62f960*/S1_2000", ("X3", "2000"): "293785a6*/X3_2000",
}
ERR = ("miss", "merge", "split", "boundary", "false")


def vdir(pat):
    a, b = pat.split("/")
    m = glob.glob(R + f"orx_{a}_meso_diag/{b}")
    assert len(m) == 1, (pat, m)
    return m[0]


def per_nucleus(npz):
    D = np.load(npz)
    P, G, cp = D["pred"].astype(np.int64), D["gt"].astype(np.int64), D["cp"].astype(np.float32)
    gs, ps = ndi.find_objects(G), ndi.find_objects(P)
    out = []
    for gi, sl in enumerate(gs, start=1):
        if sl is None:
            continue
        gm = G[sl] == gi
        cpg = cp[sl][gm]
        cz, cy, cx = [int(round(float(np.mean(i)) + s.start)) for i, s in zip(np.nonzero(gm), sl)]
        r = dict(g=gi, nz_g=int(gm.any((1, 2)).sum()), cov=float((cpg > -2).mean()), cpc=float(cp[cz, cy, cx]),
                 cpmed=float(np.median(cpg)))
        pv = P[sl][gm]; pv = pv[pv > 0]
        if pv.size == 0:
            r.update(matched=False, iou=0.0, iou_xy=0.0, iou_zo=0.0); out.append(r); continue
        p = int(np.bincount(pv).argmax()); psl = ps[p - 1]
        u = tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(sl, psl))
        g2, p2 = G[u] == gi, P[u] == p
        inter_z = (g2 & p2).sum((1, 2)); uni_z = (g2 | p2).sum((1, 2))
        ag, ap = g2.sum((1, 2)), p2.sum((1, 2))
        zg, zp = ag > 0, ap > 0; both = zg & zp
        iou = inter_z.sum() / uni_z.sum()
        iou_xy = inter_z[both].sum() / max(uni_z[both].sum(), 1)
        iou_zo = ag[both].sum() / max(ag[zg].sum() + ap[zp & ~zg].sum(), 1)
        r.update(matched=True, iou=float(iou), iou_xy=float(iou_xy), iou_zo=float(iou_zo), nz=float(zp.sum() / zg.sum()),
                 dxy=float(np.sqrt(ap.max() / ag.max())), V=float(ap.sum() / ag.sum()))
        out.append(r)
    return out


def job(key_npz):
    key, npz = key_npz
    return key, os.path.basename(npz)[:-4], per_nucleus(npz)


if __name__ == "__main__":
    jobs = []
    meta = {}
    for (m, s), pat in VAR.items():
        d = vdir(pat)
        S = json.load(open(d + "/summary.json"))["cells"]
        thr = sorted({float(k.split("_")[0][2:]) for k in S}, reverse=True)
        curve = {t: (S[f"cp{t:g}_vol0.15"]["per_ds"]["mesospim"]["f1_05"], S[f"cp{t:g}_vol0.15"]["per_ds"]["mesospim"]["f1_07"]) for t in thr}
        c = S["cp-2_vol0.15"]["per_ds"]["mesospim"]
        sd = json.load(open(d + "/shape_diag.json"))["all"]
        rows = [x for x in json.load(open(d + "/rows.json")) if x["thr"] == -2.0 and x["frac"] == 0.15]
        meta[f"{m}|{s}"] = dict(dir=d, f5=c["f1_05"], f7=c["f1_07"], err=c["err"], n_gt=c["n_gt"], n_pred=c["n_pred"], curve=curve,
                                shape=sd, per_vol={x["id"]: dict(f1=x["f1"], n_pred=x["n_pred"], n_gt=x["n_gt"]) for x in rows})
        jobs += [(f"{m}|{s}", f) for f in sorted(glob.glob(d + "/masks/*.npz"))]
    with mp.get_context("spawn").Pool(10) as p:
        res = p.map(job, jobs, chunksize=1)
    nuc = {}
    for key, vol, rs in res:
        nuc.setdefault(key, {})[vol] = rs
    out = {}
    print("== mesoSPIM c1–c5（源数据体素、gt 协议、单视图）pooled；F1 为 cp−2_vol0.15；形状统计为 cp−2 未删小块的预测，对每个真值核取重叠最多的预测")
    print(f"{'模型':4s} {'种子':5s} {'F1@.5':>6s} {'F1@.7':>6s} {'pred/gt':>7s} | " + " ".join(f"{e[:5]:>6s}" for e in ERR)
          + f" | {'中位IoU':>6s} {'IoU_xy':>6s} {'层数比':>6s} {'xy径比':>6s} {'体积比':>6s} | {'命中':>5s} {'xy对时':>6s} {'z对时':>6s} | "
          f"{'cov':>5s} {'cp质心':>6s} {'cp中位':>6s} | 最佳阈值")
    for key in sorted(meta, key=lambda k: (int(k.split("|")[1]), ["2D", "A1", "C", "X1", "S1", "X3", "X2"].index(k.split("|")[0]))):
        M = meta[key]; allr = [r for v in nuc[key].values() for r in v]
        mt = [r for r in allr if r["matched"]]
        med = lambda k, L: float(np.median([r[k] for r in L]))
        hit = np.mean([r["iou"] >= 0.5 for r in allr]); hxy = np.mean([r["iou_zo"] >= 0.5 for r in allr]); hz = np.mean([r["iou_xy"] >= 0.5 for r in allr])
        best = max(M["curve"].items(), key=lambda t: t[1][0])
        o = dict(f5=M["f5"], f7=M["f7"], pred_gt=M["n_pred"] / M["n_gt"], err=M["err"], med_iou=med("iou", mt), med_iou_xy=med("iou_xy", mt),
                 med_iou_zo=med("iou_zo", mt), med_nz=med("nz", mt), med_dxy=med("dxy", mt), med_V=med("V", mt), hit=float(hit),
                 hit_if_xy_fixed=float(hxy), hit_if_z_fixed=float(hz), cov=med("cov", allr), cpc=med("cpc", allr), cpmed=med("cpmed", allr),
                 frac_cpc_gt_m2=float(np.mean([r["cpc"] > -2 for r in allr])), best_thr=best[0], best_f5=best[1][0], curve=M["curve"],
                 per_vol=M["per_vol"], n_gt=len(allr), matched=len(mt))
        out[key] = o
        m, s = key.split("|")
        print(f"{m:4s} {s:5s} {o['f5']:6.3f} {o['f7']:6.3f} {o['pred_gt']:7.3f} | " + " ".join(f"{M['err'][e]:6.3f}" for e in ERR)
              + f" | {o['med_iou']:6.2f} {o['med_iou_xy']:6.2f} {o['med_nz']:6.2f} {o['med_dxy']:6.2f} {o['med_V']:6.2f} | {o['hit']:5.2f} {o['hit_if_xy_fixed']:6.2f} {o['hit_if_z_fixed']:6.2f} | "
              f"{o['cov']:5.2f} {o['cpc']:6.2f} {o['cpmed']:6.2f} | cp{best[0]:g} {best[1][0]:.3f}")
    print("\n== 逐体积 F1@0.5（cp−2_vol0.15）")
    vols = sorted(next(iter(out.values()))["per_vol"])
    for key, o in out.items():
        print(f"  {key:8s} " + " ".join(f"{v} {o['per_vol'][v]['f1']:.3f}" for v in vols))
    json.dump(out, open(f"{_NSP_SCRATCH}/r6/diag/mesoC.json", "w"), indent=1)
    print(f"写出 {_NSP_SCRATCH}/r6/diag/mesoC.json")
