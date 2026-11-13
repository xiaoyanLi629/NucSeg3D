"""第六轮诊断 D 的便宜办法试算（零训练、CPU、只读已存的 cellprob 与逐阈值计数）：按“尺度一致”逐体积选 cellprob 阈值。
推理流程本来就给了直径 d（gt 协议为真值直径，完整流程为 r2a2 选出的 d_used），期望核体积 V = π/6·d³·vxy/vz（与 eval_gt 的体积下限同式）。
对每个阈值 t，用 m(t) = #(cellprob > t) / 预测核数(t, 不删小块) 估计预测核的平均体积，
  R1：t* = argmin_t |log(m(t)/V)|；
  R2（保守）：m(−2)/V ≥ 0.7 时仍用 −2，否则取 m(t)/V ≥ 0.85 的最高阈值（都不到则 −6）。
在 mesoSPIM c1–c5（第五轮各模型各种子，源数据体素）与 D3 的域内验证子集（每套至多 2 个体积，二维与 A1 种子 1000；去 mesospim）上，
比较固定 cp−2 与 R1/R2 的 F1@0.5（体积下限 0.15，pooled）。只用于判断方向，不是预定规则。输出 $NUCSEG3D_SCRATCH/r6/diag/sizeThr.json。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob
import json
import math
import multiprocessing as mp
import os

import numpy as np

R = f"{_NSP_ROOT}/results/"
import sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/development/round6'))
from mesoC import VAR as _V                                            # 第五轮各模型各种子的 meso_diag 子目录
VAR = {f"{m}|{s}": p for (m, s), p in _V.items()}
VAR["D3域内 2D|1000"] = "004b4e35*/2d_s1000_val"; VAR["D3域内 A1|1000"] = "004b4e35*/a1_s1000_val"


def vdir(pat):
    a, b = pat.split("/"); m = glob.glob(R + f"orx_{a}_meso_diag/{b}"); assert len(m) == 1, (pat, m); return m[0]


def job(args):
    key, d = args
    rows = json.load(open(d + "/rows.json"))
    by = {(x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True)
    out = []
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        if key.startswith("D3") and ds == "mesospim":
            continue
        cp = np.load(f)["cp"].astype(np.float32)
        r0 = by[(vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
        r1 = min(thrs, key=lambda t: abs(math.log(max(m[t], 1e-9))))
        r2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
        rec = dict(ds=ds, id=vid, ratio_m=m, t_r1=r1, t_r2=r2)
        for nm, t in (("fix", -2.0), ("r1", r1), ("r2", r2)):
            x = by[(vid, t, 0.15)]
            rec[nm] = dict(tp5=x["tp5"], tp7=x["tp7"], n_gt=x["n_gt"], n_pred=x["n_pred"])
        out.append(rec)
    return key, out


def pooled(recs, nm, k="tp5"):
    tp = sum(r[nm][k] for r in recs); den = sum(r[nm]["n_gt"] + r[nm]["n_pred"] for r in recs)
    return 2 * tp / max(den, 1)


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(8) as p:
        res = dict(p.map(job, [(k, vdir(v)) for k, v in VAR.items()]))
    print("== mesoSPIM c1–c5 pooled F1@0.5/F1@0.7（vol0.15）：固定 cp−2 | R1 | R2；括号内为各体积选出的阈值与 m(−2)/V")
    for key in sorted((k for k in res if not k.startswith("D3")), key=lambda k: (int(k.split("|")[1]), k)):
        R_ = res[key]
        print(f"  {key:8s} " + " | ".join(f"{nm} {pooled(R_, nm):.3f}/{pooled(R_, nm, 'tp7'):.3f}" for nm in ("fix", "r1", "r2"))
              + "  (" + " ".join(f"{r['id']}:{r['t_r1']:g}/{r['t_r2']:g}/{r['ratio_m'][-2.0]:.2f}" for r in R_) + ")")
    print("\n== D3 域内验证子集（12 套，每套至多 2 个体积）：逐数据集 pooled F1@0.5，再宏平均")
    for key in [k for k in res if k.startswith("D3")]:
        R_ = res[key]; dss = sorted({r["ds"] for r in R_})
        mac = {nm: np.mean([pooled([r for r in R_ if r["ds"] == d], nm) for d in dss]) for nm in ("fix", "r1", "r2")}
        mac7 = {nm: np.mean([pooled([r for r in R_ if r["ds"] == d], nm, "tp7") for d in dss]) for nm in ("fix", "r1", "r2")}
        print(f"  {key}: " + " | ".join(f"{nm} {mac[nm]:.4f}/{mac7[nm]:.4f}" for nm in ("fix", "r1", "r2")))
        for d in dss:
            rr = [r for r in R_ if r["ds"] == d]
            print(f"     {d:12s} m(−2)/V " + ",".join(f"{r['ratio_m'][-2.0]:.2f}" for r in rr) + "  选出 R1 " + ",".join(f"{r['t_r1']:g}" for r in rr)
                  + " R2 " + ",".join(f"{r['t_r2']:g}" for r in rr) + "  F1 " + " ".join(f"{nm} {pooled(rr, nm):.3f}" for nm in ("fix", "r1", "r2")))
    json.dump({k: v for k, v in res.items()}, open(f"{_NSP_SCRATCH}/r6/diag/sizeThr.json", "w"), indent=1, ensure_ascii=False)
    print(f"写出 {_NSP_SCRATCH}/r6/diag/sizeThr.json")
