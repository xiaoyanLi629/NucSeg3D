"""L3：用 S1 自己上一遍的预测估直径（不靠二维模型），写成 eval_gt --diam_from 能读的 rows.json。
输入：一次带 --save_masks 的 eval_gt / thr_eval 结果目录（masks/<ds>__<id>.npz 里有 cellprob −2 的预测 pred（未删小块）、vox、
上一遍用的直径 d）。对每个体积：先按上一遍的直径删掉体积 < 0.15 × π/6·d³·vxy/vz 的预测（与主流程的体积下限同式同比例），
再取剩下预测核的等效直径中位数（与真值直径同一个函数 eq_diam_xy_px），夹到 [4, 160] 像素；一个核都不剩时沿用上一遍的直径
（记为 fallback）。真值只读来写诊断列 d_gt，不参与估计。
用法：python l3_diam.py --src <结果目录> | --state_key <排队器任务 key>  --out <输出目录>
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse, glob, json, os
import numpy as np

RES = f"{_NSP_ROOT}/results"
FRAC, MIN_OBJ, LO, HI = 0.15, 1, 4.0, 160.0          # 事先写定，不在任何数据上调


def eq_diam_xy_px(lab, vox):                           # 与 src/eval_cpsam.py 的同名函数逐字相同
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    d_um = np.median((6 * cnt * np.prod(vox) / np.pi) ** (1 / 3))
    return float(d_um / vox[1])


ap = argparse.ArgumentParser()
ap.add_argument("--src"); ap.add_argument("--state_key"); ap.add_argument("--out", required=True)
a = ap.parse_args()
src = a.src
if a.state_key:
    run = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))[a.state_key]["run"]
    c = [d for d in glob.glob(f"{RES}/orx_{run}_*") if os.path.isdir(d + "/masks")]
    assert len(c) == 1, c
    src = c[0]
fs = sorted(glob.glob(src + "/masks/*.npz"))
assert fs, f"{src} 没有 masks/"
rows = []
for f in fs:
    ds, vid = os.path.basename(f)[:-4].split("__", 1)
    z = np.load(f)
    pred, vox, d = z["pred"], tuple(float(v) for v in z["vox"]), float(z["d"])
    cnt = np.bincount(pred.ravel())
    exp_vol = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    keep = np.where(cnt >= FRAC * exp_vol)[0]; keep = keep[keep > 0]
    n = len(keep)
    if n >= MIN_OBJ:
        p = np.where(np.isin(pred, keep), pred, 0)
        d_new, fb = float(np.clip(eq_diam_xy_px(p, vox), LO, HI)), False
    else:
        d_new, fb = d, True
    rows.append(dict(ds=ds, id=vid, d_used=d_new, d_prev=d, n_obj=int(n), fallback=fb, d_gt=eq_diam_xy_px(z["gt"], vox)))
os.makedirs(a.out, exist_ok=True)
assert not os.path.exists(a.out + "/rows.json"), "输出已存在，不覆盖"
json.dump(rows, open(a.out + "/rows.json", "w"), indent=0)
json.dump(dict(src=src, n=len(rows), fallback=sum(r["fallback"] for r in rows), frac=FRAC, min_obj=MIN_OBJ, clip=[LO, HI]),
          open(a.out + "/meta.json", "w"), indent=1)
e = np.array([abs(np.log(r["d_used"] / r["d_gt"])) for r in rows if r["d_gt"] > 0])
print(f"L3_DIAM_DONE {a.out}：{len(rows)} 个体积，fallback {sum(r['fallback'] for r in rows)}，|log(d/d_gt)| 中位 {np.median(e):.3f}（诊断用）")
