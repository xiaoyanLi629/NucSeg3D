"""nc G 系列的 orx script 入口：同 r6/thr_eval.py（阈值网格 0、−1、−2、−3、−4、−6 + --save_masks，再用 sizeThr_dir.py 做规则 T），
只是调用 gx_eval.py（eval_gt 加 --gx_* 开关）而不是 eval_gt.py。
每遍结束后另存 ruleT.json（逐体积 m(t)/V 与规则 T 选出的阈值，与 sizeThr_dir.py 的 R2 逐字同规则），
加 --gx_drop_masks 时随后删掉 masks/（G4 的 npz 只为规则 T 服务，删掉省共享盘）。
另加 --gx_noise_list c1 c2 ...：对每个噪声档各跑一遍，结果放 <out>/noise_<c>/（G4 一个作业跑完同一模型、同一 z 降采样倍数的全部噪声档）。
用法：python gx_thr_eval.py --out <结果目录> --model <模型> [eval_gt 参数] [--gx_* 开关] [--gx_noise_list 0 0.25 0.5]"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np


def rule_t(d):
    """与 sizeThr_dir.run 相同的 R2 规则，结果写 <d>/ruleT.json。"""
    rows = json.load(open(d + "/rows.json"))
    by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True)
    recs = []
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        cp = np.load(f)["cp"].astype(np.float32)
        r0 = by[(ds, vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
        t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
        recs.append(dict(ds=ds, id=vid, m={f"{t:g}": v for t, v in m.items()}, t=t2))
    json.dump(recs, open(d + "/ruleT.json", "w"), indent=0)
    return len(recs)

args = sys.argv[1:]
i = args.index("--out"); out = args[i + 1]; rest = args[:i] + args[i + 2:]
drop = "--gx_drop_masks" in rest
rest = [x for x in rest if x != "--gx_drop_masks"]
levels = None
if "--gx_noise_list" in rest:
    j = rest.index("--gx_noise_list"); k = j + 1
    while k < len(rest) and not rest[k].startswith("--"):
        k += 1
    levels = rest[j + 1:k]; rest = rest[:j] + rest[k:]
    assert levels and "--gx_noise" not in rest, (levels, rest)
for lv in (levels or [None]):
    o = out if lv is None else os.path.join(out, f"noise_{float(lv):g}")
    extra = [] if lv is None else ["--gx_noise", lv]
    cmd = [sys.executable, "gx_eval.py", "--out", o, "--thrs", "0", "-1", "-2", "-3", "-4", "-6", "--save_masks"] + rest + extra
    print("RUN", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "sizeThr_dir.py", o], check=True)
    print("ruleT.json", rule_t(o), "个体积", flush=True)
    if drop:
        shutil.rmtree(os.path.join(o, "masks"))
