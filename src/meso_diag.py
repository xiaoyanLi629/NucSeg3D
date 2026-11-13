"""round 5：mesoSPIM 诊断（orx script 模式；不用于选择，只用于找原因）。

在 mesoSPIM 的 c1–c5（c3 是验证体积，其余四个按设计从不参与训练；测试体积 visual 不碰）上，逐个“模型 + 评测选项”
跑 eval_gt（gt 协议、单视图，cellprob 阈值扩到 −6，存预测与网络输出），再跑 shape_diag 做逐核形状分析。
用法：python meso_diag.py --out <结果目录> --variants 名=模型[:选项] ...   选项用逗号分隔，如 a1_s1000_xz=orx_t3d_a1_s1000:--view_perm,102
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--variants", nargs="+", required=True)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--thrs", default="0,-1,-2,-3,-4,-6")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
sj = os.path.join(here, "..", "mesospim_dev.json")
M = f"{_NSP_ROOT}/models/models"
res = {}
for v in a.variants:
    name, spec = v.split("=", 1)
    model, _, extra = spec.partition(":")
    model = model if (os.path.isabs(model) or not model.startswith("orx_t3d")) else f"{M}/{model}"
    o = f"{a.out}/{name}"
    cmd = [sys.executable, "eval_gt.py", "--model", model, "--out", o, "--split_json", sj, "--only", "mesospim", "--workers", str(a.workers),
           "--save_masks", "--thrs", *a.thrs.split(",")] + ([x for x in extra.split(",") if x] if extra else [])
    print("RUN", name, " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "shape_diag.py", o], check=True)
    S = json.load(open(f"{o}/summary.json"))["cells"]
    res[name] = {k: [round(S[k]["per_ds"]["mesospim"]["f1_05"], 4), round(S[k]["per_ds"]["mesospim"]["f1_07"], 4)] for k in S if k.endswith("_vol0.15")}
    print(f"MESO {name:22s} " + "  ".join(f"{k.split('_')[0]} {x[0]:.3f}/{x[1]:.3f}" for k, x in res[name].items()), flush=True)
json.dump(res, open(f"{a.out}/meso_diag.json", "w"), indent=1)
print("FINAL_METRICS " + json.dumps({k: v.get("cp-2_vol0.15") for k, v in res.items()}), flush=True)
