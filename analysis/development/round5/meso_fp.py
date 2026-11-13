"""mesoSPIM 改用源数据体素后，只重跑 mesoSPIM 的对照评测（orx script 模式；清单须已改为 5×10.52×10.52 µm）。
对每个模型：完整流程（eval_val，与第四轮二维对照相同的参数：joint r2a2、7 个尺度、iso、iso_final，每套至多 4 个）在测试集与验证集上
只跑 mesospim（visual、c3）；gt 协议（eval_gt）在测试集与验证集上只跑 mesospim。结果分别写到 <out>/<名>_fp_<split>/ 与 <out>/<名>_gt_<split>/。
用法：python meso_fp.py --out <结果目录> --models orx_v5_2d_s0 ... base [--parts fp gt] [--splits test val]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse, json, os, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--parts", nargs="+", default=["fp", "gt"])
ap.add_argument("--splits", nargs="+", default=["test", "val"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ROOT = f"{_NSP_ROOT}"
FP = ["--thrs", "0", "-1", "-2", "--fracs", "0", "0.15", "0.25", "0.35", "--max_per_ds", "4", "--diam_mode", "joint",
      "--scales", "6", "10", "17", "30", "52", "90", "150", "--joint_rule", "r2a2", "--iso", "--iso_final", "--only", "mesospim"]
res = {}
for m0 in a.models:
    nm = os.path.basename(m0.rstrip("/"))
    # eval_val 要的是权重路径（给名字时 Cellpose 会去网上下载）：base -> 原版 cpsam_v2，其余在共享盘 models/models/ 下
    m = m0 if os.path.isabs(m0) else (f"{ROOT}/models/base/cpsam_v2" if m0 == "base" else f"{ROOT}/models/models/{m0}")
    assert os.path.exists(m), m
    for sp in a.splits:
        if "fp" in a.parts:
            pre = os.path.relpath(f"{a.out}/{nm}_fp", f"{ROOT}/results")           # eval_val 写到 results/<prefix>_<split>
            subprocess.run([sys.executable, "eval_val.py", "--model", m, "--split", sp, "--prefix", pre] + FP, check=True)
            g = json.load(open(f"{ROOT}/results/{pre}_{sp}/grid.json"))["cp-2_vol0.15"]
            rows = [r for r in json.load(open(f"{ROOT}/results/{pre}_{sp}/rows.json")) if r.get("thr") == -2.0 and r.get("frac") == 0.15]
            res[f"{nm} fp {sp}"] = dict(f1=[round(r["f1"], 4) for r in rows], f1_7=[round(r.get("f1_7", float("nan")), 4) for r in rows],
                                        d_used=[r.get("d_used") for r in rows], ids=[r["id"] for r in rows])
            print(f"MESOFP {nm} fp {sp} {res[f'{nm} fp {sp}']}", flush=True)
        if "gt" in a.parts:
            o = f"{a.out}/{nm}_gt_{sp}"
            subprocess.run([sys.executable, "eval_gt.py", "--model", m, "--out", o, "--split", sp, "--only", "mesospim", "--workers", "1"], check=True)
            c = json.load(open(f"{o}/summary.json"))["cells"]["cp-2_vol0.15"]["per_ds"]["mesospim"]
            res[f"{nm} gt {sp}"] = [round(c["f1_05"], 4), round(c["f1_07"], 4)]
            print(f"MESOFP {nm} gt {sp} {res[f'{nm} gt {sp}']}", flush=True)
json.dump(res, open(f"{a.out}/meso_fp.json", "w"), indent=1)
print("FINAL_METRICS " + json.dumps(res), flush=True)
