"""源体素敏感性（第二轮审稿 M6；投稿前事后补加，只推理、只报告）：BBBC050 与 NISNet3D zebrafish 测试集 gt 协议全部体积（44 + 16 = 60 个），
用源数据 z/xy 重新推理（--vox_override bbbc050=2.5/1/1 nisnet_zf=3.97/1/1；xy 仍记为 1 像素，源值见 AGENTS.md 2026-11-08 体素来源条）。
模型：S1 五种子（视图规则，两套都是三视图）+ 规则 T、二维种子 0 + 规则 T、原版（固定 cp−2，同一运行里也存了网格）。
其余参数与正式 tGT_* 作业逐项相同（复制 r6/jobs.json 的 args，只加 --only 与新的体素覆盖）；各向同性跟流按 eval_gt 原规则（新 z/xy 都 ≥ 2，两套都走插值）。
GPU 只用 hostA 的 [0, 1]。闸门 $NUCSEG3D_SHARED/nc_gpu/GO_VOX。设计见 DESIGN_VOX.md。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
BASE = "fcb392eb-d4b8-4954-be38-29734741a9d1"
J = {j["key"]: j for j in json.load(open(f"{_NSP_SCRATCH}/r6/jobs.json"))}
OV = ["--vox_override", "mesospim=5/10.52/10.52", "bbbc050=2.5/1/1", "nisnet_zf=3.97/1/1"]
DESC = f"nc VOX 源体素敏感性（第二轮审稿 M6，投稿前事后补加；设计见 {_NSP_SCRATCH}/nc/gpu/DESIGN_VOX.md）；只报告，不参与任何选择。"
out = [dict(key="VOX_stage", kind="cmd", deps=[], wait_file=f"{_NSP_SHARED}/nc_gpu/GO_VOX", cmd="true")]
src = [(f"tGT_S1_{s}", f"S1 种子 {s[1:]}") for s in ("s0", "s1000", "s2000")] + [(f"S5_tGT_S1_{s}", f"S1 种子 {s[1:]}") for s in ("s3000", "s4000")] + [("tGT_2D_s0", "二维种子 0"), ("tGT_base", "原版")]
for k, lab in src:
    a = list(J[k]["config"]["args"]); i = a.index("--vox_override"); del a[i:i + 2]
    a += ["--only", "bbbc050", "nisnet_zf"] + OV
    key = "VOX_" + k.replace("S5_", "").replace("tGT_", "")
    out.append(dict(key=key, kind="new", parent=BASE, title=f"nc VOX：{lab}，BBBC050 与 NISNet3D zebrafish 测试集用源数据体素（事后，只报告）", desc=DESC,
                    gpus=[0, 1], mem=14300 if "S1" in k else 10000, block=False, deps=["VOX_stage"], notify=True, copy_thr=True, copy_files=[],
                    config=dict(script="thr_eval.py", args=a, share_gpu=True)))
json.dump(out, open(f"{_NSP_SCRATCH}/nc/gpu/jobs_VOX.json", "w"), ensure_ascii=False, indent=1)
for j in out[1:]: print(j["key"], j["mem"], " ".join(j["config"]["args"]))
