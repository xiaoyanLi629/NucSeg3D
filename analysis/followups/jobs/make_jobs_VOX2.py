"""VOX2（第三轮审稿 MF2；投稿前事后补加，只推理、只报告）。设计见 DESIGN_VOX2.md。
A：二维种子 1000/2000/3000/4000 在 BBBC050 与 NISNet3D zebrafish 测试集 60 个体积上用源体素重跑——逐项复制 VOX_2D_s0，只换模型。
B：NIS3D 4 个（Info.txt 写 1×1×1 µm，原用按核形状估的 z）与 Purdue 3 个人体光谱体积（Chen 等 2022：0.54×0.54×1.04 µm，z/xy 1.93，原按各向同性）
   用源体素重跑：S1 五种子、二维五种子、原版；参数复制正式 tGT_* 作业，只加 --only/--ids 与体素覆盖。
GPU 只用 hostA 的 [0, 1]；闸门 $NUCSEG3D_SHARED/nc_gpu/GO_VOX2。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import copy, json
J = {j["key"]: j for j in json.load(open(f"{_NSP_SCRATCH}/r6/jobs.json"))}
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]
DESC = f"nc VOX2 源体素敏感性补充（第三轮审稿 MF2，投稿前事后补加；设计见 {_NSP_SCRATCH}/nc/gpu/DESIGN_VOX2.md）；只报告，不参与任何选择。"
IDS = ["Drosophila_1_xR25", "Drosophila_2_xR25", "MusMusculus_1_xR25", "MusMusculus_2_xR25",
       "Diabetic_Biopsy_Human_Spectral_1-5800-2500", "Diabetic_Biopsy_Human_Spectral_3-3480_1070", "Kidney_Cortex_Human_Spectral_1-1317-1381"]
OVB = ["--vox_override", "mesospim=5/10.52/10.52", "nis3d=1/1/1", "purdue=1.93/1/1"]
out = [dict(key="VOX2_stage", kind="cmd", deps=[], wait_file=f"{_NSP_SHARED}/nc_gpu/GO_VOX2", cmd="true")]
for s in SEEDS[1:]:
    j = copy.deepcopy(J["VOX_2D_s0"]); a = j["config"]["args"]; a[a.index("--model") + 1] = f"orx_v5_2d_{s}"
    j.update(key=f"VOX2_2D_{s}", title=f"nc VOX2：二维种子 {s[1:]}，BBBC050 与 NISNet3D zebrafish 测试集用源数据体素（事后，只报告）", desc=DESC, deps=["VOX2_stage"])
    out.append(j)
src = [(("S5_" if s in ("s3000", "s4000") else "") + f"tGT_S1_{s}", f"S1 种子 {s[1:]}") for s in SEEDS] + \
      [(("S5_" if s in ("s3000", "s4000") else "") + f"tGT_2D_{s}", f"二维种子 {s[1:]}") for s in SEEDS] + [("tGT_base", "原版")]
for k, lab in src:
    a = list(J[k]["config"]["args"]); i = a.index("--vox_override"); del a[i:i + 2]
    a += ["--only", "nis3d", "purdue", "--ids", *IDS] + OVB
    key = "VOX2N_" + k.replace("S5_", "").replace("tGT_", "")
    out.append(dict(key=key, kind="new", parent="fcb392eb-d4b8-4954-be38-29734741a9d1",
                    title=f"nc VOX2：{lab}，NIS3D 4 个与 Purdue 3 个人体体积用源数据体素（事后，只报告）", desc=DESC, gpus=[0, 1],
                    mem=14300 if "S1" in k else 10000, block=False, deps=["VOX2_stage"], notify=True, copy_thr=True, copy_files=[],
                    config=dict(script="thr_eval.py", args=a, share_gpu=True)))
json.dump(out, open(f"{_NSP_SCRATCH}/nc/gpu/jobs_VOX2.json", "w"), ensure_ascii=False, indent=1)
for j in out[1:]: print(j["key"], j["mem"], " ".join(j["config"]["args"])[:60], "...", " ".join(j["config"]["args"])[-120:])
