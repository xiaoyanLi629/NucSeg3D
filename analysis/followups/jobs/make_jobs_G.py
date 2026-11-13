"""生成 nc G 系列作业草稿 $NUCSEG3D_SCRATCH/nc/gpu/jobs_G.json（排队器格式；不并入 r6/jobs.json，由主代理登记后合并）。
设计见同目录 DESIGN.md。全部作业挂在闸门 $NUCSEG3D_SHARED/nc_gpu/GO 之后（G_stage 等它，其余都依赖 G_stage）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json

BASE = "fcb392eb-d4b8-4954-be38-29734741a9d1"      # L 系列基础节点（R7-T 代码 + 排除清单补丁，eval_gt/infer3d 与 R7-T 相同）
GD = f"{_NSP_SCRATCH}/nc/gpu"
SH = f"{_NSP_SHARED}/nc_gpu"
PY = f"{_NSP_SHARED}/env_torch/bin/python"
MD = f"{_NSP_ROOT}/models/models"
GPUS = [0, 1, "hostB:3", "hostB:2"]
DESC = f"nc G 系列（投稿分析方案 4.2/4.4 节，设计见 {_NSP_SCRATCH}/nc/gpu/DESIGN.md）；只报告，不参与任何选择。"
VOX = ["--vox_override", "mesospim=5/10.52/10.52"]
S1 = lambda s: ["--model", f"{MD}/orx_t3d_r5s1_s{s}", "--workers", "2", "--box", "384", "--views", "3", "--views1_ds", "cellos", "parhyale", "mesospim"]
D2 = lambda s: ["--model", f"orx_v5_2d_s{s}", "--workers", "2", "--box", "384"]
BS = ["--model", "base", "--workers", "2", "--box", "384"]
GX = [f"{GD}/gx_eval.py", f"{GD}/gx_thr_eval.py", f"{_NSP_SCRATCH}/r6/diag/sizeThr_dir.py"]
J = []


def new(key, title, args, mem, deps=("G_stage",), script="gx_thr_eval.py", files=GX, gpus=GPUS, block=False):
    J.append(dict(key=key, kind="new", parent=BASE, title=title, desc=DESC, gpus=list(gpus), mem=mem, block=block, deps=list(deps),
                  notify=True, copy_thr=False, copy_files=list(files), config=dict(script=script, args=list(args), share_gpu=True)))


J.append(dict(key="G_stage", kind="cmd", deps=[], wait_file=f"{SH}/GO",
              cmd=f"mkdir -p {SH} && cd {GD}/lists && cp -rn g4_keys.txt g6_test_keys.txt g4_diam md5sums.txt {SH}/ && cd {SH} && md5sum -c md5sums.txt"))

# ---- G5：同一张卡、独占、依次跑三个模型（最先排，block 挡住这张卡）
g5 = [("s1", f"{MD}/orx_t3d_r5s1_s0"), ("d2", "orx_v5_2d_s0"), ("base", "base")]
prev = "G_stage"
for tag, m in g5:
    new(f"G5_{tag}", f"nc G5：推理时间与峰值显存（hostB GPU3 独占）：{tag}", ["--models", f"{tag}={m}"], 30000, deps=[prev],
        script="g5_bench.py", files=[f"{GD}/g5_bench.py"], gpus=["hostB:3"], block=True)
    prev = f"G5_{tag}"

# ---- G6：各向同性跟流开关（S1 种子 0；开的结果复用 a69ee4fa / 07992b55）
new("G6v_off", "nc G6：验证集 gt 协议，关掉各向同性跟流：S1 种子 0", S1(0) + VOX + ["--gx_iso", "off"], 14300)
new("G6t_off", "nc G6：测试集 gt 协议（z/xy≥2 的 144 个体积），关掉各向同性跟流：S1 种子 0（只报告）",
    S1(0) + VOX + ["--split", "test", "--gx_iso", "off", "--gx_min_ratio", "2"], 14300)

# ---- A4：三维残差开关
new("A4", "nc A4：三维残差开关的流场差（测试集每套 2 个体积）：S1 种子 0", ["--model", f"{MD}/orx_t3d_r5s1_s0", "--split", "test", "--per_ds", "2"],
    9000, script="a4_resid.py", files=[f"{GD}/a4_resid.py"])

# ---- G4：z 降采样 × 噪声（每个作业 = 一个模型实例 × 一个倍数，内部跑三档噪声）
G4A = ["--split", "test", "--only", "bbbc050", "celegans", "nis3d", "nucverse3d", "purdue", "tapenade",
       "--diam_from", f"{SH}/g4_diam", "--gx_keys", f"{SH}/g4_keys.txt", "--gx_noise_list", "0", "0.25", "0.5", "--gx_drop_masks"]
for k in (1, 2, 4, 8):
    for s in (0, 1000, 2000):
        new(f"G4_S1_s{s}_z{k}", f"nc G4：z 降采样 {k} 倍 × 噪声 0/0.25/0.5：S1 种子 {s}", S1(s) + VOX + G4A + ["--gx_zdec", str(k)], 14300)
        new(f"G4_2D_s{s}_z{k}", f"nc G4：z 降采样 {k} 倍 × 噪声 0/0.25/0.5：二维种子 {s}", D2(s) + VOX + G4A + ["--gx_zdec", str(k)], 18000)
    new(f"G4_base_z{k}", f"nc G4：z 降采样 {k} 倍 × 噪声 0/0.25/0.5：原版", BS + VOX + G4A + ["--gx_zdec", str(k)], 18000)

# ---- G4 附：mesoSPIM 旧体素 0.16（按真值核形状估的口径）的 S1 敏感性分析（gt 协议，只 mesospim 测试体积）
for s in (0, 1000, 2000):
    new(f"G4m_S1_s{s}", f"nc G4 附：mesoSPIM 旧体素 z/xy 0.16 的敏感性分析：S1 种子 {s}",
        S1(s) + ["--split", "test", "--only", "mesospim", "--vox_override", "mesospim=0.16/1/1"], 14300)

# ---- A3：CPU，后台跑（不挡排队器）；读已有 tGT 运行的 masks
J.append(dict(key="A3", kind="cmd", deps=["G_stage"],
              cmd=f"mkdir -p {SH}/A3 && cd {GD} && (nohup bash -c 'CUDA_VISIBLE_DEVICES= {PY} a3_calib.py --out {SH}/A3 --src {GD}/wt/src --per_ds 2 "
                  f"&& CUDA_VISIBLE_DEVICES= {PY} a3_calib.py --out {SH}/A3 --src {GD}/wt/src --per_ds 0' > {SH}/A3/a3.log 2>&1 &) && echo A3 已在后台启动"))

keys = [j["key"] for j in J]
assert len(keys) == len(set(keys)) and len({j["title"] for j in J if j["kind"] == "new"}) == sum(j["kind"] == "new" for j in J)
json.dump(J, open(f"{GD}/jobs_G.json", "w"), ensure_ascii=False, indent=1)
print(len(J), "个作业，其中占 GPU", sum(j["kind"] == "new" for j in J))
