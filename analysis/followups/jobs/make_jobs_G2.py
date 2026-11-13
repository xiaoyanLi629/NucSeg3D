"""生成 nc G 系列第二批作业草稿 $NUCSEG3D_SCRATCH/nc/gpu/jobs_G2.json（G7、A5；不并入 r6/jobs.json）。设计见 DESIGN_G2.md。
全部挂在闸门 $NUCSEG3D_SHARED/nc_gpu/GO2 之后。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json

BASE = "fcb392eb-d4b8-4954-be38-29734741a9d1"
GD = f"{_NSP_SCRATCH}/nc/gpu"
SH = f"{_NSP_SHARED}/nc_gpu"
PY = f"{_NSP_SHARED}/env_torch/bin/python"
MD = f"{_NSP_ROOT}/models/models"
GPUS = [0, 1, "hostB:3", "hostB:2"]
DESC = f"nc G 系列第二批（投稿分析方案 4.2/4.4 节，设计见 {_NSP_SCRATCH}/nc/gpu/DESIGN_G2.md）；只报告，不参与任何选择。"
J = []


def new(key, title, script, args, mem, deps):
    J.append(dict(key=key, kind="new", parent=BASE, title=title, desc=DESC, gpus=GPUS, mem=mem, block=False, deps=deps, notify=True,
                  copy_thr=False, copy_files=[f"{GD}/{script}"], config=dict(script=script, args=args, share_gpu=True)))


J.append(dict(key="G2b_stage", kind="cmd", deps=[], wait_file=f"{SH}/GO2",
              cmd=f"mkdir -p {SH} && cd {GD}/lists && cp -n iso30_index.json md5sums_G2.txt {SH}/ && cd {SH} && md5sum -c md5sums_G2.txt "
                  f"&& md5sum -c md5sums.txt"))
new("G7", "nc G7：PDAC 类器官三层分析与荧光定量，S1 + 各向同性跟流 + 规则 T（对照二维、原版、旧直径 31.25）", "g7_pdac.py", [], 20000, ["G2b_stage"])
new("A5_feat", "nc A5：S1 编码器特征（训练 910 + 验证 88 + 测试 227 个体积，中间 8 层）", "a5_feat.py", ["--model", f"{MD}/orx_t3d_r5s1_s0"], 8000,
    ["G2b_stage"])
J.append(dict(key="A5_dist", kind="cmd", deps=["A5_feat"],
              cmd=(f"mkdir -p {SH}/A5 && R=$(python3 -c \"import json;print(json.load(open('{_NSP_SCRATCH}/r6/sched_state.json'))['A5_feat']['run'])\") "
                   f"&& F={_NSP_ROOT}/results/orx_${{R}}_a5_feat/feats.npz && test -f $F && cd {GD}/wt/src "
                   f"&& (nohup env CUDA_VISIBLE_DEVICES= {PY} {GD}/a5_dist.py --feats $F --out {SH}/A5 > {SH}/A5/a5_dist.log 2>&1 &) && echo A5_dist 已在后台启动 $F")))
keys = [j["key"] for j in J]
assert len(keys) == len(set(keys))
json.dump(J, open(f"{GD}/jobs_G2.json", "w"), ensure_ascii=False, indent=1)
print(len(J), "个作业，其中占 GPU", sum(j["kind"] == "new" for j in J))
