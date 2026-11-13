"""生成 M1 消融作业草稿 jobs_M1.json（审稿意见 M1，投稿前事后补加）：S1 五个种子把三维解码器输出层置零（--gx_zero_dec），
其余参数与正式测评逐项相同（复制 r6/jobs.json 里 tGT_S1_*/tFP_S1_*/S5_tGT_S1_*/S5_tFP_S1_* 的 args；验证集参照复制 a69ee4fa 的 args），只报告。
闸门 $NUCSEG3D_SHARED/nc_gpu/GO_M1。设计见 DESIGN_M1.md。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
BASE = "fcb392eb-d4b8-4954-be38-29734741a9d1"
GD = f"{_NSP_SCRATCH}/nc/gpu"
GPUS = [0, 1, "hostB:0", "hostB:1", "hostB:2", "hostB:3", "hostC:0", "hostC:1", "hostC:2"]
DESC = f"nc M1 消融（审稿意见 M1，投稿前事后补加；设计见 {_NSP_SCRATCH}/nc/gpu/DESIGN_M1.md）：三维解码器输出层置零；只报告，不参与任何选择。"
J = {j["key"]: j for j in json.load(open(f"{_NSP_SCRATCH}/r6/jobs.json"))}
FILES = [f"{GD}/gx_eval.py", f"{GD}/gx_thr_eval.py", f"{_NSP_SCRATCH}/r6/diag/sizeThr_dir.py"]
out = [dict(key="M1_stage", kind="cmd", deps=[], wait_file=f"{_NSP_SHARED}/nc_gpu/GO_M1", cmd=f"ls {_NSP_ROOT}/models/models/orx_t3d_r5s1_s4000 >/dev/null")]
def add(key, title, args):
    out.append(dict(key=key, kind="new", parent=BASE, title=title, desc=DESC, gpus=GPUS, mem=14300, block=False, deps=["M1_stage"], notify=True,
                    copy_thr=False, copy_files=FILES, config=dict(script="gx_thr_eval.py", args=list(args) + ["--gx_zero_dec"], share_gpu=True)))
for s in ("s0", "s1000", "s2000", "s3000", "s4000"):
    pre = "S5_" if s in ("s3000", "s4000") else ""
    for prot, nm in (("tGT", "测试集 gt 协议 227 个体积"), ("tFP", "测试集完整流程 45 个体积")):
        src = J[f"{pre}{prot}_S1_{s}"]["config"]["args"]
        assert f"orx_t3d_r5s1_{s}" in " ".join(src), (s, src)
        add(f"M1_{prot}_{s}", f"nc M1 消融：S1 种子 {s[1:]} 三维解码器置零，{nm}（事后，只报告）", src)
add("M1_val_s0", "nc M1 消融：S1 种子 0 三维解码器置零，验证集 gt 协议 88 个体积（参照，事后，只报告）",
    ["--model", f"{_NSP_ROOT}/models/models/orx_t3d_r5s1_s0", "--workers", "2", "--box", "384", "--views", "3", "--views1_ds", "cellos", "parhyale", "mesospim"])
json.dump(out, open(f"{GD}/jobs_M1.json", "w"), ensure_ascii=False, indent=1)
for j in out[1:]: print(j["key"], " ".join(j["config"]["args"])[-170:])
