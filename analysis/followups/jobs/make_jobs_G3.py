"""生成 nc G 系列第三批作业草稿 $NUCSEG3D_SCRATCH/nc/gpu/jobs_G3.json（G1 补调参、G3 u-Segment3D；不并入 r6/jobs.json）。设计见 DESIGN_G3.md。全部挂在闸门 GO3 之后。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json

BASE = "fcb392eb-d4b8-4954-be38-29734741a9d1"
GD = f"{_NSP_SCRATCH}/nc/gpu"
SH = f"{_NSP_SHARED}/nc_gpu"
PY = f"{_NSP_SHARED}/env_torch/bin/python"
GPUS = [0, 1, "hostB:3", "hostB:2"]
RES = f"{_NSP_ROOT}/results"
DESC = f"nc G 系列第三批（设计见 {_NSP_SCRATCH}/nc/gpu/DESIGN_G3.md）；只在验证集上选，测试集只报告。"
FR = ["0", "0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75"]
MU = dict(tapenade=["0.5", "0.7", "1", "1.4", "2"], anystar=["0.25", "0.35", "0.5", "0.7", "1"], cellos=["0.7", "1", "1.4", "2", "2.8"])
J = []


def new(key, title, script, args, mem, deps, files, gpus=GPUS):
    J.append(dict(key=key, kind="new", parent=BASE, title=title, desc=DESC, gpus=list(gpus), mem=mem, block=False, deps=list(deps), notify=True,
                  copy_thr=False, copy_files=[f"{GD}/{x}" for x in files], config=dict(script=script, args=list(args), share_gpu=True)))


J.append(dict(key="G3b_stage", kind="cmd", deps=[], wait_file=f"{SH}/GO3", cmd=f"mkdir -p {SH}/regrid {SH}/G3 && ls {SH}/regrid {SH}/G3 >/dev/null"))
F1 = ["bench_others2.py", "tf_wrap.py"]
for m in ("tapenade", "anystar", "cellos"):
    new(f"GRv_{m}", f"nc G1 补调参：{m} 验证集 36 个体积，放宽倍数与体积下限网格", "tf_wrap.py",
        ["--model", m, "--pub", f"{SH}/regrid/{m}", "--stage", "val", "--mults", *MU[m], "--fracs", *FR], 12000, ["G3b_stage"], F1)
    new(f"GRt_{m}", f"nc G1 补调参：{m} 测试集 45 个体积按选中参数跑一次（只报告；与旧参数相同则保留旧结果）", "tf_wrap.py",
        ["--model", m, "--pub", f"{SH}/regrid/{m}", "--stage", "test"], 12000, [f"GRv_{m}"], F1)
J.append(dict(key="GR_table", kind="cmd", deps=[f"GRt_{m}" for m in ("tapenade", "anystar", "cellos")],
              cmd=f"cd {GD}/g1 && (nohup env CUDA_VISIBLE_DEVICES= {PY} {GD}/g1_table2.py {SH}/regrid > {GD}/g1/g1_table2.log 2>&1 &) && echo GR_table 已在后台启动"))
F3 = ["useg_eval.py", "useg_cons.py"]
VOX = ["--workers", "2", "--box", "384", "--vox_override", "mesospim=5/10.52/10.52"]
M = dict(base="base", d2="orx_v5_2d_s0"); RD = dict(base="orx_r5merged_base_test", d2="orx_r5merged_2d_s0_test")
state = f"import json;S=json.load(open('{_NSP_SCRATCH}/r6/sched_state.json'));print('%s='+'%s/orx_'+S['%s']['run']+'_useg_eval')"


def spec(k, key):
    return "$(python3 -c \"" + state % (k, RES, key) + "\")"


for k, name in M.items():
    new(f"G3v1_{k}", f"nc G3：u-Segment3D 验证集 gt 协议 88 个体积（v0–v6）：{k}", "useg_eval.py", ["--model", name, *VOX], 18000, ["G3b_stage"], F3)
J.append(dict(key="G3_sel1", kind="cmd", deps=["G3v1_base", "G3v1_d2"],
              cmd=f"{PY} {GD}/useg_select.py --stage 1 --out {SH}/G3/sel1.json {spec('base', 'G3v1_base')} {spec('d2', 'G3v1_d2')}"))
for k, name in M.items():
    new(f"G3v2_{k}", f"nc G3：u-Segment3D 验证集，第一轮最好的一组加流一致性过滤：{k}", "useg_eval.py",
        ["--model", name, *VOX, "--useg_sel", f"{SH}/G3/sel1.json", "--useg_key", k, "--useg_field", "stage1"], 18000, ["G3_sel1"], F3)
def spec2(k):
    a = f"import json;S=json.load(open('{_NSP_SCRATCH}/r6/sched_state.json'));print('%s='+','.join('%s/orx_'+S[q]['run']+'_useg_eval' for q in ('G3v1_%s','G3v2_%s')))" % (k, RES, k, k)
    return '$(python3 -c "' + a + '")'
J.append(dict(key="G3_sel2", kind="cmd", deps=["G3v2_base", "G3v2_d2"], cmd=f"{PY} {GD}/useg_select.py --stage 2 --out {SH}/G3/selected.json {spec2('base')} {spec2('d2')}"))
for k, name in M.items():
    S = ["--useg_sel", f"{SH}/G3/selected.json", "--useg_key", k, "--save_masks"]
    new(f"G3tg_{k}", f"nc G3：u-Segment3D 测试集 gt 协议 227 个体积（只报告）：{k}", "useg_eval.py", ["--model", name, *VOX, "--split", "test", *S], 18000, ["G3_sel2"], F3)
    new(f"G3tf_{k}", f"nc G3：u-Segment3D 测试集完整流程 45 个体积（只报告）：{k}", "useg_eval.py",
        ["--model", name, *VOX, "--split", "test", "--max_per_ds", "4", "--diam_from", f"{RES}/{RD[k]}", *S], 18000, ["G3_sel2"], F3)
keys = [j["key"] for j in J]
assert len(keys) == len(set(keys)) and len({j["title"] for j in J if j["kind"] == "new"}) == sum(j["kind"] == "new" for j in J)
json.dump(J, open(f"{GD}/jobs_G3.json", "w"), ensure_ascii=False, indent=1)
print(len(J), "个作业，其中占 GPU", sum(j["kind"] == "new" for j in J))
