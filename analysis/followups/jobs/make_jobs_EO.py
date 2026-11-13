"""EO 对照（第三轮审稿 SF2；投稿前事后补加，只报告）：三维阶段训练与 S1 完全相同（数据、步数、学习率日程、种子、A1 → C/X1 → 权重平均），
只是三维解码器输出层恒为零并冻结（train3d.py 新增 --zero_dec，补丁文件 eo/train3d.py 由 copy_files 覆盖进节点 src/），
得到“只再训练编码器与二维头”的模型 EO；评测与 S1 完全相同（验证集 gt、测试集 gt 与完整流程，视图规则，规则 T 离线）。
种子 0/1000/2000/3000/4000；GPU 只用 hostA 的 [0, 1]；闸门 $NUCSEG3D_SHARED/nc_gpu/GO_EO。设计见 DESIGN_EO.md。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import copy, json
GD = f"{_NSP_SCRATCH}/nc/gpu"; MD = f"{_NSP_ROOT}/models/models"
BASE = "fcb392eb-d4b8-4954-be38-29734741a9d1"
J = {j["key"]: j for j in json.load(open(f"{_NSP_SCRATCH}/r6/jobs.json"))}
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]
DESC = f"nc EO 对照（第三轮审稿 SF2，投稿前事后补加；设计见 {_NSP_SCRATCH}/nc/gpu/DESIGN_EO.md）：解码器恒零、只训编码器；只报告。"
VAL = ["--workers", "2", "--box", "384", "--views", "3", "--views1_ds", "cellos", "parhyale", "mesospim", "--vox_override", "mesospim=5/10.52/10.52"]
out = [dict(key="EO_stage", kind="cmd", deps=[], wait_file=f"{_NSP_SHARED}/nc_gpu/GO_EO",
            cmd=f"test -f {_NSP_SCRATCH}/cpft_iso30.done && test -f " + GD + "/eo/train3d.py")]


def node(key, title, cfg, deps, mem, files=(), block=False, copy_thr=False):
    out.append(dict(key=key, kind="new", parent=BASE, title=title, desc=DESC, gpus=[0, 1], mem=mem, block=block, deps=list(deps), notify=True,
                    copy_thr=copy_thr, copy_files=list(files), config=dict(cfg, share_gpu=True)))


for s in SEEDS:
    n = s[1:]; pre = "S5_" if s in ("s3000", "s4000") else ""
    a1 = ["--variant", "dec3d", "--init", f"orx_v5_2d_{s}", "--seed", n, "--steps", "20000", "--zero_dec"]
    ft = ["--variant", "dec3d", "--init", f"orx_v5_2d_{s}", "--init3d", f"{MD}/orx_t3d_eo_a1_{s}", "--seed", n, "--steps", "3000", "--warmup", "100", "--save_every", "1000", "--zero_dec"]
    x1 = ["--aniso_aug", "0.33", "--aniso_xy", "1", "--aniso_lo", "2", "--aniso_hi", "10", "--aniso_blur", "0.3", "--aniso_blur_max", "0.6"]
    tr = lambda name, args: dict(mode="gt", train3d=dict(name=name, args=args), no_eval=True, iso_workers=6)
    F = [f"{GD}/eo/train3d.py"]
    node(f"EO_A1_{s}", f"nc EO：A1 种子 {n}（解码器恒零，2 万步，从二维种子 {n} 出发，只训练）", tr(f"orx_t3d_eo_a1_{s}", a1), ["EO_stage"], 17000, F, block=True)
    node(f"EO_C_{s}", f"nc EO：C 种子 {n}（解码器恒零，A1 后 3000 步，只训练）", tr(f"orx_t3d_eo_c_{s}", ft), [f"EO_A1_{s}"], 17000, F, block=True)
    node(f"EO_X1_{s}", f"nc EO：X1 种子 {n}（解码器恒零，A1 后 3000 步 + 各向异性模拟，只训练）", tr(f"orx_t3d_eo_x1_{s}", ft + x1), [f"EO_A1_{s}"], 17000, F, block=True)
    out.append(dict(key=f"EO_S1_{s}", kind="cmd", deps=[f"EO_C_{s}", f"EO_X1_{s}"],
                    cmd=f"{_NSP_SHARED}/env_torch/bin/python {_NSP_SCRATCH}/r3_aniso/avg_weights.py {MD}/orx_t3d_eo_c_{s} {MD}/orx_t3d_eo_x1_{s} {MD}/orx_t3d_eo_s1_{s} 0.5"))
    M = ["--model", f"{MD}/orx_t3d_eo_s1_{s}"]
    node(f"EO_v_{s}", f"nc EO：种子 {n} 验证集 gt 协议 88 个体积（只报告）", dict(script="thr_eval.py", args=M + VAL), [f"EO_S1_{s}"], 14300, copy_thr=True)
    for prot, lab in (("tGT", "测试集 gt 协议 227 个体积"), ("tFP", "测试集完整流程 45 个体积")):
        a = list(J[f"{pre}{prot}_S1_{s}"]["config"]["args"]); a[a.index("--model") + 1] = f"{MD}/orx_t3d_eo_s1_{s}"
        node(f"EO_{prot}_{s}", f"nc EO：种子 {n} {lab}（只报告）", dict(script="thr_eval.py", args=a), [f"EO_S1_{s}"], 14300, copy_thr=True)
json.dump(out, open(f"{GD}/jobs_EO.json", "w"), ensure_ascii=False, indent=1)
print(len(out), "个作业；", sum(j["kind"] == "new" for j in out), "个占 GPU")
for j in out[1:8]: print(j["key"], json.dumps(j.get("config", j.get("cmd")), ensure_ascii=False)[:230])
