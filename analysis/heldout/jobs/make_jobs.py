"""生成 jobs_L.json（L1/L2/L3 作业草稿，格式同 $NUCSEG3D_SCRATCH/r6/jobs.json；不合并进 r6/jobs.json）。
占位符 <NC_L_BASE> 是打好 nc_L.diff 的基础节点 id，发出前由主代理建节点后替换；<D0>、<VPASS> 是 L3 在验证集上定型后的取值。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json, os

H = os.path.dirname(os.path.abspath(__file__))
SH = f"{_NSP_SHARED}/nc_heldout"            # 共享盘副本（三台机器都能读）；L_stage 负责拷贝并核对 md5
M = f"{_NSP_ROOT}/models/models"
RES = f"{_NSP_ROOT}/results"
PY = f"{_NSP_SHARED}/env_torch/bin/python"
BASE = "<NC_L_BASE>"
GO = f"{SH}/GO"                                         # 总闸门：主代理确认后 touch，之前任何作业都不发
ISO_OK = f"{_NSP_SHARED}/tools/hostB_iso30.ok"
W = [f"hostB:{i}" for i in (3, 2, 1, 0)]
G2D = [0, 1]                                            # 二维训练读 hostA 本地解包的 v5_2d
ALL = ["hostC:0", "hostC:1", "hostC:2"] + W[::-1] + [0, 1]
VOX = ["--vox_override", "mesospim=5/10.52/10.52"]
V3 = ["--views", "3", "--views1_ds", "cellos", "parhyale", "mesospim"]
SEL = ["bbbc050", "blastospim", "nucverse3d"]
L2EV = json.load(open(f"{H}/L2_eval_test.json"))
DESC = f"nc 大实验（投稿分析方案 4.5 节，设计见 {_NSP_SCRATCH}/nc/heldout/DESIGN.md）；只报告，不参与任何选择。"
J = []


def job(key, title, cfg=None, gpus=ALL, mem=10000, deps=(), kind="new", block=False, wait=None, cmd=None):
    j = dict(key=key)
    if kind == "cmd":
        j.update(kind="cmd", deps=list(deps), cmd=cmd)
        if wait:
            j["wait_file"] = wait
        J.append(j); return
    j.update(kind=kind, parent=BASE, title=title, desc=DESC, gpus=gpus, mem=mem, block=block, deps=list(deps),
             notify=True, copy_thr="script" in cfg and cfg["script"] == "thr_eval.py", config=cfg)
    if wait:
        j["wait_file"] = wait
    J.append(j)


def thr(model, extra, two_d):
    a = ["--model", model, "--workers", "1" if two_d else "2", "--box", "384"] + ([] if two_d else V3) + extra + VOX
    return {"script": "thr_eval.py", "args": a, "share_gpu": True}


def link(key, src_key, name):
    return (f"python3 -c \"import json,os; r=json.load(open('{_NSP_SCRATCH}/r6/sched_state.json'))['{src_key}']['run']; "
            f"src='{RES}/orx_'+r+'_test'; assert os.path.exists(src+'/rows.json'), src; os.symlink(src, '{RES}/{name}'); print('链接', src)\"")


# ---------------- 0. 拷贝名单到共享盘并核对 md5（总闸门 GO）
job("L_stage", None, kind="cmd", wait=GO,
    cmd=f"mkdir -p {SH} && cd {_NSP_SCRATCH}/nc/heldout && cp -n L1_exclude.json L2_*_exclude.json L1_eval_test.json L2_eval_test.json "
        f"l3_diam.py md5sums.txt {SH}/ && cp -rn L3_fixed_d17 L3_fixed_d30 L3_fixed_d52 {SH}/ && cd {SH} && md5sum -c md5sums.txt")

# ---------------- L1：留出 bbbc050、blastospim、nucverse3d，整条链种子 0 重训
EX1 = f"{SH}/L1_exclude.json"
job("L1_2D", "nc L1：二维微调种子 0，排除三套留出数据（bbbc050、blastospim、nucverse3d）与验证体积；训练后跑测试集 r2a2（给完整流程的直径）",
    {"model": "orx_nc_l1_2d_s0", "train": {"name": "orx_nc_l1_2d_s0", "args": ["--seed_offset", "0", "--exclude_json", EX1],
                                           "exclude_val": False, "slices": "v5_2d"},
     "thrs": [0, -1, -2], "fracs": [0, 0.15, 0.25, 0.35], "val_per_ds": 4, "test_per_ds": 4, "diam_mode": "joint",
     "scales": [6, 10, 17, 30, 52, 90, 150], "share_gpu": True, "splits": ["test"], "parallel_splits": True,
     "joint_rule": "r2a2", "iso": True, "iso_final": True},
    gpus=G2D, mem=22000, block=True, deps=["L_stage"])
job("L1_link", None, kind="cmd", deps=["L1_2D"], cmd=link("L1_link", "L1_2D", "orx_nc_l1_2d_s0_test"))
A1ARGS = ["--variant", "dec3d", "--init", "orx_nc_l1_2d_s0", "--seed", "0"]
job("L1_A1", "nc L1：A1 种子 0（2 万步，从 L1 二维出发，排除三套留出数据）",
    {"mode": "gt", "no_eval": True, "iso_workers": 6, "share_gpu": True,
     "train3d": {"name": "orx_nc_l1_a1_s0", "args": A1ARGS + ["--steps", "20000", "--exclude_ids", EX1]}},
    gpus=W, mem=17000, deps=["L1_2D"], wait=ISO_OK)
FT = A1ARGS + ["--init3d", f"{M}/orx_nc_l1_a1_s0", "--steps", "3000", "--warmup", "100", "--save_every", "1000", "--exclude_ids", EX1]
X1AUG = ["--aniso_aug", "0.33", "--aniso_xy", "1", "--aniso_lo", "2", "--aniso_hi", "10", "--aniso_blur", "0.3", "--aniso_blur_max", "0.6"]
job("L1_C", "nc L1：C 种子 0（L1 A1 对照微调 3000 步，只训练）",
    {"mode": "gt", "no_eval": True, "iso_workers": 6, "share_gpu": True, "train3d": {"name": "orx_nc_l1_c_s0", "args": FT}},
    gpus=W, mem=17000, deps=["L1_A1"], wait=ISO_OK)
job("L1_X1", "nc L1：X1 种子 0（L1 A1 微调 3000 步 + xy 降采样模拟，只训练）",
    {"mode": "gt", "no_eval": True, "iso_workers": 6, "share_gpu": True, "train3d": {"name": "orx_nc_l1_x1_s0", "args": FT + X1AUG}},
    gpus=W, mem=17000, deps=["L1_A1"], wait=ISO_OK)
S1 = f"{M}/orx_nc_l1_s1_s0"
job("L1_S1", None, kind="cmd", deps=["L1_C", "L1_X1"],
    cmd=f"{PY} {_NSP_SCRATCH}/r3_aniso/avg_weights.py {M}/orx_nc_l1_c_s0 {M}/orx_nc_l1_x1_s0 {S1} 0.5")
ONLY = ["--only"] + SEL
D2 = f"{RES}/orx_nc_l1_2d_s0_test"
job("L1_tGT_S1", "nc L1：测试集 gt 协议（规则 T）：L1 S1，三套留出数据全部测试体积", thr(S1, ["--split", "test"] + ONLY, False),
    mem=14300, deps=["L1_S1"])
job("L1_tGT_2D", "nc L1：测试集 gt 协议（规则 T）：L1 二维，三套留出数据全部测试体积", thr("orx_nc_l1_2d_s0", ["--split", "test"] + ONLY, True),
    deps=["L1_2D"])
job("L1_tFP_S1", "nc L1：测试集完整流程（规则 T）：L1 S1，直径取 L1 二维的 r2a2", thr(S1, ["--split", "test", "--max_per_ds", "4", "--diam_from", D2] + ONLY, False),
    mem=14300, deps=["L1_S1", "L1_link"])
job("L1_tFP_2D", "nc L1：测试集完整流程（规则 T）：L1 二维", thr("orx_nc_l1_2d_s0", ["--split", "test", "--max_per_ds", "4", "--diam_from", D2] + ONLY, True),
    deps=["L1_link"])
job("L1_vS1", "nc L1：验证集 88 个体积（规则 T）：L1 S1（域内是否受损的检查，只报告）", thr(S1, [], False), mem=14300, deps=["L1_S1"])

# ---------------- L2：每套 1/2/5/10 个训练体积，从 L1 S1（三维）/ L1 二维出发各 1000 步
for d in SEL:
    ids = L2EV[d]
    for k in (1, 2, 5, 10):
        ex = f"{SH}/L2_{d}_k{k}_exclude.json"
        n3, n2 = f"orx_nc_l2_3d_{d}_k{k}_s0", f"orx_nc_l2_2d_{d}_k{k}_s0"
        job(f"L2_3D_{d}_k{k}", f"nc L2：{d} 取 {k} 个训练体积，从 L1 S1 按 C 配方微调 1000 步（三维）",
            {"mode": "gt", "no_eval": True, "iso_workers": 6, "share_gpu": True,
             "train3d": {"name": n3, "args": A1ARGS + ["--init3d", S1, "--steps", "1000", "--warmup", "100", "--save_every", "1000",
                                                       "--exclude_ids", ex, "--guard_parent", EX1]}},
            gpus=W, mem=17000, deps=["L1_S1"], wait=ISO_OK)
        job(f"L2_2D_{d}_k{k}", f"nc L2：{d} 取 {k} 个训练体积，从 L1 二维按二维配方缩到 1000 步（100 轮 × 10 张）",
            {"model": n2, "no_eval": True, "share_gpu": True,
             "train": {"name": n2, "exclude_val": False, "slices": "v5_2d",
                       "args": ["--seed_offset", "0", "--exclude_json", ex, "--base", f"{M}/orx_nc_l1_2d_s0", "--epochs", "100", "--nimg", "10"]}},
            gpus=G2D, mem=22000, block=False, deps=["L1_2D"])
        ev = ["--split", "test", "--only", d, "--ids"] + ids
        job(f"L2_tGT_3D_{d}_k{k}", f"nc L2：测试集 gt 协议（规则 T）：{d} k={k} 三维", thr(f"{M}/{n3}", ev, False), mem=14300, deps=[f"L2_3D_{d}_k{k}"])
        job(f"L2_tGT_2D_{d}_k{k}", f"nc L2：测试集 gt 协议（规则 T）：{d} k={k} 二维", thr(n2, ev, True), deps=[f"L2_2D_{d}_k{k}"])

# L2 的 k=0 对照：同样 1000 步、不放回任何留出体积，分开“多训 1000 步”与“多给数据”的作用（三套共用一对）
job("L2_3D_k0", "nc L2：k=0 对照，从 L1 S1 按 C 配方再训 1000 步、不加留出体积（三维）",
    {"mode": "gt", "no_eval": True, "iso_workers": 6, "share_gpu": True,
     "train3d": {"name": "orx_nc_l2_3d_k0_s0", "args": A1ARGS + ["--init3d", S1, "--steps", "1000", "--warmup", "100", "--save_every", "1000",
                                                    "--exclude_ids", EX1, "--guard_parent", EX1]}},
    gpus=W, mem=17000, deps=["L1_S1"], wait=ISO_OK)
job("L2_2D_k0", "nc L2：k=0 对照，从 L1 二维再训 1000 步（100 轮 × 10 张）、不加留出体积",
    {"model": "orx_nc_l2_2d_k0_s0", "no_eval": True, "share_gpu": True,
     "train": {"name": "orx_nc_l2_2d_k0_s0", "exclude_val": False, "slices": "v5_2d",
               "args": ["--seed_offset", "0", "--exclude_json", EX1, "--base", f"{M}/orx_nc_l1_2d_s0", "--epochs", "100", "--nimg", "10"]}},
    gpus=G2D, mem=22000, block=False, deps=["L1_2D"])
EV0 = ["--split", "test", "--only"] + SEL + ["--ids"] + sum((L2EV[d] for d in SEL), [])
job("L2_tGT_3D_k0", "nc L2：测试集 gt 协议（规则 T）：k=0 对照 三维", thr(f"{M}/orx_nc_l2_3d_k0_s0", EV0, False), mem=14300, deps=["L2_3D_k0"])
job("L2_tGT_2D_k0", "nc L2：测试集 gt 协议（规则 T）：k=0 对照 二维", thr("orx_nc_l2_2d_k0_s0", EV0, True), deps=["L2_2D_k0"])

# ---------------- L3：S1 自估直径。验证集定型（S1 种子 0，88 个体积；d0 ∈ {17, 30, 52}，一次或两次细化）
OS1 = f"{M}/orx_t3d_r5s1_s0"
for d0 in (17, 30, 52):
    p1 = f"L3v_P1_d{d0}"
    job(p1, f"nc L3：验证集第 1 遍（固定直径 {d0}，只存 cp−2 预测）：S1 种子 0",
        {"script": "eval_gt.py", "args": ["--model", OS1, "--workers", "2", "--box", "384"] + V3 +
         ["--thrs", "-2", "--save_masks", "--diam_from", f"{SH}/L3_fixed_d{d0}"] + VOX, "share_gpu": True},
        mem=14300, deps=["L_stage"])
    job(f"L3v_D1_d{d0}", None, kind="cmd", deps=[p1],
        cmd=f"{PY} {SH}/l3_diam.py --state_key {p1} --out {SH}/L3/val_d{d0}_r1")
    job(f"L3v_P2_d{d0}", f"nc L3：验证集第 2 遍（一次细化，规则 T）：d0={d0}",
        thr(OS1, ["--diam_from", f"{SH}/L3/val_d{d0}_r1"], False), mem=14300, deps=[f"L3v_D1_d{d0}"])
    job(f"L3v_D2_d{d0}", None, kind="cmd", deps=[f"L3v_P2_d{d0}"],
        cmd=f"{PY} {SH}/l3_diam.py --state_key L3v_P2_d{d0} --out {SH}/L3/val_d{d0}_r2")
    job(f"L3v_P3_d{d0}", f"nc L3：验证集第 3 遍（两次细化，规则 T）：d0={d0}",
        thr(OS1, ["--diam_from", f"{SH}/L3/val_d{d0}_r2"], False), mem=14300, deps=[f"L3v_D2_d{d0}"])

# 测试集只报告一次：在 L3 定型（AGENTS.md 写下 d0 与细化次数，touch L3_frozen.ok）之后才发；<D0> 与 <VPASS> 由定型结果替换
FROZEN = f"{SH}/L3_frozen.ok"
for s in (0, 1000, 2000, 3000, 4000):
    m = f"{M}/orx_t3d_r5s1_s{s}"
    p1 = f"L3t_P1_s{s}"
    job(p1, f"nc L3：测试集完整流程第 1 遍（固定直径 <D0>）：S1 种子 {s}",
        {"script": "eval_gt.py", "args": ["--model", m, "--workers", "2", "--box", "384"] + V3 +
         ["--split", "test", "--max_per_ds", "4", "--thrs", "-2", "--save_masks", "--diam_from", f"{SH}/L3_fixed_d<D0>"] + VOX, "share_gpu": True},
        mem=14300, deps=["L_stage"], wait=FROZEN)
    job(f"L3t_D1_s{s}", None, kind="cmd", deps=[p1], cmd=f"{PY} {SH}/l3_diam.py --state_key {p1} --out {SH}/L3/test_s{s}_r1")
    job(f"L3t_P2_s{s}", f"nc L3：测试集完整流程第 2 遍（规则 T，只报告）：S1 种子 {s}",
        thr(m, ["--split", "test", "--max_per_ds", "4", "--diam_from", f"{SH}/L3/test_s{s}_r1"], False), mem=14300, deps=[f"L3t_D1_s{s}"])
    # <VPASS> = 2 时再加 L3t_D2 / L3t_P3（与验证集同样的两步），= 1 时到此为止

json.dump(J, open(f"{H}/jobs_L.json", "w"), ensure_ascii=False, indent=1)
print(len(J), "个作业；GPU 作业", sum(j.get("kind") != "cmd" for j in J))
