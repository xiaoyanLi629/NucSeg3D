"""生成 jobs_L1seeds.json：L1 种子 1000、2000 的作业草稿（不并入 r6/jobs.json）。
配置从 r6/jobs.json 里实际发出的 L1 种子 0 作业机械派生，只换种子相关的字段；L1_vS1 按要求不做，L2、L3 不加种子。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import copy, json, os, re

H = os.path.dirname(os.path.abspath(__file__))
SH = f"{_NSP_SHARED}/nc_heldout"
M = f"{_NSP_ROOT}/models/models"
GO = f"{SH}/GO_seeds"
SEEDS = (1000, 2000)
MD5 = "eee3dda52aef78038e1851e5fe4df783"
G2D = ["hostC:0", "hostC:1", 0, 1]
W = [f"hostB:{i}" for i in (3, 2, 1, 0)]
EVAL = ["hostC:0", "hostC:1", "hostB:0", "hostB:1", "hostB:2", "hostB:3", 0, 1]
KEYS = ["L1_2D", "L1_link", "L1_A1", "L1_C", "L1_X1", "L1_S1", "L1_tGT_S1", "L1_tGT_2D", "L1_tFP_S1", "L1_tFP_2D"]
live = {j["key"]: j for j in json.load(open(f"{_NSP_SCRATCH}/r6/jobs.json"))}
nk = lambda k, s: k.replace("L1_", f"L1s{s}_", 1)


def reseed(x, s):
    """把种子 0 的取值换成 s：权重名 _s0、--seed 0、--seed_offset 0；其余一律不动。"""
    if isinstance(x, dict):
        return {k: reseed(v, s) for k, v in x.items()}
    if isinstance(x, list):
        out = [reseed(v, s) for v in x]
        for i in range(1, len(x)):
            if x[i - 1] in ("--seed", "--seed_offset"):
                assert x[i] == "0", x
                out[i] = str(s)
        return out
    if isinstance(x, str):
        return re.sub(r"_s0(?![0-9])", f"_s{s}", x)
    return x


J = [dict(key="L1s_stage", kind="cmd", deps=[], wait_file=GO,
          cmd=f"cd {SH} && echo \"{MD5}  L1_exclude.json\" | md5sum -c -")]
for s in SEEDS:
    for k in KEYS:
        j = copy.deepcopy(live[k])
        j = reseed(j, s)
        j["key"] = nk(k, s)
        j["deps"] = [nk(d, s) for d in live[k]["deps"] if d != "L_stage"]
        if j["kind"] == "cmd":
            j["cmd"] = j["cmd"].replace("L1_2D", nk("L1_2D", s))
        else:
            t = j["title"].replace("种子 0", f"种子 {s}")
            j["title"] = t if f"种子 {s}" in t else f"{t}（种子 {s}）"
            j["desc"] = live[k]["desc"] + f"L1 补种子 {s}：配方、排除清单、评测体积与种子 0 相同，只换种子。"
        if k == "L1_2D":
            j["gpus"], j["deps"] = G2D, ["L1s_stage"]
        elif k in ("L1_A1", "L1_C", "L1_X1"):
            j["gpus"] = W
        elif j["kind"] != "cmd":
            j["gpus"] = EVAL
        if k in ("L1_A1", "L1_tGT_2D"):                     # 权重写完就发，不等二维作业的 r2a2（实测约 12 小时）跑完
            j["deps"] = ["L1s_stage"]
            j["wait_file"] = f"{M}/orx_nc_l1_2d_s{s}.orx_done"
        J.append(j)
json.dump(J, open(f"{H}/jobs_L1seeds.json", "w"), ensure_ascii=False, indent=1)
print(len(J), "个作业；GPU 作业", sum(j["kind"] != "cmd" for j in J))
