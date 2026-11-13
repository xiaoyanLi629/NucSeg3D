"""生成 G3 敏感性分析作业 jobs_G3d.json：u-Segment3D 官方默认 v0（不做流一致性过滤，多级 Otsu、decay 0、sigma 1），原版与二维 s0，
测试集 gt 227 个与完整流程 45 个体积各一次，只报告（AGENTS.md 已登记为事后分析）。参数除聚合设置外与 G3tg/G3tf 逐项相同；
useg_eval.py 已是临时文件放共享盘的版本。闸门 $NUCSEG3D_SHARED/nc_gpu/GO3d。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json, copy
J = {j["key"]: j for j in json.load(open(f"{_NSP_SCRATCH}/nc/gpu/jobs_G3.json"))}
GPUS = [1, "hostB:0", "hostB:1", "hostB:2", "hostB:3"]
out = [dict(key="G3d_stage", kind="cmd", deps=[], wait_file=f"{_NSP_SHARED}/nc_gpu/GO3d", cmd=f"ls {_NSP_SHARED}/nc_gpu/G3 >/dev/null")]
for src, k, prot in (("G3tg_base", "base", "gt 协议 227 个体积"), ("G3tf_base", "base", "完整流程 45 个体积"), ("G3tg_d2", "d2", "gt 协议 227 个体积"), ("G3tf_d2", "d2", "完整流程 45 个体积")):
    j = copy.deepcopy(J[src]); a = j["config"]["args"]
    i = a.index("--useg_sel"); del a[i:i + 4]                       # 去掉 --useg_sel <json> --useg_key <k>
    assert "--useg_key" not in a and "--useg_sel" not in a, a
    a += ["--useg_vars", "0"]
    j.update(key=src.replace("G3t", "G3d"), title=f"nc G3 敏感性：u-Segment3D 官方默认 v0，测试集{prot}（事后分析，只报告）：{k}",
             desc="nc G3 敏感性分析（官方默认设置，事后补做，AGENTS.md 已登记）；只报告，不参与任何选择。", gpus=GPUS, deps=["G3d_stage"])
    out.append(j)
json.dump(out, open(f"{_NSP_SCRATCH}/nc/gpu/jobs_G3d.json", "w"), ensure_ascii=False, indent=1)
for j in out[1:]: print(j["key"], j["mem"], " ".join(j["config"]["args"])[:260])
