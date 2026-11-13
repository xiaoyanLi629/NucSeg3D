"""把 jobs_VOX2.json 与 jobs_EO.json 并入 r6/jobs.json 末尾（主代理 2026-11-09 授权，AGENTS.md 331ec39 已登记）。
加一个 cmd 作业 VOX2_done：依赖全部 VOX2 GPU 作业，完成后写 $NUCSEG3D_SHARED/nc_gpu/VOX2_finished；EO_stage 改为同时等这个标记
（wait_file），保证 EO 训练不与 VOX2 抢卡。备份 jobs.json.bak_vox2eo。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json, os, shutil
P = f"{_NSP_SCRATCH}/r6/jobs.json"
V = json.load(open(f"{_NSP_SCRATCH}/nc/gpu/jobs_VOX2.json")); E = json.load(open(f"{_NSP_SCRATCH}/nc/gpu/jobs_EO.json"))
MK = f"{_NSP_SHARED}/nc_gpu/VOX2_finished"
V.append(dict(key="VOX2_done", kind="cmd", deps=[j["key"] for j in V if j["kind"] == "new"], cmd=f"date > {MK}"))
st = [j for j in E if j["key"] == "EO_stage"][0]
st["deps"] = ["VOX2_done"]
J = json.load(open(P)); keys = {j["key"] for j in J}
new = V + E
assert not keys & {j["key"] for j in new}, keys & {j["key"] for j in new}
titles = {j.get("title") for j in J if j.get("title")}
assert not titles & {j["title"] for j in new if j.get("title")}
assert all(set(map(str, j.get("gpus", [0, 1]))) <= {"0", "1"} for j in new), "只能用 hostA [0, 1]"
shutil.copy(P, P + ".bak_vox2eo")
J += new
json.dump(J, open(P + ".tmp", "w"), ensure_ascii=False, indent=1); os.replace(P + ".tmp", P)
print("merged", len(new), "total", len(J))
