"""jobs_L3t_final.json：L3 测试集作业定型版（d0=30、两次细化、S1 评测 --workers 1）。从 r6/jobs.json 里现有 15 个 L3t 条目派生，只改 <D0>、--workers、GPU 列表；
每个种子新增 L3t_D2_s*、L3t_P3_s*（照验证集 L3v_D2/L3v_P3 的写法）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import copy, json
J = json.load(open(f"{_NSP_SCRATCH}/r6/jobs.json")); L = {j["key"]: j for j in J}
V = {k: L[k] for k in ("L3v_D2_d30", "L3v_P3_d30")}
st = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SH = f"{_NSP_SHARED}/nc_heldout"
def fix(x):
    if isinstance(x, dict): return {k: fix(v) for k, v in x.items()}
    if isinstance(x, list):
        o = [fix(v) for v in x]
        for i in range(1, len(x)):
            if x[i - 1] == "--workers": o[i] = "1"
        return o
    return x.replace("<D0>", "30") if isinstance(x, str) else x
out = []
for s in (0, 1000, 2000, 3000, 4000):
    keys = [f"L3t_P1_s{s}", f"L3t_D1_s{s}", f"L3t_P2_s{s}"]
    for k in keys:
        assert k in L and k not in st, k
        j = fix(copy.deepcopy(L[k]))
        if "gpus" in j: j["gpus"] = [g for g in j["gpus"] if g != "hostC:2"]
        out.append(j)
    p2 = out[-1]
    d2 = copy.deepcopy(V["L3v_D2_d30"]); d2["key"] = f"L3t_D2_s{s}"; d2["deps"] = [f"L3t_P2_s{s}"]
    assert "--state_key L3v_P2_d30" in d2["cmd"] and "L3/val_d30_r2" in d2["cmd"]
    d2["cmd"] = d2["cmd"].replace("--state_key L3v_P2_d30", f"--state_key L3t_P2_s{s}").replace("L3/val_d30_r2", f"L3/test_s{s}_r2")
    p3 = copy.deepcopy(p2); p3["key"] = f"L3t_P3_s{s}"; p3["deps"] = [f"L3t_D2_s{s}"]
    a = p3["config"]["args"]; i = a.index("--diam_from"); assert a[i + 1] == f"{SH}/L3/test_s{s}_r1"; a[i + 1] = f"{SH}/L3/test_s{s}_r2"
    p3["title"] = f"nc L3：测试集完整流程第 3 遍（两次细化，规则 T，只报告）：S1 种子 {s}"
    p3.pop("wait_file", None)
    out += [d2, p3]
json.dump(out, open("jobs_L3t_final.json", "w"), ensure_ascii=False, indent=1)
# 自检
keys = [j["key"] for j in out]; assert len(keys) == len(set(keys)) == 25
titles = [j["title"] for j in out if j.get("kind") != "cmd"]; assert len(titles) == len(set(titles)); oth = {j.get("title") for j in J if not j["key"].startswith("L3t_")}; assert not set(titles) & oth; assert all(j["title"] == L[j["key"]]["title"].replace("<D0>", "30") for j in out if j["key"] in L and j.get("kind") != "cmd")
assert "<D0>" not in json.dumps(out, ensure_ascii=False) and "hostC:2" not in json.dumps(out)
for j in out:
    if j.get("kind") != "cmd": assert j["config"]["args"][j["config"]["args"].index("--workers") + 1] == "1"
    for d in j["deps"]: assert d in keys or d in L, d
print(len(out), "个作业，自检通过"); print(json.dumps(out[:5] + [out[3], out[4]], ensure_ascii=False)[:2500])
