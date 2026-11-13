import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
exec(open("l1_cmp.py").read().split("res = {}")[0])
ev = json.load(open(f"{_NSP_SCRATCH}/nc/heldout/L2_eval_test.json"))
keep = {(d, i) for d, v in ev.items() for i in v}
runs = {}
for d in SEL:
    for k in (1, 2, 5, 10):
        for m in ("3D", "2D"): runs[f"{m}_{d}_k{k}"] = f"L2_tGT_{m}_{d}_k{k}"
for m in ("3D", "2D"): runs[f"{m}_k0"] = f"L2_tGT_{m}_k0"
runs.update({"L1_S1": "L1_tGT_S1", "L1_2D": "L1_tGT_2D", "base": "tGT_base"})
for s in (0, 1000, 2000, 3000, 4000):
    p = "" if s < 3000 else "S5_"
    runs[f"S1_s{s}"] = f"{p}tGT_S1_s{s}"; runs[f"2D_s{s}"] = f"{p}tGT_2D_s{s}"
import os
out = json.load(open("l2_cmp.json")) if os.path.exists("l2_cmp.json") else {}
for n, k in runs.items():
    if n in out: continue
    d = rdir(k)
    if not os.path.exists(d + "/rows.json"): print(n, "未完成，跳过"); continue
    kk = keep if "_k" not in n or n.endswith("_k0") else {x for x in keep if x[0] == n.split("_")[1] or (n.split("_")[1] + "_" + n.split("_")[2]) == x[0]}
    rc = recs(d, kk); assert set(rc) == kk, (n, len(rc), len(kk))
    out[n] = dict(fix=score(rc, "fix"), r2=score(rc, "r2"))
    json.dump(out, open("l2_cmp.json", "w"), default=float)
    print(n, len(rc), {s: [round(v, 3) for v in x[:2]] for s, x in out[n]["r2"][0].items()}, flush=True)
json.dump(out, open("l2_cmp.json", "w"), default=float)
print("L2_DONE")
