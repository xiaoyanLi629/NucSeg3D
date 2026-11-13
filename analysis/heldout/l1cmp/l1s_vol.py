import json, os
exec(open("l1_cmp.py").read().split("res = {}")[0])
OUT = "l1s_vol.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for proto in ("tGT", "tFP"):
    keep = set(recs(rdir(f"L1_{proto}_2D")).keys())
    runs = {"base": f"{proto}_base"}
    for s in (0, 1000, 2000):
        for m in ("S1", "2D"):
            runs[f"L1_{m}_s{s}"] = f"L1_{proto}_{m}" if s == 0 else f"L1s{s}_{proto}_{m}"
            runs[f"F_{m}_s{s}"] = f"{proto}_{m}_s{s}"
    for n, k in runs.items():
        key = f"{proto}|{n}"
        if key in out: continue
        rc = recs(rdir(k), keep); assert set(rc) == keep, (key, len(rc))
        out[key] = {f"{a}__{b}": dict(t=v["t"], d=v["d"], **{nm: {q: v[nm][q] for q in ("tp5", "tp7", "n_gt", "n_pred")} for nm in ("fix", "r2")}) for (a, b), v in rc.items()}
        json.dump(out, open(OUT, "w")); print(key, len(rc), flush=True)
print("VOL_DONE")
