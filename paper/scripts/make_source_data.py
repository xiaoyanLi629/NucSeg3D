"""Source Data for Figs 2-6: one CSV per panel, read-only from the same files the figure scripts
(paper/scripts/figs/fig*.py) read; access paths copied from those scripts. Output: panels/*.csv + manifest.json."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import csv, glob, json, os
import numpy as np
O = f"{_NSP_SCRATCH}/nc/source_data/panels"; os.makedirs(O, exist_ok=True)
MAN = {}
def w(name, header, rows, src, note=""):
    with open(f"{O}/{name}.csv", "w", newline="") as f:
        c = csv.writer(f, lineterminator="\n"); c.writerow(header); c.writerows(rows)
    MAN[name] = dict(source=src, note=note, n_rows=len(rows))

# ---------------- Fig 2
ST = f"{_NSP_SCRATCH}/nc/stats"
R = json.load(open(f"{ST}/nc_stats_results.json"))
for j, lab in enumerate(("F1@0.5", "F1@0.7")):
    e = R["results"]["gt"]["all"][lab]["vs2D"]; s2 = e["S2"]
    rows = [[ds, d["n"], d["effect"], d["se"], d["effect"] - 1.96 * d["se"] if d["n"] >= 2 else "", d["effect"] + 1.96 * d["se"] if d["n"] >= 2 else "",
             s2["weights"].get(ds, ""), "pooled" if ds in s2["pooled_ds"] else "plotted only (single volume)"] for ds, d in sorted(e["per_ds"].items())]
    rows.append(["Pooled (random effects, Hartung-Knapp)", s2["k"], s2["mu"], s2["se_hk"], s2["ci_hk"][0], s2["ci_hk"][1], "", f"I2={s2['I2']}; tau2={s2['tau2']}; p_HK={s2['p_hk']}"])
    w(f"Fig2{'ab'[j]}", ["dataset", "n_test_volumes", "S1_minus_2D", "se", "lower_95", "upper_95", "pooling_weight", "note"], rows,
      f"nc_stats_results.json results/gt/all/{lab}/vs2D", "gt protocol; per-dataset difference of volume-level means over five seeds")
REF = {(m, sub): json.load(open(f"{ST}/ref/s5_test_{m}_5seeds_basefixed{sfx}.json")) for m in ("gt", "fp") for sub, sfx in (("all", ""), ("C1", "_clean_test"))}
rows = []
for m, sub, glab in [("gt", "all13", "13"), ("gt", "all", "12"), ("gt", "C1", "C1"), ("fp", "all13", "13"), ("fp", "all", "12"), ("fp", "C1", "C1")]:
    mac = REF[(m, "all" if sub == "all13" else sub)]["macro"]
    for lab in ("F1@0.5", "F1@0.7"):
        tag = [t for t in mac[lab] if ("mesospim" in t) != (sub == "all13")][0]
        ent = mac[lab][tag]
        d = np.array(ent["S1_seeds"]) - np.array(ent["two_d_seeds"])
        mean, sd = d.mean(), d.std(ddof=1)
        for s, a, b in zip((0, 1000, 2000, 3000, 4000), ent["S1_seeds"], ent["two_d_seeds"]):
            rows.append(["gt" if m == "gt" else "full pipeline", glab, lab, s, a, b, a - b, mean, sd, bool(mean > 2 * sd and mean >= 0.01)])
w("Fig2c", ["protocol", "datasets", "metric", "seed", "S1_macro", "2D_macro", "difference", "mean_difference", "sd_difference", "confirmed"], rows,
  "nc/stats/ref/s5_test_{gt,fp}_5seeds_basefixed{,_clean_test}.json macro", "confirmation rule: mean > 2 s.d. and >= 0.01")
SEL = json.load(open(f"{_NSP_SCRATCH}/nc/figs/ortho/selection.json"))
w("Fig2d", ["selection_json"], [[json.dumps(SEL, ensure_ascii=True)]], "nc/figs/ortho/selection.json",
  "image panel: volume identifiers and selection values; per-volume F1 in Supplementary Data 2")

# ---------------- Fig 3
E = json.load(open(f"{_NSP_SCRATCH}/nc/errors/results.json"))
M12 = [k for k in E["gt"]["A1"]["all"] if "mesospim" in k][0]
e = E["gt"]["A1"]["all"][M12]; comps = ("miss", "merge", "split", "boundary", "false")
rows = []
for m in ("S1", "2D", "base"):
    rows.append([m, "mean"] + [e[m]["decomp"][c] for c in comps] + [e[m]["one_minus_f1"]])
    for i, sd in enumerate(e[m].get("seeds", [])):
        rows.append([m, f"seed index {i}"] + [sd["decomp"][c] for c in comps] + [sum(sd["decomp"][c] for c in comps)])
w("Fig3a", ["model", "seed", *comps, "one_minus_F1@0.5"], rows, "nc/errors/results.json gt/A1/all/12 datasets",
  "seed index 0-4 = seeds 0,1000,2000,3000,4000; original has one model")
s4 = E["gt"]["S4"]["all"][M12]; g = np.round(0.5 + 0.05 * np.arange(10), 2)
rows = []
for m in ("S1", "2D", "base"):
    cs = s4[m].get("curve_seeds") or [s4[m]["curve"]]
    for i, t in enumerate(g):
        rows.append([m, t, s4[m]["curve"][i]] + [c[i] for c in cs] + [""] * (5 - len(cs)))
d_seed = np.array(s4["S1"]["curve_seeds"]) - np.array(s4["2D"]["curve_seeds"])
for i, t in enumerate(g):
    rows.append(["S1 minus 2D", t, s4["S1_minus_2D_curve"][i]] + list(d_seed[:, i]))
w("Fig3b", ["model", "IoU_threshold", "F1_mean", "seed0", "seed1000", "seed2000", "seed3000", "seed4000"], rows, "nc/errors/results.json gt/S4/all/12 datasets")
A4 = json.load(open(f"{_NSP_SCRATCH}/nc/gpu/a4/a4_interp.json")); cv = A4["v1"]["curve"]
rows = [[b] + [cv[k][i] for k in ("dz", "dxy", "dcp")] for i, b in enumerate(A4["bins"])]
w("Fig3c", ["signed_distance_to_boundary_voxels", "mean_abs_change_axial_flow", "mean_abs_change_lateral_flow", "mean_change_foreground_logit"], rows,
  "nc/gpu/a4/a4_interp.json v1 (25 test volumes, 13 datasets, S1 seed 0)",
  f"example image panel: {A4['fig_volume']}; region means: {json.dumps(A4['v1']['region'])}")
G6 = {k: json.load(open(f"{_NSP_SCRATCH}/nc/gpu/g6/g6_{k}.json")) for k in ("val", "test")}
rows = []
for split in ("val", "test"):
    T = G6[split]["T_12"]
    for k, (lab, key) in enumerate((("F1@0.5", "ci05"), ("F1@0.7", "ci07"))):
        rows.append([split, lab, "on minus off", T["on"][k], T["off"][k], -T["diff"][k], -T[key][1], -T[key][0], ""])
    on, off = G6[split]["T_err_on"], G6[split]["T_err_off"]
    rows.append([split, "merge rate", "on / off", on["merge"], off["merge"], "", "", "", f"n_gt={on['n_gt']}; anisotropic volumes={G6[split]['n_aniso']}"])
w("Fig3d", ["split", "metric", "quantity", "isotropic_on", "isotropic_off", "difference", "lower_95", "upper_95", "note"], rows,
  "nc/gpu/g6/g6_{val,test}.json T_12, T_err_on/off", "S1 seed 0, rule T, 12 datasets")

# ---------------- Fig 4
G = json.load(open(f"{_NSP_SCRATCH}/nc/gpu/g4/g4_summary.json"))
rows = []
for met, lab in (("f05", "F1@0.5"), ("f07", "F1@0.7")):
    for cc in ("0", "0.25", "0.5"):
        for k in (1, 2, 4, 8):
            C = G["conditions"][f"z{k}_n{cc}"]; e = C[f"all_{met}"]
            i = 0 if met == "f05" else 1
            seeds = [C.get(f"S1_{s}", {}).get("all", [None, None])[i] for s in (0, 1000, 2000)] + [C.get(f"2D_{s}", {}).get("all", [None, None])[i] for s in (0, 1000, 2000)]
            rows.append([lab, cc, k, e["S1"][0], e["S1"][1], e["D2"][0], e["D2"][1], e["base"], e["diff"][0], e["diff"][1], e["diff"][2]] + seeds)
w("Fig4abc", ["metric", "noise_x_contrast", "k", "S1_mean", "S1_sd", "2D_mean", "2D_sd", "original", "diff_mean", "diff_min", "diff_max",
              "S1_seed0", "S1_seed1000", "S1_seed2000", "2D_seed0", "2D_seed1000", "2D_seed2000"], rows,
  "nc/gpu/g4/g4_summary.json conditions/z{k}_n{c}", f"macro over {len(G['datasets'])} datasets ({', '.join(G['datasets'])}), 20 volumes; panels a, b use mean/sd, panel c uses diff columns")

# ---------------- Fig 5
H = f"{_NSP_SCRATCH}/nc/heldout/l1cmp"; V = json.load(open(f"{H}/l1s_vol.json")); L2 = json.load(open(f"{H}/l2_cmp.json"))
SETS = ("bbbc050", "blastospim", "nucverse3d")
def F(proto, n, g_, i):
    rule = "fix" if n == "base" else "r2"
    vs = [v[rule] for k, v in V[f"{proto}|{n}"].items() if k.startswith(g_ + "__")]
    return 2 * sum(x["tp7" if i else "tp5"] for x in vs) / max(sum(x["n_gt"] + x["n_pred"] for x in vs), 1)
def Gm(proto, n, g_, i):
    return float(np.mean([F(proto, n, s, i) for s in SETS])) if g_ == "macro" else F(proto, n, g_, i)
rows = []
for i, lab in ((0, "F1@0.5"), (1, "F1@0.7")):
    for g_ in ("macro", *SETS):
        for n, desc in (("L1_S1", "3D retrained without held-out data"), ("L1_2D", "2D retrained without held-out data"), ("F_S1", "S1 production"), ("F_2D", "2D production")):
            for s in (0, 1000, 2000):
                rows.append([lab, g_, desc, s, Gm("tGT", f"{n}_s{s}", g_, i)])
        rows.append([lab, g_, "Original Cellpose-SAM", "", Gm("tGT", "base", g_, i)])
w("Fig5a", ["metric", "dataset", "model", "seed", "pooled_F1"], rows, "nc/heldout/l1cmp/l1s_vol.json (tGT)", "gt protocol; rule T for S1/2D, fixed threshold for original")
f = lambda n, d, i, rule="r2": L2[n][rule][0][d][i]  # noqa: E731
rows = []
for d in SETS:
    for m, key, l1, ref in (("3D", "3D", "L1_S1", "S1"), ("2D", "2D", "L1_2D", "2D")):
        for i, lab in ((0, "F1@0.5"), (1, "F1@0.7")):
            rows.append([d, m, lab, 0, f(f"{key}_k0", d, i)] + [f(f"{key}_{d}_k{k}", d, i) for k in (1, 2, 5, 10)]
                        + [f(l1, d, i), float(np.mean([f(f"{ref}_s{s}", d, i) for s in (0, 1000, 2000, 3000, 4000)]))])
    for i, lab in ((0, "F1@0.5"), (1, "F1@0.7")):
        rows.append([d, "Original Cellpose-SAM", lab, "", f("base", d, i, "fix"), "", "", "", "", "", ""])
w("Fig5b", ["dataset", "model", "metric", "seed", "k0", "k1", "k2", "k5", "k10", "retrained_before_finetuning", "production_mean_5_seeds"], rows,
  "nc/heldout/l1cmp/l2_cmp.json", "pooled F1 on the data-efficiency test volumes; seed 0")

# ---------------- Fig 6
G1 = json.load(open(f"{_NSP_SCRATCH}/nc/gpu/g1/g1_table2.json"))["results"]
meth = [("S1_mean", "S1 (mean of 3 seeds)"), ("D2_mean", "2D (mean of 3 seeds)"), ("base", "Original Cellpose-SAM"), ("tapenade_sd", "Tapenade StarDist"), ("anystar", "AnyStar"), ("cellos_sd", "Cellos StarDist")]
rows = []
for ds in sorted(G1["S1_mean"]["per_ds"]):
    rows.append([ds] + [G1[k]["per_ds"][ds]["f1_05"] for k, _ in meth])
rows.append(["Macro, 12 datasets"] + [G1[k]["macro12_f1_05"] for k, _ in meth])
rows.append(["Macro, C1"] + [G1[k]["c1_f1_05"] for k, _ in meth])
w("Fig6a", ["dataset"] + [m for _, m in meth], rows, "nc/gpu/g1/g1_table2.json", "F1@0.5, 44 full-pipeline test volumes with GT diameters; Tapenade and Cellos StarDist saw their own test images in training")
G3 = json.load(open(f"{_NSP_SCRATCH}/nc/gpu/g3/g3_compare.json"))
rows = []
for proto in ("gt", "fp"):
    M = G3[proto]["macro"]
    for key, lab in (("S1", "S1"), ("d2", "2D, direct"), ("U_d2", "2D + u-Segment3D"), ("base", "Original, direct"), ("U_base", "Original + u-Segment3D")):
        for met, ml in (("tp5", "F1@0.5"), ("tp7", "F1@0.7")):
            v = M[key][met]
            rows.append(["gt" if proto == "gt" else "full pipeline", lab, ml, v[0], v[1] if key == "S1" else ""])
w("Fig6b", ["protocol", "model", "metric", "macro_F1_13_datasets", "S1_sd_5_seeds"], rows, "nc/gpu/g3/g3_compare.json", "u-Segment3D and 2D: seed 0")
BL = f"{_NSP_SHARED}/baselines/results"
CMP = {m: json.load(open(f"{BL}/eval/compare_test_{m}.json")) for m in ("gt", "fp")}
CMPF = {m: json.load(open(f"{BL}/eval/compare_test_{m}_fixed.json")) for m in ("gt", "fp")}
PST = {m: json.load(open(f"{BL}/eval/paired_stats_{m}.json")) for m in ("gt", "fp")}
REFG = json.load(open(f"{ST}/ref/s5_test_gt_5seeds_basefixed.json"))["macro"]
rows = []
for m, key in (("S1", "S1"), ("2D", "二维"), ("Original Cellpose-SAM", "原版"), ("nnU-Net", "nnU-Net"), ("StarDist-3D", "StarDist")):
    for lab in ("F1@0.5", "F1@0.7"):
        if m in ("S1", "2D"):
            sd = np.array(REFG[lab]["13 套"]["S1_seeds" if m == "S1" else "two_d_seeds"]); rows.append(["absolute", "gt", m, lab, sd.mean(), sd.min(), sd.max(), "", ""])
        elif m.startswith("Original"):
            rows.append(["absolute", "gt", m, lab, CMPF["gt"][lab]["macro13"][key], "", "", "", ""])
        else:
            rows.append(["absolute", "gt", m, lab, CMP["gt"][lab]["macro13"][key], "", "", "", ""])
for bkey, bm in (("nnU-Net", "nnU-Net"), ("StarDist", "StarDist-3D")):
    for proto in ("gt", "fp"):
        for lab in ("F1@0.5", "F1@0.7"):
            e = PST[proto][f"{bkey}|13|{lab}"]
            conf = "确认" in e["confirm"] and "未" not in e["confirm"]
            rows.append(["S1 minus " + bm, "gt" if proto == "gt" else "full pipeline", "", lab, e["mean"], e["boot_ci"][0], e["boot_ci"][1],
                         ";".join(f"{v:.6f}" for v in e["seed_diffs"]), "confirmed" if conf else "not confirmed"])
w("Fig6c", ["quantity", "protocol", "model", "metric", "value", "lower_or_min", "upper_or_max", "seed_differences", "confirmation"], rows,
  "baselines/results/eval/compare_test_{gt,fp}{,_fixed}.json; paired_stats_{gt,fp}.json; nc/stats/ref/s5_test_gt_5seeds_basefixed.json", "macro over 13 datasets")
G5 = f"{BL}/g5"
J = {m: json.load(open(f"{G5}/{fn}.json")) for m, fn in (("S1", "s1"), ("2D", "d2"), ("base", "base"), ("nnunet", "nnunet"), ("stardist3d", "stardistB"))}
rows = []
for m, key, lab in [("S1", "t_ruleT", "S1, rule T"), ("S1", "t_fixed", "S1, fixed"), ("2D", "t_ruleT", "2D, rule T"), ("base", "t_fixed", "Original"),
                    ("nnunet", "t_fixed", "nnU-Net"), ("stardist3d", "t_fixed", "StarDist-3D")]:
    for vid, p in sorted(J[m]["P"].items()):
        rows.append(["per volume", lab, vid, p[key], ""])
XS = {int(k): v for fn in sorted(glob.glob(f"{G5}/stardistB_x*.json")) for k, v in json.load(open(fn))["X"].items()}
for m, sfx, lab in [("S1", "_v3", "S1 (3 views)"), ("S1", "_v1", "S1 (1 view)"), ("2D", "_vNone", "2D"), ("base", "_vNone", "Original"), ("nnunet", "", "nnU-Net"), ("stardist3d", "", "StarDist-3D")]:
    X = XS if m == "stardist3d" else J[m]["X"]
    sides = sorted(int(str(k).split("_")[0]) for k in X if str(k).endswith(sfx) or (sfx == "" and "_" not in str(k)))
    get = (lambda s_: X[s_]) if m == "stardist3d" else (lambda s_: X[f"{s_}{sfx}"])
    for s_ in sides:
        x = get(s_); rows.append(["cube", lab, s_, x["t_fixed"], x.get("peak_torch_mib", x.get("peak_fw_mib")) / 1024])
w("Fig6d", ["series", "model", "volume_id_or_cube_side", "time_s", "peak_memory_GiB"], rows, "baselines/results/g5/*.json",
  f"GPU {J['S1']['gpu']}; per-volume times on the 45 full-pipeline volumes; cubes = central crops of an isotropic NucVerse3D volume, faster of two runs; host load not controlled")
json.dump(MAN, open(f"{O}/manifest.json", "w"), indent=1, ensure_ascii=False)
print("panels", len(MAN), {k: v["n_rows"] for k, v in MAN.items()})
