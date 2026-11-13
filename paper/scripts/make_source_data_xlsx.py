"""Assemble paper/source_data/Source_Data.xlsx for the final figure set (paper/figures/final/, 2026-11-10).

Inputs (all local; md5 of every input is written to the 'Sources' sheet):
  source_data/panels/*.csv            per-panel tables built on the server by scripts/make_source_data.py
                                       (their server inputs and md5s are listed in SERVER_MD5 below)
  figures/final/source_data/*.csv     tables exported by the figure scripts (nc-figures, FINAL set)
  figures/final/*.provenance.json     figure provenance (volume identifiers, rules)
  figures/assets_viz/ISO-a3.provenance.json, V2-h2_angle_quiver_xz.provenance.json
  source_data/inputs/flowerr_per_volume.csv, flowerr_results.json (copied from $NUCSEG3D_SCRATCH/nc/flowerr/)
Old -> new numbering: Fig2a-c same; Fig3a <- old 3a; Fig3b <- old 3c (decoder zeroed); Fig3c <- former FigS_mech d; Fig3d <- old 3d (isotropic); Fig3e (ISO-a4) and Fig3g (V2-h3) image panels; old 3b -> FigS_mech c;
Fig4, Fig5 same; Fig7a-d <- old Fig6a-d; Fig2d, Fig3d-f, Fig6, FigS_* new.
Review round 1 (2026-11-11): Fig1c added; Fig5a -> Fig5c-e_pooled and Fig5b -> Fig5f, with new Fig5a-e sheets from Fig5.provenance.json;
Fig6 rebuilt from Fig6.provenance.json (GT, original, nnU-Net, 2D, S1; F1@0.5 and F1@0.7), old Fig6 sheet -> FigS_mm7;
Fig7a new (heat map + radar); old Fig7a -> Fig7b, old Fig7d -> Fig7c, old Fig7c -> Fig7d, old Fig7b -> Fig7e."""
import csv, hashlib, json, pathlib
from openpyxl import Workbook
from openpyxl.styles import Font

ROOT = pathlib.Path(__file__).resolve().parent.parent
SD = ROOT / "source_data"; P = SD / "panels"; FIN = ROOT / "figures" / "final"; FSD = FIN / "source_data"; AV = ROOT / "figures" / "assets_viz"
SERVER_MD5 = {  # inputs of scripts/make_source_data.py on hostA (md5sum, 2026-11-10)
    "nc/stats/nc_stats_results.json": "e55e64c2fe9b592eac4e1831a53f1092",
    "nc/stats/ref/s5_test_gt_5seeds_basefixed.json": "7df9250e7e0e350eb1d77e51be44ab8b",
    "nc/stats/ref/s5_test_fp_5seeds_basefixed.json": "7449936e593170f77ea9b645e7d6c4eb",
    "nc/stats/ref/s5_test_gt_5seeds_basefixed_clean_test.json": "07034a8febb46463c2a6b7aec8e3990e",
    "nc/stats/ref/s5_test_fp_5seeds_basefixed_clean_test.json": "6819c006ef63afacdcfc35dee814f61d",
    "nc/figs/ortho/selection.json": "b3fe53a57e15f3cb487dd3769db43e6d",
    "nc/errors/results.json": "01b398578a871f84d76efb5646f2ed58",
    "nc/gpu/a4/a4_interp.json": "bdfe76fbdff469a19582f9f831fd31ea",
    "nc/gpu/g6/g6_val.json": "d801aca33cfb982cde52121c5078a295",
    "nc/gpu/g6/g6_test.json": "f65cc969a323a06307d7320247b88450",
    "nc/gpu/g4/g4_summary.json": "3ea0513f2de64cc9dc7e24e8233a1c5c",
    "nc/heldout/l1cmp/l1s_vol.json": "4696645056be0acd6b7321796fa8c606",
    "nc/heldout/l1cmp/l2_cmp.json": "8ab5093a5ab1caff63e0729e2c105a96",
    "nc/gpu/g1/g1_table2.json": "d6cfb809017440898af19e1aee9766f0",
    "nc/gpu/g3/g3_compare.json": "c95ecb89028cd04af8bd6349c99e64cb",
    "baselines/results/eval/compare_test_gt.json": "724731969482c98e0b4c213ac7c79eb5",
    "baselines/results/eval/compare_test_gt_fixed.json": "e37d54465d5d4cf709e63151879a49f5",
    "baselines/results/eval/paired_stats_gt.json": "0f25ff5f0f98a6c842e1bf8a34c8d21b",
    "baselines/results/eval/paired_stats_fp.json": "63d0669e14b9eaa10d56e26b18deebec",
    "baselines/results/eval/nnunet5f_stats/compare_test_gt.json": "cf66e17d1a5b4149635e3ff14deac18a",
    "baselines/results/eval/nnunet5f_stats/paired_stats_gt.json": "3cb57a63a85806778a63be8706110532",
    "baselines/results/eval/nnunet5f_stats/paired_stats_fp.json": "091039c3e5da6d869a6001b4f0272f9b",
}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731
rel = lambda p: str(pathlib.Path(p).resolve().relative_to(ROOT))  # noqa: E731
rcsv = lambda p: list(csv.reader(open(p, encoding="utf-8")))  # noqa: E731
J = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
ZH = {"单视图3套": "single view, 3 datasets", "三视图10套": "three views, 10 datasets"}

SHEETS = []   # (name, description, rows, [local sources], [server sources], note)
def add(name, desc, rows, local=(), server=(), note=""):
    SHEETS.append((name, desc, rows, list(local), list(server), note))

# ---- Fig. 1c (review round 1, 2026-11-11): numbers printed in panel c (Fig1.provenance.json items 20-21)
F1P = J(FIN / "Fig1.provenance.json"); IT = F1P["items"]
assert "17.1 / 7.2" in IT[20]["shown"] and "-3.18" in IT[21]["shown"]
add("Fig1c", "Fig. 1c. How z information enters the flow (schematic with data): per-voxel angle between predicted flow and the 3D target on one ground-truth nucleus (mouse embryo E15_Ab2, nucleus 62, seed 0) and the 12-dataset macro difference in median angle.",
    [["quantity", "value", "source"],
     ["median angle, 2D control (three views fused), nucleus 62 (degrees)", 17.1, IT[20]["source"]],
     ["median angle, S1 as evaluated, nucleus 62 (degrees)", 7.2, IT[20]["source"]],
     ["voxels (ground-truth foreground with |target| > 0.5)", 125, IT[20]["source"]],
     ["S1 - 2D difference in median angle, macro over 12 datasets, five seeds (degrees)", -3.18, IT[21]["source"]],
     ["95% interval lower (degrees)", -3.41, IT[21]["source"]], ["95% interval upper (degrees)", -2.89, IT[21]["source"]],
     ["macro median angle S1 (degrees)", 12.4, IT[21]["source"]], ["macro median angle 2D control (degrees)", 15.6, IT[21]["source"]]],
    [FIN / "Fig1.provenance.json", SD / "inputs" / "flowerr_results.json"], [], "post hoc; S1 is trained on this target. Panels a, b, d, e are schematics without plotted data.")
# ---- Fig. 2 (R7-Fig2-B)
F2 = J(FIN / "Fig2.provenance.json"); P2 = F2["panels"]; ORD = F2["row_order"]["order"]; ZXY = F2["row_order"]["z/xy"]
for j_, lab in enumerate(("F1@0.5", "F1@0.7")):
    old = {r_[0]: r_ for r_ in rcsv(P / f"Fig2{'ab'[j_]}.csv")}
    hdr = ["dataset", "z_over_xy_of_median_rule_volume"] + old["dataset"][1:]
    rows = [hdr]
    for ds in ORD:
        e, se, n = P2["ab"[j_]]["per_ds"][ds]
        r_ = old[ds]; assert abs(float(r_[2]) - e) < 1e-12 and int(r_[1]) == n, (ds, lab)
        rows.append([ds, ZXY[ds]] + r_[1:])
    pooled = [r_ for k_, r_ in old.items() if k_.startswith("Pooled")][0]
    assert abs(float(pooled[2]) - P2["ab"[j_]]["pooled"][0]) < 1e-12
    rows.append([pooled[0], ""] + pooled[1:])
    add(f"Fig2{'ab'[j_]}", f"Fig. 2{'ab'[j_]}. Per-dataset S1 - 2D difference in {lab} (gt protocol; difference of volume-level means over five seeds), 95% interval (+-1.96 s.e.), pooling weight; rows in ascending z/xy of each dataset's median-rule test volume (as plotted); last row: random-effects estimate with Hartung-Knapp interval.",
        rows, [P / f"Fig2{'ab'[j_]}.csv", FIN / "Fig2.provenance.json"], ["nc/stats/nc_stats_results.json", "nc/figs/ortho/selection.json"])
c2 = P2["c"]; cols = ("F1@0.5", "F1@0.7", "iou", "count")
rows = [["dataset", "z_over_xy", "n_test_volumes", "dF1@0.5 (S1 - 2D)", "dF1@0.7 (S1 - 2D)", "d median matched IoU (S1 - 2D)", "d |count error| (|2D bias| - |S1 bias|)", "per-dataset win (dot)"]]
for ds in ORD:
    rows.append([ds, ZXY[ds], P2["a"]["per_ds"][ds][2]] + [c2["cells"][ds][k_] for k_ in cols] + [";".join(c2["win_dots"][ds])])
rows.append(["Macro (13 datasets)", "", ""] + [c2["macro"][k_][0] for k_ in cols] + ["confirmed: " + ", ".join(f"{k_}={c2['macro'][k_][1]}" for k_ in cols)])
rows += [[], ["colour scale limits (symmetric, +-)"] + [f"{k_}: {v_}" for k_, v_ in c2["colour_scales"].items()], ["sources"] + [f"{k_}: {v_}".replace("13 套", "13 datasets") for k_, v_ in c2["sources"].items()]]
add("Fig2c", "Fig. 2c. Dataset x metric heat map (gt protocol, test set): S1 - 2D in F1@0.5 and F1@0.7 (pooled per dataset, mean of five seeds), in the median IoU of matched nuclei, and the reduction in absolute count error (|2D bias| - |S1 bias|; positive favours S1); dot = per-dataset bootstrap judgement 'better'; macro row over 13 datasets with confirmation.",
    rows, [FIN / "Fig2.provenance.json"], ["nc/stats/ref/s5_test_gt_5seeds_basefixed.json"],
    "count column from the CPUX analysis (Supplementary Table stab:cpux; macro not confirmed); matched-IoU column from the boundary analysis (md5 " + F2["sources_md5"]["boundary"] + ")")
add("Fig2d", "Fig. 2d. Macro F1 of S1 and the 2D control per seed and paired difference by protocol and subset (13 = all datasets; 12 = without mesoSPIM; C1 = clean subset); confirmation rule mean > 2 s.d. and >= 0.01.",
    rcsv(P / "Fig2c.csv"), [P / "Fig2c.csv"], [k for k in SERVER_MD5 if k.startswith("nc/stats/ref/")])
F6 = J(FIN / "Fig6.provenance.json"); O1, O2 = J(FIN / "FigS_ortho_1.provenance.json"), J(FIN / "FigS_ortho_2.provenance.json")
OV = {**O1["volumes"], **O2["volumes"]}
rows = [["dataset", "volume_id", "F1@0.5_S1_seed0", "F1@0.5_2D_seed0", "F1@0.5_original", "z_over_xy", "volume_S1_minus_2D", "dataset_median_S1_minus_2D"]]
for ds, v2 in P2["e"]["volumes"].items():
    v = OV[ds]; assert v["id"] == v2["id"] and v["f1"]["S1"] == v2["f1"]["S1"] and v["f1"]["2D"] == v2["f1"]["2D"]
    rows.append([ds, v["id"], v["f1"]["S1"], v["f1"]["2D"], v["f1"]["base"], v["ratio"], v["diff"], v["median_diff"]])
add("Fig2e", "Fig. 2e. 3D renderings (image panel) of the median-rule test volume of five datasets (gt protocol, seed 0), S1 and the 2D control; identifiers and per-volume F1@0.5. Same volumes as Fig. 6 and Supplementary Fig. (T grid).",
    rows, [FIN / "Fig2.provenance.json", FIN / "FigS_ortho_1.provenance.json", FIN / "FigS_ortho_2.provenance.json"], ["nc/figs/ortho/selection.json"], "Rule: MM_RULE.md amendment 2026-11-10 (Fig. 2d dataset rule; median-rule volumes)")  # F6 (R8) no longer carries the rule text; string as in the previous workbook
f2 = P2["f"]
add("Fig2f", "Fig. 2f. Semi-transparent cube rendering (image panel, no numbers plotted).",
    [["dataset", "volume_id", "model and seed", "z_over_xy", "voxel_um_zyx", "crop_zyx", "face_opacity"],
     [f2["dataset"], f2["volume"], f2["masks"], f2["ratio"], str(f2["voxel_um_zyx"]), str(f2["crop_zyx"]), f2["face_opacity"]]],
    [FIN / "Fig2.provenance.json"], [])
# ---- Fig. 3 (R7-Fig3)
F3 = J(FIN / "Fig3.provenance.json"); P3 = F3["panels"]; a3 = P3["a"]; comps = ("miss", "merge", "split", "boundary", "false", "total")
rows = [["dataset", "z_over_xy"] + [f"{c_} (2D - S1, percentage points)" for c_ in comps]]
for ds in a3["order"]:
    rows.append([ds, ZXY[ds]] + [a3["cells_pp_2D_minus_S1"][ds][c_] for c_ in comps])
rows.append(["Macro (12 datasets)", ""] + [a3["macro_pp_2D_minus_S1"][c_] for c_ in comps])
rows += [[], ["Absolute macro components (percentage points of 1 - F1@0.5), as given in the legend"], ["model"] + list(comps)]
for m_, v_ in F3["caption_text_moved"]["a_absolute_macro_pp"].items():
    rows.append([m_] + v_)
rows += [[], ["Per-seed absolute components (fractions, not percentage points)"]] + rcsv(P / "Fig3a.csv")
add("Fig3a", "Fig. 3a. Error decomposition by dataset: 2D - S1 in each component of 1 - F1@0.5 (percentage points; positive = fewer errors with S1; gt protocol, 12 datasets, mean of five seeds), rows in ascending z/xy; macro row; absolute macro components per model; per-seed absolute components.",
    rows, [FIN / "Fig3.provenance.json", P / "Fig3a.csv"], ["nc/errors/results.json"], f"colour scale +-{a3['colour_scale_pp']} pp; source: {a3['source']}".replace("套", "datasets"))
r = rcsv(FSD / "Fig3b_decoder_region_means.csv"); c = rcsv(P / "Fig3c.csv")
add("Fig3b", "Fig. 3b. Decoder output zeroed (S1 seed 0, 25 test volumes, 13 datasets): region means (core, boundary band, background) of the absolute change in axial flow (dz), lateral flow (dxy) and foreground logit (dcp), and of flow magnitudes with the decoder on/off; below, the same changes binned by signed distance to the nearest ground-truth boundary.",
    r + [[]] + c, [FSD / "Fig3b_decoder_region_means.csv", P / "Fig3c.csv"], ["nc/gpu/a4/a4_interp.json"])
rows = [["metric", "view group", "model", "mean", "sd_over_5_seeds", "confirmed (S1 - Z only)"]]
for k_, v_ in P3["c"]["values"].items():
    met, grp, mod = k_.split("/")
    rows.append([met, grp, {"Z": "S1 with decoder output zeroed"}.get(mod, mod), v_[0], v_[1], v_[2] if len(v_) > 2 else ""])
add("Fig3c", "Fig. 3c. Dumbbells: S1 versus S1 with the decoder output zeroed (Z), with the 2D control, by view rule; macro F1 mean and s.d. over five seeds (rule T); S1 - Z paired difference with confirmation.",
    rows, [FIN / "Fig3.provenance.json", ROOT / "data_snapshot" / "m1_strat.json"], [], P3["c"]["groups"] + "; error bars: " + P3["c"]["error_bars"])
dv = P3["d"]["values"]
rows = [["split", "metric", "on - off", "lower_95", "upper_95"]]
for sp in ("val", "test"):
    for met in ("F1@0.5", "F1@0.7"):
        rows.append([sp, met, *dv[f"{sp}/{met}"]])
rows += [[], ["split", "merge rate off (native grid)", "merge rate on (isotropic)", "ground-truth nuclei", "anisotropic volumes"]]
for sp in ("val", "test"):
    on, off, ngt, nan_ = dv[f"{sp}/merge_on_off"]; rows.append([sp, off, on, ngt, nan_])
rows += [[], ["split", "metric", "dataset", "on - off (dots)"]]
for sp in ("val", "test"):
    for met in ("F1@0.5", "F1@0.7"):
        for ds, v_ in dv[f"{sp}/{met}/per_ds_on_minus_off"].items():
            rows.append([sp, met, ds, v_])
add("Fig3d", "Fig. 3d. Isotropic flow following on versus off (S1 seed 0, rule T, 12 datasets): change in macro F1 (on - off) with 95% bootstrap interval and per-dataset changes (dots); paired merge rates off -> on (validation 5.0% -> 3.1%, test 3.3% -> 2.2%).",
    rows, [FIN / "Fig3.provenance.json", FSD / "Fig3d_isotropic_following.csv"], ["nc/gpu/g6/g6_val.json", "nc/gpu/g6/g6_test.json"], P3["d"]["note"])
IC, IB = J(AV / "ISO-c.provenance.json"), J(AV / "ISO-b.provenance.json")
rows = [["quantity", "value"]] + [[k_.replace("_", " "), v_] for k_, v_ in IC["numbers_for_caption"].items()]
rows += [[], ["transition matrix (rows: off, native grid; columns: on, isotropic)", *IB["transition_matrix"]["states"]]]
for st_, cnt in zip(IB["transition_matrix"]["states"], IB["transition_matrix"]["counts"]):
    rows.append([st_] + cnt)
rows += [[], ["alluvial flow (pairs merged in at least one condition, n = %d)" % IB["alluvial"]["n"], "pairs"]] + [[k_, v_] for k_, v_ in IB["alluvial"]["flows"].items()]
rows += [[], ["example pair (volume)", "nucleus a", "nucleus b", "z_over_xy", "evaluated outcome off", "evaluated outcome on", "threshold"]]
for pr in IB["pairs"]:
    rows.append([pr["key"], pr["a"], pr["b"], pr["z_over_xy"], pr["evaluated_outcome"]["off"], pr["evaluated_outcome"]["on"], pr["t_off"]])
add("Fig3e", "Fig. 3e. Isotropic flow following over z-stacked nucleus pairs (descriptive; S1 seed 0; 144 test volumes with z/xy >= 2): outcome counts, transition matrix, alluvial flows and the four illustrated example pairs (streamline images are illustrations).",
    rows, [AV / "ISO-c.provenance.json", AV / "ISO-b.provenance.json"], ["nc/gpu/g6/g6_test.json"], IB["alluvial"]["rest"])
pd_, sm = rcsv(FSD / "Fig3f_flowerr_per_dataset.csv"), rcsv(FSD / "Fig3f_flowerr_summary.csv")
add("Fig3f", "Fig. 3f. FLOWERR: change in median angle (degrees) between predicted and ground-truth-derived 3D flow, S1 - 2D, per dataset (mean and per seed) and macro summary (region a primary; regions b, c; 95% interval); per-volume values in sheet Fig3f_per_volume.",
    pd_ + [[]] + sm, [FSD / "Fig3f_flowerr_per_dataset.csv", FSD / "Fig3f_flowerr_summary.csv", SD / "inputs" / "flowerr_results.json"], [])
add("Fig3f_per_volume", "Fig. 3f, per-volume FLOWERR values (227 test volumes): delta of median angle per region (a, b, c) and seed, median angles per model and seed (region a), and per-volume ground-truth flow and F1 fields as exported.",
    rcsv(SD / "inputs" / "flowerr_per_volume.csv"), [SD / "inputs" / "flowerr_per_volume.csv"], [])
g3 = P3["g"]
add("Fig3g", "Fig. 3g. Exploded stack of planes (image panel, no numbers plotted): ground truth, 2D control and S1 outlines on one test volume.",
    [["dataset", "volume_id", "seed", "planes_z", "xz_section_y", "xy_section_z", "crop_zyx", "voxel_um_zyx", "selection", "F1@0.5_S1", "F1@0.5_2D"],
     [g3["dataset"], g3["volume"], g3["seed"], ";".join(map(str, g3["planes"])), g3["xz_pulled_y"], g3["xy_pulled_z"], str(g3["crop_zyx"]), str(g3["voxel_um_zyx"]), g3["selection"], g3["F1@0.5"]["S1"], g3["F1@0.5"]["2D"]]],
    [FIN / "Fig3.provenance.json"], [], g3["masks"])
# ---- Fig. 4
add("Fig4", "Fig. 4a-c. Axial subsampling (every kth plane) and noise: macro F1 over six datasets (20 volumes) for S1 and the 2D control (mean, s.d., per seed) and the original model; diff columns (mean, min, max over seed-paired differences) give panel c.",
    rcsv(P / "Fig4abc.csv"), [P / "Fig4abc.csv"], ["nc/gpu/g4/g4_summary.json"])
# ---- Fig. 5 (R8-Fig5, panels a-f; review round 1, 2026-11-11: re-lettered from the earlier a-b layout)
F5 = J(FIN / "Fig5.provenance.json"); P5 = F5["panels"]
rows = [["dataset", "volume_id", "plane_z", "n_test_volumes", "display"]]
for ds, v_ in P5["a"]["thumbnails"].items():
    rows.append([ds, v_["volume"], v_["plane_z"], v_["n_test"], v_["lut"]])
add("Fig5a", "Fig. 5a. Design schematic (image panel): raw xy thumbnails of the representative volumes of panel b and the number of test volumes per held-out dataset.",
    rows, [FIN / "Fig5.provenance.json"], [], P5["a"]["note"])
rows = [["dataset", "volume_id", "n_gt", "n_eligible", "median d F1@0.7 (retrained - production)", "model", "F1@0.5", "F1@0.7"] + list(P5["b"]["volumes"]["bbbc050"]["prod"]["classes"])]
for ds, v_ in P5["b"]["volumes"].items():
    for mk, mn in (("prod", "production S1, seed 0"), ("ret", "retrained S1, seed 0")):
        rows.append([ds, v_["volume"], v_["n_gt"], v_["n_eligible"], v_["median_rule_d_F1@0.7"], mn, v_[mk]["F1@0.5"], v_[mk]["F1@0.7"]] + list(v_[mk]["classes"].values()))
add("Fig5b", "Fig. 5b. Representative held-out test volumes (3D renders; gt protocol, rule T): per-volume F1@0.5 / F1@0.7 of production and retrained S1 (seed 0) and the number of predicted nuclei per render colour class (match = IoU >= 0.7, steel blue; iou57 = matched with IoU 0.5-0.7, light blue #A1CCF7).",
    rows, [FIN / "Fig5.provenance.json"], ["nc/heldout/l1cmp/l1s_vol.json"], "selection rule: retrained - production F1@0.7 of S1 closest to the dataset median among volumes with >= 10 ground-truth nuclei")
rows = [["metric", "dataset", "models", "seed 0", "seed 1000", "seed 2000", "mean", "sd", "confirmed"]]
for k_, v_ in P5["c"]["values"].items():
    met, ds = k_.split("/")
    for mk in ("production", "retrained"):
        x = v_[mk]; rows.append([met, ds, mk, *x["seeds"], x["mean"], x["sd"], x["confirmed"]])
add("Fig5c", "Fig. 5c. The 3D advantage S1 - 2D (pooled F1, gt protocol) in the production and the retrained models, per seed and mean; confirmation rule |mean| > 2 s.d. and >= 0.01.",
    rows, [FIN / "Fig5.provenance.json"], ["nc/heldout/l1cmp/l1s_vol.json"])
rows = [["dataset", "metric", "model", "seed 0", "seed 1000", "seed 2000", "mean", "sd", "confirmed"]]
for k_, v_ in P5["d"]["values"].items():
    ds, met, mod = k_.split("/")
    rows.append([ds, met, {"Δ": "difference (change in the 3D advantage)"}.get(mod, mod), *v_["seeds"], v_["mean"], v_["sd"], v_["confirmed"]])
add("Fig5d", "Fig. 5d. Cost of withholding: retrained - production pooled F1 (gt protocol) for S1, the 2D control and their difference, per seed and mean, with confirmation; liver and fly are the two NucVerse3D subsets.",
    rows, [FIN / "Fig5.provenance.json"], ["nc/heldout/l1cmp/l1s_vol.json"], P5["d"]["frame"])
rows = [["metric", "dataset", "model", "n_volumes", "median per-volume F1 (black tick)", "pooled F1 seed 0", "pooled F1 seed 1000", "pooled F1 seed 2000"]]
for k_, v_ in P5["e"]["values"].items():
    met, ds, mod = k_.split("/")
    rows.append([met, ds, {"F_S1": "production S1", "L1_S1": "retrained S1", "F_2D": "production 2D control", "L1_2D": "retrained 2D control"}[mod], v_["n_vol"], v_["median_per_volume"], *v_["pooled_seeds"]])
add("Fig5e", "Fig. 5e. Per-volume F1 distributions (violins over volumes; per-volume F1 averaged over seeds 0, 1000, 2000): medians and pooled F1 per seed (dots); per-seed pooled F1 of all models and the original model in sheet Fig5c-e_pooled.",
    rows, [FIN / "Fig5.provenance.json"], ["nc/heldout/l1cmp/l1s_vol.json"])
add("Fig5c-e_pooled", "Fig. 5c-e (underlying values; formerly sheet Fig5a). Held-out datasets: pooled F1 (gt protocol) of retrained and production 3D and 2D models per seed and of the original Cellpose-SAM.",
    rcsv(P / "Fig5a.csv"), [P / "Fig5a.csv"], ["nc/heldout/l1cmp/l1s_vol.json"])
f5 = P5["f"]
rows = rcsv(P / "Fig5b.csv") + [[], ["Plotted gaps (percentage points / 100 = F1 units) as drawn in panel f; " + f5["note"]]]
for key, lab in (("gapS1", "S1 minus production S1"), ("gap2D", "2D control minus production 2D control"), ("diff", "S1 - 2D")):
    rows += [[], [lab, *f5[key]["cols"]]] + [[k_.replace("  ", " "), *v_] for k_, v_ in f5[key]["values"].items()]
add("Fig5f", "Fig. 5f (formerly sheet Fig5b). Data efficiency (seed 0): pooled F1 after fine-tuning with k volumes of each held-out dataset added back; retrained model before fine-tuning; production mean of five seeds; original model; below, the gaps plotted in the three heat maps.",
    rows, [P / "Fig5b.csv", FIN / "Fig5.provenance.json"], ["nc/heldout/l1cmp/l2_cmp.json"])
# ---- Fig. 6 (R8-Fig6: ground truth + four methods, F1@0.5 and F1@0.7 per tile)
rows = [["dataset", "volume_id", "z_over_xy", "column", "F1@0.5", "F1@0.7", "tp at IoU 0.5", "tp at IoU 0.7", "n_gt", "n_pred", "source record"]]
MN = {"base": "Original Cellpose-SAM (fixed threshold)", "nnunet": "nnU-Net", "2D": "2D control (seed 0, rule T)", "S1": "S1 (seed 0, rule T)"}
for ds in F6["datasets"]:
    rw = F6["rows"][ds]; n_gt = rw["f1"]["S1"]["n_gt"]
    rows.append([ds, rw["volume"], rw["z_xy"], "Ground truth (number of nuclei)", "", "", "", "", n_gt, "", ""])
    for mk in ("base", "nnunet", "2D", "S1"):
        x = rw["f1"][mk]; assert x["n_gt"] == n_gt
        rows.append([ds, rw["volume"], rw["z_xy"], MN[mk], x["f1_05"], x["f1_07"], x["tp5"], x["tp7"], x["n_gt"], x["n_pred"], x["source"]])
rows += [[], ["inset (dark box): dataset", "centre label", "score (summed per-nucleus IoU disagreement)", "nuclei in window", "window voxels (z, y, x)"]]
for ds in F6["datasets"]:
    z_ = F6["rows"][ds]["zoom"]; rows.append([ds, z_["centre_label"], z_["score"], z_["n_nuclei_in_window"], str(z_["window_voxels"])])
add("Fig6", "Fig. 6. The median-rule test volume of five datasets (gt protocol) segmented by five columns: ground truth, original Cellpose-SAM, nnU-Net, 2D control and S1; per-volume F1@0.5 and F1@0.7 as printed on each tile, with true positives and object counts; inset windows. StarDist-3D and u-Segment3D: sheet FigS_mm7.",
    rows, [FIN / "Fig6.provenance.json"], ["baselines/results/eval/compare_test_gt.json"], "Rule file " + F6["rule_file"] + " (md5 " + F6["rule_md5"] + ")")
r6 = rcsv(FSD / "Fig6_per_volume_counts_F1.csv")
add("FigS_mm7", "Supplementary Fig. (six-method callouts; formerly sheet Fig6). Per-volume counts and F1@0.5 on the five median-rule volumes (gt protocol): ground-truth nuclei hit at IoU >= 0.5 (tp), n_gt, n_pred and F1@0.5 per model (S1 and 2D seed 0; original; nnU-Net; StarDist-3D; u-Segment3D consensus of the 2D control).",
    r6, [FSD / "Fig6_per_volume_counts_F1.csv", FIN / "Fig6.provenance.json"], ["baselines/results/eval/compare_test_gt.json"], F6.get("equal_f1_note", "NIS3D: 2D 0.41406 (tp 318, n_pred 653) vs u-Segment3D 0.41440 (tp 328, n_pred 700) - different, both round to 0.414. BlastoSPIM: original, 2D, u-Segment3D and S1 all tp 59 / n_gt 60 / n_pred 59 -> 0.99160, genuinely equal."))  # note text from the R7 Fig6 provenance (previous workbook)
# ---- Fig. 7 (R6-Fig7-B, panels a-e; review round 1: re-lettered)
F7 = J(FIN / "Fig7.provenance.json"); P7 = F7["panels"]; COL7 = F7["ordering"]["column_order"]
rows = [["metric", "model"] + COL7 + ["macro13", "macro12"]]
for met, mods in P7["a"]["heatmap"].items():
    for mk, v_ in mods.items():
        rows.append([met, mk] + [v_.get(c_, "") for c_ in COL7 + ["macro13", "macro12"]])
rows += [[], ["Radar (F1@0.7, same values as the F1@0.7 heat-map rows; scale " + P7["a"]["radar_scale"] + ")"], ["model"] + COL7]
for mk, v_ in P7["a"]["radar_F1@0.7"].items():
    rows.append([mk] + [v_[c_] for c_ in COL7])
add("Fig7a", "Fig. 7a. Model x dataset heat maps of pooled F1@0.5 and F1@0.7 (gt protocol, 227 test volumes; S1 and 2D control mean of five seeds; original, nnU-Net (single fold and five-fold ensemble, nnunet5f) and StarDist-3D one run each) with 13- and 12-dataset macro, and the F1@0.7 radar.",
    rows, [FIN / "Fig7.provenance.json"], ["nc/stats/ref/s5_test_gt_5seeds_basefixed.json", "baselines/results/eval/compare_test_gt.json", "baselines/results/eval/nnunet5f_stats/compare_test_gt.json"], F7["ordering"]["columns"])
add("Fig7b", "Fig. 7b (formerly sheet Fig7a). Per-dataset F1@0.5 of S1, the 2D control (means of three seeds), the original model and three published tools on 44 full-pipeline test volumes with ground-truth diameters; macro rows over 12 datasets and over C1.",
    rcsv(P / "Fig6a.csv"), [P / "Fig6a.csv"], ["nc/gpu/g1/g1_table2.json"], "hatched (tool trained on material including these test images): " + "; ".join(t_[1] for t_ in P7["b"]["hatched"]))
pv = [r_ for r_ in rcsv(P / "Fig6d.csv") if r_[0] == "per volume"]
cube7d = rcsv(FSD / "Fig7d_cube_time_memory.csv"); cube7d[0] = cube7d[0] + ["note"]
for r_ in cube7d[1:]:   # footnote requested by nc-figures (FINAL-v3); checked against baselines/results/g5/nnunet.json X["128"]
    r_.append("nvidia-smi poll missed the peak (764 MiB recorded < framework 1512 MiB); framework value shown"
              if (r_[0], r_[1]) == ("nnU-Net", "128") else "")
pts = [["Plotted points: model", "median time (s)", "q1", "q3", "time key", "peak memory 256^3 (GiB)", "macro F1@0.5 (gt, 13 datasets, 227 volumes)", "macro F1@0.7 (gt, 13 datasets, 227 volumes)", "timed volumes"]]
pts += [[mk, v_["t"], v_["q1"], v_["q3"], v_["time_key"], v_["mem"], v_["f05"], v_["f07"], v_["n"]] for mk, v_ in P7["c"]["points"].items()]
pts += [["front (not dominated)", "F1@0.5: " + ", ".join(P7["c"]["pareto_front"]["f05"]), "F1@0.7: " + ", ".join(P7["c"]["pareto_front"]["f07"])]]
add("Fig7c", "Fig. 7c (formerly sheet Fig7d). Accuracy against inference cost: plotted points (13-dataset macro F1, gt protocol, 227 volumes, against median time on the 45 full-pipeline volumes; bubble = peak GPU memory on a 256^3 cube); run time per volume (summary and raw) and time and peak GPU memory on cubes of increasing size (memory = max of framework-reported and nvidia-smi peak).",
    pts + [[]] + rcsv(FSD / "Fig7d_per_volume_time.csv") + [[]] + cube7d + [[]] + [["series", "model", "volume_id", "time_s"]] + [r_[:4] for r_ in pv],
    [FIN / "Fig7.provenance.json", FSD / "Fig7d_per_volume_time.csv", FSD / "Fig7d_cube_time_memory.csv", P / "Fig6d.csv"], [], "server inputs: baselines/results/g5/*.json; host load not controlled. " + P7["c"]["note"])
CF = {"S1 优（确认）": "S1 better (confirmed)", "基线优（确认）": "baseline better (confirmed)", "未确认": "not confirmed"}
rows = rcsv(P / "Fig6c.csv") + [[], ["Plotted squares and dots (panel d): comparison", "protocol", "datasets", "metric", "mean of five seed-paired macro differences", "boot lower", "boot upper", "seed rule"] + sorted(P7["d"]["nnU-Net|gt|13|F1@0.5"]["per_ds_d"])]
for k_, v_ in P7["d"].items():
    cmp_, prot, st, met = k_.split("|")
    rows.append([cmp_, {"fp": "full pipeline"}.get(prot, prot), st, met, v_["mean"], *v_["boot_ci"], CF.get(v_["confirm"], v_["confirm"])] + [v_["per_ds_d"].get(d_, "") for d_ in sorted(P7["d"]["nnU-Net|gt|13|F1@0.5"]["per_ds_d"])])
add("Fig7d", "Fig. 7d (formerly sheet Fig7c). Native 3D baselines: absolute macro F1 (gt, 13 datasets) and S1 minus baseline with 95% volume-bootstrap interval, seed differences and confirmation; below, the plotted squares (filled = confirmed by the seed rule) and per-dataset dots.",
    rows, [P / "Fig6c.csv", FIN / "Fig7.provenance.json"],
    ["baselines/results/eval/compare_test_gt.json", "baselines/results/eval/compare_test_gt_fixed.json", "baselines/results/eval/paired_stats_gt.json", "baselines/results/eval/paired_stats_fp.json", "baselines/results/eval/nnunet5f_stats/paired_stats_gt.json", "baselines/results/eval/nnunet5f_stats/paired_stats_fp.json", "nc/stats/ref/s5_test_gt_5seeds_basefixed.json"])
rows = rcsv(P / "Fig6b.csv") + [[], ["Plotted bars (panel e): protocol/metric/model", "macro F1 (13 datasets)"]] + [[k_, v_] for k_, v_ in P7["e"].items()]
add("Fig7e", "Fig. 7e (formerly sheet Fig7b). Consensus 3D reconstruction (u-Segment3D) versus direct inference: macro F1 over 13 datasets (S1 mean and s.d. of five seeds; others seed 0); U_ = u-Segment3D consensus, d2 = 2D control, base = original model.",
    rows, [P / "Fig6b.csv", FIN / "Fig7.provenance.json"], ["nc/gpu/g3/g3_compare.json"])
# ---- Supplementary figures
add("FigS_mech_c", "Supplementary Fig. (mechanism) c. S1 - 2D in F1 against IoU threshold (gt protocol, 12 datasets; mean of five seed-paired differences); full curves per model and seed below.",
    rcsv(FSD / "FigS_mech_c_F1_IoU_S1_minus_2D.csv") + [[]] + rcsv(P / "Fig3b.csv"), [FSD / "FigS_mech_c_F1_IoU_S1_minus_2D.csv", P / "Fig3b.csv"], ["nc/errors/results.json"])
AN = J(FIN / "FigS_anisotropy3D.provenance.json")
add("FigS_anisotropy3D", "Supplementary Fig. (anisotropy 3D). Illustration only; no metric is computed or shown (statistics in Fig. 4). Volume and crop as selected by the pre-written rule.",
    # R9 provenance no longer carries run_log (data and crop unchanged from R8-F5-a, per its 'items'); values as in the previous workbook
    [["volume", "crop_yx", "diameter_px", "voxel_zyx_um"], (["celegans/pha4I2L_0408072", "[31, 150, 96]", 12.42082752636985, "[0.122, 0.116, 0.116]"] if "run_log" not in AN else
     [AN["run_log"]["key"][0] + "/" + AN["run_log"]["key"][1], str(AN["run_log"]["crop_yx"]), AN["run_log"]["d"], str(AN["run_log"]["vox"])])],
    [FIN / "FigS_anisotropy3D.provenance.json"], [])
add("FigS_Tgrid", "Supplementary Fig. (T grid). 3D renderings (image panel) of the same five median-rule volumes as Fig. 2e for all models; per-model counts and F1 in sheet Fig6.",
    rows := [["dataset", "volume_id"]] + [[ds, F6["rows"][ds]["volume"]] for ds in F6["datasets"]], [FIN / "FigS_Tgrid.provenance.json"], [])
rows = [["figure", "dataset", "volume_id", "z_over_xy", "volume_S1_minus_2D", "dataset_median_S1_minus_2D", "F1@0.5_S1", "F1@0.5_2D", "F1@0.5_original", "threshold_S1", "threshold_2D", "threshold_original"]]
for fig, O in (("FigS_ortho_1", O1), ("FigS_ortho_2", O2)):
    for ds in O["order"]:
        v = O["volumes"][ds]
        rows.append([fig, ds, v["id"], v["ratio"], v["diff"], v["median_diff"], v["f1"]["S1"], v["f1"]["2D"], v["f1"]["base"], v["thr"]["S1"], v["thr"]["2D"], v["thr"]["base"]])
add("FigS_ortho", "Supplementary Figs (orthogonal views, parts 1 and 2). Image panels: the median-rule test volume per dataset (gt protocol, seed 0), with selection values and per-volume F1.",
    rows, [FIN / "FigS_ortho_1.provenance.json", FIN / "FigS_ortho_2.provenance.json"], ["nc/figs/ortho/selection.json"])

# ---- Supplementary Table (stab:cpux), no figure: CPUX (AGENTS.md b8932d9, 1d6edcc; post hoc, test set report-only)
CX = ROOT / "data_snapshot" / "round5" / "cpux"; C = J(CX / "results.json")
TR = {"浅": "shallow", "中": "middle", "深": "deep", "低": "low", "高": "high", "全部": "all", "×": " x ", "S1 更小": "S1 smaller", "二维更小": "2D smaller"}
def tr(x):
    if not isinstance(x, str):
        return x
    for a_, b_ in TR.items():
        x = x.replace(a_, b_)
    return x
RD = {"n": "number of nuclei", "vol_med": "median nuclear volume", "nn_med": "median nearest-neighbour distance"}
ST = {"abs_bias": "primary: macro |bias| (bias = mean signed relative error per dataset)", "mean_abs_err": "secondary: macro mean |relative error| per volume"}
rows = [["A. Morphometric readouts: S1 vs 2D (gt protocol, test set, five seeds; confirmation = |mean of the five seed differences| > 2 s.d. (ddof = 1) and 95% dataset-stratified volume-bootstrap interval excluding 0, no minimum effect size; negative differences favour S1; original model descriptive only)"],
        ["readout", "datasets", "statistic", "n_volumes", "S1", "2D", "original", "diff_seed0", "diff_seed1000", "diff_seed2000", "diff_seed3000", "diff_seed4000",
         "diff_mean", "diff_sd", "boot_lower", "boot_upper", "confirmed", "direction", "S1_minus_original", "S1_minus_original_lower", "S1_minus_original_upper"]]
for rk, rv in C["analysis_A"]["readouts"].items():
    for st in ("12", "13"):
        for sk in ("abs_bias", "mean_abs_err"):
            e = rv["sets"][st][sk]
            rows.append([RD[rk], st, ST[sk], rv["n_vol"], e["S1"], e["two_d"], e["base"], *e["diff_seeds"], e["diff_mean"], e["diff_sd"], *e["boot_ci"],
                         e["confirmed"], tr(e["direction"]), e["desc_vs_base"]["diff"], *e["desc_vs_base"]["boot_ci"]])
rows += [[], ["B. Error by relative depth and local SNR (descriptive; strata: depth thirds of z; SNR thirds from each dataset's own validation nuclei (snr_own, primary) or pooled validation nuclei (snr_pooled, sensitivity); hit5/hit7 = fraction of GT nuclei matched at IoU >= 0.5 / 0.7; biou = boundary IoU of matched pairs (all matched; biou_common = matched by both))"],
         ["stratification", "stratum", "datasets", "n_nuclei", "measure", "S1", "2D", "diff_seed0", "diff_seed1000", "diff_seed2000", "diff_seed3000", "diff_seed4000", "diff_mean", "diff_sd", "boot_lower", "boot_upper"]]
for key, e in C["analysis_B"]["strata"].items():
    kind, stratum, st = key.split("|")
    for m in ("hit5", "hit7", "biou", "biou_common"):
        if m in e:
            x = e[m]
            rows.append([kind, tr(stratum), st, e.get("n_nuclei"), m, x["S1"], x["two_d"], *x["diff_seeds"], x["diff_mean"], x["diff_sd"], *x["boot_ci"]])
add("SuppTable_CPUX", "Supplementary Table (stab:cpux): morphometric bias (A) and error by depth and local SNR (B), S1 vs the 2D control; post hoc analysis registered before computation (design md5 " + C["design_md5"] + "), test set report-only; not shown in a figure.",
    rows, [CX / "results.json", CX / "DESIGN_CPUX.md"], [], "Section A judgements are not corrected for multiplicity (six judgements: three readouts x 12/13 datasets); section B is descriptive. Datasets: 12 = without mesoSPIM (main), 13 = all.")
pvc = rcsv(CX / "per_volume.csv"); pvc[0] = [tr(h) for h in pvc[0]]
add("SuppTable_CPUX_per_volume", "Supplementary Table (stab:cpux), per-volume values (227 test volumes): relative errors of the three readouts per model (seed means), and per-stratum hit rates of S1 and the 2D control.",
    pvc, [CX / "per_volume.csv"], [])
# ---- write workbook
import re
LABEL = {"base": "Original Cellpose-SAM", "nnunet": "nnU-Net", "stardist3d": "StarDist-3D", "useg_2d": "u-Segment3D consensus of 2D control", "useg_base": "Original + u-Segment3D"}
def relabel(x):
    """Internal model keys -> names used in the figures and text (cell values and header tokens)."""
    if x in LABEL:
        return LABEL[x]
    return re.sub(r"(?<![A-Za-z])base(?![A-Za-z])", "original_Cellpose-SAM", x) if re.search(r"[_/]base(?![A-Za-z])|(?<![A-Za-z])base_", x) else x
num = lambda x: x if not isinstance(x, str) else (float(x) if _isnum(x) else relabel(x))  # noqa: E731
def _isnum(s):
    try:
        float(s); return s.strip().lower() not in ("nan", "inf", "-inf", "true", "false", "")
    except ValueError:
        return False
wb = Workbook(); ws = wb.active; ws.title = "README"
ws.append(["Source Data for Figs 2-7 and Supplementary Figures"]); ws["A1"].font = Font(bold=True)
ws.append(["Each sheet holds the numbers plotted in one figure panel; image panels list the volumes shown. F1 values are fractions (0-1); angles in degrees. Seeds are training seeds 0, 1000, 2000, 3000, 4000. gt = ground-truth-diameter protocol; full pipeline = automatic diameters. Fig. 1 is a schematic; the numbers printed in panel c are in sheet Fig1c. Per-volume F1 of all models: Supplementary Data 2. Sheet 'Sources' lists every input file with its md5."])
ws.append([]); ws.append(["Sheet", "Content", "Data rows"])
for cc in ws[4]: cc.font = Font(bold=True)
src = wb.create_sheet("Sources"); src.append(["sheet", "input file", "location", "md5"])
for cc in src[1]: cc.font = Font(bold=True)
for name, desc, rows, local, server, note in SHEETS:
    s = wb.create_sheet(name); s.append([desc]); s["A1"].font = Font(italic=True)
    if note: s.append(["Note: " + note])
    s.append([])
    for i, row in enumerate(rows):
        s.append([num(x) for x in row])
        if i == 0:
            for cc in s[s.max_row]: cc.font = Font(bold=True)
    ws.append([name, desc, sum(1 for r_ in rows[1:] if r_)])
    for p in local: src.append([name, rel(p), "paper/ (this repository)", md5(p)])
    for k in server: src.append([name, k, "analysis server (released with the analysis code)", SERVER_MD5[k]])
out = SD / "Source_Data.xlsx"; wb.save(out); print("wrote", out, len(SHEETS), "data sheets")
