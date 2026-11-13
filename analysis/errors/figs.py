"""由 results.json 画图（matplotlib，PNG）。用法：python figs.py"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = f"{_NSP_SCRATCH}/nc/errors"
R = json.load(open(f"{OUT}/results.json"))
COL = {"S1": "#1f6fb4", "2D": "#e08a1e", "base": "#7a7a7a"}
LAB = {"S1": "Ours (S1, true 3D)", "2D": "2D fine-tuned", "base": "Cellpose-SAM (orig.)"}
for f in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"):
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
M12 = "12 套（去 mesospim）"


def fig_a1():
    comps = ("miss", "merge", "split", "boundary", "false")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, proto in zip(axs[:2], ("gt", "fp")):
        e = R[proto]["A1"]["all"][M12]
        x = np.arange(3); bottom = np.zeros(3)
        cc = plt.cm.Set2(np.arange(len(comps)))
        for c, colr in zip(comps, cc):
            v = np.array([e[nm]["decomp"][c] for nm in ("S1", "2D", "base")])
            ax.bar(x, v, bottom=bottom, color=colr, label=c); bottom += v
        ax.set_xticks(x, ["S1", "2D", "orig."]); ax.set_ylabel("1 − F1@0.5 (macro, 12 sets)")
        ax.set_title(f"A1 error decomposition – {'GT-diameter protocol (227 vols)' if proto == 'gt' else 'full pipeline (45 vols)'}")
        ax.legend(fontsize=8)
    ax = axs[2]
    e = R["gt"]["A1"]["all"][M12]
    ks = ("zsplit", "split", "zmerge", "xymerge", "merge", "miss", "boundary")
    w = 0.27; x = np.arange(len(ks))
    for i, nm in enumerate(("S1", "2D", "base")):
        ax.bar(x + (i - 1) * w, [100 * e[nm]["gt_rate"][k] for k in ks], w, color=COL[nm], label=LAB[nm])
    ax.set_xticks(x, ["z-split", "split (all)", "z-merge", "xy-merge", "merge (all)", "miss", "boundary"], rotation=30)
    ax.set_ylabel("per 100 GT nuclei (macro, 12 sets)"); ax.set_title("GT-level error rates (GT protocol)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/a1_decomposition.png", dpi=150); plt.close(fig)


def fig_strata():
    S = R["gt"]["A1_strata"]["all"]
    vars_ = (("relvol", "nucleus volume / volume median"), ("depth", "relative z position"), ("crowd", "crowding quintile (10-NN dist / diam.)"), ("zxy", "z/xy voxel ratio"))
    fig, axs = plt.subplots(3, 4, figsize=(17, 11))
    for j, (v, xl) in enumerate(vars_):
        ent = S[v]
        for i, oc in enumerate(("err", "zsplit", "zmerge")):
            ax = axs[i, j]
            for nm in ("S1", "2D", "base"):
                y = [np.nan if a is None else 100 * a for a in ent[nm][oc]["macro"]]
                ax.plot(range(len(ent["labels"])), y, "o-", color=COL[nm], label=LAB[nm])
                ax.plot(range(len(ent["labels"])), [100 * a for a in ent[nm][oc]["pooled"]], ":", color=COL[nm], alpha=0.6)
            ax.set_xticks(range(len(ent["labels"])), ent["labels"], fontsize=8)
            ax.set_ylabel({"err": "GT nuclei not matched (%)", "zsplit": "z-split (%)", "zmerge": "in z-merge (%)"}[oc])
            if i == 2:
                ax.set_xlabel(xl)
            if i == 0:
                ax.set_title(f"{xl}\n n = " + ", ".join(str(n) for n in ent["n"]), fontsize=9)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("A1 strata (GT protocol, 12 sets; solid = dataset-macro, dotted = pooled)")
    fig.tight_layout(); fig.savefig(f"{OUT}/a1_strata.png", dpi=140); plt.close(fig)


def fig_s4():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    g = [0.5 + 0.05 * i for i in range(10)]
    for ax, proto in zip(axs[:2], ("gt", "fp")):
        e = R[proto]["S4"]["all"][M12]
        for nm in ("S1", "2D", "base"):
            ax.plot(g, e[nm]["curve"], "o-", color=COL[nm], label=f"{LAB[nm]} (AUC {e[nm]['auc_norm']:.3f})")
        ax.set_xlabel("IoU threshold"); ax.set_ylabel("F1 (macro, 12 sets)"); ax.legend(fontsize=8)
        ax.set_title(f"S4 F1–IoU – {'GT protocol' if proto == 'gt' else 'full pipeline'}")
    ax = axs[2]
    for proto, ls in (("gt", "-"), ("fp", "--")):
        for sn in ("all", "C1"):
            e = R[proto]["S4"][sn]
            k = M12 if sn == "all" else [x for x in e if "去 mesospim" in x][0]
            ax.plot(g, e[k]["S1_minus_2D_curve"], ls, marker="o", label=f"S1−2D {proto} {sn}")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("IoU threshold"); ax.set_ylabel("ΔF1"); ax.legend(fontsize=8)
    ax.set_title("S1 − 2D across IoU thresholds")
    fig.tight_layout(); fig.savefig(f"{OUT}/s4_f1_iou.png", dpi=150); plt.close(fig)


def fig_c3():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, nm in enumerate(("S1", "2D")):
        pv = R["C3"][nm]["per_vol"]
        lr = [math.log2(r["ratio"]) for r in pv]; cost = [r["f5_gt"] - r["f5_fp"] for r in pv]
        axs[2].scatter(lr, cost, s=14, color=COL[nm], label=nm, alpha=0.7)
    pv = R["C3"]["S1"]["per_vol"]
    axs[0].hist([math.log2(r["ratio"]) for r in pv], bins=25, color="#555")
    axs[0].axvline(0, color="k", lw=0.6)
    axs[0].set_xlabel("log2(estimated d / GT d)"); axs[0].set_ylabel("volumes"); axs[0].set_title("C3 diameter error (45 full-pipeline vols)")
    dss = sorted({r["ds"] for r in pv})
    for nm, off in (("S1", -0.2), ("2D", 0.2)):
        pvv = R["C3"][nm]["per_vol"]
        axs[1].bar(np.arange(len(dss)) + off, [np.mean([r["f5_gt"] - r["f5_fp"] for r in pvv if r["ds"] == d]) for d in dss], 0.4, color=COL[nm], label=nm)
    axs[1].set_xticks(range(len(dss)), dss, rotation=60, fontsize=8); axs[1].set_ylabel("F1@0.5 (GT d) − F1@0.5 (estimated d)")
    axs[1].axhline(0, color="k", lw=0.5); axs[1].legend(); axs[1].set_title("C3 cost per dataset")
    axs[2].axhline(0, color="k", lw=0.5); axs[2].axvline(0, color="k", lw=0.5)
    axs[2].set_xlabel("log2(estimated d / GT d)"); axs[2].set_ylabel("F1@0.5 cost"); axs[2].legend(); axs[2].set_title("cost vs diameter error")
    fig.tight_layout(); fig.savefig(f"{OUT}/c3_diameter.png", dpi=150); plt.close(fig)


def fig_c4():
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    for ax, proto in zip(axs, ("gt", "fp")):
        dss = sorted(R[proto]["C4"]["S1"]["all"]["per_ds_gap_F1@0.5"])
        for i, nm in enumerate(("S1", "2D", "base")):
            g = R[proto]["C4"][nm]["all"]["per_ds_gap_F1@0.5"]
            ax.bar(np.arange(len(dss)) + (i - 1) * 0.27, [g[d] for d in dss], 0.27, color=COL[nm], label=LAB[nm])
        ax.set_xticks(range(len(dss)), dss, rotation=60, fontsize=8); ax.set_ylabel("oracle-threshold F1@0.5 − rule T")
        ax.set_title(f"C4 per-volume oracle gap – {'GT protocol' if proto == 'gt' else 'full pipeline'}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/c4_oracle_gap.png", dpi=150); plt.close(fig)


def fig_a2():
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for nm in ("S1", "2D"):
        e = R["gt"]["A2"].get(nm)
        if not e:
            continue
        c = e["all"]["review_curve"]
        ax.plot(c["frac"], c["recall"], color=COL[nm], label=f"{nm} (seeds {e['seeds'][0].replace('_', ',')}; AUROC {e['all']['auroc_pooled']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="random review")
    ax.set_xlabel("fraction of predicted nuclei reviewed (most uncertain first)"); ax.set_ylabel("fraction of wrong nuclei found")
    ax.set_title("A2 seed-agreement triage (GT protocol, 12 sets)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/a2_review_curve.png", dpi=150); plt.close(fig)


for f in (fig_a1, fig_strata, fig_s4, fig_c3, fig_c4, fig_a2):
    try:
        f(); print("ok", f.__name__)
    except Exception as e:  # 单张图失败不挡其他图
        import traceback
        traceback.print_exc(); print("FAIL", f.__name__, e)
