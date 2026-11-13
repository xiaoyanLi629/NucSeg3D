"""G2 验证集选型（PLAN.md 第 4 节）：每个方法 8 组，选 13 套宏平均 pooled F1@0.5 最高的一格，并列取 F1@0.7 高者。
只看体积下限 0.15、0.25 两档（vol0 不在预先写定的网格里，不参与选择）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
B = f"{_NSP_SHARED}/baselines/results/eval"
TN = {1: "t1 优化阈值", 2: "t2 默认0.5/0.4"}
NN = {1: "argmax", 2: "P>0.6", 3: "P>0.75", 4: "P>0.9"}


def cells(name, tag):
    S = json.load(open(f"{B}/{name}/summary.json"))
    out = []
    for k, c in S["cells"].items():
        if c["frac"] in (0.15, 0.25):
            out.append(dict(tag=tag(int(c["thr"])), frac=c["frac"], key=k, f5=c["macro_f1_05"], f7=c["macro_f1_07"],
                            per=c["per_ds"], err=c["err_macro"], n=S["n_volumes"], nds=S["n_datasets"]))
    return out


groups = {"nnU-Net": cells("nnunet_val", lambda v: NN[v]),
          "StarDist": cells("stardist_A_val", lambda v: "A " + TN[v]) + cells("stardist_B_val", lambda v: "B " + TN[v])}
sel = {}
for m, g in groups.items():
    g.sort(key=lambda c: (-round(c["f5"], 10), -c["f7"]))
    print(f"== {m}（{len(g)} 组，{g[0]['n']} 个体积 / {g[0]['nds']} 套）")
    for c in g:
        print(f"  {c['tag']:16s} vol{c['frac']:<5g} F1@0.5 {c['f5']:.4f}  F1@0.7 {c['f7']:.4f}")
    b = g[0]
    sel[m] = {k: b[k] for k in ("tag", "frac", "key", "f5", "f7")}
    print("  选中：", b["tag"], b["frac"], "误差分解", {k: round(v, 3) for k, v in b["err"].items()})
    print("  逐套 F1@0.5/0.7：", "  ".join(f"{d} {v['f1_05']:.3f}/{v['f1_07']:.3f}" for d, v in b["per"].items()))
json.dump(sel, open(f"{B}/selection_val.json", "w"), ensure_ascii=False, indent=1)
