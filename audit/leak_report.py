"""r9 泄漏审计：逐个测试体积归类并生成 leak_audit.md 的逐体积附表与 clean_test.json（只用审计结果，不看任何模型分数）。
类别：
  a  像素级重复，或与训练/验证体积互为原样裁块（含线性缩放后相同）；
  b  与训练/验证体积同一标本或同一原始体积，但空间上不相交；
  c  与训练/验证体积同一时间序列（相邻帧或同一胚胎的不同时间点）；
  d  未发现重复、重叠或同序列关系。
用法：python leak_report.py <leak 目录> <manifest.csv> <val_split.json> <输出 clean_test.json> <输出附表.md>
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict

L, MAN, VAL, OUTJ, OUTMD = sys.argv[1:6]
rows = list(csv.DictReader(open(MAN)))
vs = {(d, i) for d, ids in json.load(open(VAL)).items() for i in ids}
part = {(r["ds"], r["id"]): ("test" if r["split"] == "test" else ("val" if (r["ds"], r["id"]) in vs else "train")) for r in rows}
tests = defaultdict(list)
for r in rows:
    if r["split"] == "test":
        tests[r["ds"]].append(r["id"])
J = lambda p: json.load(open(f"{L}/{p}"))
rec = {}


def put(ds, i, cat, partner, ppart, evidence, src):
    rec[(ds, i)] = dict(cat=cat, partner=partner, partner_part=ppart, evidence=evidence, source=src)


# ---- 通用证据：每个测试体积在 train/val 里指纹最像的体积，以及包含、投影重叠的最高值
near = {}
for ds in tests:
    try:
        P = J(f"pairs/{ds}.json")
    except FileNotFoundError:
        continue
    for t, nb in P["nearest_test"].items():
        if nb:
            c, nm, v = max((v["c"], k, v) for k, v in nb.items())
            near[(ds, t)] = (round(c, 3), nm, v["id"], v["part"])
cont3, contm = {}, {}
for ds in ("tapenade", "mesospim", "bbbc050", "cellos", "nucverse3d", "celegans", "mouse_embryo"):
    for r in J(f"contain/{ds}_3d.json"):
        if "test" not in (r["ps"], r["pb"]) or r["ps"] == r["pb"]:
            continue
        t, o = (r["small"], r["big"]) if r["ps"] == "test" else (r["big"], r["small"])
        v = r.get("ncc_fine") if r.get("ncc_fine") is not None else r["ncc_coarse"]
        if v > cont3.get((ds, t), (-9,))[0]:
            cont3[(ds, t)] = (round(v, 3), o, part[(ds, o)])
for ds in ("bbbc050", "cellos", "celegans", "mouse_embryo", "nucverse3d", "mesospim", "tapenade", "nis3d"):
    for r in J(f"contain/{ds}_mip.json")["pairs"]:
        if "test" not in (r["pa"], r["pb"]) or r["pa"] == r["pb"]:
            continue
        t, o = (r["a"], r["b"]) if r["pa"] == "test" else (r["b"], r["a"])
        if r["mip_overlap_ncc"] > contm.get((ds, t), (-9,))[0]:
            contm[(ds, t)] = (round(r["mip_overlap_ncc"], 3), o, part[(ds, o)])
for r in J("extra/partial.json")["gonuclear_mip"]:
    if "test" in (r["pa"], r["pb"]) and r["pa"] != r["pb"]:
        t, o = (r["a"], r["b"]) if r["pa"] == "test" else (r["b"], r["a"])
        if r["mip_overlap_ncc"] > contm.get(("gonuclear", t), (-9,))[0]:
            contm[("gonuclear", t)] = (round(r["mip_overlap_ncc"], 3), o, part[("gonuclear", o)])


def generic(ds, t):
    s = []
    if (ds, t) in near:
        c, nm, o, p = near[(ds, t)]
        s.append(f"指纹最像 {o}（{p}，{nm} {c}）")
    if (ds, t) in cont3:
        c, o, p = cont3[(ds, t)]
        s.append(f"三维包含最高 {c}（{o}，{p}）")
    if (ds, t) in contm:
        c, o, p = contm[(ds, t)]
        s.append(f"投影部分重叠最高 {c}（{o}，{p}）")
    return "；".join(s) if s else "数据集内没有训练/验证体积可比"


# ---- (a) tapenade：测试体积包含训练/验证块的原样裁块
X = J("extra/tapenade.json")
by = defaultdict(list)
for h in X["hits"]:
    by[h["big"]].append(h)
for t, H in by.items():
    pp = Counter(h["ps"] for h in H)
    put("tapenade", t, "a", ",".join(sorted(h["small"] for h in H)), "/".join(f"{k} {v}" for k, v in sorted(pp.items())),
        f"{len(H)} 个训练/验证 64³ 块是本体积的原样裁块：全部逐像素相同、相关 1.0，块内标注与测试标注相同（逐核 F1 1.0、匹配核平均 IoU 1.0）",
        "leak/contain/tapenade_3d.json、leak/extra/tapenade.json")
# (c) tapenade：square.tif 的其他帧
sq = X["square_consecutive_corr"]
for t in tests["tapenade"]:
    if t.startswith("annotator2_square") and t not in by:
        k = int(t[-2:])
        nbr = min((0, 6, 16), key=lambda j: abs(j - k))
        pt_ = by[f"annotator2_square{nbr:02d}"]
        pids = sorted(h["small"] for h in pt_)
        pcnt = Counter(h["ps"] for h in pt_)
        put("tapenade", t, "c", f"{pids[0]}–{pids[-1]}（共 {len(pids)} 块，都在 annotator2_square{nbr:02d} 里）",
            "/".join(f"{k} {v}" for k, v in sorted(pcnt.items())),
            f"与 square00/06/16 同属四维文件 square.tif（17×90×192×192，相邻帧原分辨率相关 {min(sq):.2f}–{max(sq):.2f}，与 square00 的相关随帧距单调下降），"
            f"离最近的含训练块帧 {abs(nbr - k)} 帧；本体积里没有找到训练/验证块（{generic('tapenade', t)}）",
            "leak/extra/tapenade.json、leak/contain/tapenade_3d.json")
# ---- mouse_embryo
M = J("extra/mouse.json")
e7 = M["E7_Ab1_in_E5_Ab1"]
put("mouse_embryo", "E7_Ab1", "a", "E5_Ab1", "train",
    f"E7_Ab1 是 E5_Ab1 在 (z,y,x)=({e7['pos'][0]},{e7['pos'][1]},{e7['pos'][2]}) 处的原样裁块：逐像素相同、相关 {e7['corr']:.4f}；"
    f"标注不同（{e7['lab_n_a']} 对 {e7['lab_n_b']} 个核，逐核 F1 {e7['lab_f1']}，匹配核平均 IoU {e7['lab_mean_iou_matched']}）",
    "leak/contain/mouse_embryo_3d.json、leak/extra/mouse.json")
pmap = {}
for p in M["partial3d"]:
    if "test" in (p["pa"], p["pb"]) and p["pa"] != p["pb"]:
        t, o = (p["a"], p["b"]) if p["pa"] == "test" else (p["b"], p["a"])
        if p["masked3d_ncc"] > pmap.get(t, (-9,))[0]:
            pmap[t] = (p["masked3d_ncc"], o, part[("mouse_embryo", o)])
# ---- blastospim：BlastoSPIM1 测试帧与训练/验证帧同一延时序列
S = J("seq/seq.json")
V = J("verify/blastospim.json")
full = {}
for r in V:
    if "test" in (r["pa"], r["pb"]) and r["pa"] != r["pb"]:
        t, o = (r["a"], r["b"]) if r["pa"] == "test" else (r["b"], r["a"])
        full[(t, o)] = r["corr_full"]
for t, v in S["blastospim"]["test"].items():
    if t.startswith("BlastoSPIM1"):
        n = v["nearest_same_key"]
        fc = full.get((t, n["id"]))
        put("blastospim", t, "c", n["id"], n["part"],
            f"同一编号 {v['key'].split(':')[1]} 的延时序列，相差 {abs(n['dt'])} 帧，块平均网格相关 {n['c_hi']:.3f}"
            + (f"，原分辨率相关 {fc:.3f}" if fc is not None else "") + "，非逐像素相同",
            "leak/seq/seq.json、leak/verify/blastospim.json")
# ---- parhyale
for t, v in S["parhyale"]["test"].items():
    n = v["nearest_same_key"]
    put("parhyale", t, "c", n["id"], n["part"], f"同一胚胎的延时序列（tp01–tp50 同一文件），相差 {abs(n['dt'])} 个时间点，块平均网格相关 {n['c_hi']:.3f}",
        "leak/seq/seq.json")
# ---- nis3d
for r in J("extra/nis3d.json"):
    put("nis3d", r["R"], "b", r["L"], r["L_part"],
        f"同一原图 {r['volume']}（{tuple(r['raw_shape'])}）沿 x 切开：左块等于原图前 {r['L_width']} 列、右块等于其余 {r['raw_shape'][2] - r['L_width']} 列，"
        f"逐像素核对（最大差 {r['max_abs_diff_L']:.0f}/{r['max_abs_diff_R']:.0f}），两部分互不重叠、紧邻",
        "leak/extra/nis3d.json、leak/nis3d/nis3d.json")
# ---- purdue
best = {}
for r in J("coords/coords.json")["purdue"]:
    if "test" in (r["pa"], r["pb"]) and r["pa"] != r["pb"]:
        t, o = (r["a"], r["b"]) if r["pa"] == "test" else (r["b"], r["a"])
        if r["masked_ncc"] > best.get(t, (-9,))[0]:
            best[t] = (r["masked_ncc"], o, r["group"], r["overlap_frac_at_best"])
grp_members = defaultdict(list)
for r in rows:
    if r["ds"] == "purdue" and r["split"] == "train":
        grp_members[re.search(r"ground_truth_and_synthetic/([^/]+)/", r["source"]).group(1)].append(r["id"])
for t, (c, o, g, ov) in best.items():
    put("purdue", t, "b", ",".join(sorted(grp_members[g])), "/".join(sorted({part[("purdue", x)] for x in grp_members[g]})),
        f"与 {len(grp_members[g])} 个训练/验证子体积同属原始体积 {g}；任意平移、重叠 ≥ 5% 时三维带掩码互相关最高 {c:.3f}（与 {o}），"
        f"远低于真实重叠的 1.0（阳性对照见正文），判定为空间不相交",
        "leak/coords/coords.json、leak/extra/purdue.json")
# ---- mesospim
for t in tests["mesospim"]:
    put("mesospim", t, "b", "c1,c2,c3,c4,c5", "train(c1,c2,c4,c5)/val(c3)",
        f"数据来源说明这些体积取自同一个小鼠脑；{generic('mesospim', t)}，远低于重叠所需的接近 1。注意 c1–c5 从不参与训练，"
        f"第五轮起作为开发集用于选择",
        "leak/contain/mesospim_3d.json、leak/contain/mesospim_mip.json、data/std/README.md")
# ---- 其余：d
hpbest = {}
for r in J("extra/hp.json"):
    if "test" in (r.get("pa"), r.get("pb")) and r.get("pa") != r.get("pb") and "ncc_raw" in r:
        t, o = (r["a"], r["b"]) if r["pa"] == "test" else (r["b"], r["a"])
        if r["ncc_raw"] > hpbest.get((r["ds"], t), {}).get("ncc_raw", -9):
            hpbest[(r["ds"], t)] = dict(r, other=o, other_part=part[(r["ds"], o)])
for ds, ids in tests.items():
    for t in ids:
        if (ds, t) in rec:
            continue
        ev = generic(ds, t)
        if ds == "mouse_embryo" and t in pmap:
            c, o, p = pmap[t]
            ev += f"；三维部分重叠复核最高 {c}（{o}，{p}）"
        if (ds, t) in hpbest:
            h = hpbest[(ds, t)]
            ev += (f"；三维最佳平移下与 {h['other']}（{h['other_part']}）原图相关 {h['ncc_raw']}、高通（σ=1）后 {h['ncc_hp1']}、"
                   f"重叠区标注逐核 F1 {h['lab_f1']}" + ("（这一阶段每个胚胎只有 1–2 个核，F1 不能说明是同一标注）" if ds == "bbbc050" else "")
                   + "，非逐像素相同，属形态相似")
        if ds == "mouse_embryo" and t == "E9_Ab3":
            ev += "；与训练体积 E9_Ab1 编号前缀相同（含义不明），三维带掩码互相关 0.428，未见重叠"
        if ds == "blastospim":
            ev += "；BlastoSPIM2 每个编号只有一帧，最像的训练/验证体积块平均网格相关 ≤ 0.45"
        if ds == "nisnet_zf":
            ev += "；裁块在原图里定位，属 Embryo3（原图 elifeData/C2-wtstk1.tif），训练/验证只有 Embryo1、Embryo2，没有重叠"
        if ds == "celegans":
            d8 = re.search(r"_(\d{6})\d$", t)
            same = [r["id"] for r in rows if r["ds"] == "celegans" and r["split"] == "train" and d8 and re.search(r"_(\d{6})\d$", r["id"])
                    and re.search(r"_(\d{6})\d$", r["id"]).group(1) == d8.group(1)]
            if same:
                ev += f"；与训练/验证个体 {','.join(same)} 同日成像（按编号推断），但为不同个体"
        if ds == "bbbc032":
            ev = "该数据集只有这一个体积（无训练/验证可比）；评测已停用"
        put(ds, t, "d", "—", "—", ev, "leak/pairs、leak/contain、leak/extra")
# ---- 输出
assert all((ds, t) in rec for ds, ids in tests.items() for t in ids)
gt = {ds: ids for ds, ids in tests.items() if ds != "bbbc032"}                 # gt 协议的 227 个测试体积（13 套）
assert sum(len(v) for v in gt.values()) == 227, sum(len(v) for v in gt.values())
cnt = {ds: Counter(rec[(ds, t)]["cat"] for t in ids) for ds, ids in tests.items()}
key = lambda ds, t: [ds, t]
clean = sorted(key(ds, t) for ds, ids in gt.items() for t in ids if rec[(ds, t)]["cat"] == "d")
json.dump(dict(
    description="r9 泄漏审计建议的干净测试子集：只保留 (d) 类（未发现与任何训练/验证体积重复、重叠或同序列关系）的测试体积。"
                "范围是 gt 协议的全部 227 个测试体积（13 套，不含已停用的 bbbc032）。只依据图像、标注与清单的审计结果，没有看任何模型分数。",
    key_format="[数据集, 体积 id]，体积 id 与评测结果 masks/<数据集>__<体积 id>.npz 的文件名一致，也就是 test_r6.py 里 out[(ds, vid)] 的键",
    categories={"a": "像素级重复，或与训练/验证体积互为原样裁块（含线性缩放后相同）",
                "b": "与训练/验证体积同一标本或同一原始体积，但空间上不相交",
                "c": "与训练/验证体积同一时间序列（相邻帧或同一胚胎的不同时间点）",
                "d": "未发现重复、重叠或同序列关系"},
    clean=clean, n_clean=len(clean),
    kept_excluded_by_ds={ds: dict(n=len(ids), kept=cnt[ds]["d"], excluded=len(ids) - cnt[ds]["d"],
                                  a=cnt[ds]["a"], b=cnt[ds]["b"], c=cnt[ds]["c"]) for ds, ids in sorted(gt.items())},
    optional_not_included={
        "b_same_specimen_spatially_disjoint": sorted(key(ds, t) for ds, ids in gt.items() for t in ids if rec[(ds, t)]["cat"] == "b"),
        "c_same_time_series": sorted(key(ds, t) for ds, ids in gt.items() for t in ids if rec[(ds, t)]["cat"] == "c")},
    excluded_a=sorted(key(ds, t) for ds, ids in gt.items() for t in ids if rec[(ds, t)]["cat"] == "a"),
    all_227=[dict(key=key(ds, t), **rec[(ds, t)]) for ds in sorted(gt) for t in sorted(gt[ds])],
    not_in_gt_protocol={"bbbc032|BMP4blastocyst": rec[("bbbc032", "BMP4blastocyst")]},
), open(OUTJ, "w"), indent=1, ensure_ascii=False)
with open(OUTMD, "w") as f:
    f.write("| 数据集 | 测试体积 | (a) 重复/包含 | (b) 同标本不相交 | (c) 同序列 | (d) 未发现 | 建议保留 | 排除 |\n|---|---|---|---|---|---|---|---|\n")
    tot = Counter()
    for ds in sorted(gt):
        c = cnt[ds]
        tot.update(dict(n=len(gt[ds]), a=c["a"], b=c["b"], c=c["c"], d=c["d"]))
        f.write(f"| {ds} | {len(gt[ds])} | {c['a']} | {c['b']} | {c['c']} | {c['d']} | {c['d']} | {len(gt[ds]) - c['d']} |\n")
    f.write(f"| 合计（13 套） | {tot['n']} | {tot['a']} | {tot['b']} | {tot['c']} | {tot['d']} | {tot['d']} | {tot['n'] - tot['d']} |\n\n")
    f.write("bbbc032 的唯一体积 BMP4blastocyst 属 (d)（该数据集没有训练/验证体积可比），评测已停用，不在上表与 clean 里。\n\n")
    for ds in sorted(gt):
        f.write(f"#### {ds}\n\n| 测试体积 | 类别 | 对应的训练/验证体积 | 所在划分 | 证据 | 出处 |\n|---|---|---|---|---|---|\n")
        for t in sorted(gt[ds]):
            v = rec[(ds, t)]
            partner = v["partner"] if len(v["partner"]) < 120 else v["partner"][:117] + "…"
            f.write(f"| {t} | {v['cat']} | {partner} | {v['partner_part']} | {v['evidence']} | {v['source']} |\n")
        f.write("\n")
print(json.dumps({ds: dict(c) for ds, c in cnt.items()}, ensure_ascii=False), "clean", len(clean))
