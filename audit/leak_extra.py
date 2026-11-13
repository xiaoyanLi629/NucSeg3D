"""r9 泄漏审计的定点复核（接 leak_audit.py 的结果；只读图像、标签、清单与 raw 原图，不读任何模型输出，不用 GPU，单进程）。
用法：python leak_extra.py <步骤> ...，结果写到 $NUCSEG3D_SCRATCH/r9/leak/extra/<步骤>.json
  tapenade   129 个“训练/验证块 ⊂ 测试体积”的命中逐个原分辨率复核（逐像素是否相同、相关），并比较两份标注；
             annotator2/square.tif 的结构（各帧之间的相关）；训练/验证块彼此之间的重叠。
  mouse      E7_Ab1 ⊂ E5_Ab1 原分辨率复核与标注比较；E5_Ab2 / E7_Ab2 标注比较；投影重叠最高的跨划分对做三维带掩码互相关。
  nis3d      xL75 与 xR25 和 raw 原图逐层比对：左块是否等于原图前 W 列、右块是否等于其余列。
  partial    bbbc050、celegans、cellos、nucverse3d、gonuclear 投影重叠最高的跨划分对做三维带掩码互相关（部分重叠复核）。
  purdue     阳性对照：两个训练子体积 BABB-383-350 与 BABB-405-237 在找到的平移下重叠区是否逐像素相同。
  blasto2    BlastoSPIM2 的 8 个测试体积在 train/val 里最像的体积。
  hp         bbbc050、celegans 高相关对的鉴别：最佳平移下的原图相关、高通后相关、逐像素是否相同、重叠区标注匹配，附训练-训练参照对。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leak_audit import (OUT, RAW, STD, cross, down, drop_cache, load_fp, manifest, masked_best, match_labels,  # noqa: E402
                        read_vol)

EX = f"{OUT}/extra"
ROWS = {(r["ds"], r["id"]): r for r in manifest()}


def img(ds, i):
    return read_vol(f"{STD}/{ROWS[(ds, i)]['image']}")


def lab(ds, i):
    return read_vol(f"{STD}/{ROWS[(ds, i)]['mask']}", None)


def corr(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def crop_compare(ds, small, big, pos, search=0):
    """small 在 big 里的位置 pos（z,y,x）；search > 0 时在 ±search 范围内找相关最高的整数位置。返回图像与标注比较。"""
    S, Bv = img(ds, small), img(ds, big)
    best = None
    rng = range(-search, search + 1)
    for dz in rng:
        for dy in rng:
            for dx in rng:
                z0, y0, x0 = pos[0] + dz, pos[1] + dy, pos[2] + dx
                if min(z0, y0, x0) < 0 or z0 + S.shape[0] > Bv.shape[0] or y0 + S.shape[1] > Bv.shape[1] or x0 + S.shape[2] > Bv.shape[2]:
                    continue
                c = corr(S, Bv[z0:z0 + S.shape[0], y0:y0 + S.shape[1], x0:x0 + S.shape[2]])
                if best is None or c > best[0]:
                    best = (c, (z0, y0, x0))
    c, (z0, y0, x0) = best
    sl = (slice(z0, z0 + S.shape[0]), slice(y0, y0 + S.shape[1]), slice(x0, x0 + S.shape[2]))
    Bc = Bv[sl]
    A = np.vstack([Bc.ravel(), np.ones(Bc.size)]).T.astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, S.ravel().astype(np.float64), rcond=None)
    resid = S.ravel() - A @ coef
    rec = dict(small=small, big=big, ps=ROWS[(ds, small)]["part"], pb=ROWS[(ds, big)]["part"], pos=[int(z0), int(y0), int(x0)],
               corr=round(c, 6), identical=bool(np.array_equal(S, Bc)), linear_fit=[float(coef[0]), float(coef[1])],
               linear_resid_rms_over_std=float(resid.std() / max(S.std(), 1e-9)))
    Ls, Lb = lab(ds, small), lab(ds, big)[sl]
    rec.update({f"lab_{k}": v for k, v in match_labels(Ls, Lb).items()})
    rec["lab_identical_after_relabel"] = bool(rec["lab_f1"] == 1.0 and rec["lab_mean_iou_matched"] == 1.0)
    return rec


def masked3d(ds, a, b, f):
    A, B = img(ds, a), img(ds, b)
    A, B = (down(A, f), down(B, f)) if f != 1 else (A, B)
    c, k = masked_best(A, B, 0.25)
    return dict(a=a, pa=ROWS[(ds, a)]["part"], b=b, pb=ROWS[(ds, b)]["part"], factor=f, masked3d_ncc=round(c, 4), index=k)


def step_tapenade():
    hits = [r for r in json.load(open(f"{OUT}/contain/tapenade_3d.json")) if (r.get("ncc_fine") or 0) >= 0.99]
    out = dict(hits=[])
    for r in hits:
        rec = crop_compare("tapenade", r["small"], r["big"], r["pos_fine"])
        out["hits"].append(rec)
    h = out["hits"]
    out["summary"] = dict(n=len(h), identical=sum(x["identical"] for x in h), corr_min=min(x["corr"] for x in h),
                          lab_f1_min=min(x["lab_f1"] for x in h), lab_f1_median=float(np.median([x["lab_f1"] for x in h])),
                          lab_identical=sum(x["lab_identical_after_relabel"] for x in h),
                          by_test={t: sum(1 for x in h if x["big"] == t) for t in sorted({x["big"] for x in h})})
    # annotator2/square.tif 的结构与各帧之间的相关
    import tifffile
    p = f"{RAW}/tapenade/stardist_training_datasets/annotator2/square.tif"
    with tifffile.TiffFile(p) as t:
        out["square_tif"] = dict(shape=list(t.series[0].shape), axes=t.series[0].axes, imagej=bool(t.is_imagej),
                                 imagej_meta={k: v for k, v in (t.imagej_metadata or {}).items() if k in ("frames", "slices", "channels", "finterval", "spacing", "unit")})
    drop_cache(p)
    sq = [f"annotator2_square{k:02d}" for k in range(17)]
    vols = {s: img("tapenade", s) for s in sq}
    out["square_consecutive_corr"] = [round(corr(vols[sq[k]], vols[sq[k + 1]]), 4) for k in range(16)]
    out["square00_vs_others_corr"] = [round(corr(vols[sq[0]], vols[s]), 4) for s in sq[1:]]
    json.dump(out, open(f"{EX}/tapenade.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["summary"], ensure_ascii=False), json.dumps(out["square_tif"], ensure_ascii=False), out["square_consecutive_corr"], flush=True)


def step_mouse():
    out = {}
    m3 = json.load(open(f"{OUT}/contain/mouse_embryo_3d.json"))
    r = next(x for x in m3 if x["small"] == "E7_Ab1" and x["big"] == "E5_Ab1")
    pos = [r["pos_fine"][0], 2 * r["pos_fine"][1], 2 * r["pos_fine"][2]]           # 复核因子 (1,2,2) → 原分辨率
    out["E7_Ab1_in_E5_Ab1"] = crop_compare("mouse_embryo", "E7_Ab1", "E5_Ab1", pos, search=3)
    print(json.dumps(out["E7_Ab1_in_E5_Ab1"], ensure_ascii=False), flush=True)
    out["E5_Ab2_vs_E7_Ab2"] = crop_compare("mouse_embryo", "E7_Ab2", "E5_Ab2", [0, 0, 0])
    print(json.dumps(out["E5_Ab2_vs_E7_Ab2"], ensure_ascii=False), flush=True)
    mp = json.load(open(f"{OUT}/contain/mouse_embryo_mip.json"))["pairs"]
    top = [p for p in mp if not (p["same_shape"] and p["mip_overlap_ncc"] > 0.99)][:12]
    extra = [("E9_Ab3", "E9_Ab1"), ("E7_Ab1", "E7_Ab2"), ("E7_Ab1", "E5_Ab2")]
    out["partial3d"] = [masked3d("mouse_embryo", p["a"], p["b"], (1, 4, 4)) for p in top] + \
                       [masked3d("mouse_embryo", a, b, (1, 4, 4)) for a, b in extra]
    for x in out["partial3d"]:
        print("   ", json.dumps(x, ensure_ascii=False), flush=True)
    json.dump(out, open(f"{EX}/mouse.json", "w"), indent=1, ensure_ascii=False)


def step_nis3d():
    import tifffile
    out = []
    rows = [r for r in ROWS.values() if r["ds"] == "nis3d" and r["id"].endswith("_xL75")]
    for L in rows:
        base = L["id"][:-5]
        R = ROWS[("nis3d", base + "_xR25")]
        rawp = f"{RAW}/{L['source'][4:]}"
        with tifffile.TiffFile(f"{STD}/{L['image']}") as tl, tifffile.TiffFile(f"{STD}/{R['image']}") as tr, tifffile.TiffFile(rawp) as tw:
            Z = len(tl.pages)
            WL = tl.pages[0].shape[-1]
            rawz = tw.series[0].shape
            eqL = eqR = True; dmaxL = dmaxR = 0.0; ok_shape = True
            for z in range(Z):
                a = tl.pages[z].asarray(); b = tr.pages[z].asarray(); w = tw.pages[z].asarray() if len(tw.pages) == Z else None
                if w is None:
                    ok_shape = False; break
                wl, wr = w[:, :WL], w[:, WL:]
                if wl.shape != a.shape or wr.shape != b.shape:
                    ok_shape = False; break
                dL = np.abs(a.astype(np.float64) - wl.astype(np.float64)).max(); dR = np.abs(b.astype(np.float64) - wr.astype(np.float64)).max()
                dmaxL = max(dmaxL, float(dL)); dmaxR = max(dmaxR, float(dR))
                eqL &= bool(dL == 0); eqR &= bool(dR == 0)
        for p in (f"{STD}/{L['image']}", f"{STD}/{R['image']}", rawp):
            drop_cache(p)
        rec = dict(volume=base, L=L["id"], L_part=L["part"], R=R["id"], R_part=R["part"], raw_shape=list(rawz), L_width=int(WL),
                   shapes_consistent=ok_shape, L_equals_raw_first_cols=eqL, R_equals_raw_remaining_cols=eqR, max_abs_diff_L=dmaxL, max_abs_diff_R=dmaxR)
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    json.dump(out, open(f"{EX}/nis3d.json", "w"), indent=1, ensure_ascii=False)


def step_partial():
    plan = {"bbbc050": (20, 1), "celegans": (6, (2, 4, 4)), "cellos": (6, (2, 4, 4)), "nucverse3d": (4, (1, 4, 4))}
    out = {}
    for ds, (k, f) in plan.items():
        mp = json.load(open(f"{OUT}/contain/{ds}_mip.json"))["pairs"][:k]
        out[ds] = [dict(masked3d(ds, p["a"], p["b"], f), mip_overlap_ncc=p["mip_overlap_ncc"]) for p in mp]
        print(ds, [(x["a"], x["pa"], x["b"], x["pb"], x["mip_overlap_ncc"], x["masked3d_ncc"]) for x in out[ds]], flush=True)
    # gonuclear：体素记录有微小差异，所以 contain 步骤没有比；这里在投影上比（xy 各 1/4）
    meta, z = load_fp("gonuclear")
    ids = [m["id"] for m in meta]
    mips = {i: down(np.asarray(z[f"{i}|mip"], np.float32), 4) for i in ids}
    g = []
    for a_ in range(len(ids)):
        for b_ in range(a_ + 1, len(ids)):
            if cross(meta[a_]["part"], meta[b_]["part"]):
                c, kk = masked_best(mips[ids[a_]], mips[ids[b_]], 0.25)
                g.append(dict(a=ids[a_], pa=meta[a_]["part"], b=ids[b_], pb=meta[b_]["part"], mip_overlap_ncc=round(c, 4)))
    out["gonuclear_mip"] = sorted(g, key=lambda x: -x["mip_overlap_ncc"])
    print("gonuclear", out["gonuclear_mip"][:6], flush=True)
    json.dump(out, open(f"{EX}/partial.json", "w"), indent=1, ensure_ascii=False)


def step_purdue():
    """阳性对照：masked NCC 在两个训练子体积之间找到 NCC 1.0 的平移；在该平移（两种符号约定）±4 体素内找重叠区逐像素相同的平移。"""
    a, b = "BABB-cleared_kidney_1-383-350", "BABB-cleared_kidney_1-405-237"
    A, B = img("purdue", a), img("purdue", b)
    rec = next(r for r in json.load(open(f"{OUT}/coords/coords.json"))["purdue"] if {r["a"], r["b"]} == {a, b})
    s0 = rec["shift_zyx"]
    best = None
    cands = sorted({(sg * s0[0] + i, sg * s0[1] + j, sg * s0[2] + k) for sg in (1, -1)
                    for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)})
    for dz, dy, dx in cands:
        za, zb = max(0, dz), max(0, -dz); ya, yb = max(0, dy), max(0, -dy); xa, xb = max(0, dx), max(0, -dx)
        nz, ny, nx = min(A.shape[0] - za, B.shape[0] - zb), min(A.shape[1] - ya, B.shape[1] - yb), min(A.shape[2] - xa, B.shape[2] - xb)
        if nz <= 0 or ny < 8 or nx < 8:
            continue
        if np.array_equal(A[za:za + nz, ya:ya + ny, xa:xa + nx], B[zb:zb + nz, yb:yb + ny, xb:xb + nx]):
            v = nz * ny * nx
            if best is None or v > best[0]:
                best = (v, (dz, dy, dx), (nz, ny, nx))
    out = dict(a=a, b=b, masked_ncc_shift=s0, found_identical_overlap=best is not None)
    if best:
        out.update(shift_zyx=list(best[1]), overlap_shape=list(best[2]), overlap_frac=round(best[0] / A.size, 4),
                   note="A[max(0,dz):, max(0,dy):, max(0,dx):] 与 B[max(0,-dz):, ...] 的重叠区逐像素相同")
    print(json.dumps(out, ensure_ascii=False), flush=True)
    json.dump(out, open(f"{EX}/purdue.json", "w"), indent=1, ensure_ascii=False)


def step_blasto2():
    P = json.load(open(f"{OUT}/pairs/blastospim.json"))
    out = {t: nb for t, nb in P["nearest_test"].items() if t.startswith("BlastoSPIM2")}
    for t, nb in out.items():
        print(t, json.dumps(nb, ensure_ascii=False), flush=True)
    json.dump(out, open(f"{EX}/blasto2.json", "w"), indent=1, ensure_ascii=False)


def overlap_views(shA, shB, s):
    """平移 s（B 相对 A，约定同 coords：A[max(0,s):] 对 B[max(0,-s):]）下两者的重叠切片。"""
    sa, sb = [], []
    for ax in range(3):
        a0 = max(0, s[ax]); b0 = max(0, -s[ax])
        n = min(shA[ax] - a0, shB[ax] - b0)
        if n <= 0:
            return None
        sa.append(slice(a0, a0 + n)); sb.append(slice(b0, b0 + n))
    return tuple(sa), tuple(sb)


def hpf(v, sig):
    from scipy.ndimage import gaussian_filter
    v = v.astype(np.float32)
    return v - gaussian_filter(v, sig)


def pair_detail(ds, a, b, f):
    """先在因子 f 降采样后做三维带掩码互相关找平移，再在原分辨率 ±f 内细化（重叠 ≥ 25%），
    在最佳平移下比较：原图相关、高通（减去 σ=1、3 的高斯模糊）后的相关、是否逐像素相同，以及重叠区内两份标注的逐核匹配。
    同一图像的重复在高通后仍接近 1、标注 F1 接近 1；不同个体只是轮廓相似时高通相关会明显下降。"""
    A, B = img(ds, a), img(ds, b)
    fs = (f,) * 3 if np.isscalar(f) else tuple(f)
    Ad, Bd = (A, B) if fs == (1, 1, 1) else (down(A, fs), down(B, fs))
    c0, k = masked_best(Ad, Bd, 0.25)
    s0 = [(int(k[ax]) - (Bd.shape[ax] - 1)) * fs[ax] for ax in range(3)]
    best = None
    rz, ry, rx = (range(-q + 1 if q > 1 else 0, q if q > 1 else 1) for q in fs)
    for dz in rz:
        for dy in ry:
            for dx in rx:
                sft = [s0[0] + dz, s0[1] + dy, s0[2] + dx]
                v = overlap_views(A.shape, B.shape, sft)
                if v is None or A[v[0]].size < 0.25 * min(A.size, B.size):
                    continue
                c = corr(A[v[0]], B[v[1]])
                if best is None or c > best[0]:
                    best = (c, sft, v)
    if best is None:
        return dict(ds=ds, a=a, b=b, note="没有满足重叠 ≥ 25% 的平移")
    c, sft, (va, vb) = best
    oa, ob = A[va], B[vb]
    rec = dict(ds=ds, a=a, pa=ROWS[(ds, a)]["part"], b=b, pb=ROWS[(ds, b)]["part"], coarse_masked_ncc=round(c0, 4), factor=list(fs),
               shift=[int(x) for x in sft], overlap_frac=round(oa.size / min(A.size, B.size), 3), ncc_raw=round(c, 4),
               ncc_hp1=round(corr(hpf(oa, 1), hpf(ob, 1)), 4), ncc_hp3=round(corr(hpf(oa, 3), hpf(ob, 3)), 4),
               identical=bool(np.array_equal(oa, ob)))
    la, lb = lab(ds, a)[va], lab(ds, b)[vb]
    rec.update({f"lab_{kk}": vv for kk, vv in match_labels(la, lb).items()})
    return rec


def step_hp():
    """bbbc050 与 celegans 的高相关是真重复还是形态相似：跨划分投影重叠最高的对，加同一胚胎不同时间、不同个体的训练-训练对作参照。"""
    P3 = json.load(open(f"{EX}/partial.json"))
    plan = []
    for p in P3["bbbc050"][:10]:
        plan.append(("bbbc050", p["a"], p["b"], 1, "跨划分，投影重叠最高"))
    for a, b in (("train_Emb01_t051", "train_Emb01_t101"), ("train_Emb03_t101", "train_Emb03_t151"), ("train_Emb02_t101", "train_Emb03_t101"),
                 ("train_Emb09_t101", "train_Emb10_t101"), ("train_Emb01_t051", "train_Emb04_t051")):
        plan.append(("bbbc050", a, b, 1, "参照：训练-训练（同一胚胎相隔 50 帧，或不同胚胎）"))
    for p in P3["celegans"][:6]:
        plan.append(("celegans", p["a"], p["b"], (2, 4, 4), "跨划分，投影重叠最高"))
    for a, b in (("cnd1threeL1_1229061", "cnd1threeL1_1229062"), ("hlh1fourL1_0417076", "hlh1fourL1_0417077"), ("elt3L1_0503072", "unc54L1_0123071")):
        plan.append(("celegans", a, b, (2, 4, 4), "参照：训练-训练（同日不同个体，或不同日不同个体）"))
    out = []
    for ds, a, b, f, why in plan:
        rec = pair_detail(ds, a, b, f); rec["role"] = why
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    json.dump(out, open(f"{EX}/hp.json", "w"), indent=1, ensure_ascii=False)


def step_b1adj():
    """只看 BlastoSPIM1：同一编号内按时间排序的相邻帧，块平均网格相关按划分组合分开统计（BlastoSPIM2 每个编号只有一帧，不算）。"""
    import re
    meta, z = load_fp("blastospim")
    ids = [m["id"] for m in meta]
    pt = {m["id"]: m["part"] for m in meta}
    shp = {m["id"]: tuple(m["shape"]) for m in meta}
    key = {}
    for i in ids:
        m = re.match(r"^(BlastoSPIM1)_(?:train|val|test)(?:_[a-z]+SNR)?_(.+)_(\d+)_image_\d+$", i)
        if m:
            key[i] = (m.group(2), int(m.group(3)))
    by = {}
    for i, (k, t) in key.items():
        by.setdefault(k, []).append((t, i))
    acc = {}
    for k, L in by.items():
        L.sort()
        for (t1, i), (t2, j) in zip(L, L[1:]):
            if shp[i] != shp[j]:
                continue
            a = np.asarray(z[f"{i}|hi"], np.float32).ravel(); b = np.asarray(z[f"{j}|hi"], np.float32).ravel()
            acc.setdefault("-".join(sorted([pt[i], pt[j]])), []).append((t2 - t1, float(a @ b / a.size)))
    out = {k: dict(n=len(v), dt_median=float(np.median([d for d, _ in v])), c_min=round(min(c for _, c in v), 4),
                   c_median=round(float(np.median([c for _, c in v])), 4), c_max=round(max(c for _, c in v), 4)) for k, v in acc.items()}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    json.dump(out, open(f"{EX}/b1adj.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    os.makedirs(EX, exist_ok=True)
    for s in sys.argv[1:]:
        dict(tapenade=step_tapenade, mouse=step_mouse, nis3d=step_nis3d, partial=step_partial, purdue=step_purdue, blasto2=step_blasto2,
             hp=step_hp, b1adj=step_b1adj)[s]()
