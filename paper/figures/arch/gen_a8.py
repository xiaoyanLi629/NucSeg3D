"""Fig. 1 v8 (A8-b): information-dense S1 architecture (panel a) in the A7-b visual language (stage containers,
cuboids, stream colours, 3D-viewer warm-grey panes), plus a training-schedule strip; panels b-d from gen_a7 (A7-b style).
Every drawn fact is listed with its source in A8-b.provenance.json (main.tex Methods / Results, code references).
Real-data assets: existing renders; assets8/out_*.png = raw S1 outputs on one xz section (make_a8_assets.py).
Usage: python gen_a8.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fig import FS, LW, PAL, Fig, export, tint, M as PM  # noqa: E402
from gen_a3 import DEP, tex  # noqa: E402
from gen_a4 import REFS as REFS4, DF, hatch_defs, num  # noqa: E402
import gen_a7 as G  # noqa: E402
from gen_a7 import (T, card, cub, block, arr, funnel, slab7, inset, legend7, stream_legend, panel_cmp7, panel_target7,  # noqa: E402
                    panel_infer7, F_CARD, F_LAB, F_NOTE, MUTE, AW, C2D, C3D, S2D, S3D, BASE, RED, LOSS, REFS7, REFS6, REF_IMG)

HERE = os.path.dirname(os.path.abspath(__file__))
INK = PAL["fg"]
PT = 0.3528
G.FAM["IV"] = ("#F0F0EE", "#E3E2DD", "#55565C")
ND = PAL["new_d"]

REFS8 = [
    ("MA-SAM (arXiv 2309.08842) Fig. 1", "frozen vs trainable parts of the 2D encoder distinguished; caption names the reshape between [B,N,H/16,W/16,c] and [B,c,N,H/16,W/16] -> A8: explicit 'Z planes -> batch' and re-stack shapes on the arrows"),
    ("SAM-Med3D (arXiv 2310.15161) Fig. 3", "module families labelled with their 3D components (3D PE, 3D conv, LayerNorm) -> A8: kernel shapes written inside modules"),
    ("SAM 2 (arXiv 2408.00714) Fig. 3", "per-frame processing drawn as a stream through shared modules; caption states what is omitted -> A8: per-plane encoder with shared weights, omitted details named in the caption"),
    ("UNETR (arXiv 2103.10504) Fig. 2", "output sizes written on the diagram (caption: 'Output sizes are given for P=16 and K=768'); transformer features reshaped back to 3D -> A8: tensor shape under every tensor glyph"),
    ("Swin-UNETR (arXiv 2201.01266) Figs. 1-2", "input split into non-overlapping patches; sub-mechanism moved to its own panel -> A8: patch-grid plane with a magnified 8x8 patch; zoom insets"),
    ("nnFormer (arXiv 2109.03201) Fig. 2", "overall U-shape plus zoomed sub-panels (b-e) annotated with kernel K, stride S, norm -> A8: residual-block inset with GN / conv 3^3 / dilation (d,1,1)"),
    ("3DSAM-adapter (arXiv 2306.13465) Figs. 1-2", "volumetric adaptation of a 2D ViT, adapter detail in a separate zoom -> A8: new 3D modules in their own colour family"),
    ("MedSAM2 (arXiv 2504.03600) Fig. 1", "real example data (2D slices and 3D renders) placed next to the architecture -> A8: real input planes, real S1 output maps, real target renders inside the architecture panel"),
    ("Cellpose-SAM (bioRxiv 2025.04.28.651001) Fig. 1", "previous architecture, new architecture, training stages and training examples combined in one figure -> A8: training-schedule strip (IV) in the same panel"),
    ("Cellpose3 (bioRxiv 2024.02.10.579780) Fig. 1e", "trainable network chained to a fixed network with the losses listed as boxes -> A8: loss box fed by output maps and the 3D target"),
    ("CellSAM (bioRxiv 2023.11.17.567630) Fig. 1", "two-stage training with frozen parts stated in the figure -> A8: frozen patch embedding hatched; stage-wise learning rates in the schedule strip"),
    ("StarDist-3D (arXiv 1908.03636) Fig. 1", "small geometric glyphs (rays, polyhedra) beside the network diagram -> A8: 3x3x3 voxel-kernel glyphs spread along z with receptive-field brackets"),
    ("note", "techniques were taken from captions and surrounding text read via WebFetch (figure images not inspected by the survey agent); Segment Anything for Microscopy, PlantSeg and EmbedSeg were also read but put architecture in supplementary figures or show pipelines with real data rather than network diagrams"),
]


# ======================================================================== glyphs
def plane_stack(f, x, y, w, L, n, pitch, img):
    """Staggered stack of n horizontal xy planes (front edge at y + k*pitch), textured with a real plane."""
    dx, dy = L * DEP[0], L * DEP[1]
    for k in range(n - 1, -1, -1):
        yk = y + k * pitch
        tex(f, img, w, 0, -dx, -dy, x + dx, yk + dy)
        f.poly([(x, yk), (x + w, yk), (x + w + dx, yk + dy), (x + dx, yk + dy)], "none", C2D["st"], 0.6 * LW, chk=False)
    return x + w + dx


def patch_plane(f, x, y, s, cell=(10, 20)):
    f.image(x, y, s, s, "in_xy.png", fit="none")
    c = s / 32
    lines = "".join(f"M{x + k * c:.2f},{y:.2f} V{y + s:.2f} M{x:.2f},{y + k * c:.2f} H{x + s:.2f} " for k in range(1, 32))
    f.add(f'<path d="{lines}" stroke="#ffffff" stroke-width="{0.5 * LW:.3f}" opacity="0.45" fill="none"/>')
    f.rect(x, y, s, s, rx=0, fill="none", stroke=C2D["st"], sw=0.6 * LW, chk=False)
    r, q = cell
    hx, hy = x + q * c, y + r * c
    f.rect(hx, hy, c, c, rx=0, fill="none", stroke=RED, sw=0.8 * LW, chk=False)
    return hx, hy, c


def magnifier(f, x, y, s, src):
    hx, hy, c = src
    f.rect(x, y, s, s, rx=0, fill="#ffffff", stroke=RED, sw=0.7 * LW, chk=False)
    g = s / 8
    cells = "".join(f'<rect x="{x + i * g:.2f}" y="{y + j * g:.2f}" width="{g:.3f}" height="{g:.3f}" fill="{tint(BASE, 0.15 + 0.55 * ((i * 3 + j * 5) % 8) / 8)}"/>'
                    for i in range(8) for j in range(8))
    f.add(cells)
    f.rect(x, y, s, s, rx=0, fill="none", stroke=RED, sw=0.7 * LW, chk=False)
    for (a, b), (c_, d) in (((hx, hy), (x, y + s)), ((hx + c, hy), (x + s, y + s))):
        f.path(f"M{a:.2f},{b:.2f} L{c_:.2f},{d:.2f}", stroke=RED, arrow=False, dash="0.5 0.4", chk=False, sw=0.5 * LW)


def token_grid(f, x, y, s, L):
    dx, dy = L * DEP[0], L * DEP[1]
    f.poly([(x, y), (x + s, y), (x + s + dx, y + dy), (x + dx, y + dy)], C2D["top"], C2D["st"], 0.7 * LW, chk=False)
    f.poly([(x + s, y), (x + s + dx, y + dy), (x + s + dx, y + s + dy), (x + s, y + s)], C2D["side"], C2D["st"], 0.7 * LW, chk=False)
    f.rect(x, y, s, s, rx=0, fill=C2D["front"], stroke=C2D["st"], sw=0.7 * LW, chk=False)
    g = s / 8
    d_ = "".join(f"M{x + k * g:.2f},{y:.2f} V{y + s:.2f} M{x:.2f},{y + k * g:.2f} H{x + s:.2f} " for k in range(1, 8))
    f.add(f'<path d="{d_}" stroke="{C2D["st"]}" stroke-width="{0.5 * LW:.3f}" opacity="0.7" fill="none"/>')
    for k in range(1, 6):                       # channel slices on the side face
        t = k / 6
        f.add(f'<line x1="{x + s + dx * t:.2f}" y1="{y + dy * t:.2f}" x2="{x + s + dx * t:.2f}" y2="{y + s + dy * t:.2f}" stroke="{C2D["st"]}" stroke-width="{0.5 * LW:.3f}" opacity="0.6"/>')
    return x + s + dx


def attention_motif(f, x, y, s, n=6, q=(2, 3)):
    g = s / n
    f.rect(x, y, s, s, rx=0.3, fill="#ffffff", stroke=C2D["st"], sw=0.5 * LW, chk=False)
    qx, qy = x + (q[1] + 0.5) * g, y + (q[0] + 0.5) * g
    ls = "".join(f"M{qx:.2f},{qy:.2f} L{x + (j + 0.5) * g:.2f},{y + (i + 0.5) * g:.2f} " for i in range(n) for j in range(n) if (i, j) != q)
    f.add(f'<path d="{ls}" stroke="{S2D}" stroke-width="{0.5 * LW:.3f}" opacity="0.45" fill="none"/>')
    for i in range(n):
        for j in range(n):
            col = RED if (i, j) == q else tint(BASE, 0.35)
            f.circle(x + (j + 0.5) * g, y + (i + 0.5) * g, g * 0.18 if (i, j) != q else g * 0.3, fill=col, chk=False)


def voxel_kernel(f, cx, ymid, d, pz, nz=39):
    """z column of nz planes (faint) with the three 3x3 taps of a conv 3^3 dilated (d,1,1) along z; returns RF bracket x."""
    w, dp = 4.6, 2.0
    dx, dy = dp * DEP[0], dp * DEP[1]
    x0 = cx - (w + dx) / 2
    for q in range(nz - 1, -1, -1):
        off = q - nz // 2
        yy = ymid + off * pz
        f.add(f'<polygon points="{x0:.2f},{yy:.2f} {x0 + w:.2f},{yy:.2f} {x0 + w + dx:.2f},{yy + dy:.2f} {x0 + dx:.2f},{yy + dy:.2f}" '
              f'fill="{tint(BASE, 0.20)}" stroke="none"/>')
    for off in (d, 0, -d):                      # draw lower taps first
        yy = ymid + off * pz
        for i in range(3):
            for j in range(3):
                u0, u1 = i / 3, (i + 1) / 3
                v0, v1 = j / 3, (j + 1) / 3
                P = [(x0 + w * u0 + dx * v0, yy + dy * v0), (x0 + w * u1 + dx * v0, yy + dy * v0),
                     (x0 + w * u1 + dx * v1, yy + dy * v1), (x0 + w * u0 + dx * v1, yy + dy * v1)]
                f.add('<polygon points="' + " ".join(f"{a:.2f},{b:.2f}" for a, b in P) + f'" fill="{RED if (i, j) == (1, 1) else tint(RED, 0.55)}" '
                      f'stroke="#ffffff" stroke-width="{0.5 * LW:.3f}"/>')
    return x0 + w + dx + 0.6


def pixel_shuffle(f, x, cy):
    shades = [tint(RED, 0.15 + 0.7 * k / 7) for k in range(8)]
    sh = 1.15
    y0 = cy - 4 * sh
    for k in range(8):
        f.rect(x, y0 + k * sh, 2.2, sh - 0.12, rx=0.15, fill=shades[k], stroke="none", chk=False)
    f.rect(x - 0.2, y0 - 0.2, 2.6, 8 * sh + 0.28, rx=0.3, fill="none", stroke=C3D["st"], sw=0.5 * LW, chk=False)
    gx = x + 6.4
    arr(f, x + 2.6, cy, gx - 0.3, ND)
    g = 0.95
    gy = cy - 4 * g
    for r in range(8):
        for c in range(8):
            f.add(f'<rect x="{gx + c * g:.2f}" y="{gy + r * g:.2f}" width="{g - 0.08:.3f}" height="{g - 0.08:.3f}" fill="{shades[r]}"/>')
    f.rect(gx - 0.2, gy - 0.2, 8 * g + 0.32, 8 * g + 0.32, rx=0.3, fill="none", stroke=C3D["st"], sw=0.5 * LW, chk=False)
    return gx + 8 * g + 0.2


# ======================================================================== panel a
def panel_arch8(f, ox, oy, w=183.0, letter="a"):
    hatch_defs(f)
    f.plabel(ox, oy + 3.4, letter, "S1 architecture and training")
    legend7(f, ox + 70.0, oy + 3.6)
    # ---------------------------------------------------------------- tier 1: per-plane 2D encoder
    t1y, t1h = oy + 6.2, 35.8
    card(f, ox + 0.5, t1y, w - 1.0, t1h, "I", "I. Per-plane 2D encoder",
         "applied to every plane with shared weights · patch embedding frozen, all other encoder weights fine-tuned")
    cyA = oy + 23.2
    l1, l2, l3, l4 = oy + 33.4, oy + 35.7, oy + 38.0, oy + 40.3
    # input volume (real renders)
    ix, isz, iL = ox + 2.6, 10.0, 5.0
    block(f, ix, cyA - 5.0, isz, isz, iL, "#000", "#000", "#000", C2D["st"], front="../assets3/in_front.png", top="../assets3/in_top.png",
          side="../assets3/in_side.png")
    icx = ix + isz / 2 + iL * DEP[0] / 2
    T(f, icx, l1, "input", F_LAB, 700, a="middle")
    T(f, icx, l2, ["1×Z×256×256", "Z = 32 planes"], F_NOTE, c=MUTE, a="middle")
    iend = ix + isz + iL * DEP[0]
    # staggered xy planes
    px0 = iend + 3.6
    arr(f, iend + 0.2, cyA, px0 - 0.1, S2D)
    pend = plane_stack(f, px0, cyA - 4.2, 9.0, 6.0, 6, 2.0, "../assets/in_xy.png")
    f.path(f"M{px0 - 1.4:.2f},{cyA + 6.6:.2f} V{cyA - 7.0:.2f}", stroke=MUTE, sw=0.6 * LW, chk=False)
    T(f, px0 - 1.4, cyA - 7.9, "z", F_NOTE, c=MUTE, a="middle", italic=True)
    pcx = (px0 + pend) / 2
    T(f, pcx, l1, "Z = 32 xy planes", F_LAB, 700, a="middle")
    T(f, pcx, l2, ["Z×256×256", "→ batch of Z planes"], F_NOTE, c=MUTE, a="middle")
    # one plane with its 8x8-pixel patch grid
    qx0, qs = pend + 3.0, 12.4
    arr(f, pend + 0.2, cyA, qx0 - 0.25, S2D)
    src = patch_plane(f, qx0, cyA - qs / 2, qs)
    magnifier(f, qx0 + qs - 3.6, oy + 12.0, 3.4, src)
    T(f, qx0 + qs + 0.6, oy + 14.6, "8×8 px", F_NOTE, 700, c=ND)
    T(f, qx0 + qs / 2, l1, "one plane", F_LAB, 700, a="middle")
    T(f, qx0 + qs / 2, l2, ["32×32 patches", "of 8×8 px"], F_NOTE, c=MUTE, a="middle")
    # patch embedding (frozen) + position embedding
    ex0, ew0, eh0 = qx0 + qs + 3.0, 8.4, 4.0
    arr(f, qx0 + qs + 0.2, cyA, ex0 - 0.25, S2D)
    cub(f, ex0, cyA - eh0 / 2, ew0, eh0, 1.6, C2D, "patch", size=F_NOTE, hatch=True)
    pyy = cyA + 4.4
    cub(f, ex0, pyy, ew0, eh0, 1.6, C2D, "+ pos.", size=F_NOTE)
    T(f, ex0 + ew0 / 2 + 0.6, cyA - 3.6, "8×8 conv", F_NOTE, c=MUTE, a="middle")
    T(f, ex0 + ew0 / 2, l2 + 0.6, "embedding", F_NOTE, c=MUTE, a="middle")
    eend = ex0 + ew0 + 1.6 * DEP[0]
    qq = eend + 2.0
    f.plus_node(qq, cyA, r=1.0, col=S2D)
    arr(f, eend + 0.2, cyA, qq - 1.15, S2D)
    f.path(f"M{eend + 0.2:.2f},{pyy + eh0 / 2:.2f} H{qq:.2f} V{cyA + 1.25:.2f}", stroke=S2D, sw=1.1 * LW)
    # token tensor
    tx0, ts = qq + 3.0, 9.0
    arr(f, qq + 1.05, cyA, tx0 - 0.25, S2D)
    tend = token_grid(f, tx0, cyA - ts / 2, ts, 6.0)
    T(f, (tx0 + tend) / 2, l1, "tokens", F_LAB, 700, a="middle")
    T(f, (tx0 + tend) / 2, l2, ["32×32×1024", "per plane"], F_NOTE, c=MUTE, a="middle")
    # ViT stack x24
    sx, sw_, bh, bd = tend + 3.0, 20.5, 4.0, 2.4
    arr(f, tend + 0.2, cyA, sx - 0.25, S2D)
    ys = [cyA - 8.2, cyA - 3.6, cyA + 3.8]
    for yy, nm in zip(ys, ["block 1", "block 2", "block 24"]):
        cub(f, sx, yy, sw_, bh, bd, C2D, nm, size=F_LAB)
    T(f, sx + sw_ / 2, cyA + 2.0, "···", FS["mod"], 700, c=C2D["st"], a="middle")
    send = sx + sw_ + bd * DEP[0]
    scx = (sx + send) / 2
    T(f, scx, oy + 12.7, "ViT-L encoder · 304 M · lr 10^{−5}", F_LAB, 700, a="middle")
    T(f, scx, l1, "24 global-attention blocks", F_NOTE, 700, a="middle")
    T(f, scx, l2, "16 heads · rel. position", F_NOTE, c=MUTE, a="middle")
    num(f, sx - 1.6, cyA + 6.6, 1)
    # neck
    nx = send + 2.8
    arr(f, send + 0.2, cyA, nx - 0.25, S2D)
    cub(f, nx, cyA - 5.0, 5.2, 10.0, 2.0, dict(front=tint(BASE, 0.40), top=tint(BASE, 0.22), side=tint(BASE, 0.60), st=C2D["st"]), "neck", size=F_NOTE)
    nend = nx + 5.2 + 2.0 * DEP[0]
    T(f, (nx + nend) / 2, l1, ["1×1 conv", "1024→256,", "3×3 conv, LN"], F_NOTE, c=MUTE, a="middle")
    # per-plane feature map
    fx0, fs0, fL = nend + 2.8, 8.0, 4.0
    arr(f, nend + 0.2, cyA, fx0 - 0.25, S2D)
    for k in (2, 1):
        cub(f, fx0 + k * 1.3, cyA - fs0 / 2 - k * 1.0, fs0, fs0, fL, C2D, op=0.45)
    cub(f, fx0, cyA - fs0 / 2, fs0, fs0, fL, C2D)
    fend = fx0 + fs0 + fL * DEP[0] + 2.6
    T(f, fx0 + fs0 / 2 + 1.8, l1, "features", F_LAB, 700, a="middle")
    T(f, fx0 + fs0 / 2 + 1.8, l2, ["256×32×32", "× Z planes"], F_NOTE, c=MUTE, a="middle")
    f.prov_add("input planes: the 32-plane network window is processed as a batch of Z xy planes (B*Z) by the 2D encoder; texture = real xy plane of mouse embryo E15_Ab2 (assets/in_xy.png)",
               "net3d.py:285-286 (encoder on B*Z planes); make_assets.py in_xy (z = 39, plane with most GT foreground)")
    f.prov_add("patch embedding 8x8 conv, stride 8 -> 32x32 tokens x 1024 per 256x256 plane, weights not trained; absolute position embedding (1,32,32,1024), trained",
               "cellpose vit.py:109-116 (CPSAM); net3d.py:202-205 patch-embed requires_grad False, :228-232 _embed")
    f.prov_add("ViT-L encoder: 24 blocks, width 1024, 16 heads, relative position bias, global attention in every block; 304.4 M parameters, lr 1e-5 (Methods: AdamW learning rate 1e-5 encoder)",
               "segment_anything build_sam.py:27-34, 66-80; cellpose vit.py:118-120; train3d.py:144; main.tex Methods '3D model'")
    f.prov_add("neck: 1x1 conv 1024->256, LayerNorm2d, 3x3 conv 256->256, LayerNorm2d -> 256x32x32 per plane", "segment_anything image_encoder.py:88-104")
    # transformer-block inset (right of tier 1)
    tix, tiy, tiw, tih = fend + 2.4, oy + 11.6, ox + w - 1.6 - (fend + 2.4), 29.4
    inset(f, tix, tiy, tiw, tih, "transformer block (×24)", "#4a5577")
    T(f, tix + tiw - 1.4, tiy + 7.4, "tokens 32×32×1024", F_NOTE, c=MUTE, a="end")
    f.path(f"M{send - 0.2:.2f},{ys[0] - 1.45:.2f} L{tix - 0.1:.2f},{tiy + 1.0:.2f}", stroke=C2D["st"], arrow=False, dash="0.7 0.5", chk=False, sw=0.6 * LW)
    f.path(f"M{send - 0.2:.2f},{ys[2] + bh:.2f} L{send + 1.2:.2f},{ys[2] + bh:.2f}", stroke=C2D["st"], arrow=False, chk=False, sw=0.0001)
    cx0, cw = tix + 4.2, 11.0
    items = [("LN", 2.8), ("MHSA", 3.4), ("+", 0), ("LN", 2.8), ("MLP", 3.4), ("+", 0)]
    yy = tiy + 5.6
    pos = []
    for i, (nm, hh) in enumerate(items):
        if nm == "+":
            f.plus_node(cx0 + cw / 2, yy + 1.0, r=1.0, col=S2D)
            pos.append(yy + 1.0)
            yy += 2.0
        else:
            cub(f, cx0, yy, cw, hh, 0.9, C2D, nm, size=F_NOTE)
            pos.append(yy + hh / 2)
            yy += hh
        if i < 5:
            arr(f, cx0 + cw / 2, yy + 0.1, cx0 + cw / 2, yy + 1.0, S2D)
            yy += 1.1
    xr = cx0 - 1.6
    for a_, b_ in ((tiy + 5.0, pos[2]), (pos[2] + 1.4, pos[5])):
        f.path(f"M{cx0 + cw / 2:.2f},{a_:.2f} H{xr:.2f} V{b_:.2f} H{cx0 + cw / 2 - 1.15:.2f}", stroke=S2D, sw=0.9 * LW)
    ax_ = cx0 + cw + 1.6
    T(f, ax_, pos[1] - 0.4, ["16 heads, global,", "rel. position"], F_NOTE, c=MUTE)
    T(f, ax_, pos[4] - 0.4, ["1024→4096→1024", "GELU"], F_NOTE, c=MUTE)
    ms = 8.6
    mx = tix + tiw - ms - 1.6
    attention_motif(f, mx, tiy + 9.6, ms)
    T(f, mx + ms / 2, tiy + 9.6 + ms + 2.4, ["query token", "↔ all tokens"], F_NOTE, c=MUTE, a="middle")
    f.prov_add("transformer block: x + MHSA(LN(x)); x + MLP(LN(x)); MHSA 16 heads with relative position bias, global (every token attends to all tokens of its plane; motif schematic); MLP 1024->4096->1024 GELU",
               "segment_anything image_encoder.py Block l.151-175; build_sam.py:27-34, 66-80; cellpose vit.py (window_size 0)")

    # ---------------------------------------------------------------- tier 2: decoder (II) and output (III)
    t2y, t2h = t1y + t1h + 2.8, 59.4
    wII = 100.0
    xII, xIII = ox + 0.5, ox + 0.5 + wII + 1.5
    wIII = ox + w - 0.5 - xIII
    card(f, xII, t2y, wII, t2h, "II", "II. 3D residual decoder", "new · 14.3 M parameters · lr 2·10^{−4}")
    card(f, xIII, t2y, wIII, t2h, "III", "III. Output & training target", "target: diffusion in each GT nucleus")
    cy = t2y + 21.2
    ret = t1y + t1h + 1.4
    sx0 = xII + 3.9
    f.path(f"M{fend - 2.4:.2f},{cyA:.2f} H{fend + 0.4:.2f} V{ret:.2f} H{xII + 1.15:.2f} V{cy:.2f} H{sx0 - 0.3:.2f}", stroke=S2D, sw=AW)
    slab7(f, sx0, cy - 5.5, 7.0, 11.0, 4.0, 5, C2D)
    send2 = sx0 + 7.0 + 4.0 * DEP[0]
    T(f, (sx0 + send2) / 2, cy - 9.0, "stack along z", F_NOTE, c=MUTE, a="middle")
    T(f, sx0 + 4.0, cy + 8.7, "Z×256×32×32", F_NOTE, c=MUTE, a="middle")
    vx, vw, vh, vL = send2 + 2.6, 9.0, 12.0, 6.0
    arr(f, send2 + 0.2, cy, vx - 0.25, S2D)
    block(f, vx, cy - vh / 2, vw, vh, vL, C2D["front"], C2D["top"], C2D["side"], C2D["st"], top="../assets3/feat_enc_top.png")
    for k_ in range(1, 5):
        f.add(f'<line x1="{vx + k_ * vw / 5:.2f}" y1="{cy - vh / 2:.2f}" x2="{vx + k_ * vw / 5:.2f}" y2="{cy + vh / 2:.2f}" stroke="{C2D["st"]}" stroke-width="{0.5 * LW:.3f}" opacity="0.55"/>')
        f.add(f'<line x1="{vx:.2f}" y1="{cy - vh / 2 + k_ * vh / 5:.2f}" x2="{vx + vw:.2f}" y2="{cy - vh / 2 + k_ * vh / 5:.2f}" stroke="{C2D["st"]}" stroke-width="{0.5 * LW:.3f}" opacity="0.55"/>')
    vend = vx + vw + vL * DEP[0]
    T(f, vx + vw / 2 + 1.5, cy + 8.7, "256×Z×32×32", F_NOTE, c=MUTE, a="middle")
    f.prov_add("plane features re-stacked along z: (B*Z,256,32,32) -> (B,256,Z,32,32)", "net3d.py:288")
    c1 = vend + 2.6
    arr(f, vend + 0.2, cy, c1 - 0.25, S3D)
    cub(f, c1, cy - 5.0, 2.2, 10.0, 2.4, C3D)
    T(f, c1 + 1.6, cy - 7.4, "1^{3}", F_NOTE, c=ND, a="middle")
    cend = c1 + 2.2 + 2.4 * DEP[0]
    rf = [1]
    for dd in (1, 2, 4, 8):
        rf.append(rf[-1] + 2 * dd + 2)
    r0, rbw, rbL, rstep = cend + 3.0, 3.6, 5.6, 14.0
    foot = rbw + rbL * DEP[0]
    arr(f, cend + 0.15, cy, r0 - 0.2, S3D)
    arc_y = cy - 10.6
    pz, kmid = 0.25, cy + 13.4
    tops, bots = [], []
    for j, dd in enumerate((1, 2, 4, 8)):
        rx = r0 + j * rstep
        block(f, rx, cy - 6.0, rbw, 12.0, rbL, C3D["front"], C3D["top"], C3D["side"], C3D["st"], sw=0.8 * LW,
              top="../assets3/feat_dec_top.png" if j == 3 else None)
        pn = rx + foot + 2.0
        f.path(f"M{rx + foot + 0.1:.2f},{cy:.2f} H{pn - 1.0:.2f}", stroke=S3D, sw=AW, arrow=False)
        f.plus_node(pn, cy, r=1.0, col=S3D)
        f.path(f"M{rx - 1.4:.2f},{cy:.2f} V{arc_y:.2f} H{pn:.2f} V{cy - 1.25:.2f}", stroke=S3D, sw=0.9 * LW)
        if j < 3:
            arr(f, pn + 1.05, cy, rx + rstep - 0.2, S3D)
        kc = rx + foot / 2
        bx = voxel_kernel(f, kc, kmid, dd, pz)
        r = rf[j + 1]
        y0b, y1b = kmid - (r // 2) * pz - 0.3, kmid + (r // 2) * pz + 0.3
        f.add(f'<path d="M{bx:.2f},{y0b:.2f} h0.6 V{y1b:.2f} h-0.6" stroke="{ND}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
        T(f, bx + 1.1, kmid + 0.8, f"{r}", F_LAB, 700, c=ND)
        tops.append((bx + 0.6, y0b))
        bots.append((bx + 0.6, y1b))
        T(f, kc, cy + 20.9, f"d = {dd}", F_LAB, 700, c=ND, a="middle")
    last_pn = r0 + 3 * rstep + foot + 2.0
    for P in (tops, bots):
        f.path("M" + " L".join(f"{a:.2f},{b:.2f}" for a, b in P), stroke=tint(RED, 0.6), arrow=False, dash="0.6 0.5", chk=False, sw=0.5 * LW)
    T(f, r0 - 2.2, kmid - 1.0, "RF_{z}", F_NOTE, c=MUTE, a="end")
    T(f, r0 - 2.2, kmid + 2.4, "(planes)", F_NOTE, c=MUTE, a="end")
    T(f, r0 - 2.2, cy + 20.9, "z-dilation", F_NOTE, c=MUTE, a="end")
    num(f, last_pn + 2.6, cy + 20.3, 2)
    T(f, xII + 3.0, cy + 13.0, ["red: 3×3×3 taps of the first", "conv at z = −d, 0, +d;", "bracket: z receptive field", "after the block (planes)"], F_NOTE, c=MUTE, ls=1.2)
    f.prov_add("decoder: 1x1x1 Conv3d 256->256, four ResBlock3D (residual connection drawn as the arc to the sum node) with z-dilation 1,2,4,8, 256 channels, shape kept 256xZx32x32; 14.3 M params, lr 2e-4; voxel glyphs: the 3x3x3 kernel of the first conv with taps at z offsets -d, 0, +d; brackets: z receptive field after each block 5, 11, 21, 39 planes (derived: RF_k = RF_(k-1) + 2 d_k + 2)",
               "net3d.py:69-96; main.tex Methods '3D model' (1x1x1 input conv; four residual blocks, two 3x3x3 convs, 256 channels, GN 16 groups + GELU, first conv dilated by 1,2,4,8; 14.3 M); train3d.py:145")
    # residual-block inset
    rxi, ryi, rwi, rhi = xII + 1.2, t2y + t2h - 15.4, wII - 2.4, 14.2
    inset(f, rxi, ryi, rwi, rhi, "residual block (×4)", ND, "256×Z×32×32 in and out · GroupNorm 16 groups")
    funnel(f, (r0 - 1.4, cy + 21.9), (last_pn, cy + 21.9), (rxi + 6.0, ryi - 0.1), (rxi + rwi - 6.0, ryi - 0.1), tint(RED, 0.7))
    seq2 = [("GN", 4.4), ("GELU", 6.2), ("conv 3^{3}, dilation (d,1,1)", 26.0), ("GN", 4.4), ("GELU", 6.2), ("conv 3^{3}", 9.6)]
    bx_, by_, bh_ = rxi + 4.6, ryi + 6.2, 4.6
    first = bx_
    for i, (nm, bw_) in enumerate(seq2):
        conv = nm.startswith("conv")
        cc = C3D if conv else dict(front="#ffffff", top=tint(RED, 0.05), side=tint(RED, 0.16), st=C3D["st"])
        cub(f, bx_, by_, bw_, bh_, 1.0, cc, nm, size=F_NOTE, bold=conv)
        bx_ += bw_ + 1.0 * DEP[0]
        if i < 5:
            arr(f, bx_ + 0.15, by_ + bh_ / 2, bx_ + 1.7, S3D)
            bx_ += 1.9
    pxn = bx_ + 2.6
    f.plus_node(pxn, by_ + bh_ / 2, r=1.1, col=S3D)
    arr(f, bx_ + 0.15, by_ + bh_ / 2, pxn - 1.25, S3D)
    f.path(f"M{first - 1.6:.2f},{by_ + bh_ / 2:.2f} V{by_ - 1.4:.2f} H{pxn:.2f} V{by_ + bh_ / 2 - 1.3:.2f}", stroke=S3D, sw=0.9 * LW)
    f.path(f"M{rxi + 1.0:.2f},{by_ + bh_ / 2:.2f} H{first - 0.2:.2f}", stroke=S3D, sw=0.9 * LW, arrow=False)
    T(f, pxn + 2.4, by_ + bh_ / 2 + 0.7, "x + F(x)", F_NOTE, c=MUTE, italic=True)
    f.prov_add("ResBlock3D: x + conv3(GELU(GN(conv3_dil(GELU(GN(x)))))), GroupNorm 16 groups; first conv dilation (d,1,1)", "net3d.py:69-78; main.tex Methods '3D model'")

    # ---------------- stage III
    zx = xIII + 2.6
    arr(f, last_pn + 1.05, cy, zx - 0.25, S3D)
    zc = dict(front="#ffffff", top="#ffffff", side=tint(RED, 0.06), st=RED)
    cub(f, zx, cy - 5.0, 2.6, 10.0, 2.4, zc, dash="0.7 0.45")
    T(f, zx + 1.3, cy + 0.9, "0", F_NOTE, 700, c=RED, a="middle")
    T(f, zx + 2.3, cy - 7.4, "1^{3} · 4·64", F_NOTE, c=ND, a="middle")
    zend = zx + 2.6 + 2.4 * DEP[0]
    qx3 = zend + 4.2
    f.plus_node(qx3, cy, r=1.6)
    arr(f, zend + 0.2, cy, qx3 - 1.85, S3D)
    T(f, qx3 - 2.6, cy - 1.2, "Δ", F_LAB, 700, c=RED, a="middle")
    num(f, zx + 1.3, cy + 8.6, 3)
    hx, hy, hw, hh = xIII + 2.6, t2y + 6.8, 26.0, 4.2
    cub(f, hx, hy, hw, hh, 1.8, C2D, "2D head: 1×1 conv → 3·64", size=F_NOTE)
    vcx, vtop = vx + vw / 2 + vL * DEP[0] / 2, cy - vh / 2 + vL * DEP[1] / 2
    f.path(f"M{vcx:.2f},{vtop - 0.1:.2f} V{hy + hh / 2:.2f} H{hx - 0.25:.2f}", stroke=S2D, sw=AW)
    T(f, (vcx + xII + wII) / 2 + 6.0, hy + hh / 2 - 1.0, "same features, per plane", F_NOTE, c=MUTE, a="middle", italic=True)
    f.path(f"M{qx3:.2f},{hy + hh + 0.2:.2f} V{cy - 1.85:.2f}", stroke=S2D, sw=AW)
    T(f, qx3 + 1.0, cy - 6.6, "d_{y}, d_{x}, p", F_NOTE, c=MUTE)
    f.prov_add("2D head = Cellpose-SAM 1x1 Conv2d(256 -> 3*64) per plane; zero-initialised 1x1x1 output conv gives 4 x 8 x 8 values per 8x8-pixel token (4*64), added to the 2D heads for (dy, dx, p); dz = Delta_z",
               "net3d.py:287-295; cellpose vit.py:122; main.tex Methods '3D model'")
    # pixel shuffle
    psx = qx3 + 3.6
    arr(f, qx3 + 1.8, cy, psx - 0.4, INK)
    psend = pixel_shuffle(f, psx, cy)
    T(f, (psx + psend) / 2, cy + 8.6, "pixel shuffle ×8", F_NOTE, 700, a="middle")
    T(f, (psx + psend) / 2, cy + 10.9, ["4·64 per token", "→ 4×8×8 px"], F_NOTE, c=MUTE, a="middle")
    f.prov_add("pixel shuffle x8: the 4*64 values of each token become 4 channels of an 8x8 pixel block (glyph schematic: one channel, colour = value index)",
               "net3d.py:294 (pixel_shuffle 8); main.tex Methods ('4 x 8 x 8 values per 8 x 8-pixel token')")
    # real output maps
    ox0 = psend + 3.2
    arr(f, psend + 0.1, cy, ox0 - 0.3, INK)
    tw_, th_ = 9.6, 9.6 * 28 / 48
    names = [("dz", "d_{z}"), ("dy", "d_{y}"), ("dx", "d_{x}"), ("p", "p")]
    for k, (nm, lab) in enumerate(names):
        cxk = ox0 + (k % 2) * (tw_ + 1.4)
        cyk = cy - 7.4 + (k // 2) * (th_ + 4.0)
        f.image(cxk, cyk, tw_, th_, f"../assets8/out_{nm}.png", fit="none")
        f.rect(cxk, cyk, tw_, th_, rx=0, fill="none", stroke=tint(BASE, 0.6), sw=0.5 * LW, chk=False)
        T(f, cxk + tw_ / 2, cyk + th_ + 2.3, lab, F_NOTE, 700, a="middle")
    oend = ox0 + 2 * tw_ + 1.4
    T(f, (ox0 + oend) / 2, cy - 8.6, "4×Z×256×256", F_NOTE, 700, a="middle")
    T(f, (ox0 + oend) / 2, cy + 13.6, "S1 · xz · nucleus 62", F_NOTE, c=MUTE, a="middle")
    num(f, oend + 1.2, cy - 9.2, 4)
    f.prov_add("output maps: raw S1 seed-0 outputs (dz, dy, dx = 5 x unit flow, palette diverging clipped at +-5; p = sigmoid(logit)) on the xz section through GT nucleus 62 of E15_Ab2, identity view, one 32-plane window",
               "assets3/views_E15_Ab2_n62.npz (arch_views.py) -> make_a8_assets.py -> assets8/out_*.png, assets8_log.json")
    # 3D target + loss
    lx = oend + 3.4
    lw_ = xIII + wIII - 1.4 - lx
    ts_ = 7.4
    tgx1, tgx2 = lx + 0.2, lx + lw_ - ts_ - 0.2
    tgy = t2y + 5.6
    th8 = ts_ * 679 / 555
    tgy = t2y + 5.2
    f.image(tgx1, tgy, ts_, th8, "../assets8/diff_heat.png")
    f.image(tgx2, tgy, ts_, th8, "../assets8/diff_flow.png")
    arr(f, tgx1 + ts_ + 0.2, tgy + th8 / 2, tgx2 - 0.2, LOSS)
    T(f, tgx1 + ts_ / 2, tgy + th8 + 2.1, "heat T", F_NOTE, c=MUTE, a="middle")
    T(f, tgx2 + ts_ / 2, tgy + th8 + 2.1, "u = ∇T/|∇T|", F_NOTE, c=MUTE, a="middle")
    ts_ = th8
    ly, lh = cy - 2.0, 12.4
    f.rect(lx, ly, lw_, lh, rx=1.0, fill=tint(LOSS, 0.10), stroke=LOSS, sw=0.8 * LW)
    T(f, lx + lw_ / 2, ly + 2.9, "loss", F_LAB, 700, c=LOSS, a="middle")
    T(f, lx + lw_ / 2, ly + 5.5, ["½ MSE(flows, 5*u*)", "+ BCE(p, fg)"], F_NOTE, 700, c=LOSS, a="middle", ls=1.2)
    T(f, lx + lw_ / 2, ly + 10.6, "inside the volume", F_NOTE, c=MUTE, a="middle")
    arr(f, (tgx1 + ts_ + tgx2) / 2, tgy + ts_ + 0.6, (tgx1 + ts_ + tgx2) / 2, ly - 0.25, LOSS, dash="0.9 0.6")
    arr(f, oend + 0.2, cy + 3.0, lx - 0.25, cy + 3.0, LOSS, dash="0.9 0.6")
    f.prov_add("3D target: heat released at the most central voxel of the GT nucleus diffuses inside the nucleus; unit gradient = flow u (tiles: nucleus 62, E15_Ab2); loss = MSE(flow, 5u)/2 + BCE(logit, fg) over voxels inside the volume",
               "data3d.py:28-81 flows3d_gpu; render_diffusion.py (assets3/diff_heat.png, diff_flow.png); train3d.py:33-43; main.tex Methods '3D model'")
    # output-composition inset
    zxi, zyi, zwi, zhi = xIII + 1.2, t2y + t2h - 23.6, wIII - 2.4, 22.4
    inset(f, zxi, zyi, zwi, zhi, "output composition", ND, "step 0 = 2D control, d_{z} = 0")
    funnel(f, (zx - 0.4, cy + 11.2), (qx3 + 0.4, cy + 11.2), (zxi + 4.0, zyi - 0.1), (zxi + 13.0, zyi - 0.1), tint(RED, 0.7))
    cA, cB = zxi + 15.0, zxi + 43.0
    for k_, (dy_, t_) in enumerate(((9.6, "Δ is added to the"), (12.9, "2D head for p,"), (16.2, "d_{y} and d_{x};"), (19.9, "Δ alone gives d_{z}"))):
        T(f, zxi + 53.0, zyi + dy_, t_, F_NOTE, c=MUTE)
    T(f, cA, zyi + 7.0, "step 0", F_LAB, 700, a="middle")
    T(f, cB, zyi + 7.0, "trained", F_LAB, 700, c=ND, a="middle")
    rowsZ = [("d_{z}", "0", "Δ_{z}"), ("d_{y}", "2D", "2D+Δ"), ("d_{x}", "2D", "2D+Δ"), ("p", "2D", "2D+Δ")]
    for k_, (chn, a_, b_) in enumerate(rowsZ):
        yy = zyi + 8.2 + k_ * 3.55
        T(f, zxi + 4.0, yy + 2.2, chn, F_NOTE)
        zero = a_ == "0"
        cub(f, cA - 5.0, yy, 10.0, 3.0, 0.8, zc if zero else C2D, a_, bold=zero, dash="0.7 0.45" if zero else None, lc=RED if zero else INK)
        cub(f, cB - 6.0, yy, 12.0, 3.0, 0.8, C3D, b_, bold=False)
    arr(f, cA + 6.0, zyi + 8.2 + 3.55 + 1.5, cB - 6.6, zyi + 8.2 + 3.55 + 1.5, INK)
    T(f, (cA + cB) / 2, zyi + 8.2 + 3.55 + 0.8, "train", F_NOTE, c=MUTE, a="middle", italic=True)
    f.prov_add("output composition (dz, dy, dx, p) = (Delta_z, 2D head + Delta); Delta = 0 at step 0 (output layer initialised to zero), so the untrained model reproduces the 2D control with dz = 0",
               "net3d.py:89-90, 294; main.tex Results 'A zero-initialized residual decoder...' and Methods '3D model'")

    # ---------------------------------------------------------------- tier 3: training schedule
    t3y, t3h = t2y + t2h + 2.4, 17.2
    card(f, ox + 0.5, t3y, w - 1.0, t3h, "IV", "IV. Training schedule", "AdamW throughout · five seeds per stage")
    by0, bh0 = t3y + 5.6, 10.8
    boxes = [
        (ox + 2.0, 30.0, C2D, ["*2D control* · lr 10^{−5}", "from Cellpose-SAM cpsam_v2", "6,056 slices · 100 × 1,000", "batch 1 · wd 0.1"]),
        (None, 39.0, C3D, ["*A1* · lr 10^{−5} enc. / 2·10^{−4} dec.", "3D model on the 2D encoder", "20,000 steps · 32×256×256", "wd 0.05 · 300 warm-up · cosine"]),
    ]
    xcur = ox + 2.0
    rights = []
    for i, (_, bw_, c, lines) in enumerate(boxes):
        f.rect(xcur, by0, bw_, bh0, rx=0.8, fill=c["front"], stroke=c["st"], sw=0.7 * LW, chk=False)
        T(f, xcur + 1.4, by0 + 2.6, lines, F_NOTE, ls=1.3)
        rights.append(xcur + bw_)
        xcur += bw_ + 4.2
        if i == 0:
            arr(f, rights[-1] + 0.2, by0 + bh0 / 2, xcur - 0.25, S2D)
    fx = xcur + 3.2
    fw = 46.0
    fy1, fy2, fh = by0, by0 + 5.8, 5.0
    for yy, lines in ((fy1, ["*C* · +3,000 steps · unchanged · 100 warm-up"]), (fy2, ["*X1* · +3,000 steps · simulated lateral downsampling"])):
        f.rect(fx, yy, fw, fh, rx=0.8, fill=C3D["front"], stroke=C3D["st"], sw=0.7 * LW, chk=False)
        T(f, fx + 1.4, yy + 3.0, lines, F_NOTE)
    jx = rights[-1] + 1.6
    f.path(f"M{rights[-1] + 0.2:.2f},{by0 + bh0 / 2:.2f} H{jx:.2f}", stroke=S3D, sw=AW, arrow=False)
    for yy in (fy1 + fh / 2, fy2 + fh / 2):
        f.path(f"M{jx:.2f},{by0 + bh0 / 2:.2f} V{yy:.2f} H{fx - 0.25:.2f}", stroke=S3D, sw=AW)
    s1x = fx + fw + 4.6
    s1w = 23.0
    mx_ = fx + fw + 1.6
    for yy in (fy1 + fh / 2, fy2 + fh / 2):
        f.path(f"M{fx + fw + 0.2:.2f},{yy:.2f} H{mx_:.2f} V{by0 + bh0 / 2:.2f}", stroke=S3D, sw=AW, arrow=False)
    arr(f, mx_, by0 + bh0 / 2, s1x - 0.25, S3D)
    f.rect(s1x, by0, s1w, bh0, rx=0.8, fill=tint(RED, 0.30), stroke=RED, sw=0.9 * LW, chk=False)
    T(f, s1x + 1.4, by0 + 2.4, ["*S1 = ½(C + X1)*", "parameter-wise", "average (model soup),", "weight fixed at 0.5"], F_NOTE, ls=1.3)
    nx_ = s1x + s1w + 2.0
    T(f, nx_, by0 + 2.4, ["X1: probability 0.33,", "interval 2–10 voxels,", "blur 0.3–0.6× interval;", "no noise augmentation"], F_NOTE, c=MUTE, ls=1.3)
    f.prov_add("training schedule: 2D control = Cellpose-SAM cpsam_v2 fine-tuned on 6,056 slices, AdamW lr 1e-5, wd 0.1, 100 epochs of 1,000 slices; A1 = 3D model from the seed-matched 2D encoder, 20,000 steps on 32x256x256 patches, lr 1e-5 encoder / 2e-4 decoder, wd 0.05, 300 warm-up steps, cosine decay, no noise augmentation; C and X1 = 3,000 further steps each, 100 warm-up; X1 simulated lateral downsampling (probability 0.33, interval 2-10 network voxels, Gaussian blur 0.3-0.6 x interval); S1 = parameter-wise average (C + X1)/2, weight fixed at 0.5 (model soup); five seeds per stage",
               "main.tex Methods '2D fine-tuning' and '3D model'; Results 'A zero-initialized residual decoder...' (model soup)")

    # ---------------------------------------------------------------- rationale + stream legend
    ry = t3y + t3h + 3.3
    notes = ["per-plane encoding keeps the 2D foundation model's representations, which native 3D models do not inherit",
             "axial resolution is several-fold poorer than lateral: dilation widens z context to 39 planes without extra weights",
             "zero-initialised output: the network starts as the 2D model and learns only the volumetric correction",
             "the axial flow comes from 3D context, not from summing flows predicted on separate orthogonal slices"]
    for k_, t_ in enumerate(notes):
        colx = ox + 1.0 + (k_ % 2) * 91.5
        rowy = ry + (k_ // 2) * 3.4
        num(f, colx + 1.6, rowy - 0.7, k_ + 1)
        T(f, colx + 4.0, rowy, t_, F_NOTE)
    f.prov_add("rationale 1-4 wording", "main.tex Introduction l.44-50 (as in gen_a4 panel_arch4)")
    stream_legend(f, ox + 0.5, ry + 4.9, w - 1.0)
    return ry + 4.9 + 4.6


def layout_b8():
    G.STYLE["v"] = "b"
    f = Fig(183.0, 219.4)
    end = panel_arch8(f, 0.0, 0.0, letter="a")
    yb = end + 1.6
    panel_cmp7(f, 0.0, yb, 120.5, letter="b")
    panel_target7(f, 122.0, yb, 60.5, letter="c")
    panel_infer7(f, 0.0, yb + 42.2, 183.0, letter="d")
    return f


def main():
    cid = "A8-b"
    f = layout_b8()
    f.el = [e for e in f.el if not (e.startswith("<text") and e.endswith("></text>") and ">" + "</text>" in e and e.count("<text") == 1 and e.split(">", 1)[1] == "</text>")]
    res = export(f, cid)
    print(cid, "issues:", len(res.get("issues", [])), "min font", res.get("min_font_pt"), "H", f.H)
    for s_ in res.get("issues", [])[:80]:
        print("   ", s_)
    json.dump(dict(figure=cid, code_snapshot=_nsp_os.path.join(_NSP_REPO, 'src'),
                   model="orx_t3d_r5s1_s0/model3d.json", palette="paper/figures/palette.json",
                   generator="paper/figures/arch/gen_a8.py (layout_b8); panels b-d from gen_a7.py (A7-b style); assets: make_a7_assets.py, make_a8_assets.py",
                   reference_image=REF_IMG, colour_scheme="3D-viewer warm grey panes #EDECE9 (Supplementary_Data_3D_viewer.html)",
                   references=[dict(figure=a, technique=b) for a, b in REFS8 + REFS7 + REFS6 + REFS4], items=f.prov),
              open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
