"""Fig. 1 variant A8-d: information-dense S1 architecture in the dense-Nature style of A7-d (3D-viewer warm-grey scheme).

a  architecture: main row as A7-d plus five zoom insets (i per-plane tokenisation with the full shape trace, ii transformer block with a
   global-attention token-grid motif, iii dilated 3x3x3 kernels as voxel grids along z with the receptive-field cone and residual paths,
   iv zero-init composition, pixel shuffle x8 as sub-pixel rearrangement and the four real output maps, v diffusion target and loss)
b  training schedule strip (2D control -> A1 -> C / X1 -> S1 = (C + X1)/2) with the optimiser facts of main.tex Methods
c-e compacted A7-d panels (z information, 3D target, inference).
Every fact is listed with its source in A8-d.provenance.json. New real-data rasters: make_a8d_assets.py -> assets8d/.
Usage: python gen_a8d.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fig import LW, PAL, Fig, export, tint  # noqa: E402
from gen_a3 import DEP, block  # noqa: E402
from gen_a4 import REFS as REFS4, hatch_defs, num  # noqa: E402
import gen_a7d as G  # noqa: E402
from gen_a7d import (ACC2D, ACC3D, ACCT, BASE, BLOCK, C2D, C3D, CZ, F_L, F_N, F_S, INK, LOSS, MUTE, PT, RED, RULE, S2D, S3D,  # noqa: E402
                     a7, cub, funnel, harrow, inset, plabel, section, slab, tw)

HERE = os.path.dirname(os.path.abspath(__file__))
A8LOG = json.load(open(os.path.join(HERE, "assets8d", "a8d_assets_log.json")))
DF = G.DF
H_PAGE = 224.5
WEB_REFS = [
    ("MA-SAM, arXiv 2309.08842, Fig. 1 (arxiv.org/html/2309.08842)", "reshape [BN,H,W] <-> [B,c,N,H,W] written on the figure so 3D operations fit a 2D backbone; trainable vs frozen parts marked per module"),
    ("SAM-Med3D, arXiv 2310.15161, Figs. 1 and 3", "orientation key in a corner; per-component frozen/learnable markers; 3D positional encoding named on the module"),
    ("SAM 2, arXiv 2408.00714, Figs. 1-3", "lettered sub-panels (task / model / data) and a step-labelled strip; caption states which paths are omitted"),
    ("UNETR, arXiv 2103.10504, Fig. 2", "tensor sizes written on every connection (P = 16, K = 768); channel counts at input and output"),
    ("Swin-UNETR, arXiv 2201.01266, Figs. 1-2", "explicit token grid with window size (8x8x8 tokens, 4x4x4 windows) to show what attention sees"),
    ("nnFormer, arXiv 2109.03201, Figs. 2-3", "overview plus lettered zoom panels of each block type; abbreviations (K, DK, S, Norm) defined on the figure"),
    ("Cellpose-SAM, bioRxiv 2025.04.28.651001, Fig. 1", "old and new architectures side by side; a training-stage strip; parameter split between encoder and head"),
    ("Cellpose3, bioRxiv 2024.02.10.579780, Fig. 1e", "separate training-protocol schematic with the loss terms as coloured boxes"),
    ("CellSAM, arXiv 2311.11004, Fig. 1", "two-stage training with frozen versus fine-tuned modules named per stage"),
    ("EmbedSeg, arXiv 2101.10033, Fig. B.1", "intermediate network outputs (seediness, embeddings) shown as real maps next to the schematic"),
    ("StarDist-3D, arXiv 1908.03636, Fig. 1", "lettered strip: training targets -> network -> post-processing, all on real data"),
    ("micro-SAM (muSAM), bioRxiv 2023.08.21.551956", "not read: the bioRxiv page refused automated access (HTTP 403)"),
]
TAGC = "#7d7d84"


def tag(f, x, y, s):
    """Small grey tag (roman numeral) linking a main-row module to its zoom inset; (x, y) = top-left."""
    w = 1.6 + len(s) * 1.05
    f.rect(x, y, w, 2.7, rx=0.5, fill=TAGC, stroke="none", chk=False)
    f.add(f'<text x="{x + w / 2:.2f}" y="{y + 1.95:.2f}" font-size="{5.0 * PT:.3f}" font-weight="700" fill="#ffffff" '
          f'text-anchor="middle" font-style="italic">{s}</text>')
    return w


def srcbox(f, x0, y0, x1, y1, col):
    """Thin dashed box (stage colour) around exactly the modules an inset expands; checked for text crossings."""
    f.rect(x0, y0, x1 - x0, y1 - y0, rx=1.0, fill="none", stroke=col, sw=0.55 * LW, dash="1.0 0.6", chk=True)


def funnel9(f, box, ins, col):
    """Two dashed lines: source box bottom-left -> inset top-left, bottom-right -> top-right (checked for text crossings)."""
    (bx0, by1, bx1), (ix0, iy0, ix1) = box, ins
    for a, b, c, d in ((bx0, by1, ix0, iy0), (bx1, by1, ix1, iy0)):
        f.path(f"M{a:.2f},{b:.2f} L{c:.2f},{d:.2f}", stroke=col, arrow=False, dash="0.7 0.5", chk=True, sw=0.55 * LW)


def inset8(f, x, y, w, h, t, title, accent, right=None):
    f.rect(x, y, w, h, rx=1.2, fill="#ffffff", stroke=accent, sw=0.55 * LW, chk=False)
    tw_ = tag(f, x + 1.4, y + 1.35, t)
    f.text(x + 2.6 + tw_, y + 3.5, title, F_S, 700, c=accent)
    if right:
        f.text(x + w - 1.4, y + 3.5, right, F_N, c=MUTE, a="end")


# ======================================================================== panel a
def panel_arch8(f, ox, oy, w=183.0, letter="a"):
    hatch_defs(f)
    plabel(f, ox, oy + 3.4, letter, "S1 architecture")
    items = [(C2D, False, None, "2D, fine-tuned (lr 10^{−5})"), (C3D, False, None, "new (lr 2·10^{−4})"),
             (CZ, False, "0.7 0.45", "zero-initialised"), (C2D, True, None, "frozen")]
    widths = [5.3 + tw(lab, F_L) + 3.6 for *_, lab in items]
    cx = ox + w - 0.5 - sum(widths) + 3.6
    for (c, hatch, dash, lab), wd in zip(items, widths):
        cub(f, cx, oy + 1.0, 3.4, 2.9, 1.5, c, hatch=hatch, dash=dash)
        f.text(cx + 5.3, oy + 3.4, lab, F_L)
        cx += wd

    cy0, ch = oy + 5.4, 101.2
    xI, wI = ox + 0.5, 57.5
    xII, wII = xI + wI + 1.4, 69.0
    xIII = xII + wII + 1.4
    wIII = ox + w - 0.5 - xIII
    section(f, xI, cy0, wI, ch, BLOCK, "I. Per-plane 2D encoder", accent=ACC2D)
    section(f, xII, cy0, wII, ch, BLOCK, "II. 3D residual decoder", accent=ACC3D)
    section(f, xIII, cy0, wIII, ch, BLOCK, "III. Output and training target", accent=ACCT)
    cy = oy + 29.2
    lab1, lab2 = cy + 8.8, cy + 11.1

    # ---------------- main row (as A7-d)
    ix, iy_, isz, iL = xI + 2.0, oy + 15.2, 11.5, 5.8
    block(f, ix, iy_, isz, isz, iL, "#000", "#000", "#000", C2D["st"], front="../assets3/in_front.png", top="../assets3/in_top.png",
          side="../assets3/in_side.png")
    icx = ix + isz / 2 + iL * DEP[0] / 2
    f.text(icx, oy + 30.4, "input", F_L, 700, a="middle")
    f.text(icx, oy + 32.7, "1×Z×256×256", F_N, c=MUTE, a="middle")
    f.text(icx, oy + 34.9, "Z = 32 planes", F_N, c=MUTE, a="middle")
    f.prov_add("input window 1 x Z x 256 x 256, Z = 32 planes in training; textures = real E15_Ab2 data on the isotropic grid",
               "net3d.py:268-272; train3d.py:218 patch (32,256,256); main.tex Methods '3D model' (32 x 256 x 256 patches); make_a3_assets.py")
    bh, pitch, bd = 4.3, 5.0, 2.6
    b1 = oy + 16.4
    px = ix + isz + iL * DEP[0] + 2.4
    pw = 7.4
    harrow(f, ix + isz + iL * DEP[0] + 0.2, px - 0.25, b1 + bh / 2, S2D)
    cub(f, px, b1, pw, bh, 1.8, C2D, "patch", hatch=True)
    f.text(px + pw / 2 + 0.6, b1 - 2.2, "8×8, frozen", F_N, c=MUTE, a="middle")
    pyy = b1 + 7.6
    cub(f, px, pyy, pw, bh, 1.8, C2D, "+ pos.")
    f.text(px + pw / 2, pyy + bh + 2.4, "embedding", F_N, c=MUTE, a="middle")
    f.text(px + pw / 2, pyy + bh + 4.6, "per plane", F_N, c=MUTE, a="middle")
    f.text(px + pw / 2, pyy + bh + 6.8, "32×32×1024", F_N, c=MUTE, a="middle")
    sx = px + pw + 6.2
    qx = px + pw + 1.8 * DEP[0] + 1.9
    f.plus_node(qx, b1 + bh / 2, r=1.0, col=S2D)
    harrow(f, px + pw + 1.8 * DEP[0] + 0.2, qx - 1.15, b1 + bh / 2, S2D)
    harrow(f, qx + 1.05, sx - 0.25, b1 + bh / 2, S2D)
    f.path(f"M{px + pw + 1.8 * DEP[0] + 0.2:.2f},{pyy + bh / 2:.2f} H{qx:.2f} V{b1 + bh / 2 + 1.25:.2f}", stroke=S2D, sw=1.1 * LW)
    sw_ = 19.0
    ys = [b1, b1 + pitch, None, b1 + 2 * pitch + 3.3, b1 + 3 * pitch + 3.3]
    names = ["block 1", "block 2", None, "block 24", "neck"]
    # source boxes (drawn first, behind the modules): i = tokenisation (input, patch, + pos., sum); ii = the 24 blocks
    BOX_I = (ox + 1.4, oy + 11.0, qx + 1.6, oy + 40.0)
    srcbox(f, *BOX_I, C2D["st"])
    BOX_II = (sx - 0.7, b1 + bd * DEP[1] - 0.6, sx + sw_ + bd * DEP[0] + 0.7, ys[3] + bh + 0.5)
    srcbox(f, *BOX_II, C2D["st"])
    for yy, nm in zip(ys, names):
        if yy is None:
            continue
        cc = C2D if nm != "neck" else dict(front=tint(BASE, 0.40), top=tint(BASE, 0.22), side=tint(BASE, 0.60), st=C2D["st"])
        cub(f, sx, yy, sw_, bh, bd, cc, nm if nm != "neck" else "neck → 256")
    f.text(sx + sw_ / 2, b1 + 2 * pitch + 1.7, "···", F_S, 700, c=C2D["st"], a="middle")
    scx = sx + sw_ / 2 + bd * DEP[0] / 2
    f.text(scx, oy + 10.2, "ViT-L · 24 blocks · 304 M", F_L, 700, a="middle")
    f.text(scx, oy + 13.6, "global attention · lr 10^{−5}", F_N, c=MUTE, a="middle")
    num(f, sx - 2.4, b1 + 2 * pitch + 1.6, 1)
    tag(f, (BOX_I[0] + BOX_I[2]) / 2 - 1.3, BOX_I[3] + 0.4, "i")
    tag(f, sx + sw_ - 4.6, b1 + 2 * pitch + 0.4, "ii")
    f.prov_add("patch embedding 8x8 conv, stride 8 (SAM 16x16 kernel subsampled), weights not trained; absolute position embedding (1,32,32,1024), trained",
               "cellpose vit.py:109-116 (CPSAM); net3d.py:202-205 patch-embed requires_grad False, :228-232 _embed")
    f.prov_add("ViT-L encoder: 24 blocks, width 1024, 16 heads, MLP ratio 4, relative position bias, global attention in every block; all B*Z planes, shared weights; 304.4 M encoder-side parameters, lr 1e-5",
               "segment_anything build_sam.py:27-34, 66-80; cellpose vit.py:118-120; net3d.py:285-286; train3d.py:144, 211-214; train log; main.tex Fig. 1 legend")

    # stage II main row
    ny = ys[4] + bh / 2
    s0 = xII + 3.0
    f.path(f"M{sx + sw_ + bd * DEP[0] + 0.2:.2f},{ny:.2f} H{xII - 0.7:.2f} V{cy:.2f} H{s0 - 0.3:.2f}", stroke=S2D, sw=1.4 * LW)
    sh_, sd_ = 11.0, 5.0
    slab(f, s0, cy - sh_ / 2, 8.6, sh_, sd_, 5, C2D)
    f.text(s0 + 4.3 + sd_ * DEP[0] / 2, cy - sh_ / 2 - 4.2, "stack along z", F_N, c=MUTE, a="middle")
    f.text(s0 + 4.3, lab1, "Z×256×32×32", F_N, c=MUTE, a="middle")
    vx = s0 + 8.6 + sd_ * DEP[0] + 2.6
    harrow(f, vx - 2.4, vx - 0.25, cy, S2D)
    vw, vh, vL = 9.5, 12.0, 6.0
    block(f, vx, cy - vh / 2, vw, vh, vL, C2D["front"], C2D["top"], C2D["side"], C2D["st"], top="../assets3/feat_enc_top.png")
    for k in range(1, 5):
        gx_, gy_ = vx + k * vw / 5, cy - vh / 2 + k * vh / 5
        f.add(f'<path d="M{gx_:.2f},{cy - vh / 2:.2f} V{cy + vh / 2:.2f} M{vx:.2f},{gy_:.2f} H{vx + vw:.2f}" stroke="{C2D["st"]}" '
              f'stroke-width="{0.5 * LW:.3f}" opacity="0.6" fill="none"/>')
    f.text(vx + vw / 2 + 1.0, lab1, "256×Z×32×32", F_N, c=MUTE, a="middle")
    f.prov_add("plane features re-stacked to (B,256,Z,32,32) by view + permute", "net3d.py:288 f3 = f.view(B,Z,...).permute(0,2,1,3,4)")
    vend = vx + vw + vL * DEP[0]
    c1 = vend + 2.6
    harrow(f, vend + 0.2, c1 - 0.25, cy, S3D)
    cub(f, c1, cy - 5.0, 2.2, 10.0, 2.4, C3D)
    f.text(c1 + 1.1, lab1, "1^{3}", F_N, c=PAL["new_d"], a="middle")
    rf = [1]
    for dd in (1, 2, 4, 8):
        rf.append(rf[-1] + 2 * dd + 2)
    r0, rbw, rbL, rstep = c1 + 2.2 + 2.4 * DEP[0] + 2.0, 3.4, 5.4, 6.6
    harrow(f, c1 + 2.2 + 2.4 * DEP[0] + 0.15, r0 - 0.2, cy, S3D)
    for j, dd in enumerate((1, 2, 4, 8)):
        rx = r0 + j * rstep
        if j:
            harrow(f, rx - rstep + rbw + rbL * DEP[0] + 0.1, rx - 0.15, cy, S3D)
        block(f, rx, cy - 6.0, rbw, 12.0, rbL, C3D["front"], C3D["top"], C3D["side"], C3D["st"], sw=0.8 * LW,
              top="../assets3/feat_dec_top.png" if j == 3 else None)
        f.text(rx + rbw / 2, lab1, f"d={dd}", F_N, 700, c=PAL["new_d"], a="middle")
        f.text(rx + rbw / 2, lab2, f"{rf[j + 1]}", F_N, c=MUTE, a="middle")
    f.text(r0 - 1.6, lab2, "RF_{z}", F_N, c=MUTE, a="end")
    dend = r0 + 3 * rstep + rbw + rbL * DEP[0]
    dmid = (c1 + dend) / 2 + 1.0
    f.text(dmid, oy + 15.4, "3D residual decoder · 14.3 M", F_L, 700, c=PAL["new_d"], a="middle")
    f.text(dmid, oy + 18.6, "4 blocks · z-dilation d · lr 2·10^{−4}", F_N, c=MUTE, a="middle")
    num(f, c1 - 0.6, oy + 16.0, 2)
    BOX_III = (r0 - 0.8, cy - 6.0 + rbL * DEP[1] - 0.5, dend + 0.4, lab2 + 1.9)
    srcbox(f, *BOX_III, PAL["new_l"])
    tag(f, dend - 4.65, lab2 - 1.6, "iii")
    f.prov_add("decoder: 1x1x1 Conv3d 256->256, four ResBlock3D (two 3x3x3 convs, 256 channels, GroupNorm 16 groups + GELU before each; first conv dilated along z by 1,2,4,8), 14.3 M params, lr 2e-4; z receptive field after blocks 1-4 = 5, 11, 21, 39 planes (derived: 1 + sum(2d + 2))",
               "net3d.py:69-96; model3d.json dec_*; train3d.py:145; train log '新增模块 14.3M'; main.tex Methods '3D model'")

    # stage III main row
    zx = xIII + 2.2
    harrow(f, dend + 0.2, zx - 0.25, cy, S3D)
    cub(f, zx, cy - 5.0, 2.6, 10.0, 2.4, CZ, dash="0.7 0.45")
    f.text(zx + 1.3, cy + 0.9, "0", F_N, 700, c=RED, a="middle")
    f.text(zx + 1.3, lab1, "1^{3}", F_N, c=PAL["new_d"], a="middle")
    f.text(zx + 2.2, cy - 8.2, "4·64", F_N, c=MUTE, a="middle")
    f.text(zx + 2.2, cy - 10.4, "GN · GELU", F_N, c=MUTE, a="middle")
    num(f, zx + 1.3, lab2 - 0.5, 3)
    tag(f, zx + 3.4, lab2 - 1.9, "iv")
    qx3 = zx + 2.6 + 2.4 * DEP[0] + 5.0
    f.plus_node(qx3, cy, r=1.6)
    harrow(f, zx + 2.6 + 2.4 * DEP[0] + 0.2, qx3 - 1.85, cy, S3D)
    f.text(qx3 - 2.9, cy - 1.1, "Δ", F_L, 700, c=RED, a="middle")
    hx, hy, hw, hh = xIII + 2.2, oy + 10.9, 26.0, 4.4
    cub(f, hx, hy, hw, hh, 1.8, C2D, "2D head: 1×1 conv → 3·64", size=F_N)
    # the 2D head reads the per-plane neck features f (net3d.py:287 tok2d = self.net.out(f)), i.e. the neck output before re-stacking
    jx = xII - 0.7
    f.path(f"M{jx:.2f},{cy:.2f} V{hy + hh / 2:.2f} H{hx - 0.25:.2f}", stroke=S2D, sw=1.4 * LW)
    f.circle(jx, cy, 0.6, fill=S2D, chk=False)
    f.text((jx + xII + wII) / 2, hy + hh / 2 - 1.0, "per-plane neck features (Z×256×32×32)", F_N, c=MUTE, a="middle", italic=True)
    f.path(f"M{qx3:.2f},{hy + hh + 0.2:.2f} V{cy - 1.85:.2f}", stroke=S2D, sw=1.4 * LW)
    f.text(qx3 + 1.0, cy - 12.0, "d_{y}, d_{x}, p", F_N, c=MUTE)
    f.prov_add("2D head = Cellpose-SAM 1x1 Conv2d(256 -> 3*64) per plane; Delta added for (dy, dx, p); dz = Delta_z", "net3d.py:287-295; cellpose vit.py:122")
    ox_, ow, oL = qx3 + 4.6, 10.0, 6.0
    f.arrow(qx3 + 1.8, cy, ox_ - 0.25, cy, sw=1.4 * LW)
    BOX_IV = (zx - 0.8, cy - 5.0 + 2.4 * DEP[1] - 0.6, ox_ - 0.6, lab2 + 1.6)
    srcbox(f, *BOX_IV, PAL["new_l"])
    f.text((qx3 + ox_) / 2 + 0.2, cy + 3.4, "×8", F_N, c=MUTE, a="middle")
    block(f, ox_, cy - 6.0, ow, 12.0, oL, "#ffffff", "#ffffff", tint(BASE, 0.10), INK, top="../assets3/out_top.png", sw=0.8 * LW)
    f.image(ox_ + 0.4, cy - 5.6, ow - 0.8, 11.2, a7("flow_glyphs_pal.png"))
    f.text(ox_ + ow / 2 + 1.5, lab1, "(d_{z}, d_{y}, d_{x}, p)", F_N, 700, a="middle")
    f.text(ox_ + ow / 2 + 1.5, lab2 + 1.0, "4×Z×256×256", F_N, c=MUTE, a="middle")
    num(f, ox_ - 1.9, cy - 4.4, 4)
    lx = ox_ + ow + oL * DEP[0] + 2.0
    lw_ = xIII + wIII - 1.4 - lx
    ly, lh = cy - 7.4, 14.8
    f.rect(lx, ly, lw_, lh, rx=1.0, fill=tint(LOSS, 0.14), stroke=LOSS, sw=0.7 * LW)
    f.text(lx + lw_ / 2, ly + 3.2, "loss", F_L, 700, c=LOSS, a="middle")
    f.text(lx + lw_ / 2, ly + 5.9, ["½ MSE(flows, 5*u*)", "+ BCE(p, fg)"], F_N, 700, c=LOSS, a="middle", ls=1.25)
    f.text(lx + lw_ / 2, ly + 10.9, ["vs 3D target,", "in the volume"], F_N, c=MUTE, a="middle", ls=1.25)
    f.arrow(ox_ + ow + oL * DEP[0] + 0.2, cy, lx - 0.25, cy, stroke=LOSS, sw=1.4 * LW, dash="0.9 0.6")
    BOX_V = (lx - 0.6, ly - 0.6, lx + lw_ + 0.6, ly + lh + 0.6)
    srcbox(f, *BOX_V, LOSS)
    tag(f, lx + lw_ / 2 - 1.1, BOX_V[3] + 0.4, "v")
    f.prov_add("output (B,4,Z,256,256) = [dz, dy, dx, cellprob] after pixel shuffle x8; glyph = S1 seed-0 output on E15_Ab2 nucleus 90 (illustrative glyph only: the only exported 3D flow crop, export/flow_mouse_embryo.npz, is centred on nucleus 90 and contains only 167 edge voxels of nucleus 62, so no equivalent nucleus-62 glyph exists; all other panels use nucleus 62)",
               "net3d.py:295,299; arch/render_glyphs.py")
    f.prov_add("loss: MSE(net flow, 5 x unit target)/2 + BCE(cellprob, nucleus) over voxels inside the volume", "train3d.py:33-43; main.tex Methods '3D model'")

    # ================= insets
    # ---- i: per-plane tokenisation and shape trace
    ex, ew = xI + 1.2, wI - 2.4
    iy, ih = oy + 43.4, 29.0
    inset8(f, ex, iy, ew, ih, "i", "per-plane tokenisation", "#4a5577", "B = batch")
    funnel9(f, (BOX_I[0], BOX_I[3], BOX_I[2]), (ex, iy, ex + ew), C2D["st"])
    rowc = iy + 12.2
    vs_, vL_ = 7.0, 3.0
    vx0 = ex + 2.2
    block(f, vx0, rowc - vs_ / 2 + 0.8, vs_, vs_, vL_, "#000", "#000", "#000", C2D["st"], front="../assets3/in_front.png",
          top="../assets3/in_top.png", side="../assets3/in_side.png", sw=0.6 * LW)
    f.text(ex + 1.2, iy + 20.0, "B×1×Z×256×256", F_N, c=MUTE)
    a0 = vx0 + vs_ + vL_ * DEP[0] + 0.5
    harrow(f, a0 + 0.6, a0 + 6.6, rowc, S2D)
    f.text(a0 + 3.8, rowc - 1.1, "reshape", F_N, c=MUTE, a="middle")
    ps_ = 7.4
    p0 = a0 + 7.0
    for k in range(3, -1, -1):
        dxk, dyk = k * 1.25, -k * 0.95
        f.image(p0 + dxk, rowc - ps_ / 2 + dyk + 1.4, ps_, ps_, f"plane_{4 - k}.png", fit="xMidYMid slice")
        f.rect(p0 + dxk, rowc - ps_ / 2 + dyk + 1.4, ps_, ps_, rx=0, stroke="#ffffff", sw=0.6 * LW, chk=False)
    f.text(p0 + 5.6, iy + 20.0, "B·Z×1×256×256", F_N, c=MUTE, a="middle")
    a1 = p0 + ps_ + 3 * 1.25 + 0.5
    harrow(f, a1, a1 + 2.4, rowc, S2D)
    gx0, gsz = a1 + 2.9, 10.0
    gy0 = rowc - gsz / 2 + 0.6
    f.image(gx0, gy0, gsz, gsz, "plane_2.png", fit="xMidYMid slice")
    lines = "".join(f"M{gx0 + k * gsz / 32:.2f},{gy0:.2f} V{gy0 + gsz:.2f} M{gx0:.2f},{gy0 + k * gsz / 32:.2f} H{gx0 + gsz:.2f} "
                    for k in range(1, 32))
    f.add(f'<path d="{lines}" stroke="#ffffff" stroke-width="{0.25 * PT:.3f}" opacity="0.45" fill="none"/>')
    pr, pc = A8LOG["patch"]["patch_row_col"]
    hx0, hy0, hs = gx0 + pc * gsz / 32, gy0 + pr * gsz / 32, gsz / 32
    f.rect(hx0 - 0.15, hy0 - 0.15, hs + 0.3, hs + 0.3, rx=0, stroke=RED, sw=0.9 * LW, chk=False)
    f.text(gx0 + gsz / 2, iy + 20.0, "32×32 patches", F_N, c=MUTE, a="middle")
    zx0, zs = gx0 + gsz + 1.6, 5.6
    zy0 = gy0 + 0.2
    f.image(zx0, zy0, zs, zs, "../assets8d/patch8.png")
    f.rect(zx0, zy0, zs, zs, rx=0, stroke=RED, sw=0.7 * LW, chk=False)
    for p_, q_ in (((hx0 + hs + 0.15, hy0 - 0.15), (zx0, zy0)), ((hx0 + hs + 0.15, hy0 + hs + 0.15), (zx0, zy0 + zs))):
        f.path(f"M{p_[0]:.2f},{p_[1]:.2f} L{q_[0]:.2f},{q_[1]:.2f}", stroke=RED, arrow=False, dash="0.5 0.4", chk=False, sw=0.5 * LW)
    f.text(zx0 + zs / 2, zy0 + zs + 2.3, "8×8 px", F_N, c=MUTE, a="middle")
    tkx = zx0 + zs + 0.9
    for q in range(12):
        f.rect(tkx, zy0 - 0.4 + q * 0.62, 1.3, 0.55, rx=0, fill=tint(BASE, 0.25 + 0.06 * (q % 5)), stroke="none", chk=False)
    f.rect(tkx, zy0 - 0.4, 1.3, 12 * 0.62 - 0.07, rx=0.2, stroke=C2D["st"], sw=0.5 * LW, chk=False)
    f.text(tkx + 0.65, zy0 + 10.7, "1024", F_N, 700, c=S2D, a="middle")
    tl = ["8×8 conv, stride 8 → one 1024-d token per patch;",
          "B·Z×32×32×1024 → + pos. → 24 blocks → neck (1×1, LN,",
          "3×3, LN) → B·Z×256×32×32 → view, permute → B×256×Z×32×32"]
    for k, t in enumerate(tl):
        f.text(ex + 1.6, iy + 22.6 + k * 2.2, t, F_N, c=INK if k else MUTE)
    f.prov_add("tokenisation: the volume (B,1,Z,256,256) is reshaped to B*Z planes (B*Z,1,256,256); each 8x8-pixel patch -> one 1024-d token (8x8 conv, stride 8); 32x32 tokens per plane; + absolute position embedding; 24 blocks; neck 1x1 conv 1024->256, LayerNorm2d, 3x3 conv, LayerNorm2d -> (B*Z,256,32,32); view + permute -> (B,256,Z,32,32). Shown plane = input plane_2; highlighted patch row 15, col 21 (assets8d/patch8.png)",
               "net3d.py:285-288; cellpose vit.py:109-116; segment_anything image_encoder.py:88-104; make_a8d_assets.py (patch)")

    # ---- ii: transformer block with global-attention token motif
    iy2, ih2 = iy + ih + 1.4, cy0 + ch - 1.2 - (iy + ih + 1.4)
    inset8(f, ex, iy2, ew, ih2, "ii", "transformer block (×24)", "#4a5577", "tokens 32×32×1024")
    seq = [("LN", 4.2), ("MHSA", 9.0), ("+", 0), ("LN", 4.2), ("MLP", 9.0), ("+", 0)]
    bx, by_, bh_ = ex + 4.2, iy2 + 8.4, 5.0
    xs_ = []
    for i, (nm, bw_) in enumerate(seq):
        if nm == "+":
            f.plus_node(bx + 1.3, by_ + bh_ / 2, r=1.2, col=S2D)
            xs_.append(bx + 1.3)
            bx += 2.6
        else:
            cub(f, bx, by_, bw_, bh_, 1.1, C2D, nm)
            xs_.append(bx + bw_ / 2)
            bx += bw_ + 1.1 * DEP[0]
        if i < 5:
            harrow(f, bx + 0.15, bx + 1.85, by_ + bh_ / 2, S2D)
            bx += 2.0
    for a_, b_ in ((ex + 2.4, xs_[2]), (xs_[2] + 1.8, xs_[5])):
        f.path(f"M{a_:.2f},{by_ + bh_ / 2:.2f} V{by_ - 2.0:.2f} H{b_:.2f} V{by_ + bh_ / 2 - 1.4:.2f}", stroke=S2D, sw=0.9 * LW)
    f.path(f"M{ex + 1.0:.2f},{by_ + bh_ / 2:.2f} H{ex + 4.0:.2f}", stroke=S2D, sw=0.9 * LW, arrow=False)
    f.text(xs_[1], by_ + bh_ + 2.5, ["16 heads, global,", "rel. position"], F_N, c=MUTE, a="middle")
    f.text(xs_[4], by_ + bh_ + 2.5, ["1024→4096", "→1024, GELU"], F_N, c=MUTE, a="middle")
    # token grid: one query token attends to every token of its plane
    n_, cs = 8, 1.32
    tgx, tgy = ex + 2.2, iy2 + 19.0
    for r in range(n_):
        for c in range(n_):
            f.rect(tgx + c * cs, tgy + r * cs, cs - 0.22, cs - 0.22, rx=0.15, fill=tint(BASE, 0.30), stroke="none", chk=False)
    qr, qc = 3, 4
    qxc, qyc = tgx + qc * cs + (cs - 0.22) / 2, tgy + qr * cs + (cs - 0.22) / 2
    att = []
    for r in range(n_):
        for c in range(n_):
            if (r, c) != (qr, qc):
                att.append(f"M{qxc:.2f},{qyc:.2f} L{tgx + c * cs + (cs - 0.22) / 2:.2f},{tgy + r * cs + (cs - 0.22) / 2:.2f}")
    f.add(f'<path d="{" ".join(att)}" stroke="{S2D}" stroke-width="{0.25 * PT:.3f}" opacity="0.55" fill="none"/>')
    f.rect(tgx + qc * cs, tgy + qr * cs, cs - 0.22, cs - 0.22, rx=0.15, fill=S2D, stroke="none", chk=False)
    gend = tgx + n_ * cs + 1.6
    al = ["global attention: one token (dark) attends to",
          "all 32×32 = 1,024 tokens of its own plane;",
          "planes never exchange information here",
          "patch embed frozen; pos. embed, blocks and",
          "neck fine-tuned; checkpointed per block"]
    for k, t in enumerate(al):
        f.text(gend, iy2 + 20.4 + k * 2.2, t, F_N, c=INK if k < 3 else MUTE)
    f.prov_add("transformer block: x + MHSA(LN(x)); x + MLP(LN(x)); 16 heads, relative position bias, global attention (Cellpose-SAM window_size 0) over the 32x32 = 1,024 tokens of one plane (planes are separate batch items); MLP 1024->4096->1024 GELU; per-block gradient checkpointing; patch embed frozen, everything else in the encoder trained",
               "segment_anything image_encoder.py Block l.151-175; build_sam.py:27-34, 66-80; cellpose vit.py:118-120; net3d.py:202-205, 285-286; main.tex Methods '3D model' (per-block gradient checkpointing)")

    # ---- iii: dilated 3x3x3 kernels along z, receptive-field cone, residual paths
    rxi, rwi = xII + 1.2, wII - 2.4
    ih3 = cy0 + ch - 1.2 - iy
    inset8(f, rxi, iy, rwi, ih3, "iii", "residual block (×4) and z-dilation", PAL["new_d"], "256×Z×32×32")
    funnel9(f, (BOX_III[0], BOX_III[3], BOX_III[2]), (rxi, iy, rxi + rwi), PAL["new_l"])
    seq2 = [("GN", 4.0), ("GELU", 5.8), ("conv 3^{3}", 9.4), ("GN", 4.0), ("GELU", 5.8), ("conv 3^{3}", 9.4)]
    bx, by_ = rxi + 4.2, iy + 8.4
    first = bx
    for i, (nm, bw_) in enumerate(seq2):
        conv = nm.startswith("conv")
        cub(f, bx, by_, bw_, bh_, 1.1, C3D if conv else dict(front="#ffffff", top=tint(RED, 0.05), side=tint(RED, 0.16), st=C3D["st"]), nm, bold=conv)
        if i == 2:
            f.text(bx + bw_ / 2, by_ + bh_ + 2.5, "dilation (d,1,1)", F_N, 700, c=PAL["new_d"], a="middle")
        if i == 5:
            f.text(bx + bw_ / 2, by_ + bh_ + 2.5, "dilation 1", F_N, c=MUTE, a="middle")
        bx += bw_ + 1.1 * DEP[0]
        if i < 5:
            harrow(f, bx + 0.15, bx + 1.55, by_ + bh_ / 2, S3D)
            bx += 1.7
    pxn = bx + 2.6
    f.plus_node(pxn, by_ + bh_ / 2, r=1.2, col=S3D)
    harrow(f, bx + 0.15, pxn - 1.35, by_ + bh_ / 2, S3D)
    f.path(f"M{first - 1.6:.2f},{by_ + bh_ / 2:.2f} V{by_ - 2.0:.2f} H{pxn:.2f} V{by_ + bh_ / 2 - 1.4:.2f}", stroke=S3D, sw=0.9 * LW)
    f.path(f"M{rxi + 1.0:.2f},{by_ + bh_ / 2:.2f} H{first - 0.2:.2f}", stroke=S3D, sw=0.9 * LW, arrow=False)
    f.text(rxi + 1.6, iy + 19.2, "GroupNorm (16 groups) · GELU · 256 channels", F_N, c=MUTE)
    # kernel taps along z
    pz = 0.66
    zt = iy + 24.6
    z0 = zt + 19 * pz
    colx = [rxi + 13.0 + j * 13.4 for j in range(4)]
    kw, kc, kd = 5.0, 3, 2.4
    # cone first (behind)
    hw_ = [(rf[j + 1] - 1) / 2 * pz for j in range(4)]
    cone = [(colx[0] - 1.0, z0 - hw_[0])] + [(colx[j] + kw + kd * DEP[0] + 0.6, z0 - hw_[j]) for j in range(4)] + \
           [(colx[j] + kw + kd * DEP[0] + 0.6, z0 + hw_[j]) for j in range(3, -1, -1)] + [(colx[0] - 1.0, z0 + hw_[0])]
    f.poly(cone, tint(RED, 0.10), None, chk=False)
    for j, dd in enumerate((1, 2, 4, 8)):
        x0_ = colx[j]
        # plane ticks (39 planes)
        ticks = "".join(f"M{x0_ - 0.6:.2f},{zt + q * pz:.2f} h{kw + kd * DEP[0] + 1.0:.2f} " for q in range(39))
        f.add(f'<path d="{ticks}" stroke="{tint(BASE, 0.35)}" stroke-width="{0.25 * PT:.3f}" fill="none"/>')
        for off in (-dd, 0, dd):
            yy = z0 + off * pz
            cw_ = kw / kc
            ddx, ddy = kd * DEP[0] / kc, kd * DEP[1] / kc
            for a_ in range(kc):            # depth rows back to front
                for b_ in range(kc):
                    xa = x0_ + b_ * cw_ + (kc - 1 - a_) * ddx
                    ya = yy + (kc - 1 - a_) * ddy
                    pts = [(xa, ya), (xa + cw_, ya), (xa + cw_ + ddx, ya + ddy), (xa + ddx, ya + ddy)]
                    f.poly(pts, tint(RED, 0.55) if off else RED, "#ffffff", 0.25 * PT, chk=False)
        f.text(x0_ + kw / 2 + 0.8, zt - 2.0, f"d = {dd}", F_N, 700, c=PAL["new_d"], a="middle")
        bxb = x0_ + kw + kd * DEP[0] + 1.3
        f.add(f'<path d="M{bxb:.2f},{z0 - hw_[j]:.2f} h0.6 V{z0 + hw_[j]:.2f} h-0.6" stroke="{PAL["new_d"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
        f.text(bxb + 1.0, z0 + 0.8, f"{rf[j + 1]}", F_L, 700, c=PAL["new_d"])
    f.text(rxi + 1.6, zt - 2.0, "planes", F_N, c=MUTE)
    f.text(rxi + 5.4, z0 - 19 * pz + 0.7, "−19", F_N, c=MUTE, a="end")
    f.text(rxi + 5.4, z0 + 0.7, "0", F_N, c=MUTE, a="end")
    f.text(rxi + 5.4, z0 + 19 * pz + 0.7, "+19", F_N, c=MUTE, a="end")
    f.path(f"M{rxi + 7.6:.2f},{zt:.2f} V{zt + 38 * pz:.2f}", stroke=MUTE, sw=0.6 * LW, chk=False)
    f.text(rxi + 8.4, zt + 38 * pz - 0.4, "z", F_N, c=MUTE, italic=True)
    # residual path strip
    ry = zt + 38 * pz + 3.0
    rx0 = colx[0] - 1.0
    for j in range(4):
        bxj = colx[j] + 0.2
        cub(f, bxj, ry, 5.0, 3.0, 1.0, C3D, f"{j + 1}", size=F_N)
        pnx = bxj + 5.0 + 1.0 * DEP[0] + 2.6
        f.plus_node(pnx, ry + 1.5, r=0.9, col=S3D)
        harrow(f, bxj + 5.0 + 1.0 * DEP[0] + 0.1, pnx - 1.0, ry + 1.5, S3D)
        f.path(f"M{bxj - 1.0:.2f},{ry + 1.5:.2f} V{ry - 1.0:.2f} H{pnx:.2f} V{ry + 0.55:.2f}", stroke=S3D, sw=0.7 * LW)
        if j < 3:
            harrow(f, pnx + 0.95, colx[j + 1] - 1.0 + 0.0, ry + 1.5, S3D)
        else:
            harrow(f, pnx + 0.95, pnx + 3.0, ry + 1.5, S3D)
    harrow(f, rx0 - 1.6, colx[0] + 0.15, ry + 1.5, S3D)
    f.text(rxi + 1.6, ry + 2.3, "x + F(x)", F_N, c=MUTE)
    f.text(rxi + 1.6, iy + ih3 - 1.5, "red: 3×3×3 taps at z − d, z, z + d; shade: z receptive field after the block", F_N, c=MUTE)
    f.prov_add("dilated kernel diagram: first 3x3x3 conv of block i samples planes z-d, z, z+d (d = 1, 2, 4, 8; dilation (d,1,1)), second conv undilated; z receptive field after blocks 1-4 = 5, 11, 21, 39 planes (cone, +-19 planes after block 4); residual x + F(x) per block; GroupNorm 16 groups, GELU, 256 channels",
               "net3d.py:69-78 ResBlock3D, :72 dilation=(zdil,1,1); model3d.json dec_zdil [1,2,4,8], dec_groups 16, dec_ch 256; main.tex Methods '3D model'")

    # ---- iv: zero-init composition, pixel shuffle, real output maps
    zxi, zwi = xIII + 1.2, wIII - 2.4
    ih4 = 38.6
    inset8(f, zxi, iy, zwi, ih4, "iv", "output composition", PAL["new_d"])
    funnel9(f, (BOX_IV[0], BOX_IV[3], BOX_IV[2]), (zxi, iy, zxi + zwi), PAL["new_l"])
    cA, cB = zxi + 8.6, zxi + 19.8
    f.text(cA, iy + 7.2, "step 0", F_N, 700, a="middle")
    f.text(cB, iy + 7.2, "trained", F_N, 700, c=PAL["new_d"], a="middle")
    rowsZ = [("d_{z}", "0", "Δ_{z}"), ("d_{y}", "2D", "2D+Δ"), ("d_{x}", "2D", "2D+Δ"), ("p", "2D", "2D+Δ")]
    for k_, (chn, a_, b_) in enumerate(rowsZ):
        yy = iy + 8.6 + k_ * 3.7
        f.text(zxi + 1.6, yy + 2.3, chn, F_N)
        zero = a_ == "0"
        cub(f, cA - 3.6, yy, 7.2, 3.0, 0.8, CZ if zero else C2D, a_, size=F_N, bold=zero, dash="0.7 0.45" if zero else None, lc=RED if zero else INK)
        cub(f, cB - 4.4, yy, 8.8, 3.0, 0.8, C3D, b_, size=F_N, bold=False)
    harrow(f, cA + 4.4, cB - 4.8, iy + 8.6 + 3.7 + 1.5, INK)
    # pixel shuffle: 64 channels of one token -> 8x8 pixel block
    import colorsys  # noqa: F401
    def ramp(i):
        t = i / 63.0
        c0, c1 = (0xF3, 0xD4, 0xCE), (0xB8, 0x3A, 0x28)
        return "#%02X%02X%02X" % tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
    sx0, sy0, sh = zxi + 30.0, iy + 8.2, 13.6
    bhh = sh / 64
    for i in range(64):
        f.add(f'<rect x="{sx0:.2f}" y="{sy0 + i * bhh:.3f}" width="1.6" height="{bhh + 0.01:.3f}" fill="{ramp(i)}"/>')
    f.rect(sx0, sy0, 1.6, sh, rx=0, stroke=C3D["st"], sw=0.5 * LW, chk=False)
    f.text(sx0 + 0.8, sy0 - 1.0, "64 ch", F_N, c=MUTE, a="middle")
    gxs, cg = sx0 + 6.0, 1.45
    for i in range(64):
        r_, c_ = divmod(i, 8)
        f.add(f'<rect x="{gxs + c_ * cg:.2f}" y="{sy0 + 1.4 + r_ * cg:.2f}" width="{cg - 0.12:.2f}" height="{cg - 0.12:.2f}" fill="{ramp(i)}"/>')
    harrow(f, sx0 + 2.0, gxs - 0.4, sy0 + sh / 2, INK)
    f.text(gxs + 4 * cg, sy0 - 1.0, "8×8 px", F_N, c=MUTE, a="middle")
    f.text(zxi + 1.6, iy + 25.6, "×8 pixel shuffle: each token's 64 channels per", F_N, c=MUTE)
    f.text(zxi + 1.6, iy + 27.8, "map fill its 8×8 pixels (4·64 → 4 maps)", F_N, c=MUTE)
    # four real output maps
    tw4 = (zwi - 3.2 - 3 * 1.0) / 4
    th4 = tw4 * 28 / 48
    ty4 = iy + 29.4
    for k, (nm, lab) in enumerate((("dz", "d_{z}"), ("dy", "d_{y}"), ("dx", "d_{x}"), ("p", "p"))):
        tx4 = zxi + 1.6 + k * (tw4 + 1.0)
        f.image(tx4, ty4, tw4, th4, f"../assets8d/out_{nm}.png")
        f.text(tx4 + tw4 / 2, ty4 + th4 + 2.2, lab, F_N, 700, a="middle")
    f.prov_add("output composition (dz, dy, dx, p) = (Delta_z, 2D head + Delta); Delta = 0 at step 0 so S1 = 2D control with dz = 0; output conv 256 -> 4*64 = 4 x 8 x 8 values per 8 x 8-pixel token, pixel shuffle x8 rearranges the 64 channels of each map into the token's 8x8 pixels",
               "net3d.py:89-90, 294-299 (F.pixel_shuffle(tok, 8)); main.tex Methods '3D model' ('4 x 8 x 8 values per 8 x 8-pixel token')")
    f.prov_add("output maps: S1 seed-0 raw outputs d_z, d_y, d_x (shown /5, palette diverging, clipped to +-1) and p (sigmoid of logit) on the xz section through GT nucleus 62 of E15_Ab2 (identity view, one 32-plane window); outline = nucleus 62",
               "assets8d/out_*.png from make_a8d_assets.py <- assets3/views_E15_Ab2_n62.npz (arch_views.py on the server)")

    # ---- v: diffusion target and loss
    iy5 = iy + ih4 + 1.4
    ih5 = cy0 + ch - 1.2 - iy5
    inset8(f, zxi, iy5, zwi, ih5, "v", "3D target and loss", LOSS)
    ht = 7.4
    hw5 = ht * 56 / 49
    hx5 = zxi + 1.6
    for k, s in enumerate((10, 40, 160, 522)):
        xk = hx5 + k * (hw5 + 0.9)
        f.image(xk, iy5 + 5.6, hw5, ht, f"../assets8d/heat_{s}.png")
        f.text(xk + hw5 / 2, iy5 + 5.6 + ht + 2.1, str(s), F_N, c=MUTE, a="middle")
    fx5 = hx5 + 4 * (hw5 + 0.9) + 2.6
    harrow(f, fx5 - 2.4, fx5 - 0.3, iy5 + 5.6 + ht / 2, INK)
    f.image(fx5, iy5 + 5.0, 8.4, 8.4, "../assets3/diff_flow.png")
    f.text(fx5 + 4.2, iy5 + 5.6 + ht + 2.1, "u", F_N, 700, a="middle", italic=True)
    f.text(zxi + 1.6, iy5 + 17.4, "heat T from the central voxel, iterations (log)", F_N, c=MUTE)
    f.text(zxi + 1.6, iy5 + 20.7, "u = ∇T/∣∇T∣; ½ MSE(flows, 5*u*) + BCE(p, fg)", F_N, 700, c=LOSS)
    f.prov_add(f"target: heat source at the most central voxel of the GT nucleus, diffusion restricted to the nucleus for niter = {DF['niter']} iterations (nucleus 62), unit gradient = target flow u; snapshots after 10, 40, 160, 522 iterations (xz slice through the source, log scale); loss = MSE(flows, 5u)/2 + BCE(p, fg)",
               "data3d.py:28-81 flows3d_gpu; assets3/diffusion.npz (diffusion_data.py) -> make_a8d_assets.py heat_*.png; render_diffusion.py diff_flow.png; train3d.py:33-43")

    # ================= rationale notes + stream legend
    ry = cy0 + ch + 3.3
    notes = ["per-plane encoding keeps the 2D foundation model's representations, which native 3D models do not inherit",
             "axial resolution is several-fold poorer than lateral: dilation widens z context to 39 planes without extra weights",
             "zero-initialised output: the network starts as the 2D model and learns only the volumetric correction",
             "the axial flow comes from 3D context, not from summing flows predicted on separate orthogonal slices"]
    for k_, t_ in enumerate(notes):
        colx_ = ox + 0.5 + (k_ % 2) * 91.8
        rowy = ry + (k_ // 2) * 3.4
        num(f, colx_ + 1.6, rowy - 0.65, k_ + 1)
        f.text(colx_ + 3.9, rowy, t_, F_N)
    f.prov_add("rationale 1-4 wording", "main.tex Introduction l.44-50 (as in gen_a4.panel_arch4)")
    y = ry + 4.6
    h = 4.2
    f.rect(ox + 0.5, y, w - 1.0, h, rx=1.0, fill="none", stroke="#B9B9BD", sw=0.55 * LW, dash="1.2 0.8", chk=False)
    yy = y + h / 2
    sitems = [("arrow", S2D, None, "per-plane 2D stream"), ("arrow", S3D, None, "3D stream"),
              ("arrow", LOSS, "0.9 0.6", "loss vs 3D target"), ("plus", INK, None, "sum"),
              ("arrow", INK, None, "composition / output"), ("dots", INK, None, "repeated"), ("funnel", MUTE, None, "dashed box → zoom inset"), ("tag", MUTE, None, "inset key")]
    cx = ox + 3.0
    for kind, col, dash, lab in sitems:
        if kind == "arrow":
            f.arrow(cx, yy, cx + 5.0, yy, stroke=col, sw=1.4 * LW, dash=dash, chk=False)
            tx = cx + 6.2
        elif kind == "plus":
            f.plus_node(cx + 1.3, yy, r=1.1)
            tx = cx + 3.4
        elif kind == "dots":
            f.text(cx + 1.8, yy + 0.5, "···", F_L, 700, a="middle")
            tx = cx + 4.4
        elif kind == "tag":
            tag(f, cx, yy - 1.35, "i")
            tx = cx + 4.0
        else:
            f.path(f"M{cx:.2f},{yy - 1.2:.2f} L{cx + 5.0:.2f},{yy + 1.2:.2f}", stroke="#9a9aa0", arrow=False, dash="0.7 0.5", chk=False, sw=0.6 * LW)
            tx = cx + 6.4
        f.text(tx, yy + 0.65, lab, F_N)
        cx = tx + tw(lab, F_N) + 3.6
    return y + h


# ======================================================================== panel b: training schedule
def lr_spark(f, x, y, w, h, kind):
    """Shape of the learning-rate schedule stated in Methods (not to scale between models)."""
    f.add(f'<path d="M{x:.2f},{y:.2f} V{y + h:.2f} H{x + w:.2f}" stroke="{MUTE}" stroke-width="{0.5 * LW:.3f}" fill="none"/>')
    pts = []
    if kind == "2d":                     # warm-up 10 epochs, flat to 50, halved every 5 epochs to 100
        for e in range(0, 101):
            v = min(e / 10, 1.0) if e <= 50 else 0.5 ** ((e - 50) // 5 + 1) if e > 50 else 1.0
            if 45 < e <= 50:
                v = 1.0
            pts.append((e / 100, v))
    else:                                # linear warm-up then cosine decay
        wu, n = (300, 20000) if kind == "a1" else (100, 3000)
        for i in range(0, 201):
            s = n * i / 200
            v = s / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / (n - wu)))
            pts.append((s / n, v))
    d = "M" + " L".join(f"{x + 0.3 + a * (w - 0.6):.2f},{y + h - 0.2 - b * (h - 0.6):.2f}" for a, b in pts)
    f.add(f'<path d="{d}" stroke="{INK}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')


def panel_sched8(f, ox, oy, w, letter="b"):
    plabel(f, ox, oy + 3.4, letter, "Training schedule")
    f.text(ox + w - 0.5, oy + 3.4, "every stage with five seeds (0, 1000, 2000, 3000, 4000); each 3D model starts from the seed-matched 2D model",
           F_N, c=MUTE, a="end")
    by, bh = oy + 5.4, 21.0
    section(f, ox, by, w, bh, BLOCK)
    cy_, ch_ = by + 2.2, 16.6
    ym = cy_ + ch_ / 2

    def card(x, y, wd, ht, head, lines, fill, dark, ghost=True, bold_lines=()):
        if ghost:
            for k in (3, 2, 1):
                f.rect(x + k * 0.7, y - k * 0.6, wd, ht, rx=1.0, fill="#ffffff", stroke=RULE, sw=0.5 * LW, op=0.9, chk=False)
        f.rect(x, y, wd, ht, rx=1.0, fill="#ffffff", stroke=RULE, sw=0.6 * LW, chk=False)
        f.add(f'<path d="M{x:.2f},{y + 3.9:.2f} V{y + 1.0:.2f} Q{x:.2f},{y:.2f} {x + 1.0:.2f},{y:.2f} H{x + wd - 1.0:.2f} '
              f'Q{x + wd:.2f},{y:.2f} {x + wd:.2f},{y + 1.0:.2f} V{y + 3.9:.2f} Z" fill="{fill}" stroke="none"/>')
        f.text(x + 1.4, y + 2.85, head, F_L, 700, c=dark)
        for k, t in enumerate(lines):
            f.text(x + 1.4, y + 6.4 + k * 2.45, t, F_N, 700 if k in bold_lines else 400, c=INK if k in bold_lines else MUTE)

    x = ox + 1.6
    w0 = 19.0
    card(x, cy_, w0, ch_, "Cellpose-SAM", ["released weights", "(cpsam_v2)", "2D ViT-L encoder", "+ 1×1 conv head"],
         tint(BASE, 0.22), "#4a5577", ghost=False)
    x1 = x + w0 + 3.2
    harrow(f, x + w0 + 0.3, x1 - 0.3, ym, S2D)
    w1 = 43.0
    card(x1, cy_, w1, ch_, "2D fine-tuning → 2D control", ["AdamW, lr 10^{−5}, wd 0.1", "6,056 slices (xy, xz, yz)", "+ crypt sections; batch 1;", "100 epochs × 1,000 slices"],
         tint(PAL["ctl"], 0.30), PAL["ctl_d"])
    lr_spark(f, x1 + w1 - 11.4, cy_ + 7.0, 10.0, 6.0, "2d")
    f.text(x1 + w1 - 6.4, cy_ + 15.0, "lr", F_N, c=MUTE, a="middle")
    x2 = x1 + w1 + 4.6
    harrow(f, x1 + w1 + 0.3, x2 - 0.3, ym, S2D)
    f.text((x1 + w1 + x2) / 2, ym - 1.0, "init", F_N, c=MUTE, a="middle")
    w2 = 43.0
    card(x2, cy_, w2, ch_, "A1 · 3D model", ["lr 10^{−5} enc / 2·10^{−4} dec", "910 volumes; 32×256×256", "20,000 steps; wd 0.05;", "300 warm-up, cosine decay"],
         tint(RED, 0.22), PAL["new_d"])
    lr_spark(f, x2 + w2 - 11.4, cy_ + 7.0, 10.0, 6.0, "a1")
    f.text(x2 + w2 - 6.4, cy_ + 15.0, "lr", F_N, c=MUTE, a="middle")
    x3 = x2 + w2 + 5.0
    w3 = 41.0
    yC, hC = cy_ - 0.6, 6.4
    yX, hX = cy_ + 6.4, ch_ - 6.0
    for (yy, hh3, head, lns) in ((yC, hC, "C · +3,000 steps", ["unchanged data; 100 warm-up"]),
                                 (yX, hX, "X1 · +3,000 steps", ["simulated lateral downsampling:", "p 0.33, interval 2–10 voxels,", "blur 0.3–0.6 × interval"])):
        for k in (2, 1):
            f.rect(x3 + k * 0.6, yy - k * 0.25, w3, hh3, rx=1.0, fill="#ffffff", stroke=RULE, sw=0.5 * LW, op=0.9, chk=False)
        f.rect(x3, yy, w3, hh3, rx=1.0, fill="#ffffff", stroke=RULE, sw=0.6 * LW, chk=False)
        f.rect(x3, yy, 1.0, hh3, rx=0.4, fill=RED, stroke="none", chk=False)
        f.text(x3 + 2.0, yy + 2.8, head, F_L, 700, c=PAL["new_d"])
        for k, ln in enumerate(lns):
            f.text(x3 + 2.0, yy + 5.0 + k * 2.15, ln, F_N, c=MUTE)
    hh3 = hC
    xs_ = x2 + w2 + 0.3
    for yy in (yC + hC / 2, yX + hX / 2):
        f.path(f"M{xs_:.2f},{ym:.2f} C{xs_ + 2.8:.2f},{ym:.2f} {x3 - 2.8:.2f},{yy:.2f} {x3 - 0.3:.2f},{yy:.2f}", stroke=S3D, sw=1.2 * LW)
    # weight average glyph
    gx0 = x3 + w3 + 2.6
    o = (gx0, ym)
    tC, tX = (gx0 + 4.6, ym - 5.0), (gx0 + 5.6, ym + 5.0)
    mid = ((tC[0] + tX[0]) / 2, (tC[1] + tX[1]) / 2)
    f.arrow(o[0], o[1], tC[0], tC[1], stroke=PAL["new_d"], sw=0.8 * LW)
    f.arrow(o[0], o[1], tX[0], tX[1], stroke=PAL["new_d"], sw=0.8 * LW)
    f.path(f"M{tC[0]:.2f},{tC[1]:.2f} L{tX[0]:.2f},{tX[1]:.2f}", stroke=MUTE, arrow=False, dash="0.6 0.4", sw=0.6 * LW)
    f.circle(mid[0], mid[1], 0.9, fill=RED, stroke="#ffffff", sw=0.5 * LW)
    f.circle(o[0], o[1], 0.45, fill=INK, chk=False)
    f.text(tC[0], tC[1] - 1.9, "θ_{C}", F_N, c=PAL["new_d"], a="middle")
    f.text(tX[0], tX[1] + 2.6, "θ_{X1}", F_N, c=PAL["new_d"], a="middle")
    sxb = ox + w - 1.6 - 11.5
    f.arrow(mid[0] + 1.1, mid[1], sxb - 0.3, mid[1], sw=1.0 * LW)
    for k in (3, 2, 1):
        f.rect(sxb + k * 0.5, ym - 6.0 - k * 0.5, 11.5, 12.0, rx=1.2, fill=RED, stroke="none", op=0.25, chk=False)
    f.rect(sxb, ym - 6.0, 11.5, 12.0, rx=1.2, fill=RED, stroke="none", chk=False)
    f.text(sxb + 5.75, ym - 0.6, "S1", 7.0, 700, c="#ffffff", a="middle")
    f.text(sxb + 5.75, ym + 3.4, "½(C + X1)", F_N, 700, c="#ffffff", a="middle")
    f.prov_add("2D fine-tuning (2D control): Cellpose-SAM cpsam_v2 on 6,056 2D slices (xy and isotropically resampled xz, yz planes) plus crypt sections; AdamW lr 1e-5, wd 0.1, batch 1, 100 epochs of 1,000 slices; lr linear warm-up 10 epochs, constant to epoch 50, then halved every 5 epochs; random 256x256 crops; five seeds",
               "main.tex Methods '2D fine-tuning' (l.156)")
    f.prov_add("A1: 3D model from the seed-matched 2D model, 910 volumes, 20,000 steps on 32x256x256 patches, AdamW lr 1e-5 encoder / 2e-4 decoder, wd 0.05, 300 warm-up steps, cosine decay",
               "main.tex Methods '3D model' (l.159); main.tex Results l.63 (910 volumes); model3d.json init")
    f.prov_add("C and X1: 3,000 further steps from A1, same optimiser, 100 warm-up steps; X1 with simulated lateral downsampling (p 0.33, interval 2-10 network voxels, Gaussian blur 0.3-0.6 x interval); S1 = parameter-wise average (C + X1)/2, weight fixed before evaluation; five seeds per stage",
               "main.tex Methods '3D model' (l.159); orx_t3d_r5s1_s0/model3d.json avg_of, avg_alpha 0.5")
    f.prov_add("lr sparklines: shapes of the schedules stated in Methods (2D: warm-up 10 epochs, flat to 50, halved every 5; 3D: linear warm-up then cosine); axes not to scale", "main.tex Methods l.156, l.159")


# ======================================================================== layout
def layout_a8d():
    f = Fig(183.0, H_PAGE)
    ya = panel_arch8(f, 0.0, 0.0, letter="a")
    ob = ya + 1.4
    panel_sched8(f, 0.0, ob, 183.0, letter="b")
    oc = ob + 5.4 + 21.0 + 1.4
    G.panel_cmp7(f, 0.0, oc, 119.6, letter="c", bh=32.0, ph=11.6)
    G.panel_target7(f, 121.0, oc, 61.5, letter="d", bh=32.0, th=16.5)
    oe = oc + 5.4 + 32.0 + 1.2
    G.panel_infer7(f, 0.0, oe, 183.0, letter="e", bottom=H_PAGE)
    return f


def main():
    cid = "A8-d"
    G.ensure_crops()
    f = layout_a8d()
    res = export(f, cid)
    print(cid, "issues:", len(res.get("issues", [])), "min font", res.get("min_font_pt"))
    for s_ in res.get("issues", [])[:80]:
        print("   ", s_)
    json.dump(dict(figure=cid, size_mm=[183.0, H_PAGE], code_snapshot=_nsp_os.path.join(_NSP_REPO, 'src'),
                   model="orx_t3d_r5s1_s0/model3d.json", palette="paper/figures/palette.json; blocks #EDECE9 (3D-viewer scheme, user 11-11)",
                   generator="paper/figures/arch/gen_a8d.py (panel_arch8, panel_sched8; panels c-e = gen_a7d.panel_cmp7/target7/infer7 compacted)",
                   new_assets=dict(script="paper/figures/arch/make_a8d_assets.py", log=A8LOG),
                   references_web=[dict(figure=a, technique=b) for a, b in WEB_REFS],
                   references_style=[dict(figure=a, technique=b) for a, b in G.REFS7 + REFS4], items=f.prov),
              open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)
    gal = os.path.join(HERE, "..", "gallery", "assets")
    for ext in ("pdf", "png", "provenance.json"):
        shutil.copy(os.path.join(HERE, f"{cid}.{ext}"), os.path.join(gal, f"{cid}.{ext}"))


if __name__ == "__main__":
    main()
