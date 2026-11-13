"""Fig. 1 variant A7-d ("dense Nature multi-panel"): all four panels redrawn in one coherent system.

Panel a keeps every fact of gen_a4.panel_arch4 (S1 architecture) and draws it with pastel 3D cuboid modules, stacked ViT blocks,
tensor slabs, colour-coded streams, stage blocks with grey tabs and zoom insets joined by dashed funnels (reference: the user's
three-stage network figure). Panels b-d keep the data, assets and every number of gen_a3.panel_cmp2, gen_a4.panel_target3d and
gen_a4.panel_infer4, laid out on a strict grid with light tinted section blocks, grey tabs, one type hierarchy
(title 7 pt bold, sub-titles 6 pt bold, labels 5.5 pt, notes 5 pt muted), enlarged renders and step cards with tinted header strips
(reference: Nature multi-panel figures 0.png, 1.png, 2.png). Hand-coded SVG -> vector PDF + 600-dpi PNG via gen_fig.export.
Cropped copies of existing renders (whitespace trimmed only, no pixel edits) are written to assets7d/.
Usage: python gen_a7d.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
_NSP_STYLE_REF = _nsp_os.environ.get('NUCSEG3D_STYLE_REF', '/path/to/style_reference_images')
import json
import math
import os
import shutil
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fig import LW, PAL, P as PJ, Fig, export, tint, M as PM  # noqa: E402
from gen_a3 import DEP, LOG3, LOGA, block  # noqa: E402
from gen_a4 import REFS as REFS4, hatch_defs, num  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A7 = os.path.join(HERE, "assets7d")
INK = PAL["fg"]
MUTE = PAL["mute"]
RD = json.load(open(os.path.join(HERE, "assets3", "ruleT_diam.json")))
DF = json.load(open(os.path.join(HERE, "assets3", "diffusion.json")))
PT = 0.3528

REFDIR = f"{_NSP_STYLE_REF}/"
REFS7 = [
    (REFDIR + "Weixin Image_2026-11-11_114115_289.png",
     "panel a: stages as pastel containers with grey corner tabs carrying roman-numeral names; modules as 3D cuboids "
     "(front face, lighter top, darker side) with the label inside; repeated blocks as a stack of cuboids with an ellipsis; "
     "tensors as translucent slab glyphs with dashed outline; arrows colour-coded by stream with elbow routing; stream legend "
     "in a dashed strip under the panel"),
    (REFDIR + "0.png",
     "dense multi-panel grid: every panel fills its cell, small consistent type, plots with light gridlines and thin axes, "
     "legends set inline beside the data"),
    (REFDIR + "1.png",
     "strict left/top alignment of panel letters and blocks; colour bars and legends in the panel's top-right corner; one muted grey "
     "for secondary annotations"),
    (REFDIR + "2.png",
     "light tinted section blocks behind groups of panels (pink / blue tints, no heavy borders), small section headers, "
     "workflow steps as boxes with tinted header strips, dashed group borders"),
]

# ---------------------------------------------------------------- type hierarchy (pt)
F_T, F_S, F_L, F_N = 7.0, 6.0, 5.5, 5.0

# ---------------------------------------------------------------- colours (all from palette.json)
BASE, RED, LOSS, CTL = PM["base"], PM["S1"], PAL["loss"], PM["2D"]
C2D = dict(front=tint(BASE, 0.24), top=tint(BASE, 0.12), side=tint(BASE, 0.45), st=tint(BASE, 0.95))
C3D = dict(front=tint(RED, 0.20), top=tint(RED, 0.10), side=tint(RED, 0.38), st=tint(RED, 0.80))
CZ = dict(front="#ffffff", top="#ffffff", side=tint(RED, 0.06), st=RED)
S2D = "#56658F"                  # per-plane 2D stream: palette 'base' darkened (as A6-b)
S3D = RED
# section blocks: one warm grey (user, 11-11: the Supplementary_Data_3D_viewer.html look); page and cards white.
# Model semantics are kept by a coloured accent bar on each tab (2D blue family, S1 red, target/loss brown).
BLOCK = "#EDECE9"
BG2D = BG3D = BGT = BGN = BLOCK
ACC2D, ACC3D, ACCT = "#56658F", RED, LOSS
TAB = "#DCDAD5"
TABT = "#3a3a3e"
RULE = PAL["rule"]
GRID = "#ECECEA"


# ======================================================================== cropped renders (whitespace trimmed only)
CROPS = {
    "diff_source.png": ("assets3/diff_source.png", (165, 60, 730, 748)),
    "diff_heat.png": ("assets3/diff_heat.png", (165, 60, 730, 748)),
    "target_F1d.png": ("assets/target_F1d.png", (30, 30, 1608, 1551)),
    "flowfollow_F1d.png": ("assets/flowfollow_F1d.png", (30, 30, 1608, 1551)),
    "cmp_2d_xyonly.png": ("assets3/cmp_2d_xyonly.png", (85, 63, 623, 461)),
    "cmp_s1_view1.png": ("assets3/cmp_s1_view1.png", (85, 63, 623, 461)),
    "ang_2d.png": ("assets3/ang_2d.png", (85, 63, 623, 461)),
    "ang_s1.png": ("assets3/ang_s1.png", (85, 63, 623, 461)),
    "view_z.png": ("assets3/view_z.png", (110, 60, 390, 270)),
    "view_y.png": ("assets3/view_y.png", (110, 60, 390, 270)),
    "view_x.png": ("assets3/view_x.png", (110, 60, 390, 270)),
    "view_mean.png": ("assets3/view_mean.png", (110, 60, 390, 270)),
    "flow_glyphs_pal.png": ("assets/flow_glyphs_pal.png", (115, 135, 875, 795)),
}


def ensure_crops():
    os.makedirs(A7, exist_ok=True)
    for dst, (src, box) in CROPS.items():
        p = os.path.join(A7, dst)
        im = Image.open(os.path.join(HERE, src))
        im.crop(box).save(p)
    return {k: dict(source=v[0], crop_box_px=v[1]) for k, v in CROPS.items()}


def a7(n):
    return "../assets7d/" + n


def aspect(n):
    w, h = Image.open(os.path.join(A7, n)).size
    return w / h


# ======================================================================== shared drawing helpers
def tw(s, size, bold=False):
    """Approximate rendered width (mm) of a label in Helvetica."""
    import re
    s = re.sub(r"[_^]\{([^}]*)\}", r"\1", s).replace("*", "")
    return len(s) * size * PT * (0.56 if bold else 0.52)


def plabel(f, x, y, letter, title):
    f.text(x, y, letter, 8.0, 700)
    f.text(x + 3.8, y, title, F_T, 700)


def section(f, x, y, w, h, fill, tab=None, tab_note=None, dash=None, border=None, accent=None):
    """Warm-grey block (no border) with an optional small grey tab in the top-left corner; accent = semantic colour bar."""
    f.rect(x, y, w, h, rx=1.4, fill=fill, stroke=border, sw=0.6 * LW, dash=dash, chk=False)
    if tab:
        a0 = 1.6 if accent else 0.0
        twd = tw(tab, F_S, True) + 3.2 + a0
        f.add(f'<path d="M{x:.2f},{y + 4.2:.2f} V{y + 1.4:.2f} Q{x:.2f},{y:.2f} {x + 1.4:.2f},{y:.2f} H{x + twd:.2f} '
              f'V{y + 4.2:.2f} Z" fill="{TAB}" stroke="none"/>')
        if accent:
            f.rect(x + 1.2, y + 1.0, 1.1, 2.3, rx=0.3, fill=accent, stroke="none", chk=False)
        f.text(x + 1.5 + a0, y + 3.05, tab, F_S, 700, c=TABT)
        if tab_note:
            f.text(x + twd + 1.4, y + 3.05, tab_note, F_N, c=MUTE, italic=True)
        return twd
    return 0


def cub(f, x, y, w, h, d, c, label=None, size=F_L, bold=True, hatch=False, dash=None, op=1.0, lc=None, sw=None):
    """Pastel cuboid (front x, y, w, h; depth d along 35 deg); lighter top, darker side; label centred on the front."""
    dx, dy = d * DEP[0], d * DEP[1]
    top = [(x, y), (x + w, y), (x + w + dx, y + dy), (x + dx, y + dy)]
    side = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    sw = sw or 0.8 * LW
    f.poly(top, c["top"], c["st"], sw, op=op, chk=False, dash=dash)
    f.poly(side, c["side"], c["st"], sw, op=op, chk=False, dash=dash)
    f.poly(front, "url(#hatch)" if hatch else c["front"], c["st"], sw, op=op, chk=False, dash=dash)
    if label:
        lines = label if isinstance(label, (list, tuple)) else [label]
        fs = size * PT
        y0 = y + h / 2 - (len(lines) - 1) * fs * 1.18 / 2 + fs * 0.36
        f.text(x + w / 2, y0, lines, size, 700 if bold else 400, c=lc or INK, a="middle")
    return x + w + dx, y + dy


def slab(f, x, y, w, h, d, n, c):
    """Tensor glyph: dashed outline holding n thin translucent slices and an ellipsis."""
    dx, dy = d * DEP[0], d * DEP[1]
    out = [(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h), (x, y + h)]
    f.poly(out, "#ffffff", "#9a9aa0", 0.6 * LW, op=0.85, chk=False, dash="0.7 0.45")
    f.add(f'<path d="M{x:.2f},{y:.2f} H{x + w:.2f} L{x + w + dx:.2f},{y + dy:.2f} M{x + w:.2f},{y:.2f} V{y + h:.2f}" stroke="#9a9aa0" '
          f'stroke-width="{0.6 * LW:.3f}" stroke-dasharray="0.7 0.45" fill="none"/>')
    sl = (w - 0.8) / (n + 1.6)
    k = 0
    for i in range(n + 1):
        if i == n // 2:
            f.add(f'<text x="{x + 0.4 + (k + 0.8) * sl:.2f}" y="{y + h / 2 + 0.2:.2f}" font-size="{5 * PT:.3f}" font-weight="700" '
                  f'fill="{c["st"]}" text-anchor="middle">···</text>')
            k += 1.6
            continue
        cub(f, x + 0.4 + k * sl, y + 0.35, sl * 0.42, h - 0.7, d * 0.82, c, op=0.78, sw=0.5 * LW)
        k += 1


def harrow(f, x0, x1, y, col, **k):
    f.arrow(x0, y, x1, y, stroke=col, sw=1.4 * LW, **k)


def inset(f, x, y, w, h, title, accent, right=None):
    f.rect(x, y, w, h, rx=1.2, fill="#ffffff", stroke=accent, sw=0.55 * LW, chk=False)
    f.text(x + 1.6, y + 3.5, title, F_S, 700, c=accent)
    if right:
        f.text(x + w - 1.4, y + 3.5, right, F_N, c=MUTE, a="end")


def funnel(f, pts, col):
    for (a, b), (c, d) in pts:
        f.path(f"M{a:.2f},{b:.2f} L{c:.2f},{d:.2f}", stroke=col, arrow=False, dash="0.7 0.5", chk=False, sw=0.6 * LW)


def ncirc(f, x, y, k, fill):
    f.circle(x, y, 1.45, fill=fill, stroke="none", chk=False)
    f.add(f'<text x="{x:.2f}" y="{y + 0.62:.2f}" font-size="{5.0 * PT:.3f}" font-weight="700" fill="#ffffff" text-anchor="middle">{k}</text>')


# ======================================================================== panel a: architecture
def panel_arch7(f, ox, oy, w=183.0, letter="a"):
    hatch_defs(f)
    plabel(f, ox, oy + 3.4, letter, "S1 architecture")
    # module legend, right-aligned in the title row
    items = [(C2D, False, None, "2D, fine-tuned (lr 10^{−5})"), (C3D, False, None, "new (lr 2·10^{−4})"),
             (CZ, False, "0.7 0.45", "zero-initialised"), (C2D, True, None, "frozen")]
    widths = [5.3 + tw(lab, F_L) + 3.6 for *_, lab in items]
    cx = ox + w - 0.5 - sum(widths) + 3.6
    for (c, hatch, dash, lab), wd in zip(items, widths):
        cub(f, cx, oy + 1.0, 3.4, 2.9, 1.5, c, hatch=hatch, dash=dash)
        f.text(cx + 5.3, oy + 3.4, lab, F_L)
        cx += wd

    cy0, ch = oy + 5.4, 65.4
    xI, wI = ox + 0.5, 57.5
    xII, wII = xI + wI + 1.4, 69.0
    xIII = xII + wII + 1.4
    wIII = ox + w - 0.5 - xIII
    section(f, xI, cy0, wI, ch, BG2D, "I. Per-plane 2D encoder", accent=ACC2D)
    section(f, xII, cy0, wII, ch, BG3D, "II. 3D residual decoder", accent=ACC3D)
    section(f, xIII, cy0, wIII, ch, BGT, "III. Output and training target", accent=ACCT)
    cy = oy + 29.2                                # main-row centre in stages II and III
    lab1, lab2 = cy + 8.8, cy + 11.1              # shape-label rows

    # ---------------- I: input -> patch / pos -> ViT stack -> neck
    ix, iy_, isz, iL = xI + 2.0, oy + 15.2, 11.5, 5.8
    block(f, ix, iy_, isz, isz, iL, "#000", "#000", "#000", C2D["st"], front="../assets3/in_front.png", top="../assets3/in_top.png",
          side="../assets3/in_side.png")
    icx = ix + isz / 2 + iL * DEP[0] / 2
    f.text(icx, oy + 30.4, "input", F_L, 700, a="middle")
    f.text(icx, oy + 32.7, "1×Z×256×256", F_N, c=MUTE, a="middle")
    f.text(icx, oy + 34.9, "Z = 32 planes", F_N, c=MUTE, a="middle")
    f.prov_add("input window 1 x Z x 256 x 256, Z = 32 planes; textures = real E15_Ab2 data on the isotropic grid",
               "net3d.py:268-272; train3d.py:218 patch (32,256,256); model3d.json infer.patch_z 32; make_a3_assets.py (gen_a4.panel_arch4)")
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
    for yy, nm in zip(ys, names):
        if yy is None:
            continue
        cc = C2D if nm != "neck" else dict(front=tint(BASE, 0.40), top=tint(BASE, 0.22), side=tint(BASE, 0.60), st=C2D["st"])
        cub(f, sx, yy, sw_, bh, bd, cc, nm if nm != "neck" else "neck → 256")
    f.text(sx + sw_ / 2, b1 + 2 * pitch + 1.7, "···", F_S, 700, c=C2D["st"], a="middle")
    scx = sx + sw_ / 2 + bd * DEP[0] / 2
    f.text(scx + 1.0, oy + 10.4, "ViT-L · 24 blocks · 304 M", F_L, 700, a="middle")
    f.text(scx + 1.0, oy + 13.9, "global attention · lr 10^{−5}", F_N, c=MUTE, a="middle")
    num(f, sx - 2.4, b1 + 2 * pitch + 1.6, 1)
    f.prov_add("patch embedding 8x8 conv, stride 8 (SAM 16x16 kernel subsampled), weights not trained; absolute position embedding (1,32,32,1024), trained",
               "cellpose vit.py:109-116 (CPSAM); net3d.py:202-205 patch-embed requires_grad False, :228-232 _embed")
    f.prov_add("ViT-L encoder: 24 blocks, width 1024, 16 heads, MLP ratio 4, relative position bias, global attention in every block; applied to all B*Z planes with shared weights; 304.4 M encoder-side parameters, lr 1e-5",
               "segment_anything build_sam.py:27-34, 66-80; cellpose vit.py:118-120; net3d.py:285-286; train3d.py:144, 211-214; train log")
    f.prov_add("neck: 1x1 conv 1024->256, LayerNorm2d, 3x3 conv 256->256, LayerNorm2d", "segment_anything image_encoder.py:88-104")

    # ---------------- II: stack along z -> volume -> 1^3 -> 4 residual blocks
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
    for k in range(1, 5):                         # feature grid (own lines: >= 0.25 pt)
        gx_, gy_ = vx + k * vw / 5, cy - vh / 2 + k * vh / 5
        f.add(f'<path d="M{gx_:.2f},{cy - vh / 2:.2f} V{cy + vh / 2:.2f} M{vx:.2f},{gy_:.2f} H{vx + vw:.2f}" stroke="{C2D["st"]}" '
              f'stroke-width="{0.5 * LW:.3f}" opacity="0.6" fill="none"/>')
    f.text(vx + vw / 2 + 1.0, lab1, "256×Z×32×32", F_N, c=MUTE, a="middle")
    f.prov_add("plane features re-stacked to (B,256,Z,32,32)", "net3d.py:288")
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
    f.prov_add("decoder: 1x1x1 Conv3d 256->256, four ResBlock3D with z-dilation 1,2,4,8 (shape kept 256xZx32x32), 14.3 M params, lr 2e-4; z receptive field after blocks 1-4 = 5, 11, 21, 39 planes (derived)",
               "net3d.py:69-96; model3d.json dec_*; train3d.py:145 lr_dec 2e-4; train log '新增模块 14.3M'")

    # ---------------- III: zero-init 1^3 -> Delta (+) 2D head -> x8 -> output; loss
    zx = xIII + 2.2
    harrow(f, dend + 0.2, zx - 0.25, cy, S3D)
    cub(f, zx, cy - 5.0, 2.6, 10.0, 2.4, CZ, dash="0.7 0.45")
    f.text(zx + 1.3, cy + 0.9, "0", F_N, 700, c=RED, a="middle")
    f.text(zx + 1.3, lab1, "1^{3}", F_N, c=PAL["new_d"], a="middle")
    f.text(zx + 2.2, cy - 8.2, "4·64", F_N, c=MUTE, a="middle")
    num(f, zx + 1.3, lab2 - 0.5, 3)
    qx3 = zx + 2.6 + 2.4 * DEP[0] + 5.0
    f.plus_node(qx3, cy, r=1.6)
    harrow(f, zx + 2.6 + 2.4 * DEP[0] + 0.2, qx3 - 1.85, cy, S3D)
    f.text(qx3 - 2.9, cy - 1.1, "Δ", F_L, 700, c=RED, a="middle")
    hx, hy, hw, hh = xIII + 2.2, oy + 10.9, 26.0, 4.4
    cub(f, hx, hy, hw, hh, 1.8, C2D, "2D head: 1×1 conv → 3·64", size=F_N)
    vcx, vtop = vx + vw / 2 + vL * DEP[0] / 2, cy - vh / 2 + vL * DEP[1] / 2
    f.path(f"M{vcx:.2f},{vtop - 0.1:.2f} V{hy + hh / 2:.2f} H{hx - 0.25:.2f}", stroke=S2D, sw=1.4 * LW)
    f.text((vcx + xII + wII) / 2 - 2.0, hy + hh / 2 - 1.0, "same features, per plane", F_N, c=MUTE, a="middle", italic=True)
    f.path(f"M{qx3:.2f},{hy + hh + 0.2:.2f} V{cy - 1.85:.2f}", stroke=S2D, sw=1.4 * LW)
    f.text(qx3 + 1.0, cy - 12.0, "d_{y}, d_{x}, p", F_N, c=MUTE)
    f.prov_add("2D head = Cellpose-SAM 1x1 Conv2d(256 -> 3*64) per plane; Delta added for (dy, dx, p); dz = Delta_z; x8 pixel shuffle back to 256x256",
               "net3d.py:287-295; cellpose vit.py:122")
    ox_, ow, oL = qx3 + 4.6, 10.0, 6.0
    f.arrow(qx3 + 1.8, cy, ox_ - 0.25, cy, sw=1.4 * LW)
    f.text((qx3 + ox_) / 2 + 0.2, cy + 3.4, "×8", F_N, c=MUTE, a="middle")
    block(f, ox_, cy - 6.0, ow, 12.0, oL, "#ffffff", "#ffffff", tint(BASE, 0.10), INK, top="../assets3/out_top.png", sw=0.8 * LW)
    f.image(ox_ + 0.4, cy - 5.6, ow - 0.8, 11.2, a7("flow_glyphs_pal.png"))
    f.text(ox_ + ow / 2 + 1.5, lab1, "(d_{z}, d_{y}, d_{x}, p)", F_N, 700, a="middle")
    f.text(ox_ + ow / 2 + 1.5, lab2 + 1.0, "4×Z×256×256", F_N, c=MUTE, a="middle")
    num(f, ox_ - 1.9, cy - 4.4, 4)
    f.prov_add("output (B,4,Z,256,256) = [dz, dy, dx, cellprob] after pixel shuffle x8; glyph = S1 seed-0 output on E15_Ab2 nucleus 90, coloured by d_z (whitespace-cropped copy in assets7d)",
               "net3d.py:295,299; arch/render_glyphs.py; assets/flow_glyphs_pal.json")
    lx = ox_ + ow + oL * DEP[0] + 2.0
    lw_ = xIII + wIII - 1.4 - lx
    ly, lh = cy - 7.4, 14.8
    f.rect(lx, ly, lw_, lh, rx=1.0, fill=tint(LOSS, 0.14), stroke=LOSS, sw=0.7 * LW)
    f.text(lx + lw_ / 2, ly + 3.2, "loss", F_L, 700, c=LOSS, a="middle")
    f.text(lx + lw_ / 2, ly + 5.9, ["½ MSE(flows, 5*u*)", "+ BCE(p, fg)"], F_N, 700, c=LOSS, a="middle", ls=1.25)
    f.text(lx + lw_ / 2, ly + 10.9, ["vs 3D target,", "in the volume"], F_N, c=MUTE, a="middle", ls=1.25)
    f.arrow(ox_ + ow + oL * DEP[0] + 0.2, cy, lx - 0.25, cy, stroke=LOSS, sw=1.4 * LW, dash="0.9 0.6")
    f.prov_add("loss: MSE(net flow, 5 x unit target)/2 + BCE(cellprob, nucleus) over voxels inside the volume; AdamW, bf16 autocast, grad clip 1.0",
               "train3d.py:33-43, 215, 275-280")

    # ================= zoom insets
    iy, ih = oy + 43.4, 26.6
    bh_ = 5.0
    # ---- I: transformer block
    ex, ew = xI + 1.2, wI - 2.4
    inset(f, ex, iy, ew, ih, "transformer block (×24)", "#4a5577", "tokens 32×32×1024")
    seq = [("LN", 4.2), ("MHSA", 9.0), ("+", 0), ("LN", 4.2), ("MLP", 9.0), ("+", 0)]
    bx, by_ = ex + 4.2, iy + 9.0
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
        f.path(f"M{a_:.2f},{by_ + bh_ / 2:.2f} V{by_ - 2.2:.2f} H{b_:.2f} V{by_ + bh_ / 2 - 1.4:.2f}", stroke=S2D, sw=0.9 * LW)
    f.path(f"M{ex + 1.0:.2f},{by_ + bh_ / 2:.2f} H{ex + 4.0:.2f}", stroke=S2D, sw=0.9 * LW, arrow=False)
    f.text(xs_[1], by_ + bh_ + 2.5, ["16 heads, global,", "rel. position"], F_N, c=MUTE, a="middle")
    f.text(xs_[4], by_ + bh_ + 2.5, ["1024→4096", "→1024, GELU"], F_N, c=MUTE, a="middle")
    f.text(ex + 1.6, iy + ih - 4.3, "neck: 1×1 conv 1024→256 · LN · 3×3 conv · LN → 32×32×256", F_N)
    f.text(ex + 1.6, iy + ih - 1.7, "patch embed frozen; all other encoder weights fine-tuned", F_N, c=MUTE)
    f.prov_add("transformer block: x + MHSA(LN(x)); x + MLP(LN(x)); MHSA 16 heads with relative position bias, global; MLP 1024->4096->1024 GELU",
               "segment_anything image_encoder.py Block (norm1, attn, norm2, mlp, shortcut) l.151-175; build_sam.py:27-34, 66-80")
    funnel(f, [((sx, ys[4] + bh + 0.3), (ex + 6.0, iy - 0.2)), ((sx + sw_, ys[4] + bh + 0.3), (ex + ew - 6.0, iy - 0.2))], C2D["st"])

    # ---- II: residual block + dilated taps
    rxi, rwi = xII + 1.2, wII - 2.4
    inset(f, rxi, iy, rwi, ih, "residual block (×4)", PAL["new_d"], "256×Z×32×32 in and out")
    seq2 = [("GN", 4.0), ("GELU", 5.8), ("conv 3^{3}", 9.4), ("GN", 4.0), ("GELU", 5.8), ("conv 3^{3}", 9.4)]
    bx, by_ = rxi + 4.2, iy + 9.0
    first = bx
    for i, (nm, bw_) in enumerate(seq2):
        conv = nm.startswith("conv")
        cub(f, bx, by_, bw_, bh_, 1.1, C3D if conv else dict(front="#ffffff", top=tint(RED, 0.05), side=tint(RED, 0.16), st=C3D["st"]),
            nm, bold=conv)
        if i == 2:
            f.text(bx + bw_ / 2, by_ + bh_ + 2.5, "dilation (d,1,1)", F_N, 700, c=PAL["new_d"], a="middle")
        bx += bw_ + 1.1 * DEP[0]
        if i < 5:
            harrow(f, bx + 0.15, bx + 1.55, by_ + bh_ / 2, S3D)
            bx += 1.7
    pxn = bx + 2.6
    f.plus_node(pxn, by_ + bh_ / 2, r=1.2, col=S3D)
    harrow(f, bx + 0.15, pxn - 1.35, by_ + bh_ / 2, S3D)
    f.path(f"M{first - 1.6:.2f},{by_ + bh_ / 2:.2f} V{by_ - 2.2:.2f} H{pxn:.2f} V{by_ + bh_ / 2 - 1.4:.2f}", stroke=S3D, sw=0.9 * LW)
    f.path(f"M{rxi + 1.0:.2f},{by_ + bh_ / 2:.2f} H{first - 0.2:.2f}", stroke=S3D, sw=0.9 * LW, arrow=False)
    nz, pz = 39, 0.185
    top = iy + 18.6
    for j, dd in enumerate((1, 2, 4, 8)):
        cxj = rxi + 3.0 + j * 10.2
        cw, cdp = 4.4, 1.8
        ddx, ddy = cdp * DEP[0], cdp * DEP[1]
        for q in range(nz):
            yy = top + q * pz
            tap = (q - nz // 2) in (-dd, 0, dd)
            col = RED if tap else tint(BASE, 0.22)
            f.add(f'<polygon points="{cxj:.2f},{yy:.2f} {cxj + cw:.2f},{yy:.2f} {cxj + cw + ddx:.2f},{yy + ddy:.2f} {cxj + ddx:.2f},{yy + ddy:.2f}" '
                  f'fill="{col}" stroke="none"/>')
        r = rf[j + 1]
        q0, q1 = nz // 2 - r // 2, nz // 2 + r // 2
        y0b, y1b = top + q0 * pz - 0.4, top + q1 * pz + 0.2
        bxb = cxj + cw + ddx + 0.8
        f.add(f'<path d="M{bxb:.2f},{y0b:.2f} h0.6 V{y1b:.2f} h-0.6" stroke="{PAL["new_d"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
        f.text(bxb + 1.2, (y0b + y1b) / 2 + 0.8, f"{r}", F_L, 700, c=PAL["new_d"])
    capx = rxi + 3.0 + 3 * 10.2 + 4.4 + 1.8 * DEP[0] + 6.4
    f.text(capx, iy + 18.8, ["red: z taps of the", "dilated conv; bracket:", "z receptive field", "(planes) after block"], F_N, c=MUTE)
    f.prov_add("ResBlock3D: x + conv3(GELU(GN(conv3_dil(GELU(GN(x)))))), GroupNorm 16 groups; first conv dilation (d,1,1); taps and receptive fields 5/11/21/39",
               "net3d.py:69-78; model3d.json dec_zdil [1,2,4,8]")
    rb2 = r0 + 2 * rstep + rbw / 2
    funnel(f, [((rb2 - 1.6, lab2 + 0.9), (rxi + 5.0, iy - 0.2)), ((rb2 + 1.6, lab2 + 0.9), (rxi + rwi - 5.0, iy - 0.2))], PAL["new_l"])

    # ---- III: output composition
    zxi, zwi = xIII + 1.2, wIII - 2.4
    inset(f, zxi, iy, zwi, ih, "output composition", PAL["new_d"])
    cA, cB = zxi + 13.0, zxi + zwi - 10.0
    f.text(cA, iy + 7.0, "step 0", F_N, 700, a="middle")
    f.text(cB, iy + 7.0, "trained", F_N, 700, c=PAL["new_d"], a="middle")
    rowsZ = [("d_{z}", "0", "Δ_{z}"), ("d_{y}", "2D", "2D+Δ"), ("d_{x}", "2D", "2D+Δ"), ("p", "2D", "2D+Δ")]
    for k_, (chn, a_, b_) in enumerate(rowsZ):
        yy = iy + 8.6 + k_ * 3.9
        f.text(zxi + 3.0, yy + 2.3, chn, F_N)
        zero = a_ == "0"
        cub(f, cA - 4.6, yy, 9.2, 3.2, 0.9, CZ if zero else C2D, a_, size=F_N, bold=zero, dash="0.7 0.45" if zero else None,
            lc=RED if zero else INK)
        cub(f, cB - 5.4, yy, 10.8, 3.2, 0.9, C3D, b_, size=F_N, bold=False)
    harrow(f, cA + 5.6, cB - 6.0, iy + 8.6 + 3.9 + 1.6, INK)
    f.text(zxi + 1.6, iy + ih - 1.5, "step 0 = 2D control, d_{z} = 0", F_N, c=MUTE)
    f.prov_add("output composition (dz, dy, dx, p) = (Delta_z, 2D head + Delta); Delta = 0 at step 0, so S1 equals the seed-matched 2D model (2D control) with dz = 0",
               "net3d.py:89-90, 294; model3d.json init orx_v5_2d_s0; net3d.py:359-371 self-test")
    funnel(f, [((zx - 0.6, lab2 + 1.6), (zxi + 5.0, iy - 0.2)), ((zx + 4.6, lab2 + 1.6), (zxi + zwi - 5.0, iy - 0.2))], PAL["new_l"])

    # ================= rationale notes + stream legend
    ry = cy0 + ch + 3.3
    notes = ["per-plane encoding keeps the 2D foundation model's representations, which native 3D models do not inherit",
             "axial resolution is several-fold poorer than lateral: dilation widens z context to 39 planes without extra weights",
             "zero-initialised output: the network starts as the 2D model and learns only the volumetric correction",
             "the axial flow comes from 3D context, not from summing flows predicted on separate orthogonal slices"]
    for k_, t_ in enumerate(notes):
        colx = ox + 0.5 + (k_ % 2) * 91.8
        rowy = ry + (k_ // 2) * 3.4
        num(f, colx + 1.6, rowy - 0.65, k_ + 1)
        f.text(colx + 3.9, rowy, t_, F_N)
    f.prov_add("rationale 1-4 wording", "main.tex Introduction l.44-50 (as in gen_a4.panel_arch4); 'without extra weights' = dilation adds no parameters (net3d.py:72)")
    y = ry + 4.6
    h = 4.2
    f.rect(ox + 0.5, y, w - 1.0, h, rx=1.0, fill="none", stroke="#B9B9BD", sw=0.55 * LW, dash="1.2 0.8", chk=False)
    yy = y + h / 2
    sitems = [("arrow", S2D, None, "per-plane 2D stream"), ("arrow", S3D, None, "3D stream"),
              ("arrow", LOSS, "0.9 0.6", "loss against the 3D target"), ("plus", INK, None, "element-wise sum"),
              ("dots", INK, None, "repeated block"), ("funnel", MUTE, None, "zoom")]
    cx = ox + 3.0
    for kind, col, dash, lab in sitems:
        if kind == "arrow":
            f.arrow(cx, yy, cx + 7.0, yy, stroke=col, sw=1.4 * LW, dash=dash, chk=False)
            tx = cx + 8.4
        elif kind == "plus":
            f.plus_node(cx + 1.3, yy, r=1.1)
            tx = cx + 3.4
        elif kind == "dots":
            f.text(cx + 1.8, yy + 0.5, "···", F_L, 700, a="middle")
            tx = cx + 4.4
        else:
            f.path(f"M{cx:.2f},{yy - 1.2:.2f} L{cx + 5.0:.2f},{yy + 1.2:.2f}", stroke="#9a9aa0", arrow=False, dash="0.7 0.5", chk=False, sw=0.6 * LW)
            tx = cx + 6.4
        f.text(tx, yy + 0.65, lab, F_N)
        cx = tx + tw(lab, F_N) + 7.5


# ======================================================================== panel b: how z enters the flow
def colorbar_uz(f, x, y, w, h):
    gid = "cbuz"
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in ((0, PAL["div"][0]), (0.5, PAL["div"][1]), (1, PAL["div"][2])))
    f.defs.append(f'<linearGradient id="{gid}" x1="0" x2="1">{stops}</linearGradient>')
    f.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="url(#{gid})" stroke="{RULE}" stroke-width="{0.5 * LW:.3f}"/>')


def panel_cmp7(f, ox, oy, w, letter="b", bh=36.0, ph=16.0):
    plabel(f, ox, oy + 3.4, letter, "How z information enters the flow (xz section through one nucleus)")
    by = oy + 5.4
    xl, wl = ox + 0.5, 55.0
    xr = xl + wl + 1.4
    wr = ox + w - xr
    section(f, xl, by, wl, bh, BG2D, "same single xy view", accent=ACC2D)
    section(f, xr, by, wr, bh, BGT, "angle to the 3D target, as evaluated", accent=ACCT)
    ar = aspect("cmp_2d_xyonly.png")
    pw = ph * ar
    py = by + 11.2
    gapL = wl - 2 * pw
    xa = xl + gapL / 3
    xb = xa + pw + gapL / 3
    for xx, l1, l2, col in ((xa, "2D model, single xy view", "d_{z} ≡ 0", PAL["ctl_d"]), (xb, "S1, same single view", "d_{z} predicted", PAL["new_d"])):
        f.text(xx + pw / 2, by + 6.9, l1, F_L, 700, c=col, a="middle")
        f.text(xx + pw / 2, by + 9.0, l2, F_N, c=MUTE, a="middle")
    f.image(xa, py, pw, ph, a7("cmp_2d_xyonly.png"))
    f.image(xb, py, pw, ph, a7("cmp_s1_view1.png"))
    f.arrow(xa + pw + 0.4, py + ph / 2, xb - 0.4, py + ph / 2, stroke=PAL["new"], sw=1.2 * LW)
    cbw = 12.0
    cbx = xl + wl / 2 - cbw / 2
    colorbar_uz(f, cbx, py + ph + 1.6, cbw, 1.3)
    f.text(cbx - 0.8, py + ph + 2.8, "u_{z}  −1", F_N, c=MUTE, a="end")
    f.text(cbx + cbw + 0.8, py + ph + 2.8, "+1", F_N, c=MUTE)
    f.prov_add("left pair: 2D control on xy planes only (u_z = 0 by construction) vs S1 single identity view on the same planes, nucleus 62 (whitespace-cropped copies in assets7d)",
               "make_a3_assets.py cmp_2d_xyonly / cmp_s1_view1 (v2v4_crop.npz c2d_out, s1_out; nc-viz viz_run.py); gen_a3.panel_cmp2")
    # right: angle maps
    med = LOG3["angle_nucleus62"]["median_deg"]
    gapR = wr - 2 * pw - 7.0
    xc = xr + gapR / 3 + 0.4
    xd = xc + pw + gapR / 3
    f.text(xc + pw / 2, by + 6.9, "2D control (3 views fused)", F_L, 700, c=PAL["ctl_d"], a="middle")
    f.text(xd + pw / 2, by + 6.9, "S1", F_L, 700, c=PAL["new_d"], a="middle")
    f.image(xc, py, pw, ph, a7("ang_2d.png"))
    f.image(xd, py, pw, ph, a7("ang_s1.png"))
    f.text(xc + pw / 2, py + ph + 2.6, f"median {med['2D']:.0f}°", F_L, c=INK, a="middle")
    f.text(xd + pw / 2, py + ph + 2.6, f"median {med['S1']:.0f}°", F_L, c=INK, a="middle")
    gid = "angbar7"
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in ((0, "#ffffff"), (0.33, PM["cellos"]), (0.67, PM["S1"]), (1, PM["S1_dark"])))
    f.defs.append(f'<linearGradient id="{gid}" x1="0" x2="0" y1="1" y2="0">{stops}</linearGradient>')
    bx = xd + pw + 1.6
    f.add(f'<rect x="{bx:.2f}" y="{py:.2f}" width="1.3" height="{ph:.2f}" fill="url(#{gid})" stroke="{PAL["box"]}" stroke-width="{0.5 * LW:.3f}"/>')
    f.text(bx + 1.9, py + 1.7, "90°", F_N, c=MUTE)
    f.text(bx + 1.9, py + ph, "0°", F_N, c=MUTE)
    f.text(xr + wr / 2, by + bh - 3.6, "all 12 datasets, 5 seeds:", F_N, c=MUTE, a="middle")
    f.text(xr + wr / 2, by + bh - 1.5, "S1 − 2D median angle −3.18° (−3.41 to −2.89)", F_N, 700, c=PAL["new_d"], a="middle")
    f.prov_add(f"right pair: per-voxel 3D angle to the GT-derived target (FLOWERR definition, region GT fg & |t|>0.5) for the 2D control as evaluated (3 views fused) and S1 as evaluated, nucleus 62: medians {med['2D']:.1f} / {med['S1']:.1f} deg (n = {LOG3['angle_nucleus62']['n_vox']} voxels)",
               "make_a3_assets.py angle section; definition flowerr_angles.py; dP from thr_eval npz runs 81116913 / 07992b55")
    f.prov_add("macro over 12 datasets, five seeds: S1 - 2D difference in median angle -3.18 deg (-3.41 to -2.89); post hoc, S1 trained on this target",
               "NUMBERS.md:631 (primary.set12_region_a); main.tex Fig. 3 caption f; source_data/inputs/flowerr_results.json")


# ======================================================================== panel c: 3D training target
def panel_target7(f, ox, oy, w, letter="c", bh=36.0, th=19.5):
    plabel(f, ox, oy + 3.4, letter, "3D training target")
    by = oy + 5.4
    section(f, ox, by, w, bh, BGT, "GT nuclei, mouse embryo", accent=ACCT)
    ws = [th * aspect("diff_source.png"), th * aspect("diff_heat.png"), th * aspect("target_F1d.png")]
    gap = (w - 2.0 - sum(ws)) / 2
    ty = by + 6.4
    names = [("diff_source.png", "heat source at", "the label centre"),
             ("diff_heat.png", "diffuse inside the", f"nucleus ({DF['niter']} steps)"),
             ("target_F1d.png", "target flow u, 4 nuclei", "(colour u_{z})")]
    x = ox + 1.0
    xs = []
    for i, (nm, l1, l2) in enumerate(names):
        f.rect(x - 0.4, ty - 0.4, ws[i] + 0.8, th + 0.8, rx=0.8, fill="#ffffff", stroke=RULE, sw=0.5 * LW, chk=False)
        f.image(x, ty, ws[i], th, a7(nm))
        f.text(x + ws[i] / 2, ty + th + 2.6, l1, F_N, c=MUTE, a="middle")
        f.text(x + ws[i] / 2, ty + th + 4.8, l2, F_N, c=MUTE, a="middle")
        xs.append(x)
        if i < 2:
            f.arrow(x + ws[i] + 0.6, ty + th / 2, x + ws[i] + gap - 0.6, ty + th / 2, sw=1.0 * LW)
        x += ws[i] + gap
    # the grey arrow inside the tile-3 render is the +z direction, 3 um long (F1-d.provenance.json "style")
    f.text(xs[2] + 0.2, ty + th + 8.4, "grey arrow: +z, 3 µm", F_N, c=MUTE)
    f.prov_add(f"3D Cellpose target: heat source at the label's most central voxel, diffusion restricted to the nucleus for niter = 6 x sum of bounding-box extents ({DF['niter']} here), unit gradient = flow; GT nucleus 62 of E15_Ab2 resampled to the training grid (whitespace-cropped copies in assets7d)",
               "data3d.py:28-81 flows3d_gpu (centre l.51-57, niter l.58-62, update l.71-74, gradient l.75-77); diffusion_data.py, render_diffusion.py")
    f.prov_add("tile 3: nc-viz F1-d_target_light, GT-derived 3D target on E15_Ab2 GT 33, 51, 68, 50 (rule F1_group), one sink per nucleus; illustrative, re-interpolated along z for display; grey z arrow = 3 um along +z",
               "paper/figures/assets_viz/F1-d.provenance.json ('style': grey z arrow (3 um))")


# ======================================================================== panel d: inference
def mini_axes(f, x, y, w, h):
    f.add(f'<path d="M{x:.2f},{y:.2f} V{y + h:.2f} H{x + w:.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}" fill="none"/>')


def gridline(f, x0, x1, y):
    f.add(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="{GRID}" stroke-width="{0.5 * LW:.3f}"/>')


def panel_infer7(f, ox, oy, w, letter="d", bottom=170.0):
    plabel(f, ox, oy + 3.4, letter, "Inference on one test volume (mouse embryo, S1 seed 0)")
    # header-strip key, right-aligned in the title row
    keys = [(tint(CTL, 0.30), PAL["ctl_d"], "2D control"), (tint(RED, 0.22), PAL["new_d"], "S1 network"), (tint(BASE, 0.22), "#5d6680", "processing")]
    kx = ox + w - 0.5 - sum(4.2 + tw(l, F_N) + 3.0 for *_, l in keys) + 3.0
    for fill, st, lab in keys:
        f.rect(kx, oy + 1.2, 3.2, 2.6, rx=0.5, fill=fill, stroke="none", chk=False)
        f.text(kx + 4.2, oy + 3.4, lab, F_N, c=MUTE)
        kx += 4.2 + tw(lab, F_N) + 3.0
    n = 7
    gap = 2.6
    section(f, ox, oy + 5.0, w, bottom - (oy + 5.0), BLOCK)
    bw = (w - 2.0 - (n - 1) * gap) / n
    by, bh = oy + 6.0, bottom - 1.0 - (oy + 6.0)
    dh = max(0.0, 34.0 - bh)                                       # compaction relative to A7-d (bh = 34)
    xs = [ox + 1.0 + i * (bw + gap) for i in range(n)]
    heads = ["diameter", "isotropic", "z windows", "three views", "follow flow", "rule T", "3D nuclei"]
    kind = ["c", "p", "s", "s", "p", "p", "s"]
    col = dict(c=(tint(CTL, 0.30), PAL["ctl_d"]), s=(tint(RED, 0.22), PAL["new_d"]), p=(tint(BASE, 0.22), "#5d6680"))
    hs = 4.6
    for i, x in enumerate(xs):
        fill, dark = col[kind[i]]
        f.rect(x, by, bw, bh, rx=1.2, fill="#ffffff", stroke=RULE, sw=0.7 * LW, chk=False)
        f.add(f'<path d="M{x:.2f},{by + hs:.2f} V{by + 1.2:.2f} Q{x:.2f},{by:.2f} {x + 1.2:.2f},{by:.2f} H{x + bw - 1.2:.2f} '
              f'Q{x + bw:.2f},{by:.2f} {x + bw:.2f},{by + 1.2:.2f} V{by + hs:.2f} Z" fill="{fill}" stroke="none"/>')
        ncirc(f, x + 2.4, by + hs / 2, i + 1, dark)
        f.text(x + 4.6, by + hs / 2 + 0.75, heads[i], F_S, 700, c=INK)
        if i < n - 1:
            f.arrow(x + bw + 0.3, by + hs / 2, x + bw + gap - 0.3, by + hs / 2, sw=0.9 * LW)
    c0 = by + hs                                                    # content top
    # 1 diameter
    x = xs[0]
    rows = [r for r in RD["diam"]["E15_Ab1"] if r["thr"] == 0.0]
    sc_all = [6, 10, 17, 30, 52, 90, 150]
    gx0, gy0, gw, gh = x + 5.6, c0 + 2.6, bw - 7.0, 15.5 - dh
    lx = lambda s_: gx0 + gw * (math.log(s_) - math.log(6)) / (math.log(150) - math.log(6))
    ly = lambda v: gy0 + gh - gh * (v + 0.6) / 1.4
    for v_, lab_ in ((-0.5, "−0.5"), (0.0, "0"), (0.5, "0.5")):
        gridline(f, gx0, gx0 + gw, ly(v_))
        f.add(f'<line x1="{gx0 - 0.6:.2f}" y1="{ly(v_):.2f}" x2="{gx0:.2f}" y2="{ly(v_):.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}"/>')
        f.text(gx0 - 0.9, ly(v_) + 0.7, lab_, F_N, c=MUTE, a="end")
    mini_axes(f, gx0, gy0, gw, gh)
    for key, cc in (("stab", CTL), ("score", INK)):
        pts = [(lx(r["scale"]), ly(r["stab"] if key == "stab" else r["stab"] - r["cons"])) for r in rows]
        f.add('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + f'" stroke="{cc}" stroke-width="{0.9 * LW:.3f}" fill="none"/>')
        for a, b in pts:
            f.circle(a, b, 0.45, fill=cc, chk=False)
    sel = RD["diam_sel"]["E15_Ab1"][0]
    rs = [r for r in rows if r["scale"] == sel["scale"]][0]
    f.circle(lx(sel["scale"]), ly(rs["stab"] - rs["cons"]), 1.0, fill="none", stroke=RED, sw=1.0 * LW, chk=False)
    for s_ in sc_all:
        f.add(f'<line x1="{lx(s_):.2f}" y1="{gy0 + gh:.2f}" x2="{lx(s_):.2f}" y2="{gy0 + gh + 0.6:.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}"/>')
    for s_ in (6, 30, 150):
        f.text(lx(s_), gy0 + gh + 2.6, str(s_), F_N, c=MUTE, a="middle")
    f.text(x + bw / 2 + 1.5, gy0 + gh + 5.0, "scale (px)", F_N, c=MUTE, a="middle")
    f.text(gx0 + 1.0, gy0 + gh - 1.2, "E15_Ab1", F_N, c=MUTE)
    f.text(x + 1.3, by + bh - 3.9, "blue stability; black", F_N, c=MUTE)
    f.text(x + 1.3, by + bh - 1.5, f"minus |log d̂/d|; d̂ = {sel['d_used']:.0f}", F_N, c=MUTE)
    f.prov_add("diameter scan: 2D control seed 0, cp 0: stability (blue) and stability - |log(d_est/scale)| (black) for mouse_embryo E15_Ab1 (E15_Ab2 not in the 4-per-dataset full-pipeline sample); scales 6 and 10 px skipped (rescaled volume > 8e8 voxels); selected scale 30, threshold -2, d_used 33.8 px (GT 32.5); candidate scales 6,10,17,30,52,90,150",
               "server results/orx_4a362b0f-..._test rows_all.json, rows.json (eval_val.py joint r2a2, l.160-199); arch_views.py -> assets3/ruleT_diam.json; gen_a4.panel_infer4")
    # 2 isotropic
    x = xs[1]
    tt = bw - 2.6
    ih2 = 10.8 - dh / 2
    f.image(x + 1.3, c0 + 1.0, tt, ih2, "xz_native.png", fit="xMidYMid slice")
    f.text(x + bw / 2, c0 + ih2 + 3.4, "z/xy = 4.65 → linear z", F_N, c=MUTE, a="middle")
    f.image(x + 1.3, c0 + ih2 + 4.8, tt, ih2, "xz_iso.png", fit="xMidYMid slice")
    f.text(x + bw / 2, c0 + 2 * ih2 + 7.4, "×30/d, isotropic", F_N, c=MUTE, a="middle")
    f.prov_add("isotropic resampling of E15_Ab2 (z/xy 4.65): native xz plane vs linear z interpolation; rescale by 30/d", "eval_gt.py:217-222; infer3d.py:211-217; make_assets.py")
    # 3 windows
    x = xs[2]
    st, Zp = LOGA["windows"]["starts"], LOGA["windows"]["Z_network"]
    a0, a1 = x + 1.4, x + bw - 1.4
    base, amp = c0 + 15.0 - dh, 11.0 - dh
    for g in (0.5, 1.0):
        gridline(f, a0, a1, base - amp * g)
    s_ = (a1 - a0) / Zp
    for i, s0 in enumerate(st):
        pts = [(a0 + (s0 + k + 0.5) * s_, base - amp * math.sin(math.pi * (k + 0.5) / 32) ** 2) for k in range(32)]
        f.add('<path d="M' + " L".join(f"{p:.2f},{q:.2f}" for p, q in pts) + f'" stroke="{RED if i % 2 == 0 else PAL["new_l"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
    f.add(f'<line x1="{a0:.2f}" y1="{base:.2f}" x2="{a1:.2f}" y2="{base:.2f}" stroke="{INK}" stroke-width="{0.6 * LW:.3f}"/>')
    f.text(x + bw / 2, base + 2.8, f"z′ = {Zp} planes", F_N, c=MUTE, a="middle")
    f.text(x + 1.3, by + bh - 6.3, f"{len(st)} windows of 32,", F_N, c=MUTE)
    f.text(x + 1.3, by + bh - 3.9, "stride 23–24,", F_N, c=MUTE)
    f.text(x + 1.3, by + bh - 1.5, "sin² blending", F_N, c=MUTE)
    f.prov_add("z windows of 32 planes, 25% overlap (stride 23-24, Z' = 426, 18 windows for this volume), sin^2 taper, weighted average",
               "infer3d.py:56-60 _starts, 129-132 _taper, 161-186 run_volume; model3d.json infer; assets_log.json windows")
    # 4 three views
    x = xs[3]
    ar = aspect("view_z.png")
    vh_ = min((bw - 3.4) / 2 / ar, (bw - 3.4) / 2 / ar - dh / 2)
    vw_ = vh_ * ar
    pos = [("z", x + 1.2, c0 + 1.4), ("y", x + 2.2 + vw_, c0 + 1.4), ("x", x + 1.2, c0 + vh_ + 5.0), ("mean", x + 2.2 + vw_, c0 + vh_ + 5.0)]
    for k_, px_, py_ in pos:
        f.image(px_, py_, vw_, vh_, a7(f"view_{k_}.png"))
        lab = f"{k_} as z" if k_ != "mean" else "mean"
        f.text(px_ + vw_ / 2, py_ + vh_ + 2.4, lab, F_N, 700 if k_ == "mean" else 400, c=MUTE if k_ != "mean" else PAL["new_d"], a="middle")
    f.text(x + 1.3, by + bh - 1.5, "u_{z}, nucleus 62", F_N, c=MUTE)
    f.prov_add("three views: S1 seed-0 raw outputs with each volume axis as network z (one 32-plane window per view centred on GT nucleus 62), u_z on the xz section, and their mean (whitespace-cropped copies in assets7d)",
               "arch_views.py (server, CPU) -> assets3/views_E15_Ab2_n62.npz; infer3d.py:63-96 VIEW_PERMS, run_views")
    # 5 follow flow
    x = xs[4]
    ar5 = aspect("flowfollow_F1d.png")
    ih5 = min((bw - 1.6) / ar5, (bw - 1.6) / ar5 - dh)
    f.image(x + bw / 2 - ih5 * ar5 / 2, c0 + 1.0, ih5 * ar5, ih5, a7("flowfollow_F1d.png"))
    f.text(x + 1.3, by + bh - 3.9, "one sink per nucleus", F_N, c=MUTE)
    f.text(x + 1.3, by + bh - 1.5, "grey arrow: +z, 3 µm", F_N, c=MUTE)
    f.prov_add("follow-flow render: nc-viz F1-d_follow_light (S1 seed 0, E15_Ab2 GT 33, 51, 68, 50, all matched, one sink each; strands coloured by u_z; grey arrow 3 um along +z); re-interpolated along z for display",
               "paper/figures/assets_viz/F1-d.provenance.json (rule F1_group, selection_rules.json); infer3d.py:258-260 flow following")
    # 6 rule T
    x = xs[5]
    tr = RD["ruleT"]
    ts = [-6, -4, -3, -2, -1, 0]
    gx0, gy0, gw, gh = x + 5.6, c0 + 2.6, bw - 7.0, 15.5 - dh
    lx2 = lambda t: gx0 + gw * (t + 6) / 6.0
    ly2 = lambda v: gy0 + gh - gh * (v - 0.5) / 1.2
    for v_ in (0.7, 1.0, 1.5):
        gridline(f, gx0, gx0 + gw, ly2(v_))
        f.add(f'<line x1="{gx0 - 0.6:.2f}" y1="{ly2(v_):.2f}" x2="{gx0:.2f}" y2="{ly2(v_):.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}"/>')
        f.text(gx0 - 0.9, ly2(v_) + 0.7, f"{v_:.1f}", F_N, c=LOSS if v_ == 0.7 else MUTE, a="end")
    mini_axes(f, gx0, gy0, gw, gh)
    for yv in (0.7, 0.85):
        f.add(f'<line x1="{gx0:.2f}" y1="{ly2(yv):.2f}" x2="{gx0 + gw:.2f}" y2="{ly2(yv):.2f}" stroke="{LOSS}" stroke-width="{0.5 * LW:.3f}" stroke-dasharray="0.6 0.4"/>')
    pts = [(lx2(t), ly2(tr["m"][f"{float(t)}"])) for t in ts]
    f.add('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + f'" stroke="{INK}" stroke-width="{0.9 * LW:.3f}" fill="none"/>')
    for a, b in pts:
        f.circle(a, b, 0.45, fill=INK, chk=False)
    mc = tr["m"][f"{tr['chosen']}"]
    f.circle(lx2(tr["chosen"]), ly2(mc), 1.0, fill="none", stroke=RED, sw=1.0 * LW, chk=False)
    f.text(lx2(tr["chosen"]) - 1.2, ly2(mc) + 2.7, f"{tr['m']['-2.0']:.2f}", F_N, 700, c=PAL["new_d"], a="end")
    for t in (-6, -2, 0):
        f.text(lx2(t), gy0 + gh + 2.6, f"{t}".replace("-", "−"), F_N, c=MUTE, a="middle")
    f.text(x + bw / 2 + 1.5, gy0 + gh + 5.0, "threshold t", F_N, c=MUTE, a="middle")
    f.text(gx0 + gw, gy0 + 1.6, "m(t)", F_N, c=MUTE, a="end")
    f.text(x + 1.3, by + bh - 3.9, f"m(−2) = {tr['m']['-2.0']:.2f} ≥ 0.7", F_N, c=MUTE)
    f.text(x + 1.3, by + bh - 1.5, "→ t = −2 kept", F_N, 700, c=PAL["new_d"])
    f.prov_add(f"rule T on E15_Ab2 (S1 seed 0): m(t) = N_fg(t)/(N_obj(t) V) for t in 0..-6, m(-2) = {tr['m']['-2.0']:.2f} >= 0.7 so t = -2; dashed lines 0.7 and 0.85",
               "sizeThr_dir.py:16-28 recomputed by arch_views.py from run 07992b55 thr_eval rows.json and masks/mouse_embryo__E15_Ab2.npz cp")
    # 7 3D nuclei
    x = xs[6]
    mh = by + bh - 4.6 - (c0 + 0.8)
    mw = mh * 410 / 600
    f.image(x + (bw - mw) / 2, c0 + 0.8, mw, mh, "masks_S1.png")
    f.text(x + 1.3, by + bh - 1.5, f"{tr['n_pred']['-2.0']} objects; F1 {tr['f1_by_thr']['-2.0']:.2f}", F_N, c=MUTE)
    f.prov_add("final S1 masks of E15_Ab2 (render nc-figures), objects before size filter and F1@0.5 after it at t = -2", "E15_Ab2_iso.provenance.json; thr_eval rows.json")


# ======================================================================== layout
def layout_a7d():
    f = Fig(183.0, 170.0)
    panel_arch7(f, 0.0, 0.0, letter="a")
    panel_cmp7(f, 0.0, 86.4, 119.6, letter="b")
    panel_target7(f, 121.0, 86.4, 61.5, letter="c")
    panel_infer7(f, 0.0, 129.0, 183.0, letter="d")
    return f


def main():
    cid = "A7-d"
    crops = ensure_crops()
    f = layout_a7d()
    res = export(f, cid)
    print(cid, "issues:", len(res.get("issues", [])), "min font", res.get("min_font_pt"))
    for s_ in res.get("issues", [])[:80]:
        print("   ", s_)
    json.dump(dict(figure=cid, code_snapshot=_nsp_os.path.join(_NSP_REPO, 'src'),
                   model="orx_t3d_r5s1_s0/model3d.json", palette="paper/figures/palette.json",
                   colour_update="11-11 user: warm grey blocks #EDECE9, white page and cards (Supplementary_Data_3D_viewer.html look); 2D blue / S1 red semantics kept as tab accent bars, module fills and step header strips",
                   generator="paper/figures/arch/gen_a7d.py (panel_arch7, panel_cmp7, panel_target7, panel_infer7, layout_a7d)",
                   content_sources="panel a = gen_a4.panel_arch4 / gen_a6.panel_arch6; b = gen_a3.panel_cmp2; c = gen_a4.panel_target3d (tile3 target_F1d); d = gen_a4.panel_infer4",
                   type_hierarchy_pt=dict(panel_letter=8.0, panel_title=F_T, subtitle=F_S, label=F_L, note=F_N),
                   cropped_assets=crops,
                   references=[dict(figure=a, technique=b) for a, b in REFS7 + REFS4], items=f.prov),
              open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)
    gal = os.path.join(HERE, "..", "gallery", "assets")
    for ext in ("pdf", "png", "provenance.json"):
        shutil.copy(os.path.join(HERE, f"{cid}.{ext}"), os.path.join(gal, f"{cid}.{ext}"))


if __name__ == "__main__":
    main()
