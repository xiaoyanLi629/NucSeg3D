"""Fig. 1 v7 (A7-b): all four panels redrawn in one visual system following the user's reference figure
(pastel tinted cards with dashed grey borders and grey corner tabs, 3D cuboid modules, slab tensors, colour-coded
stream arrows, numbered step markers). Hand-coded SVG -> vector PDF + 600-dpi PNG via gen_fig.export.
Assets: existing renders/glyphs; assets7/ holds tight crops of some of them (make_a7_assets.py, margins only).
Usage: python gen_a7.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fig import FS, LW, PAL, Fig, export, tint, M as PM  # noqa: E402
from gen_a3 import DEP, LOG3, tex, block as block3  # noqa: E402
from gen_a4 import REFS as REFS4, RD, DF, LOGA, hatch_defs, num  # noqa: E402
from gen_a6 import REFS6, REF_IMG, cub as cub3, C2D, C3D, S2D, S3D, BASE, RED, LOSS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK = PAL["fg"]
PT = 0.3528
# ---------------------------------------------------------------- one typographic hierarchy
F_CARD, F_LAB, F_NOTE = 6.0, 5.5, 5.0
MUTE = PAL["mute"]
# ---------------------------------------------------------------- one container language
FAM = dict(  # stage colour families: body tint, header-strip tint, accent
    I=("#EEF3FA", "#DCE5F2", S2D),
    II=("#FFF5EC", "#FBE3CF", PAL["new_d"]),
    III=("#F0F7EE", "#DCEDD7", LOSS),
)
CBORD, TAB = "#A9A9AD", "#DCDCDE"
GREY_ARROW = "#5B6070"
AW = 1.4 * LW                      # stream arrow weight (0.7 pt)
REFS7 = [
    ("same reference figure as A6 (" + REF_IMG + ")",
     "A7 extends the reference language to panels b-d: every panel sits in pastel tinted cards with dashed grey borders and grey corner tabs "
     "(or a slim tinted header strip with a numbered circle for the 7 inference steps), stage colour families shared with panel a "
     "(blue = per-plane 2D / preprocessing, warm = 3D network, green = output / target / postprocessing), one arrow style (0.7 pt, same head) "
     "throughout, numbered step markers, white image wells and white plot areas with light gridlines; dashed funnel lines from each "
     "module to its zoom inset (MA-SAM style)"),
]


# ---------------------------------------------------------------- variant style ("b" reference cuboids, "c" flat Nature/Cell)
STYLE = {"v": "b"}
WARM = dict(pane="#EDECE9", tab="#DDDCD7", strip="#E3E2DD", line="#E2E2DE", bord="#B5B4AE")   # user's 3D-viewer scheme


def flat(f, x, y, w, h, c, label=None, size=F_NOTE, bold=True, hatch=False, dash=None, lc=None, op=1.0, shadow=True):
    rx = min(0.9, h / 3, w / 3)
    if shadow and op >= 1.0:
        f.rect(x + 0.18, y + 0.42, w, h, rx=rx, fill="#2a2a30", op=0.07, chk=False)
        f.rect(x + 0.08, y + 0.2, w, h, rx=rx, fill="#2a2a30", op=0.07, chk=False)
    f.rect(x, y, w, h, rx=rx, fill="url(#hatch)" if hatch else c["front"], stroke=c["st"], sw=0.6 * LW, dash=dash, op=op, chk=False)
    if label:
        lines = label if isinstance(label, (list, tuple)) else [label]
        fs = size * PT
        y0 = y + h / 2 - (len(lines) - 1) * fs * 1.18 / 2 + fs * 0.36
        f.text(x + w / 2, y0, lines, size, 700 if bold else 400, c=lc or INK, a="middle")


def cub(f, x, y, w, h, d, c, label=None, size=FS["small"], bold=True, hatch=False, dash=None, op=1.0, lc=None, sw=None):
    if STYLE["v"] == "b":
        return cub3(f, x, y, w, h, d, c, label, size=size, bold=bold, hatch=hatch, dash=dash, op=op, lc=lc, sw=sw)
    flat(f, x, y, w + d * DEP[0], h, c, label, size=size, bold=bold, hatch=hatch, dash=dash, lc=lc, op=op)


def block(f, x, y, w, h, L, ff, ft, fs, stroke, front=None, top=None, side=None, sw=None, grid=0, op=1.0, flat_w=None, planes=0):
    if STYLE["v"] == "b":
        return block3(f, x, y, w, h, L, ff, ft, fs, stroke, front=front, top=top, side=side, sw=sw, grid=grid, op=op)
    W = flat_w or w + L * DEP[0]
    if planes:                                     # flat stack of planes behind the front tile
        for k in range(planes, 0, -1):
            off = k * (W - w) / planes
            f.rect(x + off, y - k * 0.9, w, h, rx=0.6, fill="#ffffff" if ff == "#000" else ff, stroke=stroke, sw=0.5 * LW, op=0.55, chk=False)
        W = w
    f.rect(x + 0.18, y + 0.42, W, h, rx=0.6, fill="#2a2a30", op=0.07, chk=False)
    f.rect(x, y, W, h, rx=0.6, fill=ff if ff != "#000" else "#111", stroke=stroke, sw=0.6 * LW, chk=False)
    if front:
        tex(f, front, W, 0, 0, h, x, y)
        f.rect(x, y, W, h, rx=0.6, fill="none", stroke=stroke, sw=0.6 * LW, chk=False)


def T(f, x, y, s, size=F_NOTE, w=400, c=None, a="start", **k):
    return f.text(x, y, s, size, w, c=c or INK, a=a, **k)


def card(f, x, y, w, h, fam, tab=None, sub=None):
    acc = FAM[fam][2]
    if STYLE["v"] == "c":
        f.rect(x, y, w, h, rx=1.6, fill="#ffffff", stroke=WARM["line"], sw=0.6 * LW)
        f.add(f'<path d="M{x + 0.2:.2f},{y + 4.6:.2f} V{y + 1.6:.2f} Q{x + 0.2:.2f},{y + 0.2:.2f} {x + 1.6:.2f},{y + 0.2:.2f} '
              f'H{x + w - 1.6:.2f} Q{x + w - 0.2:.2f},{y + 0.2:.2f} {x + w - 0.2:.2f},{y + 1.6:.2f} V{y + 4.6:.2f} Z" fill="{tint(acc, 0.12)}" stroke="none"/>')
        f.add(f'<rect x="{x + 0.2:.2f}" y="{y + 4.45:.2f}" width="{w - 0.4:.2f}" height="0.3" fill="{acc}" opacity="0.85"/>')
        tw = len(tab or "") * 1.13 + 3.2
        if tab:
            T(f, x + 1.8, y + 3.35, tab, F_CARD, 700, c=acc if acc != S2D else "#3d4a6e")
            if sub:
                T(f, x + tw + 1.4, y + 3.35, sub, F_NOTE, c=MUTE, italic=True)
        return
    f.rect(x, y, w, h, rx=1.6, fill=WARM["pane"], stroke=WARM["bord"], sw=0.75 * LW, dash="1.4 0.8")
    if tab:
        tw = len(tab) * 1.13 + 3.2
        f.add(f'<path d="M{x + 0.25:.2f},{y + 4.6:.2f} V{y + 1.6:.2f} Q{x + 0.25:.2f},{y + 0.25:.2f} {x + 1.6:.2f},{y + 0.25:.2f} '
              f'H{x + tw:.2f} V{y + 4.6:.2f} Z" fill="{WARM["tab"]}" stroke="none"/>')
        f.add(f'<rect x="{x + 0.25:.2f}" y="{y + 1.0:.2f}" width="0.8" height="2.9" fill="{acc}"/>')
        T(f, x + 1.8, y + 3.35, tab, F_CARD, 700, c="#3a3a3e")
        if sub:
            T(f, x + tw + 1.4, y + 3.35, sub, F_NOTE, c=MUTE, italic=True)


def step_card(f, x, y, w, h, fam, k, title):
    """Card with a slim tinted header strip and a numbered circle (panel d)."""
    body, strip, acc = FAM[fam]
    if STYLE["v"] == "c":
        body, strip = "#ffffff", tint(acc, 0.12)
        f.rect(x, y, w, h, rx=1.6, fill=body, stroke=WARM["line"], sw=0.6 * LW)
    else:
        body, strip = WARM["pane"], WARM["strip"]
        f.rect(x, y, w, h, rx=1.6, fill=body, stroke=WARM["bord"], sw=0.75 * LW, dash="1.4 0.8")
    f.add(f'<path d="M{x + 0.25:.2f},{y + 4.8:.2f} V{y + 1.6:.2f} Q{x + 0.25:.2f},{y + 0.25:.2f} {x + 1.6:.2f},{y + 0.25:.2f} '
          f'H{x + w - 1.6:.2f} Q{x + w - 0.25:.2f},{y + 0.25:.2f} {x + w - 0.25:.2f},{y + 1.6:.2f} V{y + 4.8:.2f} Z" fill="{strip}" stroke="none"/>')
    if STYLE["v"] == "c":
        f.add(f'<rect x="{x + 0.2:.2f}" y="{y + 4.65:.2f}" width="{w - 0.4:.2f}" height="0.3" fill="{acc}" opacity="0.85"/>')
    stepnum(f, x + 2.6, y + 2.55, k, acc)
    T(f, x + 4.9, y + 3.5, title, F_CARD, 700, c="#2f2f33")


def stepnum(f, cx, cy, k, col):
    f.circle(cx, cy, 1.55, fill=col, stroke="#ffffff", sw=0.5 * LW, chk=False)
    f.add(f'<text x="{cx:.2f}" y="{cy + 0.64:.2f}" font-size="{5.0 * PT:.3f}" font-weight="700" fill="#ffffff" text-anchor="middle">{k}</text>')


def well(f, x, y, w, h):
    st = WARM["line"] if STYLE["v"] == "c" else "#D3D2CC"
    f.rect(x, y, w, h, rx=0.8, fill="#ffffff" if STYLE["v"] == "b" else "#F5F5F3", stroke=st, sw=0.5 * LW, chk=False)


def arr(f, x0, y0, x1, y1, col=GREY_ARROW, dash=None, chk=True):
    if isinstance(y1, str):            # horizontal shorthand: arr(f, x0, y, x1, colour)
        col, y1 = y1, y0
    f.arrow(x0, y0, x1, y1, stroke=col, sw=AW, dash=dash, chk=chk)


def funnel(f, p0, p1, q0, q1, col):
    for (a, b), (c, d) in ((p0, q0), (p1, q1)):
        f.path(f"M{a:.2f},{b:.2f} L{c:.2f},{d:.2f}", stroke=col, arrow=False, dash="0.7 0.5", chk=False, sw=0.6 * LW)


def slab7(f, x, y, w, h, d, n, c):
    dx, dy = d * DEP[0], d * DEP[1]
    if STYLE["v"] == "c":                        # flat: a fanned stack of planes with an ellipsis
        pw_, ph_ = w * 0.62, h
        offs = [0, 1, 2, None, 3, 4]
        step = (w + dx - pw_) / 4
        for o in offs:
            if o is None:
                continue
            f.rect(x + o * step, y - o * 0.55, pw_, ph_, rx=0.5, fill=c["front"], stroke=c["st"], sw=0.5 * LW, op=0.9, chk=False)
        return
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
        cub(f, x + 0.4 + k * sl, y + 0.35, sl * 0.42, h - 0.7, d * 0.82, c, op=0.8, sw=0.5 * LW)
        k += 1


def inset(f, x, y, w, h, title, col, right=None):
    f.rect(x, y, w, h, rx=1.3, fill="#ffffff", stroke=col, sw=0.6 * LW)
    T(f, x + 1.6, y + 3.5, title, F_CARD, 700, c=col)
    if right:
        T(f, x + w - 1.4, y + 3.5, right, F_NOTE, c=MUTE, a="end")


def legend7(f, x, y):
    items = [(C2D, False, None, "2D, fine-tuned (lr 10^{−5})"), (C3D, False, None, "new (lr 2·10^{−4})"),
             (dict(front="#ffffff", top="#ffffff", side="#ffffff", st=RED), False, "0.7 0.45", "zero-initialised"),
             (C2D, True, None, "frozen")]
    cx = x
    for c, hatch, dash, lab in items:
        cub(f, cx, y - 2.5, 3.4, 2.9, 1.6, c, hatch=hatch, dash=dash)
        T(f, cx + 5.4, y, lab, F_LAB)
        cx += 5.4 + len(lab.replace("^{", "").replace("}", "")) * 1.0 + 4.6


def stream_legend(f, x, y, w):
    h = 4.6
    if STYLE["v"] == "c":
        f.rect(x, y, w, h, rx=1.0, fill="none", stroke=WARM["line"], sw=0.6 * LW)
    else:
        f.rect(x, y, w, h, rx=1.0, fill="none", stroke=WARM["bord"], sw=0.6 * LW, dash="1.4 0.8")
    yy = y + h / 2
    items = [("arrow", S2D, None, "per-plane 2D stream"), ("arrow", S3D, None, "3D stream"),
             ("arrow", LOSS, "0.9 0.6", "loss against the 3D target"), ("plus", INK, None, "element-wise sum"),
             ("dots", INK, None, "repeated block")]
    cx = x + 3.0
    for kind, col, dash, lab in items:
        if kind == "arrow":
            f.arrow(cx, yy, cx + 7.0, yy, stroke=col, sw=AW, dash=dash, chk=False)
            tx = cx + 8.4
        elif kind == "plus":
            f.plus_node(cx + 1.4, yy, r=1.2)
            tx = cx + 3.6
        else:
            T(f, cx + 1.8, yy + 0.5, "···", F_LAB, 700, a="middle")
            tx = cx + 4.4
        T(f, tx, yy + 0.7, lab, F_NOTE)
        cx = tx + len(lab) * 0.93 + 8.0


# ======================================================================== a: architecture
def panel_arch7(f, ox, oy, w=183.0, letter="a"):
    hatch_defs(f)
    f.plabel(ox, oy + 3.4, letter, "S1 architecture")
    legend7(f, ox + 66.0, oy + 3.6)
    cy0, ch = oy + 6.2, 67.8
    xI, wI = ox + 0.5, 56.0
    xII, wII = xI + wI + 1.5, 70.0
    xIII = xII + wII + 1.5
    wIII = ox + w - 0.5 - xIII
    card(f, xI, cy0, wI, ch, "I", "I. Per-plane 2D encoder", "every plane, shared weights")
    card(f, xII, cy0, wII, ch, "II", "II. 3D residual decoder", "new · lr 2·10^{−4}")
    card(f, xIII, cy0, wIII, ch, "III", "III. Output & training target")
    cy = oy + 31.0
    lab1, lab2 = oy + 40.4, oy + 42.8
    gap = 2.6                                    # module-to-module arrow length (even rhythm)

    # ---------------- stage I
    bh, pitch, bd = 4.0, 4.6, 2.6
    b1 = oy + 20.2
    ix, iy_, isz, iL = xI + 2.0, b1 + 0.4, 11.0, 5.6
    block(f, ix, iy_, isz, isz, iL, "#000", "#000", "#000", C2D["st"], front="../assets3/in_front.png", top="../assets3/in_top.png",
          side="../assets3/in_side.png", planes=3)
    icx = ix + isz / 2 + iL * DEP[0] / 2
    T(f, icx, oy + 35.2, "input", F_LAB, 700, a="middle")
    T(f, icx, oy + 37.6, "1×Z×256×256", F_NOTE, c=MUTE, a="middle")
    T(f, icx, oy + 39.9, "Z = 32 planes", F_NOTE, c=MUTE, a="middle")
    iend = ix + isz + iL * DEP[0]
    px, pw = iend + gap, 7.6
    arr(f, iend + 0.2, b1 + bh / 2, px - 0.25, S2D)
    cub(f, px, b1, pw, bh, 1.8, C2D, "patch", size=F_LAB, hatch=True)
    T(f, px + pw / 2 + 0.6, b1 - 2.4, "8×8, frozen", F_NOTE, c=MUTE, a="middle")
    pyy = b1 + 7.0
    cub(f, px, pyy, pw, bh, 1.8, C2D, "+ pos.", size=F_LAB)
    T(f, px + pw / 2, pyy + bh + 2.6, ["embedding", "per plane", "32×32×1024"], F_NOTE, c=MUTE, a="middle")
    pend = px + pw + 1.8 * DEP[0]
    qx = pend + 2.1
    sx = qx + 3.1
    f.plus_node(qx, b1 + bh / 2, r=1.0, col=S2D)
    arr(f, pend + 0.2, b1 + bh / 2, qx - 1.15, S2D)
    arr(f, qx + 1.05, b1 + bh / 2, sx - 0.25, S2D)
    f.path(f"M{pend + 0.2:.2f},{pyy + bh / 2:.2f} H{qx:.2f} V{b1 + bh / 2 + 1.25:.2f}", stroke=S2D, sw=1.1 * LW)
    sw_ = 19.0
    ys = [b1, b1 + pitch, b1 + 2 * pitch + 2.9, b1 + 3 * pitch + 2.9]
    for yy, nm in zip(ys, ["block 1", "block 2", "block 24", "neck → 256"]):
        cc = C2D if not nm.startswith("neck") else dict(front=tint(BASE, 0.40), top=tint(BASE, 0.22), side=tint(BASE, 0.60), st=C2D["st"])
        cub(f, sx, yy, sw_, bh, bd, cc, nm, size=F_LAB)
    T(f, sx + sw_ / 2, b1 + 2 * pitch + 1.6, "···", FS["mod"], 700, c=C2D["st"], a="middle")
    scx = sx + sw_ / 2 + bd * DEP[0] / 2
    T(f, scx - 1.2, oy + 13.2, "ViT-L · 24 blocks · 304 M", F_LAB, 700, a="middle")
    T(f, scx - 1.2, oy + 16.6, "global attention · lr 10^{−5}", F_NOTE, c=MUTE, a="middle")
    num(f, sx - 2.0, b1 + 2 * pitch + 1.2, 1)
    f.prov_add("patch embedding 8x8 conv, stride 8 (SAM 16x16 kernel subsampled), weights not trained; absolute position embedding (1,32,32,1024), trained",
               "cellpose vit.py:109-116 (CPSAM); net3d.py:202-205 patch-embed requires_grad False, :228-232 _embed")
    f.prov_add("ViT-L encoder: 24 blocks, width 1024, 16 heads, MLP ratio 4, relative position bias, global attention in every block; applied to all B*Z planes with shared weights; 304.4 M encoder-side parameters, lr 1e-5",
               "segment_anything build_sam.py:27-34, 66-80; cellpose vit.py:118-120; net3d.py:285-286; train3d.py:144, 211-214; train log")
    f.prov_add("neck: 1x1 conv 1024->256, LayerNorm2d, 3x3 conv 256->256, LayerNorm2d", "segment_anything image_encoder.py:88-104")

    # ---------------- stage II
    ny = ys[3] + bh / 2
    s0 = xII + 2.6
    f.path(f"M{sx + sw_ + bd * DEP[0] + 0.2:.2f},{ny:.2f} H{xII - 0.75:.2f} V{cy:.2f} H{s0 - 0.3:.2f}", stroke=S2D, sw=AW)
    sh_, sd_, slw = 11.0, 5.0, 7.6
    slab7(f, s0, cy - sh_ / 2, slw, sh_, sd_, 5, C2D)
    send = s0 + slw + sd_ * DEP[0]
    T(f, (s0 + send) / 2, cy - sh_ / 2 - 4.6, "stack along z", F_NOTE, c=MUTE, a="middle")
    T(f, s0 + slw / 2 + 1.0, lab1, "Z×256×32×32", F_NOTE, c=MUTE, a="middle")
    vx, vw, vh, vL = send + gap, 9.5, 12.0, 6.0
    arr(f, send + 0.2, cy, vx - 0.25, S2D)
    block(f, vx, cy - vh / 2, vw, vh, vL, C2D["front"], C2D["top"], C2D["side"], C2D["st"], top="../assets3/feat_enc_top.png", planes=3)
    for k_ in range(1, 5):
        f.add(f'<line x1="{vx + k_ * vw / 5:.2f}" y1="{cy - vh / 2:.2f}" x2="{vx + k_ * vw / 5:.2f}" y2="{cy + vh / 2:.2f}" stroke="{C2D["st"]}" stroke-width="{0.5 * LW:.3f}" opacity="0.55"/>')
        f.add(f'<line x1="{vx:.2f}" y1="{cy - vh / 2 + k_ * vh / 5:.2f}" x2="{vx + vw:.2f}" y2="{cy - vh / 2 + k_ * vh / 5:.2f}" stroke="{C2D["st"]}" stroke-width="{0.5 * LW:.3f}" opacity="0.55"/>')
    T(f, vx + vw / 2 + 1.5, lab1, "256×Z×32×32", F_NOTE, c=MUTE, a="middle")
    f.prov_add("plane features re-stacked to (B,256,Z,32,32)", "net3d.py:288")
    vend = vx + vw + vL * DEP[0]
    c1 = vend + gap
    arr(f, vend + 0.2, cy, c1 - 0.25, S3D)
    cub(f, c1, cy - 5.0, 2.2, 10.0, 2.4, C3D)
    T(f, c1 + 1.6, cy - 7.4, "1^{3}", F_NOTE, c=PAL["new_d"], a="middle")
    rf = [1]
    for dd in (1, 2, 4, 8):
        rf.append(rf[-1] + 2 * dd + 2)
    cend = c1 + 2.2 + 2.4 * DEP[0]
    r0, rbw, rbL, rstep = cend + gap, 3.4, 5.4, 6.6
    rfoot = rbw + rbL * DEP[0] if STYLE["v"] == "b" else 3.9
    arr(f, cend + 0.15, cy, r0 - 0.2, S3D)
    for j, dd in enumerate((1, 2, 4, 8)):
        rx = r0 + j * rstep
        if j:
            arr(f, rx - rstep + rfoot + 0.1, cy, rx - 0.15, S3D)
        block(f, rx, cy - 6.0, rbw, 12.0, rbL, C3D["front"], C3D["top"], C3D["side"], C3D["st"], sw=0.8 * LW, flat_w=rfoot,
              top="../assets3/feat_dec_top.png" if j == 3 else None)
        T(f, rx + rbw / 2, lab1, f"{dd}", F_LAB, 700, c=PAL["new_d"], a="middle")
        T(f, rx + rbw / 2, lab2, f"{rf[j + 1]}", F_LAB, c=INK, a="middle")
    dend = r0 + 3 * rstep + rfoot
    # aligned mini-table under the decoder: row headers + a thin rule
    T(f, r0 - 1.3, lab1, "d", F_NOTE, c=MUTE, a="end", italic=True)
    T(f, r0 - 1.3, lab2, "RF_{z}", F_NOTE, c=MUTE, a="end")
    f.add(f'<line x1="{r0 - 6.0:.2f}" y1="{lab1 - 2.3:.2f}" x2="{dend - 1.0:.2f}" y2="{lab1 - 2.3:.2f}" stroke="{tint(RED, 0.5)}" stroke-width="{0.5 * LW:.3f}"/>')
    T(f, (c1 + dend) / 2 + 0.5, oy + 19.2, "3D residual decoder · 14.3 M", F_LAB, 700, c=PAL["new_d"], a="middle")
    num(f, (c1 + dend) / 2 - 17.6, oy + 18.6, 2)
    f.prov_add("decoder: 1x1x1 Conv3d 256->256, four ResBlock3D with z-dilation 1,2,4,8 (shape kept 256xZx32x32), 14.3 M params, lr 2e-4; z receptive field after blocks 1-4 = 5, 11, 21, 39 planes (derived)",
               "net3d.py:69-96; model3d.json dec_*; train3d.py:145 lr_dec 2e-4; train log '新增模块 14.3M'")

    # ---------------- stage III
    zx = xIII + 2.8
    arr(f, dend + 0.2, cy, zx - 0.25, S3D)
    zc = dict(front="#ffffff", top="#ffffff", side=tint(RED, 0.06), st=RED)
    cub(f, zx, cy - 5.0, 2.6, 10.0, 2.4, zc, dash="0.7 0.45")
    T(f, zx + 1.3, cy + 0.9, "0", F_NOTE, 700, c=RED, a="middle")
    T(f, zx + 2.3, cy - 7.4, "1^{3} · 4·64", F_NOTE, c=PAL["new_d"], a="middle")
    zend = zx + 2.6 + 2.4 * DEP[0]
    qx3 = zend + gap + 1.6
    f.plus_node(qx3, cy, r=1.6)
    arr(f, zend + 0.2, cy, qx3 - 1.85, S3D)
    T(f, qx3 - 2.9, cy - 1.1, "Δ", F_LAB, 700, c=RED, a="middle")
    hx, hy, hw, hh = xIII + 2.8, oy + 12.8, 27.5, 4.4
    cub(f, hx, hy, hw, hh, 1.8, C2D, "2D head: 1×1 conv → 3·64", size=F_LAB)
    vcx, vtop = vx + vw / 2 + vL * DEP[0] / 2, cy - vh / 2 + vL * DEP[1] / 2
    f.path(f"M{vcx:.2f},{vtop - 0.1:.2f} V{hy + hh / 2:.2f} H{hx - 0.25:.2f}", stroke=S2D, sw=AW)
    T(f, (vcx + xII + wII) / 2, hy + hh / 2 - 1.0, "same features, per plane", F_NOTE, c=MUTE, a="middle", italic=True)
    f.path(f"M{qx3:.2f},{hy + hh + 0.2:.2f} V{cy - 1.85:.2f}", stroke=S2D, sw=AW)
    T(f, qx3 + 1.0, cy - 12.0, "d_{y}, d_{x}, p", F_NOTE, c=MUTE)
    f.prov_add("2D head = Cellpose-SAM 1x1 Conv2d(256 -> 3*64) per plane; Delta added for (dy, dx, p); dz = Delta_z; x8 pixel shuffle back to 256x256",
               "net3d.py:287-295; cellpose vit.py:122")
    ox_, ow, oL = qx3 + 1.6 + gap + 0.8, 10.0, 5.6
    arr(f, qx3 + 1.8, cy, ox_ - 0.25, INK)
    T(f, (qx3 + ox_) / 2 + 0.3, cy + 3.5, "×8", F_NOTE, c=MUTE, a="middle")
    block(f, ox_, cy - 6.0, ow, 12.0, oL, "#ffffff", "#ffffff", tint(BASE, 0.10), INK, top="../assets3/out_top.png", sw=0.8 * LW, planes=3)
    f.image(ox_ + 0.3, cy - 5.6, ow - 0.6, 11.2, "flow_glyphs_pal.png")
    oend = ox_ + ow + oL * DEP[0]
    T(f, ox_ + ow / 2 + 1.5, lab1, "(d_{z}, d_{y}, d_{x}, p)", F_LAB, 700, a="middle")
    T(f, ox_ + ow / 2 + 1.5, lab2 + 1.1, "4×Z×256×256", F_NOTE, c=MUTE, a="middle")
    num(f, ox_ - 1.6, cy - 4.6, 4)
    num(f, zx + 1.3, lab2 - 0.6, 3)
    lx = oend + gap
    lw_ = xIII + wIII - 1.6 - lx
    ly, lh = cy - 7.2, 14.4
    f.rect(lx, ly, lw_, lh, rx=1.0, fill=tint(LOSS, 0.10), stroke=LOSS, sw=0.8 * LW)
    T(f, lx + lw_ / 2, ly + 3.0, "loss", F_LAB, 700, c=LOSS, a="middle")
    T(f, lx + lw_ / 2, ly + 5.7, ["½ MSE(flows, 5*u*)", "+ BCE(p, fg)"], F_NOTE, 700, c=LOSS, a="middle", ls=1.25)
    T(f, lx + lw_ / 2, ly + 10.6, ["vs the 3D target,", "inside the volume"], F_NOTE, c=MUTE, a="middle", ls=1.25)
    arr(f, oend + 0.2, cy, lx - 0.25, LOSS, dash="0.9 0.6")
    f.prov_add("loss: MSE(net flow, 5 x unit target)/2 + BCE(cellprob, nucleus) over voxels inside the volume; AdamW, bf16 autocast, grad clip 1.0",
               "train3d.py:33-43, 215, 275-280")

    # ================= zoom insets + funnels
    iy, ih = oy + 47.0, 25.9
    ex, ew = xI + 1.2, wI - 2.4
    inset(f, ex, iy, ew, ih, "transformer block (×24)", "#4a5577", "tokens 32×32×1024")
    funnel(f, (sx, ys[3] + bh + 0.3), (sx + sw_, ys[3] + bh + 0.3), (ex + 5.0, iy - 0.2), (ex + ew - 5.0, iy - 0.2), C2D["st"])
    seq = [("LN", 4.4), ("MHSA", 9.2), ("+", 0), ("LN", 4.4), ("MLP", 9.2), ("+", 0)]
    bx, by_, bh_ = ex + 4.0, iy + 8.4, 5.0
    xs_ = []
    for i, (nm, bw_) in enumerate(seq):
        if nm == "+":
            f.plus_node(bx + 1.3, by_ + bh_ / 2, r=1.2, col=S2D)
            xs_.append(bx + 1.3)
            bx += 2.6
        else:
            cub(f, bx, by_, bw_, bh_, 1.1, C2D, nm, size=F_LAB)
            xs_.append(bx + bw_ / 2)
            bx += bw_ + 1.1 * DEP[0]
        if i < 5:
            arr(f, bx + 0.15, by_ + bh_ / 2, bx + 1.85, S2D)
            bx += 2.0
    for a_, b_ in ((ex + 2.4, xs_[2]), (xs_[2] + 1.8, xs_[5])):
        f.path(f"M{a_:.2f},{by_ + bh_ / 2:.2f} V{by_ - 2.0:.2f} H{b_:.2f} V{by_ + bh_ / 2 - 1.4:.2f}", stroke=S2D, sw=0.9 * LW)
    f.path(f"M{ex + 1.0:.2f},{by_ + bh_ / 2:.2f} H{ex + 4.0:.2f}", stroke=S2D, sw=0.9 * LW, arrow=False)
    T(f, xs_[1], by_ + bh_ + 2.5, ["16 heads, global,", "rel. position"], F_NOTE, c=MUTE, a="middle")
    T(f, xs_[4], by_ + bh_ + 2.5, ["1024→4096", "→1024, GELU"], F_NOTE, c=MUTE, a="middle")
    T(f, ex + 1.6, iy + ih - 4.3, "neck: 1×1 conv 1024→256 · LN · 3×3 conv · LN → 32×32×256", F_NOTE)
    T(f, ex + 1.6, iy + ih - 1.7, "patch embed frozen; all other encoder weights fine-tuned", F_NOTE, c=MUTE)
    f.prov_add("transformer block: x + MHSA(LN(x)); x + MLP(LN(x)); MHSA 16 heads with relative position bias, global; MLP 1024->4096->1024 GELU",
               "segment_anything image_encoder.py Block (norm1, attn, norm2, mlp, shortcut) l.151-175; build_sam.py:27-34, 66-80")
    # II
    rxi, rwi = xII + 1.2, wII - 2.4
    inset(f, rxi, iy, rwi, ih, "residual block (×4)", PAL["new_d"], "256×Z×32×32 in and out")
    fy0 = lab2 + 1.5
    f.add(f'<path d="M{r0:.2f},{fy0 - 0.6:.2f} V{fy0:.2f} H{dend - 1.0:.2f} V{fy0 - 0.6:.2f}" stroke="{tint(RED, 0.7)}" stroke-width="{0.6 * LW:.3f}" fill="none"/>')
    funnel(f, (r0, fy0), (dend - 1.0, fy0), (rxi + 5.0, iy - 0.2), (rxi + rwi - 5.0, iy - 0.2), tint(RED, 0.7))
    seq2 = [("GN", 4.2), ("GELU", 6.2), ("conv 3^{3}", 9.6), ("GN", 4.2), ("GELU", 6.2), ("conv 3^{3}", 9.6)]
    bx = rxi + 4.0
    first = bx
    for i, (nm, bw_) in enumerate(seq2):
        conv = nm.startswith("conv")
        cc = C3D if conv else dict(front="#ffffff", top=tint(RED, 0.05), side=tint(RED, 0.16), st=C3D["st"])
        cub(f, bx, by_, bw_, bh_, 1.1, cc, nm, size=F_LAB, bold=conv)
        if i == 2:
            T(f, bx + bw_ / 2, by_ + bh_ + 2.15, "dilation (d,1,1)", F_NOTE, 700, c=PAL["new_d"], a="middle")
        bx += bw_ + 1.1 * DEP[0]
        if i < 5:
            arr(f, bx + 0.15, by_ + bh_ / 2, bx + 1.55, S3D)
            bx += 1.7
    pxn = bx + 2.6
    f.plus_node(pxn, by_ + bh_ / 2, r=1.2, col=S3D)
    arr(f, bx + 0.15, by_ + bh_ / 2, pxn - 1.35, S3D)
    f.path(f"M{first - 1.6:.2f},{by_ + bh_ / 2:.2f} V{by_ - 2.0:.2f} H{pxn:.2f} V{by_ + bh_ / 2 - 1.4:.2f}", stroke=S3D, sw=0.9 * LW)
    f.path(f"M{rxi + 1.0:.2f},{by_ + bh_ / 2:.2f} H{first - 0.2:.2f}", stroke=S3D, sw=0.9 * LW, arrow=False)
    nz, pz = 39, 0.2
    top = iy + 17.4
    for j, dd in enumerate((1, 2, 4, 8)):
        cxj = rxi + 3.4 + j * 10.5
        cw, cdp = 4.4, 1.8
        ddx, ddy = cdp * DEP[0], cdp * DEP[1]
        for q in range(nz):
            yy = top + q * pz
            off = q - nz // 2
            col = RED if off in (-dd, 0, dd) else tint(BASE, 0.22)
            f.add(f'<polygon points="{cxj:.2f},{yy:.2f} {cxj + cw:.2f},{yy:.2f} {cxj + cw + ddx:.2f},{yy + ddy:.2f} {cxj + ddx:.2f},{yy + ddy:.2f}" '
                  f'fill="{col}" stroke="none"/>')
        r = rf[j + 1]
        q0, q1 = nz // 2 - r // 2, nz // 2 + r // 2
        y0b, y1b = top + q0 * pz - 0.4, top + q1 * pz + 0.2
        bxb = cxj + cw + ddx + 0.8
        f.add(f'<path d="M{bxb:.2f},{y0b:.2f} h0.6 V{y1b:.2f} h-0.6" stroke="{PAL["new_d"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
        T(f, bxb + 1.2, (y0b + y1b) / 2 + 0.8, f"{r}", F_LAB, 700, c=PAL["new_d"])
    capx = rxi + 3.4 + 3 * 10.5 + 4.4 + 1.8 * DEP[0] + 6.0
    T(f, capx, iy + 16.6, ["red: z taps of the", "dilated conv; bracket:", "z receptive field", "(planes) after block"], F_NOTE, c=MUTE, ls=1.2)
    f.prov_add("ResBlock3D: x + conv3(GELU(GN(conv3_dil(GELU(GN(x)))))), GroupNorm 16 groups; first conv dilation (d,1,1)", "net3d.py:69-78")
    # III
    zxi, zwi = xIII + 1.2, wIII - 2.4
    inset(f, zxi, iy, zwi, ih, "output composition", PAL["new_d"], "step 0 = 2D control, d_{z} = 0")
    funnel(f, (zx - 0.4, lab2 + 1.2), (qx3 + 1.0, lab2 + 1.2), (zxi + 5.0, iy - 0.2), (zxi + 22.0, iy - 0.2), tint(RED, 0.7))
    cA, cB = zxi + 13.0, zxi + zwi - 10.5
    T(f, cA, iy + 7.1, "step 0", F_LAB, 700, a="middle")
    T(f, cB, iy + 7.1, "trained", F_LAB, 700, c=PAL["new_d"], a="middle")
    rowsZ = [("d_{z}", "0", "Δ_{z}"), ("d_{y}", "2D", "2D+Δ"), ("d_{x}", "2D", "2D+Δ"), ("p", "2D", "2D+Δ")]
    for k_, (chn, a_, b_) in enumerate(rowsZ):
        yy = iy + 8.4 + k_ * 3.75
        T(f, zxi + 3.0, yy + 2.3, chn, F_NOTE)
        zero = a_ == "0"
        cub(f, cA - 4.6, yy, 9.2, 3.2, 0.8, zc if zero else C2D, a_, bold=zero, dash="0.7 0.45" if zero else None, lc=RED if zero else INK)
        cub(f, cB - 5.4, yy, 10.8, 3.2, 0.8, C3D, b_, bold=False)
    arr(f, cA + 5.6, iy + 8.4 + 3.75 + 1.6, cB - 6.0, iy + 8.4 + 3.75 + 1.6, INK)
    f.prov_add("output composition (dz, dy, dx, p) = (Delta_z, 2D head + Delta); Delta = 0 at step 0, so S1 equals the seed-matched 2D model (2D control) with dz = 0",
               "net3d.py:89-90, 294; model3d.json init orx_v5_2d_s0; net3d.py:359-371 self-test")

    # ================= rationale + stream legend
    ry = cy0 + ch + 3.3
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


# ======================================================================== b: how z enters the flow
def panel_cmp7(f, ox, oy, w, letter="b"):
    f.plabel(ox, oy + 3.0, letter, "How z information enters the flow (xz section through one nucleus)")
    ty, th = oy + 5.2, 35.2
    w1 = 56.0
    w2 = w - w1 - 2.0
    x1, x2 = ox + 0.5, ox + 0.5 + w1 + 2.0
    card(f, x1, ty, w1, th, "I", "single view: 2D vs S1")
    card(f, x2, ty, w2 - 0.5, th, "II", "angle to the 3D target, as evaluated")
    gh = 16.8
    gw = gh * 530 / 390
    gy = ty + 11.6
    # card 1
    xa = x1 + 4.0
    xb = x1 + w1 - 2.0 - gw
    zarrow(f, x1 + 1.7, gy + 2.5, gy + 8.0)
    for xg, nm, h1, c1, h2 in ((xa, "cmp_2d_xyonly", "2D control, single xy view", PAL["ctl_d"], "d_{z} ≡ 0"),
                               (xb, "cmp_s1_view1", "S1, same single view", PAL["new_d"], "d_{z} predicted")):
        T(f, xg + gw / 2, ty + 7.6, h1, F_LAB, 700, c=c1, a="middle")
        T(f, xg + gw / 2, ty + 10.0, h2, F_NOTE, c=MUTE, a="middle")
        f.image(xg, gy, gw, gh, f"../assets7/{nm}.png")
    arr(f, xa + gw + 0.5, gy + gh / 2, xb - 0.5, gy + gh / 2, S3D)
    cbw = 14.0
    cbx = x1 + w1 / 2 - cbw / 2 + 2.0
    f.colorbar(cbx, ty + th - 3.6, cbw, 1.3, "", "", None)
    T(f, cbx - 1.0, ty + th - 2.35, "d_{z}  −1", F_NOTE, c=MUTE, a="end")
    T(f, cbx + cbw + 1.0, ty + th - 2.35, "+1", F_NOTE, c=MUTE)
    # card 2
    med = LOG3["angle_nucleus62"]["median_deg"]
    xc = x2 + 4.2
    xd = xc + gw + 2.6
    zarrow(f, x2 + 1.7, gy + 2.5, gy + 8.0)
    T(f, x2 + w2 - 0.5 - 1.4, ty + 3.35, "arrows: prediction", F_NOTE, c=MUTE, a="end")
    for xg, nm, h1, c1, m_ in ((xc, "ang_2d", "2D control (3 views fused)", PAL["ctl_d"], med["2D"]),
                               (xd, "ang_s1", "S1", PAL["new_d"], med["S1"])):
        T(f, xg + gw / 2, ty + 7.6, h1, F_LAB, 700, c=c1, a="middle")
        T(f, xg + gw / 2, ty + 10.0, f"median {m_:.0f}°", F_NOTE, c=MUTE, a="middle")
        f.image(xg, gy, gw, gh, f"../assets7/{nm}.png")
    gid = "angbar7"
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in ((0, "#ffffff"), (0.33, PM["cellos"]), (0.67, PM["S1"]), (1, PM["S1_dark"])))
    f.defs.append(f'<linearGradient id="{gid}" x1="0" x2="0" y1="1" y2="0">{stops}</linearGradient>')
    bx = xd + gw + 2.4
    f.add(f'<rect x="{bx:.2f}" y="{gy + 1.0:.2f}" width="1.4" height="{gh - 2.0:.2f}" rx="0.3" fill="url(#{gid})" stroke="{PAL["box"]}" stroke-width="{0.5 * LW:.3f}"/>')
    T(f, bx + 2.0, gy + 2.6, "90°", F_NOTE, c=MUTE)
    T(f, bx + 2.0, gy + gh - 1.0, "0°", F_NOTE, c=MUTE)
    sy = ty + th - 6.4
    sx0, sw0 = x2 + 1.6, w2 - 0.5 - 3.2
    f.rect(sx0, sy, sw0, 5.2, rx=0.9, fill=tint(RED, 0.13), stroke=tint(RED, 0.45), sw=0.5 * LW, chk=False)
    T(f, sx0 + sw0 / 2, sy + 2.25, "all 12 datasets, 5 seeds:", F_NOTE, 700, c=PAL["new_d"], a="middle")
    T(f, sx0 + sw0 / 2, sy + 4.45, "S1 − 2D median angle −3.18° (−3.41 to −2.89)", F_NOTE, 700, c=PAL["new_d"], a="middle")
    f.prov_add("left pair: 2D control on xy planes only (u_z = 0 by construction) vs S1 single identity view on the same planes, nucleus 62",
               "make_a3_assets.py cmp_2d_xyonly / cmp_s1_view1 (v2v4_crop.npz c2d_out, s1_out; nc-viz viz_run.py); assets7 = margin crops (make_a7_assets.py)")
    f.prov_add(f"right pair: per-voxel 3D angle to the GT-derived target (FLOWERR definition, region GT fg & |t|>0.5) for the 2D control as evaluated (3 views fused) and S1 as evaluated, nucleus 62: medians {med['2D']:.1f} / {med['S1']:.1f} deg (n = {LOG3['angle_nucleus62']['n_vox']} voxels)",
               "make_a3_assets.py angle section; definition flowerr_angles.py; dP from thr_eval npz runs 81116913 / 07992b55")
    f.prov_add("macro over 12 datasets, five seeds: S1 - 2D difference in median angle -3.18 deg (-3.41 to -2.89); macro medians S1/2D 12.4/15.6 deg; post hoc, S1 trained on this target",
               "NUMBERS.md:631 (primary.set12_region_a); main.tex Fig. 3 caption f; source_data/inputs/flowerr_results.json")


# ======================================================================== c: 3D training target
def panel_target7(f, ox, oy, w, letter="c"):
    f.plabel(ox, oy + 3.0, letter, "3D training target")
    ty, th = oy + 5.2, 35.2
    card(f, ox, ty, w, th, "III", "GT nuclei, mouse embryo")
    T(f, ox + w - 1.6, ty + 3.35, "grey arrow in ③: +z, 3 µm", F_NOTE, c=MUTE, a="end")
    rh = 18.8
    ws = [rh * 559 / 683, rh * 559 / 683, rh * 1638 / 1581]
    ag = 4.2
    tot = sum(ws) + 2 * ag
    x = ox + (w - tot) / 2
    ry = ty + 6.2
    names = [("../assets7/diff_source.png", ["heat source at", "the label centre"]),
             ("../assets7/diff_heat.png", ["diffuse inside the", f"nucleus ({DF['niter']} steps)"]),
             ("target_F1d.png", ["target flow u,", "4 nuclei (colour u_{z})"])]
    for i, ((nm, cap), wi) in enumerate(zip(names, ws)):
        well(f, x - 0.4, ry - 0.4, wi + 0.8, rh + 0.8)
        f.image(x, ry, wi, rh, nm)
        cx0 = x - 0.4 if i < 2 else x - 1.6
        stepnum(f, cx0 + 1.2, ry + rh + 3.0, i + 1, LOSS)
        T(f, cx0 + 3.3, ry + rh + 3.65, cap, F_NOTE, c=MUTE, ls=1.4)
        if i < 2:
            arr(f, x + wi + 0.7, ry + rh / 2, x + wi + ag - 0.7, ry + rh / 2, LOSS)
        x += wi + ag
    f.prov_add("tile 3: nc-viz F1-d_target_light, GT-derived 3D target (flows3d_gpu of S1 data3d.py) on E15_Ab2 GT 33, 51, 68, 50 (rule F1_group), one sink per nucleus; illustrative, re-interpolated along z for display; grey z arrow = 3 um (F1-d.provenance.json 'style')",
               "paper/figures/assets_viz/F1-d.provenance.json")
    f.prov_add(f"3D Cellpose target: heat source at the label's most central voxel, diffusion restricted to the nucleus for niter = 6 x sum of bounding-box extents ({DF['niter']} here), unit gradient = flow; GT nucleus 62 of E15_Ab2 resampled to the training grid",
               "data3d.py:28-81 flows3d_gpu (centre l.51-57, niter l.58-62, update l.71-74, gradient l.75-77); diffusion_data.py, render_diffusion.py; assets7 = margin crops")


# ======================================================================== d: inference
def zarrow(f, x, y0, y1, lab_left=False):
    """Small +z arrow for 2D xz sections (z grows downwards in every section: ylim inverted in make_a3_assets.py)."""
    f.path(f"M{x:.2f},{y0:.2f} L{x:.2f},{y1:.2f}", stroke=MUTE, sw=0.6 * LW, chk=False)
    if lab_left:
        T(f, x - 0.8, y1 - 0.4, "+z", F_NOTE, c=MUTE, a="end")
    else:
        T(f, x, y1 + 2.3, "+z", F_NOTE, c=MUTE, a="middle")


def plot_area(f, x, y, w, h, xt, yt):
    f.rect(x, y, w, h, rx=0.2, fill="#ffffff", stroke="none", chk=False)
    for v in xt:
        f.add(f'<line x1="{v:.2f}" y1="{y:.2f}" x2="{v:.2f}" y2="{y + h:.2f}" stroke="#E4E4E6" stroke-width="{0.5 * LW:.3f}"/>')
    for v in yt:
        f.add(f'<line x1="{x:.2f}" y1="{v:.2f}" x2="{x + w:.2f}" y2="{v:.2f}" stroke="#E4E4E6" stroke-width="{0.5 * LW:.3f}"/>')
    f.add(f'<path d="M{x:.2f},{y:.2f} V{y + h:.2f} H{x + w:.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}" fill="none"/>')


def panel_infer7(f, ox, oy, w, letter="d"):
    f.plabel(ox, oy + 3.0, letter, "Inference on one test volume (mouse embryo, S1 seed 0)")
    n, gap = 7, 2.6
    bw = (w - 1.0 - (n - 1) * gap) / n
    by, bh = oy + 5.2, 33.8
    xs = [ox + 0.5 + i * (bw + gap) for i in range(n)]
    heads = ["diameter", "isotropic", "z windows", "three views", "follow flow", "rule T", "3D nuclei"]
    fams = ["I", "I", "II", "II", "III", "III", "III"]
    for i, x in enumerate(xs):
        step_card(f, x, by, bw, bh, fams[i], i + 1, heads[i])
        if i < n - 1:
            arr(f, x + bw + 0.15, by + 2.5, x + bw + gap - 0.15, by + 2.5, GREY_ARROW, chk=False)
    lw_d = 0.9 * LW
    # 1 diameter
    x = xs[0]
    rows = [r for r in RD["diam"]["E15_Ab1"] if r["thr"] == 0.0]
    gx0, gy0, gw, gh = x + 5.4, by + 7.2, bw - 8.2, 13.2
    lx = lambda s_: gx0 + gw * (math.log(s_) - math.log(6)) / (math.log(150) - math.log(6))
    ly = lambda v: gy0 + gh - gh * (v + 0.6) / 1.4
    plot_area(f, gx0, gy0, gw, gh, [lx(s_) for s_ in (6, 30, 150)], [ly(v) for v in (-0.5, 0, 0.5)])
    for v_, lab_ in ((-0.5, "−0.5"), (0.0, "0"), (0.5, "0.5")):
        T(f, gx0 - 0.8, ly(v_) + 0.7, lab_, F_NOTE, c=MUTE, a="end")
    for key, col in (("stab", PAL["ctl"]), ("score", INK)):
        pts = [(lx(r["scale"]), ly(r["stab"] if key == "stab" else r["stab"] - r["cons"])) for r in rows]
        f.add('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + f'" stroke="{col}" stroke-width="{lw_d:.3f}" fill="none"/>')
        for a, b in pts:
            f.circle(a, b, 0.45, fill=col, chk=False)
    sel = RD["diam_sel"]["E15_Ab1"][0]
    rs = [r for r in rows if r["scale"] == sel["scale"]][0]
    f.circle(lx(sel["scale"]), ly(rs["stab"] - rs["cons"]), 1.0, fill="none", stroke=PAL["new"], sw=1.0 * LW, chk=False)
    for s_ in (6, 30, 150):
        T(f, lx(s_), gy0 + gh + 2.5, f"{s_}", F_NOTE, c=MUTE, a="middle")
    T(f, gx0 + gw / 2, by + 25.5, "scale (px)", F_NOTE, c=MUTE, a="middle")
    T(f, x + 1.4, by + 28.6, ["blue stability; black", f"minus ∣log d̂/d∣; d̂ = {sel['d_used']:.0f}"], F_NOTE, c=MUTE, ls=1.5)
    T(f, gx0 + 1.0, gy0 + gh - 1.2, "E15_Ab1", F_NOTE, c=MUTE)
    f.prov_add("diameter scan: 2D control seed 0, cp 0: stability (blue) and stability - |log(d_est/scale)| (black), both dimensionless on one shared y axis for mouse_embryo E15_Ab1; scales 6 and 10 px skipped (rescaled volume > 8e8 voxels); selected scale 30, threshold -2, d_used 33.8 px (GT 32.5)",
               f"server {_NSP_ROOT}/results/orx_4a362b0f-..._test/rows_all.json and rows.json (eval_val.py joint r2a2); arch_views.py -> assets3/ruleT_diam.json")
    # 2 isotropic
    x = xs[1]
    tt = bw - 2.8
    for yy, nm in ((by + 6.0, "xz_native.png"), (by + 19.4, "xz_iso.png")):
        well(f, x + 1.4 - 0.3, yy - 0.3, tt + 0.6, 10.8 + 0.6)
        f.image(x + 1.4, yy, tt, 10.8, nm, fit="xMidYMid slice")
    T(f, x + bw / 2, by + 18.6, "z/xy = 4.65 → linear z", F_NOTE, c=MUTE, a="middle")
    T(f, x + bw / 2, by + 32.3, "×30/d, isotropic", F_NOTE, c=MUTE, a="middle")
    f.prov_add("isotropic resampling of E15_Ab2 (z/xy 4.65): native xz plane vs linear z interpolation", "eval_gt.py:217-222; infer3d.py:211-217; make_assets.py")
    # 3 windows
    x = xs[2]
    st, Zp = LOGA["windows"]["starts"], LOGA["windows"]["Z_network"]
    a0, a1 = x + 1.6, x + bw - 1.6
    base, amp = by + 19.2, 11.0
    f.rect(a0 - 0.3, base - amp - 0.9, a1 - a0 + 0.6, amp + 1.2, rx=0.2, fill="#ffffff", stroke="none", chk=False)
    for v in (0.5, 1.0):
        f.add(f'<line x1="{a0:.2f}" y1="{base - amp * v:.2f}" x2="{a1:.2f}" y2="{base - amp * v:.2f}" stroke="#E4E4E6" stroke-width="{0.5 * LW:.3f}"/>')
    s_ = (a1 - a0) / Zp
    for i, s0 in enumerate(st):
        pts = [(a0 + (s0 + k + 0.5) * s_, base - amp * math.sin(math.pi * (k + 0.5) / 32) ** 2) for k in range(32)]
        f.add('<path d="M' + " L".join(f"{p:.2f},{q:.2f}" for p, q in pts) + f'" stroke="{PAL["new"] if i % 2 == 0 else PAL["new_l"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
    f.add(f'<line x1="{a0:.2f}" y1="{base:.2f}" x2="{a1:.2f}" y2="{base:.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}"/>')
    T(f, x + bw / 2, base + 2.7, f"z′ = {Zp} planes", F_NOTE, c=MUTE, a="middle")
    T(f, x + 1.4, by + 27.3, [f"{len(st)} windows of 32,", "stride 23–24,", "sin² blending"], F_NOTE, c=MUTE, ls=1.2)
    # 4 three views
    x = xs[3]
    vw_ = (bw - 3.6) / 2
    vh_ = vw_ * 199 / 269
    pos = [("z", x + 1.2, by + 6.6), ("y", x + 2.4 + vw_, by + 6.6), ("x", x + 1.2, by + 17.8), ("mean", x + 2.4 + vw_, by + 17.8)]
    for k_, px_, py_ in pos:
        f.image(px_, py_, vw_, vh_, f"../assets7/view_{k_}.png")
        lab = f"{k_} as z" if k_ != "mean" else "mean"
        T(f, px_ + vw_ / 2, py_ + vh_ + 2.3, lab, F_NOTE, c=MUTE if k_ != "mean" else PAL["new_d"], a="middle")
    T(f, x + 1.4, by + 31.6, "d_{z}, nucleus 62", F_NOTE, c=MUTE)
    zarrow(f, x + bw - 1.8, by + 28.9, by + 32.6, lab_left=True)
    f.prov_add("three views: S1 seed-0 raw outputs with each volume axis as network z (one 32-plane window per view centred on GT nucleus 62), u_z on the xz section, and their mean; the pipeline blends all windows with sin^2 weights",
               "arch_views.py (server, CPU) -> assets3/views_E15_Ab2_n62.npz; infer3d.py:63-96 VIEW_PERMS, run_views; assets7 = margin crops")
    # 5 follow flow
    x = xs[4]
    iw = bw - 2.8
    well(f, x + 1.4 - 0.3, by + 6.0 - 0.3, iw + 0.6, iw * 1581 / 1638 + 0.6)
    f.image(x + 1.4, by + 6.0, iw, iw * 1581 / 1638, "flowfollow_F1d.png")
    T(f, x + 1.4, by + 30.0, ["one sink per nucleus", "grey arrow: +z, 3 µm"], F_NOTE, c=MUTE, ls=1.2)
    f.prov_add("follow-flow render: nc-viz F1-d_follow_light (S1 seed 0, E15_Ab2 GT 33, 51, 68, 50, all matched at IoU >= 0.72, one sink each; strands coloured by u_z; grey arrow 3 um along +z)",
               "paper/figures/assets_viz/F1-d.provenance.json (rule F1_group, selection_rules.json)")
    # 6 rule T
    x = xs[5]
    tr = RD["ruleT"]
    ts = [-6, -4, -3, -2]                      # rule-T grid (main.tex Methods 'Rule T'); t = 0, -1 are not candidates
    gx0, gy0, gw, gh = x + 5.6, by + 7.2, bw - 7.0, 13.2
    lx2 = lambda t: gx0 + gw * (t + 6.4) / 4.8
    ly2 = lambda v: gy0 + gh - gh * (v - 0.5) / 1.2
    plot_area(f, gx0, gy0, gw, gh, [lx2(t) for t in (-6, -4, -2)], [ly2(v) for v in (1.0, 1.5)])
    for v_ in (0.7, 1.0, 1.5):
        T(f, gx0 - 0.8, ly2(v_) + 0.7, f"{v_:.1f}", F_NOTE, c=PAL["loss"] if v_ == 0.7 else MUTE, a="end")
    for yv in (0.7, 0.85):
        f.add(f'<line x1="{gx0:.2f}" y1="{ly2(yv):.2f}" x2="{gx0 + gw:.2f}" y2="{ly2(yv):.2f}" stroke="{PAL["loss"]}" stroke-width="{0.5 * LW:.3f}" stroke-dasharray="0.6 0.4"/>')
    pts = [(lx2(t), ly2(tr["m"][f"{float(t)}"])) for t in ts]
    f.add('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + f'" stroke="{INK}" stroke-width="{lw_d:.3f}" fill="none"/>')
    for a, b in pts:
        f.circle(a, b, 0.45, fill=INK, chk=False)
    f.circle(lx2(tr["chosen"]), ly2(tr["m"][f"{tr['chosen']}"]), 1.0, fill="none", stroke=PAL["new"], sw=1.0 * LW, chk=False)
    T(f, lx2(tr["chosen"]) - 1.2, ly2(tr["m"][f"{tr['chosen']}"]) + 2.6, f"{tr['m']['-2.0']:.2f}", F_NOTE, 700, c=PAL["new_d"], a="end")
    T(f, gx0 + 0.5, ly2(0.85) - 0.45, "0.85", F_NOTE, c=PAL["loss"])
    for t in (-6, -4, -2):
        T(f, lx2(t), gy0 + gh + 2.5, f"{t}".replace("-", "−"), F_NOTE, c=MUTE, a="middle")
    T(f, gx0 + gw / 2, by + 25.5, "threshold t", F_NOTE, c=MUTE, a="middle")
    T(f, x + 1.4, by + 28.6, f"m(−2) = {tr['m']['-2.0']:.2f} ≥ 0.7", F_NOTE, c=MUTE)
    T(f, x + 1.4, by + 31.0, "→ t = −2 kept", F_NOTE, c=PAL["new_d"])
    T(f, gx0 + gw - 0.6, gy0 + 2.0, "m(t)", F_NOTE, c=MUTE, a="end")
    f.prov_add(f"rule T on E15_Ab2 (S1 seed 0): m(t) = N_fg(t)/(N_obj(t) V) on the rule grid t in {{-6,-4,-3,-2}} (Methods); dashed lines 0.7 (t = -2 kept if m(-2) >= 0.7) and 0.85 (fallback: highest t with m(t) >= 0.85); m(-2) = {tr['m']['-2.0']:.2f} >= 0.7 so t = -2",
               "sizeThr_dir.py:16-28 recomputed by arch_views.py from run 07992b55 thr_eval rows.json and masks/mouse_embryo__E15_Ab2.npz cp")
    # 7 3D nuclei
    x = xs[6]
    mh = 19.6
    mw = mh * 410 / 600
    f.image(x + (bw - mw) / 2, by + 5.8, mw, mh, "masks_S1.png")
    T(f, x + 1.4, by + 27.7, [f"{tr['n_pred']['-2.0']} objects before", "size filter; F1@0.5",
                              f"{tr['f1_by_thr']['-2.0']:.2f} after"], F_NOTE, c=MUTE, ls=1.3)
    f.prov_add("final S1 masks of E15_Ab2 (render nc-figures), objects before size filter and F1@0.5 after it at t = -2", "E15_Ab2_iso.provenance.json; thr_eval rows.json")


STYLE_NOTE = {
    "A7-b": "reference-figure direction: dashed stage containers with grey corner tabs and 3D cuboid modules; container panes in the user's "
            "3D-viewer warm grey (#EDECE9 panes, white cards inside, tabs #DDDCD7 with a stage-accent stripe); model semantics unchanged "
            "(2D = palette 'base' blue family, S1 #E64B35 accents)",
    "A7-c": "contrasting flat 'Nature/Cell' direction: white cards with hairline borders (#E2E2DE) and a thin tinted header bar with one "
            "accent colour per stage (blue-grey 2D stream / S1 dark red / loss brown); flat rounded modules with soft vector drop shadows "
            "(two offset low-opacity rects, no raster filters); volumes as flat plane stacks; image wells #F5F5F3 from the user's 3D-viewer scheme",
}


def layout_b7():
    STYLE["v"] = "b"
    f = Fig(183.0, 170.0)
    panel_arch7(f, 0.0, 0.0, letter="a")
    panel_cmp7(f, 0.0, 88.6, 120.5, letter="b")
    panel_target7(f, 122.0, 88.6, 60.5, letter="c")
    panel_infer7(f, 0.0, 130.8, 183.0, letter="d")
    return f


def layout_c7():
    STYLE["v"] = "c"
    f = Fig(183.0, 170.0)
    panel_arch7(f, 0.0, 0.0, letter="a")
    panel_cmp7(f, 0.0, 88.6, 120.5, letter="b")
    panel_target7(f, 122.0, 88.6, 60.5, letter="c")
    panel_infer7(f, 0.0, 130.8, 183.0, letter="d")
    return f


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for cid, fn in (("A7-b", layout_b7), ("A7-c", layout_c7)):
        if which not in ("all", cid):
            continue
        export_one(cid, fn)


def export_one(cid, fn):
    f = fn()
    res = export(f, cid)
    print(cid, "issues:", len(res.get("issues", [])), "min font", res.get("min_font_pt"))
    for s_ in res.get("issues", [])[:80]:
        print("   ", s_)
    json.dump(dict(figure=cid, code_snapshot=_nsp_os.path.join(_NSP_REPO, 'src'),
                   model="orx_t3d_r5s1_s0/model3d.json", palette="paper/figures/palette.json",
                   generator=f"paper/figures/arch/gen_a7.py ({fn.__name__}); asset crops: make_a7_assets.py -> assets7/",
                   style=STYLE_NOTE[cid],
                   reference_image=REF_IMG,
                   references=[dict(figure=a, technique=b) for a, b in REFS7 + REFS6 + REFS4], items=f.prov),
              open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
