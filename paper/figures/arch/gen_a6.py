"""Fig. 1 v6 (A6-b): panel a (S1 architecture) redrawn in the visual language of a three-stage network figure
(stage containers with roman-numeral tabs, pastel 3D cuboid modules, slab tensors, colour-coded streams, dashed stream legend).
Panels b-d are unchanged from A5-b (gen_a4.layout_b5). Hand-coded SVG -> vector PDF + 600-dpi PNG via gen_fig.export.
Usage: python gen_a6.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
_NSP_STYLE_REF = _nsp_os.environ.get('NUCSEG3D_STYLE_REF', '/path/to/style_reference_images')
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fig import FS, LW, PAL, Fig, export, tint, M as PM  # noqa: E402
from gen_a3 import DEP, block, panel_cmp2  # noqa: E402
from gen_a4 import REFS as REFS4, hatch_defs, num, panel_infer4, panel_target3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK = PAL["fg"]
REF_IMG = f"{_NSP_STYLE_REF}/Weixin Image_2026-11-11_114115_289.png"
REFS6 = [
    ("user-supplied reference figure (" + REF_IMG + "), top: three-stage network figure; bottom: module figure",
     "stages grouped in pastel-tinted containers with dashed grey borders and a grey corner tab carrying a roman-numeral stage name; "
     "modules drawn as 3D cuboids (front face, lighter top, darker side) with the label inside; repeated blocks as stacked cuboids with an ellipsis; "
     "feature tensors as translucent slab glyphs (thin slices, dashed outline, ellipsis) labelled with their shapes; "
     "arrows colour-coded by stream with elbow routing; stream legend in a dashed box at the bottom; pastel palette "
     "(container tints derived from #D6E6F3 / #FFF0C1-#FACE9F / #D7F0D5; module tints kept on the paper's semantic colours: "
     "inherited 2D = palette 'base' #8491B4 family, new 3D = S1 #E64B35 family)"),
]

# ---------------------------------------------------------------- colours
BASE, RED, LOSS = PM["base"], PM["S1"], PAL["loss"]
C2D = dict(front=tint(BASE, 0.24), top=tint(BASE, 0.12), side=tint(BASE, 0.45), st=tint(BASE, 0.95))
C3D = dict(front=tint(RED, 0.20), top=tint(RED, 0.10), side=tint(RED, 0.38), st=tint(RED, 0.80))
S2D = "#56658F"            # per-plane 2D stream arrows (darkened palette 'base')
S3D = RED                  # 3D stream arrows
CONT = dict(I=("#EEF3FA", "I. Per-plane 2D encoder"), II=("#FFF5EC", "II. 3D residual decoder"),
            III=("#F0F7EE", "III. Output & training target"))
CBORD = "#A9A9AD"
TAB = "#DCDCDE"
PT = 0.3528


def cub(f, x, y, w, h, d, c, label=None, size=FS["small"], bold=True, hatch=False, dash=None, op=1.0, lc=None, sw=None):
    """Pastel cuboid: front (x, y, w, h), depth d along 35 deg; lighter top, darker side; label centred in the front face."""
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


def slab(f, x, y, w, h, d, n, c, dots=True):
    """Tensor glyph: dashed outline box holding n thin translucent slices with an ellipsis (reference style)."""
    dx, dy = d * DEP[0], d * DEP[1]
    out = [(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h), (x, y + h)]
    f.poly(out, "#ffffff", "#9a9aa0", 0.6 * LW, op=0.85, chk=False, dash="0.7 0.45")
    f.add(f'<path d="M{x:.2f},{y:.2f} H{x + w:.2f} L{x + w + dx:.2f},{y + dy:.2f} M{x + w:.2f},{y:.2f} V{y + h:.2f}" stroke="#9a9aa0" '
          f'stroke-width="{0.6 * LW:.3f}" stroke-dasharray="0.7 0.45" fill="none"/>')
    sl = (w - 0.8) / (n + (1.6 if dots else 0))
    k = 0
    for i in range(n + (1 if dots else 0)):
        if dots and i == n // 2:
            f.add(f'<text x="{x + 0.4 + (k + 0.8) * sl:.2f}" y="{y + h / 2 + 0.2:.2f}" font-size="{5 * PT:.3f}" font-weight="700" '
                  f'fill="{c["st"]}" text-anchor="middle">···</text>')
            k += 1.6
            continue
        sx = x + 0.4 + k * sl
        sd = d * 0.82
        cub(f, sx, y + 0.35, sl * 0.42, h - 0.7, sd, c, op=0.78, sw=0.45 * LW)
        k += 1


def container(f, x, y, w, h, key, sub=None):
    fill, name = CONT[key]
    f.rect(x, y, w, h, rx=1.6, fill=fill, stroke=CBORD, sw=0.75 * LW, dash="1.4 0.8")
    tw = len(name) * 1.05 + 3.0
    f.add(f'<path d="M{x + 0.25:.2f},{y + 4.3:.2f} V{y + 1.6:.2f} Q{x + 0.25:.2f},{y + 0.25:.2f} {x + 1.6:.2f},{y + 0.25:.2f} '
          f'H{x + tw:.2f} V{y + 4.3:.2f} Z" fill="{TAB}" stroke="none"/>')
    f.text(x + 1.6, y + 3.1, name, 5.6, 700, c="#3a3a3e")
    if sub:
        f.text(x + tw + 1.4, y + 3.1, sub, FS["small"], c=PAL["mute"], italic=True)


def inset(f, x, y, w, h, title, accent, right=None):
    f.rect(x, y, w, h, rx=1.3, fill="#ffffff", stroke=accent, sw=0.6 * LW, op=1.0)
    f.text(x + 1.6, y + 3.4, title, FS["body"], 700, c=accent if accent != C2D["st"] else "#4a5577")
    if right:
        f.text(x + w - 1.4, y + 3.4, right, FS["small"], c=PAL["mute"], a="end")


def harrow(f, x0, x1, y, col, **k):
    f.arrow(x0, y, x1, y, stroke=col, sw=1.4 * LW, **k)


def mini(f, x, y, w, h, c, label, bold=True):
    """Small flat module with a shallow depth, used inside the zoom insets."""
    cub(f, x, y, w, h, 1.1, c, label, bold=bold)
    return x + w + 1.1 * DEP[0]


def legend6(f, x, y):
    items = [(C2D, False, None, "2D, fine-tuned (lr 10^{−5})"), (C3D, False, None, "new (lr 2·10^{−4})"),
             (dict(front="#ffffff", top="#ffffff", side="#ffffff", st=RED), False, "0.7 0.45", "zero-initialised"),
             (C2D, True, None, "frozen")]
    cx = x
    for c, hatch, dash, lab in items:
        cub(f, cx, y - 2.4, 3.4, 2.9, 1.6, c, hatch=hatch, dash=dash)
        f.text(cx + 5.4, y, lab, FS["small"])
        cx += 5.4 + len(lab.replace("^{", "").replace("}", "")) * 0.93 + 3.4


def stream_legend(f, x, y, w):
    h = 4.6
    f.rect(x, y, w, h, rx=1.0, fill="none", stroke=CBORD, sw=0.6 * LW, dash="1.4 0.8")
    yy = y + h / 2
    items = [("arrow", S2D, None, "per-plane 2D stream"), ("arrow", S3D, None, "3D stream"),
             ("arrow", LOSS, "0.9 0.6", "loss against the 3D target"), ("plus", INK, None, "element-wise sum"),
             ("dots", INK, None, "repeated block")]
    cx = x + 3.0
    for kind, col, dash, lab in items:
        if kind == "arrow":
            f.arrow(cx, yy, cx + 7.0, yy, stroke=col, sw=1.4 * LW, dash=dash, chk=False)
            tx = cx + 8.4
        elif kind == "plus":
            f.plus_node(cx + 1.4, yy, r=1.2)
            tx = cx + 3.6
        else:
            f.text(cx + 1.8, yy + 0.5, "···", FS["body"], 700, a="middle")
            tx = cx + 4.4
        f.text(tx, yy + 0.65, lab, FS["small"])
        cx = tx + len(lab) * 0.93 + 7.0


# ======================================================================== panel a, v6
def panel_arch6(f, ox, oy, w=183.0, letter="a"):
    hatch_defs(f)
    f.plabel(ox, oy + 3.0, letter, "S1 architecture")
    legend6(f, ox + 46.0, oy + 3.2)
    cy0, ch = oy + 5.6, 71.0                     # containers
    xI, wI = ox + 0.5, 57.5
    xII, wII = xI + wI + 1.5, 69.0
    xIII = xII + wII + 1.5
    wIII = ox + w - 0.5 - xIII
    container(f, xI, cy0, wI, ch, "I", "every plane, shared weights")
    container(f, xII, cy0, wII, ch, "II", "new, 3D context")
    container(f, xIII, cy0, wIII, ch, "III")
    cy = oy + 33.5                               # main row centre (stage II / III)
    lab1, lab2 = oy + 42.4, oy + 44.7            # shape label rows under the main row

    # ---------------- stage I: input -> patch / pos -> ViT stack -> neck
    ix, iy_, isz, iL = xI + 2.0, oy + 16.6, 11.0, 5.6
    block(f, ix, iy_, isz, isz, iL, "#000", "#000", "#000", C2D["st"], front="../assets3/in_front.png", top="../assets3/in_top.png",
          side="../assets3/in_side.png")
    icx = ix + isz / 2 + iL * DEP[0] / 2
    f.text(icx, oy + 31.6, "input", FS["body"], 700, a="middle")
    f.text(icx, oy + 33.9, "1×Z×256×256", FS["small"], c=PAL["mute"], a="middle")
    f.text(icx, oy + 36.2, "Z = 32 planes", FS["small"], c=PAL["mute"], a="middle")
    bh, pitch, bd = 4.2, 4.95, 2.6
    b1 = oy + 17.4
    px = ix + isz + iL * DEP[0] + 2.4
    pw = 7.4
    harrow(f, ix + isz + iL * DEP[0] + 0.2, px - 0.25, b1 + bh / 2, S2D)
    cub(f, px, b1, pw, bh, 1.8, C2D, ["patch 8×8"] if False else "patch", hatch=True)
    f.text(px + pw / 2 + 0.6, b1 - 2.5, "8×8, frozen", FS["small"], c=PAL["mute"], a="middle")
    pyy = b1 + 7.6
    cub(f, px, pyy, pw, bh, 1.8, C2D, "+ pos.")
    f.text(px + pw / 2, pyy + bh + 2.4, "embedding", FS["small"], c=PAL["mute"], a="middle")
    f.text(px + pw / 2, pyy + bh + 4.7, "per plane", FS["small"], c=PAL["mute"], a="middle")
    f.text(px + pw / 2, pyy + bh + 7.0, "32×32×1024", FS["small"], c=PAL["mute"], a="middle")
    sx = px + pw + 6.2                            # ViT stack left edge
    qx = px + pw + 1.8 * DEP[0] + 1.9           # sum node between patch and stack
    f.plus_node(qx, b1 + bh / 2, r=1.0, col=S2D)
    harrow(f, px + pw + 1.8 * DEP[0] + 0.2, qx - 1.15, b1 + bh / 2, S2D)
    harrow(f, qx + 1.05, sx - 0.25, b1 + bh / 2, S2D)
    f.path(f"M{px + pw + 1.8 * DEP[0] + 0.2:.2f},{pyy + bh / 2:.2f} H{qx:.2f} V{b1 + bh / 2 + 1.25:.2f}", stroke=S2D, sw=1.1 * LW)
    sw_ = 19.0
    ys = [b1, b1 + pitch, None, b1 + 2 * pitch + 3.4, b1 + 3 * pitch + 3.4]
    names = ["block 1", "block 2", None, "block 24", "neck"]
    for yy, nm in zip(ys, names):
        if yy is None:
            continue
        cc = C2D if nm != "neck" else dict(front=tint(BASE, 0.40), top=tint(BASE, 0.22), side=tint(BASE, 0.60), st=C2D["st"])
        cub(f, sx, yy, sw_, bh, bd, cc, nm if nm != "neck" else "neck → 256")
    f.text(sx + sw_ / 2, b1 + 2 * pitch + 1.75, "···", FS["mod"], 700, c=C2D["st"], a="middle")
    scx = sx + sw_ / 2 + bd * DEP[0] / 2
    f.text(scx, oy + 11.5, "ViT-L · 24 blocks · 304 M", FS["small"], 700, a="middle")
    f.text(scx, oy + 14.8, "global attention · lr 10^{−5}", FS["small"], c=PAL["mute"], a="middle")
    num(f, sx - 2.4, b1 + 2 * pitch + 1.6, 1)
    f.prov_add("patch embedding 8x8 conv, stride 8 (SAM 16x16 kernel subsampled), weights not trained; absolute position embedding (1,32,32,1024), trained",
               "cellpose vit.py:109-116 (CPSAM); net3d.py:202-205 patch-embed requires_grad False, :228-232 _embed")
    f.prov_add("ViT-L encoder: 24 blocks, width 1024, 16 heads, MLP ratio 4, relative position bias, global attention in every block; applied to all B*Z planes with shared weights; 304.4 M encoder-side parameters, lr 1e-5",
               "segment_anything build_sam.py:27-34, 66-80; cellpose vit.py:118-120; net3d.py:285-286; train3d.py:144, 211-214; train log")
    f.prov_add("neck: 1x1 conv 1024->256, LayerNorm2d, 3x3 conv 256->256, LayerNorm2d", "segment_anything image_encoder.py:88-104")

    # ---------------- stage II: stack along z -> volume -> 1^3 -> 4 residual blocks
    ny = ys[4] + bh / 2
    s0 = xII + 3.0
    f.path(f"M{sx + sw_ + bd * DEP[0] + 0.2:.2f},{ny:.2f} H{xII - 0.75:.2f} V{cy:.2f} H{s0 - 0.3:.2f}", stroke=S2D, sw=1.4 * LW)
    sh_, sd_ = 11.0, 5.0
    slab(f, s0, cy - sh_ / 2, 8.6, sh_, sd_, 5, C2D)
    f.text(s0 + 4.3 + sd_ * DEP[0] / 2, cy - sh_ / 2 - 4.4, "stack along z", FS["small"], c=PAL["mute"], a="middle")
    f.text(s0 + 4.3, lab1, "Z×256×32×32", FS["small"], c=PAL["mute"], a="middle")
    vx = s0 + 8.6 + sd_ * DEP[0] + 2.6
    harrow(f, vx - 2.4, vx - 0.25, cy, S2D)
    vw, vh, vL = 9.5, 12.0, 6.0
    block(f, vx, cy - vh / 2, vw, vh, vL, C2D["front"], C2D["top"], C2D["side"], C2D["st"], top="../assets3/feat_enc_top.png", grid=5)
    f.text(vx + vw / 2 + 1.0, lab1, "256×Z×32×32", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("plane features re-stacked to (B,256,Z,32,32)", "net3d.py:288")
    vend = vx + vw + vL * DEP[0]
    c1 = vend + 2.6
    harrow(f, vend + 0.2, c1 - 0.25, cy, S3D)
    cub(f, c1, cy - 5.0, 2.2, 10.0, 2.4, C3D)
    f.text(c1 + 1.1, lab1, "1^{3}", FS["small"], c=PAL["new_d"], a="middle")
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
        f.text(rx + rbw / 2, lab1, f"d={dd}", FS["small"], 700, c=PAL["new_d"], a="middle")
        f.text(rx + rbw / 2, lab2, f"{rf[j + 1]}", FS["small"], c=PAL["mute"], a="middle")
    f.text(r0 - 1.6, lab2, "RF_{z}", FS["small"], c=PAL["mute"], a="end")
    dend = r0 + 3 * rstep + rbw + rbL * DEP[0]
    f.text((c1 + dend) / 2, oy + 21.4, "3D residual decoder · 14.3 M · lr 2·10^{−4}", FS["small"], 700, c=PAL["new_d"], a="middle")
    f.text((r0 + dend) / 2, oy + 23.7, "4 residual blocks, z-dilation d", FS["small"], c=PAL["mute"], a="middle")
    num(f, c1 - 1.0, oy + 21.0, 2)
    f.prov_add("decoder: 1x1x1 Conv3d 256->256, four ResBlock3D with z-dilation 1,2,4,8 (shape kept 256xZx32x32), 14.3 M params, lr 2e-4; z receptive field after blocks 1-4 = 5, 11, 21, 39 planes (derived)",
               "net3d.py:69-96; model3d.json dec_*; train3d.py:145 lr_dec 2e-4; train log '新增模块 14.3M'")

    # ---------------- stage III: zero-init 1^3 -> Delta (+) 2D head -> x8 -> output; loss
    zx = xIII + 2.2
    harrow(f, dend + 0.2, zx - 0.25, cy, S3D)
    zc = dict(front="#ffffff", top="#ffffff", side=tint(RED, 0.06), st=RED)
    cub(f, zx, cy - 5.0, 2.6, 10.0, 2.4, zc, dash="0.7 0.45")
    f.text(zx + 1.3, cy + 0.9, "0", FS["small"], 700, c=RED, a="middle")
    f.text(zx + 1.3, lab1, "1^{3}", FS["small"], c=PAL["new_d"], a="middle")
    f.text(zx + 2.2, cy - 8.4, "4·64", FS["small"], c=PAL["mute"], a="middle")
    num(f, zx + 1.3, lab2 + 0.2, 3)
    qx3 = zx + 2.6 + 2.4 * DEP[0] + 5.0
    f.plus_node(qx3, cy, r=1.6)
    harrow(f, zx + 2.6 + 2.4 * DEP[0] + 0.2, qx3 - 1.85, cy, S3D)
    f.text(qx3 - 2.9, cy - 1.1, "Δ", FS["body"], 700, c=RED, a="middle")
    # 2D head (per plane) on top of stage III, fed from the stacked features
    hx, hy, hw, hh = xIII + 2.2, oy + 11.4, 26.0, 4.4
    cub(f, hx, hy, hw, hh, 1.8, C2D, "2D head: 1×1 conv → 3·64")
    vcx, vtop = vx + vw / 2 + vL * DEP[0] / 2, cy - vh / 2 + vL * DEP[1] / 2
    f.path(f"M{vcx:.2f},{vtop - 0.1:.2f} V{hy + hh / 2:.2f} H{hx - 0.25:.2f}", stroke=S2D, sw=1.4 * LW)
    f.text((vcx + xII + wII) / 2, hy + hh / 2 - 1.0, "same features, per plane", FS["small"], c=PAL["mute"], a="middle", italic=True)
    f.path(f"M{qx3:.2f},{hy + hh + 0.2:.2f} V{cy - 1.85:.2f}", stroke=S2D, sw=1.4 * LW)
    f.text(qx3 + 1.0, cy - 11.4, "d_{y}, d_{x}, p", FS["small"], c=PAL["mute"])
    f.prov_add("2D head = Cellpose-SAM 1x1 Conv2d(256 -> 3*64) per plane; Delta added for (dy, dx, p); dz = Delta_z; x8 pixel shuffle back to 256x256",
               "net3d.py:287-295; cellpose vit.py:122")
    # output volume
    ox_, ow, oL = qx3 + 4.6, 10.0, 6.0
    f.arrow(qx3 + 1.8, cy, ox_ - 0.25, cy, sw=1.4 * LW)
    f.text((qx3 + ox_) / 2 + 0.2, cy + 3.4, "×8", FS["small"], c=PAL["mute"], a="middle")
    block(f, ox_, cy - 6.0, ow, 12.0, oL, "#ffffff", "#ffffff", tint(BASE, 0.10), INK, top="../assets3/out_top.png", sw=0.8 * LW)
    f.image(ox_ + 0.3, cy - 5.6, ow - 0.6, 11.2, "flow_glyphs_pal.png")
    f.text(ox_ + ow / 2 + 1.5, lab1, "(d_{z}, d_{y}, d_{x}, p)", FS["small"], 700, a="middle")
    f.text(ox_ + ow / 2 + 1.5, lab2 + 1.0, "4×Z×256×256", FS["small"], c=PAL["mute"], a="middle")
    num(f, ox_ - 1.9, cy - 4.4, 4)
    # loss box
    lx = ox_ + ow + oL * DEP[0] + 2.0
    lw_ = xIII + wIII - 1.5 - lx
    ly, lh = cy - 6.8, 13.6
    f.rect(lx, ly, lw_, lh, rx=1.0, fill=tint(LOSS, 0.10), stroke=LOSS, sw=0.8 * LW)
    f.text(lx + lw_ / 2, ly + 3.2, "loss", FS["small"], 700, c=LOSS, a="middle")
    f.text(lx + lw_ / 2, ly + 5.7, ["½ MSE(flows, 5*u*)", "+ BCE(p, fg)"], FS["small"], 700, c=LOSS, a="middle", ls=1.3)
    f.text(lx + lw_ / 2, ly + 10.4, ["vs 3D target,", "in the volume"], FS["small"], c=PAL["mute"], a="middle", ls=1.3)
    f.arrow(ox_ + ow + oL * DEP[0] + 0.2, cy, lx - 0.25, cy, stroke=LOSS, sw=1.4 * LW, dash="0.9 0.6")
    f.prov_add("loss: MSE(net flow, 5 x unit target)/2 + BCE(cellprob, nucleus) over voxels inside the volume; AdamW, bf16 autocast, grad clip 1.0",
               "train3d.py:33-43, 215, 275-280")

    # ================= zoom insets inside each stage
    iy, ih = oy + 47.2, 28.4
    # ---- I: transformer block
    ex, ew = xI + 1.2, wI - 2.4
    inset(f, ex, iy, ew, ih, "transformer block (×24)", C2D["st"], "tokens 32×32×1024")
    seq = [("LN", 4.2), ("MHSA", 9.0), ("+", 0), ("LN", 4.2), ("MLP", 9.0), ("+", 0)]
    bx, by_, bh_ = ex + 4.2, iy + 9.0, 5.0
    xs_ = []
    for i, (nm, bw_) in enumerate(seq):
        if nm == "+":
            f.plus_node(bx + 1.3, by_ + bh_ / 2, r=1.2, col=S2D)
            xs_.append(bx + 1.3)
            bx += 2.6
        else:
            mini(f, bx, by_, bw_, bh_, C2D, nm)
            xs_.append(bx + bw_ / 2)
            bx += bw_ + 1.1 * DEP[0]
        if i < 5:
            harrow(f, bx + 0.15, bx + 1.85, by_ + bh_ / 2, S2D)
            bx += 2.0
    for a_, b_ in ((ex + 2.4, xs_[2]), (xs_[2] + 1.8, xs_[5])):
        f.path(f"M{a_:.2f},{by_ + bh_ / 2:.2f} V{by_ - 2.2:.2f} H{b_:.2f} V{by_ + bh_ / 2 - 1.4:.2f}", stroke=S2D, sw=0.9 * LW)
    f.path(f"M{ex + 1.0:.2f},{by_ + bh_ / 2:.2f} H{ex + 4.0:.2f}", stroke=S2D, sw=0.9 * LW, arrow=False)
    f.text(xs_[1], by_ + bh_ + 2.6, ["16 heads, global,", "rel. position"], FS["small"], c=PAL["mute"], a="middle")
    f.text(xs_[4], by_ + bh_ + 2.6, ["1024→4096", "→1024, GELU"], FS["small"], c=PAL["mute"], a="middle")
    f.text(ex + 1.6, iy + ih - 4.5, "neck: 1×1 conv 1024→256 · LN · 3×3 conv · LN → 32×32×256", FS["small"])
    f.text(ex + 1.6, iy + ih - 1.9, "patch embed frozen; all other encoder weights fine-tuned", FS["small"], c=PAL["mute"])
    f.prov_add("transformer block: x + MHSA(LN(x)); x + MLP(LN(x)); MHSA 16 heads with relative position bias, global; MLP 1024->4096->1024 GELU",
               "segment_anything image_encoder.py Block (norm1, attn, norm2, mlp, shortcut) l.151-175; build_sam.py:27-34, 66-80")
    # dashed zoom guides: stack -> inset
    for gx0, gx1 in ((sx, ex + 6.0), (sx + sw_, ex + ew - 6.0)):
        f.path(f"M{gx0:.2f},{ys[4] + bh + 0.3:.2f} L{gx1:.2f},{iy - 0.2:.2f}", stroke=C2D["st"], arrow=False, dash="0.7 0.5", chk=False, sw=0.6 * LW)

    # ---- II: residual block + dilated taps
    rxi, rwi = xII + 1.2, wII - 2.4
    inset(f, rxi, iy, rwi, ih, "residual block (×4)", PAL["new_d"], "256×Z×32×32 in and out")
    seq2 = [("GN", 4.0), ("GELU", 5.8), ("conv 3^{3}", 9.4), ("GN", 4.0), ("GELU", 5.8), ("conv 3^{3}", 9.4)]
    bx, by_ = rxi + 4.2, iy + 9.0
    first = bx
    for i, (nm, bw_) in enumerate(seq2):
        conv = nm.startswith("conv")
        mini(f, bx, by_, bw_, bh_, C3D if conv else dict(front="#ffffff", top=tint(RED, 0.05), side=tint(RED, 0.16), st=C3D["st"]), nm, bold=conv)
        if i == 2:
            f.text(bx + bw_ / 2, by_ + bh_ + 2.6, "dilation (d,1,1)", FS["small"], 700, c=PAL["new_d"], a="middle")
        bx += bw_ + 1.1 * DEP[0]
        if i < 5:
            harrow(f, bx + 0.15, bx + 1.55, by_ + bh_ / 2, S3D)
            bx += 1.7
    pxn = bx + 2.6
    f.plus_node(pxn, by_ + bh_ / 2, r=1.2, col=S3D)
    harrow(f, bx + 0.15, pxn - 1.35, by_ + bh_ / 2, S3D)
    f.path(f"M{first - 1.6:.2f},{by_ + bh_ / 2:.2f} V{by_ - 2.2:.2f} H{pxn:.2f} V{by_ + bh_ / 2 - 1.4:.2f}", stroke=S3D, sw=0.9 * LW)
    f.path(f"M{rxi + 1.0:.2f},{by_ + bh_ / 2:.2f} H{first - 0.2:.2f}", stroke=S3D, sw=0.9 * LW, arrow=False)
    nz, pz = 39, 0.215
    top = iy + 18.4
    for j, dd in enumerate((1, 2, 4, 8)):
        cxj = rxi + 3.0 + j * 10.2
        cw, cdp = 4.4, 1.8
        ddx, ddy = cdp * DEP[0], cdp * DEP[1]
        for q in range(nz):
            yy = top + q * pz
            off = q - nz // 2
            tap = off in (-dd, 0, dd)
            col = RED if tap else tint(BASE, 0.22)
            f.add(f'<polygon points="{cxj:.2f},{yy:.2f} {cxj + cw:.2f},{yy:.2f} {cxj + cw + ddx:.2f},{yy + ddy:.2f} {cxj + ddx:.2f},{yy + ddy:.2f}" '
                  f'fill="{col}" stroke="none"/>')
        r = rf[j + 1]
        q0, q1 = nz // 2 - r // 2, nz // 2 + r // 2
        y0b, y1b = top + q0 * pz - 0.4, top + q1 * pz + 0.2
        bxb = cxj + cw + ddx + 0.8
        f.add(f'<path d="M{bxb:.2f},{y0b:.2f} h0.6 V{y1b:.2f} h-0.6" stroke="{PAL["new_d"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
        f.text(bxb + 1.2, (y0b + y1b) / 2 + 0.8, f"{r}", FS["body"], 700, c=PAL["new_d"])
    capx = rxi + 3.0 + 3 * 10.2 + 4.4 + 1.8 * DEP[0] + 6.4
    f.text(capx, iy + 19.6, ["red: z taps of the", "dilated conv; bracket:", "z receptive field", "(planes) after block"], FS["small"], c=PAL["mute"])
    f.prov_add("ResBlock3D: x + conv3(GELU(GN(conv3_dil(GELU(GN(x)))))), GroupNorm 16 groups; first conv dilation (d,1,1)", "net3d.py:69-78")

    # ---- III: output composition
    zxi, zwi = xIII + 1.2, wIII - 2.4
    inset(f, zxi, iy, zwi, ih, "output composition", PAL["new_d"])
    cA, cB = zxi + 13.0, zxi + zwi - 10.0
    f.text(cA, iy + 7.4, "step 0", FS["small"], 700, a="middle")
    f.text(cB, iy + 7.4, "trained", FS["small"], 700, c=PAL["new_d"], a="middle")
    rowsZ = [("d_{z}", "0", "Δ_{z}"), ("d_{y}", "2D", "2D+Δ"), ("d_{x}", "2D", "2D+Δ"), ("p", "2D", "2D+Δ")]
    for k_, (chn, a_, b_) in enumerate(rowsZ):
        yy = iy + 8.9 + k_ * 4.1
        f.text(zxi + 3.0, yy + 2.5, chn, FS["small"])
        zero = a_ == "0"
        cub(f, cA - 4.6, yy, 9.2, 3.5, 0.9, zc if zero else C2D, a_, bold=zero, dash="0.7 0.45" if zero else None, lc=RED if zero else INK)
        cub(f, cB - 5.4, yy, 10.8, 3.5, 0.9, C3D, b_, bold=False)
    harrow(f, cA + 5.6, cB - 6.0, iy + 8.9 + 4.1 + 1.75, INK)
    f.text(zxi + 1.6, iy + ih - 1.9, "step 0 = 2D control, d_{z} = 0", FS["small"], c=PAL["mute"])
    f.prov_add("output composition (dz, dy, dx, p) = (Delta_z, 2D head + Delta); Delta = 0 at step 0, so S1 equals the seed-matched 2D model (2D control) with dz = 0",
               "net3d.py:89-90, 294; model3d.json init orx_v5_2d_s0; net3d.py:359-371 self-test")

    # ================= rationale + stream legend
    ry = cy0 + ch + 3.2
    notes = ["per-plane encoding keeps the 2D foundation model's representations, which native 3D models do not inherit",
             "axial resolution is several-fold poorer than lateral: dilation widens z context to 39 planes without extra weights",
             "zero-initialised output: the network starts as the 2D model and learns only the volumetric correction",
             "the axial flow comes from 3D context, not from summing flows predicted on separate orthogonal slices"]
    for k_, t_ in enumerate(notes):
        colx = ox + 1.0 + (k_ % 2) * 91.5
        rowy = ry + (k_ // 2) * 3.5
        num(f, colx + 1.6, rowy - 0.7, k_ + 1)
        f.text(colx + 4.0, rowy, t_, FS["small"])
    f.prov_add("rationale 1-4 wording", "main.tex Introduction l.44-50 (as in gen_a4 panel_arch4)")
    stream_legend(f, ox + 0.5, ry + 4.9, w - 1.0)


def layout_b6():
    """A5-b with panel a redrawn (panel_arch6); panels b-d identical to layout_b5."""
    f = Fig(183.0, 170.0)
    panel_arch6(f, 0.0, 0.0, letter="a")
    panel_cmp2(f, 0.0, 91.5, 119.0, letter="b")
    panel_target3d(f, 122.0, 91.5, 61.0, letter="c", tile3="target_F1d.png")
    panel_infer4(f, 0.0, 129.0, 183.0, letter="d")
    return f


def main():
    cid = "A6-b"
    f = layout_b6()
    res = export(f, cid)
    print(cid, "issues:", len(res.get("issues", [])), "min font", res.get("min_font_pt"))
    for s_ in res.get("issues", [])[:80]:
        print("   ", s_)
    json.dump(dict(figure=cid, code_snapshot=_nsp_os.path.join(_NSP_REPO, 'src'),
                   model="orx_t3d_r5s1_s0/model3d.json", palette="paper/figures/palette.json",
                   generator="paper/figures/arch/gen_a6.py (panel_arch6, layout_b6); panels b-d from gen_a4.py",
                   reference_image=REF_IMG,
                   references=[dict(figure=a, technique=b) for a, b in REFS6 + REFS4], items=f.prov),
              open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
