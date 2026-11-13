"""Fig. 1 v4 (A4-a / A4-b): detailed architecture of S1 with design rationale, 3D training target and data-backed inference.
Hand-coded SVG -> vector PDF + 600-dpi PNG (gen_fig.export: overlap and font-size check in Chrome).
Reference figures consulted (see A4-*.provenance.json "references"): MA-SAM Fig. 1, SAM-Med3D Fig. 3, UNETR Fig. 2, Swin-UNETR
Figs. 1-2, 3DSAM-adapter Fig. 1, SAM 2 Fig. 3, nnFormer Fig. 1, Cellpose-SAM Fig. 1, MedSAM2 Fig. 1b, Cellpose3 Fig. 1e.
Usage: python gen_a4.py [a|b|all]
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
from gen_a3 import DEP, LOG3, LOGA, block, panel_cmp2, tex  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK = PAL["fg"]
RD = json.load(open(os.path.join(HERE, "assets3", "ruleT_diam.json")))
DF = json.load(open(os.path.join(HERE, "assets3", "diffusion.json")))
REFS = [
    ("MA-SAM (arXiv 2309.08842) Fig. 1", "input cube -> staggered slice stack with reshape [B,N,H,W]->[BN,H,W]; trapezoid zoom from one block to a transformer-block strip with residual elbows; frozen vs trained by fill + icon"),
    ("SAM-Med3D (arXiv 2310.15161) Fig. 3", "dashed funnel lines from the attention block to a tinted zoom panel; module families with fixed colours"),
    ("UNETR (arXiv 2103.10504) Fig. 2", "transformer block as vertical stack Norm/MHA/Norm/MLP with '× 12'; shapes written under connections; colour-coded small 3D cubes for convolution types"),
    ("Swin-UNETR (arXiv 2201.01266) Figs. 1-2", "feature maps as isometric cuboids with shape labels; exploded 3D grid cube for a 3D receptive field"),
    ("3DSAM-adapter (arXiv 2306.13465) Fig. 1", "new small modules boxed separately with their own kernel shapes"),
    ("SAM 2 (arXiv 2408.00714) Fig. 3", "translucent ghost copies behind a module + oblique axis for repetition along a dimension"),
    ("nnFormer (arXiv 2109.03201) Fig. 1", "short design-motivation notes attached to modules"),
    ("Cellpose-SAM (bioRxiv 2025.04.28.651001) Fig. 1", "key encoder parameters written inside the block (8x8 patches, 256x256, global attention everywhere)"),
    ("MedSAM2 (arXiv 2504.03600) Fig. 1b", "slice index along z for per-slice processing"),
    ("Cellpose3 (Nat. Methods 2025) Fig. 1e", "loss terms as coloured text boxes linked to the data they compare"),
]


def hatch_defs(f):
    f.defs.append(f'<pattern id="hatch" width="1.2" height="1.2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                  f'<rect width="1.2" height="1.2" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="1.2" stroke="{PAL["inh"]}" stroke-width="0.35"/></pattern>')


def num(f, x, y, k):
    """Numbered rationale marker."""
    f.circle(x, y, 1.55, fill=INK, stroke="#ffffff", sw=0.5 * LW, chk=False)
    f.add(f'<text x="{x:.2f}" y="{y + 0.62:.2f}" font-size="{5.0 * 0.3528:.3f}" font-weight="700" fill="#ffffff" text-anchor="middle">{k}</text>')


def legend(f, x, y):
    items = [(PAL["inh_f"], PAL["inh"], None, "2D, fine-tuned (lr 10^{−5})"), (PAL["new_f"], PAL["new"], None, "new (lr 2·10^{−4})"),
             ("#ffffff", PAL["new"], "0.8 0.5", "zero-initialised"), ("url(#hatch)", PAL["inh"], None, "frozen")]
    cx = x
    for fill, st, dash, lab in items:
        f.rect(cx, y - 2.5, 3.8, 3.0, rx=0.6, fill=fill, stroke=st, sw=0.9 * LW, dash=dash)
        f.text(cx + 4.8, y, lab, FS["small"])
        cx += 4.8 + len(lab.replace("^{", "").replace("}", "")) * 0.98 + 3.2


# ======================================================================== b: 3D training target
def panel_target3d(f, ox, oy, w, letter="b", tile3=None):
    f.plabel(ox, oy + 3.0, letter, "3D training target")
    s = 17.0
    gap = (w - 3 * s) / 2
    ys = oy + 6.6
    names = [("diff_source.png", "heat source at", "the label centre"),
             ("diff_heat.png", f"diffuse inside the", f"nucleus ({DF['niter']} steps)"),
             ("diff_flow.png", "flow u = ∇T/|∇T|", "(colour u_{z})")]
    if tile3:
        names[2] = (tile3, "target flow u, 4 nuclei", "(colour u_{z})")
    for i, (nm, l1, l2) in enumerate(names):
        x = ox + i * (s + gap)
        f.image(x, ys, s, s, nm if nm == tile3 else "../assets3/" + nm)
        f.text(x + s / 2, ys + s + 2.4, l1, FS["small"], c=PAL["mute"], a="middle")
        f.text(x + s / 2, ys + s + 4.7, l2, FS["small"], c=PAL["mute"], a="middle")
        if i < 2:
            f.arrow(x + s + 0.4, ys + s / 2, x + s + gap - 0.4, ys + s / 2)
    f.text(ox + w - 0.5, oy + 3.0, "GT nucleus, training grid" if not tile3 else "GT nuclei, mouse embryo", FS["small"], c=PAL["mute"], a="end")
    if tile3:
        f.prov_add("tile 3: nc-viz F1-d_target_light, GT-derived 3D target (flows3d_gpu of S1 data3d.py) on E15_Ab2 GT 33, 51, 68, 50 (rule F1_group), one sink per nucleus; illustrative, re-interpolated along z for display",
                   "paper/figures/assets_viz/F1-d.provenance.json")
    f.prov_add(f"3D Cellpose target: heat source at the label's most central voxel, diffusion restricted to the nucleus for niter = 6 x sum of bounding-box extents ({DF['niter']} here), unit gradient = flow; GT nucleus 62 of E15_Ab2 resampled to the training grid",
               "data3d.py:28-81 flows3d_gpu (centre l.51-57, niter l.58-62, update l.71-74, gradient l.75-77); diffusion_data.py, render_diffusion.py")


# ======================================================================== c: detailed architecture
def panel_arch4(f, ox, oy, w=183.0, letter="c"):
    hatch_defs(f)
    f.plabel(ox, oy + 3.0, letter, "S1 architecture")
    legend(f, ox + 46.0, oy + 3.0)
    fy, fh, L = oy + 16.0, 15.0, 9.0
    cy = fy + fh / 2
    # ---- input window
    x = ox + 1.0
    block(f, x, fy, 15.0, fh, L, "#000", "#000", "#000", PAL["inh"], front="../assets3/in_front.png", top="../assets3/in_top.png",
          side="../assets3/in_side.png")
    f.text(x + 7.5, fy + fh + 3.4, "input", FS["body"], 700, a="middle")
    f.text(x + 7.5, fy + fh + 5.8, "1×Z×256×256", FS["small"], c=PAL["mute"], a="middle")
    f.text(x + 7.5, fy + fh + 8.2, "Z = 32 planes", FS["small"], c=PAL["mute"], a="middle")
    # ---- patch embedding (frozen) + position embedding (fine-tuned)
    px = x + 15.0 + L * DEP[0] + 3.0
    f.arrow(px - 2.6, cy, px - 0.3, cy)
    f.rect(px, fy - 1.0, 9.0, 7.6, rx=0.8, fill="url(#hatch)", stroke=PAL["inh"], sw=0.9 * LW)
    f.text(px + 4.5, fy + 2.3, "patch", FS["small"], 700, a="middle")
    f.text(px + 4.5, fy + 4.8, "8×8", FS["small"], 700, a="middle")
    f.rect(px, fy + 8.4, 9.0, 7.6, rx=0.8, fill=PAL["inh_f"], stroke=PAL["inh"], sw=0.9 * LW)
    f.text(px + 4.5, fy + 11.6, "+ pos.", FS["small"], 700, a="middle")
    f.text(px + 4.5, fy + 14.1, "embed", FS["small"], 700, a="middle")
    f.text(px + 4.5, fy - 4.6, "per plane", FS["small"], c=PAL["mute"], a="middle")
    f.text(px + 4.5, fy - 2.3, "32×32×1024", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("patch embedding 8x8 conv, stride 8 (SAM 16x16 kernel subsampled), weights not trained; absolute position embedding (1,32,32,1024), trained",
               "cellpose vit.py:109-116 (CPSAM); net3d.py:202-205 patch-embed requires_grad False, :228-232 _embed")
    # ---- ViT-L: 24 blocks, ghosts along z
    vx, vw, vh = px + 12.0, 27.0, 19.0
    f.arrow(px + 9.3, cy, vx - 0.3, cy)
    for k in (3, 2, 1):
        f.rect(vx + k * 1.4, cy - vh / 2 - k * 1.0, vw, vh, rx=1.4, fill=PAL["inh_f"], stroke=PAL["inh"], sw=0.7 * LW, op=0.35, chk=False)
    f.rect(vx, cy - vh / 2, vw, vh, rx=1.4, fill=PAL["inh_f"], stroke=PAL["inh"], sw=1.1 * LW)
    for b in range(24):
        bx = vx + 1.4 + b * (vw - 2.8) / 24
        hl = b == 11
        f.add(f'<rect x="{bx:.2f}" y="{cy - vh / 2 + 1.6:.2f}" width="{(vw - 2.8) / 24 - 0.25:.2f}" height="{vh - 8.4:.2f}" '
              f'fill="{PAL["inh"] if not hl else INK}" opacity="{0.55 if not hl else 0.9}"/>')
    f.text(vx + vw / 2, cy + vh / 2 - 4.1, "ViT-L: 24 blocks", FS["small"], 700, a="middle")
    f.text(vx + vw / 2, cy + vh / 2 - 1.5, "global attention", FS["small"], a="middle")
    f.text(vx + vw / 2 + 2.0, fy - 5.6, "each plane separately,", FS["small"], c=PAL["mute"], a="middle")
    f.text(vx + vw / 2 + 2.0, fy - 3.3, "shared weights · 304 M", FS["small"], c=PAL["mute"], a="middle")
    num(f, vx - 1.2, cy + vh / 2 + 1.4, 1)
    f.prov_add("ViT-L encoder: 24 blocks, width 1024, 16 heads, MLP ratio 4, relative position bias, global attention in every block (Cellpose-SAM sets window_size 0); applied to all B*Z planes with shared weights; 304.4 M trainable encoder-side parameters, lr 1e-5",
               "segment_anything build_sam.py:27-34, 66-80; cellpose vit.py:118-120; net3d.py:285-286; train3d.py:144, 211-214; train log")
    # ---- neck
    nx = vx + vw + 4.8 + 2.4
    f.arrow(vx + vw + 4.6, cy, nx - 0.3, cy)
    f.rect(nx, cy - 6.0, 7.0, 12.0, rx=1.0, fill=PAL["inh_f"], stroke=PAL["inh"], sw=1.0 * LW)
    f.text(nx + 3.5, cy - 0.4, "neck", FS["small"], 700, a="middle")
    f.text(nx + 3.5, cy + 2.1, "→256", FS["small"], a="middle")
    f.prov_add("neck: 1x1 conv 1024->256, LayerNorm2d, 3x3 conv 256->256, LayerNorm2d", "segment_anything image_encoder.py:88-104")
    # ---- per-plane features -> restacked volume
    sx = nx + 9.4
    f.arrow(nx + 7.3, cy, sx - 0.3, cy)
    for k in range(5):
        f.poly([(sx + k * 1.15, cy - 4.0 - k * 0.8), (sx + 1.8 + k * 1.15, cy - 5.2 - k * 0.8), (sx + 1.8 + k * 1.15, cy + 3.6 - k * 0.8),
                (sx + k * 1.15, cy + 4.8 - k * 0.8)], fill=PAL["inh_f2"], stroke=PAL["inh"], sw=0.5 * LW, chk=False)
    f.text(sx + 3.6, fy + fh + 3.4, "Z × 256", FS["small"], c=PAL["mute"], a="middle")
    f.text(sx + 3.6, fy + fh + 5.8, "×32×32", FS["small"], c=PAL["mute"], a="middle")
    fx = sx + 10.4
    f.arrow(sx + 7.0, cy, fx - 0.3, cy)
    f.text(sx + 2.0, fy - 3.6, "stack", FS["small"], c=PAL["mute"], a="middle")
    f.text(sx + 2.0, fy - 1.3, "along z", FS["small"], c=PAL["mute"], a="middle")
    fw = 11.0
    block(f, fx, fy, fw, fh, L, PAL["inh_f"], PAL["inh_f2"], PAL["inh_f2"], PAL["inh"], top="../assets3/feat_enc_top.png", grid=6)
    f.text(fx + fw / 2 + 2.0, fy + fh + 3.4, "256×Z×32×32", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("plane features re-stacked to (B,256,Z,32,32)", "net3d.py:288")
    # ---- decoder: 1x1x1 in, 4 residual blocks
    dx0 = fx + fw + L * DEP[0] + 3.0
    f.arrow(fx + fw + L * DEP[0] + 0.4, cy, dx0 - 0.3, cy)
    f.rect(dx0, fy + 1.0, 2.2, fh - 2.0, rx=0.4, fill=PAL["new_f"], stroke=PAL["new"], sw=0.9 * LW)
    f.text(dx0 + 1.1, fy + fh + 3.4, "1^{3}", FS["small"], c=PAL["new_d"], a="middle")
    rb_w, rb_L, step_ = 3.6, 7.0, 7.6
    rf = [1]
    for dd in (1, 2, 4, 8):
        rf.append(rf[-1] + 2 * dd + 2)
    r0 = dx0 + 3.6
    for j, dd in enumerate((1, 2, 4, 8)):
        rx = r0 + j * step_
        block(f, rx, fy, rb_w, fh, rb_L, PAL["new_f"], tint(PAL["new"], 0.32), tint(PAL["new"], 0.32), PAL["new"], sw=0.9 * LW,
              top="../assets3/feat_dec_top.png" if j == 3 else None)
        f.text(rx + rb_w / 2, fy + fh + 3.4, f"d={dd}", FS["small"], 700, c=PAL["new_d"], a="middle")
        f.text(rx + rb_w / 2, fy + fh + 5.8, f"{rf[j + 1]}", FS["small"], c=PAL["mute"], a="middle")
    f.text(r0 - 2.0, fy + fh + 5.8, "RF_{z}", FS["small"], c=PAL["mute"], a="end")
    dend = r0 + 3 * step_ + rb_w + rb_L * DEP[0]
    f.text((dx0 + dend) / 2 + 2.0, fy - 5.4, "3D residual decoder · 14.3 M", FS["small"], 700, c=PAL["new_d"], a="middle")
    num(f, dx0 + 1.1, fy - 2.4, 2)
    f.prov_add("decoder: 1x1x1 Conv3d 256->256, four ResBlock3D with z-dilation 1,2,4,8 (shape kept 256xZx32x32), 14.3 M params, lr 2e-4; z receptive field after blocks 1-4 = 5, 11, 21, 39 planes (derived)",
               "net3d.py:69-96; model3d.json dec_*; train3d.py:145 lr_dec 2e-4; train log '新增模块 14.3M'")
    # ---- zero-init output + pixel shuffle
    zx = dend + 2.6
    f.arrow(dend + 0.3, cy, zx - 0.3, cy)
    f.rect(zx, fy + 1.0, 2.6, fh - 2.0, rx=0.5, fill="#ffffff", stroke=PAL["new"], sw=1.0 * LW, dash="0.8 0.5")
    f.text(zx + 1.3, cy + 0.9, "0", FS["small"], 700, c=PAL["new"], a="middle")
    f.text(zx + 1.3, fy + fh + 3.4, "1^{3}", FS["small"], c=PAL["new_d"], a="middle")
    f.text(zx + 1.3, fy - 2.3, "4·64", FS["small"], c=PAL["mute"], a="middle")
    num(f, zx + 1.3, fy + fh + 6.0, 3)
    sx2 = zx + 9.5
    f.plus_node(sx2, cy, r=1.6)
    f.arrow(zx + 2.9, cy, sx2 - 1.8, cy, stroke=PAL["new"])
    f.text(zx + 6.1, cy - 1.0, "Δ", FS["body"], 700, c=PAL["new"], a="middle")
    # ---- 2D head branch above
    hx, hy, hw, hh = fx + fw / 2 + 0.5 * L * DEP[0] - 13.0, oy + 4.2, 25.0, 5.4
    fxc = fx + fw / 2 + 0.5 * L * DEP[0]
    f.path(f"M{fxc:.2f},{fy + 0.5 * L * DEP[1] - 0.2:.2f} L{fxc:.2f},{hy + hh + 0.3:.2f}")
    f.rect(hx, hy, hw, hh, rx=1.2, fill=PAL["inh_f"], stroke=PAL["inh"], sw=1.0 * LW)
    f.text(hx + hw / 2, hy + 3.6, "2D head: 1×1 conv → 3·64", FS["small"], 700, a="middle")
    f.path(f"M{hx + hw + 0.3:.2f},{hy + hh / 2:.2f} L{sx2:.2f},{hy + hh / 2:.2f} L{sx2:.2f},{cy - 1.8:.2f}")
    f.text(sx2 - 0.8, hy + hh / 2 + 2.6, "d_{y}, d_{x}, p", FS["small"], c=PAL["mute"], a="end")
    f.prov_add("2D head = Cellpose-SAM 1x1 Conv2d(256 -> 3*64) per plane; Delta added for (dy, dx, p); dz = Delta_z; x8 pixel shuffle back to 256x256",
               "net3d.py:287-295; cellpose vit.py:122")
    # ---- output
    ox_ = sx2 + 5.0
    f.arrow(sx2 + 1.8, cy, ox_ - 0.3, cy)
    f.text((sx2 + ox_) / 2 + 0.6, cy + 3.0, "×8", FS["small"], c=PAL["mute"], a="middle")
    ow = ox + w - 0.5 - L * DEP[0] - ox_
    block(f, ox_, fy, ow, fh, L, "#ffffff", "#ffffff", tint(PAL["inh"], 0.10), INK, top="../assets3/out_top.png", sw=0.9 * LW)
    f.image(ox_ + 0.3, fy + 0.4, ow - 0.6, fh - 0.8, "flow_glyphs_pal.png")
    f.text(ox_ + ow / 2 + 2.0, fy + fh + 3.4, "(d_{z}, d_{y}, d_{x}, p)", FS["small"], 700, a="middle")
    f.text(ox_ + ow / 2 + 2.0, fy + fh + 7.2, "4×Z×256×256", FS["small"], c=PAL["mute"], a="middle")
    num(f, ox_ + ow / 2 - 8.0, fy + fh + 2.8, 4)
    # ---- loss
    lx = sx2 - 22.0
    lw_, ly = ox + w - 0.5 - lx, fy + fh + 8.2
    f.rect(lx, ly, lw_, 7.0, rx=1.2, fill=PAL["loss_f"], stroke=PAL["loss"], sw=0.9 * LW)
    f.text(lx + lw_ / 2, ly + 3.0, "½ MSE(flows, 5*u*) + BCE(p, fg)", FS["small"], 700, c=PAL["loss"], a="middle")
    f.text(lx + lw_ / 2, ly + 5.7, "vs the 3D target, inside the volume", FS["small"], c=PAL["mute"], a="middle")
    f.path(f"M{ox_ + ow + L * DEP[0] - 1.0:.2f},{fy + fh + L * DEP[1] + 0.6:.2f} L{ox_ + ow + L * DEP[0] - 1.0:.2f},{ly - 0.3:.2f}", stroke=PAL["loss"], dash="0.8 0.5")
    f.prov_add("loss: MSE(net flow, 5 x unit target)/2 + BCE(cellprob, nucleus) over voxels inside the volume; AdamW, bf16 autocast, grad clip 1.0",
               "train3d.py:33-43, 215, 275-280")
    # ================= zoom insets
    iy, ih = oy + 47.0, 33.0
    # inset E: transformer block
    ex, ew = ox + 1.0, 60.0
    f.rect(ex, iy, ew, ih, rx=1.5, fill=tint(PAL["inh"], 0.07), stroke=PAL["inh"], sw=0.7 * LW)
    hb = vx + 1.4 + 11 * (vw - 2.8) / 24
    f.path(f"M{hb:.2f},{cy + vh / 2 + 0.2:.2f} L{ex + 4.0:.2f},{iy - 0.2:.2f}", stroke=PAL["inh"], arrow=False, dash="0.7 0.5", chk=False)
    f.path(f"M{hb + 0.8:.2f},{cy + vh / 2 + 0.2:.2f} L{ex + ew - 4.0:.2f},{iy - 0.2:.2f}", stroke=PAL["inh"], arrow=False, dash="0.7 0.5", chk=False)
    f.text(ex + 1.6, iy + 3.4, "transformer block (×24)", FS["body"], 700)
    f.text(ex + ew - 1.4, iy + 3.4, "tokens 32×32×1024", FS["small"], c=PAL["mute"], a="end")
    seq = [("LN", 4.2, False), ("MHSA", 9.5, False), ("+", 0, True), ("LN", 4.2, False), ("MLP", 9.5, False), ("+", 0, True)]
    bx, by_, bh_ = ex + 4.0, iy + 8.4, 6.0
    xs_ = []
    for nm, bw_, plus in seq:
        if plus:
            f.plus_node(bx + 1.3, by_ + bh_ / 2, r=1.25)
            xs_.append(bx + 1.3)
            bx += 2.6
        else:
            f.rect(bx, by_, bw_, bh_, rx=0.8, fill="#ffffff", stroke=PAL["inh"], sw=0.9 * LW)
            f.text(bx + bw_ / 2, by_ + bh_ / 2 + 0.8, nm, FS["small"], 700, a="middle")
            xs_.append(bx + bw_ / 2)
            bx += bw_
        if nm != "+" or len(xs_) < 6:
            if len(xs_) < 6:
                f.arrow(bx + 0.2, by_ + bh_ / 2, bx + 2.0, by_ + bh_ / 2, sw=0.7 * LW)
                bx += 2.2
    # residual elbows
    for a_, b_ in ((ex + 2.6, xs_[2]), (xs_[2] + 1.6, xs_[5])):
        f.path(f"M{a_:.2f},{by_ + bh_ / 2:.2f} L{a_:.2f},{by_ - 1.6:.2f} L{b_:.2f},{by_ - 1.6:.2f} L{b_:.2f},{by_ + bh_ / 2 - 1.5:.2f}", sw=0.7 * LW)
    f.text(xs_[1], by_ + bh_ + 2.6, "16 heads, global,", FS["small"], c=PAL["mute"], a="middle")
    f.text(xs_[1], by_ + bh_ + 4.9, "rel. position", FS["small"], c=PAL["mute"], a="middle")
    f.text(xs_[4], by_ + bh_ + 2.6, "1024→4096", FS["small"], c=PAL["mute"], a="middle")
    f.text(xs_[4], by_ + bh_ + 4.9, "→1024, GELU", FS["small"], c=PAL["mute"], a="middle")
    f.text(ex + 1.6, iy + ih - 4.4, "neck: 1×1 conv 1024→256 · LN · 3×3 conv · LN → 32×32×256", FS["small"])
    f.text(ex + 1.6, iy + ih - 1.8, "patch embed frozen; all other encoder weights fine-tuned", FS["small"], c=PAL["mute"])
    f.prov_add("transformer block: x + MHSA(LN(x)); x + MLP(LN(x)); MHSA 16 heads with relative position bias, global; MLP 1024->4096->1024 GELU",
               "segment_anything image_encoder.py Block (norm1, attn, norm2, mlp, shortcut) l.151-175; build_sam.py:27-34, 66-80")
    # inset R: residual block + dilation
    rxi, rwi = ex + ew + 2.5, 74.0
    f.rect(rxi, iy, rwi, ih, rx=1.5, fill=tint(PAL["new"], 0.05), stroke=PAL["new_l"], sw=0.7 * LW)
    rb2 = r0 + 2 * step_ + rb_w / 2
    f.path(f"M{rb2 - 1.5:.2f},{fy + fh + 6.6:.2f} L{rxi + 4.0:.2f},{iy - 0.2:.2f}", stroke=PAL["new_l"], arrow=False, dash="0.7 0.5", chk=False)
    f.path(f"M{rb2 + 1.5:.2f},{fy + fh + 6.6:.2f} L{rxi + rwi - 4.0:.2f},{iy - 0.2:.2f}", stroke=PAL["new_l"], arrow=False, dash="0.7 0.5", chk=False)
    f.text(rxi + 1.6, iy + 3.4, "residual block (×4)", FS["body"], 700, c=PAL["new_d"])
    f.text(rxi + rwi - 1.4, iy + 3.4, "256×Z×32×32 in and out", FS["small"], c=PAL["mute"], a="end")
    seq2 = [("GN", 4.0), ("GELU", 5.6), ("conv 3^{3}", 9.2), ("GN", 4.0), ("GELU", 5.6), ("conv 3^{3}", 9.2)]
    bx, by_ = rxi + 4.0, iy + 8.4
    first = bx
    for i, (nm, bw_) in enumerate(seq2):
        conv = nm.startswith("conv")
        f.rect(bx, by_, bw_, bh_, rx=0.8, fill=PAL["new_f"] if conv else "#ffffff", stroke=PAL["new"], sw=0.9 * LW)
        f.text(bx + bw_ / 2, by_ + bh_ / 2 + 0.8, nm, FS["small"], 700 if conv else 400, a="middle")
        if i == 2:
            f.text(bx + bw_ / 2, by_ + bh_ + 2.6, "dilation (d,1,1)", FS["small"], c=PAL["new_d"], a="middle")
        bx += bw_
        if i < 5:
            f.arrow(bx + 0.2, by_ + bh_ / 2, bx + 1.6, by_ + bh_ / 2, sw=0.7 * LW)
            bx += 1.8
    pxn = bx + 2.6
    f.plus_node(pxn, by_ + bh_ / 2, r=1.25)
    f.arrow(bx + 0.2, by_ + bh_ / 2, pxn - 1.4, by_ + bh_ / 2, sw=0.7 * LW)
    f.path(f"M{first - 1.4:.2f},{by_ + bh_ / 2:.2f} L{first - 1.4:.2f},{by_ - 1.6:.2f} L{pxn:.2f},{by_ - 1.6:.2f} L{pxn:.2f},{by_ + bh_ / 2 - 1.4:.2f}", sw=0.7 * LW)
    # dilation columns (z vertical)
    nz, pz = 39, 0.29
    top = iy + 19.2
    for j, dd in enumerate((1, 2, 4, 8)):
        cxj = rxi + 4.0 + j * 16.6
        cw, cdp = 5.0, 2.0
        ddx, ddy = cdp * DEP[0], cdp * DEP[1]
        for q in range(nz):
            yy = top + q * pz
            off = q - nz // 2
            tap = off in (-dd, 0, dd)
            col = PAL["new"] if tap else tint(PAL["inh"], 0.22)
            f.add(f'<polygon points="{cxj:.2f},{yy:.2f} {cxj + cw:.2f},{yy:.2f} {cxj + cw + ddx:.2f},{yy + ddy:.2f} {cxj + ddx:.2f},{yy + ddy:.2f}" '
                  f'fill="{col}" stroke="none"/>')
        r = rf[j + 1]
        q0, q1 = nz // 2 - r // 2, nz // 2 + r // 2
        y0b, y1b = top + q0 * pz - 0.4, top + q1 * pz + 0.2
        bxb = cxj + cw + ddx + 1.0
        f.add(f'<path d="M{bxb:.2f},{y0b:.2f} h0.7 V{y1b:.2f} h-0.7" stroke="{PAL["new_d"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
        f.text(bxb + 1.4, (y0b + y1b) / 2 + 0.8, f"{r}", FS["body"], 700, c=PAL["new_d"])
    f.text(rxi + 1.6, iy + ih - 1.4, "red: z taps of the dilated conv; bracket: z receptive field (planes) after block", FS["small"], c=PAL["mute"])
    f.prov_add("ResBlock3D: x + conv3(GELU(GN(conv3_dil(GELU(GN(x)))))), GroupNorm 16 groups; first conv dilation (d,1,1)", "net3d.py:69-78")
    # inset Z: zero-init composition
    zxi = rxi + rwi + 2.5
    zwi = ox + w - 0.5 - zxi
    f.rect(zxi, iy, zwi, ih, rx=1.5, fill=tint(PAL["new"], 0.05), stroke=PAL["new_l"], sw=0.7 * LW)
    f.text(zxi + 1.6, iy + 3.4, "output composition", FS["body"], 700, c=PAL["new_d"])
    rowsZ = [("d_{z}", "0", "Δ_{z}"), ("d_{y}", "2D", "2D+Δ"), ("d_{x}", "2D", "2D+Δ"), ("p", "2D", "2D+Δ")]
    cA, cB = zxi + 9.5, zxi + zwi - 9.0
    f.text(cA, iy + 7.6, "step 0", FS["small"], 700, a="middle")
    f.text(cB, iy + 7.6, "trained", FS["small"], 700, c=PAL["new_d"], a="middle")
    for k_, (ch, a_, b_) in enumerate(rowsZ):
        yy = iy + 9.4 + k_ * 5.0
        f.text(zxi + 2.6, yy + 2.6, ch, FS["small"])
        f.rect(cA - 4.6, yy, 9.2, 4.4, rx=0.6, fill="#ffffff" if a_ == "0" else PAL["inh_f"], stroke=PAL["new"] if a_ == "0" else PAL["inh"],
               sw=0.8 * LW, dash="0.8 0.5" if a_ == "0" else None)
        f.text(cA, yy + 2.7, a_, FS["small"], 700 if a_ == "0" else 400, c=PAL["new"] if a_ == "0" else INK, a="middle")
        f.rect(cB - 5.4, yy, 10.8, 4.4, rx=0.6, fill=PAL["new_f"], stroke=PAL["new"], sw=0.8 * LW)
        f.text(cB, yy + 2.4, b_, FS["small"], a="middle")
    f.arrow(cA + 5.0, iy + 16.0, cB - 5.8, iy + 16.0, sw=0.7 * LW)
    f.text(zxi + 1.6, iy + ih - 2.4, "step 0 = 2D control, d_{z} = 0", FS["small"], c=PAL["mute"])
    f.prov_add("output composition (dz, dy, dx, p) = (Delta_z, 2D head + Delta); Delta = 0 at step 0, so S1 equals the seed-matched 2D model (2D control) with dz = 0",
               "net3d.py:89-90, 294; model3d.json init orx_v5_2d_s0; net3d.py:359-371 self-test")
    # ================= rationale (wording follows main.tex Introduction / Results)
    ry = iy + ih + 4.0
    notes = ["per-plane encoding keeps the 2D foundation model's representations, which native 3D models do not inherit",
             "axial resolution is several-fold poorer than lateral: dilation widens z context to 39 planes without extra weights",
             "zero-initialised output: the network starts as the 2D model and learns only the volumetric correction",
             "the axial flow comes from 3D context, not from summing flows predicted on separate orthogonal slices"]
    for k_, t_ in enumerate(notes):
        colx = ox + 1.0 + (k_ % 2) * 91.5
        rowy = ry + (k_ // 2) * 3.6
        num(f, colx + 1.6, rowy - 0.7, k_ + 1)
        f.text(colx + 4.0, rowy, t_, FS["small"])
    f.prov_add("rationale 1-4 wording", "main.tex Introduction l.44-50 ('do not inherit the representations of 2D foundation models'; 'axial resolution is usually several times poorer than lateral'; 'starts as the 2D model and learns only the volumetric correction'; 'Cellpose sums flows predicted on orthogonal slices into a 3D flow field ... cannot learn where a nucleus ends along the optical axis'); 'without extra weights' = dilation adds no parameters (net3d.py:72)")


# ======================================================================== d: inference with real data
def mini_axes(f, x, y, w, h):
    f.add(f'<path d="M{x:.2f},{y:.2f} V{y + h:.2f} H{x + w:.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}" fill="none"/>')


def panel_infer4(f, ox, oy, w, letter="d"):
    f.plabel(ox, oy + 3.0, letter, "Inference on one test volume (mouse embryo, S1 seed 0)")
    n = 7
    gap = 2.4
    bw = (w - 1.0 - (n - 1) * gap) / n
    by, bh = oy + 5.6, 34.0
    xs = [ox + 0.5 + i * (bw + gap) for i in range(n)]
    heads = ["1 diameter", "2 isotropic", "3 z windows", "4 three views", "5 follow flow", "6 rule T", "7 3D nuclei"]
    for i, x in enumerate(xs):
        f.rect(x, by, bw, bh, rx=1.2, fill="#ffffff", stroke=PAL["box"], sw=0.8 * LW)
        f.text(x + 1.3, by + 3.3, heads[i], FS["body"], 700)
        if i < n - 1:
            f.arrow(x + bw + 0.3, by + 5.0, x + bw + gap - 0.3, by + 5.0, sw=0.8 * LW)
    # 1 diameter: stability - inconsistency vs scale (E15_Ab1, run 4a362b0f)
    x = xs[0]
    rows = [r for r in RD["diam"]["E15_Ab1"] if r["thr"] == 0.0]
    sc = [r["scale"] for r in rows]
    sc_all = [6, 10, 17, 30, 52, 90, 150]
    gx0, gy0, gw, gh = x + 5.4, by + 7.0, bw - 7.0, 14.0
    mini_axes(f, gx0, gy0, gw, gh)
    for v_, lab_ in ((-0.5, "−0.5"), (0.0, "0"), (0.5, "0.5")):
        yv_ = gy0 + gh - gh * (v_ + 0.6) / 1.4
        f.add(f'<line x1="{gx0 - 0.6:.2f}" y1="{yv_:.2f}" x2="{gx0:.2f}" y2="{yv_:.2f}" stroke="{INK}" stroke-width="{0.4 * LW:.3f}"/>')
        f.text(gx0 - 0.9, yv_ + 0.7, lab_, FS["small"], c=PAL["mute"], a="end")
    lx = lambda s_: gx0 + gw * (math.log(s_) - math.log(6)) / (math.log(150) - math.log(6))
    ly = lambda v: gy0 + gh - gh * (v + 0.6) / 1.4          # range -0.6 .. 0.8
    for key, col, lab in (("stab", PAL["ctl"], "stability"), ("score", INK, "stab − |log(d̂/d)|")):
        pts = []
        for r in rows:
            v = r["stab"] if key == "stab" else r["stab"] - r["cons"]
            pts.append((lx(r["scale"]), ly(v)))
        f.add('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + f'" stroke="{col}" stroke-width="{0.9 * LW:.3f}" fill="none"/>')
        for a, b in pts:
            f.circle(a, b, 0.45, fill=col, chk=False)
    sel = RD["diam_sel"]["E15_Ab1"][0]
    f.circle(lx(sel["scale"]), ly([r for r in rows if r["scale"] == sel["scale"]][0]["stab"] - [r for r in rows if r["scale"] == sel["scale"]][0]["cons"]), 1.0,
             fill="none", stroke=PAL["new"], sw=1.0 * LW, chk=False)
    for s_ in sc_all:
        f.add(f'<line x1="{lx(s_):.2f}" y1="{gy0 + gh:.2f}" x2="{lx(s_):.2f}" y2="{gy0 + gh + 0.6:.2f}" stroke="{INK}" stroke-width="{0.4 * LW:.3f}"/>')
    f.text(lx(6), gy0 + gh + 2.6, "6", FS["small"], c=PAL["mute"], a="middle")
    f.text(lx(30), gy0 + gh + 2.6, "30", FS["small"], c=PAL["mute"], a="middle")
    f.text(lx(150), gy0 + gh + 2.6, "150", FS["small"], c=PAL["mute"], a="middle")
    f.text(x + bw / 2, by + 26.4, "scale (px)", FS["small"], c=PAL["mute"], a="middle")
    f.text(x + 1.3, by + 29.0, "blue stability; black", FS["small"], c=PAL["mute"])
    f.text(x + 1.3, by + 31.6, f"minus |log d̂/d|; d̂ = {sel['d_used']:.0f}", FS["small"], c=PAL["mute"])
    f.text(gx0 + 1.0, gy0 + gh - 1.2, "E15_Ab1", FS["small"], c=PAL["mute"])
    f.prov_add("diameter scan: 2D control seed 0, cp 0: stability (blue) and stability - |log(d_est/scale)| (black), both dimensionless on one shared y axis for mouse_embryo E15_Ab1 (E15_Ab2 is not in the 4-per-dataset full-pipeline sample); scales 6 and 10 px skipped (rescaled volume > 8e8 voxels); selected scale 30, threshold -2, d_used 33.8 px (GT 32.5)",
               f"server {_NSP_ROOT}/results/orx_4a362b0f-..._test/rows_all.json and rows.json (eval_val.py joint r2a2); arch_views.py -> assets3/ruleT_diam.json")
    # 2 isotropic
    x = xs[1]
    tt = bw - 3.0
    f.image(x + 1.5, by + 5.0, tt, 11.5, "xz_native.png", fit="xMidYMid slice")
    f.image(x + 1.5, by + 19.0, tt, 11.5, "xz_iso.png", fit="xMidYMid slice")
    f.text(x + bw / 2, by + 18.2, "z/xy = 4.65 → linear z", FS["small"], c=PAL["mute"], a="middle")
    f.text(x + bw / 2, by + 33.0, "×30/d, isotropic", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("isotropic resampling of E15_Ab2 (z/xy 4.65): native xz plane vs linear z interpolation", "eval_gt.py:217-222; infer3d.py:211-217; make_assets.py")
    # 3 windows
    x = xs[2]
    st, Zp = LOGA["windows"]["starts"], LOGA["windows"]["Z_network"]
    a0, a1 = x + 1.4, x + bw - 1.4
    base, amp = by + 18.0, 9.0
    s_ = (a1 - a0) / Zp
    for i, s0 in enumerate(st):
        pts = [(a0 + (s0 + k + 0.5) * s_, base - amp * math.sin(math.pi * (k + 0.5) / 32) ** 2) for k in range(32)]
        f.add('<path d="M' + " L".join(f"{p:.2f},{q:.2f}" for p, q in pts) + f'" stroke="{PAL["new"] if i % 2 == 0 else PAL["new_l"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
    f.add(f'<line x1="{a0:.2f}" y1="{base:.2f}" x2="{a1:.2f}" y2="{base:.2f}" stroke="{INK}" stroke-width="{0.6 * LW:.3f}"/>')
    f.text(x + bw / 2, base + 3.0, f"z′ = {Zp} planes", FS["small"], c=PAL["mute"], a="middle")
    f.text(x + 1.3, by + 26.4, f"{len(st)} windows of 32,", FS["small"], c=PAL["mute"])
    f.text(x + 1.3, by + 29.2, "stride 23–24,", FS["small"], c=PAL["mute"])
    f.text(x + 1.3, by + 32.0, "sin² blending", FS["small"], c=PAL["mute"])
    # 4 three views (real per-view outputs)
    x = xs[3]
    vw_ = (bw - 4.0) / 2
    vh_ = vw_ * 28 / 48
    pos = [("z", x + 1.3, by + 7.0), ("y", x + 2.7 + vw_, by + 7.0), ("x", x + 1.3, by + 18.0), ("mean", x + 2.7 + vw_, by + 18.0)]
    for k_, px_, py_ in pos:
        f.image(px_, py_, vw_, vh_, f"../assets3/view_{k_}.png")
        lab = f"{k_} as z" if k_ != "mean" else "mean"
        f.text(px_ + vw_ / 2, py_ + vh_ + 2.3, lab, FS["small"], c=PAL["mute"] if k_ != "mean" else PAL["new_d"], a="middle")
    f.text(x + 1.3, by + 31.8, "u_{z}, nucleus 62", FS["small"], c=PAL["mute"])
    f.prov_add("three views: S1 seed-0 raw outputs with each volume axis as network z (one 32-plane window per view centred on GT nucleus 62), u_z on the xz section, and their mean; the pipeline blends all windows with sin^2 weights",
               "arch_views.py (server, CPU) -> assets3/views_E15_Ab2_n62.npz; infer3d.py:63-96 VIEW_PERMS, run_views")
    # 5 follow flow
    x = xs[4]
    f.image(x + 0.8, by + 4.6, bw - 1.6, 24.0, "flowfollow_F1d.png")
    f.text(x + 1.3, by + 31.0, "one sink per nucleus", FS["small"], c=PAL["mute"])
    f.prov_add("follow-flow render: nc-viz F1-d_follow_light (S1 seed 0, E15_Ab2 GT 33, 51, 68, 50, all matched at IoU >= 0.72, one sink each; strands coloured by u_z; grey arrow 3 um along +z); fields stored at native planes and re-interpolated along z for display",
               "paper/figures/assets_viz/F1-d.provenance.json (rule F1_group, selection_rules.json)")
    # 6 rule T: m(t) curve
    x = xs[5]
    tr = RD["ruleT"]
    ts = [-6, -4, -3, -2, -1, 0]
    gx0, gy0, gw, gh = x + 5.4, by + 7.0, bw - 7.0, 14.0
    mini_axes(f, gx0, gy0, gw, gh)
    for v_ in (0.7, 1.0, 1.5):
        yv_ = gy0 + gh - gh * (v_ - 0.5) / 1.2
        f.add(f'<line x1="{gx0 - 0.6:.2f}" y1="{yv_:.2f}" x2="{gx0:.2f}" y2="{yv_:.2f}" stroke="{INK}" stroke-width="{0.4 * LW:.3f}"/>')
        f.text(gx0 - 0.9, yv_ + 0.7, f"{v_:.1f}", FS["small"], c=PAL["loss"] if v_ == 0.7 else PAL["mute"], a="end")
    lx2 = lambda t: gx0 + gw * (t + 6) / 6.0
    ly2 = lambda v: gy0 + gh - gh * (v - 0.5) / 1.2         # 0.5 .. 1.7
    pts = [(lx2(t), ly2(tr["m"][f"{float(t)}"])) for t in ts]
    for yv, lab in ((0.7, "0.7"), (0.85, "0.85")):
        f.add(f'<line x1="{gx0:.2f}" y1="{ly2(yv):.2f}" x2="{gx0 + gw:.2f}" y2="{ly2(yv):.2f}" stroke="{PAL["loss"]}" stroke-width="{0.5 * LW:.3f}" stroke-dasharray="0.6 0.4"/>')
    f.add('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + f'" stroke="{INK}" stroke-width="{0.9 * LW:.3f}" fill="none"/>')
    for a, b in pts:
        f.circle(a, b, 0.45, fill=INK, chk=False)
    f.circle(lx2(tr["chosen"]), ly2(tr["m"][f"{tr['chosen']}"]), 1.0, fill="none", stroke=PAL["new"], sw=1.0 * LW, chk=False)
    f.text(lx2(tr["chosen"]) - 1.2, ly2(tr["m"][f"{tr['chosen']}"]) + 2.6, f"{tr['m']['-2.0']:.2f}", FS["small"], 700, c=PAL["new_d"], a="end")
    for t in (-6, -2, 0):
        f.text(lx2(t), gy0 + gh + 2.6, f"{t}".replace("-", "−"), FS["small"], c=PAL["mute"], a="middle")
    f.text(x + bw / 2, by + 26.4, "threshold t", FS["small"], c=PAL["mute"], a="middle")
    f.text(x + 1.3, by + 29.0, f"m(−2) = {tr['m']['-2.0']:.2f} ≥ 0.7", FS["small"], c=PAL["mute"])
    f.text(gx0 + gw, gy0 + 1.6, "m(t)", FS["small"], c=PAL["mute"], a="end")
    f.text(x + 1.3, by + 31.6, f"→ t = −2 kept", FS["small"], c=PAL["new_d"])
    f.prov_add(f"rule T on E15_Ab2 (S1 seed 0): m(t) = N_fg(t)/(N_obj(t) V) for t in 0..-6, m(-2) = {tr['m']['-2.0']:.2f} >= 0.7 so t = -2",
               "sizeThr_dir.py:16-28 recomputed by arch_views.py from run 07992b55 thr_eval rows.json and masks/mouse_embryo__E15_Ab2.npz cp")
    # 7 3D nuclei
    x = xs[6]
    f.image(x + 0.8, by + 4.6, bw - 1.6, 24.0, "masks_S1.png")
    f.text(x + 1.3, by + 31.0, f"{tr['n_pred']['-2.0']} objects; F1 {tr['f1_by_thr']['-2.0']:.2f}", FS["small"], c=PAL["mute"])
    f.prov_add("final S1 masks of E15_Ab2 (render nc-figures), objects before size filter and F1@0.5 after it at t = -2", "E15_Ab2_iso.provenance.json; thr_eval rows.json")


# ======================================================================== layouts
def layout_a():
    f = Fig(183.0, 170.0)
    panel_cmp2(f, 0.0, 0.0, 119.0, letter="a")
    panel_target3d(f, 122.0, 0.0, 61.0, letter="b")
    panel_arch4(f, 0.0, 37.5, letter="c")
    panel_infer4(f, 0.0, 129.0, 183.0, letter="d")
    return f


def layout_b():
    """Variant b: architecture first (largest), then the evidence row (a-pair + angle) beside the 3D target, inference last."""
    f = Fig(183.0, 170.0)
    panel_arch4(f, 0.0, 0.0, letter="a")
    panel_cmp2(f, 0.0, 91.5, 119.0, letter="b")
    panel_target3d(f, 122.0, 91.5, 61.0, letter="c")
    panel_infer4(f, 0.0, 129.0, 183.0, letter="d")
    return f


def layout_a5():
    f = Fig(183.0, 170.0)
    panel_cmp2(f, 0.0, 0.0, 119.0, letter="a")
    panel_target3d(f, 122.0, 0.0, 61.0, letter="b", tile3="target_F1d.png")
    panel_arch4(f, 0.0, 37.5, letter="c")
    panel_infer4(f, 0.0, 129.0, 183.0, letter="d")
    return f


def layout_b5():
    """Final choice (user, 11-10): A4-b layout with the A5-a real-nuclei 3D target tile."""
    f = Fig(183.0, 170.0)
    panel_arch4(f, 0.0, 0.0, letter="a")
    panel_cmp2(f, 0.0, 91.5, 119.0, letter="b")
    panel_target3d(f, 122.0, 91.5, 61.0, letter="c", tile3="target_F1d.png")
    panel_infer4(f, 0.0, 129.0, 183.0, letter="d")
    return f


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for cid, fn in (("A4-a", layout_a), ("A4-b", layout_b), ("A5-a", layout_a5), ("A5-b", layout_b5)):
        if which not in ("all", cid[-1], cid):
            continue
        f = fn()
        res = export(f, cid)
        print(cid, "issues:", len(res.get("issues", [])), "min font", res.get("min_font_pt"))
        for s_ in res.get("issues", [])[:80]:
            print("   ", s_)
        json.dump(dict(figure=cid, code_snapshot=_nsp_os.path.join(_NSP_REPO, 'src'),
                       model="orx_t3d_r5s1_s0/model3d.json", palette="paper/figures/palette.json",
                       references=[dict(figure=a, technique=b) for a, b in REFS], items=f.prov),
                  open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
