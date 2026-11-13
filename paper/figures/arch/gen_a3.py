"""Fig. 1 v3 (A3-a / A3-b): architecture-centred schematic of S1, hand-coded SVG -> vector PDF + 600-dpi PNG.

Reuses the drawing primitives, palette loading, overlap / font-size check and Chrome export of gen_fig.py.
Real-data textures: make_a3_assets.py (assets3/), make_assets.py (assets/), nc-viz and nc-figures renders (copied to assets/).
Every fact drawn is listed with its code location in A3-*.provenance.json.
Usage: python gen_a3.py [a|b|all]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
import base64
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fig import FS, LW, PAL, PT, Fig, export, tint, M as PAL_M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = "../assets3/"
LOG3 = json.load(open(os.path.join(HERE, "assets3", "a3_assets_log.json")))
LOGA = json.load(open(os.path.join(HERE, "assets", "assets_log.json")))
DEP = (math.cos(math.radians(35)), -math.sin(math.radians(35)))
INK = PAL["fg"]


def b64p(name):
    p = os.path.join(HERE, "assets", name)
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def tex(f, name, a, b, c, d, e, g):
    """Image on the unit square mapped by matrix(a,b,c,d,e,f)."""
    f.add(f'<image class="im" x="0" y="0" width="1" height="1" preserveAspectRatio="none" href="{b64p(name)}" '
          f'transform="matrix({a:.4f},{b:.4f},{c:.4f},{d:.4f},{e:.3f},{g:.3f})"/>')


def block(f, x, y, w, h, L, ff, ft, fs, stroke, front=None, top=None, side=None, sw=None, grid=0, op=1.0):
    """Oblique cuboid: front face (x, y, w, h); depth L along 35 deg. Optional textures for the three faces."""
    dx, dy = L * DEP[0], L * DEP[1]
    sw = sw or 0.8 * LW
    topP = [(x, y), (x + w, y), (x + w + dx, y + dy), (x + dx, y + dy)]
    sideP = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
    frontP = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    f.poly(topP, ft, None, chk=False, op=op)
    if top:
        tex(f, top, w, 0, dx, dy, x, y)
    f.poly(sideP, fs, None, chk=False, op=op)
    if side:
        tex(f, side, dx, dy, 0, h, x + w, y)
    f.poly(frontP, ff, None, chk=False, op=op)
    if front:
        tex(f, front, w, 0, 0, h, x, y)
    if grid:
        for k in range(1, grid):
            f.add(f'<line x1="{x + k * w / grid:.2f}" y1="{y:.2f}" x2="{x + k * w / grid:.2f}" y2="{y + h:.2f}" stroke="{stroke}" stroke-width="{0.35 * LW:.3f}" opacity="0.6"/>')
            f.add(f'<line x1="{x:.2f}" y1="{y + k * h / grid:.2f}" x2="{x + w:.2f}" y2="{y + k * h / grid:.2f}" stroke="{stroke}" stroke-width="{0.35 * LW:.3f}" opacity="0.6"/>')
    for P in (topP, sideP, frontP):
        f.poly(P, "none", stroke, sw=sw, chk=True)
    return x + w + dx, y + dy


def chip(f, x, y, fill, stroke, label):
    f.rect(x, y - 2.6, 4.2, 3.2, rx=0.7, fill=fill, stroke=stroke, sw=1.0 * LW)
    f.text(x + 5.4, y, label, FS["body"])


# ======================================================================== panel a: architecture
def panel_arch(f, ox, oy, w=183.0, insets="below", letter="a"):
    f.plabel(ox, oy + 3.0, letter, "S1: a 2D foundation encoder inside a native 3D network")
    chip(f, ox + w - 60.5, oy + 3.0, PAL["inh_f"], PAL["inh"], "inherited (2D control)")
    chip(f, ox + w - 22.5, oy + 3.0, PAL["new_f"], PAL["new"], "new in S1")
    fy, fh, L = oy + 22.0, 20.0, 12.0
    cy = fy + fh / 2
    # ---- input window, textured with real data
    x0 = ox + 2.5
    block(f, x0, fy, 20.0, fh, L, "#000", "#000", "#000", PAL["inh"], front="../assets3/in_front.png",
          top="../assets3/in_top.png", side="../assets3/in_side.png")
    f.text(x0 + 10.0, fy + fh + 4.0, "input window", FS["body"], 700, a="middle")
    f.text(x0 + 10.0, fy + fh + 6.5, "1 × Z × 256 × 256", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("input window of 32 planes x 256 x 256 (model input B,1,Z,256,256); textures = real E15_Ab2 data on the resized isotropic grid, z window 162-193",
               "net3d.py:268-272; train3d.py:218 patch (32,256,256); model3d.json infer.patch_z 32; make_a3_assets.py")
    # axis tripod (x right, y down, z = depth)
    tx, ty = x0 + 20.0 + L * DEP[0] + 7.0, fy + fh + 7.5
    f.path(f"M{tx:.2f},{ty:.2f} L{tx + 4.0:.2f},{ty:.2f}", sw=0.7 * LW, chk=False)
    f.path(f"M{tx:.2f},{ty:.2f} L{tx:.2f},{ty + 3.6:.2f}", sw=0.7 * LW, chk=False)
    f.path(f"M{tx:.2f},{ty:.2f} L{tx + 3.4 * DEP[0]:.2f},{ty + 3.4 * DEP[1]:.2f}", sw=0.7 * LW, chk=False, stroke=PAL["new"])
    f.text(tx + 4.8, ty + 1.0, "x", FS["small"], italic=True)
    f.text(tx + 1.0, ty + 4.6, "y", FS["small"], italic=True)
    f.text(tx + 3.6, ty - 2.2, "z", FS["small"], italic=True, c=PAL["new"])
    # ---- planes, one at a time
    px = x0 + 20.0 + L * DEP[0] + 4.0
    f.arrow(px - 3.2, cy, px - 0.4, cy)
    ps = 11.0
    for k in range(4, -1, -1):
        dxk, dyk = k * 1.6, -k * 1.15
        f.image(px + dxk, cy - ps / 2 + dyk + 1.5, ps, ps, f"plane_{4 - k}.png", fit="xMidYMid slice", op=1.0 if k == 0 else 0.85)
        f.rect(px + dxk, cy - ps / 2 + dyk + 1.5, ps, ps, rx=0, stroke="#ffffff", sw=0.7 * LW, chk=False)
    f.text(px + 7.6, fy - 6.2, "Z planes,", FS["small"], c=PAL["mute"], a="middle")
    f.text(px + 7.6, fy - 3.8, "encoded separately", FS["small"], c=PAL["mute"], a="middle")
    # ---- encoder with ghosted copies along z
    ex, ew, eh = px + ps + 6.4 + 4.0, 13.0, 22.0
    f.arrow(px + ps + 6.6, cy, ex - 0.4, cy)
    for k in (3, 2, 1):
        f.trapezoid(ex + k * 1.6, cy - eh / 2 - k * 1.15, ew, eh, 0.42, PAL["inh_f"], PAL["inh"], op=0.40, chk=False)
    f.trapezoid(ex, cy - eh / 2, ew, eh, 0.42, PAL["inh_f"], PAL["inh"])
    f.text(ex + ew / 2 - 0.5, cy - 1.0, "ViT-L", FS["mod"], 700, a="middle")
    f.text(ex + ew / 2 - 0.5, cy + 1.6, "encoder", FS["mod"], 700, a="middle")
    f.text(ex + ew / 2 + 1.0, fy + fh + 4.0, "shared over", FS["small"], c=PAL["mute"], a="middle")
    f.text(ex + ew / 2 + 1.0, fy + fh + 6.4, "planes · 304 M", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("2D ViT-L encoder + neck run on every z plane (planes folded into the batch), shared weights, fully fine-tuned (304.4 M)",
               "net3d.py:285-286, 237-259; model3d.json freeze_encoder false, lora 0; train log '编码器侧 304.4M'")
    # ---- re-stacked feature volume
    fx = ex + ew + 4.8 + 4.6
    f.arrow(ex + ew + 4.8, cy, fx - 0.4, cy)
    f.text(ex + ew + 4.2, fy - 6.2, "re-stack", FS["small"], c=PAL["mute"], a="middle")
    f.text(ex + ew + 4.2, fy - 3.8, "along z", FS["small"], c=PAL["mute"], a="middle")
    fw = 14.0
    block(f, fx, fy, fw, fh, L, PAL["inh_f"], PAL["inh_f2"], PAL["inh_f2"], PAL["inh"], top="../assets3/feat_enc_top.png", grid=8)
    f.text(fx + fw / 2, fy + fh + 4.0, "plane features", FS["body"], 700, a="middle")
    f.text(fx + fw / 2, fy + fh + 6.5, "256 × Z × 32 × 32", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("per-plane neck features re-stacked into a 3D feature volume (B,256,Z,32,32); top-face texture = PC1 of real S1 encoder features on the xz section of this window",
               "net3d.py:288; make_a3_assets.py feat_enc_top (nc-viz viz_run.py hooks)")
    # ---- 2D head above
    hx, hy, hw, hh = fx + fw + 1.0, oy + 7.0, 24.0, 8.0
    fxc = fx + fw / 2 + 0.5 * L * DEP[0]
    f.path(f"M{fxc:.2f},{fy + 0.5 * L * DEP[1] - 0.2:.2f} L{fxc:.2f},{hy + hh / 2:.2f} L{hx - 0.4:.2f},{hy + hh / 2:.2f}")
    f.rect(hx, hy, hw, hh, rx=1.5, fill=PAL["inh_f"], stroke=PAL["inh"], sw=1.1 * LW)
    f.text(hx + hw / 2, hy + 3.3, "2D head", FS["mod"], 700, a="middle")
    f.text(hx + hw / 2, hy + 6.3, "1×1 conv, per plane", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("2D head = Cellpose-SAM 1x1 Conv2d(256 -> 3*8*8) per plane -> (dy, dx, p)", "net3d.py:287; cellpose vit.py:122")
    # ---- 3D residual decoder volume
    dx_ = fx + fw + L * DEP[0] + 4.6
    f.arrow(fx + fw + L * DEP[0] + 0.6, cy, dx_ - 0.4, cy)
    dw = 16.0
    block(f, dx_, fy, dw, fh, L, PAL["new_f"], tint(PAL["new"], 0.32), tint(PAL["new"], 0.32), PAL["new"],
          top="../assets3/feat_dec_top.png", sw=1.2 * LW)
    f.text(dx_ + dw / 2, fy + 4.6, "3D residual", FS["body"], 700, c=PAL["new_d"], a="middle")
    f.text(dx_ + dw / 2, fy + 7.0, "decoder", FS["body"], 700, c=PAL["new_d"], a="middle")
    f.text(dx_ + dw / 2, fy + 10.8, "4 blocks of", FS["small"], a="middle")
    f.text(dx_ + dw / 2, fy + 13.2, "3×3×3 convs,", FS["small"], a="middle")
    f.text(dx_ + dw / 2, fy + 15.6, "z-dilation", FS["small"], a="middle")
    f.text(dx_ + dw / 2, fy + 18.0, "1, 2, 4, 8", FS["small"], 700, c=PAL["new_d"], a="middle")
    f.text(dx_ + dw / 2, fy + fh + 4.0, "decoder features", FS["body"], 700, c=PAL["new_d"], a="middle")
    f.text(dx_ + dw / 2, fy + fh + 6.5, "256 × Z × 32 × 32", FS["small"], c=PAL["mute"], a="middle")
    f.text(dx_ + dw / 2, fy + fh + 9.0, "14.3 M params", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("Dec3D: 1x1x1 conv in, 4 ResBlock3D (GN-GELU-Conv3d 3^3 with z-dilation d, GN-GELU-Conv3d 3^3, residual), d = 1,2,4,8, 256 ch, GroupNorm 16; 14.3 M params; top-face texture = PC1 of real S1 decoder last hidden features",
               "net3d.py:69-96; model3d.json dec_ch 256, dec_blocks 4, dec_zdil [1,2,4,8], dec_groups 16; train log '新增模块 14.3M'")
    # ---- zero-init output conv and sum
    zx = dx_ + dw + L * DEP[0] + 1.6
    f.rect(zx, fy + 1.0, 3.0, fh - 2.0, rx=0.7, fill="#ffffff", stroke=PAL["new"], sw=1.0 * LW, dash="0.9 0.6")
    f.text(zx + 1.5, cy + 1.0, "0", FS["mod"], 700, c=PAL["new"], a="middle")
    f.text(zx + 1.5, fy + fh + 4.0, "zero-init", FS["small"], c=PAL["new_d"], a="middle")
    f.text(zx + 1.5, fy + fh + 6.4, "1×1×1", FS["small"], c=PAL["new_d"], a="middle")
    sx = zx + 9.0
    f.plus_node(sx, cy)
    f.arrow(zx + 3.4, cy, sx - 1.9, cy, stroke=PAL["new"])
    f.text((zx + 3.4 + sx - 1.9) / 2, cy - 1.0, "Δ", FS["body"], 700, c=PAL["new"], a="middle")
    f.path(f"M{hx + hw + 0.4:.2f},{hy + hh / 2:.2f} L{sx:.2f},{hy + hh / 2:.2f} L{sx:.2f},{cy - 1.9:.2f}")
    f.text(sx - 0.8, hy + hh / 2 + 2.9, "d_{y}, d_{x}, p", FS["small"], c=PAL["mute"], a="end")
    f.prov_add("zero-initialised 1x1x1 output conv; Delta added to the 2D head tokens for (dy, dx, p), dz from Delta only; x8 pixel shuffle",
               "net3d.py:89-90, 289-295")
    # ---- output volume
    ox_ = sx + 4.4
    f.arrow(sx + 1.9, cy, ox_ - 0.4, cy)
    ow = ox + w - 0.5 - L * DEP[0] - ox_
    block(f, ox_, fy, ow, fh, L, "#ffffff", "#ffffff", tint(PAL["inh"], 0.10), PAL["fg"], top="../assets3/out_top.png", sw=0.9 * LW)
    f.image(ox_ + 0.4, fy + 0.6, ow - 0.8, fh - 1.2, "flow_glyphs_pal.png")
    f.text(ox_ + ow / 2 + 2.0, fy + fh + 4.0, "3D flow + foreground", FS["body"], 700, a="middle")
    f.text(ox_ + ow / 2 + 2.0, fy + fh + 6.5, "(d_{z}, d_{y}, d_{x}, p) · 4 × Z × 256 × 256", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("output (B,4,Z,256,256) = [dz, dy, dx, cellprob]; top-face texture = real S1 single-view u_z on the window's xz section (diverging ramp), faded by sigmoid(p); front = S1 flow glyphs on E15_Ab2 nucleus 90 (direction verified inward)",
               "net3d.py:295-299; make_a3_assets.py out_top; render_glyphs.py")
    if insets == "below":
        inset_dil(f, ox + 1.0, oy + 58.5, 90.0, 41.0)
        inset_zero(f, ox + 94.0, oy + 58.5, w - 95.0, 41.0)


def inset_dil(f, x, y, w, h, spread=21.0):
    f.rect(x, y, w, h, rx=1.6, fill=tint(PAL["new"], 0.05), stroke=PAL["new_l"], sw=0.8 * LW)
    f.text(x + 2.0, y + 3.8, "Dilated 3×3×3 kernels widen the view along z", FS["mod"], 700, c=PAL["new_d"])
    rf = [1]
    for dd in (1, 2, 4, 8):
        rf.append(rf[-1] + 2 * dd + 2)
    nz, pz = 39, 0.62
    top = y + 7.6
    for j, dd in enumerate((1, 2, 4, 8)):
        cx = x + 6.0 + j * spread
        cw, cd = 7.0, 3.0
        ddx, ddy = cd * DEP[0], cd * DEP[1]
        for q in range(nz):
            yy = top + 1.8 + q * pz
            off = q - nz // 2
            tap = off in (-dd, 0, dd)
            col = PAL["new"] if tap else tint(PAL["inh"], 0.22)
            op = 1.0 if tap else 0.9
            f.add(f'<polygon points="{cx:.2f},{yy:.2f} {cx + cw:.2f},{yy:.2f} {cx + cw + ddx:.2f},{yy + ddy:.2f} {cx + ddx:.2f},{yy + ddy:.2f}" '
                  f'fill="{col}" stroke="{"#ffffff" if not tap else PAL["new_d"]}" stroke-width="{0.3 * LW:.3f}" opacity="{op}"/>')
            if tap:   # 3 x 3 lateral taps on this plane
                for a in (1, 2):
                    f.add(f'<line x1="{cx + a * cw / 3:.2f}" y1="{yy:.2f}" x2="{cx + a * cw / 3 + ddx:.2f}" y2="{yy + ddy:.2f}" stroke="#ffffff" stroke-width="{0.35 * LW:.3f}"/>')
                    f.add(f'<line x1="{cx + a * ddx / 3:.2f}" y1="{yy + a * ddy / 3:.2f}" x2="{cx + cw + a * ddx / 3:.2f}" y2="{yy + a * ddy / 3:.2f}" stroke="#ffffff" stroke-width="{0.35 * LW:.3f}"/>')
        # cumulative receptive field bracket
        r = rf[j + 1]
        q0, q1 = nz // 2 - r // 2, nz // 2 + r // 2
        y0b, y1b = top + 1.8 + q0 * pz - 0.3, top + 1.8 + q1 * pz + 0.3
        bx = cx + cw + ddx + 1.4
        f.add(f'<path d="M{bx:.2f},{y0b:.2f} h0.8 V{y1b:.2f} h-0.8" stroke="{PAL["new_d"]}" stroke-width="{0.7 * LW:.3f}" fill="none"/>')
        f.text(bx + 1.6, (y0b + y1b) / 2 - 0.4, f"{r}", FS["body"], 700, c=PAL["new_d"])
        f.text(bx + 1.6, (y0b + y1b) / 2 + 2.0, "planes", FS["small"], c=PAL["mute"])
        f.text(cx + cw / 2 + 1.0, top + 1.8 + nz * pz + 3.0, f"block {j + 1}: d = {dd}", FS["small"], a="middle")
    f.text(x + 2.0, y + h - 1.8, "red: kernel taps at z − d, z, z + d; brackets: receptive field along z after each block",
           FS["small"], c=PAL["mute"])
    f.prov_add("z receptive field after blocks 1-4 = 5, 11, 21, 39 planes (each block: 3^3 conv with z-dilation d adds 2d, plain 3^3 conv adds 2; 1x1x1 in/out convs add 0); lateral taps 3x3",
               "derived from net3d.py:72-73 (kernel 3, padding/dilation (d,1,1)) and :86,89 (1x1x1); model3d.json dec_zdil [1,2,4,8]")


def inset_zero(f, x, y, w, h):
    f.rect(x, y, w, h, rx=1.6, fill=tint(PAL["new"], 0.05), stroke=PAL["new_l"], sw=0.8 * LW)
    f.text(x + 2.0, y + 3.8, "Zero-initialised residual: S1 starts as the 2D model", FS["mod"], 700, c=PAL["new_d"])
    f.text(x + 2.0, y + 8.0, "(d_{z}, d_{y}, d_{x}, p) = (Δ_{z}, 2D head + Δ_{yxp}),  Δ = 0 at step 0", FS["small"])
    pw, ph = 24.0, 17.8
    p1x, p2x, py = x + 3.0, x + 3.0 + pw + 12.0, y + 14.4
    f.text(p1x + pw / 2, py - 1.6, "step 0", FS["body"], 700, a="middle")
    f.text(p2x + pw / 2, py - 1.6, "after training", FS["body"], 700, c=PAL["new_d"], a="middle")
    f.image(p1x, py, pw, ph, "../assets3/cmp_2d_xyonly.png")
    f.image(p2x, py, pw, ph, "../assets3/cmp_s1_view1.png")
    f.arrow(p1x + pw + 1.5, py + ph / 2, p2x - 1.5, py + ph / 2)
    f.text((p1x + pw + p2x) / 2, py + ph / 2 - 1.2, "learn Δ", FS["small"], c=PAL["new_d"], a="middle")
    f.text(p1x + pw / 2, py + ph + 2.6, "= 2D control, d_{z} ≡ 0", FS["small"], c=PAL["mute"], a="middle")
    f.text(p2x + pw / 2, py + ph + 2.6, "S1, same single view", FS["small"], c=PAL["mute"], a="middle")
    # colour key
    kx = p2x + pw + 3.0
    f.colorbar(kx + 2.2, py + 4.0, 9.0, 1.3, "", "", None)
    f.text(kx + 6.7, py + 2.4, "u_{z}", FS["small"], c=PAL["mute"], a="middle")
    f.text(kx + 2.2, py + 8.0, "−1", FS["small"], c=PAL["mute"], a="middle")
    f.text(kx + 11.2, py + 8.0, "+1", FS["small"], c=PAL["mute"], a="middle")
    f.text(kx, py + 12.2, ["xz section,", "nucleus 62"], FS["small"], c=PAL["mute"])
    f.prov_add("step 0: S1 is built from the seed-matched 2D model (init orx_v5_2d_s0 = 2D control seed 0) and the zero-init Delta -> its output equals the 2D head per plane with dz = 0; shown: 2D control on xy planes only; after training: S1 single (identity) view on the same planes",
               "net3d.py:89-90, 294, 359-371 (self-test: (dy,dx,cp) = 2D output up to float error, dz = 0); model3d.json init orx_v5_2d_s0; make_a3_assets.py cmp_2d_xyonly / cmp_s1_view1 (v2v4_crop.npz c2d_out, s1_out)")


# ======================================================================== panel b: how z enters
def panel_cmp(f, ox, oy, w, letter="b"):
    f.plabel(ox, oy + 3.0, letter, "How z information enters the flow (xz section through one nucleus)")
    pw, ph = 27.0, 20.0
    gap = (w - 3 * pw - 17.0) / 2
    py = oy + 15.5
    xs = [ox + 2.0 + i * (pw + gap) for i in range(3)]
    titles = [("2D control", ["xy, xz, yz planes predicted", "separately, then fused"]),
              ("S1", ["one 3D network sees", "32 planes at once"]),
              ("3D training target", ["diffusion inside", "the GT nucleus"])]
    names = ["cmp_2d_fused.png", "cmp_s1.png", "cmp_target.png"]
    for i, (x, (t1, t2), nm) in enumerate(zip(xs, titles, names)):
        f.text(x + pw / 2, oy + 7.6, t1, FS["body"], 700, c=PAL["ctl_d"] if i == 0 else (PAL["new_d"] if i == 1 else INK), a="middle")
        f.text(x + pw / 2, oy + 10.0, t2, FS["small"], c=PAL["mute"], a="middle")
        f.image(x, py, pw, ph, "../assets3/" + nm)
    kx = ox + w - 13.0
    f.colorbar(kx + 1.0, py + 5.0, 9.0, 1.3, "", "", None)
    f.text(kx + 5.5, py + 3.4, "u_{z}", FS["small"], c=PAL["mute"], a="middle")
    f.text(kx + 1.0, py + 9.0, "−1", FS["small"], c=PAL["mute"], a="middle")
    f.text(kx + 10.0, py + 9.0, "+1", FS["small"], c=PAL["mute"], a="middle")
    f.text(kx, py + 13.0, ["arrows: unit", "flow in xz"], FS["small"], c=PAL["mute"])
    f.prov_add("comparison on GT nucleus 62, xz section at native y = 126 of E15_Ab2, native z planes: 2D control as evaluated (Cellpose 3D: xy, xz, yz flows fused) vs S1 as evaluated (3 views) vs GT-derived 3D target",
               "make_a3_assets.py (thr_eval dP npz of runs 81116913 / 07992b55; gtflow_E15_Ab2.npz = flows3d_gpu of S1 data3d.py, nc-viz); main.tex Methods 'Inference' for the 2D control's Cellpose 3D inference")


# ======================================================================== panel c: training target
def panel_target(f, ox, oy, w, letter="c"):
    f.plabel(ox, oy + 3.0, letter, "Training target")
    t = 15.0
    y1 = oy + 10.5
    f.text(ox + 1.0 + t / 2, y1 - 1.3, "GT labels", FS["small"], c=PAL["mute"], a="middle")
    f.image(ox + 1.0, y1, t, t, "train_gt_xz.png", fit="xMidYMid slice")
    f.rect(ox + 1.0, y1, t, t, rx=0, stroke=PAL["box"], sw=0.6 * LW)
    xa = ox + 1.0 + t
    tf = 18.0
    tx = xa + 8.0
    f.arrow(xa + 0.6, y1 + t / 2, tx - 0.6, y1 + t / 2)
    f.text(xa + 4.0, y1 + t / 2 - 1.2, "diffuse", FS["small"], c=PAL["mute"], a="middle")
    f.text(tx + tf / 2, y1 - 1.3, "3D flow *u* + fg", FS["small"], a="middle")
    f.image(tx, y1 + 0.2, tf, tf, "target_xz.png", fit="xMidYMid slice")
    f.text(ox + 1.0, y1 + t + 5.6, "loss ½ MSE(flows, 5*u*) + BCE(p, fg)", FS["small"])
    f.text(ox + 1.0, y1 + t + 8.1, "910 volumes, 32 × 256 × 256 crops", FS["small"], c=PAL["mute"])
    f.prov_add("3D Cellpose target by diffusion inside each nucleus (flows3d_gpu), fg channel; loss MSE/2 on 5u + BCE; 910 training volumes; 32x256x256 patches",
               "data3d.py:28-81, 326, 334; train3d.py:33-43, 218; train log '训练体积 910 个'; make_assets.py")


# ======================================================================== panel d: inference strip
def panel_infer(f, ox, oy, w, letter="d"):
    f.plabel(ox, oy + 3.0, letter, "Inference")
    n = 7
    gap = 3.4
    bw = (w - 1.0 - (n - 1) * gap) / n
    by, bh = oy + 6.0, 21.0
    xs = [ox + 0.5 + i * (bw + gap) for i in range(n)]
    heads = ["1  diameter", "2  isotropic", "3  z windows", "4  three views", "5  follow flow", "6  rule T", "7  3D nuclei"]
    for i, x in enumerate(xs):
        f.rect(x, by, bw, bh, rx=1.4, fill="#ffffff", stroke=PAL["box"], sw=0.8 * LW)
        f.text(x + 1.4, by + 3.4, heads[i], FS["body"], 700)
        if i < n - 1:
            f.arrow(x + bw + 0.4, by + bh / 2, x + bw + gap - 0.4, by + bh / 2)
    # 1 diameter: 7 candidate scales
    x = xs[0]
    cxp = x + 1.6
    for dd in [6, 10, 17, 30, 52, 90, 150]:
        rr = 0.35 + 0.42 * math.log(dd / 6.0)
        cxp += rr
        f.circle(cxp, by + 9.6, rr, fill="#ffffff", stroke=PAL["ctl"], sw=0.7 * LW, chk=False)
        cxp += rr + 0.45
    f.text(x + bw / 2, by + 15.6, ["2D control at 7", "scales; most stable"], FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("diameter: 2D control at scales 6-150 px, argmax stability - inconsistency at cp 0, then refine to d_est", "eval_val.py:160-199 (joint, r2a2); run 4a362b0f config.json")
    # 2 isotropic
    x = xs[1]
    tt = 9.4
    f.image(x + 1.2, by + 5.2, tt, tt, "xz_native.png", fit="xMidYMid slice")
    f.image(x + bw - 1.2 - tt, by + 5.2, tt, tt, "xz_iso.png", fit="xMidYMid slice")
    f.text(x + bw / 2, by + 18.4, "rescale to d = 30", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("rescale by 30/d; z/xy >= 2 -> linear z interpolation to isotropic", "infer3d.py:211-217; eval_gt.py:217-222")
    # 3 windows (real geometry for this volume)
    x = xs[2]
    st, Zp = LOGA["windows"]["starts"], LOGA["windows"]["Z_network"]
    a0, a1 = x + 1.2, x + bw - 1.2
    base, amp = by + 13.6, 6.0
    sc = (a1 - a0) / Zp
    for i, s0 in enumerate(st):
        pts = [(a0 + (s0 + k + 0.5) * sc, base - amp * math.sin(math.pi * (k + 0.5) / 32) ** 2) for k in range(32)]
        f.add('<path d="M' + " L".join(f"{p:.2f},{q:.2f}" for p, q in pts) + f'" stroke="{PAL["new"] if i % 2 == 0 else PAL["new_l"]}" '
              f'stroke-width="{0.7 * LW:.3f}" fill="none"/>')
    f.add(f'<line x1="{a0:.2f}" y1="{base:.2f}" x2="{a1:.2f}" y2="{base:.2f}" stroke="{INK}" stroke-width="{0.6 * LW:.3f}"/>')
    f.text(x + bw / 2, by + 17.0, "32 planes, sin^{2}", FS["small"], c=PAL["mute"], a="middle")
    f.text(x + bw / 2, by + 19.4, "weights", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("18 windows of 32 planes over Z' = 426 for this volume, sin^2 taper", "infer3d.py:56-60, 129-132, 161-186; assets_log.json")
    # 4 three views
    x = xs[3]
    for j in range(3):
        cx, cyb, cw = x + 1.4 + j * 6.9, by + 7.8, 3.9
        dxp, dyp = 2.2 * DEP[0], 2.2 * DEP[1]
        f.cuboid(cx, cyb, cw, cw, 2.2, PAL["inh_f"], tint(PAL["inh"], 0.08), PAL["inh_f2"], stroke=PAL["inh"], sw=0.5 * LW)
        for q in (0.35, 0.7):
            if j == 0:
                a = (cx + q * dxp, cyb + q * dyp)
                pts = [a, (a[0] + cw, a[1]), (a[0] + cw, a[1] + cw), (a[0], a[1] + cw)]
            elif j == 1:
                yy = cyb + q * cw
                pts = [(cx, yy), (cx + dxp, yy + dyp), (cx + cw + dxp, yy + dyp), (cx + cw, yy)]
            else:
                xx = cx + q * cw
                pts = [(xx, cyb), (xx + dxp, cyb + dyp), (xx + dxp, cyb + cw + dyp), (xx, cyb + cw)]
            f.poly(pts, fill=PAL["new"], stroke=PAL["new"], sw=0.3 * LW, op=0.4, chk=False)
    f.text(x + bw / 2, by + 16.6, "each axis as z;", FS["small"], c=PAL["mute"], a="middle")
    f.text(x + bw / 2, by + 19.0, "average", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("three views (each axis as network z), averaged; single view for Cellos, Parhyale, mesoSPIM", "infer3d.py:63-96; S1 run config.json --views 3 --views1_ds")
    # 5 flow following (real render)
    x = xs[4]
    f.image(x + 0.8, by + 4.4, bw - 1.6, bh - 5.0, "flowfollow_S1.png")
    f.prov_add("flow following render: S1 seed-0 on E15_Ab2 nuclei GT 33, 51, 68, 50 (one sink each)", "assets_viz/F1-a2_flow_following_S1.provenance.json (nc-viz); infer3d.py:258-260")
    # 6 rule T
    x = xs[5]
    f.text(x + 1.4, by + 8.0, ["lower the", "threshold t if", "foreground is too", "small for the", "expected volume"], FS["small"], c=PAL["mute"])
    f.prov_add("rule T: t = -2 if m(-2) >= 0.7, else highest t in {-3,-4,-6} with m(t) >= 0.85, else -6; then remove objects < 0.15 V", "sizeThr_dir.py:4,28; thr_eval.py; eval_gt.py:235,248-252")
    # 7 masks (real render)
    x = xs[6]
    f.image(x + 0.6, by + 4.2, bw - 1.2, bh - 4.6, "masks_S1.png")
    f.prov_add("3D nuclei: S1 seed-0 final masks of E15_Ab2", "render3d/out/fig1/E15_Ab2_S1_iso.png + E15_Ab2_iso.provenance.json (nc-figures)")


# ======================================================================== layouts
def layout_a():
    f = Fig(183.0, 170.0)
    panel_arch(f, 0.0, 0.0)
    panel_cmp(f, 0.0, 102.5, 120.0)
    panel_target(f, 123.0, 102.5, 60.0)
    panel_infer(f, 0.0, 141.0, 183.0)
    return f


def layout_b():
    """Variant b: comparison first (the question), then the architecture that answers it, insets side by side at the bottom."""
    f = Fig(183.0, 170.0)
    panel_cmp(f, 0.0, 0.0, 120.0, letter="a")
    panel_target(f, 123.0, 0.0, 60.0, letter="b")
    panel_arch(f, 0.0, 39.0, insets="none", letter="c")
    inset_dil(f, 1.0, 39.0 + 58.5, 90.0, 41.0)
    inset_zero(f, 94.0, 39.0 + 58.5, 88.0, 41.0)
    panel_infer(f, 0.0, 141.5, 183.0)
    return f


# ======================================================================== A3-c additions
def inset_zero_schem(f, x, y, w, h):
    """Zero-init inset without data panes (the data pair is enlarged in panel b of A3-c)."""
    f.rect(x, y, w, h, rx=1.6, fill=tint(PAL["new"], 0.05), stroke=PAL["new_l"], sw=0.8 * LW)
    f.text(x + 2.0, y + 3.8, "Zero-initialised residual", FS["mod"], 700, c=PAL["new_d"])
    f.text(x + 2.0, y + 6.6, "S1 starts exactly as the 2D model", FS["small"], c=PAL["mute"])
    cols = [(x + 6.0, "step 0", PAL["fg"]), (x + w / 2 + 4.0, "after training", PAL["new_d"])]
    chans = ["d_{z}", "d_{y}", "d_{x}", "p"]
    by0, bh_, bw_ = y + 12.6, 5.2, 7.0
    for ci, (cx, ttl, col) in enumerate(cols):
        f.text(cx + bw_ + 3.0, y + 11.0, ttl, FS["body"], 700, c=col, a="middle")
        for k, ch in enumerate(chans):
            yy = by0 + k * (bh_ + 1.2)
            f.text(cx - 0.8, yy + bh_ / 2 + 0.8, ch, FS["small"], a="end")
            if k == 0:
                if ci == 0:
                    f.rect(cx, yy, 2 * bw_ + 2.6, bh_, rx=0.6, fill="#ffffff", stroke=PAL["new"], sw=0.8 * LW, dash="0.8 0.5")
                    f.text(cx + bw_ + 1.3, yy + bh_ / 2 + 0.8, "0", FS["small"], 700, c=PAL["new"], a="middle")
                else:
                    f.rect(cx, yy, 2 * bw_ + 2.6, bh_, rx=0.6, fill=PAL["new_f"], stroke=PAL["new"], sw=0.8 * LW)
                    f.text(cx + bw_ + 1.3, yy + bh_ / 2 + 0.8, "Δ_{z}", FS["small"], 700, c=PAL["new_d"], a="middle")
            else:
                f.rect(cx, yy, bw_, bh_, rx=0.6, fill=PAL["inh_f"], stroke=PAL["inh"], sw=0.8 * LW)
                f.text(cx + bw_ / 2, yy + bh_ / 2 + 0.8, "2D", FS["small"], a="middle")
                f.text(cx + bw_ + 1.3, yy + bh_ / 2 + 0.8, "+", FS["small"], a="middle")
                if ci == 0:
                    f.rect(cx + bw_ + 2.6, yy, bw_, bh_, rx=0.6, fill="#ffffff", stroke=PAL["new"], sw=0.8 * LW, dash="0.8 0.5")
                    f.text(cx + 1.5 * bw_ + 2.6, yy + bh_ / 2 + 0.8, "0", FS["small"], 700, c=PAL["new"], a="middle")
                else:
                    f.rect(cx + bw_ + 2.6, yy, bw_, bh_, rx=0.6, fill=PAL["new_f"], stroke=PAL["new"], sw=0.8 * LW)
                    f.text(cx + 1.5 * bw_ + 2.6, yy + bh_ / 2 + 0.8, "Δ", FS["small"], 700, c=PAL["new_d"], a="middle")
    f.arrow(cols[0][0] + 2 * bw_ + 3.2, by0 + 13.0, cols[1][0] - 4.4, by0 + 13.0)
    f.text(x + 2.0, y + h - 1.8, "real example: panel b, left", FS["small"], c=PAL["mute"])
    f.prov_add("zero-init: output = (Delta_z, 2D head + Delta_yxp); Delta = 0 at step 0 so S1 = seed-matched 2D model (2D control) with dz = 0",
               "net3d.py:89-90, 294; model3d.json init orx_v5_2d_s0; net3d.py:359-371 self-test")


def panel_cmp2(f, ox, oy, w, letter="b"):
    f.plabel(ox, oy + 3.0, letter, "How z information enters the flow (xz section through one nucleus)")
    pw, ph = 22.5, 16.7
    py = oy + 13.6
    # left: same single view, step 0 (= 2D model) vs S1
    xa, xb = ox + 1.5, ox + 1.5 + pw + 6.5
    f.text(xa + pw / 2, oy + 8.0, "2D model, single xy view", FS["small"], 700, c=PAL["ctl_d"], a="middle")
    f.text(xa + pw / 2, oy + 10.4, "d_{z} ≡ 0", FS["small"], c=PAL["mute"], a="middle")
    f.text(xb + pw / 2, oy + 8.0, "S1, same single view", FS["small"], 700, c=PAL["new_d"], a="middle")
    f.text(xb + pw / 2, oy + 10.4, "d_{z} predicted", FS["small"], c=PAL["mute"], a="middle")
    f.image(xa, py, pw, ph, "../assets3/cmp_2d_xyonly.png")
    f.image(xb, py, pw, ph, "../assets3/cmp_s1_view1.png")
    f.arrow(xa + pw + 0.8, py + ph / 2, xb - 0.8, py + ph / 2, stroke=PAL["new"])
    cbx = xa + 3.0
    f.colorbar(cbx + 4.0, py + ph + 2.2, 9.0, 1.2, "", "", None)
    f.text(cbx + 2.6, py + ph + 3.3, "u_{z}  −1", FS["small"], c=PAL["mute"], a="end")
    f.text(cbx + 14.0, py + ph + 3.3, "+1", FS["small"], c=PAL["mute"])
    # divider
    dvx = xb + pw + 3.5
    f.add(f'<line x1="{dvx:.2f}" y1="{oy + 6.5:.2f}" x2="{dvx:.2f}" y2="{oy + 36.0:.2f}" stroke="{PAL["rule"]}" stroke-width="{0.8 * LW:.3f}"/>')
    # right: angle to the 3D target, as evaluated
    med = LOG3["angle_nucleus62"]["median_deg"]
    xc, xd = dvx + 3.5, dvx + 3.5 + pw + 3.0
    f.text((xc + xd + pw) / 2, oy + 8.0, "angle to the 3D target, as evaluated", FS["small"], 700, a="middle")
    f.text(xc + pw / 2, oy + 10.4, "2D control (3 views fused)", FS["small"], c=PAL["ctl_d"], a="middle")
    f.text(xd + pw / 2, oy + 10.4, "S1", FS["small"], c=PAL["new_d"], a="middle")
    f.image(xc, py, pw, ph, "../assets3/ang_2d.png")
    f.image(xd, py, pw, ph, "../assets3/ang_s1.png")
    f.text(xc + pw / 2, py + ph + 2.6, f"median {med['2D']:.0f}°", FS["small"], c=PAL["mute"], a="middle")
    f.text(xd + pw / 2, py + ph + 2.6, f"median {med['S1']:.0f}°", FS["small"], c=PAL["mute"], a="middle")
    # angle colour bar 0-90
    gid = "angbar"
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in ((0, "#ffffff"), (0.33, PAL_M["cellos"]), (0.67, PAL_M["S1"]), (1, PAL_M["S1_dark"])))
    f.defs.append(f'<linearGradient id="{gid}" x1="0" x2="0" y1="1" y2="0">{stops}</linearGradient>')
    bx = xd + pw + 2.5
    f.add(f'<rect x="{bx:.2f}" y="{py:.2f}" width="1.3" height="{ph:.2f}" fill="url(#{gid})" stroke="{PAL["box"]}" stroke-width="{0.5 * LW:.3f}"/>')
    f.text(bx + 2.0, py + 1.8, "90°", FS["small"], c=PAL["mute"])
    f.text(bx + 2.0, py + ph, "0°", FS["small"], c=PAL["mute"])
    f.text(ox + w - 0.5, py + ph + 6.6, "all 12 datasets, 5 seeds: S1 − 2D median angle −3.18° (−3.41 to −2.89)",
           FS["small"], 700, c=PAL["new_d"], a="end")
    f.prov_add("left pair: 2D control on xy planes only (u_z = 0 by construction) vs S1 single identity view on the same planes, nucleus 62",
               "make_a3_assets.py cmp_2d_xyonly / cmp_s1_view1 (v2v4_crop.npz c2d_out, s1_out; nc-viz viz_run.py)")
    f.prov_add(f"right pair: per-voxel 3D angle to the GT-derived target (FLOWERR definition, region GT fg & |t|>0.5) for the 2D control as evaluated (3 views fused) and S1 as evaluated, nucleus 62: medians {med['2D']:.1f} / {med['S1']:.1f} deg (n = {LOG3['angle_nucleus62']['n_vox']} voxels)",
               "make_a3_assets.py angle section; definition flowerr_angles.py; dP from thr_eval npz runs 81116913 / 07992b55")
    f.prov_add("macro over 12 datasets, five seeds: S1 - 2D difference in median angle -3.18 deg (-3.41 to -2.89); macro medians S1/2D 12.4/15.6 deg; post hoc, S1 trained on this target",
               "NUMBERS.md:631 (primary.set12_region_a); main.tex Fig. 3 caption f; source_data/inputs/flowerr_results.json")


def layout_c():
    """A3-a with panel b replaced: same-view pair (d_z absent vs predicted) + angle to the 3D target; zero-init inset as schematic."""
    f = Fig(183.0, 170.0)
    panel_arch(f, 0.0, 0.0, insets="none")
    inset_dil(f, 1.0, 58.5, 118.0, 41.0, spread=28.0)
    inset_zero_schem(f, 122.0, 58.5, 60.0, 41.0)
    panel_cmp2(f, 0.0, 102.5, 131.0)
    panel_target(f, 133.0, 102.5, 50.0)
    panel_infer(f, 0.0, 141.0, 183.0)
    return f


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for cid, fn in (("A3-a", layout_a), ("A3-b", layout_b), ("A3-c", layout_c)):
        if which not in ("all", cid[-1]):
            continue
        f = fn()
        res = export(f, cid)
        print(cid, "issues:", len(res.get("issues", [])), "min font", res.get("min_font_pt"))
        for s_ in res.get("issues", [])[:60]:
            print("   ", s_)
        json.dump(dict(figure=cid, code_snapshot=_nsp_os.path.join(_NSP_REPO, 'src'),
                       model="orx_t3d_r5s1_s0/model3d.json", palette="paper/figures/palette.json", assets3_log=LOG3, items=f.prov),
                  open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
