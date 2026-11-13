"""Fig. 1 whole-process schematic (gallery A2-a / A2-b): hand-coded SVG, exported with headless Chrome to vector PDF
and 600-dpi PNG. Every architectural or procedural fact drawn here is listed with its code location in
fig1_arch.provenance.json (written by this script). Raster thumbnails come from make_assets.py (real data).

Usage: python gen_fig.py [a|b|all]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import base64
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AS = os.path.join(HERE, "assets")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LOGA = json.load(open(os.path.join(AS, "assets_log.json")))

# ---------------------------------------------------------------- palette: every colour read from paper/figures/palette.json
P = json.load(open(os.path.abspath(os.path.join(HERE, "..", "palette.json"))))
M = P["models"]


def tint(hexc, a):
    """hexc at opacity a over white (pastel module fills, as palette.json advises 15-25 %)."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % tuple(round(255 - a * (255 - v)) for v in (r, g, b))


LOSS = P["source"].split(":")[1].split()[8]          # npg #7E6148 (palette source list)
PAL = dict(new=M["S1"], new_d=M["S1_dark"], new_f=tint(M["S1"], 0.16), new_l=tint(M["S1"], 0.45),
           ctl=M["2D"], ctl_d=M["2D_dark"], ctl_f=tint(M["2D"], 0.16),
           inh=M["base"], inh_f=tint(M["base"], 0.16), inh_f2=tint(M["base"], 0.32), inh_f3=tint(M["base"], 0.24),
           fg=P["text"], mute=P["muted"], rule=P["rule"], box=tint(P["muted"], 0.40),
           data_f=P["tile_light"][0], loss=LOSS, loss_f=tint(LOSS, 0.12), div=P["diverging"])

PT = 0.3528                     # mm per point
LW = 0.5 * PT                   # 0.5 pt lines
FS = dict(title=7.0, mod=6.2, body=5.5, small=5.0, tag=5.0)


def b64(name):
    p = os.path.join(AS, name)
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


MIN_PT = 5.0                    # Nature Communications minimum at final 183 mm width


def mark(s, size_pt=5.0):
    """Mini markup: x_{ab} -> subscript, x^{ab} -> superscript, *bold*. Sub/superscripts are max(5 pt, 0.8 x base)."""
    s = esc(s)
    ss = max(MIN_PT, 0.8 * size_pt) * 0.3528
    s = re.sub(r"_\{([^}]*)\}", rf'<tspan baseline-shift="sub" font-size="{ss:.3f}">\1</tspan>', s)
    s = re.sub(r"\^\{([^}]*)\}", rf'<tspan baseline-shift="super" font-size="{ss:.3f}">\1</tspan>', s)
    s = re.sub(r"\*([^*]+)\*", r'<tspan font-weight="700">\1</tspan>', s)
    return s


class Fig:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.el = []
        self.defs = []
        self.prov = []
        self._mk = set()

    # ---- primitives
    def add(self, s):
        self.el.append(s)

    def text(self, x, y, s, size=FS["body"], w=400, c=None, a="start", italic=False, cls="t", ls=1.18, halo=False):
        c = c or PAL["fg"]
        lines = s if isinstance(s, (list, tuple)) else [s]
        fs = size * PT
        out = []
        for i, ln in enumerate(lines):
            st = ' font-style="italic"' if italic else ""
            if halo:
                st += f' stroke="#ffffff" stroke-width="{0.7:.2f}" stroke-linejoin="round" paint-order="stroke"' 
            out.append(f'<text class="{cls}" x="{x:.2f}" y="{y + i * fs * ls:.2f}" font-size="{fs:.3f}" font-weight="{w}" '
                       f'fill="{c}" text-anchor="{a}"{st}>{mark(ln, size)}</text>')
        self.add("".join(out))
        return y + (len(lines) - 1) * fs * ls

    def rect(self, x, y, w, h, rx=1.2, fill="none", stroke=None, sw=LW, dash=None, op=1.0, chk=True):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        cl = ' class="chk"' if chk else ""
        st = stroke or "none"
        self.add(f'<rect{cl} x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{st}" stroke-width="{sw:.3f}"{d} opacity="{op}"/>')

    def poly(self, pts, fill="none", stroke=None, sw=LW, op=1.0, chk=True, dash=None):
        p = " ".join(f"{a:.2f},{b:.2f}" for a, b in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        cl = ' class="chk"' if chk else ""
        self.add(f'<polygon{cl} points="{p}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw:.3f}" '
                 f'stroke-linejoin="round" opacity="{op}"{d}/>')

    def path(self, d, stroke=None, sw=LW, fill="none", arrow=True, dash=None, chk=True, op=1.0, head="ah"):
        st = stroke or PAL["fg"]
        m = f' marker-end="url(#{head}{st[1:]})"' if arrow else ""
        if arrow:
            self.marker(st)
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        cl = ' class="chk ln"' if chk else ""
        self.add(f'<path{cl} d="{d}" stroke="{st}" stroke-width="{sw:.3f}" fill="{fill}"{m}{ds} opacity="{op}" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')

    def arrow(self, x0, y0, x1, y1, **k):
        self.path(f"M{x0:.2f},{y0:.2f} L{x1:.2f},{y1:.2f}", **k)

    def marker(self, col):
        if col in self._mk:
            return
        self._mk.add(col)
        self.defs.append(f'<marker id="ah{col[1:]}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="1.9" markerHeight="1.9" '
                         f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1.2 L9.4,5 L0,8.8 L2.4,5 z" fill="{col}"/></marker>')

    def image(self, x, y, w, h, name, fit="xMidYMid meet", op=1.0, transform=None, clip=None, cls="im"):
        tr = f' transform="{transform}"' if transform else ""
        cp = f' clip-path="url(#{clip})"' if clip else ""
        self.add(f'<image class="{cls}" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" preserveAspectRatio="{fit}" '
                 f'href="{b64(name)}" opacity="{op}"{tr}{cp}/>')

    def circle(self, cx, cy, r, fill="none", stroke=None, sw=LW, chk=True, op=1.0):
        cl = ' class="chk"' if chk else ""
        self.add(f'<circle{cl} cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke or "none"}" '
                 f'stroke-width="{sw:.3f}" opacity="{op}"/>')

    def plabel(self, x, y, letter, title):
        self.text(x, y, letter, 8.0, 700)
        self.text(x + 4.0, y, title, FS["title"], 700)

    def prov_add(self, what, src):
        self.prov.append(dict(shown=what, source=src))

    # ---- composite glyphs
    def plus_node(self, cx, cy, r=1.7, col=None):
        col = col or PAL["fg"]
        self.circle(cx, cy, r, fill="#ffffff", stroke=col, sw=1.6 * LW)
        self.add(f'<path d="M{cx - r * 0.55:.2f},{cy:.2f} H{cx + r * 0.55:.2f} M{cx:.2f},{cy - r * 0.55:.2f} V{cy + r * 0.55:.2f}" '
                 f'stroke="{col}" stroke-width="{1.6 * LW:.3f}"/>')

    def trapezoid(self, x, y, w, h, shrink, fill, stroke, op=1.0, expand=False, chk=True):
        d = h * shrink / 2
        if not expand:
            pts = [(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)]
        else:
            pts = [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)]
        self.poly(pts, fill, stroke, sw=1.4 * LW, op=op, chk=chk)

    def cuboid(self, x, y, w, h, dp, ftop, ffront, fside, stroke="#ffffff", sw=LW, chk=False, op=1.0):
        dx, dy = dp * math.cos(math.radians(35)), -dp * math.sin(math.radians(35))
        self.poly([(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w, y)], ftop, stroke, sw, op=op, chk=chk)
        self.poly([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], fside, stroke, sw, op=op, chk=chk)
        self.poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], ffront, stroke, sw, op=op, chk=chk)

    def colorbar(self, x, y, w, h, lab_l, lab_r, title):
        gid = f"cb{len(self.defs)}"
        stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in
                        ((0, PAL["div"][0]), (0.5, PAL["div"][1]), (1, PAL["div"][2])))
        self.defs.append(f'<linearGradient id="{gid}" x1="0" x2="1">{stops}</linearGradient>')
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="url(#{gid})" stroke="{PAL["rule"]}" '
                 f'stroke-width="{0.6 * LW:.3f}"/>')
        self.text(x - 0.7, y + h * 0.5 + 0.65, lab_l, FS["small"], c=PAL["mute"], a="end")
        self.text(x + w + 0.7, y + h * 0.5 + 0.65, lab_r, FS["small"], c=PAL["mute"])
        if title:
            self.text(x + w + 3.2, y + h * 0.5 + 0.65, title, FS["small"], c=PAL["mute"])

    # ---- output
    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" id="fig" viewBox="0 0 {self.W} {self.H}" width="{self.W}mm" '
                f'height="{self.H}mm" font-family="Helvetica, Arial, sans-serif"><defs>{"".join(self.defs)}</defs>'
                f'<rect x="0" y="0" width="{self.W}" height="{self.H}" fill="#ffffff"/>{"".join(self.el)}</svg>')


# ======================================================================== panels
def panel_a(f, ox, oy, w=57.0):
    """Training data and targets."""
    f.plabel(ox, oy + 3.0, "a", "Training data and 3D targets")
    f.text(ox + 4.0, oy + 6.6, "910 training volumes from 13 collections", FS["small"], c=PAL["mute"])
    f.prov_add("910 training volumes, 13 collections (12 benchmark datasets without mesoSPIM, plus rat brain)",
               "train3d.py:225-228 log line '训练体积 910 个' and dataset shares (A1/C/X1 run logs on server); data3d.make_loader exclude_ds=('mesospim',)")
    # row 1: native -> isotropic -> crop
    y1 = oy + 12.0
    t = 15.0
    f.text(ox + 1.0 + t / 2, y1 - 1.3, "source grid", FS["small"], c=PAL["mute"], a="middle")
    f.image(ox + 1.0, y1, t, t, "xz_native.png", fit="xMidYMid slice")
    f.rect(ox + 1.0, y1, t, t, rx=0, stroke=PAL["rule"], sw=0.6 * LW)
    f.text(ox + 1.0 + t / 2, y1 + t + 2.4, ["xz plane,", "z/xy = 4.65"], FS["small"], c=PAL["mute"], a="middle")
    x2 = ox + 1.0 + t + 9.0
    f.arrow(ox + 1.6 + t, y1 + t / 2, x2 - 0.8, y1 + t / 2, sw=LW)
    f.text(ox + 1.0 + t + 4.5, y1 + t / 2 - 1.2, "resample", FS["small"], c=PAL["mute"], a="middle")
    f.text(x2 + t / 2, y1 - 1.3, "isotropic", FS["small"], c=PAL["mute"], a="middle")
    f.image(x2, y1, t, t, "xz_iso.png", fit="xMidYMid slice")
    f.rect(x2, y1, t, t, rx=0, stroke=PAL["rule"], sw=0.6 * LW)
    f.text(x2 + t / 2, y1 + t + 2.4, ["median nuclear", "diameter 30 px"], FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("training volumes resampled to isotropic voxels with median nuclear diameter 30 px (iso30 store)",
               f"data3d.py:21 ISO_DIR={_NSP_SCRATCH}/cpft_iso30; main.tex Methods '3D model' first sentence; aniso_params docstring data3d.py:142-145 ('factors' = interpolation factors of the iso30 store)")
    # crop cuboid
    cx = x2 + t + 4.0
    f.cuboid(cx, y1 + 4.0, 7.0, 7.0, 3.6, PAL["inh_f2"], PAL["inh_f"], PAL["inh_f2"], stroke=PAL["inh"], sw=0.7 * LW)
    f.image(cx, y1 + 4.0, 7.0, 7.0, "train_img_xy.png", fit="xMidYMid slice")
    f.rect(cx, y1 + 4.0, 7.0, 7.0, rx=0, stroke=PAL["inh"], sw=0.7 * LW, chk=False)
    f.text(cx + 5.0, y1 - 1.3, "crop", FS["small"], c=PAL["mute"], a="middle")
    f.text(cx + 5.0, y1 + t + 2.4, ["32 × 256", "× 256"], FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("training patch 32 x 256 x 256", "train3d.py:142 --patch_z 32; train3d.py:218 patch = (a.patch_z, 256, 256); A1/C/X1 model3d.json train.patch_z = 32")
    f.text(ox + 1.0, y1 + t + 9.6, "augment: flips, 90° rotations, scale 0.8–1.25, gain, gamma",
           FS["small"], c=PAL["mute"])
    f.prov_add("augmentation flips, xy rot90, scale 0.8-1.25, gain 0.75-1.25, gamma 0.7-1.4", "data3d.py:130 scale=(0.8,1.25); data3d.py:214-216")
    # row 2: GT labels -> diffusion -> 3D flow target + foreground
    y2 = oy + 40.5
    t2 = 15.0
    f.text(ox + 1.0 + t2 / 2, y2 - 1.3, "GT labels", FS["small"], c=PAL["mute"], a="middle")
    f.image(ox + 1.0, y2, t2, t2, "train_gt_xz.png", fit="xMidYMid slice")
    f.rect(ox + 1.0, y2, t2, t2, rx=0, stroke=PAL["rule"], sw=0.6 * LW)
    xa = ox + 1.0 + t2
    tx = xa + 9.0
    f.arrow(xa + 0.6, y2 + t2 / 2, tx - 0.8, y2 + t2 / 2, sw=LW)
    f.text(xa + 4.5, y2 + t2 / 2 - 4.0, ["heat", "diffusion"], FS["small"], c=PAL["mute"], a="middle")
    f.text(xa + 4.5, y2 + t2 / 2 + 3.4, ["per", "nucleus"], FS["small"], c=PAL["mute"], a="middle")
    tf = 19.0
    f.text(tx + tf / 2, y2 - 1.3, "3D flow target *u*", FS["small"], c=PAL["fg"], a="middle")
    f.image(tx, y2 + 0.2, tf, tf, "target_xz.png", fit="xMidYMid slice")
    f.prov_add("3D Cellpose flow target by diffusion inside each nucleus; unit vectors (dz, dy, dx); shown: real target computed with the project's own flows3d_gpu on mouse-embryo E15_Ab2 GT",
               "data3d.py:28-81 flows3d_gpu; data3d.py:326 mu = flows3d_gpu(lb); make_assets.py")
    xp = tx + tf + 1.5
    tp = 8.0
    f.text(xp + tp / 2, y2 - 1.3, "fg", FS["small"], c=PAL["fg"], a="middle")
    f.image(xp, y2 + 2.0, tp, tp, "target_fg_xz.png", fit="xMidYMid slice")
    f.rect(xp, y2 + 2.0, tp, tp, rx=0, stroke=PAL["rule"], sw=0.6 * LW)
    f.prov_add("4th target channel = nucleus or not (foreground)", "data3d.py:334 tgt = cat([mu, (lb > 0)])")
    f.colorbar(tx + 3.0, y2 + tf + 1.6, 11.0, 1.3, "−", "+", "d_{z}")
    f.text(ox + 1.0, y2 + t2 + 4.6, ["xz planes of one", "training crop"], FS["small"], c=PAL["mute"])


def panel_b(f, ox, oy, w=124.0):
    """Network."""
    f.plabel(ox, oy + 3.0, "b", "Network: 2D encoder per plane, native 3D decoder")
    lx = ox + w - 49.0
    f.rect(lx, oy + 0.2, 4.6, 3.4, rx=0.8, fill=PAL["inh_f"], stroke=PAL["inh"], sw=1.0 * LW)
    f.text(lx + 5.8, oy + 2.9, "inherited (2D, fine-tuned)", FS["body"], c=PAL["fg"])
    f.rect(lx + 33.0, oy + 0.2, 4.6, 3.4, rx=0.8, fill=PAL["new_f"], stroke=PAL["new"], sw=1.0 * LW)
    f.text(lx + 38.8, oy + 2.9, "new in 3D", FS["body"], c=PAL["fg"])

    ym = oy + 25.0
    # ---- input plane stack (real planes, sheared)
    sx0, s = ox + 1.0, 11.0
    for i in range(5):
        k = 4 - i
        dx, dy = k * 1.5, -k * 1.5
        tr = f"translate({sx0 + dx:.2f},{ym - s / 2 + dy + 1.5:.2f}) matrix(1,0,-0.32,1,{0.32 * s:.2f},0)"
        f.image(0, 0, s, s, f"plane_{4 - k}.png", fit="none", transform=tr, op=1.0 if k == 0 else 0.92)
        f.add(f'<rect x="0" y="0" width="{s}" height="{s}" fill="none" stroke="#ffffff" stroke-width="{0.8 * LW:.3f}" transform="{tr}"/>')
    f.path(f"M{sx0 + 0.6:.2f},{ym + s / 2 + 4.2:.2f} L{sx0 + 5.6:.2f},{ym + s / 2 - 0.8:.2f}", sw=0.7 * LW, stroke=PAL["mute"], chk=False)
    f.text(sx0 + 4.4, ym + s / 2 + 5.8, "z", FS["small"], c=PAL["mute"], italic=True)
    f.text(sx0 + 9.5, oy + 9.6, "input planes", FS["body"], 700, a="middle")
    f.text(sx0 + 9.5, oy + 12.2, "1 × Z × 256 × 256", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("input (B,1,Z,256,256), Z = 32 in training", "net3d.py:268-272 forward assert C==1, H==W==256")

    # ---- encoder with ghosted copies
    ex, ew, eh = ox + 25.0, 15.0, 24.0
    for k in (3, 2, 1):
        f.trapezoid(ex + k * 1.4, ym - eh / 2 - k * 1.4, ew, eh, 0.42, PAL["inh_f"], PAL["inh"], op=0.38, chk=False)
    f.trapezoid(ex, ym - eh / 2, ew, eh, 0.42, PAL["inh_f"], PAL["inh"])
    f.text(ex + ew / 2 - 0.6, ym - 1.6, "ViT-L", FS["mod"], 700, a="middle")
    f.text(ex + ew / 2 - 0.6, ym + 1.0, "encoder", FS["mod"], 700, a="middle")
    f.text(ex + ew / 2 - 0.6, ym + 3.5, "+ neck", FS["small"], c=PAL["mute"], a="middle")
    f.text(ex + ew / 2 + 1.5, oy + 9.6, "one pass per plane,", FS["small"], c=PAL["mute"], a="middle", halo=True)
    f.text(ex + ew / 2 + 1.5, oy + 12.2, "shared weights", FS["small"], c=PAL["mute"], a="middle", halo=True)
    f.text(ex + ew / 2, ym + eh / 2 + 3.0, ["8 × 8 patches,", "304 M params"], FS["small"], c=PAL["mute"], a="middle")
    f.arrow(sx0 + s + 9.6, ym + 1.0, ex - 0.5, ym + 1.0)
    f.prov_add("2D ViT-L encoder (Cellpose-SAM) applied to every z plane (B*Z batch), shared weights; neck to 256 ch; 8x8 patches -> 32x32 tokens",
               "net3d.py:285-286 xs = x.permute(...).reshape(B*Z,1,H,W); f = self.encode(xs) -> (B*Z,256,32,32); net3d.py:237-259")
    f.prov_add("encoder fully fine-tuned (304.4 M trainable encoder-side params)", "model3d.json cfg.freeze_encoder=false, lora=0; train log '可训练参数：编码器侧 304.4M'")

    # ---- plane features -> re-stacked cuboid
    px = ex + ew + 5.0
    for k in range(5):
        f.poly([(px + k * 1.3, ym - 4.5 - k * 1.3), (px + 2.2 + k * 1.3, ym - 6.0 - k * 1.3),
                (px + 2.2 + k * 1.3, ym + 4.0 - k * 1.3), (px + k * 1.3, ym + 5.5 - k * 1.3)],
               fill=PAL["inh_f2"], stroke=PAL["inh"], sw=0.6 * LW, chk=False, op=0.95)
    f.arrow(ex + ew + 0.5, ym, px - 0.5, ym)
    f.text(px + 4.0, ym + 9.0, ["plane", "features"], FS["small"], c=PAL["mute"], a="middle")
    cx0 = px + 12.0
    f.arrow(px + 8.4, ym, cx0 - 0.5, ym)
    f.cuboid(cx0, ym - 4.5, 8.0, 9.0, 6.5, PAL["inh_f3"], PAL["inh_f2"], PAL["inh_f2"], stroke=PAL["inh"], sw=0.6 * LW)
    for k in range(1, 4):
        f.add(f'<line x1="{cx0 + k * 2.0:.2f}" y1="{ym - 4.5:.2f}" x2="{cx0 + k * 2.0:.2f}" y2="{ym + 4.5:.2f}" stroke="{PAL["inh"]}" '
              f'stroke-width="{0.4 * LW:.3f}"/>')
    f.text(cx0 + 5.5, ym + 8.2, ["re-stacked", "256 × Z × 32 × 32"], FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("plane features re-stacked to a 3D tensor (B,256,Z,32,32)", "net3d.py:288 f3 = f.view(B,Z,...).permute(0,2,1,3,4)")

    # ---- 2D head (top)
    hx, hy, hw, hh = ox + 74.0, oy + 6.0, 21.0, 8.6
    f.rect(hx, hy, hw, hh, rx=1.6, fill=PAL["inh_f"], stroke=PAL["inh"], sw=1.2 * LW)
    f.text(hx + hw / 2, hy + 3.5, "2D head", FS["mod"], 700, a="middle")
    f.text(hx + hw / 2, hy + 6.5, "1×1 conv, per plane", FS["small"], c=PAL["mute"], a="middle")
    f.path(f"M{px + 3.0:.2f},{ym - 7.4:.2f} L{px + 3.0:.2f},{hy + hh / 2:.2f} L{hx - 0.5:.2f},{hy + hh / 2:.2f}")
    f.prov_add("2D Cellpose-SAM output head = 1x1 Conv2d(256 -> 3*8*8) on per-plane neck features -> (dy, dx, cp) tokens", "net3d.py:287 tok2d = self.net.out(f)  # (B*Z,192,32,32); cellpose 4 vit.py:122 CPSAM.out = nn.Conv2d(feat_dim, nout*ps**2, kernel_size=1) (server env_torch site-packages); in the 2D model the 8x8 unpatchify is conv_transpose2d with the fixed identity W2 (vit.py:124,153), in net3d F.pixel_shuffle (net3d.py:295)")

    # ---- 3D decoder (new)
    dx0, dw, dh = hx, hw, 23.5
    dy0 = ym - dh / 2 + 4.0
    for k in (2, 1):
        f.trapezoid(dx0 + k * 1.3, dy0 - k * 1.3, dw, dh, 0.30, PAL["new_f"], PAL["new"], op=0.35, chk=False, expand=True)
    f.trapezoid(dx0, dy0, dw, dh, 0.30, PAL["new_f"], PAL["new"], expand=True)
    f.text(dx0 + dw / 2, dy0 + 6.0, "3D residual", FS["mod"], 700, c=PAL["new_d"], a="middle")
    f.text(dx0 + dw / 2, dy0 + 8.6, "decoder", FS["mod"], 700, c=PAL["new_d"], a="middle")
    f.text(dx0 + dw / 2, dy0 + 12.0, ["1×1×1 conv,", "4 residual blocks,", "GN · GELU · 1×1×1"], FS["small"], a="middle")
    f.text(dx0 + dw / 2, dy0 + 19.4, "14.3 M params", FS["small"], c=PAL["mute"], a="middle")
    f.arrow(cx0 + 8.0 + 5.4, ym, dx0 - 0.5, ym)
    f.prov_add("Dec3D: 1x1x1 Conv3d in (256->256), 4 ResBlock3D, GroupNorm(16)-GELU-1x1x1 Conv3d out to 4*64 ch", "net3d.py:81-96; model3d.json dec_ch 256, dec_blocks 4, dec_groups 16")
    f.prov_add("14.3 M new params", "train log '新增模块 14.3M'; main.tex Methods")
    zx = dx0 + dw + 1.6
    f.rect(zx, dy0 + 2.0, 3.0, dh - 4.0, rx=0.8, fill="#ffffff", stroke=PAL["new"], sw=1.0 * LW, dash="0.9 0.6")
    f.text(zx + 1.5, dy0 + dh / 2 + 0.9, "0", FS["mod"], 700, c=PAL["new"], a="middle")
    f.text(zx + 1.5, dy0 + dh + 2.4, ["zero-init", "output"], FS["small"], c=PAL["new"], a="middle")
    f.prov_add("decoder output conv zero-initialised -> untrained model reproduces the 2D prediction with dz = 0", "net3d.py:89-90 nn.init.zeros_(self.out.weight/bias); main.tex Results")

    # ---- sum node and outputs
    sx, sy = zx + 7.6, ym
    f.plus_node(sx, sy)
    f.arrow(zx + 3.4, sy, sx - 1.9, sy, stroke=PAL["new"])
    f.text((zx + 3.4 + sx - 1.9) / 2, sy - 1.0, "Δ", FS["body"], 700, c=PAL["new"], a="middle")
    f.path(f"M{hx + hw + 0.5:.2f},{hy + hh / 2:.2f} L{sx:.2f},{hy + hh / 2:.2f} L{sx:.2f},{sy - 1.9:.2f}")
    f.text(sx - 0.8, hy + hh / 2 + 3.0, "d_{y}, d_{x}, p", FS["small"], c=PAL["mute"], a="end")
    f.prov_add("residual added to 2D head tokens for (dy, dx, cp); dz only from the decoder", "net3d.py:294 tok = cat([r[:, :pp], tok2d + r[:, pp:]])")
    gx, gw = ox + w - 16.5, 16.5
    f.arrow(sx + 1.9, sy, gx + 0.6, sy)
    f.image(gx, ym - 11.0, gw, gw, "flow_glyphs_pal.png")
    f.text(gx + gw / 2, oy + 9.6, "output", FS["body"], 700, a="middle")
    f.text(gx + gw / 2 - 1.0, oy + 12.2, "4 × Z × 256 × 256", FS["small"], c=PAL["mute"], a="middle")
    f.text(gx + gw / 2 - 1.0, ym + 8.2, "(d_{z}, d_{y}, d_{x}, p)", FS["body"], a="middle")
    f.text(gx + gw / 2 - 1.0, ym + 11.9, "×8 pixel shuffle", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("output (B,4,Z,256,256) = [dz, dy, dx, cellprob] after pixel shuffle x8", "net3d.py:295,299 F.pixel_shuffle(tok, self.ps)")
    f.prov_add("flow glyph thumbnail: S1 seed-0 output dP on E15_Ab2 nucleus 90, coloured by d_z with the palette diverging ramp; direction verified: all 312 in-nucleus glyphs point inwards (mean cos to centroid 0.94)",
               "arch/render_glyphs.py (recipe of paper/scripts/render3d/fig1_assets.py:82-113, palette cmap); assets/flow_glyphs_pal.json; data export/flow_mouse_embryo.npz from flow_export.py")

    # ---- loss box
    lw0, lh0 = 23.0, 10.5
    lx0, ly0 = ox + w - lw0 - 0.3, ym + 20.0
    f.rect(lx0, ly0, lw0, lh0, rx=1.4, fill=PAL["loss_f"], stroke=PAL["loss"], sw=0.9 * LW)
    f.text(lx0 + lw0 / 2, ly0 + 3.2, "training loss", FS["body"], 700, c=PAL["loss"], a="middle")
    f.text(lx0 + lw0 / 2, ly0 + 6.1, "½ MSE(flows, 5*u*)", FS["small"], a="middle")
    f.text(lx0 + lw0 / 2, ly0 + 8.8, "+ BCE(p, fg)", FS["small"], a="middle")
    f.prov_add("loss = MSE(net flow, 5 x unit flow)/2 + BCE(cellprob), over voxels inside the volume", "train3d.py:33-43 loss_fn; train3d.py:277")
    f.text(lx0 + lw0 / 2, ly0 + lh0 + 3.4, ["AdamW, lr 10^{−5} encoder,", "2·10^{−4} decoder"], FS["small"], c=PAL["mute"], a="middle", ls=1.9)
    f.prov_add("AdamW lr_enc 1e-5, lr_dec 2e-4, wd 0.05", "train3d.py:144-146,213-215; model3d.json train")

    # ---- inset: residual block with z-dilation
    ix, iy, ih = ox + 1.0, oy + 49.0, 15.0
    iw = lx0 - 3.0 - ix
    f.rect(ix, iy, iw, ih, rx=1.6, fill=tint(PAL["new"], 0.06), stroke=PAL["new_l"], sw=0.8 * LW)
    f.path(f"M{dx0 + 5.0:.2f},{dy0 + dh - 2.5:.2f} L{dx0 + 1.0:.2f},{iy - 0.3:.2f}", stroke=PAL["new_l"], arrow=False, dash="0.8 0.6", chk=False)
    f.text(ix + 1.8, iy + 3.2, "residual block (×4)", FS["body"], 700, c=PAL["new_d"])
    blocks = [("GN", 4.6), ("GELU", 6.4), ("Conv3d 3^{3}", 12.5), ("GN", 4.6), ("GELU", 6.4), ("Conv3d 3^{3}", 12.5)]
    bx, by, bh = ix + 3.0, iy + 6.0, 4.4
    x_first = bx
    for i, (nm, bw) in enumerate(blocks):
        conv = nm.startswith("Conv")
        f.rect(bx, by, bw, bh, rx=0.9, fill=PAL["new_f"] if conv else "#ffffff", stroke=PAL["new"], sw=0.8 * LW)
        f.text(bx + bw / 2, by + bh / 2 + 0.75, nm, FS["small"], 700 if conv else 400, a="middle")
        if i == 2:
            f.text(bx + bw / 2, by + bh + 2.6, "z-dilation d", FS["small"], c=PAL["new"], a="middle")
        bx += bw
        if i < len(blocks) - 1:
            f.arrow(bx + 0.2, by + bh / 2, bx + 2.0, by + bh / 2, sw=0.7 * LW)
            bx += 2.2
    px_ = bx + 3.0
    f.plus_node(px_, by + bh / 2, r=1.3)
    f.arrow(bx + 0.2, by + bh / 2, px_ - 1.5, by + bh / 2, sw=0.7 * LW)
    f.path(f"M{x_first - 1.4:.2f},{by + bh / 2:.2f} L{x_first - 1.4:.2f},{by - 1.3:.2f} L{px_:.2f},{by - 1.3:.2f} L{px_:.2f},{by + bh / 2 - 1.5:.2f}", sw=0.7 * LW)
    f.prov_add("ResBlock3D: x + Conv3d3(GELU(GN(Conv3d3_dil(GELU(GN(x))))))", "net3d.py:69-78")
    gx0 = px_ + 4.5
    f.text(gx0, iy + 3.2, "z taps, d = 1, 2, 4, 8", FS["small"], c=PAL["mute"])
    for j, dd in enumerate((1, 2, 4, 8)):
        cxj = gx0 + 2.0 + j * 4.6
        for q in range(17):
            yy = iy + 4.6 + q * 0.42
            on = (q - 8) in (-dd, 0, dd)
            f.add(f'<rect x="{cxj - 1.3:.2f}" y="{yy:.2f}" width="2.6" height="0.28" fill="{PAL["new"] if on else PAL["rule"]}"/>')
        f.text(cxj, iy + 4.6 + 17 * 0.42 + 2.3, str(dd), FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("first conv of block i dilated along z by 1, 2, 4, 8", "net3d.py:72 dilation=(zdil,1,1); model3d.json dec_zdil [1,2,4,8]")


def panel_c(f, ox, oy, w=183.0):
    """Training schedule."""
    f.plabel(ox, oy + 3.0, "c", "Training schedule (every stage with five seeds)")
    f.text(ox + w - 0.5, oy + 3.0, "seeds 0, 1000, 2000, 3000, 4000; each 3D model starts from the seed-matched 2D model",
           FS["small"], c=PAL["mute"], a="end")
    f.prov_add("five seeds; 3D init = seed-matched 2D model", "models/models/orx_t3d_r5s1_s{0,1000,2000,3000,4000}; model3d.json init orx_v5_2d_s0 for seed 0")
    ym = oy + 15.0
    bh = 12.5

    def ghosts(x, y, bw, hh, fill, stroke, op=0.30):
        for k in (4, 3, 2, 1):
            f.rect(x + k * 0.9, y - k * 0.9, bw, hh, rx=1.6, fill=fill, stroke=stroke, sw=0.7 * LW, op=op, chk=False)

    def stage(x, y, bw, title, lines, fill, stroke, tcol=None, ghost=True):
        if ghost:
            ghosts(x, y, bw, bh, fill, stroke)
        f.rect(x, y, bw, bh, rx=1.6, fill=fill, stroke=stroke, sw=1.1 * LW)
        f.text(x + bw / 2, y + 3.8, title, FS["mod"], 700, c=tcol or PAL["fg"], a="middle")
        f.text(x + bw / 2, y + 7.0, lines, FS["small"], a="middle")

    y0 = ym - bh / 2
    f.rect(ox + 1.0, y0, 19.0, bh, rx=1.6, fill="#ffffff", stroke=PAL["inh"], sw=1.1 * LW)
    f.text(ox + 10.5, y0 + 3.8, ["Original", "Cellpose-SAM"], FS["mod"], 700, a="middle", ls=1.1)
    f.text(ox + 10.5, y0 + 10.2, "released weights", FS["small"], c=PAL["mute"], a="middle")
    x1, w1 = ox + 26.0, 30.0
    f.arrow(ox + 20.5, ym, x1 - 0.5, ym)
    stage(x1, y0, w1, "2D fine-tuning", ["6,056 planes (xy, xz, yz)", "100 epochs × 1,000 planes"], PAL["inh_f"], PAL["inh"])
    f.prov_add("2D fine-tuning: cpsam_v2 on 6,056 2D planes, 100 epochs of 1,000 sampled slices, seeds 0-4000",
               "main.tex Methods '2D fine-tuning'; 3D init 'orx_v5_2d_s0' (model3d.json init)")
    cy = oy + 27.5
    f.path(f"M{x1 + w1 / 2:.2f},{y0 + bh + 0.3:.2f} L{x1 + w1 / 2:.2f},{cy - 2.5:.2f}", stroke=PAL["ctl"])
    f.rect(x1 - 1.5, cy - 2.0, w1 + 3.0, 6.0, rx=3.0, fill=PAL["ctl_f"], stroke=PAL["ctl"], sw=1.1 * LW)
    f.text(x1 + w1 / 2, cy + 1.9, "*2D control*: Cellpose 3D inference", FS["small"], c=PAL["ctl_d"], a="middle")
    f.prov_add("2D control evaluated with Cellpose's 3D inference (xy, xz, yz combined)", "main.tex Methods 'Inference'")
    x2, w2 = x1 + w1 + 8.0, 31.0
    f.arrow(x1 + w1 + 0.5, ym, x2 - 0.5, ym)
    f.text((x1 + w1 + x2) / 2, ym - 1.4, "init", FS["small"], c=PAL["mute"], a="middle")
    stage(x2, y0, w2, "A1 · 3D model", ["910 volumes, 20,000 steps", "32 × 256 × 256 patches"], PAL["new_f"], PAL["new"], tcol=PAL["new_d"])
    f.prov_add("A1: 3D model trained 20,000 steps on 910 volumes from the seed-matched 2D model", "main.tex Methods '3D model'; C/X1 model3d.json train.init3d = orx_t3d_a1_s0; train log")
    x3, w3 = x2 + w2 + 10.0, 33.0
    yc, yx = oy + 6.4, oy + 20.6
    bh2 = 9.6
    for (yy, ttl, ln) in ((yc, "C · +3,000 steps", "unchanged training data"),
                          (yx, "X1 · +3,000 steps", "simulated lateral downsampling")):
        ghosts(x3, yy, w3, bh2, PAL["new_f"], PAL["new"])
        f.rect(x3, yy, w3, bh2, rx=1.6, fill=PAL["new_f"], stroke=PAL["new"], sw=1.1 * LW)
        f.text(x3 + w3 / 2, yy + 3.6, ttl, FS["mod"], 700, c=PAL["new_d"], a="middle")
        f.text(x3 + w3 / 2, yy + 6.9, ln, FS["small"], a="middle")
    xs = x2 + w2 + 0.5
    f.path(f"M{xs:.2f},{ym:.2f} C{xs + 5.0:.2f},{ym:.2f} {x3 - 5.0:.2f},{yc + bh2 / 2:.2f} {x3 - 0.5:.2f},{yc + bh2 / 2:.2f}")
    f.path(f"M{xs:.2f},{ym:.2f} C{xs + 5.0:.2f},{ym:.2f} {x3 - 5.0:.2f},{yx + bh2 / 2:.2f} {x3 - 0.5:.2f},{yx + bh2 / 2:.2f}")
    f.text(x3 + w3 / 2, yx + bh2 + 2.6, "p = 0.33, spacing 2–10 voxels, blur 0.3–0.6×", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("C: 3,000 steps from A1, no aniso aug; X1: 3,000 steps, aniso_aug 0.33, lateral axes (aniso_xy 1.0), interval 2-10, blur 0.3-0.6; warm-up 100",
               "orx_t3d_r5c_s0/model3d.json train (steps 3000, aniso_aug 0.0); orx_t3d_r5x1_s0 train (aniso_aug 0.33, aniso_lo 2, aniso_hi 10, aniso_xy 1.0, aniso_blur 0.3, aniso_blur_max 0.6); data3d.py:141-161,291-296")
    # weight-space average glyph
    gx0 = x3 + w3 + 5.0
    o = (gx0, oy + 16.0)
    tC = (gx0 + 7.5, oy + 8.6)
    tX = (gx0 + 9.5, oy + 24.0)
    mid = ((tC[0] + tX[0]) / 2, (tC[1] + tX[1]) / 2)
    f.arrow(o[0], o[1], tC[0], tC[1], stroke=PAL["new_d"])
    f.arrow(o[0], o[1], tX[0], tX[1], stroke=PAL["new_d"])
    f.path(f"M{tC[0]:.2f},{tC[1]:.2f} L{tX[0]:.2f},{tX[1]:.2f}", stroke=PAL["mute"], arrow=False, dash="0.7 0.5", sw=0.8 * LW)
    f.circle(mid[0], mid[1], 1.0, fill=PAL["new"], stroke="#ffffff", sw=0.6 * LW)
    f.circle(o[0], o[1], 0.55, fill=PAL["fg"], chk=False)
    f.text(tC[0] - 1.6, tC[1] - 1.2, "θ_{C}", FS["body"], c=PAL["new_d"], a="end")
    f.text(tX[0] - 0.8, tX[1] + 2.4, "θ_{X1}", FS["body"], c=PAL["new_d"], a="end")
    f.text(o[0] + 3.5, oy + 31.0, "weight average", FS["small"], c=PAL["mute"], a="middle")
    sxb, sw_ = ox + w - 19.5, 15.5
    f.arrow(mid[0] + 1.2, mid[1], sxb - 0.5, mid[1])
    ghosts(sxb, mid[1] - 6.0, sw_, 12.0, PAL["new"], PAL["new"], op=0.25)
    f.rect(sxb, mid[1] - 6.0, sw_, 12.0, rx=1.6, fill=PAL["new"], stroke=PAL["new"], sw=1.1 * LW)
    f.text(sxb + sw_ / 2, mid[1] - 0.6, "S1", 7.5, 700, c="#ffffff", a="middle")
    f.text(sxb + sw_ / 2, mid[1] + 3.3, "½ (θ_{C} + θ_{X1})", FS["small"], c="#ffffff", a="middle")
    f.prov_add("S1 = parameter-wise average 0.5*C + 0.5*X1", "orx_t3d_r5s1_s0/model3d.json avg_of [r5c_s0, r5x1_s0], avg_alpha 0.5")


def panel_d(f, ox, oy, w=183.0):
    """Inference."""
    f.plabel(ox, oy + 3.0, "d", "Inference on a test volume (S1)")
    r1 = oy + 6.5
    hgt = 21.0
    cyl = r1 + hgt / 2
    f.image(ox + 0.5, r1 - 0.5, 20.0, 21.0, "input_cutaway.png")
    f.text(ox + 10.5, r1 + hgt + 2.4, "test volume, z/xy = 4.65", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("input render: mouse-embryo E15_Ab2 test volume, orthogonal planes + GT nuclei", "paper/scripts/render3d/fig1_assets.py:125-149 (input_cutaway.png)")
    x2 = ox + 27.0
    f.arrow(ox + 21.0, cyl, x2 - 0.5, cyl)
    f.rect(x2, r1, 37.0, hgt, rx=1.6, fill=PAL["ctl_f"], stroke=PAL["ctl"], sw=0.9 * LW)
    f.text(x2 + 18.5, r1 + 3.6, "1  diameter estimate", FS["mod"], 700, a="middle")
    cxp = x2 + 3.0
    for dd in [6, 10, 17, 30, 52, 90, 150]:
        rr = 0.55 + 0.75 * math.log(dd / 6.0)
        cxp += rr
        f.circle(cxp, r1 + 10.4, rr, fill="#ffffff", stroke=PAL["ctl"], sw=0.8 * LW, chk=False)
        cxp += rr + 0.9
    f.text(x2 + 18.5, r1 + 16.0, "2D control at 7 scales; pick the most stable,", FS["small"], a="middle")
    f.text(x2 + 18.5, r1 + 18.6, "self-consistent scale; refine d", FS["small"], a="middle")
    f.prov_add("diameter-free inference: seed-matched 2D model at candidate diameters 6,10,17,30,52,90,150 px; stability-based selection",
               "eval_val.py:160-199 (diam_mode joint, joint_rule r2a2: per scale and threshold n, d_est, cons=|log(d_est/sc)| l.172, stab = mean F1@0.5 with neighbouring scales l.181, n >= min_obj 2 l.188, scale = argmax stab-cons at cp0 l.192, threshold = argmax stab l.193, refine d = d_est l.196, final run at d l.199); scales 6,10,17,30,52,90,150 + iso_final from run 4a362b0f config.json (orx_v5_2d_s0) via orx_run.py:101-103; S1 consumes d_used via eval_gt.py:22-23,211-212 --diam_from. eval_val.py identical (md5 bdfd9f74) in runs 4a362b0f and 07992b55. Wording differences vs main.tex that do not change the method: (i) eval_val.py:79 default --scales is 6,10,17,30,52,90 (no 150); the full-pipeline runs pass all seven via config.json; (ii) the 8e8-voxel skip (l.160-163) applies only after at least one scale has run, scales being visited from large to small diameter.")
    x3 = x2 + 43.0
    f.arrow(x2 + 37.5, cyl, x3 - 0.5, cyl)
    f.rect(x3, r1, 38.0, hgt, rx=1.6, fill=PAL["data_f"], stroke=PAL["box"], sw=0.9 * LW)
    f.text(x3 + 19.0, r1 + 3.6, "2  rescale to d = 30 px", FS["mod"], 700, a="middle")
    tt = 11.0
    for xx, nm in ((x3 + 3.0, "xz_native.png"), (x3 + 24.0, "xz_iso.png")):
        f.image(xx, r1 + 5.2, tt, tt, nm, fit="xMidYMid slice")
        f.rect(xx, r1 + 5.2, tt, tt, rx=0, stroke=PAL["rule"], sw=0.6 * LW)
    f.arrow(x3 + 14.5, r1 + 5.2 + tt / 2 + 1.0, x3 + 23.5, r1 + 5.2 + tt / 2 + 1.0)
    f.text(x3 + 19.0, r1 + 5.2 + tt / 2 - 0.8, "linear z", FS["small"], c=PAL["mute"], a="middle")
    f.text(x3 + 19.0, r1 + 19.4, "isotropic if z/xy ≥ 2", FS["small"], a="middle")
    f.prov_add("rescale by 30/d; z/xy >= 2 -> linear z-interpolation to isotropic grid, labels sampled back at original planes",
               "infer3d.py:211-217; eval_gt.py:217-222 (zinterp), 240 m = m[zi]")
    x4 = x3 + 44.0
    f.arrow(x3 + 38.5, cyl, x4 - 0.5, cyl)
    w4 = ox + w - x4 - 0.5
    f.rect(x4, r1, w4, hgt, rx=1.6, fill="#ffffff", stroke=PAL["box"], sw=0.9 * LW)
    f.text(x4 + 2.0, r1 + 3.6, "3  sliding 32-plane windows", FS["mod"], 700)
    wl = LOGA["windows"]
    Zp, st = wl["Z_network"], wl["starts"]
    ax0, ax1 = x4 + 3.0, x4 + w4 - 3.0
    base, amp = r1 + 15.0, 7.0
    sc = (ax1 - ax0) / Zp
    for i, s0 in enumerate(st):
        pts = [(ax0 + (s0 + kk + 0.5) * sc, base - amp * math.sin(math.pi * (kk + 0.5) / 32) ** 2) for kk in range(0, 32)]
        d = "M" + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts)
        col = PAL["new"] if i % 2 == 0 else PAL["new_l"]
        f.add(f'<path d="{d}" stroke="{col}" stroke-width="{0.9 * LW:.3f}" fill="none"/>')
    f.add(f'<line x1="{ax0:.2f}" y1="{base:.2f}" x2="{ax1:.2f}" y2="{base:.2f}" stroke="{PAL["fg"]}" stroke-width="{0.7 * LW:.3f}"/>')
    s3 = st[3]
    bx0, bx1 = ax0 + s3 * sc, ax0 + (s3 + 32) * sc
    f.add(f'<path d="M{bx0:.2f},{base + 0.8:.2f} v0.7 H{bx1:.2f} v-0.7" stroke="{PAL["mute"]}" stroke-width="{0.6 * LW:.3f}" fill="none"/>')
    f.text((bx0 + bx1) / 2, base + 3.4, "32 planes", FS["small"], c=PAL["mute"], a="middle")
    f.text(ax1, base + 3.4, f"z′ = {Zp} planes, {len(st)} windows", FS["small"], c=PAL["mute"], a="end")
    f.text(x4 + w4 - 2.0, r1 + 3.6, "sin² weights, stride 23–24", FS["small"], c=PAL["mute"], a="end")
    f.prov_add("z windows of 32 planes with 25% overlap (_starts: stride 23-24, Z' = 426, 18 windows for this volume), xy windows 256 with 25% overlap, sin^2 taper, weighted average",
               "infer3d.py:56-60 _starts, 129-132 _taper, 161-186 run_volume; model3d.json infer {patch_z 32, overlap_z 0.25, overlap_xy 0.25}; assets_log.json windows")

    # ---------------- row 2
    r2 = oy + 33.0
    h2 = 21.0
    cy2 = r2 + h2 / 2
    xv, wv = ox + 0.5, 39.0
    cxn = xv + wv - 5.0
    f.path(f"M{x4 + w4 / 2:.2f},{r1 + hgt + 0.3:.2f} L{x4 + w4 / 2:.2f},{r1 + hgt + 2.6:.2f} L{cxn:.2f},{r1 + hgt + 2.6:.2f} "
           f"L{cxn:.2f},{r2 - 0.5:.2f}")
    f.rect(xv, r2, wv, h2, rx=1.6, fill="#ffffff", stroke=PAL["box"], sw=0.9 * LW)
    f.text(xv + 2.0, r2 + 3.6, "4  three views, averaged", FS["mod"], 700)
    for j, (nm, axn) in enumerate((("z", 0), ("y", 1), ("x", 2))):
        cx = xv + 3.0 + j * 12.2
        cyb = r2 + 6.6
        cw = 6.0
        f.cuboid(cx, cyb + 3.0, cw, cw, 3.4, PAL["inh_f"], tint(PAL["inh"], 0.08), PAL["inh_f2"], stroke=PAL["inh"], sw=0.5 * LW)
        dxp, dyp = 3.4 * math.cos(math.radians(35)), -3.4 * math.sin(math.radians(35))
        for q in (0.3, 0.6, 0.9):
            if axn == 0:
                a = (cx + q * dxp, cyb + 3.0 + q * dyp)
                pts = [a, (a[0] + cw, a[1]), (a[0] + cw, a[1] + cw), (a[0], a[1] + cw)]
            elif axn == 1:
                yy = cyb + 3.0 + q * cw
                pts = [(cx, yy), (cx + dxp, yy + dyp), (cx + cw + dxp, yy + dyp), (cx + cw, yy)]
            else:
                xx = cx + q * cw
                pts = [(xx, cyb + 3.0), (xx + dxp, cyb + 3.0 + dyp), (xx + dxp, cyb + 3.0 + cw + dyp), (xx, cyb + 3.0 + cw)]
            f.poly(pts, fill=PAL["new"], stroke=PAL["new"], sw=0.4 * LW, op=0.35, chk=False)
        f.text(cx + 4.4, r2 + h2 - 1.6, f"{nm} as net z", FS["small"], c=PAL["mute"], a="middle")
    f.prov_add("three views: each volume axis in turn as the network's z (VIEW_PERMS), outputs permuted back and averaged; single view for Cellos, Parhyale, mesoSPIM",
               "infer3d.py:63-64 VIEW_PERMS, 86-96 run_views; S1 run config.json --views 3 --views1_ds cellos parhyale mesospim")
    x5, w5 = xv + wv + 6.0, 35.0
    f.arrow(xv + wv + 0.5, cy2, x5 - 0.5, cy2)
    f.rect(x5, r2, w5, h2, rx=1.6, fill="#ffffff", stroke=PAL["box"], sw=0.9 * LW)
    f.text(x5 + 2.0, r2 + 3.6, "5  flow following", FS["mod"], 700)
    f.image(x5 + 0.6, r2 + 4.2, 19.5, 16.6, "flowfollow_S1.png")
    f.text(x5 + 20.6, r2 + 8.8, ["voxels move", "along the", "3D flow; one", "sink per", "nucleus"], FS["small"], c=PAL["mute"])
    f.prov_add("flow following with Cellpose dynamics (CellposeModel._compute_masks, niter = 200 d/30) on the isotropic grid; render = S1 seed-0 trajectories on 4 neighbouring nuclei of mouse_embryo E15_Ab2 (GT 33, 51, 68, 50), one sink each",
               "infer3d.py:258-260; eval_gt.py:238-239; paper/figures/assets_viz/F1-a2_flow_following_S1.provenance.json (nc-viz, rule F1_group in selection_rules.json); display contrast only: rgb**1.8, alpha x1.9, tight crop")
    x6, w6 = x5 + w5 + 6.0, 47.0
    f.arrow(x5 + w5 + 0.5, cy2, x6 - 0.5, cy2)
    f.rect(x6, r2, w6, h2, rx=1.6, fill=PAL["loss_f"], stroke=PAL["loss"], sw=0.9 * LW)
    f.text(x6 + 2.0, r2 + 3.6, "6  rule T: foreground threshold t", FS["mod"], 700)
    f.text(x6 + 2.0, r2 + 7.6, "m(t) = N_{fg}(t) / (N_{obj}(t) V),  V = πd^{3}/6 · v_{xy}/v_{z}", FS["small"])
    f.text(x6 + 2.0, r2 + 11.2, "t = −2 if m(−2) ≥ 0.7;", FS["small"])
    f.text(x6 + 2.0, r2 + 14.2, "else highest t ∈ {−3, −4, −6} with m(t) ≥ 0.85;", FS["small"])
    f.text(x6 + 2.0, r2 + 17.2, "else t = −6; then drop objects < 0.15 V", FS["small"])
    f.prov_add("rule T thresholds and constants; size filter 0.15 x expected volume", "thr_eval.py (eval_gt --thrs 0 -1 -2 -3 -4 -6 then sizeThr_dir.py:4,28); eval_gt.py:235,248-252; main.tex Methods 'Rule T'")
    x7 = x6 + w6 + 6.0
    f.arrow(x6 + w6 + 0.5, cy2, x7 - 0.5, cy2)
    w7 = ox + w - x7 - 0.5
    f.rect(x7, r2, w7, h2, rx=1.6, fill="#ffffff", stroke=PAL["box"], sw=0.9 * LW)
    f.text(x7 + 2.0, r2 + 3.6, "7  3D nuclei", FS["mod"], 700)
    f.image(x7 + 1.5, r2 + 4.6, 11.0, 16.0, "masks_S1.png")
    f.text(x7 + 14.5, r2 + 8.6, ["labels", "sampled back", "at the original", "planes"], FS["small"], c=PAL["mute"])
    f.prov_add("3D masks: S1 seed-0 final masks of mouse_embryo E15_Ab2 (run 07992b55 thr_eval, rule-T threshold -2, fragments < 0.15 V removed)",
               "paper/scripts/render3d/out/fig1/E15_Ab2_S1_iso.png + E15_Ab2_iso.provenance.json (nc-figures; pyvista contour_labels, ortho az 35 el 25)")


# ======================================================================== layouts
def layout_a():
    f = Fig(183.0, 154.0)
    panel_a(f, 0.0, 0.0)
    panel_b(f, 59.0, 0.0, 124.0)
    panel_c(f, 0.0, 65.5)
    panel_d(f, 0.0, 99.0)
    return f


def layout_b():
    f = Fig(183.0, 154.0)
    panel_c(f, 0.0, 0.0)
    panel_a(f, 0.0, 34.0)
    panel_b(f, 59.0, 34.0, 124.0)
    panel_d(f, 0.0, 99.0)
    return f


CHECK_JS = r"""
<script>
window.addEventListener('load', () => {
  const svg = document.getElementById('fig');
  const R = (el) => { const b = el.getBBox(); const m = el.getCTM(); const sm = svg.getCTM().inverse().multiply(m);
    const pts = [[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y]) => {
      const p = svg.createSVGPoint(); p.x = x; p.y = y; return p.matrixTransform(sm); });
    const xs = pts.map(p=>p.x), ys = pts.map(p=>p.y);
    return {x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys)}; };
  const vb = svg.viewBox.baseVal;
  const texts = [...svg.querySelectorAll('text')].map(t => ({el: t, s: t.textContent, b: R(t)}));
  const ims = [...svg.querySelectorAll('image.im')].map(t => ({b: R(t)}));
  const shapes = [...svg.querySelectorAll('.chk')];
  const inter = (a, b, pad) => !(a.x1 - pad <= b.x0 || b.x1 - pad <= a.x0 || a.y1 - pad <= b.y0 || b.y1 - pad <= a.y0);
  const issues = [];
  for (let i = 0; i < texts.length; i++) {
    const a = texts[i];
    if (a.b.x0 < 0 || a.b.y0 < 0 || a.b.x1 > vb.width || a.b.y1 > vb.height) issues.push('OUT ' + a.s);
    for (let j = i + 1; j < texts.length; j++) if (inter(a.b, texts[j].b, 0.05)) issues.push('TT "' + a.s + '" x "' + texts[j].s + '"');
    for (const im of ims) if (inter(a.b, im.b, 0.05)) issues.push('TI "' + a.s + '"');
  }
  for (const sh of shapes) {
    let L; try { L = sh.getTotalLength(); } catch (e) { continue; }
    const m = svg.getCTM().inverse().multiply(sh.getCTM());
    const n = Math.max(20, Math.ceil(L / 0.15));
    const P = [];
    for (let k = 0; k <= n; k++) { const p = sh.getPointAtLength(L * k / n).matrixTransform(m); P.push(p); }
    for (const t of texts) {
      const b = t.b, pad = 0.12;
      if (P.some(p => p.x > b.x0 - pad && p.x < b.x1 + pad && p.y > b.y0 - pad && p.y < b.y1 + pad))
        issues.push('TS "' + t.s + '" x <' + sh.tagName + ' ' + (sh.getAttribute('d') || sh.getAttribute('points') || sh.getAttribute('x') || '').slice(0, 40) + '>');
    }
  }
  // minimum rendered font size (1 user unit = 1 mm at the final 183 mm width)
  let minpt = 1e9;
  for (const el of svg.querySelectorAll('text, tspan')) {
    if (!el.textContent.trim()) continue;
    const pt = parseFloat(getComputedStyle(el).fontSize) / 0.3528;
    minpt = Math.min(minpt, pt);
    if (pt < 4.995) issues.push('FONT ' + pt.toFixed(2) + ' pt "' + el.textContent + '"');
  }
  const pre = document.createElement('pre'); pre.id = 'ovl'; pre.style.display = 'none';
  pre.textContent = JSON.stringify({min_font_pt: minpt, n_text: texts.length, n_shapes: shapes.length, issues});
  document.body.appendChild(pre);
});
</script>
"""


def html(f, title):
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>'
            f'@page {{ size: {f.W}mm {f.H}mm; margin: 0 }} html, body {{ margin: 0; padding: 0; background: #ffffff }} '
            f'svg {{ display: block }} text {{ font-kerning: normal }}</style></head><body>{f.svg()}{CHECK_JS}</body></html>')


def export(f, cid):
    hp = os.path.join(HERE, f"{cid}.html")
    open(hp, "w").write(html(f, cid))
    ud = tempfile.mkdtemp(prefix="chr_")
    url = "file://" + hp
    dom = subprocess.run([CHROME, "--headless=new", f"--user-data-dir={ud}", "--virtual-time-budget=1500", "--dump-dom", url],
                         capture_output=True, text=True, timeout=180).stdout
    m = re.search(r'<pre id="ovl"[^>]*>(.*?)</pre>', dom, re.S)
    res = json.loads(m.group(1).replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")) if m else {"error": "no result"}
    json.dump(res, open(os.path.join(HERE, f"{cid}.overlap.json"), "w"), indent=1, ensure_ascii=False)
    subprocess.run([CHROME, "--headless=new", f"--user-data-dir={ud}", "--no-pdf-header-footer", "--virtual-time-budget=1500",
                    f"--print-to-pdf={os.path.join(HERE, cid + '.pdf')}", url], capture_output=True, timeout=180)
    wpx, hpx = math.ceil(f.W / 25.4 * 96), math.ceil(f.H / 25.4 * 96)
    subprocess.run([CHROME, "--headless=new", f"--user-data-dir={ud}", f"--window-size={wpx},{hpx}", "--hide-scrollbars",
                    "--force-device-scale-factor=6.25", "--virtual-time-budget=1500",
                    f"--screenshot={os.path.join(HERE, cid + '.png')}", url], capture_output=True, timeout=180)
    return res


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    for cid, fn in (("A2-a", layout_a), ("A2-b", layout_b)):
        if which not in ("all", cid[-1]):
            continue
        f = fn()
        res = export(f, cid)
        out[cid] = res
        print(cid, "issues:", len(res.get("issues", [])))
        for s in res.get("issues", [])[:80]:
            print("   ", s)
        prov = dict(figure=cid, code_snapshot=f"{_NSP_RUNS}/07992b55-0982-4116-b065-b34859c9a80b/repo/src (S1 thr_eval run, server hostA)",
                    model=f"{_NSP_ROOT}/models/models/orx_t3d_r5s1_s0 (model3d.json)",
                    palette=PAL, assets_log=LOGA, items=f.prov)
        json.dump(prov, open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
