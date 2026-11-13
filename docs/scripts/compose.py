"""Compose README animations from orbit frames (orbit.py): hero (mouse embryo, GT | 2D control | S1) and gallery (Fig. 2e, five
datasets x {2D control, S1}). Numbers are read from figures/final/Fig6.provenance.json (same volumes and per-volume F1 as Fig. 2e / Fig. 6,
gt protocol, seed 0; the file is part of the figure outputs, which are deposited with the source data). Fonts: macOS Arial. Encoding: img2webp -loop 0 -lossy -q 85 (hero) / -q 78 (gallery) -m 6 -d 70 <frames> -o docs/assets/<name>.webp.
Colours: figures/palette_viewer.json (pane, error classes) and figures/palette.json (S1 / 2D model colours)."""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PAPER = os.environ.get("NUCSEG3D_PAPER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
_SCR = os.environ.get("NUCSEG3D_FIG_SCRATCH", ".")
FR = f"{_SCR}/readme_frames"
OUTD = f"{_SCR}/readme_seq"
PV = json.load(open(f"{_PAPER}/figures/palette_viewer.json"))
PAL = json.load(open(f"{_PAPER}/figures/palette.json"))
PROV = json.load(open(f"{_PAPER}/figures/final/Fig6.provenance.json"))["rows"]
N = 72
PANE, PAGE = PV["pane"], "#FFFFFF"
FG, MUT = PAL["text"], PAL["muted"]
S1C, D2C = PAL["models"]["S1"], PAL["models"]["2D_dark"]
ERRN = PAL["error_names"]
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONTB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
DSN = {"nis3d": "NIS3D", "tapenade": "Tapenade", "nisnet_zf": "NISNet3D zebrafish", "mouse_embryo": "Mouse embryo", "blastospim": "BlastoSPIM"}
NGT = {"mouse_embryo": 129}


def f(sz, b=False):
    return ImageFont.truetype(FONTB if b else FONT, sz)


def bbox(keys):
    """Union alpha bbox over all frames of the given frame sets (one fixed crop per dataset, so the loop does not jump)."""
    a = None
    for k in keys:
        for i in range(N):
            al = np.asarray(Image.open(f"{FR}/{k}/{i:03d}.png"))[..., 3] > 8
            a = al if a is None else (a | al)
    ys, xs = np.nonzero(a)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def tile(k, i, box, w, h, margin=0.05):
    im = Image.open(f"{FR}/{k}/{i:03d}.png").crop(box)
    sc = min(w * (1 - 2 * margin) / im.width, h * (1 - 2 * margin) / im.height)
    r = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.LANCZOS)
    t = Image.new("RGBA", (w, h), Image.new("RGB", (1, 1), PANE).getpixel((0, 0)) + (255,))
    t.alpha_composite(r, ((w - r.width) // 2, (h - r.height) // 2))
    return t


def rrect(d, xy, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(xy, r, fill=fill, outline=outline, width=width)


def legend(d, cx, y, sz):
    items = [(PV["errors"][k], ERRN[k]) for k in ("match", "boundary", "split", "merge", "other", "miss")]
    ft = f(sz); sw = int(sz * 0.95); gap = int(sz * 0.45); sp = int(sz * 1.3)
    widths = [sw + gap + d.textlength(t, font=ft) for _, t in items]
    x = cx - (sum(widths) + sp * (len(items) - 1)) / 2
    for (c, t), w in zip(items, widths):
        rrect(d, (x, y + 2, x + sw, y + 2 + sw), 2, fill=c)
        d.text((x + sw + gap, y), t, font=ft, fill=MUT)
        x += w + sp


def hero():
    ds = "mouse_embryo"; ks = ["gt", "2D", "S1"]
    box = bbox([f"{ds}_{k}" for k in ks])
    T, G, M, HEAD, FOOT = 400, 14, 18, 76, 52
    W = M * 2 + 3 * T + 2 * G; H = HEAD + T + FOOT + 6
    f1 = PROV[ds]["f1"]
    os.makedirs(f"{OUTD}/hero", exist_ok=True)
    for i in range(N):
        C = Image.new("RGBA", (W, H), PAGE)
        d = ImageDraw.Draw(C)
        for j, k in enumerate(ks):
            x = M + j * (T + G); y = HEAD
            t = tile(f"{ds}_{k}", i, box, T, T)
            mask = Image.new("L", (T, T), 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, T - 1, T - 1), 10, fill=255)
            C.paste(t, (x, y), mask)
            title = {"gt": "Ground truth", "2D": "2D control", "S1": "S1 (3D fine-tuned)"}[k]
            col = {"gt": FG, "2D": D2C, "S1": S1C}[k]
            d.text((x + T / 2, 10), title, font=f(25, True), fill=col, anchor="mt")
            sub = {"gt": f"{NGT[ds]} annotated nuclei, one colour each", "2D": "per-plane Cellpose-SAM, fine-tuned in 2D",
                   "S1": "+ zero-initialized 3D residual decoder"}[k]
            d.text((x + T / 2, 44), sub, font=f(16), fill=MUT, anchor="mt")
            if k != "gt":
                s = f"F1@0.5 {f1[k]['f1_05']:.3f}   F1@0.7 {f1[k]['f1_07']:.3f}"
                ft = f(18, k == "S1"); tw = d.textlength(s, font=ft)
                rrect(d, (x + 10, y + 10, x + 24 + tw, y + 40), 6, fill="#FFFFFFD0")
                d.text((x + 16, y + 14), s, font=ft, fill=S1C if k == "S1" else FG)
            if k == "S1":
                rrect(d, (x - 1, y - 1, x + T, y + T), 10, outline=S1C, width=3)
        d.text((M + 10, HEAD + T - 28), "Mouse embryo E15_Ab2 · z/xy 4.65", font=f(16), fill=MUT)
        legend(d, W / 2, HEAD + T + 16, 19)
        C.convert("RGB").save(f"{OUTD}/hero/{i:03d}.png")


def gallery():
    dss = ["nis3d", "tapenade", "nisnet_zf", "mouse_embryo", "blastospim"]
    T, G, LAB, HEAD, FOOT, M = 236, 8, 36, 62, 46, 10
    W = M * 2 + LAB + 5 * T + 4 * G; H = HEAD + 2 * T + G + FOOT
    boxes = {ds: bbox([f"{ds}_2D", f"{ds}_S1"]) for ds in dss}
    os.makedirs(f"{OUTD}/gallery", exist_ok=True)
    for i in range(N):
        C = Image.new("RGBA", (W, H), PAGE)
        d = ImageDraw.Draw(C)
        for r, k in enumerate(["2D", "S1"]):
            y = HEAD + r * (T + G)
            lab = Image.new("RGBA", (T, LAB), (0, 0, 0, 0)); ImageDraw.Draw(lab).text((T / 2, LAB / 2), "2D control" if k == "2D" else "S1",
                                                                                        font=f(20, True), fill=D2C if k == "2D" else S1C, anchor="mm")
            lab = lab.rotate(90, expand=True); C.alpha_composite(lab, (M, y))
            for c, ds in enumerate(dss):
                x = M + LAB + c * (T + G)
                t = tile(f"{ds}_{k}", i, boxes[ds], T, T, 0.06)
                mask = Image.new("L", (T, T), 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, T - 1, T - 1), 8, fill=255)
                C.paste(t, (x, y), mask)
                v = PROV[ds]["f1"][k]
                s5, s7 = f"{v['f1_05']:.3f}", f"{v['f1_07']:.3f}"
                ft = f(15, k == "S1"); col = S1C if k == "S1" else FG
                lbl = f"F1@0.5 {s5} · @0.7 {s7}"; tw = d.textlength(lbl, font=ft)
                rrect(d, (x + T - 14 - tw, y + T - 29, x + T - 5, y + T - 5), 5, fill="#FFFFFFD0")
                d.text((x + T - 9, y + T - 9), lbl, font=ft, fill=col, anchor="rd")
                if k == "S1":
                    rrect(d, (x - 1, y - 1, x + T, y + T), 8, outline=S1C, width=2)
                if r == 0:
                    d.text((x + T / 2, 8), DSN[ds], font=f(19, True), fill=FG, anchor="mt")
                    d.text((x + T / 2, 34), f"z/xy {PROV[ds]['z_xy']:.2f}", font=f(16), fill=MUT, anchor="mt")
        legend(d, M + LAB + (W - M - LAB) / 2, H - FOOT + 13, 18)
        C.convert("RGB").save(f"{OUTD}/gallery/{i:03d}.png")


if __name__ == "__main__":
    {"hero": hero, "gallery": gallery}[sys.argv[1]]()
