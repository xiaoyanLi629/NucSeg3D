"""Shared style for NucSeg3D manuscript figures (Nature Communications).

Single column 89 mm, double column 183 mm; Arial 5-7 pt; Okabe-Ito colours.
The three models keep the same colour and marker in every figure.
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for fp in ("/usr/share/fonts/truetype/arial/arial.ttf", "/usr/share/fonts/truetype/arial/arial-bold.ttf"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)

MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM
OUT = os.environ.get("NC_FIG_OUT", f"{_NSP_SCRATCH}/nc/figs/out")

# Model colours (Okabe-Ito) and markers, used identically in all figures.
C = {
    "S1": "#0072B2",      # our model (blue)
    "S1d": "#003F63",     # dark variant of S1 (second metric, e.g. F1@0.7)
    "2D": "#E69F00",      # 2D fine-tuned control (orange)
    "base": "#6E6E6E",    # original Cellpose-SAM (grey)
    # baselines
    "tapenade": "#009E73",
    "anystar": "#CC79A7",
    "cellos": "#D55E00",
    "nnunet": "#56B4E9",
    "stardist3d": "#F0E442",
    "useg": "#000000",
}
MK = {"S1": "o", "2D": "s", "base": "^", "tapenade": "D", "anystar": "v", "cellos": "P", "nnunet": "X", "stardist3d": "*"}
LAB = {"S1": "S1", "2D": "2D control", "base": "Original Cellpose-SAM",
       "tapenade": "Tapenade StarDist", "anystar": "AnyStar", "cellos": "Cellos StarDist",
       "nnunet": "nnU-Net", "stardist3d": "StarDist-3D"}
if os.environ.get("NC_PALETTE") == "unified":
    C.update({"S1": "#2f6fde", "S1d": "#173f8f", "2D": "#E69F00", "base": "#6b6b70"})

# error components (Paul Tol muted, distinct from model colours)
EC = {"miss": "#88CCEE", "merge": "#CC6677", "split": "#DDCC77", "boundary": "#117733", "false": "#AA4499"}
PALETTE = None
if os.environ.get("NC_PALETTE") == "npg":   # v2 unified palette (paper/figures/palette.json, copied next to this file on the server)
    _pp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "palette.json")
    PALETTE = json.load(open(_pp))
    _m = PALETTE["models"]
    C.update({"S1": _m["S1"], "S1d": _m["S1_dark"], "2D": _m["2D"], "base": _m["base"], "tapenade": _m["tapenade"],
              "anystar": _m["anystar"], "cellos": _m["cellos"], "nnunet": _m["nnunet"], "stardist3d": _m["stardist3d"],
              "useg": _m["useg"]})
    EC.update({"miss": PALETTE["errors"]["miss"], "merge": PALETTE["errors"]["merge"], "split": PALETTE["errors"]["split"],
               "boundary": PALETTE["errors"]["boundary"], "false": PALETTE["errors"]["other"]})

DS_NAME = {"bbbc050": "BBBC050", "blastospim": "BlastoSPIM", "celegans": "C. elegans", "cellos": "Cellos",
           "gonuclear": "GoNuclear", "mesospim": "mesoSPIM", "mouse_embryo": "Mouse embryo", "nis3d": "NIS3D",
           "nisnet_zf": "NISNet3D zebrafish", "nucverse3d": "NucVerse3D", "parhyale": "Parhyale",
           "purdue": "Purdue", "tapenade": "Tapenade"}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "legend.frameon": False,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "lines.linewidth": 0.9, "lines.markersize": 3, "patch.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "figure.dpi": 150, "axes.unicode_minus": True,
        "mathtext.default": "regular",
    })
    if os.environ.get("NC_PALETTE") in ("unified", "npg"):
        plt.rcParams.update({"text.color": "#1d1d1f", "axes.labelcolor": "#1d1d1f", "xtick.color": "#1d1d1f",
                             "ytick.color": "#1d1d1f", "axes.edgecolor": "#6b6b70", "grid.color": "#e2e2e0"})


def panel(ax_or_fig, letter, x=-0.02, y=1.0, fig=None, **kw):
    """Bold lower-case panel letter, 8 pt, placed in figure coordinates relative to an axes."""
    if fig is None:
        fig = ax_or_fig.figure
        bb = ax_or_fig.get_position()
        x0, y0 = bb.x0 + x, bb.y1 + 0.0
    else:
        x0, y0 = x, y
    fig.text(x0 if fig is not None else x, y0, letter, fontsize=8, fontweight="bold", ha="left", va="bottom", **kw)


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=8, fontweight="bold", ha="left", va="top")


def save(fig, name, prov):
    os.makedirs(OUT, exist_ok=True)
    try:
        import figcheck
        prov["overlap_issues"] = figcheck.check(fig)
        print("overlap issues:", prov["overlap_issues"])
    except ImportError:
        pass
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    json.dump(prov, open(f"{OUT}/{name}.provenance.json", "w"), indent=1, ensure_ascii=False)
    w, h = fig.get_size_inches()
    print(f"saved {name}: {w / MM:.0f} x {h / MM:.0f} mm")
