"""Shared style for nc-viz mechanism assets (V1-V4). Palette fixed by team lead."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_FIG_SCRATCH = _nsp_os.environ.get('NUCSEG3D_FIG_SCRATCH', '/path/to/figure_scratch')
_NSP_PAPER = _nsp_os.environ.get('NUCSEG3D_PAPER', _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", "..")))
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MM = 1 / 25.4
W60 = 60 * MM
DPI = 900                      # >= 600 dpi at 60 mm width
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCR = f"{_NSP_FIG_SCRATCH}"
C = dict(S1="#E64B35", C2D="#4DBBD5", orig="#8491B4", text="#1d1d1f", shell="#9AA3B5", gt="#ffffff", gtline="#1d1d1f")  # NPG (palette.json)
# Diverging maps: navy-white-red from palette.json (replaces PRGn, 2026-11-10); DIVA has a grey middle for arrows on white.
_D = json.load(open(os.environ.get("NC_PALETTE", f"{_NSP_PAPER}/figures/palette.json")))["diverging"]
DIV = LinearSegmentedColormap.from_list("pal_div", _D)
DIVA = LinearSegmentedColormap.from_list("pal_div_a", [_D[0], "#b4b6bd", _D[-1]])


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "text.color": C["text"], "axes.labelcolor": C["text"], "xtick.color": C["text"], "ytick.color": C["text"],
        "axes.edgecolor": C["text"], "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False, "legend.fontsize": 5.5})


def save(fig, name, prov):
    """PNG at DPI + SVG + provenance JSON."""
    png = os.path.join(OUT, name + ".png"); svg = os.path.join(OUT, name + ".svg")
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg)
    w, h = fig.get_size_inches()
    prov = dict(asset=name, png=png, svg=svg, size_mm=[round(w / MM, 1), round(h / MM, 1)], dpi=DPI, **prov)
    json.dump(prov, open(os.path.join(OUT, name + ".provenance.json"), "w"), indent=1, ensure_ascii=False)
    plt.close(fig)
    return png


# ---------------------------------------------------------------- palette (unified NPG palette from nc-figures, 2026-11-10)
PALETTE_JSON = os.environ.get("NC_PALETTE", f"{_NSP_PAPER}/figures/palette.json")


def _pj():
    return json.load(open(PALETTE_JSON))


def palette():
    j = _pj(); m = j["models"]
    return dict(S1=m["S1"], S1_dark=m["S1_dark"], C2D=m["2D"], C2D_dark=m["2D_dark"], orig=m["base"], text=j["text"], muted=j["muted"],
                shell=j["shell"], shell_line="#7d879b", sink=j["text"], bg=j["bg"], gt=j["gt_outline"], errors=j["errors"])


def palette_source():
    return PALETTE_JSON + " (" + _pj()["name"] + ")"


def diverging():
    """navy-white-red ramp from palette.json 'diverging'."""
    return LinearSegmentedColormap.from_list("pal_div", _pj()["diverging"])


def diverging_arrows():
    """same ramp with a grey middle so near-zero arrows stay visible on white."""
    d = _pj()["diverging"]
    return LinearSegmentedColormap.from_list("pal_div_a", [d[0], "#b4b6bd", d[-1]])


def sequential():
    """single-hue ramp for angle errors (white -> palette navy -> deep navy); palette.json has no sequential entry."""
    return LinearSegmentedColormap.from_list("pal_seq", ["#ffffff", "#c3cbe0", "#3C5488", "#1b2647"])
