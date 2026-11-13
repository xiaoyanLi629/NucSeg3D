"""Supplementary render figures (Tgrid, MM 7-method grid, F5 anisotropy) recoloured with the unified 3D-viewer scheme
(figures/palette_viewer.json, user 2026-11-11). Patches the colour globals of rstyle / grids / cand_f5 (files not modified),
renders into a new cache and writes R8-* outputs to the gallery. Data, selection and numbers unchanged."""
import json
import os

import numpy as np
from PIL import Image

import rstyle as R

PVW = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "palette_viewer.json")))
R.ERR.update(PVW["errors"])
R.QUAL[:] = PVW["qualitative_instances"]
R.LIGHT_BG = (PVW["pane"], PVW["pane"])
_tile = R.tile


def tile_dehalo(im, dark=False, size=None, pad=0.04):
    a = np.asarray(im.convert("RGBA")).copy(); a[..., 3][a[..., 3] < PVW["dehalo_alpha_below"]] = 0
    return _tile(Image.fromarray(a), dark, size=size, pad=pad)


R.tile = tile_dehalo

import grids as G  # noqa: E402
import cand_f5 as C  # noqa: E402

G.CACHE = os.path.join(G.HERE, "out", "grid_cache_viewer"); os.makedirs(G.CACHE, exist_ok=True)
G.add_err.__defaults__ = (PVW["matched_opacity"], True)
_fin = G.finish
G.finish = lambda fig, cid, bg, prov: _fin(fig, "R8-" + cid, bg, dict(prov, palette_override="figures/palette_viewer.json"))

if __name__ == "__main__":
    G.grid_Twide(False)          # -> R8-T-grid-wide (FigS_Tgrid)
    G.grid_MM(False)             # -> R8-MM-grid (FigS_mm7)
    C.main(False, "R8-F5-a")     # FigS_anisotropy3D
