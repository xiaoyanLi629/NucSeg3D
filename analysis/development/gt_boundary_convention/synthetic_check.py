#!/usr/bin/env python
"""Sanity check of gt_conv.per_nucleus on synthetic data with known answers (CPU, tiny).

Uniform ellipsoids (semi-axes rz, ry, rx) on background 0.1, peak 1.0, Gaussian blur sigma=1 voxel, small noise.
Case A: GT = true object                      -> r_face ~0.5, ratio ~1, grad ~1, iou50 high
Case B: GT = object dilated by 2 voxels (fat) -> lower r_face, ratio < 1, grad < 1
Case C: tiny object 4-5 px wide in xy, long in z, blurred (mesospim-like geometry), GT = true object
"""
import numpy as np
from scipy import ndimage as ndi

import gt_conv as G


def make(shape, centers, radii, sigma, noise, seed=0):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    lab = np.zeros(shape, np.uint16)
    for i, (c, r) in enumerate(zip(centers, radii), 1):
        m = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2 <= 1
        lab[m] = i
    img = 0.1 + 0.9 * (lab > 0).astype(np.float32)
    img = ndi.gaussian_filter(img, sigma) + rng.normal(0, noise, shape).astype(np.float32)
    return img.astype(np.float32), lab


def run(name, img, lab, dilate=0):
    if dilate:
        lab = ndi.grey_dilation(lab, size=(2 * dilate + 1,) * 3)
    lo, hi = np.percentile(img, [1, 99.8])
    n = (img - lo) / (hi - lo)
    for i, sl in G.eligible(lab, None):
        r = G.per_nucleus(i, sl, n, lab, None, (1.0, 1.0, 1.0), float(np.median(n[lab == 0])))
        print(f"{name:28s} lab{i} run_z {r['run_z']:3d} run_x {r['run_x']:3d} r_face_z {r['r_face_z']:.2f} r_face_xy {r['r_face_xy']:.2f}"
              f" r_inner {r['r_inner']:.2f} ratio_z {r['ratio_z']:.2f} ratio_x {r['ratio_x']:.2f} grad_z {r['grad_z']:.2f}"
              f" grad_x {r['grad_x']:.2f} zr50 {r['zr_0.50']:.2f} xyr50 {r['xyr_0.50']:.2f} iou50 {r['iou_0.50']:.2f}"
              f" best_lvl {r['best_lvl']:.2f} best_iou {r['best_iou']:.2f}")


img, lab = make((60, 80, 80), [(30, 40, 40)], [(12, 15, 15)], 1.0, 0.01)
run("A big object, GT exact", img, lab)
run("B big object, GT dilated 2", img, lab, dilate=2)
img, lab = make((80, 30, 30), [(40, 15, 15)], [(12, 2.3, 2.3)], (2.0, 0.8, 0.8), 0.01)
run("C thin object, GT exact", img, lab)
run("D thin object, GT dilated 1", img, lab, dilate=1)
