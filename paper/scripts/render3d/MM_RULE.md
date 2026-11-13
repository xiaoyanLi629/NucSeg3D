# Multi-model 3D strip: selection rule (written 2026-11-09 before any multi-model rendering)

Datasets: among the 13 test datasets, the median-rule volume of ortho/selection.json (S1 - 2D F1@0.5 closest to the dataset median,
ties by most GT nuclei; seed 0, gt protocol) is used. Datasets are the four with the largest z/xy among those whose median-rule
volume has >= 50 GT nuclei: BlastoSPIM (9.62), Cellos (8.36), Parhyale (7.81), mouse embryo (4.65).
Models (columns): GT | original Cellpose-SAM (fixed -2) | 2D control (rule T) | u-Segment3D (2D control, selected variant v10) |
nnU-Net (selected cp4_vol0.25 = pred_v4, fragments < 0.25 expected volume removed) | StarDist-3D (cp2_vol0.15 = pred_v2, 0.15) |
published tools (Tapenade, AnyStar, Cellos StarDist): G1 runs saved no masks -> "n/a" | S1 (rule T, last, framed #2f6fde).
Order of the middle columns: ascending macro F1@0.5 (gt protocol, 13 datasets) as reported (compare_test_gt*.json, g3_compare.json).
Callout region: the GT nucleus cluster (the GT nucleus and its touching GT neighbours) centred on the GT nucleus with the largest
number of distinct error classes across the models (ties: most voxels), from the evaluated masks; window = 3 nuclear diameters.
Per-volume F1@0.5 under each tile comes from the per-volume result files (vols/*.json / rows.json) of the same runs; nothing new.
If a model has no saved mask for a volume, the tile reads "n/a".

## Amendment (2026-11-10, before rendering the multi-sample grids)
Multi-sample grids (T, MM, F6, H; user request) use the five datasets of Fig. 2d's representative-volume rule
(ortho/RULE.md: 12 pooled datasets ordered by z/xy, positions round(linspace(0, 11, 5)) = NIS3D, Tapenade, NISNet zebrafish,
mouse embryo, BlastoSPIM), each with its median-rule volume. Callout rule unchanged.
