# R6 Fig. 6 (multi-model 3D comparison): selection, region and zoom rules

Written 2026-11-10, before any R6-Fig6 rendering and before any per-nucleus difference was computed. This is an illustration
only. Every number printed on the figure is a per-volume F1 that already exists in the evaluation outputs; no new statistic
is computed.

## Common to both versions

1. Columns, in this order: ground truth, original Cellpose-SAM (fixed threshold −2), nnU-Net, the 2D control, S1. StarDist-3D
   and u-Segment3D are dropped from the main figure; they remain in the supplementary multi-model grid.
2. Masks. The evaluated final masks are used, gt protocol, seed 0 for S1 and the 2D control.
   - S1 and the 2D control use the threshold chosen by rule T. The masks are taken from the errors cache
     (`$NUCSEG3D_SCRATCH/nc/errors/cache/gt_{S1,2D}_s0`). Fragments smaller than 0.15 × the expected volume are removed.
   - The original Cellpose-SAM uses fixed −2 (`cache/gt_base_s0`).
   - nnU-Net uses the setting selected on validation: `pred_v4`, with fragments smaller than 0.25 × the expected volume removed.
3. Displayed region. The whole evaluated volume, cropped only to the bounding box of the ground-truth foreground, expanded by
   half a nuclear diameter (z converted by z/xy) and clipped to the volume. This removes empty or unannotated margins but cuts
   no annotated nucleus. Within a row all five panes share this box and one orthographic camera.
4. Numbers on each pane. F1@0.5 and F1@0.7 of the whole evaluated volume (not of the box or the zoom), computed as
   2·tp / (n_gt + n_pred). For S1, the 2D control and the original, the values come from the stored per-volume counts
   (`errors/per/gt_*_s0`). For nnU-Net they come from the baseline run's `rows.json` (thr 4, frac 0.25). These two sources
   were cross-checked as 227/227 identical in `boundary/`.
5. Zoom inset (one region per row, shown in every pane of the row). For every ground-truth nucleus g and every comparison
   model m ∈ {original, nnU-Net, 2D control}, let IoU_m(g) be the best IoU of g with any prediction of m (0 if none). The
   per-nucleus disagreement is s(g) = Σ_m |IoU_S1(g) − IoU_m(g)|. The absolute value makes the rule blind to who is better.
   - The region score of a candidate centre c (each ground-truth centroid) is the sum of s(g) over the nuclei whose centroids
     lie within 1.5 nuclear diameters of c, measured in isotropic xy-pixel units.
   - The highest-scoring candidate is taken; ties go to more nuclei in the window, then to the lower label.
   - The inset is the cube of side 3 nuclear diameters centred on it, clipped to the volume. It is marked in the full render
     by a white wireframe box.
6. Rendering. Black image tiles on a white page. Colours are the palette.json NPG errors with hue kept and only lightness and
   saturation changed, i.e. exactly `palette_vivid_dark.json` and `r6_render.ERRV`. Each tile is rendered at 2× its final
   pixel size and downsampled with a Lanczos filter (≥ 1000 ppi at print size). Depth peeling is used, and text is drawn as
   vector text.

## Version A: same information, more impact

Datasets and volumes are unchanged from the current Fig. 6, i.e. MM_RULE.md amendment 2026-11-10: NIS3D, Tapenade, NISNet3D
zebrafish, mouse embryo, BlastoSPIM, each with its median-rule volume from `figs/ortho_selection.json`.

## Version B: datasets chosen to match the paper's conclusion

1. Ranking. In Supplementary Table g2ds (gt protocol, F1@0.7, S1 − nnU-Net, mean over S1 seeds; source
   `data_snapshot/paired_stats_gt.json`, key `nnU-Net|13|F1@0.7`), rank all 13 datasets by the difference.
   - The 3 largest positive differences are taken: Cellos +0.135, Parhyale +0.132, NucVerse3D +0.068.
   - The 2 most negative differences are taken: mesoSPIM −0.095 (single volume) and BlastoSPIM −0.085.
   - The rule is applied literally, so mesoSPIM is included. It is the largest nnU-Net advantage in the table, although it is
     a single volume and not part of the 12-dataset pooled analyses.
2. Volume within each dataset. The median-rule volume of `figs/ortho_selection.json` (unchanged rule, S1 − 2D F1@0.5).
3. Rows are ordered S1-better first, then nnU-Net-better. The nnU-Net-better rows are labelled as such on the figure and in
   the caption.
