# R6 Fig. 4 (G4, axial subsampling and noise): rules for the enriched candidates

Written 2026-11-10 11:00 (server time), before any of the new panels below was rendered or any per-dataset value was computed.
Scope: two gallery candidates `R6-Fig4-A` and `R6-Fig4-B`. Illustration and re-display of existing G4 results only. No new
statistical test, no new inference, nothing on the test set is used to choose a setting. `final/` and `main.tex` are not touched.

## Data sources (read-only)

- Macro values for the heat maps and the compact line panels come from `$NUCSEG3D_SCRATCH/nc/gpu/g4/g4_summary.json`
  (`conditions/z{k}_n{c}/all_{f05,f07}`). These are the same values as `NUMBERS.md` (G4 section) and `final/Fig4.provenance.json`.
- The macro difference is S1 − 2D, the mean of the three paired seed differences. "Range" means [min, max] of the three paired seed
  differences (`diff` = [mean, min, max]). In the heat maps a cell gets a frame when this range excludes 0, i.e. all three paired
  seeds agree in sign. This is a descriptive marker, not a test. The legend says so.
- Per-dataset values for the small multiples: for each of the 6 G4 datasets, the pooled F1 of each model instance (S1 and 2D,
  seeds 0/1000/2000) in each condition is computed with exactly the `pick` + per-dataset pooling of `g4_summary.py` (rule-T threshold
  from each run's `noise_<c>/ruleT.json`, fragment floor 0.15, rows from `rows.json`). Shown: the mean of the three paired differences,
  with the seed range. Check: the mean over the 6 datasets must reproduce `g4_summary.json` `S1`/`D2` macro means to < 1e-12.
  These are descriptive re-displays of the G4 rows. The caption must say that per-dataset values have 1–4 volumes each.

## Representative volume and images

- Volume and crop are taken unchanged from the existing F5 illustration (`F5_RULE.md`, written 2026-11-09 before selection).
  The volume is `celegans / pha4I2L_0408072`: among the 20 G4 volumes, its per-volume F1@0.5 difference (S1 seed 0 − 2D seed 0,
  k = 8, no noise, rule T) is closest to the median. The crop is 96 × 96 px around the GT nucleus nearest the volume centre, full z.
  No new selection is made.
- Panel "degradation grid": rows k = 1, 2, 4, 8, columns noise c = 0, 0.25, 0.5. Each image is produced by applying
  `gx_eval.degrade` (imported, not re-implemented) to the full evaluated volume (`load_case`, 384 box). The noise therefore uses the
  identical random stream and σ = c × (GT-inside median − outside median) as the G4 runs. The image is then cut to the F5 crop.
- Section: the xz plane at the crop's central y (row 48), the same as F5. The z range is trimmed to the GT extent ±1 plane (as F5).
  Displayed at physical aspect by nearest-neighbour replication (z voxel = k × 0.122 µm, xy 0.116 µm).
  One fixed grey-level window is used for all 12 tiles (1st–99.8th percentile of the k = 1, c = 0 crop), so the added noise is
  visible as such. GT outlines are white on the dark image tiles.
- Panel "S1 vs 2D at k = 8, no noise": the existing F5 CPU re-inferred crop masks (`export/f5_{S1,2D}_k8.npz`, seed 0, rule-T
  threshold of the G4 run, floor 0.15). Each predicted nucleus is coloured by `rstyle.classify_preds` against the subsampled GT,
  using the palette error colours (colouring only). The k = 1 crop of the same models is shown alongside for reference. No metric is
  computed or printed from these masks, because they come from a stand-alone crop inference and need not equal the G4 full-volume masks.

## Layout and checks

- Size ≤ 183 × 170 mm, text 5–7 pt, line widths 0.25–1 pt. Raster panels are rendered at the final size with ≥ 600 ppi and
  antialiasing. Image tiles sit on a dark background; the page is white.
- Colours come from `figures/palette.json`: models, errors, and the diverging ramp `#3C5488 → #ffffff → #E64B35` for S1 − 2D
  (symmetric limits per metric).
- The overlap and font-size check is `render3d/figcheck.py`. Provenance JSON is written next to each output and lists sources,
  md5s, this rule, and every number drawn.
- No generative-AI content.
