# R6 Fig. 5 (held-out datasets): representative-volume rule

Written 2026-11-10, before any per-volume difference was computed or any volume was looked at. Illustration only: the
numbers printed on the figure are the per-volume F1 values already stored in `l1s_vol.json` (gt protocol, rule T,
fragments < 0.15 x expected volume removed); no new statistic is computed.

1. Source: `$NUCSEG3D_SCRATCH/nc/heldout/l1cmp/l1s_vol.json`, keys `tGT|L1_S1_s0` (retrained S1, seed 0, run 99107759) and
   `tGT|F_S1_s0` (production S1, seed 0, run 07992b55). Per-volume F1@0.7 = 2·tp7 / (n_gt + n_pred) of the rule-T row `r2`.
2. For each held-out dataset (BBBC050, BlastoSPIM, NucVerse3D) take its gt-protocol test volumes that have at least
   10 ground-truth nuclei (so that the rendering shows more than one or two objects; BBBC050 and BlastoSPIM contain
   many early-embryo volumes with 2-8 nuclei).
3. d = F1@0.7(retrained S1) - F1@0.7(production S1) per volume. F1@0.7 is used because the retraining loss is larger
   at the stricter threshold (manuscript Results). Pick the eligible volume whose d is closest to the median of d over
   the eligible volumes of that dataset; ties broken by more GT nuclei, then by volume id.
4. Masks: the evaluated final masks of both runs (rule-T threshold from the stored volume record; if t != -2 the mask is
   recomputed from the stored flows with `errors/common.recompute`, exactly as in `render_export.py`), fragments
   < 0.15 x expected volume removed. Image: the evaluation grid (same 384-px box as `eval_gt`).
5. Error colouring of the renderings uses `rstyle.classify_preds` (colouring only, no metric derived from it).
