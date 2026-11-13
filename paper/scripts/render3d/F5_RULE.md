# F5 anisotropy-degradation illustration: rule (written 2026-11-09 before any selection or re-inference)

Purpose: illustration only (team-lead approval 2026-11-09). No metric is computed or shown; the caption points to the G4 statistics.
Volume: among the 20 G4 test volumes (lists/g4_keys.txt), the one whose per-volume F1@0.5 difference S1 seed 0 − 2D control seed 0
at k = 8, noise 0 (rule T threshold from each run's ruleT.json, frac 0.15; read from the G4 runs' vols/*.json rows) is closest to the
median of that difference over the 20 volumes; ties by more GT nuclei, then key order.
Crop: lateral window of 96 x 96 px (eval grid, after the eval_gt 384 box) centred on the GT nucleus whose centroid is nearest the
volume centre, shifted inward at borders; full z.
Degradation: exactly G4 (gx_eval.degrade): keep every k-th plane from plane 0 of image and GT, z voxel x k, no noise; k = 1, 2, 4, 8.
Inference: same as gx_eval.run_volume (eval_gt load_model; S1 with views=3 (views1 rule for cellos/parhyale/mesospim), 2D control
Cellpose do_3D; diameter from lists/g4_diam (d_used); isotropic interpolation when z/xy >= 2; flow3D_smooth 1.0; flow_threshold 0.4,
min_size 15, max_size_fraction 0.4, niter 200/(30/d)); threshold = the per-volume rule-T threshold that the G4 run selected for this
volume and k (2D/S1, seed 0); fragments < 0.15 x expected volume removed. Run on CPU (CUDA_VISIBLE_DEVICES=""), nice 19, <= 8 threads,
not concurrently with nc-viz server jobs.
Note: the crop is inferred on its own, so its masks need not equal the full-volume masks of the G4 run; hence no numbers.
