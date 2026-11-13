# Fig. 2e cube volume: selection rule (written 2026-11-10 before any round-6 cube rendering)

Reason: the R5 cube used the mouse-embryo volume E15_Ab2 (z/xy 4.65). Its side faces are 47 native z planes stretched 4.65x,
so they are soft at print size whatever the render settings. Team-lead allowed an isotropic volume chosen by the existing
median rule among datasets with z/xy <= 1.5.

Rule: candidate datasets are those with z/xy <= 1.5 whose median-rule test volume (ortho_selection.json; S1 - 2D F1@0.5
closest to the dataset median, seed 0, gt protocol) has an exported mask/image file (render3d/export/<ds>.npz). These are
NIS3D (Zebrafish_2_xR25, d = 6.5 px) and Tapenade (annotator2_square13, d = 16.3 px). The dataset with the larger
median nuclear diameter in pixels is used, so that individual nuclei are resolvable at the 48-mm panel width (no look at
model outputs or scores). -> Tapenade, annotator2_square13, voxel 0.62 um isotropic.

Crop: the same rule as the R5 cube: xy side s = min(220, Y, X) centred on the GT-foreground centroid; z extent
min(round(s / (z/xy)), Z) centred on the plane with the most GT foreground. S1 masks of the seed-0 run (07992b55),
evaluated final masks. Faces: raw image, 1-99 % LUT of the crop with gamma 0.8 (whole-image), as in R5.
The E15_Ab2 cube is re-rendered with the new pipeline only for the before/after comparison.
