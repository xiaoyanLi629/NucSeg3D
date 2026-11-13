# NucSeg3D

Code for the manuscript **“What volumetric fine-tuning adds to a 2D foundation model for 3D nucleus segmentation”** (submitted to *Nature Communications*).

The repository contains the code that trained and evaluated the models in the paper and produced its statistics, figures and tables: the 2D fine-tuning control, the 3D model S1 (a zero-initialised residual 3D decoder over per-slice Cellpose-SAM ViT-L encoder features, trained in 3D together with the encoder and finished as a two-model weight average, a “model soup”), the inference pipeline (view rule, isotropic flow following, rule T), the evaluation protocols, the leakage audit, the native 3D and published-tool baselines, and the post hoc analyses reported in the paper and its Supplementary Information.

**Status.** This is a review release. Model weights (five seeds of S1 and of the 2D control), the standardised benchmark, the train/validation/test splits and the leakage-audit lists (C1, C2) will be deposited in an archive; **DOI: to be determined (TBD)**. No data, weights or result files are included here. The licence has not been decided yet (see `LICENSE`).

Most comments, docstrings and log messages in the code are in Chinese, because the project was developed in Chinese. Function and variable names, command-line options and file formats are in English.

## Repository layout

| Folder | Files | Contents |
|---|---|---|
| `src/` | 49 | Core package (flat modules that import each other by name): data registry and standardisation, slice and isotropic-volume generation, 2D control training, 3D network, 3D training, weight averaging, inference, evaluation, rule T, and the GPU follow-up evaluations. |
| `orx_run.py`, `run.sh`, `val_split.json`, `mesospim_*.json`, `configs/` | 6 | Experiment driver used for every training and evaluation run (each run was one `config.json`; one real example is in `configs/`), and the fixed validation split. |
| `data_prep/` | 3 | Download scripts for the public source datasets (URLs of the original repositories). |
| `audit/` | 5 | Train–test leakage audit (template matching and identity checks against every training volume) and the independent spot check. |
| `baselines/` | 24 | nnU-Net v2 (single fold and five-fold), StarDist-3D, conversion to their formats, timing (G5), evaluation and paired statistics against S1. |
| `analysis/stats/` | 17 | Five-seed statistics: paired seed differences, confirmation rule, mixed models, meta-analysis and meta-regression, covariates, Bland–Altman, reviewer-round analyses (`round3/`). |
| `analysis/five_seed/` | 8 | Five-seed result aggregation (`s5_stats.py`, `test_r6.py`) that the statistics build on, rule-T decision, Purdue tolerant-split analysis. |
| `analysis/errors/` | 9 | Error decomposition (miss, merge, split, boundary, false), boundary metrics, F1–IoU curves. |
| `analysis/flowerr/` | 5 | Angle between predicted and ground-truth-derived 3D flows. |
| `analysis/cpu_extra/` | 9 | Morphology bias tests and depth / local-SNR stratification (CPUX). |
| `analysis/followups/` | 33 | Follow-up analyses G1 (published tools), G3 (u-Segment3D), G4 (anisotropy degradation), G6, M1, A3–A5 (calibration, decoder residual, feature distance), VOX/VOX2 (source voxel sizes), EO (encoder-only control); `jobs/` holds the job definitions with the exact command lines. |
| `analysis/heldout/` | 17 | Held-out dataset experiments L1–L3 (retraining without a dataset, few-shot, estimated diameters). |
| `analysis/methods/` | 1 | Crop-window counts reported in Methods. |
| `analysis/development/` | 31 | Development-stage selection scripts (rounds 4–6: paired screening and confirmation, soup checks, diagnostics) and the ground-truth boundary-convention check. These produced the development history in the Supplementary Information. |
| `analysis/figure_exports/` | 9 | Server-side exports of volumes, flows and renderings used by the figure scripts. |
| `tests/round6/` | 19 | Equivalence and unit tests written during development (for example, that the zero-initialised decoder reproduces the 2D model exactly). |
| `paper/scripts/`, `paper/figures/` | 94 | Figure, table and source-data generators (final versions and the modules they import). The folder mirrors the manuscript folder so relative paths inside the scripts work. |

## Method components and where they are

| Component (paper term) | Code |
|---|---|
| 2D control (2D fine-tuning of Cellpose-SAM on xy and isotropic xz/yz slices) | `src/make_slices_std.py`, `src/make_slices_flow3d.py` (slices and flow targets), `src/train_cpsam2.py` (training) |
| 3D network: per-slice encoder + zero-initialised residual 3D decoder (variant `dec3d`) | `src/net3d.py` (`Dec3D`, `ResBlock3D`, output layer initialised to zero) |
| 3D training (A1, then C and X1 continuations; anisotropy simulation options `--aniso_*`) | `src/train3d.py`, `src/data3d.py` (patches from the isotropic store built by `src/make_iso_vols.py`) |
| Model soup S1 = ½(C + X1), α fixed at 0.5 | `src/avg_weights.py` (round 5) and `src/avg_weights_r6.py` (same averaging, later seeds) |
| Inference, three-view averaging and the view rule (`--views 3 --views1_ds cellos parhyale mesospim`) | `src/infer3d.py`, `src/eval_gt.py` |
| Isotropic flow following (interpolate z to an isotropic grid, follow flows there, return to the original planes) | `src/eval_gt.py` (uses `zinterp` from `src/eval_val.py`) |
| Rule T (scale-consistent foreground threshold) | `src/sizeThr_dir.py`, `src/thr_eval.py` |
| Automatic diameter search used by the primary (full-pipeline) protocol | `src/eval_val.py` (`r2a2` joint rule), passed to `eval_gt.py --diam_from` |
| gt protocol (annotated diameters) and full pipeline evaluation; F1 at IoU 0.5 / 0.7 | `src/eval_gt.py`, `src/metrics.py` |
| E3 three-seed ensemble | `src/ens_exact.py`, `src/ens_test.py` |
| Benchmark construction (13 datasets, standard format, voxel sizes) | `src/cpft_data.py`, `src/cpft_data_b3.py`, `src/standardize.py`, `src/fix_std*.py`, `src/merge_b3.py`, `src/stddata.py`, `src/val_split.py`, `data_prep/` |

## Installation

The models were trained and evaluated on Linux with NVIDIA GPUs (including RTX 5090) and Python 3.10, with PyTorch 2.14.1 built for CUDA 13. The main environment is described by `requirements/core.txt` (key packages) and `requirements/env_torch_freeze.txt` (the complete `pip freeze` of the environment that ran the experiments).

```bash
python3.10 -m venv env && source env/bin/activate
pip install -r requirements/core.txt
```

Cellpose 4.2.1 downloads the original Cellpose-SAM weights (`cpsam`) on first use. Baselines used separate environments (`requirements/nnunet.txt`, `requirements/usegment3d.txt`, `requirements/stardist.txt`). Figure and table scripts ran on macOS: the 3D renderings with Python 3.12 (`requirements/figures_render3d.txt`; `paper/scripts/render3d/r10_g.py` needs Python ≥ 3.12), the other figure and table scripts with Python 3.9 (`requirements/figures_tables.txt`).

## Paths

The original scripts used absolute paths on the compute servers. In this release every such path is read from an environment variable; the code is otherwise unchanged. Python files define the variables at the top (`_NSP_ROOT = os.environ.get("NUCSEG3D_ROOT", ...)`), shell scripts use `${NUCSEG3D_ROOT}`, and docstrings, comments, JSON and Markdown show `$NUCSEG3D_ROOT` as text.

| Variable | Meaning (layout expected below it) |
|---|---|
| `NUCSEG3D_ROOT` | Data, model and result root: `data/raw`, `data/std` (standardised benchmark with `manifest.csv`), `models/` (original and fine-tuned weights; `models/models/<name>`), `results/` (one folder per evaluation run), `iso30/`, `slices/`. |
| `NUCSEG3D_SCRATCH` | Fast local disk: unpacked isotropic store (`cpft_iso30/`), training slices (`cpft_v5/`, `cpft_slices_v3/`), and the working folders in which the analysis scripts were run (`nc/`, `r5/`, `r6/`, ...; see the table below). |
| `NUCSEG3D_SHARED` | Project folder on the shared disk: `baselines/` (nnU-Net and StarDist data and results), `nc_gpu/`, `nc_cpu_extra/`, `nc_heldout/`, the Python environment `env_torch/`. |
| `NUCSEG3D_RUNS` | Run folders of the experiment driver (`<run id>/log`, `<run id>/repo/src` code snapshot). Only read by a few merge scripts. |
| `NUCSEG3D_FS` | Root of the shared disk (only mentioned in `run.sh`). |
| `NUCSEG3D_ANYSTAR` | AnyStar repository and its TensorFlow environment (published-tool comparison G1). |
| `NUCSEG3D_ORX_ENV` | Environment script of the experiment-tracking CLI (one held-out consistency test). |
| `NUCSEG3D_PAPER` | Manuscript folder; defaults to `paper/` of this repository. Figure inputs (`data_snapshot/`, `scripts/figs/localdata/`, `scripts/render3d/export/`) go there once the source data are deposited. |
| `NUCSEG3D_SRC` | Code used by the Fig. 1 asset scripts; defaults to `src/`. |
| `NUCSEG3D_FIG_SCRATCH`, `NUCSEG3D_STYLE_REF` | Scratch folder for intermediate figure assets; folder of design reference images used only by two early layout drafts. |
| `NUCSEG3D_PROXY` | Optional HTTP proxy for the download scripts (direct download when unset). |

The analysis scripts were run from working folders on the server. They are grouped by topic here; when a script refers to its own working folder (for example `$NUCSEG3D_SCRATCH/nc/errors`) that is where it reads and writes results.

| Working folder on the server | Folder in this repository |
|---|---|
| `$NUCSEG3D_SCRATCH/nc/stats` | `analysis/stats` |
| `$NUCSEG3D_SCRATCH/r6` | `analysis/five_seed`, `analysis/development/round6`, `tests/round6` |
| `$NUCSEG3D_SCRATCH/nc/errors`, `nc/flowerr`, `nc/methods` | `analysis/errors`, `analysis/flowerr`, `analysis/methods` |
| `$NUCSEG3D_SCRATCH/nc/gpu` | `analysis/followups` (analysis code) and `src/` (evaluation entry points such as `gx_eval.py`, `useg_eval.py`, `g5_bench.py`) |
| `$NUCSEG3D_SCRATCH/nc/heldout`, `$NUCSEG3D_SHARED/nc_heldout` | `analysis/heldout` |
| `$NUCSEG3D_SHARED/nc_cpu_extra` | `analysis/cpu_extra` |
| `$NUCSEG3D_SCRATCH/r5`, `r3_aniso`, top level | `analysis/development` |
| `$NUCSEG3D_SCRATCH/nc/figs` | `analysis/figure_exports`, `paper/scripts` |
| `$NUCSEG3D_SCRATCH/r9` (audit) | `audit` |
| `$NUCSEG3D_SHARED/baselines/scripts` | `baselines` |

Where a script put a code folder on `sys.path`, that entry now points inside this repository. Several scripts originally imported a frozen copy of `src/` (the code snapshot of a particular run). Most of those copies are byte-identical to the `src/` released here. A few older snapshots (the S1 evaluation run used for Fig. 1 assets and some figure exports) predate later options; those options were added with defaults that reproduce the earlier behaviour, but bit-identity of these particular imports was not re-run for this release.

## Reproducing the results (overview)

Exact command lines of every reported run are recorded in the job definitions (`analysis/followups/jobs/`, `analysis/heldout/jobs/`) and in the run configurations (`configs/` shows the format; `orx_run.py` documents every field). The order of work is as follows.

1. **Data.** Download the source datasets (`data_prep/`, `src/fetch_gonuclear.py`), standardise them (`src/standardize.py`, then `src/fix_std.py` … `src/fix_std5.py`, `src/merge_b3.py`), and fix the validation split (`src/val_split.py`, output `val_split.json`). The standardised benchmark and splits will also be available from the archive.
2. **2D control.** Generate training slices and 2D flow targets (`src/make_slices_flow3d.py`, which calls `src/make_slices_std.py`) and fine-tune Cellpose-SAM with `src/train_cpsam2.py` (AdamW, learning rate 1e-5, weight decay 0.1, 100 epochs of 1,000 slices; seeds 0, 1000, 2000, 3000, 4000).
3. **3D model.** Build the isotropic store (`src/make_iso_vols.py`), train A1 from the 2D control of the same seed (`src/train3d.py --variant dec3d --init <2D model> ...`), continue A1 for 3,000 steps without (C) and with simulated lateral downsampling (X1: `--aniso_aug 0.33 --aniso_xy 1 --aniso_lo 2 --aniso_hi 10 --aniso_blur 0.3 --aniso_blur_max 0.6`), and average C and X1 with `src/avg_weights.py <C> <X1> <S1> 0.5`. All settings are listed in Methods.
4. **Evaluation.** Run `src/thr_eval.py` (which calls `src/eval_gt.py` and applies rule T with `src/sizeThr_dir.py`) for the gt protocol on the 227 test volumes, and with `--diam_from` (diameters from the `r2a2` search of `src/eval_val.py`) for the full pipeline on the 45 test volumes. The view rule is `--views 3 --views1_ds cellos parhyale mesospim`.
5. **Statistics.** `analysis/five_seed/s5_stats.py` aggregates the five seeds; `analysis/stats/run_pre.sh` and `analysis/stats/run_all.sh` (caching, readouts, covariates, `nc_stats.py`) and `analysis/stats/nc_report.py` produce the statistics used in the paper; the remaining folders under `analysis/` produce the follow-up and supplementary analyses.
6. **Baselines.** Data conversion `baselines/make_nnunet_raw.py`; training `baselines/nnunet_train.sh` and `baselines/train_stardist.py`; prediction and evaluation `baselines/nnunet_pred.py`, `baselines/stardist_pred.py`, `baselines/bl_eval.py` (the full command sequence is in `baselines/run_full.sh`); comparisons with S1 `baselines/compare_test*.py` and `baselines/paired_stats*.py`; timing `baselines/run_g5.sh`.
7. **Leakage audit.** `audit/leak_audit.py` (its later steps are chained by `audit/leak_chain.sh`), then `audit/leak_extra.py` and `audit/leak_report.py`; `audit/spot_leak.py` is the independent spot check.
8. **Figures and tables.** Final figure generators: Fig. 1 `paper/figures/arch/gen_a9.py` (assets: `make_a8d_assets.py`, `arch_views.py`, `diffusion_data.py`); Fig. 2 `paper/scripts/figs/fig2_r8.py`; Fig. 3 `paper/scripts/figs/fig3_r12.py`; Fig. 4 `paper/scripts/render3d/fig4r10.py`; Fig. 5 `paper/scripts/figs/fig5_r8.py`; Fig. 6 `paper/scripts/render3d/r8_fig6.py`; Fig. 7 `paper/scripts/figs/fig7_r7.py`; Supplementary figures `paper/figures/supp/gen_supp.py` with `make_supp_assets.py`. Supplementary tables: `paper/scripts/make_supp_tables.py`, `make_supp_table_eo.py`, `order_supp_tables.py`; source data: `make_source_data.py`, `make_source_data_xlsx.py`, `make_sd2.py`. Earlier layout revisions that the final generators import are kept because they are needed to run them.

## Weights, data and splits

Weights and benchmark splits: to be deposited; DOI TBD. The source datasets are public and listed, with their original repositories, in the datasets table of the paper.

## Citation

If you use this code, please cite the paper (citation to be added after publication):

```
@article{nucseg3d_tbd,
  title   = {What volumetric fine-tuning adds to a 2D foundation model for 3D nucleus segmentation},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD},
  doi     = {TBD}
}
```

## Licence

All rights reserved; the licence is to be decided (see `LICENSE`).
