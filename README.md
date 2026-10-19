# HDEEG2ECG

Code, trained weights and toolbox for

> Fraile-Vazquez, M. E. *et al.* Heartbeat signal recovered from high-density EEG enables
> brain-body neural marker analysis (submitted)

A dilated convolutional network reconstructs the ECG from 128-channel EEG. A reference-free
quality-control gate then decides, from the timing of the reconstructed R-peaks alone, whether
each reconstruction can be used. Heartbeat-evoked potentials (HEPs) are validated against a real
ECG in adults and then measured in the Healthy Brain Network, which has no cardiac channel.

**No recordings are included in this repository.** See [Data](#data).

## Contents

```
inputs/                            put the data here (see inputs/README.md); not tracked by git
outputs/                           every result is written here; created on first run
weights/final_model_weights.pt     the deployed model: 697,305 parameters, checkpoint from epoch 322
                                   (selection-split r = 0.753)
splits/split_manifest.csv          train / selection / testing membership of the 66 adult recordings
toolbox/hdeeg2ecg_toolbox.py       three-stage inference and quality-control toolbox (GUI)
python/                            model, adult validation, quality control, HBN analyses, figures
matlab/adult_validation/           HEP from true vs reconstructed R-peaks in the adult cohort
matlab/deployment/                 HEP in an ECG-free dataset (the Healthy Brain Network by default)
matlab/reference/                  group-level adult HEP reference used by the deployment pipeline
```

In the split manifest and in output file names, `val1` is the selection split and `val2` the
testing split.

## Installation

Python 3.11. The versions used for the paper are pinned in `requirements.txt`.

```bash
conda create -n hdeeg2ecg python=3.11
conda activate hdeeg2ecg
pip install -r requirements.txt
```

The pinned `torch` is the CUDA 12.1 build. For CPU-only use, install the matching CPU wheel
from pytorch.org instead. A GPU is needed only for retraining.

The MATLAB pipelines need EEGLAB 2020.0 with the AMICA, ICLabel, clean_rawdata and
zapline-plus plugins, and FieldTrip 20240110. Put them in `matlab/toolboxes/` as
`eeglab/eeglab2020_0` and `fieldtrip-20240110`, or point to them with
`setenv('EEGLAB_PATH', ...)` and `setenv('FIELDTRIP_PATH', ...)`.

## Quick start

Put the data in `inputs/` as described in [`inputs/README.md`](inputs/README.md), run the
scripts in the order below, and collect the results in `outputs/`. No script contains a
machine-specific path. Every script runs from any working directory, on Windows, macOS or
Linux.

### A. Adult cohort with simultaneous ECG

This is the validation cohort of the paper, available under a data-use agreement.

1. Put the recordings and `cohort_map.csv` in `inputs/adult/recordings/`.
2. Run the Python analyses:

   | Order | Script | Writes (in `outputs/`) | Manuscript |
   |---|---|---|---|
   | 1 | `hdeeg2ecg_model.py` | scores the released weights on the selection and testing splits: `per_subject_metrics_val1/val2.csv`, `validation_summary.json` | Table 1 inputs |
   | 2 | `validation_pooled_metrics.py` | `pooled_per_subject_metrics.csv`; cohort comparisons | Results (cohort) |
   | 3 | `qc_rhythm_features.py` | `rhythm_qc.csv`, `rhythm_qc_separation.csv` | Table 2 |
   | 4 | `qc_gate_thresholds.py` | `qc_gate_decisions.csv`; gate thresholds 0.16 and 0.29 | Methods |
   | 5 | `validation_summary_table.py` | Table 1 rows (bootstrap 95% CIs, 5,000 resamples) | Table 1 |
   | 6 | `validation_leakage_indices.py` | `leakage_compare.csv`; single-channel vs multichannel leakage | Methods |
   | 7 | `validation_heart_rate.py` | per-subject heart-rate table; Bland–Altman | Supplementary Table 3 |
   | 8 | `network_activations.py --prepare` | `network_activations_inputs.npz`: per-stage activations, electrode deletion | — |
   | 9 | `validation_electrode_deletion.py` | Supplementary Table 4 rows | Supplementary Table 4 |
   | 10 | `fig_overview.py` | `python/figures/fig_overview` | Fig. 1 |
   | 11 | `fig_leakage_vs_f1.py` | `python/figures/fig_leakage_vs_f1.png`; leakage–F1 statistics | Fig. 2 |
   | 12 | `fig_network_emergence.py --prepare`, then without `--prepare` | `python/figures/fig_network_emergence` | Fig. 3 |

3. Run the toolbox on `inputs/adult/recordings/` with `outputs/adult_validation/toolbox/` as
   the work folder (Stages 1–3).
4. In MATLAB, run `matlab/adult_validation` steps 1–6 twice, first with
   `setenv('HEP_TRIGGER','true')` and then with `setenv('HEP_TRIGGER','reconstructed')`.
   Then run steps 7 and 8.

### B. Healthy Brain Network

1. Download the HBN resting-state recordings and participant files. Put the recordings in
   `inputs/hbn/recordings/` and the `participants*.csv` files in `inputs/hbn/`.
2. Run the toolbox on `inputs/hbn/recordings/` with `outputs/deployment/hbn/toolbox/` as the
   work folder (Stages 1–3).
3. Run `python/hbn_merge_demographics.py`, then `python/hbn_gate_age_transfer.py` and
   `python/hbn_gate_margin.py` (gate rejection by age).
4. In MATLAB, run `matlab/deployment` steps 1–8.
5. Run `python/fig_hep_by_age.py` to draw Fig. 5 from the step-8 output and the adult reference.

### C. Your own dataset

1. Put the recordings in `inputs/<your_dataset>/recordings/`.
2. Run the toolbox with `outputs/deployment/<your_dataset>/toolbox/` as the work folder.
3. In MATLAB, run `setenv('HDEEG2ECG_DATASET','<your_dataset>')`, then run
   `matlab/deployment` steps 1–7. Step 8 needs participant ages in the HBN format.

`deploy_step7_group_hep.m` measures the HEP on the region of interest defined in the adult
cohort and compares it with the adult waveform. If the adult pipeline has been run, it reads
them from `outputs/adult_validation/`. Otherwise it uses `matlab/reference/adult_hep_reference.mat`,
so HBN or other datasets can be analysed without the adult recordings.

### Adult reference

`matlab/reference/adult_hep_reference.mat` holds the adult HEP at the group level (n = 14
adults, epochs on the R-peaks of the recorded ECG): the grand average over all 128 channels,
the validated cluster mask and its 67 channels, the region-of-interest waveform and its SEM,
and the split-half reliability. It holds no per-subject waveforms and no subject IDs. It is
on the 256-Hz time axis of the HBN analysis, in EGI HydroCel-128 channel order.
`make_adult_hep_reference.m` rebuilds it from the outputs of `adult_step7_hep_validation`,
`deploy_step7_group_hep` and `deploy_step8_hep_by_age`, without reading any recording. Used in
place of the adult outputs, it reproduces the paper's adult–HBN waveform agreement
(r = 0.864) and every age-band correlation in Supplementary Table 5.

## Script reference

### MATLAB steps

| Step | `adult_validation/` | `deployment/` | What it does |
|---|---|---|---|
| 1 | `adult_step1_import` | `deploy_step1_import` | load; adult: drop the unused external channel and resample to 512 Hz; deployment: keep the toolbox's 256 Hz |
| 2 | `adult_step2_preprocess` | `deploy_step2_preprocess` | filter, ZapLine-plus, ASR, bad-channel interpolation |
| 3 | `adult_step3_amica` | `deploy_step3_amica` | AMICA decomposition |
| 4 | `adult_step4_amica_components` | `deploy_step4_amica_components` | save the AMICA component matrices |
| 5 | `adult_step5_remove_ocular_ics` | `deploy_step5_remove_ocular_ics` | remove ocular components (ICLabel); cardiac components are kept |
| 6 | `adult_step6_epoch` | `deploy_step6_epoch` | epoch −300 to 600 ms around each R-peak |
| 7 | `adult_step7_hep_validation` | `deploy_step7_group_hep` | true- vs reconstructed-peak HEP against a surrogate, `ga_waveforms.png` (Fig. 4) / group HEP |
| 8 | `adult_step8_cfa_check` | `deploy_step8_hep_by_age` | cardiac-field-artifact check / HEP by age band (Supplementary Table 5; Fig. 5 is drawn by `python/fig_hep_by_age.py`) |

Each step writes a folder of the same name. The adult steps 1–6 write under
`outputs/adult_validation/true_rpeaks/` or `.../reconstructed_rpeaks/`. Each script finds the
repository from its own location, sets its folders with `matlab/hdeeg2ecg_paths.m`, and can be
run in a fresh MATLAB session.

### Supporting Python modules

These are imported by the scripts above: `fig_style.py` (figure style), `network_activations.py`
(per-stage activations, beat-locked modulation and electrode deletion), and `summary_figure.py`,
`summary_figure_panels.py` and `learned_topography.py` (loaders and helpers shared with earlier
draft figures).

### Retraining

`hdeeg2ecg_model.py` evaluates the released weights by default. Set `HDEEG2ECG_MODE=train` to
retrain with the paper configuration: 1,000 epochs maximum, early-stopping patience 80, and the
frozen split in `splits/split_manifest.csv`. Retraining is stochastic on GPU, so a new run will
not reproduce the released weights bit for bit. It writes `outputs/final_model_weights.pt`. To
analyse that model instead of the released one, set `HDEEG2ECG_WEIGHTS` to that file.

### Settings

Everything defaults to folders inside the repository. Override a setting with an environment
variable: `export NAME=value` (macOS/Linux), `$env:NAME = "value"` (Windows PowerShell), or
`setenv('NAME', 'value')` in MATLAB.

| Variable | Meaning | Default |
|---|---|---|
| `HDEEG2ECG_INPUT_DIR` | where the data are | `inputs/` |
| `HDEEG2ECG_OUTPUT_DIR` | where results are written | `outputs/` |
| `HDEEG2ECG_MODE` | `evaluate` the released weights or `train` a new model | `evaluate` |
| `HDEEG2ECG_WEIGHTS` | model weights used by every script and the toolbox | `weights/final_model_weights.pt` |
| `HDEEG2ECG_DATA_DIR` | adult recordings | `inputs/adult/recordings` |
| `HDEEG2ECG_COHORT_MAP` | cohort labels for `validation_pooled_metrics.py` | `cohort_map.csv` in the adult recordings folder |
| `HEP_TRIGGER` | adult HEP steps 1–6: `true` or `reconstructed` R-peaks | `true` |
| `HDEEG2ECG_DATASET` | deployment dataset (folder name under `inputs/` and `outputs/deployment/`) | `hbn` |
| `EEGLAB_PATH`, `FIELDTRIP_PATH` | MATLAB toolboxes | `matlab/toolboxes/...` |

## Toolbox

```bash
python toolbox/hdeeg2ecg_toolbox.py
```

| Stage | What it does | Output folder |
|---|---|---|
| 1 Reconstruct | 128-channel EEGLAB `.set` → the same recording with the reconstructed ECG added as channel 129 | `01_reconstructed` |
| 2 Mark peaks | automatic R-peak detection, with optional manual review in an R-DECO-style editor (add, drag, delete, R–R tachogram, mark bad) | `02_marked` |
| 3 Discriminate | passes a recording only if `frac_short` ≤ 0.16 and `frac_implaus` ≤ 0.29 and it was not marked bad; writes `toolbox_qc_summary.csv` | `03_good` / `03_bad` |

`frac_short` is the fraction of R–R intervals shorter than 400 ms, and `frac_implaus` is the
fraction outside 500–1,300 ms. The thresholds were fixed on the selection split. The toolbox
expects the EGI HydroCel 128-channel montage in its native channel order, as used for training.

**Standalone executable (Windows).** `HDEEG2ECG_Toolbox.exe` is attached to each
[GitHub release](https://github.com/casciolab/HDEEG2ECG/releases) and runs without a Python
installation. It leaves out PyTorch, so it covers Stages 2 and 3 on recordings already
reconstructed in Stage 1. To rebuild it, follow `toolbox/build_toolbox_exe.bat`.

## Data

- **Adult cohort** (66 recordings with simultaneous ECG). Individual-level EEG and ECG cannot be
  shared publicly under the approved protocol. Contact the corresponding author about access
  under a data-use agreement.
- **Healthy Brain Network.** Openly available from the Child Mind Institute under its data-use
  terms.

## Licence

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the paper if you use the model, the weights or the toolbox.
