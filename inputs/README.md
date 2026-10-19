# Where the data go

Put recordings in these folders and run the scripts. Every script reads from here by default
and writes its results to `outputs/`. Git tracks nothing in this folder except this README and
the empty folder skeleton, so recordings cannot be committed by accident.

```
inputs/
├── adult/                 adult cohort with simultaneous ECG
│   └── recordings/        model evaluation, Python analyses, the toolbox and the adult HEP
├── hbn/                   Healthy Brain Network, or any dataset without an ECG
│   ├── recordings/        input to the toolbox
│   └── participants*.csv  HBN participant release files (age analysis only)
└── <your_dataset>/
    └── recordings/        input to the toolbox
```

## File formats

| Folder | Contents |
|---|---|
| `adult/recordings/` | EEGLAB `.set`/`.fdt` pairs with 130 channels: 1–128 EEG (EGI HydroCel order) and two external channels, of which 129 was not used and 130 is the ECG; R-peaks as `Rpeak` events. Add `cohort_map.csv` with columns `filename,cohort` (`TD` or `ASD`). The file names must match `splits/split_manifest.csv`. |
| `hbn/recordings/`, `<your_dataset>/recordings/` | EEGLAB `.set` files with the 128 EEG channels of the EGI HydroCel montage first, in native order. No ECG is needed. |
| `hbn/participants*.csv` | The HBN participant files as released (comma- or tab-delimited). They are read by `python/hbn_merge_demographics.py`. |

## Where the results go

| Data | Toolbox work folder | Analysis results |
|---|---|---|
| adult | `outputs/adult_validation/toolbox/` | `outputs/` (Python), `outputs/adult_validation/` (MATLAB) |
| HBN | `outputs/deployment/hbn/toolbox/` | `outputs/deployment/hbn/` |
| your dataset | `outputs/deployment/<your_dataset>/toolbox/` | `outputs/deployment/<your_dataset>/` |

To keep data somewhere else, set `HDEEG2ECG_INPUT_DIR` (and `HDEEG2ECG_OUTPUT_DIR`) to a folder
with the same layout.
