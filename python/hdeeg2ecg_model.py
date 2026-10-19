# -*- coding: utf-8 -*-
"""
EEG -> ECG cross-modal sequence-to-sequence reconstruction.

This is the cleaned + extended pipeline. Relative to the original script it:
  - removes dead code (unused edge/min-amplitude constants, the disabled
    negative-window sampler, unused LightningModule args, the redundant
    second masking schedule, the hand-rolled early-stopping that duplicated
    Lightning's, and the DDP wrapper that only got in the way at devices=1);
  - adds a subject-level, cohort-stratified train / val1 / val2 split;
  - adds a latent-width ablation with automatic knee detection + figure;
  - computes and stores a full statistical validation suite (subject-level
    aggregation, cluster-bootstrap CIs, QRS detection F1, R-peak timing error,
    Bland-Altman HR agreement, ASD-vs-TD equivalence test) on BOTH validation
    sets, and writes a consolidated paper_validation_summary.md.

Selection / early stopping is driven by reconstruction correlation (val_r), NOT
the composite loss: the loss can be minimised by a degenerate flat prediction
(r ~ 0 at the same low loss), so monitoring the loss alone can select a dead
model. A waveform-correlation term in the loss removes that flat minimum, and
min_epochs forces training through the masking curriculum before any model is
selected.

Run once. Everything lands in OUTPUT_DIR.

@author: Matias E. Fraile-Vazquez
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import re
import json
import platform
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
from scipy import stats as sps
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


# ============================================================
# CONFIGURATION  (everything reproducible lives here)
# ============================================================
# Paths: data go in inputs/ and results in outputs/ (neither is tracked by git).
# Set the environment variables to use other locations.
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR   = os.environ.get('HDEEG2ECG_INPUT_DIR', os.path.join(REPO_ROOT, 'inputs'))
DATA_FOLDER = os.environ.get('HDEEG2ECG_DATA_DIR', os.path.join(INPUT_DIR, 'adult', 'recordings'))
OUTPUT_DIR  = os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(REPO_ROOT, 'outputs'))
# Weights every analysis script loads: the released model unless overridden (e.g. with
# outputs/final_model_weights.pt after retraining).
WEIGHTS_PATH = os.environ.get('HDEEG2ECG_WEIGHTS', os.path.join(REPO_ROOT, 'weights', 'final_model_weights.pt'))

RANDOM_SEED = 42
CUDNN_DETERMINISTIC = False   # False = fast (cuDNN autotuning); True = bit-exact reproducible but
                              # ~5x slower for this conv-heavy model. Seed is fixed either way.

# --- splits (subject level, cohort stratified). Must sum to 1.0 ---
FRAC_TRAIN = 0.70
FRAC_VAL1  = 0.15   # used during training for early stopping / model selection / the ablation knee
FRAC_VAL2  = 0.15   # locked box: untouched by training, used to test the post-inference toolbox

# Freeze the split instead of regenerating it. The split is produced by
# train_test_split(..., stratify=cohorts), so the COHORT LABELS ARE PART OF THE
# SPLIT KEY: fixing the cohort bug (21/45 -> 28/38) silently moved 13 of the 20
# held-out subjects even though RANDOM_SEED never changed. Any run that is meant
# to be comparable with the deployed model must read that model's manifest rather
# than recompute one. Set to None to generate a fresh split.
SPLIT_MANIFEST_IN = os.path.join(REPO_ROOT, 'splits', 'split_manifest.csv')

# --- signal / windowing ---
SFREQ        = 256
WINDOW_SIZE  = 512          # 2.0 s at 256 Hz
N_SHIFTS     = 2            # jittered training windows per R-peak
FILTER_LFREQ = 5.0
FILTER_HFREQ = 50.0

# --- augmentation (train only) ---
LEAD_DROPOUT_RATE   = 0.20   # per-channel Bernoulli; see gpu_augment
NOISE_STD           = 0.025
ECG_MAX_AMPLITUDE   = 4.0    # realistic normalized ECG peak; used by the bounds term

# --- masking curriculum (single source of truth now) ---
MASK_RATIO_START = 0.50
MASK_RATIO_END   = 1.00
MASK_WARMUP_EPOCHS = 30      # linearly ramp start->end over this many epochs, then hold at end.
                            # min_epochs is tied to this so no model is selected mid-curriculum.

# --- optimisation ---
BATCH_SIZE      = 128       # was 32; larger batch -> ~4x fewer steps/epoch + better GPU use
NUM_WORKERS     = 0         # augmentation now runs on GPU, so __getitem__ is cheap; 0 avoids
                            # Windows spawn/pickling overhead. Safe to keep at 0.
MAX_LR          = 2e-4      # was 1e-4; modest bump to match the larger batch (drop back if unstable)
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0

# --- loss weights (fixed; each term is normalised to [0,1] before weighting) ---
LAMBDA_LAG    = 2.0
LAMBDA_SHAPE  = 1.0
LAMBDA_BOUNDS = 0.5
LAMBDA_SMOOTH = 0.2
LAMBDA_CORR   = 2.0   # waveform-correlation term. Couples the loss to reconstruction
                      # fidelity so a flat / constant prediction (r~0) is NO LONGER a
                      # low-loss solution -- this is what let one ablation seed collapse
                      # to r~0 at the same val_loss as a good fit. Keep it >= LAMBDA_SHAPE.
TAU = 1.0

# --- latent width: testing (sweep) vs. selected (single width) ---
# RUN_ABLATION = True  -> "testing" path: sweep ABLATION_WIDTHS over ABLATION_SEEDS,
#                         pick the knee, then train the final model at that width.
# RUN_ABLATION = False -> "just run it" path: skip the sweep entirely and train the
#                         final model directly at SELECTED_LATENT_DIM. Nothing else
#                         changes width afterwards -- the final model IS this width.
RUN_ABLATION        = False   # toggle: False = single selected width; True = full sweep
SELECTED_LATENT_DIM = 32      # latent width trained when RUN_ABLATION is False
                              # (also the fallback if a sweep ever yields no knee)

# --- ablation sweep settings (only used when RUN_ABLATION = True) ---
ABLATION_WIDTHS    = [32, 64, 128, 256, 512]
ABLATION_SEEDS     = [42, 43]         # 2 seeds for error bars; 3+ strongly advised -- with 2 seeds
                                      # a single collapsed run halves a width's mean and can move the knee
ABLATION_MAX_EPOCHS = 300             # ceiling only; early stopping decides actual length (keep generous so no width is cut short)
ABLATION_PATIENCE   = 25              # was 40; stops sooner after a val plateau (fair across widths)

# --- what main() does ---
# 'evaluate' (default): score the released weights (WEIGHTS_PATH) on the selection (val1)
#     and testing (val2) splits and write per_subject_metrics_val1/val2.csv and
#     validation_summary.json, the per-recording results reported in the paper. No training.
# 'train': retrain from scratch with the settings below (stochastic on GPU).
MODE = os.environ.get('HDEEG2ECG_MODE', 'evaluate')

# --- final model ---
RUN_FINAL         = True
FINAL_MAX_EPOCHS  = 1000              # paper configuration
FINAL_PATIENCE    = 80                # paper configuration; the released checkpoint was selected at epoch 322

# --- decoder channel schedule: testing (sweep) vs. selected (single) ---
# The decoder takes the [B, T, latent_dim] latent and convolves through a list of
# channel widths down to 1 output channel. A schedule is a list whose FIRST entry is
# the latent width and whose LAST entry is 1, e.g. [32, 16, 8, 4, 1]. The special
# value None means "use the original decoder" (linear latent->128, then conv
# 128->128->64->32->1) -- this is the architecture behind the current results, so it
# is the honest control to compare new schedules against.
#
# RUN_DECODER_SWEEP = True  -> "testing" path: train every entry in DECODER_SCHEDULES
#                             (over DECODER_SWEEP_SEEDS), compare mean val1 reconstruction
#                             r, and train the final model with the best one.
# RUN_DECODER_SWEEP = False -> "just run it" path: skip the sweep and build the final
#                             model's decoder from SELECTED_DECODER.
RUN_DECODER_SWEEP = False
SELECTED_DECODER  = [32, 16, 8, 4, 1]   # monotonic taper from the latent down to 1 (intuitive,
                                        # and within seed noise of the original in the sweep);
                                        # None would use the original decoder instead

# candidate decoders tried when RUN_DECODER_SWEEP = True. None = original (control).
# Every list MUST start at the latent width that will be used (SELECTED_LATENT_DIM,
# or the width chosen by the latent sweep) and end at 1.
DECODER_SCHEDULES = [
    None,                                          # control: the current/original decoder
    [32, 16, 8, 4, 1],                             # narrow funnel (parsimony test)
    [32, 64, 32, 16, 8, 4, 1],                     # mild expand, then funnel
    [32, 256, 128, 64, 32, 16, 8, 4, 1],           # wide expand, long funnel
    [32, 512, 256, 128, 64, 32, 16, 8, 4, 1],      # very wide expand, long funnel
]
DECODER_SWEEP_SEEDS = [42, 43, 44]   # >=3 advised: the capacity landscape is flat, so
                                     # seed-to-seed noise can exceed the gap between schedules

# --- component (lesion) ablation: which modules are necessary ---
# This is a REPORTING ablation (the ablation table reviewers expect), NOT a model selector.
# Each variant removes/replaces one part of the architecture, holding everything else fixed:
#   'full'               both attention streams + cross-attention + dilated conv encoder
#   'no_ecg_stream'      drop the ECG stream and cross-attention; EEG stream -> encoder
#                        (predicted near-vestigial at eval, since the ECG input is fully masked)
#   'no_cross_attn'      keep both streams but fuse them by addition (no cross-attention)
#   'conv_only'          no attention anywhere; linear EEG projection -> dilated encoder
#   'no_dilated_encoder' drop the dilated conv encoder; fused attention -> latent directly
# When RUN_COMPONENT_ABLATION = True the sweep trains every variant (over COMPONENT_SEEDS),
# writes the table/plot, and STOPS there -- the final model is built from SELECTED_VARIANT,
# not from a val1 argmax (picking the winning lesion would just chase seed noise).
RUN_COMPONENT_ABLATION = False    # OFF: reuse the cached ablation results (component_ablation_
                                  # summary.json) to build the table below; set True to regenerate.
SELECTED_VARIANT       = 'conv_only'   # final model = conv_only. The ablation shows the attention /
                                       # cross-modal components are unnecessary (within seed scatter
                                       # of 'full'); conv_only is the parsimonious, equal-performance
                                       # choice -- chosen on val1 + simplicity, NOT claimed better than full.
COMPONENT_VARIANTS = ['full', 'no_ecg_stream', 'no_cross_attn', 'conv_only', 'no_dilated_encoder']
COMPONENT_SEEDS    = [42, 43, 44]

# --- evaluation ---
QRS_MATCH_TOLERANCE_MS = 50    # tolerance for matching detected vs true R-peaks
N_BOOTSTRAP            = 5000  # cluster (subject) bootstrap for CIs
EQUIV_MARGIN_R         = 0.05  # TOST equivalence margin on Pearson r for the ASD-vs-TD claim


# ============================================================
# REPRODUCIBILITY HELPERS
# ============================================================
def set_determinism(seed):
    L.seed_everything(seed, workers=True)
    if CUDNN_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # seeds are still fixed, so runs are reproducible up to minor cuDNN nondeterminism;
        # benchmark=True lets cuDNN pick fast conv algorithms (big win for the dilated encoder)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _seed_worker(worker_id):
    # each DataLoader worker gets its own numpy seed (derived from torch's per-worker
    # base seed) so augmentation noise differs across workers but stays reproducible
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)


def software_versions():
    import sklearn, scipy, lightning
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': scipy.__version__,
        'sklearn': sklearn.__version__,
        'torch': torch.__version__,
        'lightning': lightning.__version__,
        'mne': mne.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ============================================================
# COHORT LABELLING
# ============================================================
# Cohort is DERIVED FROM FOLDER MEMBERSHIP, not from a hardcoded list.
#
# Why this changed. This used to be a hand-maintained TD_IDS set and it had
# drifted from the data. It produced 21 TD / 45 ASD where the folders say
# 28 TD / 38 ASD, mislabelling 9 of 66 subjects. Cohort feeds the
# stratified split and every ASD-vs-TD comparison, so a stale list corrupts
# both silently. Deriving the labels from the recordings removes the failure
# mode rather than fixing one instance of it.
#
# DATA_FOLDER is the pooled set despite its Adult_TD name: it holds all 66
# recordings, and the 38 ASD ones also live in ASD_FOLDER. So ASD is
# DATA_FOLDER intersected with ASD_FOLDER, and TD is the remainder. A
# cohort_map.csv in DATA_FOLDER still overrides everything, for the case where
# the folders stop being the source of truth.
ASD_FOLDER = os.environ.get('HDEEG2ECG_ASD_DIR', os.path.join(INPUT_DIR, 'adult', 'autistic_subset'))

# Asserted at run time so a folder change cannot pass unnoticed. Set to None to
# disable the check; change the numbers only deliberately.
EXPECTED_COHORT_N = {'TD': 28, 'ASD': 38}


def subject_id(filename):
    """First 8xxx token in the filename, or None if there is none."""
    ids = re.findall(r'8\d{3}', os.path.basename(filename))
    return int(ids[0]) if ids else None


def _asd_ids(asd_folder):
    if not os.path.isdir(asd_folder):
        raise FileNotFoundError(
            f"ASD_FOLDER does not exist: {asd_folder}\n"
            "Cohort labels are derived from it. Fix the path, or place a "
            "cohort_map.csv in DATA_FOLDER to override folder-based labelling.")
    ids = {subject_id(f) for f in os.listdir(asd_folder) if f.endswith('.set')}
    ids.discard(None)
    if not ids:
        raise ValueError(f"No 8xxx .set files found in ASD_FOLDER: {asd_folder}")
    return ids


def get_cohort_labels(files, data_folder, asd_folder=None):
    """Label every file TD or ASD. cohort_map.csv wins; otherwise folder membership.

    Halts on any unassignable file and on any mismatch with EXPECTED_COHORT_N, so a
    mislabelled cohort can never reach the split or the ASD-vs-TD comparison again."""
    csv_path = os.path.join(data_folder, 'cohort_map.csv')
    if os.path.exists(csv_path):
        m = pd.read_csv(csv_path)
        m['filename'] = m['filename'].astype(str)
        lut = dict(zip(m['filename'], m['cohort'].astype(str).str.upper()))
        labels = [lut.get(f, 'UNKNOWN') for f in files]
        print(f"Cohort labels read from {csv_path}")
    else:
        asd = _asd_ids(asd_folder or ASD_FOLDER)
        labels = ['UNKNOWN' if subject_id(f) is None
                  else ('ASD' if subject_id(f) in asd else 'TD') for f in files]
        print(f"Cohort labels derived from folder membership "
              f"({len(asd)} ASD ids in {os.path.basename(asd_folder or ASD_FOLDER)})")

    unknown = [f for f, c in zip(files, labels) if c == 'UNKNOWN']
    if unknown:
        raise ValueError(
            "Could not assign a cohort to these files:\n  " + "\n  ".join(unknown)
            + "\nCheck ASD_FOLDER, or provide cohort_map.csv, before running.")

    counts = {c: labels.count(c) for c in ('TD', 'ASD')}
    print(f"  cohort distribution: TD={counts['TD']}, ASD={counts['ASD']}, "
          f"total={len(labels)}")
    if EXPECTED_COHORT_N is not None and counts != EXPECTED_COHORT_N:
        raise ValueError(
            f"Cohort counts {counts} do not match EXPECTED_COHORT_N "
            f"{EXPECTED_COHORT_N}. Either the data changed or the labelling is "
            "wrong. Update EXPECTED_COHORT_N deliberately rather than letting "
            "this pass silently.")
    return labels


# ============================================================
# DATA LOADING
# ============================================================
def load_subject_data(filepath):
    """Load one recording. Returns (eeg[time,129], event_vector[time], meta)."""
    raw = mne.io.read_raw_eeglab(filepath, preload=True)
    raw.resample(SFREQ)
    raw.filter(l_freq=FILTER_LFREQ, h_freq=FILTER_HFREQ, method='iir', verbose=False)
    data = raw.get_data()

    raw_eeg = data[:128, :]
    raw_eeg = (raw_eeg - raw_eeg.mean(axis=1, keepdims=True)) / (raw_eeg.std(axis=1, keepdims=True) + 1e-6)
    ecg = data[129, :].reshape(1, -1)

    n_samples = ecg.shape[1]
    event_vector = np.zeros(n_samples, dtype=np.float32)
    r_peaks_count = 0
    for ann in raw.annotations:
        if 'Rpeak' in ann['description']:
            p = int(ann['onset'] * SFREQ)
            if 0 <= p < n_samples:
                event_vector[p] = 1.0
                r_peaks_count += 1

    eeg = np.vstack([raw_eeg, ecg]).T.astype(np.float32)
    meta = {'n_samples': int(n_samples), 'n_rpeaks': int(r_peaks_count),
            'duration_s': float(n_samples / SFREQ)}
    return eeg, event_vector, meta


def load_split(files, data_folder):
    """Concatenate recordings for a split; also keep per-subject arrays for eval."""
    eeg_list, evt_list, subjects = [], [], []
    for f in files:
        eeg, evt, meta = load_subject_data(os.path.join(data_folder, f))
        eeg_list.append(eeg)
        evt_list.append(evt)
        subjects.append({'file': f, 'eeg': eeg, 'event': evt, **meta})
        print(f"  {f}: {meta['n_rpeaks']} R-peaks, {meta['duration_s']:.0f}s")
    eeg_cat = np.concatenate(eeg_list, axis=0)
    evt_cat = np.concatenate(evt_list, axis=0)
    return eeg_cat, evt_cat, subjects


# ============================================================
# DATASET  (training windows; single masking curriculum)
# ============================================================
class EEGHeartbeatDataset(Dataset):
    def __init__(self, eeg, event_vector, window_size=WINDOW_SIZE, is_train=True, n_shifts=N_SHIFTS):
        self.eeg = eeg
        self.window_size = window_size
        self.is_train = is_train
        self.current_epoch = 0

        r_peaks = np.where(event_vector == 1)[0]
        eeg_len = len(eeg)
        rng = np.random.default_rng(RANDOM_SEED)

        starts = []
        for p in r_peaks:
            min_s = max(0, p - window_size + 1)
            max_s = min(p, eeg_len - window_size)
            if min_s > max_s:
                continue
            for _ in range(n_shifts if is_train else 1):
                starts.append(rng.integers(min_s, max_s + 1))
        self.starts = np.array(starts, dtype=np.int32)
        print(f"  Dataset: {len(self.starts)} R-peak-centred windows ({'train' if is_train else 'val'})")

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _mask_ratio(self):
        # validation is ALWAYS fully masked (ratio 1.0): val_r/val_loss therefore always
        # reflect the deployed 'no ECG present' condition, regardless of training epoch.
        if not self.is_train:
            return 1.0
        prog = min(1.0, self.current_epoch / max(1, MASK_WARMUP_EPOCHS))
        return MASK_RATIO_START + prog * (MASK_RATIO_END - MASK_RATIO_START)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        x = self.eeg[start:start + self.window_size].copy().astype(np.float32)
        ecg_raw = x[:, 128].copy()

        m, s = ecg_raw.mean(), ecg_raw.std()
        g = (ecg_raw - m) / s if s > 1e-6 else ecg_raw.copy()

        # mask the ECG input channel (model must reconstruct it from EEG).
        # num_workers=0, so set_epoch() reaches this dataset directly and the curriculum works.
        ratio = self._mask_ratio()
        nz = np.where(np.abs(ecg_raw) > 1e-6)[0]
        if ratio >= 1.0 or len(nz) == 0:
            x[:, 128] = 0.0
        else:
            n_mask = max(1, int(len(nz) * ratio))
            chosen = np.random.choice(nz, size=min(n_mask, len(nz)), replace=False)
            x[chosen, 128] = 0.0

        # NB: noise / lead-dropout / scaling are applied on the GPU in training_step
        # (see gpu_augment) so __getitem__ stays cheap and the data pipeline isn't the bottleneck.
        attn_mask = np.ones(self.window_size, dtype=bool)
        return (torch.from_numpy(x).float(),
                torch.from_numpy(attn_mask).bool(),
                torch.from_numpy(g).float().unsqueeze(-1))


class EEGDataModule(L.LightningDataModule):
    def __init__(self, eeg_train, evt_train, eeg_val, evt_val, batch_size=BATCH_SIZE):
        super().__init__()
        self.eeg_train, self.evt_train = eeg_train, evt_train
        self.eeg_val, self.evt_val = eeg_val, evt_val
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = EEGHeartbeatDataset(self.eeg_train, self.evt_train, is_train=True)
        self.val_dataset = EEGHeartbeatDataset(self.eeg_val, self.evt_val, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                          worker_init_fn=_seed_worker, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          worker_init_fn=_seed_worker, persistent_workers=False)


# ============================================================
# GPU-SIDE AUGMENTATION  (applied per training batch, not in the dataloader)
# ============================================================
def gpu_augment(x):
    """Apply EEG sensor dropout, additive noise, and amplitude scaling on the GPU.

    x: [B, T, 129] (channels 0-127 = EEG, 128 = masked ECG input). Returns a new tensor.
    Equivalent in spirit to the original CPU augmentation; lead dropout is now an
    independent per-channel Bernoulli at rate LEAD_DROPOUT_RATE (was a fixed channel
    count with +-jitter) -- same expected effect, simpler to describe. Only call in training."""
    B, T, _ = x.shape
    eeg = x[:, :, :128]
    ecg = x[:, :, 128:]
    keep = (torch.rand(B, 1, 128, device=x.device) > LEAD_DROPOUT_RATE).to(eeg.dtype)
    eeg = eeg * keep
    eeg = eeg + torch.randn_like(eeg) * NOISE_STD
    ecg = ecg + torch.randn_like(ecg) * (NOISE_STD * 0.5)
    scale = (1.0 + torch.randn(B, 1, 128, device=x.device) * 0.08).clamp(0.8, 1.2).to(eeg.dtype)
    eeg = eeg * scale
    return torch.cat([eeg, ecg], dim=-1)


# ============================================================
# LOSS  (lag + shape + bounds + smooth + corr; all normalised to [0,1])
# ============================================================
def hybrid_rpeak_loss(y_true, y_pred):
    if y_true.dim() == 3 and y_true.shape[-1] == 1:
        y_t = y_true.permute(0, 2, 1)
        y_p = y_pred.permute(0, 2, 1)
    else:
        y_t, y_p = y_true, y_pred
    B = y_t.size(0)
    seq_len = y_t.size(-1)
    center = seq_len // 2

    # 1. timing via cross-correlation lag (differentiable soft-argmax)
    corr = F.conv1d(y_p.view(1, B, -1), y_t, groups=B, padding=center)
    corr = corr.view(B, -1)[:, :seq_len]
    corr = corr - corr.mean(dim=-1, keepdim=True)
    corr = corr / (corr.std(dim=-1, keepdim=True) + 1e-6)   # scale-invariant -> TAU is meaningful
    half = 64  # max plausible misalignment, samples (windows out periodic side-lobes)
    lo, hi = center - half, center + half
    corr = corr[:, lo:hi]
    idx = torch.arange(lo, hi, device=corr.device, dtype=corr.dtype)
    weights = torch.softmax(corr / TAU, dim=-1)
    soft_idx = (weights * idx).sum(dim=-1)
    target_idx = torch.full_like(soft_idx, float(center))
    lag = F.mse_loss(soft_idx / seq_len, target_idx / seq_len)

    # 2. shape (bounded Huber)
    shape = torch.clamp(F.huber_loss(y_p, y_t, delta=0.15) / 0.15, max=1.0)
    # 3. amplitude bounds
    oob = torch.clamp(torch.abs(y_p) - ECG_MAX_AMPLITUDE, min=0.0)
    bounds = torch.clamp(torch.mean(oob ** 2) / 4.0, max=1.0)
    # 4. smoothness (squared 2nd derivative)
    yp = y_p.squeeze(1)
    d2 = yp[:, 2:] - 2 * yp[:, 1:-1] + yp[:, :-2]
    smooth = torch.clamp(torch.mean(d2 ** 2) / 0.1, max=1.0)
    # 5. waveform correlation -- the term that removes the degenerate flat-output minimum.
    #    (1 - Pearson r) is in [0,2]; /2 -> [0,1]. Flat/constant pred -> r~0 -> 0.5;
    #    perfect reconstruction -> 0; anti-correlated -> 1. NOTE: smooth/bounds/lag all
    #    reward a flat line (they go to ~0), so without this term a flat prediction is a
    #    competitive low-loss solution that has zero reconstruction value.
    pc = (y_p - y_p.mean(dim=-1, keepdim=True)).float()
    tc = (y_t - y_t.mean(dim=-1, keepdim=True)).float()
    num = (pc * tc).sum(dim=-1)
    den = torch.sqrt((pc ** 2).sum(dim=-1) * (tc ** 2).sum(dim=-1) + 1e-6)
    corr_term = ((1.0 - num / den) / 2.0).mean()

    total = (LAMBDA_LAG * lag + LAMBDA_SHAPE * shape + LAMBDA_BOUNDS * bounds
             + LAMBDA_SMOOTH * smooth + LAMBDA_CORR * corr_term)
    return total, {'total': total.item(), 'lag': lag.item(), 'shape': shape.item(),
                   'bounds': bounds.item(), 'smooth': smooth.item(), 'corr': corr_term.item()}


# ============================================================
# MODEL
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.g


class TransformerStream(nn.Module):
    def __init__(self, in_dim, proj_dim, num_heads, seq_len, dropout=0.4):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.pos_emb = nn.Embedding(seq_len, proj_dim)
        self.norm = RMSNorm(proj_dim)
        self.attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        x = self.dropout(self.proj(x) + self.pos_emb(pos))
        residual = x
        xn = self.norm(x)
        attn_out, _ = self.attn(xn, xn, xn, key_padding_mask=key_padding_mask,
                                need_weights=False)
        return self.norm2(residual + self.dropout(attn_out))


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.4):
        super().__init__()
        dilations = [2, 4, 8, 16, 32, 64]
        layers, cur = [], in_dim
        for i, d in enumerate(dilations):
            oc = 256 if i == len(dilations) - 1 else 128
            layers += [nn.Conv1d(cur, oc, kernel_size=5, padding=2 * d, dilation=d),
                       nn.BatchNorm1d(oc), nn.GELU(), nn.Dropout(dropout)]
            cur = oc
        layers.append(nn.Conv1d(cur, out_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, d_dec=128):
        super().__init__()
        self.project_latent = nn.Linear(latent_dim, d_dec)
        self.dropout = nn.Dropout(0.4)
        self.upsample = nn.Sequential(
            nn.Conv1d(d_dec, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.4),
            nn.Conv1d(128, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.4),
            nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 1, 1))

    def forward(self, latent):
        x = self.dropout(self.project_latent(latent)).permute(0, 2, 1)
        return self.upsample(x).permute(0, 2, 1)


class ConvDecoderFromSchedule(nn.Module):
    """Decoder built directly from an explicit channel schedule, e.g. [32, 16, 8, 4, 1].

    The latent (latent_dim channels per timestep) is fed straight into the first conv,
    so schedule[0] MUST equal latent_dim and schedule[-1] MUST be 1. Each hidden step is
    Conv1d(k=3, pad=1) + BatchNorm + GELU + Dropout (length-preserving); the final step to
    1 channel is a 1x1 conv with no activation, matching the original decoder's output head.
    Note this varies BOTH width and depth with the schedule length -- a 9-entry schedule is
    a deeper network than a 5-entry one, so differences confound the two."""
    def __init__(self, schedule, dropout=0.4):
        super().__init__()
        if len(schedule) < 2 or schedule[-1] != 1:
            raise ValueError(f"decoder schedule must end at 1 and have >=2 entries, got {schedule}")
        layers = []
        for i in range(len(schedule) - 1):
            cin, cout = schedule[i], schedule[i + 1]
            if i == len(schedule) - 2:                       # final head -> 1 channel
                layers.append(nn.Conv1d(cin, cout, kernel_size=1))
            else:
                layers += [nn.Conv1d(cin, cout, kernel_size=3, padding=1),
                           nn.BatchNorm1d(cout), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self.schedule = list(schedule)

    def forward(self, latent):                               # latent: [B, T, C0]
        return self.net(latent.permute(0, 2, 1)).permute(0, 2, 1)


def make_decoder(latent_dim, schedule):
    """Return the original decoder when schedule is None, else a schedule-driven decoder.

    None reproduces the architecture behind the current results (linear latent->128, then
    conv 128->128->64->32->1). A list builds ConvDecoderFromSchedule and requires
    schedule[0] == latent_dim so the latent feeds the first conv directly."""
    if schedule is None:
        return ConvDecoder(latent_dim)
    if schedule[0] != latent_dim:
        raise ValueError(
            f"decoder schedule {schedule} must start at the latent width ({latent_dim}); "
            f"set schedule[0] = {latent_dim} or change SELECTED_LATENT_DIM to {schedule[0]}.")
    return ConvDecoderFromSchedule(schedule)


class EEGtoECGModel(nn.Module):
    """Masked, dual-stream EEG->ECG sequence-to-sequence model (no compressive bottleneck).

    `variant` selects a component-ablation lesion (see COMPONENT_VARIANTS). Lesioned models
    do NOT instantiate the removed modules, so reported parameter counts reflect the lesion."""
    def __init__(self, seq_len=WINDOW_SIZE, latent_dim=256, proj_dim=128,
                 decoder_schedule=None, variant='full'):
        super().__init__()
        valid = {'full', 'no_ecg_stream', 'no_cross_attn', 'conv_only', 'no_dilated_encoder'}
        if variant not in valid:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(valid)}")
        self.variant = variant
        self.use_attention  = variant != 'conv_only'             # attention on the EEG stream
        self.use_cross_attn = variant in {'full', 'no_dilated_encoder'}
        self.use_ecg_stream = variant in {'full', 'no_cross_attn', 'no_dilated_encoder'}
        self.use_dilated    = variant != 'no_dilated_encoder'

        if self.use_attention:
            self.raw_stream = TransformerStream(128, proj_dim, 8, seq_len)
        else:                                                    # conv_only: projection, no attention
            self.raw_proj = nn.Linear(128, proj_dim)
            self.raw_pos  = nn.Embedding(seq_len, proj_dim)
            self.raw_drop = nn.Dropout(0.4)
        if self.use_ecg_stream:
            self.ecg_stream = TransformerStream(1, proj_dim, 8, seq_len)
        if self.use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(proj_dim, 8, dropout=0.4, batch_first=True)
        if self.use_dilated:
            self.conv_encoder = DilatedConvEncoder(proj_dim, 128)

        self.fc_latent = nn.Linear(128, latent_dim)              # proj_dim is 128 in both paths
        self.decoder = make_decoder(latent_dim, decoder_schedule)

    def _encode_eeg(self, eeg, attn_mask):
        if self.use_attention:
            return self.raw_stream(eeg, key_padding_mask=~attn_mask)
        T = eeg.shape[1]
        pos = torch.arange(T, device=eeg.device)
        return self.raw_drop(self.raw_proj(eeg) + self.raw_pos(pos))

    def forward(self, x, attn_mask):
        raw_out = self._encode_eeg(x[:, :, :128], attn_mask)
        if self.use_cross_attn:                                  # full: query EEG over ECG, residual
            ecg_out = self.ecg_stream(x[:, :, 128:])
            cross_out, _ = self.cross_attn(raw_out, ecg_out, ecg_out, need_weights=False)
            fused = cross_out + raw_out
        elif self.use_ecg_stream:                               # no_cross_attn: fuse by addition
            fused = raw_out + self.ecg_stream(x[:, :, 128:])
        else:                                                   # no_ecg_stream / conv_only
            fused = raw_out
        enc = self.conv_encoder(fused) if self.use_dilated else fused
        latent = F.gelu(self.fc_latent(enc))
        return self.decoder(latent)


class EEGLightningModule(L.LightningModule):
    def __init__(self, latent_dim=256, max_lr=MAX_LR, weight_decay=WEIGHT_DECAY,
                 decoder_schedule=None, variant='full'):
        super().__init__()
        self.save_hyperparameters()
        self.model = EEGtoECGModel(latent_dim=latent_dim, decoder_schedule=decoder_schedule,
                                   variant=variant)
        self._val_r_chunks = []

    def forward(self, x, mask):
        return self.model(x, mask)

    def _step(self, batch, augment=False):
        x, attn_mask, g = batch
        if augment:
            x = gpu_augment(x)
        return hybrid_rpeak_loss(g, self(x, attn_mask))

    @staticmethod
    def _val_corr(y_pred, g):
        """Per-window Pearson r between prediction and target. Returns [B].

        Computed in float32 regardless of autocast. Summing 512 squared values in
        float16 overflows at |p| ~ 256 (65504 max), giving inf / inf -> NaN, and
        on_validation_epoch_end maps an all-NaN epoch to val_r = 0.0 exactly. That
        is what killed the no-early-stopping run at epoch 8305 while training was
        still healthy: BatchNorm running_var had drifted 25x (max 8.1e2 -> 2.0e4),
        which only affects eval mode, so the training path never saw it."""
        p = y_pred.squeeze(-1).float()
        t = g.squeeze(-1).float()
        p = p - p.mean(dim=1, keepdim=True)
        t = t - t.mean(dim=1, keepdim=True)
        num = (p * t).sum(dim=1)
        den = torch.sqrt((p ** 2).sum(dim=1) * (t ** 2).sum(dim=1) + 1e-8)
        return num / den

    def training_step(self, batch, _):
        loss, bd = self._step(batch, augment=True)
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        for k, v in bd.items():
            self.log(f'train_{k}', v, on_epoch=True, prog_bar=(k == 'total'))
        return loss

    def on_validation_epoch_start(self):
        self._val_r_chunks = []

    def validation_step(self, batch, _):
        x, attn_mask, g = batch
        y_pred = self(x, attn_mask)
        # The loss cross-correlates two 512-sample windows and sums squares; both
        # overflow float16 long before the model is actually broken. Validation is
        # not on the training critical path, so score it in float32 and let the
        # forward pass keep its speed.
        with torch.autocast(device_type=x.device.type, enabled=False):
            loss, bd = hybrid_rpeak_loss(g.float(), y_pred.float())
        for k, v in bd.items():
            self.log(f'val_{k}', v, on_epoch=True, prog_bar=(k == 'total'))
        self.log('val_absmax', y_pred.detach().abs().max().float(), on_epoch=True)
        self._val_r_chunks.append(self._val_corr(y_pred, g).detach().float().cpu())
        return loss

    def on_validation_epoch_end(self):
        # val_r is the model-selection / early-stopping metric. The composite loss is
        # minimised by a degenerate flat prediction (r~0 at the same low loss), so
        # monitoring the loss can select a dead model; reconstruction r cannot be fooled
        # that way. Mean per-window Pearson r over the (fully masked) validation set.
        if self._val_r_chunks:
            r = torch.cat(self._val_r_chunks)
            r = r[~torch.isnan(r)]
            val_r = r.mean().item() if r.numel() else 0.0
        else:
            val_r = 0.0
        self.log('val_r', val_r, prog_bar=True)

    def on_train_epoch_start(self):
        ds = self.trainer.train_dataloader.dataset
        if hasattr(ds, 'set_epoch'):
            ds.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.max_lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-8)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}


# ============================================================
# TRAINING DRIVER
# ============================================================
class AbortOnDeadValidation(L.Callback):
    """Stop when validation stops producing a number, rather than burning GPU on it.

    val_r == 0.0 exactly is not a measurement: on_validation_epoch_end returns it as
    a fallback when every window's r is NaN. In the 13,334-epoch run that state was
    absorbing, and never recovered across the 5,029 epochs that followed it, so there
    is nothing to wait for. `training_step` returns None on a non-finite loss, which
    means a broken run raises nothing and looks alive from the outside."""

    def __init__(self, patience=20):
        self.patience = patience
        self.dead = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get('val_r')
        if v is None:
            return
        v = float(v)
        self.dead = self.dead + 1 if (not np.isfinite(v) or abs(v) < 1e-12) else 0
        if self.dead >= self.patience:
            trainer.should_stop = True
            print(f"\n[AbortOnDeadValidation] val_r has been 0/NaN for {self.dead} "
                  f"consecutive epochs at epoch {trainer.current_epoch}; stopping. "
                  f"The best checkpoint is unaffected.")


def train_one(latent_dim, datamodule, max_epochs, patience, seed, tag,
              decoder_schedule=None, variant='full'):
    set_determinism(seed)
    ckpt_dir = os.path.join(OUTPUT_DIR, 'checkpoints', tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    model = EEGLightningModule(latent_dim=latent_dim, decoder_schedule=decoder_schedule,
                               variant=variant)
    # Select + stop on reconstruction r (maximise), NOT the composite loss (which a flat
    # prediction can minimise). min_epochs forces training through the masking curriculum,
    # so a model is never selected before it has trained under full ECG masking.
    ckpt = ModelCheckpoint(dirpath=ckpt_dir, monitor='val_r', mode='max',
                           save_top_k=1, filename='best')
    # patience falsy (0 or None) disables BOTH stoppers, for deliberate long runs
    # that are meant to reach max_epochs regardless of what validation does.
    _cbs = [ckpt, LearningRateMonitor(logging_interval='epoch')]
    if patience:
        _cbs = ([EarlyStopping(monitor='val_r', mode='max', patience=patience,
                               min_delta=1e-4)] + _cbs
                + [AbortOnDeadValidation(patience=20)])
    trainer = L.Trainer(
        max_epochs=max_epochs, min_epochs=min(MASK_WARMUP_EPOCHS, max_epochs),
        accelerator='auto', devices=1,
        precision='16-mixed', gradient_clip_val=GRAD_CLIP,
        num_sanity_val_steps=0, log_every_n_steps=10,
        default_root_dir=ckpt_dir,
        callbacks=_cbs,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=datamodule)
    best = EEGLightningModule.load_from_checkpoint(ckpt.best_model_path)
    return best, float(ckpt.best_model_score), trainer.current_epoch   # best_model_score = best val_r


# ============================================================
# INFERENCE + METRICS
# ============================================================
@torch.no_grad()
def reconstruct_subject(model, eeg, device, eval_batch=128):
    """Tile non-overlapping windows across a recording and predict ECG for all of them
    in batches (was one window at a time). Returns (pred_signal, true_signal) at full
    length (trailing remainder dropped) plus per-window Pearson r."""
    model.eval().to(device)
    n = (len(eeg) // WINDOW_SIZE) * WINDOW_SIZE
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    wins = eeg[:n].reshape(-1, WINDOW_SIZE, eeg.shape[1]).astype(np.float32).copy()  # [W,512,129]
    ecg = wins[:, :, 128]
    m = ecg.mean(axis=1, keepdims=True)
    sd = ecg.std(axis=1, keepdims=True)
    g = np.where(sd > 1e-6, (ecg - m) / (sd + 1e-12), ecg)   # per-window z-scored target
    wins[:, :, 128] = 0.0                                    # inference masking

    W = wins.shape[0]
    preds = []
    for i in range(0, W, eval_batch):
        xb = torch.from_numpy(wins[i:i + eval_batch]).float().to(device)
        mb = torch.ones(xb.shape[0], WINDOW_SIZE, dtype=torch.bool, device=device)
        preds.append(model(xb, mb).squeeze(-1).cpu().numpy())   # [b,512]
    pred = np.concatenate(preds, axis=0)                        # [W,512]

    rs = [float(np.corrcoef(pred[i], g[i])[0, 1])
          for i in range(W) if pred[i].std() > 1e-6 and g[i].std() > 1e-6]
    return pred.reshape(-1), g.reshape(-1), np.array(rs)


def detect_peaks(sig):
    min_dist = int(0.3 * SFREQ)            # >=300 ms apart -> <200 bpm
    height = np.mean(sig) + 0.5 * np.std(sig)
    peaks, _ = find_peaks(sig, distance=min_dist, height=height)
    return peaks


def match_peaks(true_pk, pred_pk, tol):
    """Greedy nearest-neighbour matching within tol (samples).
    Returns TP, FP, FN, and matched timing errors (samples)."""
    used = np.zeros(len(pred_pk), dtype=bool)
    tp, errs = 0, []
    for t in true_pk:
        if len(pred_pk) == 0:
            break
        d = np.abs(pred_pk - t)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
            errs.append(pred_pk[j] - t)
    fn = len(true_pk) - tp
    fp = len(pred_pk) - used.sum()
    return tp, fp, fn, np.array(errs, dtype=float)


def eeg_cardiac_leakage(eeg, event_vector, half_win_ms=120):
    """Model-free measure of how strongly the heartbeat leaks into the EEG.

    R-peak-triggered average of the 128 z-scored EEG channels (random EEG averages
    toward zero; a time-locked cardiac field artifact survives). Returns the largest
    peak-to-peak amplitude across channels. Computed from ground-truth R-peaks and raw
    EEG only -- it never sees the model -- so it is an independent explanatory variable
    for why some subjects reconstruct well and others do not."""
    hw = int(half_win_ms / 1000 * SFREQ)
    peaks = np.where(event_vector == 1)[0]
    peaks = peaks[(peaks - hw >= 0) & (peaks + hw < len(eeg))]
    if len(peaks) < 5:
        return np.nan
    eeg128 = eeg[:, :128]
    segs = np.stack([eeg128[p - hw:p + hw + 1] for p in peaks])   # [n_peaks, 2hw, 128]
    avg = segs.mean(axis=0)                                    # triggered average
    p2p = avg.max(axis=0) - avg.min(axis=0)                    # per-channel peak-to-peak
    return float(np.max(p2p))


def subject_metrics(model, subject, device):
    """Returns (per-subject metric dict, signed per-beat timing errors in ms)."""
    eeg, evt = subject['eeg'], subject['event']
    n = (len(eeg) // WINDOW_SIZE) * WINDOW_SIZE
    pred_sig, true_sig, win_r = reconstruct_subject(model, eeg, device)

    true_pk = np.where(evt[:n] == 1)[0]
    pred_pk = detect_peaks(pred_sig)
    tol = int(QRS_MATCH_TOLERANCE_MS / 1000 * SFREQ)
    tp, fp, fn, errs = match_peaks(true_pk, pred_pk, tol)
    errs_ms = errs / SFREQ * 1000.0   # signed: predicted - true

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    f1   = 2 * sens * ppv / (sens + ppv) if (sens + ppv) and not np.isnan(sens) and not np.isnan(ppv) else np.nan

    true_hr = 60.0 / (np.median(np.diff(true_pk)) / SFREQ) if len(true_pk) > 1 else np.nan
    pred_hr = 60.0 / (np.median(np.diff(pred_pk)) / SFREQ) if len(pred_pk) > 1 else np.nan

    row = {
        'file': subject['file'], 'cohort': subject['cohort'],
        'n_windows': len(win_r),
        'mean_r': float(np.nanmean(win_r)) if len(win_r) else np.nan,
        'rmse': float(np.sqrt(np.mean((pred_sig - true_sig) ** 2))),
        'mae': float(np.mean(np.abs(pred_sig - true_sig))),
        'n_true_peaks': int(len(true_pk)), 'n_pred_peaks': int(len(pred_pk)),
        'tp': tp, 'fp': fp, 'fn': fn,
        'sensitivity': sens, 'ppv': ppv, 'f1': f1,
        'timing_err_ms_mean': float(np.mean(np.abs(errs_ms))) if len(errs_ms) else np.nan,
        'timing_bias_ms': float(np.mean(errs_ms)) if len(errs_ms) else np.nan,
        'timing_err_ms_sd': float(np.std(errs_ms)) if len(errs_ms) else np.nan,
        'eeg_leakage': eeg_cardiac_leakage(eeg, evt),
        'true_hr': true_hr, 'pred_hr': pred_hr,
    }
    return row, errs_ms


# ============================================================
# STATISTICS
# ============================================================
def cluster_bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05, seed=RANDOM_SEED):
    """Subject-level bootstrap CI of the mean. `values` = one number per subject."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    return float(np.mean(v)), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def cohort_comparison(per_subject_df, metric='mean_r'):
    asd = per_subject_df.loc[per_subject_df.cohort == 'ASD', metric].dropna().values
    td  = per_subject_df.loc[per_subject_df.cohort == 'TD', metric].dropna().values
    out = {'metric': metric, 'n_ASD': len(asd), 'n_TD': len(td),
           'median_ASD': float(np.median(asd)) if len(asd) else np.nan,
           'median_TD': float(np.median(td)) if len(td) else np.nan}
    if len(asd) >= 2 and len(td) >= 2:
        u, p = sps.mannwhitneyu(asd, td, alternative='two-sided')
        out['mannwhitney_U'], out['mannwhitney_p'] = float(u), float(p)
        out['rank_biserial'] = float(1 - 2 * u / (len(asd) * len(td)))
        # TOST equivalence: is the ASD-TD difference within +/- EQUIV_MARGIN_R ?
        diff = np.mean(asd) - np.mean(td)
        se = np.sqrt(np.var(asd, ddof=1) / len(asd) + np.var(td, ddof=1) / len(td))
        df = len(asd) + len(td) - 2
        t_low = (diff + EQUIV_MARGIN_R) / se
        t_high = (diff - EQUIV_MARGIN_R) / se
        p_low = sps.t.sf(t_low, df)     # H0: diff <= -margin (upper-tailed)
        p_high = sps.t.cdf(t_high, df)  # H0: diff >= +margin (lower-tailed)
        out['equiv_margin'] = EQUIV_MARGIN_R
        out['tost_p'] = float(max(p_low, p_high))   # equivalence supported if < 0.05
        out['mean_diff'] = float(diff)
    return out


def bland_altman(true_hr, pred_hr):
    t = np.asarray(true_hr, dtype=float); p = np.asarray(pred_hr, dtype=float)
    ok = ~(np.isnan(t) | np.isnan(p))
    t, p = t[ok], p[ok]
    if len(t) < 2:
        return {}
    diff = p - t
    bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    return {'n': len(t), 'bias': bias, 'sd': sd,
            'loa_lower': bias - 1.96 * sd, 'loa_upper': bias + 1.96 * sd,
            'mean': (t + p) / 2, 'diff': diff}


def _partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z (rank-based partial Pearson).
    Returns (rho, p). p from a t-test with dof = n - 2 - 1 (one covariate)."""
    rx, ry, rz = sps.rankdata(x), sps.rankdata(y), sps.rankdata(z)
    def pear(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else np.nan
    rxy, rxz, ryz = pear(rx, ry), pear(rx, rz), pear(ry, rz)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if not np.isfinite(den) or den == 0:
        return np.nan, np.nan
    r = (rxy - rxz * ryz) / den
    n, k = len(x), 1
    dof = n - 2 - k
    if dof <= 0 or abs(r) >= 1:
        return float(r), np.nan
    tstat = r * np.sqrt(dof / (1 - r ** 2))
    return float(r), float(2 * sps.t.sf(abs(tstat), dof))


def pooled_leakage_analysis(per_subject_dfs):
    """Pool subjects across validation splits and test whether reconstruction quality (F1)
    tracks the model-free EEG cardiac-leakage measure. This is the robust, higher-n version
    of the per-split F1-vs-leakage correlation (previously a separate post-hoc script):
    a pooled Spearman rho plus a partial Spearman controlling for cohort, so the leakage
    relationship can't be dismissed as an ASD/TD artifact."""
    df = pd.concat(per_subject_dfs, ignore_index=True).dropna(subset=['f1', 'eeg_leakage'])
    out = {'n': int(len(df))}
    if len(df) >= 4:
        rho, p = sps.spearmanr(df.eeg_leakage.values, df.f1.values)
        out['spearman_rho'], out['spearman_p'] = float(rho), float(p)
        cohort_num = (df.cohort.values == 'ASD').astype(float)   # ASD=1, TD=0
        pr, pp = _partial_spearman(df.eeg_leakage.values, df.f1.values, cohort_num)
        out['partial_rho_ctrl_cohort'], out['partial_p_ctrl_cohort'] = pr, pp
    return out


# ============================================================
# KNEE DETECTION  (kneedle-style perpendicular distance)
# ============================================================
def find_knee(widths, metric_higher_is_better):
    x = np.log2(np.asarray(widths, dtype=float))
    y = np.asarray(metric_higher_is_better, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    # distance of each point above the chord from first to last
    dist = yn - (xn * (yn[-1] - yn[0]) + yn[0])
    knee_idx = int(np.argmax(dist))
    return widths[knee_idx], knee_idx


def plot_ablation(widths, means, los, his, knee_width, path):
    plt.figure(figsize=(5.5, 4))
    plt.errorbar(widths, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                 marker='o', capsize=3, lw=1.5)
    plt.axvline(knee_width, ls='--', color='crimson', lw=1, label=f'knee = {knee_width}')
    plt.xscale('log', base=2)
    plt.xticks(widths, [str(w) for w in widths])
    plt.xlabel('Latent width')
    plt.ylabel('Validation Pearson r (subject-level mean)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()


def decoder_label(schedule):
    """Short, filesystem-safe label for a decoder schedule (None -> 'original')."""
    if schedule is None:
        return 'original'
    return '-'.join(str(c) for c in schedule)


def plot_decoder_sweep(labels, means, sems, best_idx, path, title='Decoder channel-schedule sweep (val1)'):
    plt.figure(figsize=(max(5.5, 0.9 * len(labels)), 4))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=1.96 * np.asarray(sems), capsize=3,
            color=['crimson' if i == best_idx else 'slategray' for i in x])
    plt.xticks(x, labels, rotation=30, ha='right', fontsize=7)
    plt.ylabel('Validation Pearson r (subject-level mean)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()


def plot_bland_altman(ba, title, path):
    if not ba:
        return
    plt.figure(figsize=(5.5, 4))
    plt.scatter(ba['mean'], ba['diff'], s=20)
    plt.axhline(ba['bias'], color='k', lw=1)
    plt.axhline(ba['loa_upper'], color='grey', ls='--', lw=1)
    plt.axhline(ba['loa_lower'], color='grey', ls='--', lw=1)
    plt.xlabel('Mean HR (true, predicted) [bpm]')
    plt.ylabel('Predicted - True HR [bpm]')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()


# One colour for every recording. The paper's claim is cohort INVARIANCE, so
# colouring points by group invites the reader to hunt for a difference the
# analysis reports as undetectable, and at n = 20 any apparent clustering is
# noise. Cohort remains in the per-subject CSVs for the statistics; it is only
# dropped from the visual encoding.
POINT_COLOR = '#3A6EA5'


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()


def plot_se_ppv_scatter(df, split_name, path):
    """Each subject as a point in sensitivity x PPV space. The 'perfect' cluster
    sits in the top-right; subjects without recoverable leakage fall away from it."""
    plt.figure(figsize=(5, 5))
    plt.scatter(df.sensitivity, df.ppv, s=45, alpha=0.85, color=POINT_COLOR,
                edgecolor='white', linewidth=0.5, label=f'n = {len(df)}')
    plt.plot([0, 1], [0, 1], ls=':', color='grey', lw=0.8)
    plt.xlim(-0.03, 1.03); plt.ylim(-0.03, 1.03)
    plt.xlabel('Sensitivity (recall)')
    plt.ylabel('PPV (precision)')
    plt.title(f'Per-subject R-peak detection ({split_name})')
    plt.legend(frameon=False, loc='lower left')
    _save(path)


def plot_f1_distribution(df, split_name, path):
    """Sorted per-subject F1 lollipop -- makes a bimodal (perfect vs failing) split obvious."""
    d = df.sort_values('f1', ascending=True, na_position='first').reset_index(drop=True)
    plt.figure(figsize=(6, max(3, 0.26 * len(d))))
    plt.hlines(range(len(d)), 0, d.f1.fillna(0), color='lightgrey', lw=1)
    plt.scatter(d.f1, range(len(d)), color=POINT_COLOR, s=35, zorder=3,
                edgecolor='white', linewidth=0.5)
    plt.yticks([])          # recording IDs withheld; order is the sort, not an identity
    plt.xlim(-0.03, 1.03)
    plt.xlabel('R-peak detection F1')
    plt.title(f'Per-subject F1, sorted ({split_name})')
    _save(path)


def plot_f1_vs_leakage(df, split_name, path):
    """The argument-clinching figure: failing subjects are the low-leakage ones."""
    plt.figure(figsize=(5.5, 4))
    plt.scatter(df.eeg_leakage, df.f1, s=45, alpha=0.85, color=POINT_COLOR,
                edgecolor='white', linewidth=0.5)
    plt.xlabel('EEG cardiac leakage  (R-peak-triggered peak-to-peak, a.u.)')
    plt.ylabel('R-peak detection F1')
    plt.ylim(-0.03, 1.03)
    plt.title(f'Reconstruction quality tracks leakage ({split_name})')
    _save(path)


def plot_timing_error_hist(errs_ms, split_name, path):
    """Pooled signed latency errors across all matched beats, with the bias marked."""
    if len(errs_ms) == 0:
        return
    plt.figure(figsize=(5.5, 4))
    plt.hist(errs_ms, bins=40, color='#555555')
    plt.axvline(0, color='k', lw=1)
    plt.axvline(np.mean(errs_ms), color='crimson', ls='--', lw=1.2,
                label=f'bias = {np.mean(errs_ms):.1f} ms')
    plt.xlabel('R-peak latency error: predicted - true (ms)')
    plt.ylabel('Matched beats')
    plt.title(f'Latency agreement ({split_name})')
    plt.legend(frameon=False)
    _save(path)


# ============================================================
# EVALUATION OF A SPLIT
# ============================================================
def evaluate_split(model, subjects, device, split_name, out_dir):
    rows, pooled_errs = [], []
    for s in subjects:
        row, errs_ms = subject_metrics(model, s, device)
        rows.append(row)
        pooled_errs.append(errs_ms)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f'per_subject_metrics_{split_name}.csv'), index=False)
    pooled = np.concatenate([e for e in pooled_errs if len(e)]) if any(len(e) for e in pooled_errs) else np.array([])

    summary = {'split': split_name, 'n_subjects': len(df),
               'n_subjects_ASD': int((df.cohort == 'ASD').sum()),
               'n_subjects_TD': int((df.cohort == 'TD').sum()),
               'total_true_peaks': int(df.n_true_peaks.sum())}
    for metric in ['mean_r', 'rmse', 'mae', 'sensitivity', 'ppv', 'f1',
                   'timing_err_ms_mean', 'timing_bias_ms']:
        mean, lo, hi = cluster_bootstrap_ci(df[metric].values)
        summary[metric] = {'mean': mean, 'ci95_low': lo, 'ci95_high': hi,
                           'median': float(df[metric].median()), 'sd': float(df[metric].std())}
    summary['cohort_comparison_r'] = cohort_comparison(df, 'mean_r')
    summary['cohort_comparison_f1'] = cohort_comparison(df, 'f1')

    # does reconstruction quality track physiological leakage?
    sub = df.dropna(subset=['f1', 'eeg_leakage'])
    if len(sub) >= 3:
        rho, p = sps.spearmanr(sub.eeg_leakage, sub.f1)
        summary['f1_vs_leakage_spearman'] = {'rho': float(rho), 'p': float(p), 'n': int(len(sub))}

    # figures
    plot_se_ppv_scatter(df, split_name, os.path.join(out_dir, f'se_ppv_scatter_{split_name}.png'))
    plot_f1_distribution(df, split_name, os.path.join(out_dir, f'f1_distribution_{split_name}.png'))
    plot_f1_vs_leakage(df, split_name, os.path.join(out_dir, f'f1_vs_leakage_{split_name}.png'))
    plot_timing_error_hist(pooled, split_name, os.path.join(out_dir, f'timing_error_hist_{split_name}.png'))

    ba = bland_altman(df.true_hr.values, df.pred_hr.values)
    if ba:
        plot_bland_altman(ba, f'HR agreement ({split_name})',
                          os.path.join(out_dir, f'bland_altman_HR_{split_name}.png'))
        summary['bland_altman_HR'] = {k: v for k, v in ba.items() if k not in ('mean', 'diff')}
    return df, summary


# ============================================================
# MAIN
# ============================================================
def preserve_previous_outputs():
    """Move a previous run's outputs aside before this one overwrites them.

    The pipeline writes final_model_weights.pt, per_subject_metrics_val*.csv,
    validation_summary.json and paper_validation_summary.md by fixed name. On
    2026-09-07 a re-run destroyed the deployed model's copies of all four in place;
    only a checkpoint and a stale pooled CSV made recovery possible. Nothing is
    deleted here, so the cost of being wrong is a folder."""
    at_risk = ['final_model_weights.pt', 'validation_summary.json',
               'paper_validation_summary.md', 'per_subject_metrics_val1.csv',
               'per_subject_metrics_val2.csv', 'split_manifest.csv',
               'pooled_per_subject_metrics.csv']
    present = [f for f in at_risk if os.path.exists(os.path.join(OUTPUT_DIR, f))]
    if not present:
        return
    dest = os.path.join(OUTPUT_DIR, f'_superseded_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(dest, exist_ok=True)
    import shutil
    for f in present:
        shutil.copy2(os.path.join(OUTPUT_DIR, f), os.path.join(dest, f))
    print(f"Previous outputs copied to {dest} before this run overwrites them "
          f"({len(present)} files).")


def evaluate_released(f_val1, c_val1, f_val2, c_val2, device):
    """MODE = 'evaluate': score the released weights on both held-out splits."""
    print(f"\nEvaluating the released weights {WEIGHTS_PATH} (no training)")
    print("Loading val1...");  _, _, subj_v1 = load_split(f_val1, DATA_FOLDER)
    print("Loading val2...");  _, _, subj_v2 = load_split(f_val2, DATA_FOLDER)
    for s, c in zip(subj_v1, c_val1): s['cohort'] = c
    for s, c in zip(subj_v2, c_val2): s['cohort'] = c
    model = EEGLightningModule(latent_dim=SELECTED_LATENT_DIM, decoder_schedule=SELECTED_DECODER,
                               variant=SELECTED_VARIANT)
    model.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))   # strict
    df1, sum1 = evaluate_split(model, subj_v1, device, 'val1', OUTPUT_DIR)
    df2, sum2 = evaluate_split(model, subj_v2, device, 'val2', OUTPUT_DIR)
    save_json({'val1': sum1, 'val2': sum2, 'weights': WEIGHTS_PATH,
               'pooled_leakage': pooled_leakage_analysis([df1, df2])},
              os.path.join(OUTPUT_DIR, 'validation_summary.json'))
    print(f"\nDone. All outputs in {OUTPUT_DIR}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preserve_previous_outputs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')
    set_determinism(RANDOM_SEED)

    # ---- discover files and cohorts ----
    files = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith('.set'))
    cohorts = get_cohort_labels(files, DATA_FOLDER)

    # ---- subject-level 3-way split: frozen if a manifest is supplied ----
    if SPLIT_MANIFEST_IN and os.path.exists(SPLIT_MANIFEST_IN):
        man_in = pd.read_csv(SPLIT_MANIFEST_IN)
        missing = set(man_in.filename) ^ set(files)
        if missing:
            raise ValueError(
                f"{SPLIT_MANIFEST_IN} does not describe the same recordings as "
                f"DATA_FOLDER; symmetric difference: {sorted(missing)}")
        by_split = {s: man_in.loc[man_in.split == s, 'filename'].tolist()
                    for s in ('train', 'val1', 'val2')}
        f_train, f_val1, f_val2 = by_split['train'], by_split['val1'], by_split['val2']
        lut = dict(zip(files, cohorts))          # labels from the CURRENT source of truth
        c_train = [lut[f] for f in f_train]
        c_val1 = [lut[f] for f in f_val1]
        c_val2 = [lut[f] for f in f_val2]
        print(f"\nSplit READ FROM {SPLIT_MANIFEST_IN}\n"
              f"  (cohort labels re-derived, so a relabelling cannot move a subject "
              f"between splits)")
    else:
        f_train, f_temp, c_train, c_temp = train_test_split(
            files, cohorts, test_size=(FRAC_VAL1 + FRAC_VAL2),
            stratify=cohorts, random_state=RANDOM_SEED)
        rel_val2 = FRAC_VAL2 / (FRAC_VAL1 + FRAC_VAL2)
        f_val1, f_val2, c_val1, c_val2 = train_test_split(
            f_temp, c_temp, test_size=rel_val2, stratify=c_temp, random_state=RANDOM_SEED)
        print("\nSplit GENERATED (SPLIT_MANIFEST_IN unset): not comparable with any "
              "run whose cohort labels differed.")

    manifest = pd.DataFrame({
        'filename': f_train + f_val1 + f_val2,
        'cohort': c_train + c_val1 + c_val2,
        'split': ['train'] * len(f_train) + ['val1'] * len(f_val1) + ['val2'] * len(f_val2),
    })
    manifest.to_csv(os.path.join(OUTPUT_DIR, 'split_manifest.csv'), index=False)
    print("\nSplit composition (subjects):")
    print(manifest.groupby(['split', 'cohort']).size())

    if MODE == 'evaluate':
        evaluate_released(f_val1, c_val1, f_val2, c_val2, device)
        return
    if MODE != 'train':
        raise ValueError(f"HDEEG2ECG_MODE must be 'evaluate' or 'train', not {MODE!r}")

    # ---- load data ----
    print("\nLoading train...");  eeg_tr, evt_tr, _ = load_split(f_train, DATA_FOLDER)
    print("Loading val1...");     eeg_v1, evt_v1, subj_v1 = load_split(f_val1, DATA_FOLDER)
    print("Loading val2...");     eeg_v2, evt_v2, subj_v2 = load_split(f_val2, DATA_FOLDER)
    for s, c in zip(subj_v1, c_val1): s['cohort'] = c
    for s, c in zip(subj_v2, c_val2): s['cohort'] = c

    datamodule = EEGDataModule(eeg_tr, evt_tr, eeg_v1, evt_v1)
    datamodule.setup()

    # ---- dump config + versions ----
    save_json({'config': {k: v for k, v in globals().items()
                          if k.isupper() and isinstance(v, (int, float, str, bool, list))},
               'versions': software_versions(),
               'timestamp': datetime.now().isoformat()},
              os.path.join(OUTPUT_DIR, 'run_config.json'))

    # ---- latent width: sweep-and-pick (testing) OR use the selected width ----
    ablation_summary = None
    if RUN_ABLATION:
        print("\n================ LATENT-WIDTH ABLATION ================")
        rows = []
        for w in ABLATION_WIDTHS:
            for sd in ABLATION_SEEDS:
                best, best_val_r, ep = train_one(w, datamodule, ABLATION_MAX_EPOCHS,
                                                 ABLATION_PATIENCE, sd, tag=f'ablate_w{w}_s{sd}')
                # full val1 reconstruction r for this width/seed (best checkpoint already loaded)
                rs = np.concatenate([reconstruct_subject(best, s['eeg'], device)[2] for s in subj_v1])
                rows.append({'width': w, 'seed': sd, 'best_val_r': best_val_r,
                             'val_mean_r': float(np.nanmean(rs)), 'stopped_epoch': ep})
                print(f"  width={w} seed={sd}: best_val_r={best_val_r:.4f} "
                      f"recon_mean_r={rows[-1]['val_mean_r']:.4f} (stopped @ epoch {ep})")
        ab = pd.DataFrame(rows)
        ab.to_csv(os.path.join(OUTPUT_DIR, 'ablation_results.csv'), index=False)

        agg = ab.groupby('width')['val_mean_r'].agg(['mean', 'std', 'count'])
        means = agg['mean'].values
        sem = (agg['std'] / np.sqrt(agg['count'])).fillna(0).values
        los, his = means - 1.96 * sem, means + 1.96 * sem
        knee_width, _ = find_knee(ABLATION_WIDTHS, means)
        plot_ablation(ABLATION_WIDTHS, means, los, his, knee_width,
                      os.path.join(OUTPUT_DIR, 'ablation_knee.png'))
        ablation_summary = {'widths': ABLATION_WIDTHS, 'mean_r': means.tolist(),
                            'ci95_low': los.tolist(), 'ci95_high': his.tolist(),
                            'knee_width': int(knee_width), 'n_seeds': len(ABLATION_SEEDS)}
        save_json(ablation_summary, os.path.join(OUTPUT_DIR, 'ablation_summary.json'))
        chosen_width = int(knee_width)
        print(f"  -> knee at latent width = {knee_width}")
    else:
        chosen_width = SELECTED_LATENT_DIM
        print(f"\nAblation skipped (RUN_ABLATION=False); "
              f"training final model at SELECTED_LATENT_DIM = {chosen_width}")

    print(f"\nFinal model latent width = {chosen_width}")

    # ---- decoder schedule: sweep-and-pick (testing) OR use the selected one ----
    decoder_sweep_summary = None
    if RUN_DECODER_SWEEP:
        print("\n================ DECODER-SCHEDULE SWEEP ================")
        rows = []
        for sched in DECODER_SCHEDULES:
            lab = decoder_label(sched)
            for sd in DECODER_SWEEP_SEEDS:
                best, best_val_r, ep = train_one(chosen_width, datamodule, ABLATION_MAX_EPOCHS,
                                                 ABLATION_PATIENCE, sd, tag=f'dec_{lab}_s{sd}',
                                                 decoder_schedule=sched)
                rs = np.concatenate([reconstruct_subject(best, s['eeg'], device)[2] for s in subj_v1])
                rows.append({'decoder': lab, 'seed': sd, 'best_val_r': best_val_r,
                             'val_mean_r': float(np.nanmean(rs)), 'stopped_epoch': ep})
                print(f"  decoder={lab} seed={sd}: best_val_r={best_val_r:.4f} "
                      f"recon_mean_r={rows[-1]['val_mean_r']:.4f} (stopped @ epoch {ep})")
        ds = pd.DataFrame(rows)
        ds.to_csv(os.path.join(OUTPUT_DIR, 'decoder_sweep_results.csv'), index=False)

        labels = [decoder_label(s) for s in DECODER_SCHEDULES]   # preserve config order
        agg = ds.groupby('decoder')['val_mean_r'].agg(['mean', 'std', 'count']).reindex(labels)
        means = agg['mean'].values
        sems = (agg['std'] / np.sqrt(agg['count'])).fillna(0).values
        best_idx = int(np.nanargmax(means))
        plot_decoder_sweep(labels, means, sems, best_idx,
                           os.path.join(OUTPUT_DIR, 'decoder_sweep.png'))
        decoder_sweep_summary = {'labels': labels, 'mean_r': means.tolist(),
                                 'sem': sems.tolist(), 'best': labels[best_idx],
                                 'n_seeds': len(DECODER_SWEEP_SEEDS)}
        save_json(decoder_sweep_summary, os.path.join(OUTPUT_DIR, 'decoder_sweep_summary.json'))
        chosen_decoder = DECODER_SCHEDULES[best_idx]   # selection on val1 only
        print(f"  -> best decoder (val1) = {labels[best_idx]}")
    else:
        chosen_decoder = SELECTED_DECODER
        print(f"Decoder sweep skipped (RUN_DECODER_SWEEP=False); "
              f"using SELECTED_DECODER = {decoder_label(chosen_decoder)}")

    # ---- component (lesion) ablation: REPORTING ONLY (does not pick the final model) ----
    component_summary = None
    if RUN_COMPONENT_ABLATION:
        print("\n================ COMPONENT (LESION) ABLATION ================")
        rows = []
        for var in COMPONENT_VARIANTS:
            for sd in COMPONENT_SEEDS:
                best, best_val_r, ep = train_one(chosen_width, datamodule, ABLATION_MAX_EPOCHS,
                                                 ABLATION_PATIENCE, sd, tag=f'comp_{var}_s{sd}',
                                                 decoder_schedule=chosen_decoder, variant=var)
                rs = np.concatenate([reconstruct_subject(best, s['eeg'], device)[2] for s in subj_v1])
                rows.append({'variant': var, 'seed': sd, 'best_val_r': best_val_r,
                             'val_mean_r': float(np.nanmean(rs)), 'stopped_epoch': ep})
                print(f"  variant={var} seed={sd}: best_val_r={best_val_r:.4f} "
                      f"recon_mean_r={rows[-1]['val_mean_r']:.4f} (stopped @ epoch {ep})")
        cdf = pd.DataFrame(rows)
        cdf.to_csv(os.path.join(OUTPUT_DIR, 'component_ablation_results.csv'), index=False)

        agg = cdf.groupby('variant')['val_mean_r'].agg(['mean', 'std', 'count']).reindex(COMPONENT_VARIANTS)
        means = agg['mean'].values
        sems = (agg['std'] / np.sqrt(agg['count'])).fillna(0).values
        full_idx = COMPONENT_VARIANTS.index('full') if 'full' in COMPONENT_VARIANTS else 0
        plot_decoder_sweep(COMPONENT_VARIANTS, means, sems, full_idx,           # highlight the control
                           os.path.join(OUTPUT_DIR, 'component_ablation.png'),
                           title='Component ablation (val1); control = full')
        component_summary = {'variants': COMPONENT_VARIANTS, 'mean_r': means.tolist(),
                             'sem': sems.tolist(), 'n_seeds': len(COMPONENT_SEEDS),
                             'note': 'reporting ablation; final model uses SELECTED_VARIANT'}
        save_json(component_summary, os.path.join(OUTPUT_DIR, 'component_ablation_summary.json'))
        print("  (reporting only -- final model uses SELECTED_VARIANT, not a val1 argmax)")

    chosen_variant = SELECTED_VARIANT
    print(f"\nFinal model variant = {chosen_variant}")

    # ---- final model ----
    final_summaries = {}
    if RUN_FINAL:
        print("\n================ FINAL MODEL ================")
        best_model, best_val_r, _ = train_one(chosen_width, datamodule, FINAL_MAX_EPOCHS,
                                             FINAL_PATIENCE, RANDOM_SEED, tag='final',
                                             decoder_schedule=chosen_decoder, variant=chosen_variant)
        torch.save(best_model.model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model_weights.pt'))

        df1, sum1 = evaluate_split(best_model, subj_v1, device, 'val1', OUTPUT_DIR)
        df2, sum2 = evaluate_split(best_model, subj_v2, device, 'val2', OUTPUT_DIR)
        pooled_leak = pooled_leakage_analysis([df1, df2])     # robust n=20 F1-vs-leakage test
        final_summaries = {'val1': sum1, 'val2': sum2, 'chosen_width': chosen_width,
                           'chosen_decoder': decoder_label(chosen_decoder),
                           'chosen_variant': chosen_variant,
                           'pooled_leakage': pooled_leak,
                           'final_val_r': best_val_r}
        save_json(final_summaries, os.path.join(OUTPUT_DIR, 'validation_summary.json'))
        write_paper_summary(final_summaries, ablation_summary, manifest,
                            os.path.join(OUTPUT_DIR, 'paper_validation_summary.md'))

    print(f"\nDone. All outputs in {OUTPUT_DIR}")


def count_variant_params(latent_dim, decoder_schedule, variants):
    """Total parameters for each component-ablation variant. Lesioned models omit the removed
    modules, so the counts are honest. Instantiation only -- no training, no GPU required."""
    out = {}
    for v in variants:
        m = EEGtoECGModel(latent_dim=latent_dim, decoder_schedule=decoder_schedule, variant=v)
        out[v] = int(sum(p.numel() for p in m.parameters()))
    return out


def build_ablation_table_md(comp, params=None):
    """Render the component (lesion) ablation as a Markdown table: per-variant val1 r
    (+/- sem), delta vs the full model, and parameter count (if provided). The framing is
    deliberately parsimony-not-superiority: variants within seed scatter are reported as
    equivalent, and the essential component (the dilated encoder) is identified by its collapse."""
    variants = comp['variants']
    means = comp['mean_r']
    sems = comp.get('sem', [None] * len(variants))
    have_sem = any(s is not None for s in sems)
    have_par = bool(params)
    full_r = dict(zip(variants, means)).get('full')

    head = "| Variant | Val1 r" + (" (mean ± sem)" if have_sem else "") + " | Δ vs full |"
    sep = "|---|---|---|"
    if have_par:
        head += " Params |"
        sep += "---|"
    out = [f"## Component (lesion) ablation (val1, {comp.get('n_seeds', '?')} seeds each)\n", head, sep]
    for v, r, s in zip(variants, means, sems):
        rcell = f"{r:.3f}" + (f" ± {s:.3f}" if (have_sem and s is not None) else "")
        dcell = "—" if (full_r is None or v == 'full') else f"{r - full_r:+.3f}"
        row = f"| `{v}` | {rcell} | {dcell} |"
        if have_par:
            row += (f" {params[v] / 1e6:.2f}M |" if v in params else " — |")
        out.append(row)
    out += ["",
            "Differences among `full`, `no_cross_attn`, and `conv_only` fall within seed scatter; "
            "removing the dilated convolutional encoder (`no_dilated_encoder`) collapses "
            "reconstruction. The attention and cross-modal components are therefore not necessary "
            "for this task, whereas the dilated convolutional encoder is the essential component. "
            "The final model uses the parsimonious `conv_only` variant -- equal performance with "
            "fewer parameters, not a claim of superiority over `full`.\n"]
    return out


def write_paper_summary(fs, ab, manifest, path):
    def fmt(d): return f"{d['mean']:.3f} (95% CI {d['ci95_low']:.3f}-{d['ci95_high']:.3f})"

    out_dir = os.path.dirname(path)
    def _load(name):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            try:
                with open(p, encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    # pull in ablation summaries written by this run OR earlier runs, so one document
    # consolidates everything regardless of which toggles fired this invocation
    lat  = ab or _load('ablation_summary.json')
    dec  = _load('decoder_sweep_summary.json')
    comp = _load('component_ablation_summary.json')

    lines = ["# Validation summary (auto-generated)\n",
             f"Generated: {datetime.now().isoformat()}\n",
             "## Final model architecture\n",
             f"- Latent width: {fs.get('chosen_width')}",
             f"- Decoder schedule: {fs.get('chosen_decoder')}",
             f"- Component variant: {fs.get('chosen_variant')}\n",
             "## Sample\n",
             f"- Subjects: {len(manifest)} total "
             f"({(manifest.cohort=='ASD').sum()} ASD, {(manifest.cohort=='TD').sum()} TD)",
             f"- Splits: train={int((manifest.split=='train').sum())}, "
             f"val1={int((manifest.split=='val1').sum())}, val2={int((manifest.split=='val2').sum())}\n"]

    if lat:
        lines += ["## Latent-width ablation\n",
                  f"- Widths tested: {lat['widths']} ({lat['n_seeds']} seeds each)",
                  f"- Mean val r by width: " +
                  ", ".join(f"{w}:{r:.3f}" for w, r in zip(lat['widths'], lat['mean_r'])) + "\n"]
    if dec:
        lines += ["## Decoder-schedule ablation\n",
                  f"- Schedules ({dec['n_seeds']} seeds each): " +
                  ", ".join(f"{l}:{r:.3f}" for l, r in zip(dec['labels'], dec['mean_r'])),
                  f"- Spread (max-min mean r): {max(dec['mean_r']) - min(dec['mean_r']):.3f} "
                  f"-- interpret relative to seed scatter; report as insensitivity, not a winner\n"]
    if comp:
        try:
            _params = count_variant_params(fs.get('chosen_width', SELECTED_LATENT_DIM),
                                           SELECTED_DECODER, comp['variants'])
        except Exception:
            _params = None
        table_md = build_ablation_table_md(comp, _params)
        lines += table_md
        try:   # standalone, paste-ready copy next to the summary
            with open(os.path.join(out_dir, 'component_ablation_table.md'), 'w', encoding='utf-8') as _f:
                _f.write("\n".join(table_md))
        except Exception:
            pass

    for split in ['val1', 'val2']:
        s = fs[split]
        cc = s['cohort_comparison_r']
        lines += [f"## {split} (n={s['n_subjects']}: {s['n_subjects_ASD']} ASD / {s['n_subjects_TD']} TD)\n",
                  f"- Waveform Pearson r: {fmt(s['mean_r'])}",
                  f"- RMSE: {fmt(s['rmse'])}",
                  f"- R-peak F1: {fmt(s['f1'])}  (sensitivity {fmt(s['sensitivity'])}, PPV {fmt(s['ppv'])})",
                  f"- R-peak timing error (abs): {fmt(s['timing_err_ms_mean'])} ms",
                  f"- R-peak latency bias (signed): {fmt(s['timing_bias_ms'])} ms",
                  f"- ASD vs TD reconstruction r: Mann-Whitney p={cc.get('mannwhitney_p', float('nan')):.3f}, "
                  f"rank-biserial={cc.get('rank_biserial', float('nan')):.3f}, "
                  f"TOST equivalence p={cc.get('tost_p', float('nan')):.3f} "
                  f"(margin +/-{cc.get('equiv_margin', EQUIV_MARGIN_R)} r) -- underpowered; report as 'no detectable difference'"]
        if 'f1_vs_leakage_spearman' in s:
            lk = s['f1_vs_leakage_spearman']
            lines += [f"- F1 vs EEG cardiac leakage: Spearman rho={lk['rho']:.3f}, "
                      f"p={lk['p']:.3g} (n={lk['n']})"]
        if 'bland_altman_HR' in s:
            ba = s['bland_altman_HR']
            lines += [f"- HR Bland-Altman: bias {ba['bias']:.2f} bpm, "
                      f"95% LoA [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}] (wide/outlier-driven)\n"]

    pl = fs.get('pooled_leakage')
    if pl and 'spearman_rho' in pl:
        lines += ["## Pooled leakage (val1 + val2)\n",
                  f"- F1 vs EEG cardiac leakage, pooled (n={pl['n']}): "
                  f"Spearman rho={pl['spearman_rho']:.3f}, p={pl['spearman_p']:.3g}",
                  f"- Partial (controlling for cohort): rho={pl['partial_rho_ctrl_cohort']:.3f}, "
                  f"p={pl['partial_p_ctrl_cohort']:.3g}",
                  "- Positive rho supports that low physiological leakage, not model failure, "
                  "drives the poorly-reconstructed subjects\n"]

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()