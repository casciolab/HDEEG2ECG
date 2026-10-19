# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:45:16 2026

@author: Matias E. Fraile-Vazquez
"""

# -*- coding: utf-8 -*-
"""
Robustness check on the EEG cardiac-leakage measure.

Compares the single-channel maximum (eeg_cardiac_leakage, the measure currently used)
against a spatially-optimal multichannel measure: the peak-to-peak of the dominant
spatiotemporal component of the R-peak-triggered average (its leading singular vector) --
i.e. what an optimal linear spatial filter over the 128 channels (the kind the model's input
projection can learn) would recover. It matches the single-channel max when leakage is focal
and exceeds it when leakage is distributed across electrodes.

A per-subject permutation noise floor is subtracted: the leading singular value of a
[time x 128] matrix is inflated by chance structure across many channels, so a no-leakage
subject would otherwise score ~0.8 instead of ~0. We trigger on random (non-R-peak) times,
recompute the same statistic, and subtract its mean.

No retraining. It re-loads the val1/val2 recordings listed in split_manifest.csv, recomputes
both measures, pulls per-subject F1 from the per_subject_metrics_*.csv files the run wrote,
and reports (a) the agreement between the two measures and (b) each measure's correlation
with reconstruction F1.

Run from the scripts directory:  python leakage_compare.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sps

# load_subject_data, eeg_cardiac_leakage, DATA_FOLDER, OUTPUT_DIR, SFREQ all come from the pipeline
import hdeeg2ecg_model as P

HALF_WIN_MS = 120
N_PERM = 1000


def _dominant_p2p(avg):
    """Peak-to-peak of the leading SVD component of the (per-channel time-centered)
    triggered average -- the output an optimal linear spatial filter would produce."""
    avg = avg - avg.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(avg, full_matrices=False)
    comp = U[:, 0] * S[0]
    return float(comp.max() - comp.min())

def leakage_spatial(eeg, event_vector, half_win_ms=HALF_WIN_MS, n_perm=N_PERM, seed=0):
    """Noise-corrected spatially-optimal leakage: dominant-component p2p of the R-peak-triggered
    average, minus the mean of the same statistic computed at random trigger times."""
    hw = int(half_win_ms / 1000 * P.SFREQ)
    peaks = np.where(event_vector == 1)[0]
    peaks = peaks[(peaks - hw >= 0) & (peaks + hw < len(eeg))]      # p+hw is a valid index -> p+hw+1 slice ok
    if len(peaks) < 5:
        return np.nan
    e = eeg[:, :128]
    obs = _dominant_p2p(np.stack([e[p - hw:p + hw + 1] for p in peaks]).mean(axis=0))   # +1
    rng = np.random.default_rng(seed)
    lo, hi, n = hw, len(eeg) - hw, len(peaks)                        # hi exclusive -> max p = len-hw-1 -> p+hw = len-1, in bounds
    null = [_dominant_p2p(np.stack([e[p - hw:p + hw + 1] for p in rng.integers(lo, hi, size=n)]).mean(axis=0))  # +1
            for _ in range(n_perm)]
    return max(0.0, obs - float(np.mean(null)))


def main():
    man = pd.read_csv(os.path.join(P.OUTPUT_DIR, 'split_manifest.csv'))
    man = man[man.split.isin(['val1', 'val2'])].reset_index(drop=True)

    # per-subject F1 from the CSVs the final run wrote
    f1 = {}
    for sp in ['val1', 'val2']:
        d = pd.read_csv(os.path.join(P.OUTPUT_DIR, f'per_subject_metrics_{sp}.csv'))
        f1.update(dict(zip(d['file'].astype(str), d['f1'])))

    rows = []
    for _, r in man.iterrows():
        fn, split = str(r['filename']), r['split']
        eeg, evt, _ = P.load_subject_data(os.path.join(P.DATA_FOLDER, fn))
        lm = P.eeg_cardiac_leakage(eeg, evt)
        ls = leakage_spatial(eeg, evt)
        rows.append({'file': fn, 'split': split, 'leak_max': lm, 'leak_spatial': ls,
                     'f1': f1.get(fn, np.nan)})
        print(f"  {fn:42s} [{split}]  max={lm:6.3f}  spatial={ls:6.3f}  F1={f1.get(fn, float('nan')):.3f}")
    df = pd.DataFrame(rows).sort_values(['split', 'leak_max']).reset_index(drop=True)
    df.to_csv(os.path.join(P.OUTPUT_DIR, 'leakage_compare.csv'), index=False)

    def spear(a, b, d):
        d = d.dropna(subset=[a, b])
        if len(d) < 3:
            return float('nan'), float('nan'), len(d)
        rho, p = sps.spearmanr(d[a], d[b])
        return float(rho), float(p), len(d)

    print("\n=== agreement between the two leakage measures (Spearman) ===")
    for name, d in [('val1', df[df.split == 'val1']), ('val2', df[df.split == 'val2']), ('pooled', df)]:
        rho, p, n = spear('leak_max', 'leak_spatial', d)
        print(f"  {name:7s} rho(max, spatial) = {rho:+.3f}  (p={p:.3g}, n={n})")

    print("\n=== each measure vs reconstruction F1 (Spearman) ===")
    for name, d in [('val1', df[df.split == 'val1']), ('val2', df[df.split == 'val2']), ('pooled', df)]:
        rm, pm, n = spear('leak_max', 'f1', d)
        rs, ps, _ = spear('leak_spatial', 'f1', d)
        print(f"  {name:7s} (n={n})  F1~max: rho={rm:+.3f} (p={pm:.3g})   "
              f"F1~spatial: rho={rs:+.3f} (p={ps:.3g})")

    print(f"\nwrote leakage_compare.csv to {P.OUTPUT_DIR}")


if __name__ == '__main__':
    main()