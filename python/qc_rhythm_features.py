# -*- coding: utf-8 -*-
"""
Reference-free reconstruction QC from predicted-peak LATENCIES only (no EEG morphology, no ECG).

Interval-based plausibility features computed from the reconstruction's own peaks, validated
against F1. The model's failure mode is OVER-DETECTION (too many peaks on a poor reconstruction),
which shows up purely in the inter-peak intervals.

This version fixes the earlier omission where the per-split separation check only covered a
hand-picked subset of features (frac_implaus, pnn_large, n_pred were skipped). Both the pooled
and the within-split reports are now driven from the SAME `direction` dict, so every feature is
evaluated in both, and each split cell reports the failed-vs-good subject counts so an
"undefined" gap (a split with 0 failures or 0 successes) is distinguishable from a real verdict.

Outputs (all to OUTPUT_DIR):
    rhythm_qc.csv            -- per-subject features + F1
    rhythm_qc_separation.csv -- per-feature: rho/p pooled, pooled gap, and per-split gaps+counts
    rhythm_qc_scatter.png    -- HR-robust composite vs F1

Inference only (loads the released weights, P.WEIGHTS_PATH; no training).
Run next to the pipeline:  python qc_rhythm_features.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import hdeeg2ecg_model as P

# ---- fixed a-priori constants (must match the toolbox QC gate) ----
SHORT_MS = 400              # RR below this is sub-physiological (150 bpm); over-detection floor
RR_LO, RR_HI = 500, 1300    # plausible RR window (HR ~46-120 bpm)
JUMP_MS = 200               # |consecutive RR difference| above this = erratic
FAIL_F1 = 0.60              # F1 below this = "failure" (used ONLY for the separation check)

# split labels as written in split_manifest.csv. The pipeline writes 'val1'/'val2';
# override here if your manifest uses different names (e.g. 'Selection'/'Testing').
SPLITS = ['val1', 'val2']


def interval_features(pk, sfreq):
    """Reference-free interval features from a peak-sample array. None if < 5 RR intervals."""
    rr = np.diff(pk) / sfreq * 1000.0
    if len(rr) < 5:
        return None
    drr = np.abs(np.diff(rr))
    return {
        'apparent_hr':  float(60000.0 / np.median(rr)),
        'rr_reg':       float(1.0 / (1.0 + rr.std() / rr.mean())),
        'frac_short':   float(np.mean(rr < SHORT_MS)),
        'frac_implaus': float(np.mean((rr < RR_LO) | (rr > RR_HI))),
        'pnn_large':    float(np.mean(drr > JUMP_MS)) if len(drr) else np.nan,
        'n_pred':       int(len(pk)),
    }


# orientation so that "higher oriented value = better reconstruction" for every feature;
# the composite is added after the frame is built.
DIRECTION = {
    'apparent_hr':   -1,   # over-detection -> implausibly high HR
    'rr_reg':        +1,   # erratic over-detection -> low regularity
    'frac_short':    -1,
    'frac_implaus':  -1,
    'pnn_large':     -1,
    'n_pred':        -1,
    'implaus_score': -1,   # composite: higher = worse
}


def oriented_gap(sub, col, dirn, fail_mask):
    """Signed separation between failed and successful groups in the feature's oriented units.
    Returns (gap, n_fail, n_ok); gap is NaN when a group is empty. Positive gap = clean,
    non-overlapping separation (worst success still beats best failure)."""
    v = sub[col].values * dirn
    fl = fail_mask.values
    n_fail, n_ok = int(fl.sum()), int((~fl).sum())
    if n_fail == 0 or n_ok == 0:
        return np.nan, n_fail, n_ok
    gap = float(np.min(v[~fl]) - np.max(v[fl]))
    return gap, n_fail, n_ok


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- load the deployed model (conv_only) ----
    model = P.EEGtoECGModel(latent_dim=P.SELECTED_LATENT_DIM,
                            decoder_schedule=P.SELECTED_DECODER, variant=P.SELECTED_VARIANT)
    model.load_state_dict(torch.load(P.WEIGHTS_PATH,
                                     map_location=device))
    model.eval().to(device)

    # ---- subjects ----
    man = pd.read_csv(os.path.join(P.OUTPUT_DIR, 'split_manifest.csv'))
    man = man[man.split.isin(SPLITS)].reset_index(drop=True)
    if len(man) == 0:
        raise SystemExit(f"No subjects with split in {SPLITS}. "
                         f"Check split_manifest.csv split labels (found: "
                         f"{sorted(pd.read_csv(os.path.join(P.OUTPUT_DIR,'split_manifest.csv')).split.unique())}).")
    tol = int(P.QRS_MATCH_TOLERANCE_MS / 1000 * P.SFREQ)

    rows = []
    for _, row in man.iterrows():
        fn, split, cohort = str(row['filename']), row['split'], str(row['cohort']).upper()
        sid = fn.replace('_a0_withR.set', '').replace('sub_', '')
        eeg, evt, _ = P.load_subject_data(os.path.join(P.DATA_FOLDER, fn))
        n = (len(eeg) // P.WINDOW_SIZE) * P.WINDOW_SIZE
        pred_sig, _, _ = P.reconstruct_subject(model, eeg, device)

        true_pk = np.where(evt[:n] == 1)[0]
        pred_pk = P.detect_peaks(pred_sig)
        tp, fp, fn_, _ = P.match_peaks(true_pk, pred_pk, tol)
        sens = tp / (tp + fn_) if (tp + fn_) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        f1 = (2 * sens * ppv / (sens + ppv)
              if (sens + ppv) and not np.isnan(sens) and not np.isnan(ppv) else np.nan)

        feats = interval_features(pred_pk, P.SFREQ)
        if feats is None:
            print(f"  {sid:>8s} [{split}]  < 5 RR intervals -- skipped")
            continue
        feats.update({'id': sid, 'split': split, 'cohort': cohort, 'f1': f1})
        rows.append(feats)
        print(f"  {sid:>8s} [{split:4s}]  F1={f1:5.3f}  frac_short={feats['frac_short']:.3f}  "
              f"frac_implaus={feats['frac_implaus']:.3f}  n_pred={feats['n_pred']}")

    df = pd.DataFrame(rows)

    # ---- HR-robust composite: z-score the over-detection features (NOT raw HR), higher = worse ----
    zc = ['frac_short', 'frac_implaus', 'pnn_large']
    z = df[zc].apply(lambda c: (c - c.mean()) / (c.std() + 1e-9))
    df['implaus_score'] = z.sum(axis=1) + (df['rr_reg'].mean() - df['rr_reg']) / (df['rr_reg'].std() + 1e-9)
    df.to_csv(os.path.join(P.OUTPUT_DIR, 'rhythm_qc.csv'), index=False)

    fail = df.f1 < FAIL_F1

    # ============================================================
    # POOLED + PER-SPLIT SEPARATION FOR EVERY FEATURE (one loop, one dict)
    # ============================================================
    sep_rows = []
    header = (f"{'feature':14s} {'rho_pool':>9s} {'p':>10s} | {'pooled sep':>18s} | "
              + " | ".join(f"{sp:>14s}" for sp in SPLITS))
    print("\n" + header)
    print("-" * len(header))

    for c, d in DIRECTION.items():
        rho, p = sps.spearmanr(df.f1, df[c])
        g_pool, nf_pool, nok_pool = oriented_gap(df, c, d, fail)
        pool_tag = ("CLEAN " if g_pool > 0 else "overlap") + f" {g_pool:+.3g} ({nf_pool}v{nok_pool})" \
            if np.isfinite(g_pool) else f"n/a ({nf_pool}v{nok_pool})"

        rec = {'feature': c, 'direction': d, 'rho_pooled': rho, 'p_pooled': p,
               'gap_pooled': g_pool, 'n_fail_pooled': nf_pool, 'n_ok_pooled': nok_pool}

        split_cells = []
        for sp in SPLITS:
            s = df[df.split == sp]
            g, nf, nok = oriented_gap(s, c, d, s.f1 < FAIL_F1)
            if np.isfinite(g):
                cell = f"{'CLEAN' if g > 0 else 'overlap'} {g:+.2g} ({nf}v{nok})"
            else:
                cell = f"n/a ({nf}v{nok})"
            split_cells.append(f"{cell:>14s}")
            rec[f'gap_{sp}'] = g
            rec[f'n_fail_{sp}'] = nf
            rec[f'n_ok_{sp}'] = nok
        sep_rows.append(rec)

        print(f"{c:14s} {rho:+9.3f} {p:10.3g} | {pool_tag:>18s} | " + " | ".join(split_cells))

    sep = pd.DataFrame(sep_rows)
    sep.to_csv(os.path.join(P.OUTPUT_DIR, 'rhythm_qc_separation.csv'), index=False)

    # ---- reading guide ----
    print("\nlegend: 'CLEAN +g' = failed and successful groups do not overlap (margin g in the")
    print("        feature's oriented units); 'overlap -g' = they interleave; 'n/a' = that split")
    print("        had 0 failures or 0 successes, so a gap is undefined (see the NvN counts).")
    print("        A feature is a robust threshold only if it is CLEAN pooled AND CLEAN in each split.")

    # ============================================================
    # scatter of the HR-robust composite vs F1
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['crimson' if f else 'steelblue' for f in fail]
    ax.scatter(df['implaus_score'], df['f1'], c=colors, s=60, edgecolor='k', linewidth=0.5)
    for _, r in df.iterrows():
        ax.annotate(r['id'], (r['implaus_score'], r['f1']), fontsize=6, xytext=(3, 3),
                    textcoords='offset points')
    ax.axhline(FAIL_F1, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('rhythm implausibility score (HR-robust; higher = worse)')
    ax.set_ylabel('F1 (validation only)')
    ax.set_title('Reference-free rhythm QC vs reconstruction quality\n'
                 f'red = failure (F1<{FAIL_F1}), blue = good')
    fig.tight_layout()
    fig.savefig(os.path.join(P.OUTPUT_DIR, 'rhythm_qc_scatter.png'), dpi=150)
    plt.close(fig)

    print(f"\nwrote rhythm_qc.csv, rhythm_qc_separation.csv, rhythm_qc_scatter.png to {P.OUTPUT_DIR}")


if __name__ == '__main__':
    main()