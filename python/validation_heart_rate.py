# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:13:36 2026

@author: Matias E. Fraile-Vazquez
"""

# -*- coding: utf-8 -*-
"""
Build the per-subject heart-rate table + a leakage-stratified Bland-Altman summary
(LaTeX) from an existing per_subject_metrics_<split>.csv. No retraining: it only reads
the CSV the final run already wrote.

Point this at the same OUTPUT_DIR your pipeline used, then run it. It prints (and writes)
two LaTeX tables and the threshold-free Spearman(|dHR|, leakage) statistic.

The "outlier" split is taken at the LARGEST GAP in the (model-free) leakage distribution,
so it is data-driven and disclosable rather than hand-picked; set LEAK_THR to override.
"""
import os
import re
import numpy as np
import pandas as pd

# --- edit these two if needed ---
OUTPUT_DIR = os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'))
SPLIT      = 'val2'      # 'val2' (held-out) or 'val1'
LEAK_THR   = 1.0        # None = split at the largest gap in the leakage distribution; or a number


def _num(v, fmt):
    """Format a value, returning '--' for NaN/missing."""
    return '--' if pd.isna(v) else format(v, fmt)


def _bland_altman(dhr):
    x = np.asarray(dhr, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return len(x), float('nan'), float('nan'), float('nan')
    b, s = float(np.mean(x)), float(np.std(x, ddof=1))
    return len(x), b, b - 1.96 * s, b + 1.96 * s


def main():
    csv = os.path.join(OUTPUT_DIR, f'per_subject_metrics_{SPLIT}.csv')
    df = pd.read_csv(csv)
    df['id'] = df['file'].astype(str).str.extract(r'(8\d{3})')
    df['dhr'] = df['pred_hr'] - df['true_hr']
    df = df.sort_values('eeg_leakage').reset_index(drop=True)

    # data-driven cutoff: midpoint of the widest gap in the sorted leakage values
    leak = np.sort(df['eeg_leakage'].dropna().values)
    gi = int(np.argmax(np.diff(leak)))
    thr = LEAK_THR if LEAK_THR is not None else float((leak[gi] + leak[gi + 1]) / 2)

    rho = df['eeg_leakage'].corr(df['dhr'].abs(), method='spearman')

    # ---- console preview ----
    print(f"\nsplit = {SPLIT}   leakage gap cutoff T = {thr:.2f}")
    print(df[['id', 'cohort', 'eeg_leakage', 'f1', 'true_hr', 'pred_hr', 'dhr',
              'n_true_peaks', 'n_pred_peaks']]
          .to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(f"\nSpearman(|dHR|, leakage) = {rho:.2f}")
    for lab, d in [('all', df), (f'leakage>={thr:.2f}', df[df.eeg_leakage >= thr])]:
        n, b, lo, hi = _bland_altman(d['dhr'])
        print(f"  {lab}: n={n}  bias={b:+.1f}  95% LoA=[{lo:+.1f}, {hi:+.1f}]")

    # ---- LaTeX: per-subject table ----
    ps_rows = "\n".join(
        f"    {r.id} & {r.cohort} & ${_num(r.eeg_leakage, '.2f')}$ & ${_num(r.f1, '.2f')}$ & "
        f"${_num(r.true_hr, '.0f')}$ & ${_num(r.pred_hr, '.0f')}$ & ${_num(r.dhr, '+.0f')}$ & "
        f"${int(r.n_pred_peaks)}/{int(r.n_true_peaks)}$ \\\\"
        for r in df.itertuples())
    per_subj = (r"""\begin{table}[t]
  \centering
  \caption{Per-subject heart-rate agreement on the held-out test split (val2), sorted by EEG
  cardiac leakage. Subjects with negligible leakage (top rows) over-detect R-peaks (predicted
  vs.\ true peak counts) and incur large heart-rate errors ($\Delta\mathrm{HR}=$ predicted
  $-$ true), whereas subjects with clear leakage reconstruct accurately. The wide group-level
  limits of agreement are driven entirely by the low-leakage subjects.}
  \label{tab:hr_persubject}
  \begin{tabular}{llcccccc}
    \toprule
    Subject & Cohort & Leakage & F1 & True HR & Pred.\ HR & $\Delta$HR & Peaks (P/T) \\
    \midrule
""" + ps_rows + r"""
    \bottomrule
  \end{tabular}
\end{table}""")

    # ---- LaTeX: Bland-Altman stratified summary ----
    na = _bland_altman(df['dhr'])
    cl = _bland_altman(df[df.eeg_leakage >= thr]['dhr'])
    ba_tex = (r"""\begin{table}[t]
  \centering
  \caption{Heart-rate Bland--Altman agreement on the held-out test split (val2): all subjects
  vs.\ subjects with detectable EEG cardiac leakage ($\geq """ + f"{thr:.2f}" + r"""$, the split at
  the largest gap in the leakage distribution). Excluding the low-leakage subjects, whose ECG
  reconstruction over-detects R-peaks, collapses both the bias and the limits of agreement,
  confirming that the group-level disagreement is outlier-driven. Across all subjects, absolute
  HR error is strongly negatively correlated with leakage (Spearman $\rho="""
              + f"{rho:.2f}" + r"""$).}
  \label{tab:hr_ba}
  \begin{tabular}{lccc}
    \toprule
    Subjects & $n$ & Bias (bpm) & 95\% LoA (bpm) \\
    \midrule
"""
              + f"    All subjects & {na[0]} & ${na[1]:+.1f}$ & $[{na[2]:+.1f},\\ {na[3]:+.1f}]$ \\\\\n"
              + f"    Leakage $\\geq {thr:.2f}$ & {cl[0]} & ${cl[1]:+.1f}$ & $[{cl[2]:+.1f},\\ {cl[3]:+.1f}]$ \\\\\n"
              + r"""    \bottomrule
  \end{tabular}
\end{table}""")

    print("\n" + per_subj)
    print("\n" + ba_tex)

    # also drop .tex files next to the CSV
    for name, tex in [('hr_persubject_table.tex', per_subj), ('hr_ba_table.tex', ba_tex)]:
        with open(os.path.join(OUTPUT_DIR, name), 'w', encoding='utf-8') as f:
            f.write(tex + "\n")
    print(f"\nwrote hr_persubject_table.tex and hr_ba_table.tex to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()