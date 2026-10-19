# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:31:40 2026

@author: Matias E. Fraile-Vazquez
"""

# -*- coding: utf-8 -*-
"""
Pool the per-subject metrics from val1 and val2 and rerun the *relational* analyses
(cohort comparison + leakage correlation) with more power.

Why this is allowed: model selection (early stopping, the latent-width knee) used the
aggregate val1 reconstruction loss only. That is independent of cohort identity and of
the model-free leakage measure, so pooling for these relational claims adds power without
circularity. It is NOT valid for headline performance (r, F1): val1 informed selection,
so pooled performance is reported here as descriptive only -- cite val2 for held-out
performance.

Reads per_subject_metrics_val1.csv and per_subject_metrics_val2.csv from OUTPUT_DIR.
Writes pooled_per_subject_metrics.csv, pooled_analyses.json, pooled_analyses.md, and
three pooled figures.

@author: Matias E. Fraile-Vazquez
"""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy import stats as sps

# ---- config (match the main pipeline) ----
OUTPUT_DIR     = os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'))
EQUIV_MARGIN_R = 0.05
N_BOOTSTRAP    = 5000
RANDOM_SEED    = 42

# The cohort column inherited from per_subject_metrics_val*.csv was written by the
# deployed run, which used the old hardcoded TD_IDS list and mislabels subjects.
# Cohort is therefore re-derived here from folder membership.
# The same file the model reads (inputs/adult/recordings/cohort_map.csv, columns
# filename,cohort); a subject_id,cohort table also works. Without one, the labels
# the model wrote are kept.
_INPUT_DIR = os.environ.get('HDEEG2ECG_INPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs'))
COHORT_MAP = os.environ.get('HDEEG2ECG_COHORT_MAP', os.path.join(
    os.environ.get('HDEEG2ECG_DATA_DIR', os.path.join(_INPUT_DIR, 'adult', 'recordings')), 'cohort_map.csv'))


def relabel_cohorts(df, label):
    """Overwrite the inherited cohort column from COHORT_MAP, and say what moved.

    Raises rather than guessing if a subject is absent from the map: a silently
    unlabelled subject would be dropped from every cohort comparison downstream.
    """
    if not os.path.exists(COHORT_MAP):
        print(f'  {label}: no {COHORT_MAP}; keeping the cohort labels the model wrote')
        return df
    m = pd.read_csv(COHORT_MAP)
    if 'filename' in m.columns:
        new = df.file.map(dict(zip(m['filename'].astype(str), m['cohort'].astype(str).str.upper())))
    else:
        lut = m.set_index('subject_id').cohort.to_dict()
        sid = df.file.map(lambda f: int(re.search(r'sub_(\d+)_', f).group(1)))
        new = sid.map(lut)
    if new.isna().any():
        raise KeyError(f'{label}: absent from {os.path.basename(COHORT_MAP)}: '
                       f'{df.file[new.isna()].tolist()}')
    moved = new.values != df.cohort.values
    for f, o, n in zip(df.file[moved], df.cohort[moved], new[moved]):
        print(f'    relabelled {f}: {o} -> {n}')
    print(f'  {label}: {moved.sum()} of {len(df)} cohort labels corrected; now '
          f'{(new == "ASD").sum()} ASD / {(new == "TD").sum()} TD')
    return df.assign(cohort=new.values)


# ============================================================
# STATS (same definitions as the main script)
# ============================================================
def cluster_bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05, seed=RANDOM_SEED):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def cohort_test(df, metric):
    asd = df.loc[df.cohort == 'ASD', metric].dropna().values
    td  = df.loc[df.cohort == 'TD', metric].dropna().values
    out = {'metric': metric, 'n_ASD': int(len(asd)), 'n_TD': int(len(td)),
           'median_ASD': float(np.median(asd)) if len(asd) else np.nan,
           'median_TD': float(np.median(td)) if len(td) else np.nan}
    if len(asd) >= 2 and len(td) >= 2:
        u, p = sps.mannwhitneyu(asd, td, alternative='two-sided')
        out['mannwhitney_U'] = float(u)
        out['mannwhitney_p'] = float(p)
        out['rank_biserial'] = float(1 - 2 * u / (len(asd) * len(td)))
        diff = asd.mean() - td.mean()
        se = np.sqrt(asd.var(ddof=1) / len(asd) + td.var(ddof=1) / len(td))
        dfree = len(asd) + len(td) - 2
        out['mean_diff'] = float(diff)
        out['equiv_margin'] = EQUIV_MARGIN_R
        if se > 0:
            p_low = sps.t.sf((diff + EQUIV_MARGIN_R) / se, dfree)
            p_high = sps.t.cdf((diff - EQUIV_MARGIN_R) / se, dfree)
            out['tost_p'] = float(max(p_low, p_high))   # equivalence supported if < 0.05
    return out


def leakage_corr(df):
    d = df.dropna(subset=['f1', 'eeg_leakage'])
    if len(d) < 3:
        return {}
    rho, p = sps.spearmanr(d.eeg_leakage, d.f1)
    return {'rho': float(rho), 'p': float(p), 'n': int(len(d))}


def partial_leakage_corr(df, control='n_true_peaks'):
    """Spearman partial correlation of F1 vs leakage, controlling for `control`
    (e.g. beat count) -- closes the 'both ride on a third factor' objection."""
    d = df.dropna(subset=['f1', 'eeg_leakage', control])
    if len(d) < 5:
        return {}
    rf, rl, rc = d.f1.rank(), d.eeg_leakage.rank(), d[control].rank()
    rxy = np.corrcoef(rf, rl)[0, 1]
    rxz = np.corrcoef(rf, rc)[0, 1]
    ryz = np.corrcoef(rl, rc)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0:
        return {}
    pr = (rxy - rxz * ryz) / denom
    n = len(d)
    dfree = n - 3
    if dfree <= 0 or abs(pr) >= 1:
        return {'partial_rho': float(pr), 'n': int(n), 'control': control}
    t = pr * np.sqrt(dfree / (1 - pr ** 2))
    p = 2 * sps.t.sf(abs(t), dfree)
    return {'partial_rho': float(pr), 'p': float(p), 'n': int(n), 'control': control}


def perf_descriptives(df):
    out = {}
    for m in ['mean_r', 'f1', 'sensitivity', 'ppv', 'timing_err_ms_mean']:
        if m in df:
            mean, lo, hi = cluster_bootstrap_ci(df[m].values)
            out[m] = {'mean': mean, 'ci95_low': lo, 'ci95_high': hi}
    return out


# ============================================================
# FIGURES (pooled)
# ============================================================
# ============================================================
# REPORT
# ============================================================
def analyse(df, label):
    return {
        'label': label,
        'n_subjects': int(len(df)),
        'n_ASD': int((df.cohort == 'ASD').sum()),
        'n_TD': int((df.cohort == 'TD').sum()),
        'cohort_mean_r': cohort_test(df, 'mean_r'),
        'cohort_f1': cohort_test(df, 'f1'),
        'leakage': leakage_corr(df),
        'leakage_partial': partial_leakage_corr(df, 'n_true_peaks'),
        'performance_descriptive': perf_descriptives(df),
    }


def fmt_block(a):
    L = [f"=== {a['label']} (n={a['n_subjects']}: {a['n_ASD']} ASD / {a['n_TD']} TD) ==="]
    lk = a['leakage']
    if lk:
        L.append(f"  Leakage (F1 vs eeg_leakage): Spearman rho={lk['rho']:.3f}, p={lk['p']:.4f} (n={lk['n']})")
    pl = a['leakage_partial']
    if pl and 'p' in pl:
        L.append(f"    controlling for {pl['control']}: partial rho={pl['partial_rho']:.3f}, p={pl['p']:.4f}")
    for key, name in [('cohort_mean_r', 'reconstruction r'), ('cohort_f1', 'F1')]:
        c = a[key]
        if 'mannwhitney_p' in c:
            L.append(f"  Cohort {name}: MW p={c['mannwhitney_p']:.3f}, rank-biserial={c['rank_biserial']:.3f}, "
                     f"TOST p={c.get('tost_p', float('nan')):.3f} (margin +/-{EQUIV_MARGIN_R})")
        else:
            L.append(f"  Cohort {name}: too few per cohort to test")
    pd_ = a['performance_descriptive']
    if 'f1' in pd_:
        f = pd_['f1']; r = pd_['mean_r']
        L.append(f"  [descriptive only] r={r['mean']:.3f} [{r['ci95_low']:.3f},{r['ci95_high']:.3f}], "
                 f"F1={f['mean']:.3f} [{f['ci95_low']:.3f},{f['ci95_high']:.3f}]")
    return "\n".join(L)


def main():
    paths = {s: os.path.join(OUTPUT_DIR, f'per_subject_metrics_{s}.csv') for s in ('val1', 'val2')}
    for s, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p} -- run the main pipeline first.")

    val1 = pd.read_csv(paths['val1']); val1['split'] = 'val1'
    val2 = pd.read_csv(paths['val2']); val2['split'] = 'val2'
    # before any analysis: val2_only splits on cohort too, so both need correcting
    print('cohort labels:')
    val1 = relabel_cohorts(val1, 'val1')
    val2 = relabel_cohorts(val2, 'val2')
    pooled = pd.concat([val1, val2], ignore_index=True)
    pooled.to_csv(os.path.join(OUTPUT_DIR, 'pooled_per_subject_metrics.csv'), index=False)

    results = {
        'val2_only': analyse(val2, 'val2 only (clean)'),
        'pooled': analyse(pooled, 'val1 + val2 pooled (power-boost; relational only)'),
    }
    with open(os.path.join(OUTPUT_DIR, 'pooled_analyses.json'), 'w') as f:
        json.dump(results, f, indent=2)

    report = ("Pooled validation analyses\n"
              "Note: performance numbers below are DESCRIPTIVE ONLY. val1 informed model\n"
              "selection, so cite val2 alone for held-out performance. Pooling is valid for\n"
              "the cohort and leakage analyses because selection used aggregate loss only.\n\n"
              + fmt_block(results['val2_only']) + "\n\n" + fmt_block(results['pooled']) + "\n")
    print(report)
    with open(os.path.join(OUTPUT_DIR, 'pooled_analyses.md'), 'w') as f:
        f.write(report)

    print(f"Wrote pooled CSV, JSON and MD to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()