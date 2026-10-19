# -*- coding: utf-8 -*-
"""
Table 1: reconstruction and R-peak recovery in the held-out splits, as
participant-level means with 95% bootstrap CIs (5,000 resamples, seed 42).
The last column restricts the testing split to the recordings the reference-free
gate retains (thresholds fixed on the selection split).

Inputs (in OUTPUT_DIR): pooled_per_subject_metrics.csv  (validation_pooled_metrics.py)
                        qc_gate_decisions.csv            (qc_gate_thresholds.py)

@author: Matias E. Fraile-Vazquez
"""
import os
import re
import pandas as pd
from validation_pooled_metrics import OUTPUT_DIR, cluster_bootstrap_ci

ROWS = [('Pearson $r$', 'mean_r', 3), ('R-peak F1', 'f1', 3),
        ('\\quad Sensitivity', 'sensitivity', 3), ('\\quad PPV', 'ppv', 3),
        ('Abs.\\ timing error (ms)', 'timing_err_ms_mean', 2),
        ('Signed latency bias (ms)', 'timing_bias_ms', 2)]

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'pooled_per_subject_metrics.csv'))
df['id'] = df.file.map(lambda f: int(re.search(r'sub_(\d+)', f).group(1)))
gate = pd.read_csv(os.path.join(OUTPUT_DIR, 'qc_gate_decisions.csv'))
kept = set(gate.loc[(gate.split == 'val2') & gate.gate_pass, 'id'])

cols = {'Selection': df[df.split == 'val1'],
        'Testing': df[df.split == 'val2'],
        'Testing, QC-retained': df[(df.split == 'val2') & df.id.isin(kept)]}
print(' & '.join(['Metric'] + [f'{k} ($n{{=}}{len(v)}$)' for k, v in cols.items()]) + ' \\\\')


def cell(v, d):
    m, lo, hi = cluster_bootstrap_ci(v)
    f = lambda x: f'{x:.{d}f}' if x >= 0 else f'${x:.{d}f}$'
    return f'{f(m)} ({f(lo)}--{f(hi)})' if lo >= 0 else f'{f(m)} ({f(lo)} to {f(hi)})'


for label, col, d in ROWS:
    print(' & '.join([label] + [cell(v[col].values, d) for v in cols.values()]) + ' \\\\')
