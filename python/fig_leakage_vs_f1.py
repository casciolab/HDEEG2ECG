# -*- coding: utf-8 -*-
"""
Fig. 2: R-peak detection F1 against cardiac-field leakage for the held-out
participants (selection + testing, n = 20), with the Spearman statistics quoted
in the Results.

Input : <OUTPUT_DIR>/pooled_per_subject_metrics.csv  (written by validation_pooled_metrics.py)
Output: figures/fig_leakage_vs_f1.png next to this script

@author: Matias E. Fraile-Vazquez
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(REPO_ROOT, 'outputs'))
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
BLUE = '#3A6EA5'

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'pooled_per_subject_metrics.csv'))
df = df[df.split.isin(['val1', 'val2'])].dropna(subset=['eeg_leakage', 'f1'])
sel, test = df[df.split == 'val1'], df[df.split == 'val2']

for name, d in (('testing', test), ('pooled', df), ('selection', sel)):
    rho, p = spearmanr(d.eeg_leakage, d.f1)
    print(f'{name:9s} n={len(d):2d}  Spearman rho={rho:+.2f}  P={p:.3f}')
ok, bad = test[test.f1 >= 0.87], test[test.f1 <= 0.48]
print(f'testing gap: failures <= {bad.eeg_leakage.max():.2f}, successes >= {ok.eeg_leakage.min():.2f}')

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(sel.eeg_leakage, sel.f1, s=60, marker='o', color=BLUE, label='selection', zorder=3)
ax.scatter(test.eeg_leakage, test.f1, s=60, marker='s', facecolors='white', edgecolors=BLUE,
           linewidths=1.5, label='testing', zorder=3)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel('EEG cardiac leakage (R-peak-triggered peak-to-peak, dimensionless)')
ax.set_ylabel('R-peak detection $F_1$')
ax.set_title(f'Reconstruction quality tracks leakage (held-out, $n$ = {len(df)})')
ax.legend(loc='lower right', frameon=False, fontsize=9)
fig.tight_layout()

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, 'fig_leakage_vs_f1.png')
fig.savefig(out, dpi=300)
print('saved', out)
