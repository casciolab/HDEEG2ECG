# -*- coding: utf-8 -*-
"""
Quality-control rejection by year of age in the Healthy Brain Network (Results:
rejection rose as age fell).

Reads toolbox_qc_with_demographics.csv (from hbn_merge_demographics.py) and tabulates,
for each single-year band (17-22 pooled), how many recordings the reference-free gate
passed and rejected, with Wilson 95% intervals on the rejection rate, and the ratio of
the youngest band's rejection rate to the oldest's. Writes hbn_gate_by_age.csv.

    python hbn_gate_by_age.py

@author: Matias E. Fraile-Vazquez
"""
import os

import numpy as np
import pandas as pd

# HBN folders: the toolbox's work folder is outputs/deployment/hbn/toolbox/, and
# everything derived is written to outputs/deployment/hbn/.
HBN_DIR = os.path.join(os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')), 'deployment', 'hbn')
IN = os.path.join(HBN_DIR, 'toolbox_qc_with_demographics.csv')
OUT = os.path.join(HBN_DIR, 'hbn_gate_by_age.csv')
YEAR_BINS = list(range(5, 18)) + [23]   # 17-22 pooled; 23 because pd.cut is right-open


def wilson(k, n, z=1.96):
    p = k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return c - h, c + h


def main():
    d = pd.read_csv(IN)
    d = d[d.age.notna()].copy()
    d['reject'] = (d.Decision == 'BAD').astype(int)
    d['band'] = pd.cut(d.age, YEAR_BINS, right=False)

    g = d.groupby('band', observed=True)['reject'].agg(['sum', 'count'])
    g['rate'] = g['sum'] / g['count']
    g[['lo', 'hi']] = [wilson(k, n) for k, n in zip(g['sum'], g['count'])]
    g.index = [f'{iv.left}' if iv.right - iv.left == 1 else f'{iv.left}-{iv.right - 1}' for iv in g.index]
    g.index.name = 'age'
    g = g.rename(columns={'count': 'n', 'sum': 'rejected'})
    g.insert(1, 'passed', g['n'] - g['rejected'])

    print('age      n  passed  rejected')
    for age, r in g.iterrows():
        print(f'{age:>5s}  {int(r.n):4d}  {int(r.passed):5d}  {100 * r.rate:5.1f}%  '
              f'[{100 * r.lo:.1f}-{100 * r.hi:.1f}]')
    print(f'\nrejection rate, youngest vs oldest band: {g.rate.iloc[0] / g.rate.iloc[-1]:.2f}x')
    os.makedirs(HBN_DIR, exist_ok=True)
    g[['n', 'passed', 'rejected', 'rate', 'lo', 'hi']].to_csv(OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
