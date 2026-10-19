# -*- coding: utf-8 -*-
"""
hbn_gate_age_transfer.py -- does the reference-free gate reject by age rather than by
reconstruction failure?

The gate's pass-fractions were calibrated in adults marked at 62-82 bpm. The
deployment cohort is 5-21 years old, where resting heart rate is higher and more
variable, so a child with a genuinely fast rhythm can put R-R intervals under the
400 ms bound without anything having gone wrong with the reconstruction. Because
frac_short is the binding constraint in this corpus, the rejection rate cannot be
read as a reconstruction-failure rate until that is ruled out.

    python hbn_gate_age_transfer.py
    python hbn_gate_age_transfer.py --quality prep_quality.csv --quality-col prop_interp

Reads toolbox_qc_with_demographics.csv (from hbn_merge_demographics.py) and writes
a results file, a per-band CSV and a two-panel figure.

TWO THINGS TO READ BEFORE THE NUMBERS
-------------------------------------
1. apparent_HR is computed from the RECONSTRUCTED peak train -- the same train the
   gate's features come from. For a rejected recording it is inflated by
   over-detection and is not an estimate of that child's heart rate. Entering it as
   a predictor of pass/fail is therefore close to circular, and the age-only model
   is the primary one. apparent_HR is reported because the analysis plan asks for
   it, and it is interpretable only within the retained recordings.

2. The data-quality covariate the plan asks for (proportion of interpolated
   channels, or ASR-rejected fraction) is NOT in these files -- it lives in the EEG
   preprocessing pipeline. Supply it with --quality and it enters the full model;
   without it, that model cannot exclude two opposing effects cancelling. The
   script states this in its own output rather than letting the omission pass.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib.pyplot as plt

import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import fig_style as fs
    HAVE_FS = True
except Exception:                                # the figure still builds without it
    HAVE_FS = False

# HBN folders: the participants*.csv release files go in inputs/hbn/; the toolbox's
# work folder is outputs/deployment/hbn/toolbox/; everything derived is written to
# outputs/deployment/hbn/ (deploy_step8_hep_by_age.m reads it from there).
INPUT_DIR = os.environ.get('HDEEG2ECG_INPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs'))
HBN_DIR = os.path.join(os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')),
                       'deployment', 'hbn')
IN = os.path.join(HBN_DIR, 'toolbox_qc_with_demographics.csv')
BINS = [5, 7, 9, 11, 13, 15, 18, 22]             # years; last band wide, few subjects


# ============================================================
# LOGISTIC REGRESSION  (IRLS + Wald SEs; statsmodels is not installed)
# ============================================================
def logit_fit(X, y, names, max_iter=100, tol=1e-10):
    """Maximum-likelihood logistic regression, intercept prepended.

    Returns coefficients, Wald standard errors from the inverse observed
    information, z, p, odds ratios and 95% CIs. Hand-rolled because statsmodels is
    absent from this environment; checked against sklearn in validate_logit().
    """
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        W = np.clip(p * (1 - p), 1e-10, None)
        step = np.linalg.solve((X.T * W) @ X, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    p = 1.0 / (1.0 + np.exp(-(X @ b)))
    W = np.clip(p * (1 - p), 1e-10, None)
    se = np.sqrt(np.diag(np.linalg.inv((X.T * W) @ X)))
    z = b / se
    return pd.DataFrame({'term': ['intercept'] + list(names), 'coef': b, 'se': se,
                         'z': z, 'p': 2 * sps.norm.sf(np.abs(z)), 'OR': np.exp(b),
                         'OR_lo': np.exp(b - 1.96 * se),
                         'OR_hi': np.exp(b + 1.96 * se)})


def validate_logit():
    """Two independent checks, because a hand-rolled GLM carries a manuscript claim.

    Coefficients are checked against sklearn with regularisation off and its
    tolerance tightened -- at sklearn's default tol=1e-4 the lbfgs solver stops
    early and disagrees at ~1e-4, which is its convergence, not our error.
    Standard errors are checked against the closed form for a 2x2 table, where the
    log odds ratio has SE sqrt(1/a + 1/b + 1/c + 1/d); sklearn cannot check this
    because it does not produce standard errors at all.
    """
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 3))
    eta = 0.4 + X @ np.array([0.8, -0.5, 0.2])
    y = (rng.uniform(size=800) < 1 / (1 + np.exp(-eta))).astype(float)
    mine = logit_fit(X, y, ['a', 'b', 'c'])
    sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=5000).fit(X, y)
    ref = np.r_[sk.intercept_, sk.coef_.ravel()]
    gap = float(np.max(np.abs(mine.coef.values - ref)))
    print(f'  coefficients vs sklearn (tol 1e-10): max |diff| = {gap:.2e}')
    assert gap < 1e-6, 'IRLS coefficients disagree with sklearn'

    a, b, c, dd = 40, 60, 25, 75          # exposed: 40/100 events; unexposed: 25/100
    xx = np.r_[np.ones(a + b), np.zeros(c + dd)][:, None]
    yy = np.r_[np.ones(a), np.zeros(b), np.ones(c), np.zeros(dd)]
    m = logit_fit(xx, yy, ['x']).iloc[1]
    or_cf = (a / b) / (c / dd)
    se_cf = np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
    print(f'  2x2 odds ratio: fitted {m.OR:.6f} vs closed form {or_cf:.6f}')
    print(f'  2x2 Wald SE   : fitted {m.se:.6f} vs closed form {se_cf:.6f}')
    assert abs(m.OR - or_cf) < 1e-8 and abs(m.se - se_cf) < 1e-8, 'SEs are wrong'


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cochran_armitage(k, n, scores):
    """Trend in proportions across ordered groups."""
    k, n, s = np.asarray(k, float), np.asarray(n, float), np.asarray(scores, float)
    N = n.sum()
    pbar = k.sum() / N
    T = np.sum(s * (k - n * pbar))
    var = pbar * (1 - pbar) * (np.sum(n * s ** 2) - (np.sum(n * s) ** 2) / N)
    z = T / np.sqrt(var)
    return float(z), float(2 * sps.norm.sf(abs(z)))


# ============================================================
# ANALYSIS
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quality', default=None,
                    help='CSV with participant_id and a data-quality column')
    ap.add_argument('--quality-col', default=None)
    ap.add_argument('--outstem', default=os.path.join(HBN_DIR, 'hbn_gate_age_transfer'))
    args = ap.parse_args()

    print('=== validation ===')
    validate_logit()

    d = pd.read_csv(IN)
    n_all = len(d)
    d = d[d.age.notna()].copy()
    d['pass'] = (d.Decision == 'GOOD').astype(int)
    print('\n=== sample ===')
    print(f'  QC subjects total           : {n_all}')
    print(f'  with demographics (analysed): {len(d)} '
          f'({100*len(d)/n_all:.1f}%; {n_all-len(d)} excluded for missing age)')
    print(f'  retained {int(d["pass"].sum())}  rejected {int((1-d["pass"]).sum())}  '
          f'pass rate {100*d["pass"].mean():.1f}%')

    qcol = None
    if args.quality:
        q = pd.read_csv(args.quality)
        qcol = args.quality_col or [c for c in q.columns if c != 'participant_id'][0]
        d = d.merge(q[['participant_id', qcol]], on='participant_id', how='left')
        print(f'  quality covariate "{qcol}": {d[qcol].notna().sum()}/{len(d)} present')

    # ---- (i) retained vs rejected ----------------------------------------
    print('\n=== (i) retained vs rejected ===')
    for col, lab in (('age', 'age (y)'), ('apparent_HR', 'reconstructed HR (bpm)')):
        a, b = d.loc[d['pass'] == 1, col], d.loc[d['pass'] == 0, col]
        u, p = sps.mannwhitneyu(a, b)
        rb = 1 - 2 * u / (len(a) * len(b))
        print(f'  {lab:24s} retained {a.median():6.1f} '
              f'[{a.quantile(.25):.1f}-{a.quantile(.75):.1f}]   '
              f'rejected {b.median():6.1f} [{b.quantile(.25):.1f}-{b.quantile(.75):.1f}]'
              f'   p={p:.3g}, rank-biserial {rb:+.3f}')

    d['band'] = pd.cut(d.age, BINS, right=False)
    g = d.groupby('band', observed=True)['pass'].agg(['sum', 'count'])
    g['rate'] = g['sum'] / g['count']
    g[['lo', 'hi']] = [wilson(k, n) for k, n in zip(g['sum'], g['count'])]
    g['mid'] = [iv.left + (iv.right - iv.left) / 2 for iv in g.index]
    print('\n=== pass rate by age band (Wilson 95% CI) ===')
    for iv, r in g.iterrows():
        print(f'  {str(iv):12s} n={int(r["count"]):4d}  pass {int(r["sum"]):4d}  '
              f'{100*r["rate"]:5.1f}%  [{100*r["lo"]:.1f}-{100*r["hi"]:.1f}]')
    z_ca, p_ca = cochran_armitage(g['sum'], g['count'], g['mid'])
    print(f'  Cochran-Armitage trend: z = {z_ca:+.3f}, p = {p_ca:.4f}')

    # ---- (ii) logistic models --------------------------------------------
    print('\n=== (ii) logistic regression of pass/fail ===')
    print('  M1  pass ~ age   [PRIMARY: age is exogenous to the gate]')
    m1 = logit_fit(d[['age']].values, d['pass'].values, ['age'])
    print(m1.to_string(index=False, float_format=lambda v: f'{v:9.4f}'))

    d['age_c'] = d.age - d.age.mean()
    Xq = np.column_stack([d.age_c.values, d.age_c.values ** 2])
    m2 = logit_fit(Xq, d['pass'].values, ['age_c', 'age_c^2'])
    print('\n  M2  pass ~ age + age^2   [a flat line can hide a hump]')
    print(m2.to_string(index=False, float_format=lambda v: f'{v:9.4f}'))

    X = d[['age']].copy()
    X['apparent_HR'] = d.apparent_HR
    X['sex_M'] = (d.sex == 'M').astype(int)
    names = ['age', 'apparent_HR', 'sex_M']
    if qcol is not None and d[qcol].notna().any():
        X[qcol] = d[qcol]
        names.append(qcol)
    ok = X.notna().all(axis=1)
    m3 = logit_fit(X[ok].values, d.loc[ok, 'pass'].values, names)
    print(f'\n  M3  pass ~ {" + ".join(names)}   (n = {int(ok.sum())})')
    print('      apparent_HR derives from the reconstructed peak train and is inflated')
    print('      by over-detection in rejected recordings; its coefficient is NOT')
    print('      evidence about physiological heart rate.')
    print(m3.to_string(index=False, float_format=lambda v: f'{v:9.4f}'))
    if qcol is None:
        print('\n      INCOMPLETE: no data-quality covariate supplied (--quality), so this')
        print('      model cannot exclude two opposing effects cancelling, which the')
        print('      analysis plan explicitly requires.')

    # ---- positive control -------------------------------------------------
    ret = d[d['pass'] == 1]
    rho, p_hr = sps.spearmanr(ret.age, ret.apparent_HR)
    print('\n=== positive control: is the reconstructed rate physiological? ===')
    print(f'  retained only, HR vs age: Spearman rho = {rho:+.3f}, p = {p_hr:.3g} '
          f'(n = {len(ret)})')
    print('  resting heart rate falls with age in childhood, so a negative rho means the')
    print('  reconstructed rate carries real physiology rather than detector noise.')

    g.drop(columns='mid').to_csv(f'{args.outstem}_by_age.csv')
    fig = make_figure(d, g, m1)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{args.outstem}.{ext}', dpi=600)
    plt.close(fig)
    print(f'\nwrote {args.outstem}_by_age.csv, {args.outstem}.pdf/.png/.svg')
    write_report(f'{args.outstem}_results.txt', d, g, m1, m2, m3, names,
                 z_ca, p_ca, rho, p_hr, qcol)


def make_figure(d, g, m1):
    if HAVE_FS:
        fs.use(base_pt=7.0)
        fig = fs.figure(width='double', height_mm=74.0)
        C = fs.C
    else:
        fig = plt.figure(figsize=(183 / 25.4, 74 / 25.4))
        C = {'pass': '#12796B', 'fail': '#B4341F', 'ink': '#1A1A1A',
             'ink_soft': '#55534E', 'ink_faint': '#8C8A83', 'threshold': '#6E6C66'}
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], left=0.075, right=0.985,
                          top=0.90, bottom=0.21, wspace=0.26)

    ax = fig.add_subplot(gs[0])
    ax.errorbar(g['mid'], 100 * g['rate'],
                yerr=[100 * (g['rate'] - g['lo']), 100 * (g['hi'] - g['rate'])],
                fmt='o', ms=3.4, color=C['ink'], ecolor=C['ink_faint'],
                elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
    xs = np.linspace(d.age.min(), d.age.max(), 200)
    b0, b1 = m1.coef.values
    ax.plot(xs, 100 / (1 + np.exp(-(b0 + b1 * xs))), color=C['threshold'], lw=0.9,
            ls=(0, (3, 2)), zorder=2, label='logistic fit')
    ax.axhline(100 * d['pass'].mean(), color=C['ink_faint'], lw=0.5, ls=(0, (1.5, 1.5)))
    ax.text(d.age.max(), 100 * d['pass'].mean() + 1.5, 'cohort mean', ha='right',
            va='bottom', color=C['ink_faint'])
    for n, m, r in zip(g['count'], g['mid'], g['rate']):
        ax.annotate(f'{int(n)}', (m, 100 * r), textcoords='offset points',
                    xytext=(0, 8), ha='center', color=C['ink_faint'])
    ax.set_xlabel('age (years)')
    ax.set_ylabel('gate pass rate (%)')
    ax.set_ylim(0, 100)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title('a', loc='left', fontweight='bold')

    ax = fig.add_subplot(gs[1])
    bins = np.linspace(40, 180, 41)
    for k, col, lab in ((1, C['pass'], 'retained'), (0, C['fail'], 'rejected')):
        ax.hist(d.loc[d['pass'] == k, 'apparent_HR'], bins=bins, color=col,
                alpha=0.55, lw=0, density=True, label=lab)
    ax.axvline(120, color=C['threshold'], lw=0.6, ls=(0, (2.5, 2)))
    ax.text(122, ax.get_ylim()[1] * 0.96, '120 bpm\n(500 ms bound)', ha='left',
            va='top', color=C['threshold'])
    ax.set_xlabel('reconstructed heart rate (bpm)')
    ax.set_ylabel('density')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(loc='upper right', frameon=False)
    ax.set_title('b', loc='left', fontweight='bold')
    return fig


def write_report(path, d, g, m1, m2, m3, names, z_ca, p_ca, rho, p_hr, qcol):
    L = []
    A = L.append
    A('Gate transfer to a developmental cohort -- results')
    A('=' * 62)
    A(f'n = {len(d)} recordings with age; {int(d["pass"].sum())} retained, '
      f'{int((1 - d["pass"]).sum())} rejected (pass rate {100*d["pass"].mean():.1f}%).')
    A(f'Age median {d.age.median():.1f} y, range {d.age.min():.1f}-{d.age.max():.1f}.')
    A('')
    A('Pass rate by age band (Wilson 95% CI):')
    for iv, r in g.iterrows():
        A(f'  {str(iv):12s} n={int(r["count"]):4d}  {100*r["rate"]:5.1f}% '
          f'[{100*r["lo"]:.1f}-{100*r["hi"]:.1f}]')
    A(f'Cochran-Armitage trend across bands: z = {z_ca:+.3f}, p = {p_ca:.4f}')
    A('')
    b = m1[m1.term == 'age'].iloc[0]
    A(f'M1 logistic, pass ~ age: OR per year {b.OR:.4f} '
      f'[{b.OR_lo:.4f}-{b.OR_hi:.4f}], p = {b.p:.4f}')
    q = m2[m2.term == 'age_c^2'].iloc[0]
    curve = 'evidence of curvature' if q.p < 0.05 else 'no evidence of curvature'
    A(f'M2 quadratic age term: coef {q.coef:+.5f}, p = {q.p:.4f} ({curve})')
    A('')
    A(f'M3 full model ({", ".join(names)}):')
    A(m3.to_string(index=False))
    if qcol is None:
        A('')
        A('INCOMPLETE: no data-quality covariate (proportion of interpolated channels')
        A('or ASR-rejected fraction) was supplied. Rerun with --quality when available;')
        A('until then M3 cannot exclude two opposing effects cancelling.')
    A('')
    A(f'Positive control (retained only): reconstructed HR vs age, '
      f'Spearman rho = {rho:+.3f}, p = {p_hr:.3g}')
    A('')
    A('CAVEAT: apparent_HR is computed from the reconstructed peak train and is')
    A('inflated by over-detection in rejected recordings, so it is not an estimate')
    A('of those children heart rates. M1 (age only) is the primary test.')
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
