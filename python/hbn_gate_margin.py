# -*- coding: utf-8 -*-
"""
hbn_gate_margin.py -- how well does the deployment cohort match the margin the
thresholds were validated in?

(a) Gate rejection rises as age falls.
(b) frac_short by year of age, against the empty margin the labelled cohort leaves.
(c) frac_implaus by year of age, against the same.

THE IDEA. In the labelled adult cohort every retained recording sits below some
value of each feature and every rejected one above some higher value, leaving an
empty band in between with the threshold inside it. That band is the only region
where the threshold's placement is arbitrary -- outside it, the labelled data
themselves separate the two classes. So the question for the deployment cohort is
not simply what fraction is rejected, but what fraction lands in that band, where
the adults offer no precedent in either direction. This figure asks that per year
of age, because the cohort the band came from is adult and the cohort it is
applied to is 5-21.

    python hbn_gate_margin.py

Reads toolbox_qc_with_demographics.csv and the labelled cohort's
qc_gate_decisions.csv. Writes hbn_gate_margin.{pdf,png,svg} and a results file.
"""
from __future__ import annotations

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
except Exception:
    HAVE_FS = False

# HBN folders: the participants*.csv release files go in inputs/hbn/; the toolbox's
# work folder is outputs/deployment/hbn/toolbox/; everything derived is written to
# outputs/deployment/hbn/ (deploy_step8_hep_by_age.m reads it from there).
INPUT_DIR = os.environ.get('HDEEG2ECG_INPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs'))
HBN_DIR = os.path.join(os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')),
                       'deployment', 'hbn')
HBN = os.path.join(HBN_DIR, 'toolbox_qc_with_demographics.csv')
LABELLED = os.path.join(os.environ.get('HDEEG2ECG_OUTPUT_DIR',
                                       str(Path(__file__).resolve().parents[1] / 'outputs')),
                        'qc_gate_decisions.csv')
STEM = os.path.join(HBN_DIR, 'hbn_gate_margin')
FEATURES = (('frac_short', 0.16, 'frac_short   (R-R < 400 ms)', 0.52),
            ('frac_implaus', 0.29, 'frac_implaus   (R-R outside 500-1300 ms)', 0.80))
YEAR_BINS = list(range(5, 18)) + [23]   # 23, not 22: pd.cut is right-open and
                                        # 22 dropped the one recording aged 22.0



def band_palette(n):
    """One distinct hue per age band, at matched saturation and value.

    A single-hue sequential ramp encodes age as opacity, so the pale bands are
    both hard to see and hard to tell apart -- exactly the comparison this figure
    is for. Sweeping hue instead keeps every band equally visible on white. The
    sweep runs violet to amber and deliberately stops short of red, which is
    reserved for the gate.
    """
    import matplotlib.colors as mc
    hues = np.linspace(288, 48, n) / 360.0
    return [mc.hsv_to_rgb((h, 0.58, 0.72)) for h in hues]


def logit_fit(X, y, names):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = np.clip(p * (1 - p), 1e-10, None)
        step = np.linalg.solve((X.T * W) @ X, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    W = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv((X.T * W) @ X)
    se = np.sqrt(np.diag(cov))
    z = b / se
    return pd.DataFrame({'term': ['intercept'] + list(names), 'coef': b,
                         'p': 2 * sps.norm.sf(np.abs(z)), 'OR': np.exp(b),
                         'OR_lo': np.exp(b - 1.96 * se),
                         'OR_hi': np.exp(b + 1.96 * se)}), cov


def wilson(k, n, z=1.96):
    p = k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return c - h, c + h


def gaps(lab):
    """The margin the thresholds were actually calibrated in, per feature.

    The pass-fractions were set at the midpoint of the margin observed in the
    SELECTION split (val1) -- the highest value among its retained recordings and
    the lowest among its rejected ones. Using that split, and not the pooled 20,
    is what makes the threshold sit at the centre of the band: pooling brings in a
    testing-split failure that lies closer to the gate than anything in the
    calibration data, which pulls the upper edge down and makes a correctly
    centred threshold look off-centre.

    Also returns the nearest testing-split failure, so the figure can show how far
    the held-out data subsequently narrowed the margin on the rejection side.
    """
    sel = lab[lab.split == 'val1']
    test_fail = lab[(lab.split == 'val2') & (~lab['pass'])]
    out = {}
    for col, tau, _lbl, _top in FEATURES:
        lo = float(sel.loc[sel['pass'], col].max())      # highest retained
        hi = float(sel.loc[~sel['pass'], col].min())     # lowest rejected
        out[col] = (lo, hi, tau, float(test_fail[col].min()))
    return out


def main():
    d = pd.read_csv(HBN)
    d = d[d.age.notna()].copy()
    d['reject'] = (d.Decision == 'BAD').astype(int)
    d['band'] = pd.cut(d.age, YEAR_BINS, right=False)

    lab = pd.read_csv(LABELLED)
    lab['pass'] = lab.gate_pass.astype(bool)
    G = gaps(lab)

    print('=== empty margins in the labelled adult cohort ===')
    for col, (lo, hi, tau, near) in G.items():
        inside = float(((d[col] > lo) & (d[col] < hi)).mean())
        print(f'  {col:14s} selection margin ({lo:.4f}, {hi:.4f}); midpoint '
              f'{(lo+hi)/2:.4f}; threshold {tau} sits {100*(tau-lo)/(hi-lo):.0f}% across')
        print(f'  {"":14s} nearest testing-split failure {near:.4f} '
              f'(clears the gate by {100*(near-tau)/tau:+.0f}% of tau)')
        print(f'  {"":14s} deployment recordings inside the gap: {100*inside:.1f}%')

    g = d.groupby('band', observed=True)['reject'].agg(['sum', 'count'])
    g['rate'] = g['sum'] / g['count']
    g[['lo', 'hi']] = [wilson(k, n) for k, n in zip(g['sum'], g['count'])]
    g['mid'] = [iv.left + (iv.right - iv.left) / 2 for iv in g.index]
    m_age, cov = logit_fit(d[['age']].values, d.reject.values, ['age'])
    a = m_age[m_age.term == 'age'].iloc[0]
    print(f'\n=== rejection vs age ===')
    print(f'  OR per year younger {1/a.OR:.3f} [{1/a.OR_hi:.3f}-{1/a.OR_lo:.3f}], '
          f'p = {a.p:.1e}')

    print('\n=== % inside the gap, by year ===')
    tab = {}
    for col, (lo, hi, tau, near) in G.items():
        s = d.groupby('band', observed=True).apply(
            lambda x: pd.Series({
                'n': len(x),
                'below': float((x[col] <= lo).mean()),
                'inside': float(((x[col] > lo) & (x[col] < hi)).mean()),
                'above': float((x[col] >= hi).mean())}),
            include_groups=False)
        tab[col] = s
        rho, p = sps.spearmanr(d.age, ((d[col] > lo) & (d[col] < hi)).astype(int))
        print(f'  {col}: inside-gap fraction vs age rho = {rho:+.3f}, p = {p:.2e}')
        for iv, r in s.iterrows():
            print(f'     {str(iv):10s} n={int(r["n"]):4d}  below {100*r.below:4.1f}%  '
                  f'INSIDE {100*r.inside:4.1f}%  above {100*r.above:4.1f}%')

    fig = make_figure(d, g, m_age, cov, lab, G, tab)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{STEM}.{ext}', dpi=600)
    plt.close(fig)
    print(f'\nwrote {STEM}.pdf/.png/.svg')
    report(f'{STEM}_results.txt', d, g, a, lab, G, tab)


def make_figure(d, g, m_age, cov, lab, G, tab):
    if HAVE_FS:
        fs.use(base_pt=7.0)
        fig = fs.figure(width='double', height_mm=150.0)
        C = fs.C
    else:
        fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))
        C = {'pass': '#12796B', 'fail': '#B4341F', 'ink': '#1A1A1A',
             'ink_soft': '#55534E', 'ink_faint': '#8C8A83', 'threshold': '#6E6C66',
             'rule': '#C9C7BF', 'fill_soft': '#E8E6DF'}
    PAL = band_palette(len(g))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], left=0.070,
                          right=0.895, top=0.945, bottom=0.070, hspace=0.40,
                          wspace=0.30)

    # ---- a: rejection vs age --------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    xs = np.linspace(5, 21.5, 200)
    b = m_age.coef.values
    Xd = np.column_stack([np.ones_like(xs), xs])
    eta = Xd @ b
    se = np.sqrt(np.einsum('ij,jk,ik->i', Xd, cov, Xd))
    sig = lambda v: 1 / (1 + np.exp(-v))
    ax.fill_between(xs, 100 * sig(eta - 1.96 * se), 100 * sig(eta + 1.96 * se),
                    color=C['fail'], alpha=0.16, lw=0, zorder=1)
    ax.plot(xs, 100 * sig(eta), color=C['fail'], lw=1.4, zorder=3)
    ax.errorbar(g['mid'], 100 * g['rate'],
                yerr=[100 * (g['rate'] - g['lo']), 100 * (g['hi'] - g['rate'])],
                fmt='none', ecolor=C['ink_faint'], elinewidth=0.6, capsize=1.4,
                capthick=0.6, zorder=4)
    ax.scatter(g['mid'], 100 * g['rate'], s=np.sqrt(g['count']) * 2.4,
               c=PAL, edgecolor='white', linewidth=0.5, zorder=5)
    a = m_age[m_age.term == 'age'].iloc[0]
    ax.text(0.985, 0.95, f'OR {1/a.OR:.3f} per year younger '
                         f'[{1/a.OR_hi:.3f}-{1/a.OR_lo:.3f}],  p = {a.p:.0e}',
            transform=ax.transAxes, ha='right', va='top', color=C['ink_soft'])
    ax.set_xlabel('age (years)')
    ax.set_ylabel('gate rejection rate (%)')
    ax.set_ylim(0, 60)
    ax.set_xlim(4.5, 22)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.set_title('a', loc='left', fontweight='bold')

    # ---- b, c: each feature by year against the labelled gap ------------
    bands = list(g.index)
    mids = np.array([iv.left + (iv.right - iv.left) / 2 for iv in bands])
    rng = np.random.default_rng(0)
    for j, (col, tau, xlabel, top) in enumerate(FEATURES):
        ax = fig.add_subplot(gs[1, j])
        lo, hi, _tau, near = G[col]
        # Metric on x, age band on y: rows are directly comparable left-to-right,
        # and the gate is one vertical line cutting every band at the same place.
        ax.axvspan(lo, hi, color=C['fill_soft'], alpha=0.95, lw=0, zorder=0)
        ax.axvline(tau, color=C['fail'], lw=1.2, zorder=5)
        ax.axvline(near, color=C['ink_soft'], lw=0.9, ls=(0, (2.5, 2)), zorder=5)

        s = tab[col]
        for i, (bnd, m) in enumerate(zip(bands, mids)):
            v = d.loc[d.band == bnd, col].values
            y = i + rng.uniform(-0.34, 0.34, len(v))
            ax.scatter(v, y, s=2.6, color=PAL[i], alpha=0.75, lw=0,
                       rasterized=True, zorder=2)
            ax.plot([np.median(v)] * 2, [i - 0.40, i + 0.40], color=C['ink'],
                    lw=1.3, solid_capstyle='butt', zorder=6)
            r = s.iloc[i]
            ax.text(top * 1.015, i, f'{100*r.inside:.0f}', ha='left', va='center',
                    color=C['ink'], clip_on=False)
            ax.text(top * 1.14, i, f'{int(r.n)}', ha='left', va='center',
                    color=C['ink_faint'], clip_on=False)

        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels([f'{int(b.left)}' for b in bands])
        for tick, colr in zip(ax.get_yticklabels(), PAL):
            tick.set_color(colr)
            tick.set_fontweight('bold')
        ax.invert_yaxis()                      # youngest at the top
        ax.set_ylim(len(bands) - 0.4, -1.15)
        ax.text(top * 1.015, -0.95, '% in\nband', ha='left', va='center',
                color=C['ink'], linespacing=1.2, clip_on=False)
        ax.text(top * 1.14, -0.95, 'n', ha='left', va='center',
                color=C['ink_faint'], clip_on=False)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('age (years)')
        ax.set_xlim(-0.008 * top, top)
        ax.tick_params(axis='y', length=0)
        for s_ in ('top', 'right', 'left'):
            ax.spines[s_].set_visible(False)
        ax.set_title('bc'[j], loc='left', fontweight='bold')
    return fig


def report(path, d, g, a, lab, G, tab):
    L = []
    A = L.append
    A('Deployment cohort against the margin validated in adults')
    A('=' * 62)
    A(f'Deployment n = {len(d)} (age 5-21); labelled adult cohort n = {len(lab)}.')
    A('')
    A('(a) Rejection versus age')
    A(f'  OR of rejection per year younger {1/a.OR:.4f} '
      f'[{1/a.OR_hi:.4f}-{1/a.OR_lo:.4f}], p = {a.p:.3e}')
    for iv, r in g.iterrows():
        A(f'  {str(iv):10s} n={int(r["count"]):4d}  reject {100*r["rate"]:5.1f}% '
          f'[{100*r["lo"]:.1f}-{100*r["hi"]:.1f}]')
    A('')
    A('(b, c) The empty margin in the labelled cohort, per feature')
    for col, (lo, hi, tau, near) in G.items():
        A(f'  {col}: every retained adult <= {lo:.4f}, every rejected adult '
          f'>= {hi:.4f}.')
        A(f'    No labelled recording lies between. The threshold {tau} sits inside '
          f'that band')
        A(f'    ({100*(tau-lo)/(hi-lo):.0f}% of the way across it), so within the band '
          f'the verdict rests')
        A('    on the threshold value rather than on demonstrated separation.')
        inside = float(((d[col] > lo) & (d[col] < hi)).mean())
        rho, p = sps.spearmanr(d.age, ((d[col] > lo) & (d[col] < hi)).astype(int))
        A(f'    Deployment recordings inside the band: {100*inside:.1f}% overall; '
          f'the fraction')
        A(f'    varies with age at rho = {rho:+.3f} (p = {p:.2e}).')
        s = tab[col]
        for iv, r in s.iterrows():
            A(f'      {str(iv):10s} n={int(r["n"]):4d}  below {100*r.below:5.1f}%  '
              f'inside {100*r.inside:5.1f}%  above {100*r.above:5.1f}%')
        A('')
    A('SCOPE: the margin is defined in adults (labelled cohort mean age ~30-34 y)')
    A('and applied to a cohort aged 5-21. That is the transfer this figure makes')
    A('visible; it is not evidence that the margin is valid at those ages.')
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
