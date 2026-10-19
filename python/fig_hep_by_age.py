# -*- coding: utf-8 -*-
"""
Fig. 5: the heartbeat-evoked potential in the Healthy Brain Network, pooled and by age.

a  all ages pooled (reconstructed R-peaks) against the validated adults (R-peaks of
   the recorded ECG), with standard-error bands, on the 67-channel region of interest
b  each age band against the adult waveform, one row per band
c  agreement with the adult waveform by age (bootstrap 95% intervals) and the noise
   ceiling from each band's split-half reliability

Reads only analysis outputs: hep_by_age.mat from deploy_step8_hep_by_age.m and the
group-level adult reference (matlab/reference/adult_hep_reference.mat).

    python fig_hep_by_age.py [--hep-by-age PATH] [--reference PATH] [--outdir DIR]

@author: Matias E. Fraile-Vazquez
"""
import argparse
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat

import fig_style as fs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(REPO_ROOT, 'outputs'))
DEFAULT_IN = os.path.join(OUTPUT_DIR, 'deployment', 'hbn', 'deploy_step8_hep_by_age', 'hep_by_age.mat')
DEFAULT_REF = os.path.join(REPO_ROOT, 'matlab', 'reference', 'adult_hep_reference.mat')

ADULT = fs.C['true']          # adults, R-peaks of the recorded ECG (as in Fig. 4)
HBN = fs.C['recon']           # Healthy Brain Network, reconstructed R-peaks
# Age bands: one hue (the reconstructed-peak red), light at age 5 to dark at 17-22.
# Lightest step 2.6:1 against white. Rows and age positions identify the bands;
# colour only links panel b to panel c.
AGE_RAMP = LinearSegmentedColormap.from_list('age', ['#E58A72', HBN, '#5A1609'])
WIN = (200, 600)              # ms, analysis window
B_SPAN = (100, 600)           # ms shown in panel b


def load(path, name):
    return loadmat(path, squeeze_me=True, struct_as_record=False)[name]


def panel_pooled(ax, t, B, R):
    fs.shade_window(ax, *WIN, alpha=0.7)
    fs.zeroline(ax)
    ax.axvline(0, color=fs.C['rule'], lw=0.5, ls=(0, (1, 1.5)))
    ax.fill_between(t, R.waveform - R.sem, R.waveform + R.sem, color=ADULT, alpha=0.15, lw=0)
    ax.plot(t, R.waveform, color=ADULT, lw=1.0, label=f'Adults, recorded ECG (n = {int(R.n)})')
    ax.fill_between(t, B.wAll - B.semAll, B.wAll + B.semAll, color=HBN, alpha=0.25, lw=0)
    ax.plot(t, B.wAll, color=HBN, lw=1.0, ls=fs.LINESTYLE['recon'],
            label=f'HBN, reconstructed (n = {int(B.nAll):,})')
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks([-200, 0, 200, 400, 600])
    ax.set_xlabel('Time from R-peak (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend(loc='lower right', handlelength=2.2)          # empty corner, clear of the spike
    ax.text(sum(WIN) / 2, ax.get_ylim()[1], f'r = {float(B.rAll):.3f}', ha='center', va='top',
            color=fs.C['ink_soft'])
    ax.set_title('All ages pooled')


def panel_ridges(ax, t, B, R, colors):
    res = np.atleast_1d(B.res)
    sel = (t >= B_SPAN[0]) & (t <= B_SPAN[1])
    tt = t[sel]
    spread = max(np.ptp(np.r_[r.w[sel], R.waveform[sel]]) for r in res)
    step = 0.8 * spread                                   # rows may touch at the trough only
    fs.shade_window(ax, *WIN, alpha=0.7)
    for i, (r, col) in enumerate(zip(res, colors)):
        off = -i * step
        ax.axhline(off, color=fs.C['rule'], lw=0.4, zorder=1)
        ax.plot(tt, R.waveform[sel] + off, color=ADULT, lw=0.7, zorder=2)
        ax.fill_between(tt, r.w[sel] - r.sem[sel] + off, r.w[sel] + r.sem[sel] + off,
                        color=col, alpha=0.30, lw=0, zorder=3)
        ax.plot(tt, r.w[sel] + off, color=col, lw=1.2, zorder=4)
        ax.text(B_SPAN[0] - 12, off, str(r.label).replace('-', '–'), ha='right', va='center')
        ax.text(B_SPAN[1] + 12, off, f'{float(r.r):.2f}', ha='left', va='center',
                color=fs.C['ink_soft'])
    top = step * 0.75
    ax.text(B_SPAN[0] - 12, top, 'Age', ha='right', va='bottom', color=fs.C['ink_soft'])
    ax.text(B_SPAN[1] + 12, top, 'r', ha='left', va='bottom', color=fs.C['ink_soft'], style='italic')
    # 1-uV scale bar below the last row, left of the shaded window, clear of the traces
    y0 = -(len(res) - 1) * step - spread * 0.55 - 1.0
    ax.set_xlim(*B_SPAN)
    ax.set_ylim(y0 - 0.15, top + spread * 0.35)
    fs.despine(ax, keep=('bottom',))
    ax.set_yticks([])
    ax.set_xticks([100, 200, 300, 400, 500, 600])
    ax.set_xlabel('Time from R-peak (ms)')
    ax.plot([125, 125], [y0, y0 + 1.0], color=fs.C['ink'], lw=0.9, solid_capstyle='butt')
    ax.text(132, y0 + 0.5, '1 µV', ha='left', va='center', color=fs.C['ink_soft'])
    ax.plot([], [], color=ADULT, lw=0.7, label='Adults, recorded ECG')
    ax.plot([], [], color=colors[len(colors) // 2], lw=1.2, label='HBN age band, reconstructed')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.6)
    ax.set_title('Each age band against the adults', pad=14)


def panel_agreement(ax, B, colors):
    res = np.atleast_1d(B.res)
    x = np.array([r.ageMean for r in res], float)
    y = np.array([r.r for r in res], float)
    lo = np.array([r.rLo for r in res], float)
    hi = np.array([r.rHi for r in res], float)
    ceil = np.array([r.ceiling for r in res], float)
    ax.plot(x, ceil, color=fs.C['ink_faint'], lw=0.9, ls=(0, (3, 2)))
    ax.text(x[-1] + 0.4, ceil[-1], 'noise\nceiling', ha='left', va='center',
            color=fs.C['ink_soft'], linespacing=1.1)
    ax.plot(x, y, color=fs.C['rule'], lw=0.8, zorder=1)
    for xi, yi, l, h, col in zip(x, y, lo, hi, colors):
        ax.plot([xi, xi], [l, h], color=col, lw=1.0, solid_capstyle='butt', zorder=2)
        ax.plot(xi, yi, 'o', ms=4.2, color=col, mec='white', mew=0.6, zorder=3)
    t = B.trend
    fs.stat_text(ax, f'ρ = {float(t.rhoObs):+.2f}, P = {float(t.pObs):.3f}', x=0.97, y=0.06, va='bottom')
    ax.set_xlim(4, 21)
    ax.set_xticks([5, 10, 15, 20])
    ax.set_ylim(0.4, 1.04)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Correlation with adult HEP (r)')
    ax.set_title('Agreement with the adult HEP')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hep-by-age', default=DEFAULT_IN)
    ap.add_argument('--reference', default=DEFAULT_REF)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    B = load(args.hep_by_age, 'byAge')
    R = load(args.reference, 'ref')
    assert np.allclose(B.times, R.times), 'reference and age bands are on different time axes'
    assert np.allclose(B.refW, R.waveform, atol=1e-5), 'reference waveform differs from hep_by_age.mat'
    t = np.asarray(B.times, float) * 1000
    colors = [AGE_RAMP(v) for v in np.linspace(0, 1, len(np.atleast_1d(B.res)))]

    fs.use()
    fig = fs.figure('double', height_mm=118)
    ax_a = fig.add_axes([0.075, 0.60, 0.385, 0.34])
    ax_c = fig.add_axes([0.075, 0.10, 0.385, 0.34])
    ax_b = fig.add_axes([0.585, 0.10, 0.345, 0.807])       # title level with panel a's
    panel_pooled(ax_a, t, B, R)
    panel_agreement(ax_c, B, colors)
    panel_ridges(ax_b, t, B, R, colors)
    for ax in (ax_a, ax_c):
        fs.offset_spines(ax)
    fs.panel_label(ax_a, 'a', dx=-0.13)
    fs.panel_label(ax_c, 'c', dx=-0.13)
    fs.panel_label(ax_b, 'b', dx=-0.10, dy=1.067)
    fs.check_text_sizes(fig)
    fs.save(fig, 'fig_hep_by_age', outdir=args.outdir, formats=('pdf', 'png'))


if __name__ == '__main__':
    main()
