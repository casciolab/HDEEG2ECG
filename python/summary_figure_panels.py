# -*- coding: utf-8 -*-
"""
summary_figure_panels.py -- panels of an earlier five-panel summary figure (not in the paper).

    a  raw 128-channel EEG            "the heartbeat is invisible"
    b  reconstructed vs true ECG      "...but the model reads it out"
    c  per-beat latency error         "...to within one sample"
    d  leakage vs F1, gate verdict    "...and it knows when it failed"
    e  2,078-subject HEP carpet       "...at a scale nobody else has"

Each panel is an independent function taking plain arrays, so you can drop any
one of them into another figure or reorder them without untangling the layout.
Wire up load_inputs() to your own files; everything below it is style, not data.

    python summary_figure_panels.py

Requires fig_style.py in the same directory.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import fig_style as fs

SFREQ = 256
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


# ============================================================
# 1. DATA  -- the only part you need to edit
# ============================================================
def load_inputs(run_dir='.'):
    """Return everything the panels need.

    Replace each block marked EDIT. Shapes and units are given explicitly; if
    an array arrives with a different orientation the panels will look wrong
    rather than error, so check the shapes printed at the end of this function.
    """
    D = {}

    # ---- panel a: raw EEG, one representative retained recording -----------
    # EDIT: (n_times, 128) float, microvolts, band-passed as at model input.
    #       Take ~8 s from the middle of the recording, away from any edge.
    #   eeg = mne.io.read_raw_eeglab(path).resample(256).filter(5,50).get_data()[:128].T
    D['eeg'] = None                       # (n_times, 128)

    # ---- panel b: two recordings, true + reconstructed traces --------------
    # EDIT: 1-D arrays, same length, same time base as D['eeg'].
    #       Pick a MEDIAN-performing retained recording, not the best one, and
    #       say so in the caption. The rejected one is the honest contrast.
    D['good'] = dict(true=None, recon=None, peaks_true=None, peaks_pred=None,
                     label='retained  (F1 = 0.98)')
    D['bad'] = dict(true=None, recon=None, peaks_true=None, peaks_pred=None,
                    label='rejected  (F1 = 0.26)')

    # ---- panel c: per-beat signed latency errors, milliseconds -------------
    # EDIT: pooled over gate-retained recordings only. From match_peaks():
    #   errs_ms = np.concatenate([e for e in per_subject_errs]) / SFREQ * 1000
    D['errs_ms'] = None                   # (n_beats,)

    # ---- panel d: per-subject leakage, F1, gate verdict --------------------
    # Reads the CSV you already have. Add a 'gate_pass' column if it is absent.
    p = os.path.join(run_dir, 'validate_self_leakage.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)
        if 'gate_pass' not in df:
            df['gate_pass'] = df['f1'] >= 0.86      # EDIT: use the real verdict
        D['subjects'] = df
    else:
        D['subjects'] = None

    # ---- panel e: HBN carpet ----------------------------------------------
    # EDIT: (n_subjects, n_times) surrogate-referenced difference, averaged over
    #       the 67 cluster channels, in microvolts. times in seconds.
    D['carpet'] = None                    # (2078, n_times)
    D['carpet_times'] = None              # (n_times,)
    D['ga_true14'] = None                 # (n_times,) true-ECG GA, labelled cohort

    for k, v in D.items():
        if isinstance(v, np.ndarray):
            print(f'  {k}: {v.shape}')
    return D


# ============================================================
# 2. PANELS
# ============================================================
def panel_a_eeg(ax, eeg, sfreq=SFREQ, n_show=48, letter='a'):
    """Butterfly of the raw EEG. The point is that nothing cardiac is visible,
    so this panel is deliberately undramatic: thin grey lines, no annotation of
    beat times, a scale bar instead of axes."""
    t = np.arange(eeg.shape[0]) / sfreq
    step = max(1, eeg.shape[1] // n_show)
    for ch in range(0, eeg.shape[1], step):
        ax.plot(t, eeg[:, ch], color=fs.C['ink_faint'], lw=0.25, alpha=0.5,
                solid_joinstyle='round', rasterized=True)

    ax.set_xlim(t[0], t[-1])
    lo, hi = np.nanpercentile(eeg, [0.5, 99.5])
    pad = 0.25 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axis('off')

    fs.scalebar(ax, t[0] + 0.15, lo - 0.05 * (hi - lo), 1.0, 0.35 * (hi - lo),
                '1 s', f'{0.35*(hi-lo):.0f} µV')
    fs.stat_text(ax, '128-channel EEG\nno heartbeat discernible',
                 x=0.995, y=0.99, color=fs.C['ink_soft'])
    fs.panel_label(ax, letter, dx=-0.005, dy=1.0)
    return ax


def _trace_row(ax, d, sfreq, show_x, ylabel_pad=0.0):
    """One true/reconstructed trace pair with R-peak rug marks below."""
    t = np.arange(len(d['true'])) / sfreq
    ax.plot(t, d['true'], color=fs.C['true'], lw=0.7,
            ls=fs.LINESTYLE['true'], zorder=3)
    ax.plot(t, d['recon'], color=fs.C['recon'], lw=0.7,
            ls=fs.LINESTYLE['recon'], zorder=4)

    y0, y1 = np.nanmin(d['true']), np.nanmax(d['true'])
    span = y1 - y0
    base = y0 - 0.34 * span
    for pk, col, off in ((d['peaks_true'], fs.C['true'], 0.00),
                         (d['peaks_pred'], fs.C['recon'], 0.11)):
        if pk is None or not len(pk):
            continue
        pk = np.asarray(pk)
        pk = pk[(pk >= 0) & (pk < len(t))]
        yy = base - off * span
        ax.vlines(t[pk], yy, yy + 0.075 * span, color=col, lw=0.6)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(base - 0.20 * span, y1 + 0.18 * span)
    ax.set_yticks([])
    fs.despine(ax, keep=('bottom',) if show_x else ())
    if show_x:
        ax.set_xlabel('time (s)')
        fs.offset_spines(ax)
    else:
        ax.set_xticks([])
    ax.text(0.004, 1.02, d['label'], transform=ax.transAxes, ha='left',
            va='bottom', color=fs.C['ink_soft'], linespacing=1.3)
    return ax


def panel_b_traces(ax_good, ax_bad, good, bad, sfreq=SFREQ, letter='b'):
    _trace_row(ax_good, good, sfreq, show_x=False)
    _trace_row(ax_bad, bad, sfreq, show_x=True)
    handles = [
        Line2D([], [], color=fs.C['true'], lw=0.9, ls=fs.LINESTYLE['true'],
               label='recorded ECG'),
        Line2D([], [], color=fs.C['recon'], lw=0.9, ls=fs.LINESTYLE['recon'],
               label='reconstructed from EEG'),
    ]
    ax_good.legend(handles=handles, loc='upper right', ncol=2,
                   bbox_to_anchor=(1.0, 1.30))
    fs.panel_label(ax_good, letter, dx=-0.005, dy=1.30)
    return ax_good


def panel_c_timing(ax, errs_ms, sfreq=SFREQ, letter='c', bins=None):
    """Latency error histogram. One sample at 256 Hz is 3.9 ms, so marking it
    turns an abstract millisecond figure into 'sub-sample accuracy'.

    Pass `bins` when the errors are quantized to the sample grid, as they are
    whenever peaks come from an integer-index detector: free-running bin edges
    then fall between the achievable values and the histogram combs."""
    errs_ms = np.asarray(errs_ms, float)
    errs_ms = errs_ms[np.isfinite(errs_ms)]
    lim = float(np.nanpercentile(np.abs(errs_ms), 99.5))
    lim = max(lim, 1000.0 / sfreq * 2)
    if bins is None:
        bins = np.linspace(-lim, lim, 41)

    ax.hist(errs_ms, bins=bins, color=fs.C['recon'], alpha=0.85, lw=0)
    one_sample = 1000.0 / sfreq
    for s in (-1, 1):
        ax.axvline(s * one_sample, color=fs.C['threshold'], lw=0.5,
                   ls=(0, (2.5, 2.0)), zorder=1)
    fs.zeroline(ax, axis='x', color=fs.C['ink'], lw=0.5)

    ax.set_xlabel('latency error (ms)\nreconstructed − recorded')
    ax.set_ylabel('beats')
    ax.set_xlim(-lim, lim)
    fs.offset_spines(ax)
    fs.stat_text(ax, (f'|Δt| = {np.mean(np.abs(errs_ms)):.2f} ms\n'
                      f'bias = {np.mean(errs_ms):+.2f} ms\n'
                      f'n = {len(errs_ms):,} beats'))
    ax.text(-one_sample - 0.03 * lim, ax.get_ylim()[1] * 0.06, '±1 sample',
            ha='right', va='bottom', color=fs.C['threshold'])
    fs.panel_label(ax, letter)
    return ax


def panel_d_gate(ax, df, thresh_x=None, letter='d'):
    """Leakage vs F1, coloured by the frozen gate verdict.

    Colouring by verdict rather than by leakage is deliberate: it makes the
    panel about the decision the pipeline actually takes, which is the novel
    part, rather than about a correlation.
    """
    if 'gate_pass' not in df.columns:
        raise KeyError(
            "panel_d_gate needs a boolean 'gate_pass' column carrying the frozen "
            "gate verdict. If you only have F1, add it explicitly and say so in "
            "the caption:  df['gate_pass'] = df['f1'] >= 0.86")
    ok = df['gate_pass'].astype(bool).values
    x, y = df['leak_true'].values, df['f1'].values

    for m, col, lab in ((ok, fs.C['pass'], 'gate retains'),
                        (~ok, fs.C['fail'], 'gate rejects')):
        ax.scatter(x[m], y[m], s=16, facecolor=col, edgecolor='white',
                   linewidth=0.4, zorder=3, label=lab, clip_on=False)

    if thresh_x is None and ok.any() and (~ok).any():
        thresh_x = 0.5 * (x[~ok].max() + x[ok].min())
    if thresh_x is not None and thresh_x is not False:   # False = draw no boundary
        ax.axvline(thresh_x, color=fs.C['threshold'], lw=0.5, ls=(0, (2.5, 2.0)))

    ax.set_xscale('log')
    ax.set_xlabel('cardiac field leakage (a.u.)')
    ax.set_ylabel('R-peak detection $F_1$')
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    fs.offset_spines(ax)
    ax.legend(loc='lower right', bbox_to_anchor=(1.02, -0.02))
    fs.panel_label(ax, letter)
    return ax


def panel_e_carpet(ax_img, ax_line, M, times, ga_true=None, cax=None,
                   letter='e'):
    """Carpet plot of every subject, with the grand average beneath it.

    Rows are sorted by mean amplitude in the analysis window. Sorting makes the
    gradient visible, but it also manufactures apparent structure -- state the
    sort in the caption so it is not read as an ordering in the data.
    """
    M = np.asarray(M, float)
    order = np.argsort(np.nanmean(M, axis=1))
    norm = fs.diverging_norm(M)

    im = ax_img.imshow(M[order], aspect='auto', cmap=fs.CMAP_DIVERGING,
                       norm=norm, origin='upper', rasterized=True,
                       extent=[times[0] * 1e3, times[-1] * 1e3, M.shape[0], 0])
    ax_img.set_ylabel(f'subjects (n = {M.shape[0]:,})')
    ax_img.set_yticks([1, M.shape[0]])
    ax_img.set_xticks([])
    fs.despine(ax_img, keep=('left',))
    fs.panel_label(ax_img, letter)

    t_ms = times * 1e3
    ga = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0) / np.sqrt(M.shape[0])
    ax_line.fill_between(t_ms, ga - sem, ga + sem, color=fs.C['recon'],
                         alpha=0.22, lw=0)
    ax_line.plot(t_ms, ga, color=fs.C['recon'], lw=1.0,
                 ls=fs.LINESTYLE['true'], label=f'reconstructed peaks (n = {M.shape[0]:,})')
    if ga_true is not None:
        ax_line.plot(t_ms, ga_true, color=fs.C['true'], lw=0.9,
                     ls=fs.LINESTYLE['recon'], label='recorded ECG peaks (n = 14)')

    fs.zeroline(ax_line, axis='y')
    ax_line.set_xlim(t_ms[0], t_ms[-1])
    ax_line.set_xlabel('time from R-peak (ms)')
    ax_line.set_ylabel('µV')
    ax_line.legend(loc='lower left', ncol=2, bbox_to_anchor=(0.0, 1.0),
                   borderaxespad=0.2)
    fs.offset_spines(ax_line)

    if cax is not None:
        cb = ax_img.figure.colorbar(im, cax=cax)
        cb.set_label('µV', labelpad=1)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(width=0.5, length=2.0)
    return im


# ============================================================
# 3. ASSEMBLY
# ============================================================
def build(D, width='double', height_mm=195.0):
    fs.use(base_pt=7.0)
    fig = fs.figure(width=width, height_mm=height_mm)

    gs = GridSpec(
        6, 2, figure=fig,
        height_ratios=[1.05, 0.72, 0.72, 1.30, 1.55, 0.80],
        width_ratios=[1.0, 1.0],
        left=0.085, right=0.965, top=0.955, bottom=0.062,
        hspace=0.70, wspace=0.30,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b1 = fig.add_subplot(gs[1, :])
    ax_b2 = fig.add_subplot(gs[2, :])
    ax_c = fig.add_subplot(gs[3, 0])
    ax_d = fig.add_subplot(gs[3, 1])
    ax_e1 = fig.add_subplot(gs[4, :])
    ax_e2 = fig.add_subplot(gs[5, :], sharex=None)

    if D.get('eeg') is not None:
        panel_a_eeg(ax_a, D['eeg'])
    if D['good'].get('true') is not None:
        panel_b_traces(ax_b1, ax_b2, D['good'], D['bad'])
    if D.get('errs_ms') is not None:
        panel_c_timing(ax_c, D['errs_ms'])
    if D.get('subjects') is not None:
        panel_d_gate(ax_d, D['subjects'])
    if D.get('carpet') is not None:
        pos = ax_e1.get_position()
        cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.009, pos.height])
        panel_e_carpet(ax_e1, ax_e2, D['carpet'], D['carpet_times'],
                       D.get('ga_true14'), cax=cax)
    return fig


def main():
    D = load_inputs()
    fig = build(D)
    fs.check_text_sizes(fig)
    fs.save(fig, 'summary_figure', outdir=OUTDIR)
    plt.close(fig)


if __name__ == '__main__':
    main()