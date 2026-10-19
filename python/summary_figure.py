# -*- coding: utf-8 -*-
"""
summary_figure.py -- build the earlier five-panel summary figure (not in the paper) from
the pipeline outputs. fig_network_emergence.py reuses its loaders.

summary_figure_panels.py ships with a load_inputs() full of EDIT placeholders. This script is
that function, wired to the files the pipeline actually wrote:

    panel a  raw 128-channel EEG, one gate-retained recording   "the heartbeat is invisible"
    panel b  reconstructed vs recorded ECG, retained + rejected "...but the model reads it out"
    panel c  per-beat latency error, gate-retained recordings   "...to within one sample"
    panel d  leakage vs F1, coloured by the gate verdict        "...and it knows when it failed"
    panel e  HBN carpet                                          (optional; see --hep)

Panels a-c cannot be read off a CSV -- nothing in the run directory stores raw
traces or per-beat errors, only per-subject summaries. So this runs in two stages:

    python summary_figure.py --prepare      # loads the recordings, runs the deployed
                                         # model, caches the arrays.  GPU, ~minutes.
    python summary_figure.py                # builds figures/summary_figure.{pdf,png,svg}

Stage 2 needs no torch and no recordings, so you can re-style the figure freely
once the cache exists.

Panel e needs the Healthy Brain Network HEP arrays (the 2,078 x n_times
surrogate-referenced carpet). No code in this folder produces them, so it is off
by default; supply them with

    python summary_figure.py --hep path/to/hep_carpet.npz

where the .npz holds  carpet (n_subjects, n_times) in uV,  times (n_times,) in
seconds, and optionally  ga_true14 (n_times,) -- the true-ECG grand average of
the labelled cohort. Without --hep the figure is built as a-d.

Everything that had to be chosen rather than read is chosen by rule, not by eye:
the exemplar recordings in panels a and b are the MEDIAN-performing retained and
median-performing rejected recordings, so the figure cannot flatter the model by
showing its best case.

Requires fig_style.py, summary_figure_panels.py and hdeeg2ecg_model.py in the
same directory. Run from the scripts directory.

@author: figures for Fraile-Vazquez et al.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import fig_style as fs
import summary_figure_panels as fh

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')

# Same directory hdeeg2ecg_model.OUTPUT_DIR writes to. Kept as a
# plain constant so stage 2 does not have to import torch just to learn a path;
# --prepare checks it against the pipeline's own value and warns if they diverge.
RUN_DIR = os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'))

CACHE_NAME = 'summary_figure_inputs.npz'

SFREQ = 256
WINDOW = 512
DISPLAY_S = 8.0          # seconds of trace shown in panels a and b
SUBJ_RE = re.compile(r'8\d{3}')


def subject_id(name):
    return int(SUBJ_RE.search(os.path.basename(str(name))).group())


# ============================================================
# 1. THE PER-SUBJECT TABLE  (panel d, and the exemplar choice)
# ============================================================
def load_subject_table(run_dir):
    """One row per held-out recording: leakage, F1, and the frozen gate verdict.

    leak_true is the single-channel maximum of Section 'Impact of Baseline Cardiac
    Field Artifact Leakage' -- the paper's primary leakage measure -- taken from
    leakage_compare.csv. gate_pass is the reference-free verdict from
    qc_gate_decisions.csv, i.e. the real decision, not an F1 cut-off.
    """
    leak = pd.read_csv(os.path.join(run_dir, 'leakage_compare.csv'))
    gate = pd.read_csv(os.path.join(run_dir, 'qc_gate_decisions.csv'))

    leak['id'] = leak['file'].map(subject_id)
    df = leak.merge(gate[['id', 'gate_pass', 'frac_short', 'frac_implaus']],
                    on='id', validate='one_to_one')
    df = df.rename(columns={'leak_max': 'leak_true'})
    df['gate_pass'] = df['gate_pass'].astype(bool)

    n_pass = int(df['gate_pass'].sum())
    print(f'  subjects: {len(df)}  ({n_pass} retained, {len(df) - n_pass} rejected)')
    return df


def pick_exemplars(df):
    """Median-performing retained and median-performing rejected recording.

    Median rather than best: summary_figure_panels's own note, and the honest choice. Ties are
    broken by taking the lower F1, so the pick never drifts upward.
    """
    def median_row(sub):
        target = sub['f1'].median()
        return sub.loc[(sub['f1'] - target).abs().sort_values(kind='stable').index[0]]

    good = median_row(df[df['gate_pass']])
    bad = median_row(df[~df['gate_pass']])
    print(f"  panel-b exemplars: retained sub-{good['id']} (F1 = {good['f1']:.3f}), "
          f"rejected sub-{bad['id']} (F1 = {bad['f1']:.3f})")
    return good, bad


# ============================================================
# 2. PREPARE  -- run the deployed model, cache what the panels need
# ============================================================
def _display_eeg_uv(path, start, n, P):
    """The same recording panel b shows, in microvolts rather than z-scores.

    P.load_subject_data z-scores every channel because that is what the model
    eats; a butterfly plot labelled 'uV' must not be drawn from that. So the
    display copy is re-read with identical resampling and filtering and left in
    physical units.
    """
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
    raw.resample(SFREQ, verbose='ERROR')
    raw.filter(l_freq=P.FILTER_LFREQ, h_freq=P.FILTER_HFREQ, method='iir',
               verbose='ERROR')
    return raw.get_data()[:128, start:start + n].T * 1e6


def _display_window(n_samples):
    """Start of the shown segment: window-aligned, near the middle of the recording.

    Window-aligned so the segment begins at a real inference boundary rather than
    inside one, and mid-recording so it is away from both edges.
    """
    n_show = int(DISPLAY_S * SFREQ)
    start = ((n_samples // 2) // WINDOW) * WINDOW
    start = min(start, max(0, n_samples - n_show))
    return start, n_show


def prepare(run_dir, data_folder=None, cache_path=None):
    import torch
    import hdeeg2ecg_model as P

    if os.path.abspath(run_dir) != os.path.abspath(P.OUTPUT_DIR):
        print(f'  NOTE: --run-dir {run_dir} differs from the pipeline OUTPUT_DIR '
              f'{P.OUTPUT_DIR}; reading results from the former.')
    data_folder = data_folder or P.DATA_FOLDER
    cache_path = cache_path or os.path.join(run_dir, CACHE_NAME)

    df = load_subject_table(run_dir)
    good, bad = pick_exemplars(df)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = P.EEGtoECGModel(latent_dim=P.SELECTED_LATENT_DIM,
                          decoder_schedule=P.SELECTED_DECODER,
                          variant=P.SELECTED_VARIANT)
    net.load_state_dict(torch.load(P.WEIGHTS_PATH,
                                   map_location='cpu'))
    net.eval().to(device)
    print(f'  loaded {P.SELECTED_VARIANT} model on {device}')

    tol = int(P.QRS_MATCH_TOLERANCE_MS / 1000 * SFREQ)

    # Panel c pools beats over the recordings the gate retains -- the population the
    # deployed pipeline actually analyses. The rejected exemplar is reconstructed too,
    # for panel b, but its beats are not pooled.
    wanted = df[df['gate_pass'] | (df['id'] == bad['id'])]
    out, errs_pool, per_subject = {}, [], []

    for _, row in wanted.iterrows():
        fn = str(row['file'])
        eeg, evt, _ = P.load_subject_data(os.path.join(data_folder, fn))
        pred, true, _ = P.reconstruct_subject(net, eeg, device)
        n = len(pred)

        true_pk = np.where(evt[:n] == 1)[0]
        pred_pk = P.detect_peaks(pred)
        tp, fp, fn_, errs = P.match_peaks(true_pk, pred_pk, tol)
        errs_ms = errs / SFREQ * 1000.0

        if bool(row['gate_pass']):
            errs_pool.append(errs_ms)
            per_subject.append((row['id'], len(errs_ms),
                                np.mean(np.abs(errs_ms)), np.mean(errs_ms)))
        print(f"  sub-{row['id']}  F1 = {row['f1']:.3f}  "
              f"{len(errs_ms)} matched beats  {'retained' if row['gate_pass'] else 'rejected'}")

        for tag, ref in (('good', good), ('bad', bad)):
            if row['id'] != ref['id']:
                continue
            s, w = _display_window(n)
            sl = slice(s, s + w)
            out[f'{tag}_true'] = true[sl]
            out[f'{tag}_recon'] = pred[sl]
            out[f'{tag}_pk_true'] = true_pk[(true_pk >= s) & (true_pk < s + w)] - s
            out[f'{tag}_pk_pred'] = pred_pk[(pred_pk >= s) & (pred_pk < s + w)] - s
            out[f'{tag}_id'] = int(row['id'])
            out[f'{tag}_f1'] = float(row['f1'])
            if tag == 'good':
                out['eeg_uv'] = _display_eeg_uv(os.path.join(data_folder, fn), s, w, P)

    out['errs_ms'] = np.concatenate(errs_pool)
    out['n_retained'] = len(errs_pool)
    # (id, n_beats, mean |dt|, signed bias) -- so the caption can quote the
    # subject-level figures alongside the beat-pooled ones panel c shows.
    out['per_subject'] = np.array(per_subject, dtype=float)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **out)
    print(f'\n  cached {len(out["errs_ms"]):,} beats from {out["n_retained"]} '
          f'retained recordings -> {cache_path}')
    return cache_path


# ============================================================
# 3. PANELS THAT NEED MORE THAN summary_figure_panels'S DEFAULTS
# ============================================================
def panel_c(ax, errs_ms, sfreq=SFREQ, letter='c'):
    """Latency errors, binned on the sample grid.

    Peaks are integer sample indices at both ends of the comparison, so every
    achievable error is a multiple of 1000/sfreq ms. Free-running bin edges put
    some bins between achievable values and none in others, which renders as a
    comb and reads as noise rather than as quantization. One bin per sample
    offset shows the distribution as it actually is.
    """
    one = 1000.0 / sfreq
    e = np.asarray(errs_ms, float)
    e = e[np.isfinite(e)]
    lim = max(float(np.nanpercentile(np.abs(e), 99.5)), 2 * one)
    k = int(np.ceil(lim / one))
    bins = (np.arange(-k, k + 2) - 0.5) * one

    fh.panel_c_timing(ax, e, sfreq=sfreq, letter=letter, bins=bins)
    ax.set_xlim(bins[0], bins[-1])
    return ax


def panel_d(ax, df, draw_boundary=False, letter='d'):
    """Leakage vs F1 coloured by the gate verdict.

    summary_figure_panels.panel_d_gate defaults to drawing a vertical line at the midpoint of the
    leakage gap and styles it as the frozen decision boundary. Here that would be
    wrong: the gate is calibrated on frac_short and frac_implaus, not on leakage,
    and the paper is explicit that no absolute leakage value transfers as a
    threshold. So the line is suppressed by default and the separation is reported
    as what it is -- an observed, non-overlapping gap -- alongside the rank
    correlation the panel is really about.
    """
    fh.panel_d_gate(ax, df, thresh_x=None if draw_boundary else False, letter=letter)

    rho, p = sps.spearmanr(df['leak_true'], df['f1'])
    hi_fail = df.loc[~df['gate_pass'], 'leak_true'].max()
    lo_pass = df.loc[df['gate_pass'], 'leak_true'].min()
    p_txt = 'p < 0.001' if p < 1e-3 else f'p = {p:.3f}'
    # Top-left: the quadrant the finding itself leaves empty (no recording has
    # negligible leakage and a good reconstruction), so nothing is occluded.
    fs.stat_text(ax, (f'$\\rho$ = {rho:.2f}, {p_txt}\n'
                      f'n = {len(df)} recordings\n'
                      f'no overlap: {hi_fail:.2f} | {lo_pass:.2f}'),
                 x=0.02, y=0.99, ha='left')
    return ax


# ============================================================
# 4. ASSEMBLY
# ============================================================
def build_abcd(D, df, width='double', height_mm=132.0):
    """Panels a-d. Same proportions as summary_figure_panels.build with the carpet rows removed."""
    fs.use(base_pt=7.0)
    fig = fs.figure(width=width, height_mm=height_mm)

    gs = GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.05, 0.72, 0.72, 1.45],
        left=0.085, right=0.965, top=0.945, bottom=0.105,
        hspace=0.75, wspace=0.32,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b1 = fig.add_subplot(gs[1, :])
    ax_b2 = fig.add_subplot(gs[2, :])
    ax_c = fig.add_subplot(gs[3, 0])
    ax_d = fig.add_subplot(gs[3, 1])

    fh.panel_a_eeg(ax_a, D['eeg'])
    fh.panel_b_traces(ax_b1, ax_b2, D['good'], D['bad'])
    panel_c(ax_c, D['errs_ms'])
    panel_d(ax_d, df)
    return fig


def build_full(D, df, width='double', height_mm=195.0):
    """All five panels. Mirrors summary_figure_panels.build, but with the corrected panel d."""
    fs.use(base_pt=7.0)
    fig = fs.figure(width=width, height_mm=height_mm)

    gs = GridSpec(
        6, 2, figure=fig,
        height_ratios=[1.05, 0.72, 0.72, 1.30, 1.55, 0.80],
        width_ratios=[1.0, 1.0],
        # right margin pulled in from summary_figure_panels.build's 0.965: the panel-e colorbar
        # is hung outside the grid and its tick labels need the room.
        left=0.085, right=0.935, top=0.955, bottom=0.062,
        hspace=0.70, wspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b1 = fig.add_subplot(gs[1, :])
    ax_b2 = fig.add_subplot(gs[2, :])
    ax_c = fig.add_subplot(gs[3, 0])
    ax_d = fig.add_subplot(gs[3, 1])
    ax_e1 = fig.add_subplot(gs[4, :])
    ax_e2 = fig.add_subplot(gs[5, :])

    fh.panel_a_eeg(ax_a, D['eeg'])
    fh.panel_b_traces(ax_b1, ax_b2, D['good'], D['bad'])
    panel_c(ax_c, D['errs_ms'])
    panel_d(ax_d, df)

    pos = ax_e1.get_position()
    cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.009, pos.height])
    fh.panel_e_carpet(ax_e1, ax_e2, D['carpet'], D['carpet_times'],
                      D.get('ga_true14'), cax=cax)
    return fig


def assemble_inputs(cache_path, df, hep_path=None):
    """The dict summary_figure_panels's panels expect, built from the cache and the CSVs."""
    z = np.load(cache_path, allow_pickle=False)

    def _z(v):
        """z-score over the shown segment.

        The model's output is systematically lower in amplitude than the z-scored
        target -- that shortfall is what the normalized RMSE near 0.9 in
        per_subject_metrics_*.csv measures. Plotted on a shared axis the
        reconstruction would sit almost flat and read as a failure, when the claim
        this panel makes is about the timing of the deflections, not their height.
        Both traces are therefore scaled independently, and the caption says so.
        """
        v = np.asarray(v, float)
        return (v - v.mean()) / (v.std() + 1e-12)

    def side(tag, verdict):
        return dict(true=_z(z[f'{tag}_true']), recon=_z(z[f'{tag}_recon']),
                    peaks_true=z[f'{tag}_pk_true'], peaks_pred=z[f'{tag}_pk_pred'],
                    label=f'{verdict}   sub-{int(z[f"{tag}_id"])}   '
                          f'$F_1$ = {float(z[f"{tag}_f1"]):.2f}')

    D = {
        'eeg': z['eeg_uv'],
        'good': side('good', 'gate retains'),
        'bad': side('bad', 'gate rejects'),
        'errs_ms': z['errs_ms'],
        'subjects': df,
        'carpet': None, 'carpet_times': None, 'ga_true14': None,
    }

    if hep_path:
        h = np.load(hep_path, allow_pickle=False)
        D['carpet'] = h['carpet']
        D['carpet_times'] = h['times']
        D['ga_true14'] = h['ga_true14'] if 'ga_true14' in h.files else None
        print(f"  panel e: carpet {D['carpet'].shape}, "
              f"{D['carpet_times'][0]*1e3:.0f} to {D['carpet_times'][-1]*1e3:.0f} ms")

    meta = {'n_retained': int(z['n_retained']),
            'n_beats': int(len(z['errs_ms'])),
            'per_subject': z['per_subject'],
            'good_id': int(z['good_id']), 'good_f1': float(z['good_f1']),
            'bad_id': int(z['bad_id']), 'bad_f1': float(z['bad_f1'])}
    return D, meta


# ============================================================
# 5. CAPTION
# ============================================================
def write_caption(path, D, df, meta):
    e = np.asarray(D['errs_ms'], float)
    ps = np.asarray(meta['per_subject'], float)   # id, n_beats, mean |dt|, bias
    rho, p = sps.spearmanr(df['leak_true'], df['f1'])
    n_pass = int(df['gate_pass'].sum())
    hi_fail = df.loc[~df['gate_pass'], 'leak_true'].max()
    lo_pass = df.loc[df['gate_pass'], 'leak_true'].min()
    p_txt = 'p < 0.001' if p < 1e-3 else f'p = {p:.3f}'
    has_e = D.get('carpet') is not None

    txt = f"""Figure 1 | R-peak timing recovered from EEG alone.

(a) Raw 128-channel EEG from a gate-retained recording (sub-{meta['good_id']}),
{DISPLAY_S:.0f} s taken from the middle of the recording, resampled to {SFREQ} Hz and
band-pass filtered 5-50 Hz exactly as at model input, shown in physical units. No
heartbeat is discernible by eye.

(b) The reconstructed ECG (red, dashed) against the simultaneously recorded ECG
(black, solid) over the same segment, for the median-performing recording the
quality gate retains (sub-{meta['good_id']}, F1 = {meta['good_f1']:.2f}; top) and the
median-performing recording it rejects (sub-{meta['bad_id']}, F1 = {meta['bad_f1']:.2f};
bottom). Exemplars are the medians of their groups, not the best cases. Each
trace is z-scored over the segment shown and the vertical axis therefore carries
no units: the reconstruction is systematically lower in amplitude than the
recorded trace, which the panel does not attempt to display and which the
normalized RMSE reports separately. Rug marks below each trace give recorded
(black) and detected (red) R-peaks; the rejected recording over-detects, the
failure mode the gate is built to catch.

(c) Signed per-beat latency error, reconstructed minus recorded, pooled over the
{meta['n_retained']} held-out recordings the gate retains ({meta['n_beats']:,} matched beats,
+/-50 ms matching tolerance). One bin per sample offset, since both peak trains
are integer sample indices. Mean absolute error {np.mean(np.abs(e)):.2f} ms and bias
{np.mean(e):+.2f} ms across beats; across recordings, {ps[:, 2].mean():.2f} ms and
{ps[:, 3].mean():+.2f} ms (n = {meta['n_retained']}). One sample at {SFREQ} Hz = {1000/SFREQ:.1f} ms (dashed).

(d) R-peak detection F1 against cardiac field artifact leakage for every held-out
recording (n = {len(df)}; selection and testing splits pooled), coloured by the frozen
reference-free gate verdict ({n_pass} retained, {len(df) - n_pass} rejected). Leakage is the
single-channel maximum peak-to-peak of the R-peak-triggered average, computed on
z-scored EEG without permutation correction, and is therefore dimensionless.
Spearman rho = {rho:.2f}, {p_txt}. Failed and successful reconstructions do not overlap
on this axis (highest failure {hi_fail:.2f}, lowest success {lo_pass:.2f}), but the gate is
calibrated on the reconstructed R-R interval fractions, not on leakage, so no
boundary is drawn here.
"""
    if has_e:
        n_sub = D['carpet'].shape[0]
        txt += f"""
(e) Surrogate-referenced heartbeat-locked response for every Healthy Brain Network
recording retained by the pipeline (n = {n_sub:,}), averaged over the 67 cluster
channels defined in the labelled adult cohort. Rows are sorted by mean amplitude
in the analysis window; the sort is imposed for display and is not an ordering in
the data. Beneath, the grand average (+/-1 s.e.m.) with the true-ECG grand average
of the labelled cohort (n = 14) for comparison.
"""
    else:
        txt += """
Panel e (Healthy Brain Network carpet) is not included in this build: supply the
HEP arrays with --hep to add it.
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f'  wrote caption -> {path}')


# ============================================================
# 6. CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepare', action='store_true',
                    help='run the deployed model and cache the panel a-c arrays')
    ap.add_argument('--run-dir', default=RUN_DIR, help='pipeline output directory')
    ap.add_argument('--data-folder', default=None,
                    help='folder holding the .set recordings (default: the '
                         'pipeline DATA_FOLDER)')
    ap.add_argument('--cache', default=None, help=f'override the {CACHE_NAME} path')
    ap.add_argument('--hep', default=None,
                    help='npz with carpet, times [, ga_true14] to draw panel e')
    ap.add_argument('--outdir', default=FIG_DIR)
    ap.add_argument('--stem', default='summary_figure')
    ap.add_argument('--height-mm', type=float, default=None)
    args = ap.parse_args()

    cache = args.cache or os.path.join(args.run_dir, CACHE_NAME)

    if args.prepare:
        print('\n=== preparing panel a-c inputs ===')
        prepare(args.run_dir, args.data_folder, cache)
        print('\nnow run:  python summary_figure.py')
        return

    print('\n=== building the figure ===')
    if not os.path.exists(cache):
        raise SystemExit(
            f'no cache at {cache}.\n'
            'Panels a-c need raw traces and per-beat errors, which the run '
            'directory does not store.\nRun:  python summary_figure.py --prepare')

    df = load_subject_table(args.run_dir)
    D, meta = assemble_inputs(cache, df, args.hep)

    if D['carpet'] is not None:
        fig = build_full(D, df, height_mm=args.height_mm or 195.0)
    else:
        print('  panel e omitted (no --hep); building a-d')
        fig = build_abcd(D, df, height_mm=args.height_mm or 132.0)

    fs.check_text_sizes(fig)
    fs.save(fig, args.stem, outdir=args.outdir)
    write_caption(os.path.join(args.outdir, f'{args.stem}_caption.txt'), D, df, meta)
    plt.close(fig)


if __name__ == '__main__':
    main()
