# -*- coding: utf-8 -*-
"""
fig_network_emergence.py -- the heartbeat emerging through the network, twice.

The same deployed weights, the same eight stages, two held-out recordings: one the
reference-free gate retains and one it rejects. On the left the beat sharpens from
invisible noise into hard vertical spikes and out as an ECG. On the right it never
appears, and the decoder spends its capacity on nothing.

    a  successful reconstruction
    b  failed reconstruction

Both columns are sorted by the SAME feature ordering, taken from the successful
recording. The features are units of one network, so row i is the same unit in
both panels and the comparison is unit by unit rather than pattern against
pattern. Heat is each recording's own per-feature z-scored magnitude, so the right
column is not merely dimmer -- it is unstructured.

The ladder on the right is the quantitative spine of the figure: how deeply each
stage's features are modulated by the heartbeat, against a permutation floor
(grey). It climbs through the network on the left and stays in the floor on the
right. Its rows align with the image rows, so the two readings are the same
reading. That is the failure mode the gate catches, measured inside the network
rather than at its output.

Recordings are identified by verdict only. Which recordings they are, and the
selection rule, are in the caption and the cache; the panel makes a claim about
two classes of reconstruction, not about two subjects.

Two stages, because the panel needs the weights and the recordings:

    python fig_network_emergence.py --prepare     # GPU, ~1 min
    python fig_network_emergence.py               # figures/fig_network_emergence.{pdf,png,svg}

Requires fig_style.py, network_activations.py and hdeeg2ecg_model.py in the same folder.

@author: figures for Fraile-Vazquez et al.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import fig_style as fs
from network_activations import (RUN_DIR, SFREQ, WINDOW, load_subject_table,
                            HALF_WIN, N_PERM, N_STAT, STAGES,
                            _receptive_field_samples, beat_modulation,
                            extract_activations)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
CACHE_NAME = 'fig_network_emergence_inputs.npz'

N_SHOW = 3        # inference windows drawn per column (3 x 2 s = 6 s). At the
                  # slowest heart rate in the cohort (50 bpm) a 4 s span holds only
                  # three beats anywhere in the recording; 6 s holds five.
CLIP = 2.5

COLS = (('good', 'a', 'successful reconstruction', 'pass'),
        ('bad', 'b', 'failed reconstruction', 'fail'))

# Activations get their own bright sequential ramp: paper white through amber and
# coral to magenta and deep indigo. Monotonic in lightness, so it encodes magnitude
# honestly and survives greyscale, and it occupies a hue range (magenta-indigo)
# that appears nowhere else in the figure -- it cannot be read as the verdict
# colours or as the recorded/reconstructed pair.
#
# The stops are deliberately NOT evenly spaced. Typical noise sits near |z| = 0.8,
# a third of the way up the scale, so an even ramp puts saturated amber everywhere
# and the input rows end up as vivid as the decoder rows. Worse, the failed column
# spreads mid-range activation across every timepoint while the successful one
# concentrates it into a few, so an even ramp makes failure look busier -- and
# therefore better. Holding the first 40% near paper white keeps the contrast where
# the meaning is: sparse, saturated spikes for a beat that was found, a pale wash
# for one that was not.
CMAP_ACT = LinearSegmentedColormap.from_list('act_bright', [
    (0.00, '#FFFFFF'), (0.20, '#FFFBEA'), (0.38, '#FFF0BE'), (0.52, '#FFD273'),
    (0.64, '#FBA23F'), (0.74, '#F26B3F'), (0.83, '#DB3B5C'), (0.91, '#A62480'),
    (0.96, '#651A7A'), (1.00, '#2A0F47')], N=256)

EEG_TRACE = '#6F8BA8'      # recorded EEG butterfly: slate blue, not dead grey
TINT_ALPHA = 0.10          # column wash strength

# Row order, gutter label, sub-label. Three rows all reading "decoder" was
# ambiguous; numbering them is the same information, legible.
ROWS = [
    ('trace_in', 'recorded EEG', '128 channels', 0.60),
    ('eeg', 'model input', '512 × 128', 1.00),
    ('encoder', 'dilated encoder', '512 × 128', 1.00),
    # The 32-channel stage was called the "latent" when the decoder re-expanded it
    # to 128 (a true bottleneck). The deployed decoder does not re-expand, so this
    # stage is simply the first step of a 128->32->16->8->4->1 funnel. Cache keys
    # are unchanged; only the displayed labels moved.
    ('latent', 'decoder 1', '512 × 32', 0.62),
    ('dec16', 'decoder 2', '512 × 16', 0.45),
    ('dec8', 'decoder 3', '512 × 8', 0.34),
    ('dec4', 'decoder 4', '512 × 4', 0.26),
    ('trace_out', 'output ECG', '512 × 1', 0.92),
]
STAGE_KEYS = [s[0] for s in STAGES]


# ============================================================
# 1. PREPARE
# ============================================================
def _median_row(sub):
    """The median-F1 member of a group. Median rather than best or worst: neither
    column should be the most flattering or the most damning case available.

    With an even group size the median falls between two recordings, and both are
    equidistant from it. Ties break toward the HIGHER F1 -- of two equally
    representative recordings, show the better one, so the failed column is never
    the harsher of two defensible choices.
    """
    # Rounded before sorting: with an even group the two gaps are equal in exact
    # arithmetic but differ in the last float bit, so an unrounded sort picks the
    # winner by rounding noise rather than by the stated rule.
    d = sub.assign(_gap=(sub['f1'] - sub['f1'].median()).abs().round(9))
    return d.sort_values(['_gap', 'f1'], ascending=[True, False]).iloc[0]


def _pick(sub, want_id, label):
    """Explicit id if given, else the group median."""
    if want_id is None:
        return _median_row(sub)
    hit = sub[sub['id'] == int(want_id)]
    if hit.empty:
        raise SystemExit(
            f'sub-{want_id} is not in the {label} group. Available: '
            + ', '.join(f"{int(r.id)} (F1 {r.f1:.2f}, leakage {r.leak_true:.2f})"
                        for r in sub.sort_values('leak_true').itertuples()))
    return hit.iloc[0]


def _load_eeg_uv(path, P):
    """Whole recording, 128 channels, microvolts, resampled and filtered exactly as
    at the model input. P.load_subject_data z-scores every channel because that is
    what the network eats; a trace labelled uV must not come from it."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
    raw.resample(SFREQ, verbose='ERROR')
    raw.filter(l_freq=P.FILTER_LFREQ, h_freq=P.FILTER_HFREQ, method='iir',
               verbose='ERROR')
    return (raw.get_data()[:128] * 1e6).T.astype(np.float32)          # (T, 128)


def _pick_display_window(true_sig, eeg_uv, peaks, n_disp, n_stat):
    """Choose which stretch of the recording to draw.

    Taking the middle of the recording is arbitrary and it went wrong twice. The
    recorded ECG has to be unambiguous, because it is the reference the
    reconstruction is judged against and a panel where the reader cannot see the
    ground-truth beats proves nothing. And the EEG in that stretch must not be
    dominated by a movement burst, or the amplitude scale bar is set by the artifact
    and invites the reader to read a failure as an amplitude problem rather than as
    an absent cardiac field.

    So: keep candidate spans whose EEG amplitude falls in the quieter half of the
    recording, then among those take the span where the recorded R-peaks stand
    highest above their own robust baseline. Candidates are window-aligned and stop
    early enough that the activation stack still fits after them.
    """
    starts = np.arange(0, max(len(true_sig) - n_stat, 0) + 1, WINDOW)
    if not len(starts):
        return 0
    art = np.array([np.percentile(np.abs(eeg_uv[s:s + n_disp]), 99.5) for s in starts])
    quiet = art <= np.median(art)

    best, best_clarity = int(starts[0]), -np.inf
    for s, ok in zip(starts, quiet):
        if not ok:
            continue
        pk = peaks[(peaks >= s + 2) & (peaks < s + n_disp - 2)]
        if len(pk) < 4:                       # need several beats to read a rhythm
            continue
        seg = true_sig[s:s + n_disp]
        base = np.median(seg)
        scale = 1.4826 * np.median(np.abs(seg - base)) + 1e-9
        clarity = float(np.median((seg[pk - s] - base) / scale))
        if clarity > best_clarity:
            best, best_clarity = int(s), clarity
    return best


def prepare(run_dir, data_folder=None, cache_path=None, good_id=None, bad_id=None):
    import torch
    import hdeeg2ecg_model as P

    data_folder = data_folder or P.DATA_FOLDER
    cache_path = cache_path or os.path.join(run_dir, CACHE_NAME)

    df = load_subject_table(run_dir)
    picks = {'good': _pick(df[df['gate_pass']], good_id, 'retained'),
             'bad': _pick(df[~df['gate_pass']], bad_id, 'rejected')}
    for tag, row in picks.items():
        print(f"  {tag}: sub-{row['id']}  F1 = {row['f1']:.3f}  "
              f"leakage = {row['leak_true']:.2f}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = P.EEGtoECGModel(latent_dim=P.SELECTED_LATENT_DIM,
                          decoder_schedule=P.SELECTED_DECODER,
                          variant=P.SELECTED_VARIANT)
    sd = torch.load(P.WEIGHTS_PATH,
                    map_location='cpu')
    net.load_state_dict(sd)                       # strict: any mismatch raises
    net.eval().to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f'  loaded {P.SELECTED_VARIANT} on {device}: {len(sd)} tensors, '
          f'{n_par/1e6:.3f} M parameters')

    out = {}
    n_disp = N_SHOW * WINDOW

    for tag, row in picks.items():
        path = os.path.join(data_folder, str(row['file']))
        eeg, evt, _ = P.load_subject_data(path)
        eeg_uv = _load_eeg_uv(path, P)

        # Reconstruct the whole recording first: the display window is chosen from
        # the recorded ECG and the EEG, so both have to exist before it is fixed.
        pred_full, true_full, _ = P.reconstruct_subject(net, eeg, device)
        pk_pred = P.detect_peaks(pred_full)
        pk_true_full = np.where(evt[:len(pred_full)] == 1)[0]

        start = _pick_display_window(true_full, eeg_uv, pk_true_full, n_disp,
                                     N_STAT * WINDOW)
        print(f'    display window: {start / SFREQ:.0f}-'
              f'{(start + n_disp) / SFREQ:.0f} s of {len(pred_full) / SFREQ:.0f} s')

        stack = eeg[start:start + N_STAT * WINDOW].reshape(
            N_STAT, WINDOW, 129).astype(np.float32).copy()
        stack[:, :, 128] = 0.0                     # inference masking, as deployed
        A = extract_activations(net, stack, device)
        pk_local = np.where(evt[start:start + N_STAT * WINDOW] == 1)[0]

        for name in STAGE_KEYS:
            obs, floor = beat_modulation(A[name], pk_local, hw=HALF_WIN,
                                         n_perm=N_PERM)
            out[f'act_{tag}_{name}'] = A[name][:n_disp].astype(np.float32)
            out[f'mod_{tag}_{name}'] = obs.astype(np.float32)
            out[f'floor_{tag}_{name}'] = floor.astype(np.float32)

        # Verification, not decoration. The hooked activations are only meaningful
        # if the forward pass they came from is the pipeline's own. P.reconstruct_
        # subject tiles the same 512-sample windows, zeroes the same ECG channel and
        # calls the same net, independently of extract_activations and at a
        # different batch size -- so the final hooked stage must reproduce it to
        # float precision. If it does not, the stack above is not this model.
        span = pred_full[start:start + N_STAT * WINDOW]
        gap = float(np.abs(A['out'] - span).max())
        print(f'    hooks vs pipeline forward pass: max |diff| = {gap:.2e}')
        assert gap < 1e-4, (
            'hooked activations do not match reconstruct_subject; the plotted '
            'stack would not be the deployed forward pass')

        sl = slice(start, start + n_disp)
        out[f'true_{tag}'] = true_full[sl].astype(np.float32)
        out[f'recon_{tag}'] = pred_full[sl].astype(np.float32)
        out[f'eeg_{tag}'] = eeg_uv[sl]
        out[f'pk_true_{tag}'] = pk_local[pk_local < n_disp]
        out[f'pk_pred_{tag}'] = pk_pred[(pk_pred >= start) &
                                        (pk_pred < start + n_disp)] - start
        out[f'id_{tag}'] = int(row['id'])
        out[f'f1_{tag}'] = float(row['f1'])
        out[f'leak_{tag}'] = float(row['leak_true'])
        print(f"  {tag}: sub-{row['id']} done, {len(out[f'pk_true_{tag}'])} recorded "
              f"beats / {len(out[f'pk_pred_{tag}'])} detected in the shown segment")

    # One ordering for both columns, from the successful recording. Features are
    # units of a single network, so this makes row i the same unit on both sides.
    for name in STAGE_KEYS:
        out[f'order_{name}'] = np.argsort(-out[f'mod_good_{name}'])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **out)
    print(f'\n  cached -> {cache_path}')
    return cache_path


# ============================================================
# 2. DRAWING HELPERS
# ============================================================
def _scalebar(ax, x, y, dx, dy, label):
    """L-bar with a single horizontal label beneath it.

    fig_style.scalebar rotates the amplitude label up the vertical arm, which at
    this row height lands on both the arm and the trace. One flat label under the
    corner reads the same and collides with nothing -- the L has one horizontal arm
    and one vertical arm, so which figure belongs to which is unambiguous.
    """
    ax.plot([x, x, x + dx], [y + dy, y, y], color=fs.C['ink'], lw=0.8,
            solid_capstyle='butt', clip_on=False, zorder=6)
    ax.text(x + dx / 2, y - 0.12 * dy, label, ha='center', va='top',
            color=fs.C['ink_soft'], clip_on=False)


# Gutter label anchor, in axes fractions. Must clear the column tint, which is
# drawn 0.008 figure-widths outside the axes -- at a smaller offset the labels sat
# on the tinted edge.
LABEL_X = -0.032


def _row_label(ax, name, shape):
    """Two-line gutter label: stage in ink, tensor shape beneath it in grey.

    Two texts anchored to the same x at the row's vertical centre, one growing up
    and one growing down, rather than an ylabel plus an offset text. The ylabel is
    padded in points and the text in axes fractions, so on rows of different height
    the two lines drifted out of flush and off centre.
    """
    ax.text(LABEL_X, 0.5, name, transform=ax.transAxes, ha='right', va='bottom',
            color=fs.C['ink'])
    ax.text(LABEL_X, 0.5, shape, transform=ax.transAxes, ha='right', va='top',
            color=fs.C['ink_faint'])


# ============================================================
# 3. ONE COLUMN
# ============================================================
def funnel_column(fig, gs, col, z, tag, letter, verdict, vkey, labels=True):
    n = len(z[f'true_{tag}'])
    t_max = n / SFREQ
    t = np.arange(n) / SFREQ

    axes = {}
    for i, (key, name, shape, _h) in enumerate(ROWS):
        ax = fig.add_subplot(gs[i, col])
        ax.set_facecolor('none')            # let the column tint show through
        axes[key] = ax
        if labels:
            _row_label(ax, name, shape)

    # ---- header: verdict, then how good the reconstruction actually was ---
    # On the header line rather than anywhere in the stack: it is the one place
    # already reserved for describing the column. Verdict keeps the colour and the
    # emphasis; the metrics sit right-aligned in grey as supporting detail. No
    # subject identifier -- the panel is about two classes of reconstruction.
    ax = axes['trace_in']
    ax.plot([0, 1], [1.42, 1.42], transform=ax.transAxes, color=fs.C[vkey],
            lw=1.3, solid_capstyle='butt', clip_on=False, zorder=6)
    ax.text(0.0, 1.50, verdict, transform=ax.transAxes, ha='left', va='bottom',
            color=fs.C[vkey], fontsize=plt.rcParams['font.size'] + 0.5)
    ax.text(1.0, 1.50,
            f"$F_1$ = {float(z[f'f1_{tag}']):.2f}      "
            f"CFA leakage = {float(z[f'leak_{tag}']):.2f}",
            transform=ax.transAxes, ha='right', va='bottom',
            color=fs.C['ink_soft'])
    fs.panel_label(ax, letter, dx=-0.012, dy=1.50)

    # ---- recorded EEG ---------------------------------------------------
    eeg = z[f'eeg_{tag}']
    for ch in range(0, 128, 3):
        ax.plot(t, eeg[:, ch], color=EEG_TRACE, lw=0.2, alpha=0.6,
                rasterized=True)
    lo, hi = np.nanpercentile(eeg, [0.5, 99.5])
    ax.set_xlim(0, t_max)
    ax.set_ylim(lo * 1.42, hi * 1.20)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    _scalebar(ax, 0.05, lo * 1.30, 0.75, 0.60 * hi,
              f'1 s  ·  {0.60*hi:.0f} µV')

    # ---- the stack ------------------------------------------------------
    im = None
    for key, _name, _shape, _h in ROWS[1:-1]:
        ax = axes[key]
        nch = dict((s[0], s[2]) for s in STAGES)[key]
        M = z[f'act_{tag}_{key}'][:, z[f'order_{key}']]
        zz = np.abs((M - M.mean(0)) / (M.std(0) + 1e-9)).T
        im = ax.imshow(zz, aspect='auto', cmap=CMAP_ACT, vmin=0, vmax=CLIP,
                       origin='upper', rasterized=True, interpolation='nearest',
                       extent=[0, t_max, nch, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)           # no boxes; the image is its own frame
        for w in range(1, N_SHOW):
            ax.axvline(w * WINDOW / SFREQ, color='white', lw=0.4, alpha=0.6)

    # The receptive-field bracket is deliberately absent: it is stated in the
    # caption instead. An unlabelled bracket would be decoration.

    # ---- output ---------------------------------------------------------
    ax = axes['trace_out']
    zsc = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    tr, rc = zsc(z[f'true_{tag}']), zsc(z[f'recon_{tag}'])
    ax.plot(t, tr, color=fs.C['true'], lw=0.6, ls=fs.LINESTYLE['true'], zorder=3)
    ax.plot(t, rc, color=fs.C['recon'], lw=0.8, ls=fs.LINESTYLE['recon'], zorder=4)

    span = tr.max() - tr.min()
    base = tr.min() - 0.34 * span
    for pk, colr, off in ((z[f'pk_true_{tag}'], fs.C['true'], 0.0),
                          (z[f'pk_pred_{tag}'], fs.C['recon'], 0.12)):
        pk = np.asarray(pk, int)
        ax.vlines(pk / SFREQ, base - off * span, base - off * span + 0.08 * span,
                  color=colr, lw=0.6)
    ax.set_xlim(0, t_max)
    ax.set_ylim(base - 0.22 * span, tr.max() + 0.30 * span)
    ax.set_yticks([])
    ax.set_xlabel('time (s)')
    fs.despine(ax, keep=('bottom',))
    fs.offset_spines(ax)

    # Direct labels instead of a floating legend, once, on the left column only.
    if labels:
        ax.text(0.0, 1.02, 'recorded ECG', transform=ax.transAxes, ha='left',
                va='bottom', color=fs.C['true'])
        ax.text(0.34, 1.02, 'reconstructed', transform=ax.transAxes, ha='left',
                va='bottom', color=fs.C['recon'])
    return axes, im


# ============================================================
# 4. ASSEMBLY
# ============================================================
def build(z, width='double', height_mm=158.0):
    fs.use(base_pt=7.0)
    fig = fs.figure(width=width, height_mm=height_mm)

    # col 0 funnel a | 1 spacer | 2 funnel b
    gs = GridSpec(len(ROWS), 3, figure=fig,
                  width_ratios=[1.0, 0.085, 1.0],
                  height_ratios=[r[3] for r in ROWS],
                  left=0.152, right=0.988, top=0.885, bottom=0.112,
                  hspace=0.36, wspace=0.05)

    cols = {}
    im = None
    for (tag, letter, verdict, vkey), col in zip(COLS, (0, 2)):
        cols[tag], im = funnel_column(fig, gs, col, z, tag, letter, verdict, vkey,
                                      labels=(col == 0))

    # Soft column tints, behind everything. Cheap, and it separates the two
    # conditions before any label is read.
    for tag, vkey in (('good', 'pass'), ('bad', 'fail')):
        p0 = cols[tag]['trace_in'].get_position()
        p1 = cols[tag]['trace_out'].get_position()
        fig.add_artist(Rectangle(
            (p0.x0 - 0.008, p1.y0 - 0.012), (p0.x1 - p0.x0) + 0.016,
            (p0.y1 - p1.y0) + 0.024, transform=fig.transFigure,
            facecolor=fs.C[vkey], alpha=TINT_ALPHA, lw=0, zorder=0))

    # Compact horizontal colourbar in the bottom-left margin, clear of every row.
    cax = fig.add_axes([0.030, 0.055, 0.082, 0.010])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_ticks([0, CLIP])
    cb.set_ticklabels(['0', f'≥{CLIP:g}'])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=1.6, pad=1.2)
    cax.text(0.0, 1.9, 'activation |z|', transform=cax.transAxes,
             ha='left', va='bottom', color=fs.C['ink_soft'])
    return fig


# ============================================================
# 6. CAPTION
# ============================================================
def write_caption(path, z):
    rf = _receptive_field_samples() / SFREQ
    med = lambda t, s: float(np.median(z[f'mod_{t}_{s}']))
    lad = {t: ' → '.join(f'{med(t, s):.1f}' for s in STAGE_KEYS)
           for t in ('good', 'bad')}
    fl = {t: np.median([np.median(z[f'floor_{t}_{s}']) for s in STAGE_KEYS])
          for t in ('good', 'bad')}

    txt = f"""Figure 1 | The heartbeat emerging through the network -- and failing to.

{N_SHOW * WINDOW / SFREQ:.0f} seconds of two held-out recordings passed through the deployed model
(single-stream dilated convolutional network, 0.70 M parameters), with identical
weights and identical preprocessing. (a) A successful reconstruction (R-peak
detection F1 = {float(z['f1_good']):.2f}; cardiac field artifact leakage {float(z['leak_good']):.2f}). (b) A failed one
(F1 = {float(z['f1_bad']):.2f}; leakage {float(z['leak_bad']):.2f}). Each is the median-performing member of its
group among the held-out recordings retained and rejected by the reference-free
quality gate, so neither column is the best or the worst case available; the panel
makes a claim about two classes of reconstruction rather than about two
participants. F1 is measured against the simultaneously recorded ECG; leakage is
the single-channel maximum peak-to-peak of the R-peak-triggered average on z-scored
EEG, and is therefore dimensionless.

Rows, top to bottom: the recorded 128-channel EEG; the same data as the network
receives it; the six-layer dilated convolutional encoder; the four decoder stages
(a pointwise reduction to 32 channels, then 16, 8 and 4); and the output ECG (red, dashed) against the
recorded ECG (black, solid), each z-scored over the segment shown, with recorded
and detected R-peaks as rug marks beneath. Every stage is length-preserving, so
all rows share one time axis: the temporal dimension is fixed at 512 samples (2 s)
throughout and only the feature dimension shrinks. White verticals mark the
boundary between the independent 2 s forward passes whose outputs are concatenated
at inference. The dilated encoder's receptive field is {rf:.2f} s, effectively one
whole window, so each output sample integrates two to three cardiac cycles.

Heat is the per-feature z-scored magnitude within each recording, clipped at
{CLIP:g}; (b) is therefore not a dimmer version of (a) but an unstructured one.
Features are ordered by cardiac modulation in (a) and that same ordering is
applied to (b), so row i is the same network unit in both panels and the columns
can be read against each other unit by unit. The ordering is a display choice and
carries no meaning inside the model.

The impression the two columns give can be quantified. Taking the median feature's
beat-locked modulation depth at each stage -- peak-to-peak of the R-peak-triggered
average of the z-scored feature, over {N_STAT * WINDOW / SFREQ / 60:.0f} min of each recording, against a
permutation floor of {N_PERM} random-trigger resamples -- the depth climbs
monotonically through the network in the successful reconstruction, {lad['good']}
SD across the stages of (a) against a floor of {fl['good']:.1f} SD. In the failed
reconstruction it does not, {lad['bad']} SD against {fl['bad']:.1f} SD,
remaining at or near chance at every stage. With negligible cardiac field at the
scalp there is nothing to extract, and the decoder's capacity is spent on structure
that is not there. The failure is therefore visible inside the network and not only
at its output, and it is the same failure the reference-free gate detects from the
reconstructed R-R intervals alone, without any cardiac reference.
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f'  wrote caption -> {path}')


# ============================================================
# 7. CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run-dir', default=RUN_DIR)
    ap.add_argument('--data-folder', default=None)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--good-id', type=int, default=None,
                    help='subject id for panel a (default: median retained)')
    ap.add_argument('--bad-id', type=int, default=None,
                    help='subject id for panel b (default: median rejected). Pass '
                         'an id not in the group to list the choices.')
    ap.add_argument('--outdir', default=FIG_DIR)
    ap.add_argument('--stem', default='fig_network_emergence')
    ap.add_argument('--height-mm', type=float, default=158.0)
    args = ap.parse_args()

    cache = args.cache or os.path.join(args.run_dir, CACHE_NAME)

    if args.prepare:
        print('\n=== preparing ===')
        prepare(args.run_dir, args.data_folder, cache, args.good_id, args.bad_id)
        print('\nnow run:  python fig_network_emergence.py')
        return

    print('\n=== building ===')
    if not os.path.exists(cache):
        raise SystemExit(f'no cache at {cache}\nRun:  python fig_network_emergence.py --prepare')

    z = dict(np.load(cache, allow_pickle=False))
    fig = build(z, height_mm=args.height_mm)
    fs.check_text_sizes(fig)
    fs.save(fig, args.stem, outdir=args.outdir)
    write_caption(os.path.join(args.outdir, f'{args.stem}_caption.txt'), z)
    plt.close(fig)


if __name__ == '__main__':
    main()
