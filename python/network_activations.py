# -*- coding: utf-8 -*-
"""
network_activations.py -- per-stage activations and beat-locked modulation in the network.

fig_network_emergence.py (Fig. 3) imports its activation and statistics functions. Run on its
own, it draws an earlier draft of that figure.

The argument the paper makes is mechanistic: a cardiac field that no one can see
in the EEG is read out, by a stack of dilated convolutions, as R-peak timing. A
figure that only shows inputs and outputs asks the reader to take the middle on
trust. This one opens it.

    a  THE TRANSFORM. The same eight seconds at every stage of the deployed
       network -- recorded EEG, the 128-channel input as the model sees it, the
       dilated encoder, the 32-d latent, the decoder's 16 / 8 / 4 channels, and
       the reconstructed ECG. Time is on one shared axis because the
       architecture is length-preserving end to end, which is the whole point of
       it, and which this panel makes visible rather than asserting in a table.
       Beside each stage: how deeply that stage's features are modulated by the
       heartbeat, against a permutation floor. The number climbs monotonically
       from input to output. That is the heartbeat being extracted, quantified.

    b  Where the cardiac field lands on the scalp (R-peak-triggered average).
    c  Where the model actually reads (per-channel occlusion).
    d  b against c. The network converged on the volume-conduction topography of
       the heart without ever being told the heart exists.

    e  Per-beat latency error on the recordings the gate retains.
    f  Leakage against F1, coloured by the frozen gate verdict.

Two stages, because a and b-d need the weights and the recordings:

    python network_activations.py --prepare     # GPU, a few minutes
    python network_activations.py               # figures/network_activations.{pdf,png,svg}

Requires fig_style.py, summary_figure_panels.py, learned_topography.py, summary_figure.py and
hdeeg2ecg_model.py in the same folder.

@author: figures for Fraile-Vazquez et al.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import fig_style as fs
import learned_topography as ft
from summary_figure import (RUN_DIR, SFREQ, WINDOW, load_subject_table, panel_c,
                         panel_d, subject_id)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
CACHE_NAME = 'network_activations_inputs.npz'

N_SHOW = 4        # inference windows drawn in panel a  (4 x 2 s = 8 s)
N_STAT = 60       # windows the modulation statistic is computed over (2 min)
N_TOPO_SUBJ = 6   # highest-leakage retained recordings averaged for b-d
HALF_WIN = 40     # +/-156 ms around each R-peak, for the modulation statistic
N_PERM = 200
ABL_KS = (4, 8, 16, 32, 64)   # group sizes for the electrode-deletion curve
N_RAND = 6                    # random draws averaged per group size

# Activations get their own visual language: no hue at all. fig_style's semantic
# colours stay reserved for what they already mean (true/reconstructed,
# retain/reject) and the diverging map stays reserved for signed scalp fields, so
# the one saturated thing in panel a is the reconstructed ECG at the bottom of it.
CMAP_ACT = LinearSegmentedColormap.from_list(
    'act', ['#FFFFFF', '#E8E6DF', '#C9C7BF', '#8C8A83', '#55534E', '#1A1A1A'], N=256)

# name, label, tensor width -- in forward order.
STAGES = [
    ('eeg',     'EEG input',        128),
    ('encoder', 'dilated encoder',  128),
    ('latent',  'latent',            32),
    ('dec16',   'decoder',           16),
    ('dec8',    'decoder',            8),
    ('dec4',    'decoder',            4),
]

# Row heights. Not proportional to channel count -- 128 against 4 would leave the
# decoder invisible -- but tapering, so the figure reads as a funnel.
ROW_H = {'trace_in': 0.62, 'eeg': 1.00, 'encoder': 1.00, 'latent': 0.62,
         'dec16': 0.45, 'dec8': 0.34, 'dec4': 0.26, 'trace_out': 0.92}


# ============================================================
# 1. ACTIVATIONS
# ============================================================
def extract_activations(net, windows, device, batch=32):
    """Every intermediate tensor of the deployed conv_only model, as (T, C).

    Hooks rather than a re-implementation of forward(): the arrays are then
    guaranteed to be what the network computes, not what this script thinks it
    computes. The latent is taken as the decoder's input because the model applies
    GELU to fc_latent inline in forward() -- there is no module to hang a hook on.
    """
    import torch

    acts, pre = {}, {}

    def grab(name, channels_first):
        def hook(_m, _i, out):
            a = out.detach().float().cpu().numpy()
            acts.setdefault(name, []).append(a.transpose(0, 2, 1) if channels_first else a)
        return hook

    def grab_input(name):
        def hook(_m, inp):
            pre.setdefault(name, []).append(
                inp[0].detach().float().cpu().numpy().transpose(0, 2, 1))
        return hook

    dec = net.decoder.net
    handles = [
        net.conv_encoder.register_forward_hook(grab('encoder', False)),
        dec[0].register_forward_pre_hook(grab_input('latent')),
        dec[3].register_forward_hook(grab('dec16', True)),
        dec[7].register_forward_hook(grab('dec8', True)),
        dec[11].register_forward_hook(grab('dec4', True)),
    ]

    outs = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            xb = torch.from_numpy(windows[i:i + batch]).float().to(device)
            mb = torch.ones(xb.shape[0], WINDOW, dtype=torch.bool, device=device)
            outs.append(net(xb, mb).squeeze(-1).cpu().numpy())
    for h in handles:
        h.remove()

    A = {k: np.concatenate(v, 0).reshape(-1, np.concatenate(v, 0).shape[-1])
         for k, v in acts.items()}
    A['latent'] = np.concatenate(pre['latent'], 0).reshape(-1, 32)
    A['eeg'] = windows[:, :, :128].reshape(-1, 128)
    A['out'] = np.concatenate(outs, 0).reshape(-1)
    return A


def reconstruction_r(net, windows, targets, device, batch=64):
    """Mean per-window Pearson r, the model-selection criterion, on given inputs."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            xb = torch.from_numpy(windows[i:i + batch]).float().to(device)
            mb = torch.ones(xb.shape[0], WINDOW, dtype=torch.bool, device=device)
            out.append(net(xb, mb).squeeze(-1).cpu().numpy())
    p = np.concatenate(out, 0)
    pc = p - p.mean(1, keepdims=True)
    tc = targets - targets.mean(1, keepdims=True)
    den = np.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1)) + 1e-12
    return float(np.nanmean((pc * tc).sum(1) / den))


def group_ablation(net, windows, targets, cfa, device, ks=ABL_KS, n_rand=N_RAND, seed=0):
    """Delete k electrodes at a time and watch the reconstruction.

    Occluding one channel at a time answers the wrong question here. The cardiac
    field reaches most of the montage, so the model has every opportunity to be
    redundant, and a redundant model loses nothing measurable to any single
    electrode -- which is exactly what we find (see the caption). Deleting groups
    restores the power: if the network is reading the cardiac field, removing the
    k electrodes where that field is strongest must cost more than removing k at
    random, and removing the k weakest must cost least. The random arm is the
    control that separates 'these electrodes carry the heart' from 'this model
    dislikes missing inputs'.
    """
    order = np.argsort(-np.abs(np.asarray(cfa, float)))
    rng = np.random.default_rng(seed)

    def drop(idx):
        x = windows.copy()
        x[:, :, idx] = 0.0
        return reconstruction_r(net, x, targets, device)

    base = reconstruction_r(net, windows, targets, device)
    top, low, rand = [], [], []
    for k in ks:
        top.append(drop(order[:k]))
        low.append(drop(order[-k:]))
        rand.append(np.mean([drop(rng.choice(128, k, replace=False))
                             for _ in range(n_rand)]))
    return base, np.array(top), np.array(rand), np.array(low)


def beat_modulation(M, peaks, hw=HALF_WIN, n_perm=N_PERM, seed=0):
    """How deeply each feature is modulated by the heartbeat, in SD, vs chance.

    Peak-to-peak of the R-peak-triggered average of the z-scored feature. The
    permutation floor matters: a triggered average over a finite number of beats
    has a non-zero peak-to-peak even for a feature with no cardiac locking at all,
    and that floor differs between stages because their autocorrelation differs.
    Comparing raw modulation across stages without it would manufacture a trend.
    """
    z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    pk = peaks[(peaks - hw >= 0) & (peaks + hw < len(z))]
    if len(pk) < 5:
        return None, None

    def p2p(trig):
        ta = np.stack([z[p - hw:p + hw] for p in trig]).mean(0)
        return ta.max(0) - ta.min(0)

    obs = p2p(pk)
    rng = np.random.default_rng(seed)
    null = np.stack([p2p(rng.integers(hw, len(z) - hw, size=len(pk)))
                     for _ in range(n_perm)])
    return obs, null.mean(0)


# ============================================================
# 2. PREPARE
# ============================================================
def prepare(run_dir, data_folder=None, cache_path=None):
    import torch
    import hdeeg2ecg_model as P

    data_folder = data_folder or P.DATA_FOLDER
    cache_path = cache_path or os.path.join(run_dir, CACHE_NAME)

    df = load_subject_table(run_dir)
    retained = df[df['gate_pass']]

    # Panel a: the median-performing retained recording, so the transform is shown
    # on a typical case and not on the best one.
    tgt = retained['f1'].median()
    hero = retained.loc[(retained['f1'] - tgt).abs().sort_values(kind='stable').index[0]]
    # Panels b-d: the highest-leakage retained recordings. Occlusion needs a
    # cardiac field present to have anything to occlude, and a single-subject map
    # is noisy; the claim is about the model, not about one recording.
    topo_subj = retained.nlargest(N_TOPO_SUBJ, 'leak_true')

    print(f"  panel a: sub-{hero['id']} (F1 = {hero['f1']:.3f})")
    print(f"  panels b-d: {', '.join('sub-%d' % i for i in topo_subj['id'])}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = P.EEGtoECGModel(latent_dim=P.SELECTED_LATENT_DIM,
                          decoder_schedule=P.SELECTED_DECODER,
                          variant=P.SELECTED_VARIANT)
    net.load_state_dict(torch.load(P.WEIGHTS_PATH,
                                   map_location='cpu'))
    net.eval().to(device)
    print(f'  loaded {P.SELECTED_VARIANT} on {device}')

    out = {}

    # ---- panel a --------------------------------------------------------
    eeg, evt, _ = P.load_subject_data(os.path.join(data_folder, str(hero['file'])))
    n_full = (len(eeg) // WINDOW) * WINDOW
    start = ((n_full // 2) // WINDOW) * WINDOW
    start = min(start, n_full - N_STAT * WINDOW)

    stack = eeg[start:start + N_STAT * WINDOW].reshape(N_STAT, WINDOW, 129).astype(np.float32).copy()
    stack[:, :, 128] = 0.0                       # inference masking, as deployed
    A = extract_activations(net, stack, device)

    pk_local = np.where(evt[start:start + N_STAT * WINDOW] == 1)[0]
    n_disp = N_SHOW * WINDOW

    for name, _lab, _c in STAGES:
        M = A[name]
        obs, floor = beat_modulation(M, pk_local)
        # Features ordered by cardiac modulation so the beat-locked structure is
        # contiguous rather than scattered. Display only; disclosed in the caption.
        order = np.argsort(-obs)
        out[f'act_{name}'] = M[:n_disp][:, order].astype(np.float32)
        out[f'mod_{name}'] = obs[order].astype(np.float32)
        out[f'floor_{name}'] = floor[order].astype(np.float32)

    # The recorded ECG on the same z-scale the model was trained against, and the
    # peaks the deployed detector finds on the full recording (its amplitude
    # threshold is a whole-recording statistic, so it cannot be run on a slice).
    pred_full, true_full, _ = P.reconstruct_subject(net, eeg, device)
    pk_pred = P.detect_peaks(pred_full)
    sl = slice(start, start + n_disp)
    out['trace_true'] = true_full[sl].astype(np.float32)
    out['trace_recon'] = pred_full[sl].astype(np.float32)
    out['eeg_uv'] = _display_eeg_uv(os.path.join(data_folder, str(hero['file'])),
                                    start, n_disp, P).astype(np.float32)
    out['pk_true'] = pk_local[pk_local < n_disp]
    out['pk_pred'] = pk_pred[(pk_pred >= start) & (pk_pred < start + n_disp)] - start
    out['hero_id'] = int(hero['id'])
    out['hero_f1'] = float(hero['f1'])
    out['hero_file'] = np.array(str(hero['file']))

    # ---- panels b-d -----------------------------------------------------
    imps, cfas = [], []
    abl_base, abl_top, abl_rand, abl_low = [], [], [], []
    for _, row in topo_subj.iterrows():
        e, v, _ = P.load_subject_data(os.path.join(data_folder, str(row['file'])))
        nw = min(N_STAT, len(e) // WINDOW)
        w = e[:nw * WINDOW].reshape(nw, WINDOW, 129).astype(np.float32).copy()
        ecg = w[:, :, 128]
        tgt_w = (ecg - ecg.mean(1, keepdims=True)) / (ecg.std(1, keepdims=True) + 1e-12)
        w[:, :, 128] = 0.0
        cfa = ft.cfa_topography(e, v)
        print(f"  sweeps, sub-{row['id']} ({nw} windows)")
        imps.append(ft.channel_importance(net, w, tgt_w, device=device, verbose=False))
        cfas.append(cfa)
        b, t, r, l = group_ablation(net, w, tgt_w, cfa, device)
        abl_base.append(b), abl_top.append(t), abl_rand.append(r), abl_low.append(l)
        print(f"    base r = {b:.3f}   64 strongest removed -> {t[-1]:.3f}   "
              f"64 weakest -> {l[-1]:.3f}")

    out['abl_ks'] = np.array(ABL_KS)
    out['abl_base'] = np.array(abl_base)
    out['abl_top'] = np.array(abl_top)
    out['abl_rand'] = np.array(abl_rand)
    out['abl_low'] = np.array(abl_low)

    # cfa_topography fixes each map's sign by its own largest electrode, which is
    # not necessarily the same electrode in every recording. Align to the first map
    # before averaging, or two maps of the same field can partially cancel.
    cfas = [c if float(np.dot(c, cfas[0])) >= 0 else -c for c in cfas]

    out['importance'] = np.mean(imps, axis=0)
    out['importance_sd'] = np.std(imps, axis=0)
    out['cfa_topo'] = np.mean(cfas, axis=0)
    out['topo_ids'] = topo_subj['id'].values
    out['topo_file'] = np.array(str(topo_subj.iloc[0]['file']))

    # ---- panel e --------------------------------------------------------
    tol = int(P.QRS_MATCH_TOLERANCE_MS / 1000 * SFREQ)
    errs, per_subject = [], []
    for _, row in retained.iterrows():
        e, v, _ = P.load_subject_data(os.path.join(data_folder, str(row['file'])))
        pred, _t, _r = P.reconstruct_subject(net, e, device)
        tp_pk = np.where(v[:len(pred)] == 1)[0]
        _tp, _fp, _fn, d = P.match_peaks(tp_pk, P.detect_peaks(pred), tol)
        d_ms = d / SFREQ * 1000.0
        errs.append(d_ms)
        per_subject.append((row['id'], len(d_ms), np.mean(np.abs(d_ms)), np.mean(d_ms)))
        print(f"  sub-{row['id']}  {len(d_ms)} matched beats")

    out['errs_ms'] = np.concatenate(errs)
    out['per_subject'] = np.array(per_subject, dtype=float)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **out)
    print(f'\n  cached -> {cache_path}')
    return cache_path


def _display_eeg_uv(path, start, n, P):
    """The recorded EEG in microvolts. P.load_subject_data z-scores every channel
    because that is the model's input; a trace labelled uV must not come from it."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
    raw.resample(SFREQ, verbose='ERROR')
    raw.filter(l_freq=P.FILTER_LFREQ, h_freq=P.FILTER_HFREQ, method='iir',
               verbose='ERROR')
    return raw.get_data()[:128, start:start + n].T * 1e6


# ============================================================
# 3. PANEL A -- the transform
# ============================================================
def _receptive_field_samples(kernel=5, dilations=(2, 4, 8, 16, 32, 64)):
    return 1 + sum((kernel - 1) * d for d in dilations)


def panel_a_transform(fig, spec, z, clip=2.5):
    """Every stage of the network on one time axis.

    The heatmaps are per-feature z-scored magnitudes: what is being shown is where
    each stage is active, not its arbitrary units, and the stages differ in scale
    by orders of magnitude. Absolute values would show the last decoder layers as
    blank.
    """
    keys = ['trace_in'] + [s[0] for s in STAGES] + ['trace_out']
    gs = GridSpecFromSubplotSpec(
        len(keys), 3, subplot_spec=spec, width_ratios=[1.0, 0.165, 0.022],
        height_ratios=[ROW_H[k] for k in keys], hspace=0.32, wspace=0.055)

    n = len(z['trace_true'])
    t_max = n / SFREQ
    mod_max = max(float(z[f'mod_{s}'].max()) for s, _l, _c in STAGES) * 1.12

    axes, bars = {}, {}
    for i, k in enumerate(keys):
        axes[k] = fig.add_subplot(gs[i, 0])
        bars[k] = fig.add_subplot(gs[i, 1])
        bars[k].axis('off')
    # Short colourbar against the decoder rows. Spanning the whole stack made it
    # the tallest object in the panel, which is not what the panel is about.
    cax = fig.add_subplot(gs[4:7, 2])

    # ---- recorded EEG, as an eye sees it -------------------------------
    ax = axes['trace_in']
    eeg = z['eeg_uv']
    t = np.arange(n) / SFREQ
    for ch in range(0, 128, 3):
        ax.plot(t, eeg[:, ch], color=fs.C['ink_faint'], lw=0.2, alpha=0.55,
                rasterized=True)
    lo, hi = np.nanpercentile(eeg, [0.5, 99.5])
    ax.set_xlim(0, t_max)
    ax.set_ylim(lo * 1.15, hi * 1.15)
    ax.axis('off')
    ax.set_ylabel('')
    fs.scalebar(ax, 0.06, lo * 0.95, 1.0, 0.55 * hi, '1 s', f'{0.55*hi:.0f} µV')
    ax.text(1.0, 1.02, 'recorded EEG  ·  128 channels  ·  no heartbeat discernible',
            transform=ax.transAxes, ha='right', va='bottom', color=fs.C['ink_soft'])
    fs.panel_label(ax, 'a', dx=-0.004, dy=1.02)

    # ---- the stack ------------------------------------------------------
    im = None
    for name, label, nch in STAGES:
        ax = axes[name]
        M = z[f'act_{name}']
        zz = np.abs((M - M.mean(0)) / (M.std(0) + 1e-9)).T       # (C, T)
        im = ax.imshow(zz, aspect='auto', cmap=CMAP_ACT, vmin=0, vmax=clip,
                       origin='upper', rasterized=True,
                       extent=[0, t_max, nch, 0], interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.4)
            s.set_color(fs.C['rule'])
        ax.set_ylabel(f'{label}\n', rotation=0, ha='right', va='center',
                      labelpad=3, color=fs.C['ink'])
        ax.text(-0.012, 0.5, f'\n512 × {nch}', transform=ax.transAxes,
                rotation=0, ha='right', va='center', color=fs.C['ink_faint'])

        # window joins: the trace is the concatenation of independent forward
        # passes, and the paper argues the seams are invisible. Show where they are.
        for w in range(1, N_SHOW):
            ax.axvline(w * WINDOW / SFREQ, color='white', lw=0.35, alpha=0.55)

        bar = bars[name]
        obs, floor = z[f'mod_{name}'], z[f'floor_{name}']
        bar.barh([0], [float(np.median(obs))], height=0.55, color=fs.C['ink_soft'], lw=0)
        bar.plot([float(np.median(floor))] * 2, [-0.4, 0.4], color=fs.C['fail'],
                 lw=0.7, solid_capstyle='butt')
        bar.set_xlim(0, mod_max)
        bar.set_ylim(-0.75, 0.75)
        bar.text(float(np.median(obs)) + 0.03 * mod_max, 0,
                 f'{np.median(obs):.1f}', ha='left', va='center',
                 color=fs.C['ink'])

    # Receptive field of the dilated stack. Drawn above the encoder row rather
    # than on it: the row it describes is the darkest in the panel.
    rf_s = _receptive_field_samples() / SFREQ
    ax = axes['encoder']
    ax.plot([0.10, 0.10 + rf_s], [-5.5, -5.5], color=fs.C['recon'], lw=1.0,
            solid_capstyle='butt', zorder=5, clip_on=False)
    for xx in (0.10, 0.10 + rf_s):
        ax.plot([xx, xx], [-9.0, -2.0], color=fs.C['recon'], lw=1.0, zorder=5,
                clip_on=False)
    ax.text(0.10 + rf_s + 0.12, -5.5, f'receptive field {rf_s:.2f} s '
            '(2–3 cardiac cycles)', va='center', ha='left', color=fs.C['recon'],
            zorder=5, clip_on=False)

    # ---- reconstructed ECG ---------------------------------------------
    ax = axes['trace_out']
    tr, rc = z['trace_true'], z['trace_recon']
    zsc = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    tr, rc = zsc(tr), zsc(rc)
    ax.plot(t, tr, color=fs.C['true'], lw=0.55, ls=fs.LINESTYLE['true'], zorder=3)
    ax.plot(t, rc, color=fs.C['recon'], lw=0.75, ls=fs.LINESTYLE['recon'], zorder=4)

    span = tr.max() - tr.min()
    base = tr.min() - 0.30 * span
    for pk, col, off in ((z['pk_true'], fs.C['true'], 0.0),
                         (z['pk_pred'], fs.C['recon'], 0.10)):
        pk = np.asarray(pk, int)
        ax.vlines(pk / SFREQ, base - off * span, base - off * span + 0.07 * span,
                  color=col, lw=0.6)
    ax.set_xlim(0, t_max)
    ax.set_ylim(base - 0.18 * span, tr.max() + 0.10 * span)
    ax.set_yticks([])
    ax.set_xlabel('time (s)')
    fs.despine(ax, keep=('bottom',))
    fs.offset_spines(ax)
    ax.set_ylabel('reconstructed ECG\n', rotation=0, ha='right', va='center',
                  labelpad=3, color=fs.C['ink'])
    ax.text(-0.012, 0.5, '\n512 × 1', transform=ax.transAxes, ha='right',
            va='center', color=fs.C['ink_faint'])
    ax.legend(handles=[
        Line2D([], [], color=fs.C['true'], lw=0.9, ls=fs.LINESTYLE['true'],
               label='recorded ECG'),
        Line2D([], [], color=fs.C['recon'], lw=0.9, ls=fs.LINESTYLE['recon'],
               label=f"reconstructed  ·  sub-{int(z['hero_id'])}  ·  "
                     f"$F_1$ = {float(z['hero_f1']):.2f}")],
        loc='lower right', ncol=2, bbox_to_anchor=(1.0, 1.005),
        borderaxespad=0.0)

    # scale for the modulation bars, once, under the column
    bar = bars['trace_out']
    bar.axis('on')
    bar.set_xlim(0, mod_max)
    bar.set_ylim(0, 1)
    bar.set_yticks([])
    bar.set_xticks([0, round(mod_max / 2)])
    fs.despine(bar, keep=('bottom',))
    bar.set_xlabel('beat-locked\nmodulation (SD)', labelpad=1)

    bars['trace_in'].text(0.5, -0.2, 'median feature\nvs chance ',
                          transform=bars['trace_in'].transAxes, ha='center',
                          va='top', color=fs.C['ink_soft'])

    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, clip])
    cb.set_ticklabels(['0', f'≥{clip:g}'])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=1.6)
    cax.set_title('|z|', pad=2, loc='center', color=fs.C['ink_soft'])
    return axes, im


# ============================================================
# 4. PANELS B-D -- what the model reads
# ============================================================
def _topo(fig, ax, v, info, title, letter):
    import mne
    v = np.asarray(v, float)
    lim = float(np.nanpercentile(np.abs(v), 98))
    im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False,
                                 cmap=fs.CMAP_DIVERGING, vlim=(-lim, lim),
                                 contours=4, sensors=False, outlines='head',
                                 extrapolate='head')
    ax.set_title(title, pad=1, loc='center', color=fs.C['ink'])
    cb = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.02)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=1.6)
    cb.set_ticks([-lim, 0, lim])
    cb.set_ticklabels(['−', '0', '+'])
    fs.panel_label(ax, letter, dx=0.02, dy=1.34)
    return im


def panel_d_ablation(ax, z, letter='d'):
    """Reconstruction against how many electrodes are deleted, by which ones.

    Grey (random) is the control curve; the gap between the two coloured curves is
    the result. Deleting electrodes is an out-of-distribution perturbation, so the
    absolute values are not reconstruction quality on real data -- only the
    separation between the arms is interpretable.
    """
    ks = np.asarray(z['abl_ks'])
    x = np.arange(len(ks))
    base = np.asarray(z['abl_base'], float)
    n = len(base)

    ax.axhline(base.mean(), color=fs.C['ink_faint'], lw=0.5, ls=(0, (2, 2)),
               zorder=1)
    ax.text(-0.18, base.mean(), 'full montage', ha='left', va='bottom',
            color=fs.C['ink_faint'])

    series = (('abl_top', fs.C['recon'], 'strongest cardiac field'),
              ('abl_rand', fs.C['ink_soft'], 'random electrodes'),
              ('abl_low', fs.C['ink_faint'], 'weakest cardiac field'))
    for key, col, lab in series:
        M = np.asarray(z[key], float)
        m, s = M.mean(0), M.std(0, ddof=1) / np.sqrt(n)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, m, color=col, lw=1.0, marker='o', ms=2.8, mec='white',
                mew=0.4, label=lab, zorder=3, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.25, len(ks) - 0.75)
    ax.set_xlabel('electrodes deleted (of 128)')
    ax.set_ylabel('reconstruction $r$')
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, -0.03))
    fs.stat_text(ax, f'{n} recordings', x=0.98, y=0.62)
    fs.offset_spines(ax)
    fs.panel_label(ax, letter)

    top64 = np.asarray(z['abl_top'], float)[:, -1]
    low64 = np.asarray(z['abl_low'], float)[:, -1]
    return float(top64.mean()), float(low64.mean())


# ============================================================
# 5. ASSEMBLY
# ============================================================
def build(z, df, info, width='double', height_mm=214.0):
    fs.use(base_pt=7.0)
    fig = fs.figure(width=width, height_mm=height_mm)

    gs = GridSpec(3, 1, figure=fig, height_ratios=[2.42, 0.86, 1.00],
                  left=0.135, right=0.955, top=0.972, bottom=0.052, hspace=0.30)

    panel_a_transform(fig, gs[0], z)

    gs_bd = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1],
                                    width_ratios=[0.82, 0.82, 1.35], wspace=0.44)
    ax_b = fig.add_subplot(gs_bd[0])
    ax_c = fig.add_subplot(gs_bd[1])
    ax_d = fig.add_subplot(gs_bd[2])
    _topo(fig, ax_b, z['cfa_topo'], info, 'cardiac field\nR-peak-triggered average', 'b')
    _topo(fig, ax_c, z['importance'], info, 'model reliance\nsingle-channel occlusion', 'c')
    abl = panel_d_ablation(ax_d, z)

    gs_ef = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.30)
    ax_e = fig.add_subplot(gs_ef[0])
    ax_f = fig.add_subplot(gs_ef[1])
    panel_c(ax_e, z['errs_ms'], letter='e')
    panel_d(ax_f, df, letter='f')
    return fig, abl


# ============================================================
# 6. CAPTION
# ============================================================
def write_caption(path, z, df, abl):
    e = np.asarray(z['errs_ms'], float)
    ps = np.asarray(z['per_subject'], float)
    rho_l, p_l = sps.spearmanr(df['leak_true'], df['f1'])
    rho_occl, p_occl = sps.spearmanr(np.abs(z['cfa_topo']), z['importance'])
    n_pass = int(df['gate_pass'].sum())
    mods = ' → '.join(f'{np.median(z[f"mod_{s}"]):.1f}' for s, _l, _c in STAGES)
    floors = np.median([np.median(z[f'floor_{s}']) for s, _l, _c in STAGES])
    rf = _receptive_field_samples() / SFREQ

    txt = f"""Figure 1 | The heartbeat emerging through the network.

(a) The same {N_SHOW * WINDOW / SFREQ:.0f} s of one held-out recording (sub-{int(z['hero_id'])}, the
median-performing recording the quality gate retains, F1 = {float(z['hero_f1']):.2f}) at every
stage of the deployed model. Top, the recorded 128-channel EEG. Below it, the
same data as the network receives it and then the tensor at each subsequent
stage: the six-layer dilated convolutional encoder, the 32-dimensional latent,
and the decoder's 16, 8 and 4 channels. Every stage is length-preserving, so all
rows share one time axis; the temporal dimension is fixed at 512 samples (2 s)
throughout and only the feature dimension shrinks. Heat is the per-feature
z-scored magnitude, clipped at 2.5; features within each stage are ordered by
cardiac modulation, which is a display choice and not an ordering in the model.
White verticals mark the boundaries between the independent 2 s forward passes
whose outputs are concatenated at inference. The red bracket gives the encoder's
receptive field, {rf:.2f} s, so each output sample integrates two to three cardiac
cycles. Bottom, the reconstructed ECG (red, dashed) against the recorded ECG
(black, solid), each z-scored over the segment shown, with recorded and detected
R-peaks as rug marks. Right, the median feature's beat-locked modulation depth at
each stage -- peak-to-peak of the R-peak-triggered average of the z-scored
feature, computed over {N_STAT * WINDOW / SFREQ / 60:.0f} min -- with the permutation floor in red
({N_PERM} random-trigger resamples, median floor {floors:.1f} SD). It rises monotonically,
{mods} SD, from input to decoder: the extraction of the heartbeat, measured.

(b) Scalp distribution of the cardiac field: the leading spatial component of the
R-peak-triggered average of the raw EEG, averaged over the {len(z['topo_ids'])} highest-leakage
retained recordings. This is computed from the recorded R-peaks and never sees
the model.

(c) Scalp distribution of the model's reliance on single electrodes, over the same
recordings: the drop in per-window reconstruction r when each electrode in turn is
zeroed at the input. No electrode is individually necessary -- the largest drop is
{np.max(z['importance']):.3f} in r -- and this map does not correlate with (b) across the 128
electrodes (Spearman rho = {rho_occl:.2f}, {'p < 0.001' if p_occl < 1e-3 else 'p = %.2f' % p_occl}). We show it because it is the
obvious analysis and because its null result is informative: the cardiac field
reaches most of the montage, so the network can be, and is, redundant. Single-
channel occlusion is therefore the wrong instrument, and (d) is the right one.

(d) Reconstruction r when k electrodes are deleted at the input simultaneously,
as a function of k, for the k electrodes where the cardiac field of (b) is
strongest, the k where it is weakest, and k drawn at random ({N_RAND} draws averaged
per point). Mean +/- s.e.m. over the same {len(z['topo_ids'])} recordings. Deleting the 64
electrodes carrying the weakest cardiac field leaves reconstruction essentially
intact (r = {abl[1]:.2f} against {np.mean(z['abl_base']):.2f} on the full montage); deleting the 64 carrying
the strongest collapses it (r = {abl[0]:.2f}), with random deletion in between. The
network is reading the volume-conducted field of the heart, which it was never
told about and never given a location for. Deletion is an out-of-distribution
perturbation, so only the separation between arms is interpretable, not the
absolute values; the non-monotonicity of the strongest-field arm at small k
reflects that some recordings improve when one or two very high-amplitude
electrodes are removed.

(e) Signed per-beat latency error, reconstructed minus recorded, pooled over the
{len(ps)} held-out recordings the gate retains ({len(e):,} matched beats, +/-50 ms matching
tolerance). One bin per sample offset, since both peak trains are integer sample
indices. Mean absolute error {np.mean(np.abs(e)):.2f} ms and bias {np.mean(e):+.2f} ms across beats;
across recordings, {ps[:, 2].mean():.2f} ms and {ps[:, 3].mean():+.2f} ms. One sample at {SFREQ} Hz
= {1000/SFREQ:.1f} ms (dashed).

(f) R-peak detection F1 against cardiac field artifact leakage for every held-out
recording (n = {len(df)}; selection and testing splits pooled), coloured by the frozen
reference-free gate verdict ({n_pass} retained, {len(df) - n_pass} rejected). Leakage is the
single-channel maximum peak-to-peak of the R-peak-triggered average on z-scored
EEG, and is therefore dimensionless. Spearman rho = {rho_l:.2f},
{'p < 0.001' if p_l < 1e-3 else 'p = %.3f' % p_l}. Failed and successful reconstructions do not overlap on
this axis, but the gate is calibrated on the reconstructed R-R interval fractions
and not on leakage, so no boundary is drawn.
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
    ap.add_argument('--outdir', default=FIG_DIR)
    ap.add_argument('--stem', default='network_activations')
    ap.add_argument('--height-mm', type=float, default=214.0)
    args = ap.parse_args()

    cache = args.cache or os.path.join(args.run_dir, CACHE_NAME)

    if args.prepare:
        print('\n=== preparing ===')
        prepare(args.run_dir, args.data_folder, cache)
        print('\nnow run:  python network_activations.py')
        return

    print('\n=== building ===')
    if not os.path.exists(cache):
        raise SystemExit(f'no cache at {cache}\nRun:  python network_activations.py --prepare')

    import mne
    import hdeeg2ecg_model as P

    z = dict(np.load(cache, allow_pickle=False))
    df = load_subject_table(args.run_dir)

    raw = mne.io.read_raw_eeglab(
        os.path.join(args.data_folder or P.DATA_FOLDER, str(z['topo_file'])),
        preload=False, verbose='ERROR')
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True)[:128])

    fig, abl = build(z, df, info, height_mm=args.height_mm)
    fs.check_text_sizes(fig)
    fs.save(fig, args.stem, outdir=args.outdir)
    write_caption(os.path.join(args.outdir, f'{args.stem}_caption.txt'), z, df, abl)
    plt.close(fig)


if __name__ == '__main__':
    main()
