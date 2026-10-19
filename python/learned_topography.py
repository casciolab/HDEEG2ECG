# -*- coding: utf-8 -*-
"""
learned_topography.py -- did the model learn the physics?

Occlude each EEG channel in turn, measure the drop in reconstruction quality,
and you get a 128-value importance map: which electrodes the network actually
reads. Plot it beside the cardiac field's own scalp distribution -- the leading
spatial component of the R-peak-triggered average, which you already compute in
eeg_cardiac_leakage / leakage_spatial -- and correlate the two across channels.

If they match, the network converged on the volume-conduction topography of the
heart without ever being told the heart exists. That is a one-panel
interpretability result, and it makes the artifact-as-signal framing visual
rather than rhetorical.

Two things it also buys you:
  * a partial answer to "did you compare against a non-learned baseline?" --
    it shows what the model is using, even if it does not show what a simpler
    method would achieve;
  * a diagnostic. If importance is focal on one or two electrodes, a
    single-channel read may be nearly as good and you want to know that before
    a reviewer asks.

Needs MNE for the topoplot (montage geometry). Everything else is numpy.

    python learned_topography.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

import fig_style as fs

SFREQ = 256
WINDOW = 512
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


# ============================================================
# 1. OCCLUSION IMPORTANCE
# ============================================================
def channel_importance(model, windows, targets, device='cuda', batch=64,
                       verbose=True):
    """Drop in per-window reconstruction r when each channel is zeroed.

    windows : (n_win, 512, 129) float32, exactly as fed at inference
              (ECG channel already zeroed)
    targets : (n_win, 512)     float32, per-window z-scored ECG

    Returns (128,) array. Larger = the model relies on that electrode more.

    Zeroing is the right ablation here because the input is already z-scored
    per channel, so zero is the channel mean -- an uninformative value rather
    than an out-of-distribution one. Do not use noise; that tests robustness,
    not reliance.
    """
    import torch

    model = model.eval().to(device)

    def _r(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xb = torch.from_numpy(x[i:i + batch]).float().to(device)
                mb = torch.ones(xb.shape[0], WINDOW, dtype=torch.bool, device=device)
                out.append(model(xb, mb).squeeze(-1).cpu().numpy())
        p = np.concatenate(out, 0)
        pc = p - p.mean(1, keepdims=True)
        tc = targets - targets.mean(1, keepdims=True)
        den = np.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1)) + 1e-12
        return float(np.nanmean((pc * tc).sum(1) / den))

    base = _r(windows)
    imp = np.zeros(128, float)
    for ch in range(128):
        x = windows.copy()
        x[:, :, ch] = 0.0
        imp[ch] = base - _r(x)
        if verbose and (ch + 1) % 16 == 0:
            print(f'    channel {ch+1}/128   baseline r = {base:.4f}')
    return imp


def cfa_topography(eeg, event_vector, half_win_ms=120, sfreq=SFREQ):
    """Leading spatial component of the R-peak-triggered average: the cardiac
    field's scalp distribution. Sign-fixed so the largest-magnitude electrode is
    positive, otherwise the SVD's arbitrary sign flips the map between subjects.
    """
    hw = int(half_win_ms / 1000 * sfreq)
    pk = np.where(event_vector == 1)[0]
    pk = pk[(pk - hw >= 0) & (pk + hw < len(eeg))]
    if len(pk) < 5:
        return None
    seg = np.stack([eeg[p - hw:p + hw + 1, :128] for p in pk]).mean(0)
    seg = seg - seg.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(seg, full_matrices=False)
    v = Vt[0]
    return v * np.sign(v[np.argmax(np.abs(v))])


# ============================================================
# 2. FIGURE
# ============================================================
def plot(importance, cfa_topo, info, letter_offset=0, width='onehalf',
         height_mm=62.0):
    """Two topomaps plus a scatter. info is an mne.Info with the montage set."""
    import mne

    fs.use(base_pt=7.0)
    fig = fs.figure(width=width, height_mm=height_mm)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25],
                          left=0.02, right=0.955, top=0.86, bottom=0.17,
                          wspace=0.40)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    def _topo(ax, v, title, cmap):
        v = np.asarray(v, float)
        lim = float(np.nanpercentile(np.abs(v), 99))
        im, _ = mne.viz.plot_topomap(
            v, info, axes=ax, show=False, cmap=cmap,
            vlim=(-lim, lim), contours=4, sensors=False, outlines='head',
            extrapolate='head')
        ax.set_title(title, pad=2, loc='center')
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(width=0.5, length=1.8)
        cb.set_ticks([-lim, 0, lim])
        cb.set_ticklabels(['−', '0', '+'])
        return im

    _topo(ax0, cfa_topo, 'cardiac field\n(R-peak-triggered average)',
          fs.CMAP_DIVERGING)
    _topo(ax1, importance, 'model reliance\n(channel occlusion)',
          fs.CMAP_DIVERGING)
    fs.panel_label(ax0, 'a', dx=0.02, dy=1.10)
    fs.panel_label(ax1, 'b', dx=0.02, dy=1.10)

    x = np.abs(np.asarray(cfa_topo, float))
    y = np.asarray(importance, float)
    ax2.scatter(x, y, s=7, facecolor=fs.C['pass'], edgecolor='white',
                linewidth=0.25, alpha=0.9, clip_on=False)
    rho, p = sps.spearmanr(x, y)
    if len(x) > 2:
        b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax2.plot(xs, np.polyval(b, xs), color=fs.C['threshold'], lw=0.7,
                 ls=(0, (3, 2)), zorder=1)
    ax2.set_xlabel('cardiac field magnitude\n(|weight| per electrode)')
    ax2.set_ylabel('drop in reconstruction $r$')
    p_txt = 'p < 0.001' if p < 1e-3 else f'p = {p:.3f}'
    fs.stat_text(ax2, f'ρ = {rho:.2f}\n{p_txt}\nn = 128 electrodes')
    fs.offset_spines(ax2)
    fs.panel_label(ax2, 'c')
    return fig, dict(rho=float(rho), p=float(p))


def main():
    print('This script needs three things you must supply:')
    print('  1. the deployed model + a batch of inference windows')
    print('  2. an mne.Info carrying your EGI 128 montage')
    print('  3. one recording with ground-truth R-peaks, for the CFA topography')
    print()
    print('Sketch:')
    print("""
    import mne, numpy as np
    import hdeeg2ecg_model as P
    from learned_topography import channel_importance, cfa_topography, plot

    eeg, evt, _ = P.load_subject_data(path)              # a high-leakage subject
    n = (len(eeg) // 512) * 512
    win = eeg[:n].reshape(-1, 512, 129).astype('float32').copy()
    ecg = win[:, :, 128]
    tgt = (ecg - ecg.mean(1, keepdims=True)) / (ecg.std(1, keepdims=True) + 1e-12)
    win[:, :, 128] = 0.0

    imp  = channel_importance(model, win, tgt)           # (128,)
    topo = cfa_topography(eeg, evt)                      # (128,)

    raw  = mne.io.read_raw_eeglab(path)
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True)[:128])

    fig, stats = plot(imp, topo, info)
    print(stats)
    fs.save(fig, 'learned_topography')

    Average `imp` over several high-leakage subjects before plotting -- a
    single-subject occlusion map is noisy, and the claim is about the model,
    not about one recording.
    """)


if __name__ == '__main__':
    main()