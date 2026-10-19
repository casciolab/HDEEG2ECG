# -*- coding: utf-8 -*-
"""
fig_overview.py -- Fig. 1 of the Nature Communications manuscript.

a  Study overview: one trained model and one reference-free gate, applied in two
   lanes -- validation against simultaneous ECG, and deployment to an ECG-free
   corpus.
b  The deployed network, drawn horizontally in the machine-learning convention:
   blocks as slabs whose height scales with channel count and whose depth is the
   (constant) time axis, with a real input window and a real reconstruction.

Signals are one 2-s model window (46-48 s) from held-out subject 8004, reconstructed
here by the released weights (weights/final_model_weights.pt).

    python fig_overview.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as fs
import hdeeg2ecg_model as P

DEPLOYED = P.WEIGHTS_PATH
OUTDIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')]

SUBJECT = 'sub_8004_a0_withR.set'
W0 = 46.0                                  # s; start of one model window (a multiple of 2 s)

# ---- palette: manuscript blue family for the encoder, green for the decoder -------
INK, INK2, INK3 = '#1A1D21', '#4A5058', '#8A8F98'
BLUE, BLUE_D = '#3A6EA5', '#24507F'
ENC = ['#D6E4F3', '#C3D7EE', '#AFC9E7', '#9BBCE0', '#86ADD8', '#6F9DCF']
DEC = ['#DDF0E9', '#CFE9E0', '#B6DDD0', '#9DD1C0']      # pointwise, then three convs
DEC_D = '#2A7F69'
GREY_F, GREY_E = '#F1EFE9', '#8C8A83'
QC_F, QC_E = '#E3F1EE', '#12796B'
RED = '#C8442A'


# =====================================================================================
def load_signals():
    """Reconstruct the example recording with the released weights."""
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = P.EEGtoECGModel(latent_dim=P.SELECTED_LATENT_DIM,
                            decoder_schedule=P.SELECTED_DECODER, variant=P.SELECTED_VARIANT)
    model.load_state_dict(torch.load(DEPLOYED, map_location=dev))
    eeg, evt, _ = P.load_subject_data(os.path.join(P.DATA_FOLDER, SUBJECT))
    pred, true, _ = P.reconstruct_subject(model.eval().to(dev), eeg, dev)
    i0 = int(W0 * P.SFREQ); i1 = i0 + P.WINDOW_SIZE
    pk_true = np.where(evt[i0:i1] == 1)[0]
    pk_det = P.detect_peaks(pred)
    pk_det = pk_det[(pk_det >= i0) & (pk_det < i1)] - i0
    # show the channels that carry the heartbeat most visibly, plus a spread of others
    x = eeg[i0:i1, :128]
    hw = int(0.12 * P.SFREQ)
    allpk = np.where(evt == 1)[0]
    allpk = allpk[(allpk > hw) & (allpk < len(eeg) - hw)]
    tmpl = np.mean([eeg[p - hw:p + hw + 1, :128] for p in allpk], axis=0)
    p2p = tmpl.max(0) - tmpl.min(0)
    top = list(np.argsort(p2p)[::-1][:4])
    rest = [c for c in np.linspace(0, 127, 12).astype(int) if c not in top][:7]
    chans = sorted(top + rest)
    zs = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    return (x[:, chans], zs(pred[i0:i1]), zs(true[i0:i1]), pk_true, pk_det, chans)


# =====================================================================================
def box(ax, x, y, w, h, face, edge, lw=0.7, r=1.6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=face, ec=edge, lw=lw, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.8, ls='-', z=3, ms=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def slab(ax, x, yc, w, h, face, edge, dx=4.2, dy=2.8, lw=0.6, z=4):
    """A block drawn as a cuboid: front face h tall (channels), oblique depth (time)."""
    y0 = yc - h / 2
    top = [(x, y0 + h), (x + dx, y0 + h + dy), (x + w + dx, y0 + h + dy), (x + w, y0 + h)]
    side = [(x + w, y0), (x + w + dx, y0 + dy), (x + w + dx, y0 + h + dy), (x + w, y0 + h)]
    front = [(x, y0), (x + w, y0), (x + w, y0 + h), (x, y0 + h)]
    lighten = lambda c, f: tuple(min(1, v + (1 - v) * f) for v in plt.matplotlib.colors.to_rgb(c))
    darken = lambda c, f: tuple(v * (1 - f) for v in plt.matplotlib.colors.to_rgb(c))
    for poly, fc in ((side, darken(face, 0.14)), (top, lighten(face, 0.35)), (front, face)):
        ax.add_patch(Polygon(poly, closed=True, fc=fc, ec=edge, lw=lw, zorder=z,
                             joinstyle='round'))
    return y0, y0 + h


height = lambda c: 3.0 + 1.32 * np.sqrt(c)          # mm, front-face height for c channels


# =====================================================================================
def panel_overview(ax):
    ax.set_xlim(0, 183); ax.set_ylim(0, 50); ax.set_aspect('equal'); ax.axis('off')
    Y1, Y2, H = 33.0, 9.5, 15.0                    # lane centres and box height

    # lane headers
    for y, t in ((Y1 + H / 2 + 1.6, 'VALIDATION'), (Y2 + H / 2 + 1.6, 'DEPLOYMENT')):
        ax.text(2, y, t, fontsize=6.3, color=INK3, fontweight='bold', va='bottom')

    # data sources
    box(ax, 2, Y1 - H / 2, 38, H, GREY_F, GREY_E)
    ax.text(21, Y1 + 2.9, '66 adults', ha='center', fontsize=7.5, fontweight='bold', color=INK)
    ax.text(21, Y1 - 0.4, '128-channel EEG + ECG', ha='center', fontsize=6.5, color=INK2)
    ax.text(21, Y1 - 4.2, 'train 46 · selection 10 · testing 10', ha='center', fontsize=6.0,
            color=INK3)
    box(ax, 2, Y2 - H / 2, 38, H, GREY_F, GREY_E)
    ax.text(21, Y2 + 2.9, 'Healthy Brain Network', ha='center', fontsize=7.5,
            fontweight='bold', color=INK)
    ax.text(21, Y2 - 0.4, '3,297 recordings, no ECG', ha='center', fontsize=6.5, color=INK2)
    ax.text(21, Y2 - 4.2, 'ages 5–22 years', ha='center', fontsize=6.0, color=INK3)

    # shared model and gate, spanning both lanes
    top, bot = Y1 + H / 2, Y2 - H / 2
    box(ax, 52, bot, 34, top - bot, '#E4EDF7', BLUE, lw=0.9)
    ax.text(69, (top + bot) / 2 + 5.0, 'Reconstruction', ha='center', fontsize=7.8,
            fontweight='bold', color=BLUE_D)
    ax.text(69, (top + bot) / 2 + 0.8, 'dilated convolutional', ha='center', fontsize=6.4,
            color=INK2)
    ax.text(69, (top + bot) / 2 - 2.6, 'network (panel b)', ha='center', fontsize=6.4, color=INK2)
    ax.text(69, (top + bot) / 2 - 7.6, 'trained once, then frozen', ha='center', fontsize=5.9,
            color=INK3, style='italic')

    box(ax, 98, bot, 34, top - bot, QC_F, QC_E, lw=0.9)
    ax.text(115, (top + bot) / 2 + 5.0, 'Reference-free', ha='center', fontsize=7.8,
            fontweight='bold', color=QC_E)
    ax.text(115, (top + bot) / 2 + 1.4, 'quality-control gate', ha='center', fontsize=7.8,
            fontweight='bold', color=QC_E)
    ax.text(115, (top + bot) / 2 - 2.9, 'R–R interval plausibility', ha='center', fontsize=6.4,
            color=INK2)
    ax.text(115, (top + bot) / 2 - 7.6, 'calibrated on selection split', ha='center',
            fontsize=5.9, color=INK3, style='italic')

    # endpoints
    box(ax, 144, Y1 - H / 2, 37, H, 'white', GREY_E)
    ax.text(162.5, Y1 + 2.9, 'Against true ECG', ha='center', fontsize=7.5,
            fontweight='bold', color=INK)
    ax.text(162.5, Y1 - 0.4, 'R-peak accuracy (testing)', ha='center', fontsize=6.3, color=INK2)
    ax.text(162.5, Y1 - 4.0, 'HEP: true vs reconstructed', ha='center', fontsize=6.3, color=INK2)
    box(ax, 144, Y2 - H / 2, 37, H, 'white', GREY_E)
    ax.text(162.5, Y2 + 2.9, 'HEP at scale', ha='center', fontsize=7.5, fontweight='bold',
            color=INK)
    ax.text(162.5, Y2 - 0.4, '2,066 recordings', ha='center', fontsize=6.3, color=INK2)
    ax.text(162.5, Y2 - 4.0, 'agreement with adults, by age', ha='center', fontsize=6.3,
            color=INK2)

    for y in (Y1, Y2):
        arrow(ax, 40.4, y, 51.6, y)
        arrow(ax, 86.4, y, 97.6, y)
        arrow(ax, 132.4, y, 143.6, y)

    # ground truth is used only for evaluation, never as an input
    yg = top + 5.6
    xg = 31.0                                      # clear of the lane header
    ax.plot([xg, xg], [top + 0.3, yg], color=RED, lw=0.7, ls=(0, (2.4, 1.6)), zorder=1)
    ax.plot([xg, 162.5], [yg, yg], color=RED, lw=0.7, ls=(0, (2.4, 1.6)), zorder=1)
    arrow(ax, 162.5, yg, 162.5, top + 0.4, color=RED, lw=0.7, ls=(0, (2.4, 1.6)))
    ax.text(92, yg + 0.7, 'ground-truth ECG: evaluation only, never a model input',
            ha='center', va='bottom', fontsize=6.0, color=RED)


# =====================================================================================
def panel_network(ax, sig):
    eeg, yhat, ytrue, pk_t, pk_d, chans = sig
    ax.set_xlim(0, 183); ax.set_ylim(0, 58); ax.set_aspect('equal'); ax.axis('off')
    YC = 31.0                                      # baseline through the block centres
    t = np.linspace(0, 1, len(yhat))

    # ---- input: real EEG window --------------------------------------------------
    X0, XW, EH = 1.5, 30.0, 34.0
    n = eeg.shape[1]
    for k in range(n):
        yk = YC + EH / 2 - (k + 0.5) * EH / n
        ax.plot(X0 + t * XW, yk + 0.36 * np.clip(eeg[:, k], -4, 4) * EH / n / 2,
                color=INK2, lw=0.35, zorder=3)
    ax.text(X0 + XW / 2, YC + EH / 2 + 2.2, 'EEG input', ha='center', fontsize=7.2,
            fontweight='bold', color=INK)
    ax.text(X0 + XW / 2, YC - EH / 2 - 2.6, f'128 channels × 2 s  ({n} shown)', ha='center',
            fontsize=5.9, color=INK3)

    # ---- blocks --------------------------------------------------------------------
    blocks = []                                    # (x, channels, face, edge, below-label)
    x = 38.0
    # The positional embedding is not drawn: it remained at its random initialisation
    # during training (r = 0.995) and the trained model does not use it (Methods).
    blocks.append((x, 128, '#E4EDF7', BLUE, 'linear')); x += 11.0
    enc_x0 = x
    for i, d in enumerate((2, 4, 8, 16, 32, 64)):
        c = 256 if i == 5 else 128
        blocks.append((x, c, ENC[i], BLUE_D, f'd={d}')); x += 5.3
    blocks.append((x, 128, ENC[2], BLUE_D, '1×1')); x += 5.3
    enc_x1 = x - 1.0
    x += 6.0
    # decoder: one monotonic funnel 128 -> 32 -> 16 -> 8 -> 4 -> 1. The first stage is
    # a pointwise (1x1) linear layer with GELU; it is not a bottleneck, since nothing
    # after it re-expands, so it is drawn as part of the decoder rather than as a latent.
    dec_x0 = x
    blocks.append((x, 32, DEC[0], DEC_D, 'point-\nwise')); x += 5.6
    for i, c in enumerate((16, 8, 4)):
        blocks.append((x, c, DEC[i + 1], DEC_D, 'k=3' if i == 1 else '')); x += 5.0
    blocks.append((x, 1, DEC[3], DEC_D, '1×1')); x += 4.6
    dec_x1 = x - 1.0

    W_ = 2.4
    for bx, c, fc, ec, lab in blocks:
        y0, y1 = slab(ax, bx, YC, W_, height(c), fc, ec)
        if lab:
            ax.text(bx + W_ / 2 + 1.0, YC - height(256) / 2 - 3.2, lab, ha='center', va='top',
                    fontsize=5.4, color=INK3, linespacing=1.0)

    # group labels with brackets
    def group(xa, xb, text, color, sub=None):
        ytop = YC + height(256) / 2 + 5.4
        ax.plot([xa, xa, xb, xb], [ytop - 1.0, ytop, ytop, ytop - 1.0], color=color, lw=0.6)
        ax.text((xa + xb) / 2, ytop + 1.0, text, ha='center', va='bottom', fontsize=6.8,
                fontweight='bold', color=color)
        if sub:
            ax.text((xa + xb) / 2, ytop + 4.3, sub, ha='center', va='bottom', fontsize=5.6,
                    color=INK3)

    group(38.0, 38.0 + W_ + 4.2, 'Projection', BLUE)
    group(enc_x0, enc_x1 + 3.2, 'Dilated convolutional encoder', BLUE_D,
          'k = 5 · receptive field 1.97 s')
    group(dec_x0, dec_x1 + 3.2, 'Decoder', DEC_D, '128 → 32 → 16 → 8 → 4 → 1')

    # tensor shapes under the groups
    ys = YC - height(256) / 2 - 9.6
    for xm, s in ((38.0 + 3.0, '512×128'), ((enc_x0 + enc_x1) / 2 + 1.5, '512×128  (peak 512×256)'),
                  ((dec_x0 + dec_x1) / 2 + 1.5, '512×32 → 512×1')):
        ax.text(xm, ys, s, ha='center', va='top', fontsize=5.6, color=INK2)

    # arrows between stages
    arrow(ax, X0 + XW + 0.8, YC, 37.4, YC, color=INK3, lw=0.7)
    arrow(ax, dec_x1 + 4.4, YC, dec_x1 + 9.0, YC, color=INK3, lw=0.7)

    # ---- output: real reconstruction against recorded ECG -----------------------------
    OX, OW, OH = dec_x1 + 9.8, 183 - (dec_x1 + 9.8) - 1.5, 17.0
    sc = OH / 2 / max(np.abs(ytrue).max(), np.abs(yhat).max())
    ax.plot(OX + t * OW, YC + ytrue * sc, color=INK, lw=0.6, zorder=3)
    ax.plot(OX + t * OW, YC + yhat * sc, color=RED, lw=0.8, ls=(0, (3.0, 1.6)), zorder=4)
    ybase = YC - OH / 2 - 2.2
    for p in pk_t:
        ax.plot([OX + p / len(yhat) * OW] * 2, [ybase, ybase + 1.8], color=INK, lw=0.8)
    for p in pk_d:
        ax.plot(OX + p / len(yhat) * OW, ybase - 1.4, marker='^', ms=2.6, color=RED,
                mec='none', zorder=5)
    ax.text(OX + OW / 2, YC + height(256) / 2 + 6.4, 'Reconstructed ECG', ha='center',
            va='bottom', fontsize=6.8, fontweight='bold', color=RED)
    ax.text(OX + OW / 2, YC + height(256) / 2 + 3.1, r'$\hat{y}$ (red) vs recorded ECG (black)',
            ha='center', va='bottom', fontsize=5.6, color=INK3)
    ax.text(OX + OW / 2, ybase - 4.4, 'R-peaks: recorded  |   detected ▲  (post hoc)',
            ha='center', va='top', fontsize=5.6, color=INK3)

    # ---- the time axis is preserved end to end --------------------------------------
    yt = 1.6
    ax.add_patch(FancyArrowPatch((38.0, yt), (dec_x1 + 4.0, yt), arrowstyle='<|-|>',
                                 mutation_scale=5, color=INK3, lw=0.6))
    ax.text((38.0 + dec_x1 + 4.0) / 2, yt + 1.0,
            'time preserved at every stage: 512 samples (2 s), no downsampling',
            ha='center', va='bottom', fontsize=5.9, color=INK2)


# =====================================================================================
def main():
    fs.use()
    sig = load_signals()
    print('channels shown:', sig[5], ' true peaks:', len(sig[3]), ' detected:', len(sig[4]))

    fig = fs.figure(width='double', height_mm=122.0)
    ax_a = fig.add_axes([0.0, 0.535, 1.0, 0.445])
    ax_b = fig.add_axes([0.0, 0.0, 1.0, 0.515])
    panel_overview(ax_a)
    panel_network(ax_b, sig)
    fig.text(0.006, 0.985, 'a', fontsize=9, fontweight='bold', va='top')
    fig.text(0.006, 0.525, 'b', fontsize=9, fontweight='bold', va='top')

    for d in OUTDIRS:
        os.makedirs(d, exist_ok=True)
        fs.save(fig, 'fig_overview', outdir=d, formats=('pdf', 'png'), verbose=True)
    fs.check_text_sizes(fig, floor_pt=5.0)


if __name__ == '__main__':
    main()
