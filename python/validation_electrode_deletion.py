# -*- coding: utf-8 -*-
"""
Supplementary Table 4: reconstruction under electrode deletion.

On the six retained held-out recordings with the strongest cardiac field, electrodes
are ranked by the cardiac field's scalp distribution (the leading spatial component
of the R-peak-triggered EEG average, computed from the recorded R-peaks and raw EEG,
independently of the model). The strongest, the weakest or k random electrodes
(k = 4, 8, 16, 32, 64; six random draws averaged) are zeroed at the input and the
per-window reconstruction r is recomputed over 60 windows (2 min). Single-electrode
importance zeroes each electrode in turn.

    python validation_electrode_deletion.py --prepare   # GPU and the adult recordings; caches the sweep
    python validation_electrode_deletion.py             # prints the table from the cache

P values are two-sided Wilcoxon signed-rank tests across recordings.

@author: Matias E. Fraile-Vazquez
"""
import argparse
import os

import numpy as np
from scipy import stats as sps

from network_activations import N_STAT, RUN_DIR, SFREQ, WINDOW, load_subject_table

CACHE_NAME = 'electrode_deletion_inputs.npz'
N_TOPO_SUBJ = 6               # highest-leakage retained recordings
ABL_KS = (4, 8, 16, 32, 64)   # group sizes for the electrode-deletion curve
N_RAND = 6                    # random draws averaged per group size


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


def prepare(run_dir, data_folder=None, cache_path=None):
    """Run the deletion sweep and the single-electrode occlusion, and cache them."""
    import torch
    import hdeeg2ecg_model as P

    data_folder = data_folder or P.DATA_FOLDER
    cache_path = cache_path or os.path.join(run_dir, CACHE_NAME)

    df = load_subject_table(run_dir)
    retained = df[df['gate_pass']]
    topo_subj = retained.nlargest(N_TOPO_SUBJ, 'leak_true')
    print(f"  recordings: {', '.join('sub-%d' % i for i in topo_subj['id'])}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = P.EEGtoECGModel(latent_dim=P.SELECTED_LATENT_DIM,
                          decoder_schedule=P.SELECTED_DECODER,
                          variant=P.SELECTED_VARIANT)
    net.load_state_dict(torch.load(P.WEIGHTS_PATH,
                                   map_location='cpu'))
    net.eval().to(device)

    out = {}
    imps, cfas = [], []
    abl_base, abl_top, abl_rand, abl_low = [], [], [], []
    for _, row in topo_subj.iterrows():
        e, v, _ = P.load_subject_data(os.path.join(data_folder, str(row['file'])))
        nw = min(N_STAT, len(e) // WINDOW)
        w = e[:nw * WINDOW].reshape(nw, WINDOW, 129).astype(np.float32).copy()
        ecg = w[:, :, 128]
        tgt_w = (ecg - ecg.mean(1, keepdims=True)) / (ecg.std(1, keepdims=True) + 1e-12)
        w[:, :, 128] = 0.0
        cfa = cfa_topography(e, v)
        print(f"  sweeps, sub-{row['id']} ({nw} windows)")
        imps.append(channel_importance(net, w, tgt_w, device=device, verbose=False))
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

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **out)
    print(f'\n  cached -> {cache_path}')
    return cache_path


def sem(a):
    return a.std(0, ddof=1) / np.sqrt(len(a))


def report(cache):
    z = np.load(cache)
    ks, base = z['abl_ks'], z['abl_base']
    top, rnd, low = z['abl_top'], z['abl_rand'], z['abl_low']

    print(f'recordings: {len(base)}   full montage r = {base.mean():.3f} +/- {sem(base):.3f}')
    print(f'largest drop from deleting any single electrode: {z["importance"].max():.3f}')
    print('\nk & strongest field & random & weakest field & P vs random & P vs weakest')
    for i, k in enumerate(ks):
        cells = ' & '.join(f'${a[:, i].mean():.3f} \\pm {sem(a)[i]:.3f}$' for a in (top, rnd, low))
        p_rnd = sps.wilcoxon(top[:, i], rnd[:, i]).pvalue
        p_low = sps.wilcoxon(top[:, i], low[:, i]).pvalue
        print(f'${k}$ & {cells} & ${p_rnd:.2g}$ & ${p_low:.2g}$ \\\\')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run-dir', default=RUN_DIR)
    ap.add_argument('--data-folder', default=None)
    ap.add_argument('--cache', default=None)
    args = ap.parse_args()
    cache = args.cache or os.path.join(args.run_dir, CACHE_NAME)
    if args.prepare:
        prepare(args.run_dir, args.data_folder, cache)
    report(cache)


if __name__ == '__main__':
    main()
