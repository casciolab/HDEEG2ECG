# -*- coding: utf-8 -*-
"""
network_activations.py -- per-stage activations and beat-locked modulation in the
deployed network, and the per-recording results table.

Imported by fig_network_emergence.py (Fig. 3) and validation_electrode_deletion.py
(Supplementary Table 4); it is not run on its own.

@author: Matias E. Fraile-Vazquez
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

# Same directory hdeeg2ecg_model.OUTPUT_DIR writes to, kept as a plain constant so
# that loading the results table does not import torch.
RUN_DIR = os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'))

SFREQ = 256
WINDOW = 512
SUBJ_RE = re.compile(r'8\d{3}')

N_STAT = 60       # windows the modulation statistic is computed over (2 min)
HALF_WIN = 40     # +/-156 ms around each R-peak, for the modulation statistic
N_PERM = 200

# name, label, tensor width -- in forward order.
STAGES = [
    ('eeg',     'EEG input',        128),
    ('encoder', 'dilated encoder',  128),
    ('latent',  'latent',            32),
    ('dec16',   'decoder',           16),
    ('dec8',    'decoder',            8),
    ('dec4',    'decoder',            4),
]


def subject_id(name):
    return int(SUBJ_RE.search(os.path.basename(str(name))).group())


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


def _receptive_field_samples(kernel=5, dilations=(2, 4, 8, 16, 32, 64)):
    return 1 + sum((kernel - 1) * d for d in dilations)
