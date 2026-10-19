# -*- coding: utf-8 -*-
"""
Supplementary Table 4: reconstruction under electrode deletion.

network_activations.py --prepare runs the deletion sweep (group_ablation) and the
single-electrode occlusion map on the six retained held-out recordings with the
strongest cardiac field, and caches the results in network_activations_inputs.npz.
This script reads that cache and prints the table rows and the numbers quoted in
the Results. P values are two-sided Wilcoxon signed-rank tests across recordings.

    python validation_electrode_deletion.py [--cache path/to/network_activations_inputs.npz]

@author: Matias E. Fraile-Vazquez
"""
import argparse
import os

import numpy as np
from scipy import stats as sps

from network_activations import CACHE_NAME
from summary_figure import RUN_DIR


def sem(a):
    return a.std(0, ddof=1) / np.sqrt(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(RUN_DIR, CACHE_NAME))
    z = np.load(ap.parse_args().cache)
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


if __name__ == '__main__':
    main()
