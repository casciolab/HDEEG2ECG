# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:58:49 2026

@author: Matias E. Fraile-Vazquez
"""

# -*- coding: utf-8 -*-
"""
Derive reference-free QC pass-thresholds for frac_short and frac_implaus PRINCIPLEDLY, then
report them honestly out-of-sample.

Why this script exists: the interval BOUNDS (400 ms; [500,1300] ms) are physiological and fixed
a priori, but the PASS THRESHOLDS on the resulting fractions (how many implausible intervals we
tolerate) are a decision, not a physiological constant. This picks them by a single stated rule
on the SELECTION split only, freezes them, and evaluates the frozen gate on the held-out TESTING
split. rho does NOT set a threshold (it is unitless and says nothing about where to cut); it only
justifies WHICH features to gate on. F1 is used ONLY to validate the gate, never to fit it.

Rule B (ADOPTED, as reported in the paper): midpoint of the empirical gap between good and
failed recordings on the selection split (requires a clean gap; uses selection-split F1 only).

Rule A (reported alongside for comparison, not adopted): threshold = Qth percentile of the
feature among ALL selection recordings, a label-free alternative.

Reads rhythm_qc.csv (from qc_rhythm_features.py). Pure re-analysis; no model, no GPU.
Run:  python qc_gate_thresholds.py
"""
import os
import numpy as np
import pandas as pd

import hdeeg2ecg_model as P

FEATURES = ['frac_short', 'frac_implaus']   # the two features chosen to gate on (strong rho + clean pooled sep)
PERCENTILE = 95.0                            # Rule A knob: tolerate ~ (100-Q)% flagged among selection recordings
FAIL_F1 = 0.60                               # F1 below this = failure (for VALIDATION only)
SEL, TEST = 'val1', 'val2'                   # selection / testing split labels in split_manifest.csv / rhythm_qc.csv


def confusion(passed, is_good):
    """Gate decisions vs F1 truth. passed/is_good are boolean arrays over subjects."""
    passed, is_good = np.asarray(passed), np.asarray(is_good)
    tp_good = int(( passed &  is_good).sum())   # good recording correctly passed
    fp_flag = int((~passed &  is_good).sum())   # good recording wrongly flagged (false alarm)
    fn_pass = int(( passed & ~is_good).sum())   # failed recording wrongly passed (missed -- the dangerous one)
    tn_flag = int((~passed & ~is_good).sum())   # failed recording correctly flagged
    n_good, n_bad = int(is_good.sum()), int((~is_good).sum())
    return {
        'passed_total': int(passed.sum()), 'flagged_total': int((~passed).sum()),
        'good_passed': tp_good, 'good_flagged(false_alarm)': fp_flag,
        'failed_passed(missed)': fn_pass, 'failed_flagged(caught)': tn_flag,
        'flag_rate_on_good': fp_flag / n_good if n_good else np.nan,
        'catch_rate_on_failed': tn_flag / n_bad if n_bad else np.nan,
    }


def main():
    df = pd.read_csv(os.path.join(P.OUTPUT_DIR, 'rhythm_qc.csv'))
    df['is_good'] = df['f1'] >= FAIL_F1
    sel, test = df[df.split == SEL], df[df.split == TEST]
    if len(sel) == 0 or len(test) == 0:
        raise SystemExit(f"Need both splits. Found splits: {sorted(df.split.unique())}; "
                         f"set SEL/TEST accordingly.")

    print(f"Selection split ({SEL}): n={len(sel)} "
          f"({int(sel.is_good.sum())} good, {int((~sel.is_good).sum())} failed)")
    print(f"Testing   split ({TEST}): n={len(test)} "
          f"({int(test.is_good.sum())} good, {int((~test.is_good).sum())} failed)\n")

    thresholds = {}
    for feat in FEATURES:
        # ---- Rule A: label-free percentile on ALL selection recordings (comparison only) ----
        thr_A = float(np.percentile(sel[feat].values, PERCENTILE))

        # ---- Rule B: midpoint of good/failed gap on selection (ADOPTED) ----
        g = sel.loc[sel.is_good, feat].values
        b = sel.loc[~sel.is_good, feat].values
        if len(g) and len(b):
            gap_lo, gap_hi = float(g.max()), float(b.min())     # good below, failed above (oriented so higher=worse)
            thr_B = 0.5 * (gap_lo + gap_hi)
            clean = gap_hi > gap_lo
        else:
            thr_B, gap_lo, gap_hi, clean = np.nan, np.nan, np.nan, False

        thresholds[feat] = thr_B
        print(f"[{feat}]")
        print(f"  Rule A  (P{PERCENTILE:.0f} of all selection, label-free)  -> threshold = {thr_A:.4f}")
        print(f"  Rule B  (midpoint of good/failed gap on selection)        -> threshold = {thr_B:.4f}   [ADOPTED]"
              f"   (good.max={gap_lo:.4f}, failed.min={gap_hi:.4f}, {'clean gap' if clean else 'NO clean gap'})")
        # sanity: where does the adopted threshold sit relative to the selection classes?
        if len(g) and len(b):
            print(f"  selection good : max={g.max():.4f}  (all should be <= threshold for a good gate)")
            print(f"  selection fail : min={b.min():.4f}  (all should be >  threshold)")
        print()

    # ============================================================
    # FREEZE the Rule-B thresholds; evaluate the joint gate (pass iff BOTH satisfied)
    # ============================================================
    def gate_pass(frame):
        ok = np.ones(len(frame), dtype=bool)
        for feat in FEATURES:
            ok &= frame[feat].values <= thresholds[feat]
        return ok

    print("=" * 70)
    print("FROZEN joint gate:  pass  iff  " +
          "  AND  ".join(f"{f} <= {thresholds[f]:.4f}" for f in FEATURES))
    print("=" * 70)

    for name, frame in [(f'SELECTION ({SEL}) [in-sample: derivation set]', sel),
                        (f'TESTING ({TEST}) [OUT-OF-SAMPLE: the honest result]', test),
                        ('POOLED (both splits)', df)]:
        c = confusion(gate_pass(frame), frame.is_good.values)
        print(f"\n{name}:")
        for k, v in c.items():
            print(f"    {k:28s} {v:.3f}" if isinstance(v, float) else f"    {k:28s} {v}")

    # per-subject decisions for the table / appendix
    df['gate_pass'] = gate_pass(df)
    out = df[['id', 'split', 'f1', 'is_good'] + FEATURES + ['gate_pass']].sort_values(['split', 'f1'])
    out.to_csv(os.path.join(P.OUTPUT_DIR, 'qc_gate_decisions.csv'), index=False)
    print(f"\nwrote qc_gate_decisions.csv to {P.OUTPUT_DIR}")
    print("\nReport in the paper: thresholds derived by the gap-midpoint rule on the selection split, "
          "frozen, and\nevaluated on testing -- quote the TESTING catch_rate_on_failed and "
          "flag_rate_on_good as the\nout-of-sample result. Do NOT quote selection performance as "
          "evidence (it is in-sample).")


if __name__ == '__main__':
    main()