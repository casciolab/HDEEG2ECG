# -*- coding: utf-8 -*-
"""Merge the HBN participants release files onto toolbox_qc_summary.csv.

Discovers participants*.csv by glob, so dropping in a new release file and
re-running is all that is needed. Two things this handles that a plain
pd.concat/merge does not:

  * the release files are not all the same format -- participants.csv is
    comma-delimited and the numbered ones are tab-delimited -- so the delimiter is
    sniffed per file rather than assumed;
  * toolbox_qc_summary.csv keys on Subject, which carries a task suffix
    (sub-XXXX_task-RestingState_eeg); participant_id is the bare sub-XXXX.

Left join on the QC table: every QC subject is kept whether or not demographics
exist for it, because the denominator of any pass-rate analysis is the QC cohort,
not the subset that happens to have a demographics row.

    python hbn_merge_demographics.py
"""
import glob
import os
import re

import pandas as pd

# HBN folders: the participants*.csv release files go in inputs/hbn/; the toolbox's
# work folder is outputs/deployment/hbn/toolbox/; everything derived is written to
# outputs/deployment/hbn/ (deploy_step8_hep_by_age.m reads it from there).
INPUT_DIR = os.environ.get('HDEEG2ECG_INPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs'))
HBN_DIR = os.path.join(os.environ.get('HDEEG2ECG_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')),
                       'deployment', 'hbn')
QC = os.path.join(HBN_DIR, 'toolbox', 'toolbox_qc_summary.csv')
OUT = os.path.join(HBN_DIR, 'toolbox_qc_with_demographics.csv')
UNMATCHED = os.path.join(HBN_DIR, 'toolbox_qc_unmatched.csv')
TAB = '\t'



def safe_to_csv(df, path):
    """Write, or say plainly why not.

    These outputs get opened in Excel, which takes an exclusive lock on Windows;
    the bare pandas failure is a twenty-line traceback ending in PermissionError,
    which reads like a bug rather than like a file being open. Fall back to a
    sidecar file so the run is not wasted, and name the fix.
    """
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.replace('.csv', '.NEW.csv')
        df.to_csv(alt, index=False)
        print(f'  !! {path} is locked (open in Excel?) -- wrote {alt} instead.')
        print(f'     Close it and re-run, or rename {alt} over it.')
        return alt


def release_sort_key(path):
    m = re.search(r'participants(\d*)\.csv$', os.path.basename(path))
    return int(m.group(1)) if m and m.group(1) else 1


def read_release(path):
    with open(path, encoding='utf-8-sig') as fh:
        sep = TAB if TAB in fh.readline() else ','
    d = pd.read_csv(path, sep=sep, encoding='utf-8-sig', dtype={'participant_id': str})
    d['source_file'] = os.path.basename(path)
    return d, ('TAB' if sep == TAB else 'comma')


def main():
    os.makedirs(HBN_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, 'hbn', 'participants*.csv')), key=release_sort_key)
    if not files:
        raise SystemExit(f"no participants*.csv found in {os.path.join(INPUT_DIR, 'hbn')}")

    parts = []
    for f in files:
        d, label = read_release(f)
        parts.append(d)
        rel = sorted(d.release_number.dropna().astype(str).unique())
        print(f'  {f:20s} sep={label:5s} {len(d):4d} rows  releases={rel}')

    dem = pd.concat(parts, ignore_index=True)
    n_dup = dem.participant_id.duplicated().sum()
    print(f'\ndemographics: {len(dem)} rows, {dem.participant_id.nunique()} unique ids')
    if n_dup:
        print(f'  {n_dup} duplicate ids across releases -> keeping first by release order')
        dem = dem.drop_duplicates('participant_id', keep='first')

    qc = pd.read_csv(QC, dtype={'Subject': str})
    qc['participant_id'] = qc.Subject.str.replace(r'_task-.*$', '', regex=True)
    print(f'QC subjects: {len(qc)} rows, {qc.participant_id.nunique()} unique ids')

    m = qc.merge(dem.drop(columns=['source_file']), on='participant_id',
                 how='left', indicator=True)
    hit = int((m._merge == 'both').sum())
    print(f'\nmatched {hit}/{len(m)} ({100*hit/len(m):.1f}%), {len(m)-hit} unmatched')
    print('\nmatch by QC decision:')
    print(m.groupby('Decision')._merge.value_counts().unstack(fill_value=0).to_string())

    mm = m[m._merge == 'both']
    print(f'\nmatched: age present {mm.age.notna().sum()}/{len(mm)}, '
          f'sex present {mm.sex.notna().sum()}/{len(mm)}')
    print(f'  age median {mm.age.median():.1f} (range {mm.age.min():.1f}-{mm.age.max():.1f})')
    print(f'  sex {mm.sex.value_counts().to_dict()}')
    good = m[m.Decision == 'GOOD']
    print(f'  QC-passing (GOOD): {len(good)}, with demographics {good.age.notna().sum()}')

    m.drop(columns=['_merge']).to_csv(OUT, index=False)
    print(f'\nwrote {OUT} ({len(m)} rows x {m.shape[1]-1} cols)')
    un = m.loc[m._merge == 'left_only', ['Subject', 'participant_id', 'Decision']]
    if len(un):
        print(f'wrote {safe_to_csv(un, UNMATCHED)} ({len(un)} rows)')


if __name__ == '__main__':
    main()
