# -*- coding: utf-8 -*-
"""
EEG-to-ECG Inference Toolbox  (3-stage, pure Python)
=====================================================
Stage 1  RECONSTRUCT   EEG (128ch) -> ECG, saved as a 129-channel .set (ECG = channel 129)
Stage 2  MARK PEAKS    on the reconstructed ECG, written as events INTO each .set:
                         - Pipeline detect_peaks   (automatic; the frac-VALIDATED detector)
                         - Manual editor           (interactive R-DECO-style: R-DECO seed + Add/
                                                    Adjust(drag)/Delete + RR tachogram + MARK BAD)
Stage 3  DISCRIMINATE  read in-file peaks -> frac_short / frac_implaus -> GOOD / BAD

No MATLAB. The manual editor reproduces R-DECO's correction workflow and layout (linked RR
tachogram). R-DECO's detector itself (peak_detection.m) is now ported into this file and used as
the editor seed and the "R-DECO" re-detect button: adaptive_thresholding, peaks_in_ecg,
post_processing and the minute-segmentation driver are exact ports; env_secant and calc_pos_opt
are standard reconstructions (those two helpers are referenced by peak_detection.m but were never
in any shared file) -- drop in the real .m for a byte-exact match.

@author: Matias E. Fraile-Vazquez
"""

import os
import sys
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import numpy as np
try:
    import torch
    _HAS_TORCH = True
except Exception:           # rating-only build: torch absent -> Stage 1 disabled, editor still works
    torch, _HAS_TORCH = None, False
import mne
import pandas as pd
import warnings
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION  -- edit for your machine
# ============================================================
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(_REPO, 'python')
WEIGHTS_PATH = os.environ.get('HDEEG2ECG_WEIGHTS', os.path.join(_REPO, 'weights', 'final_model_weights.pt'))

DEVICE = 'cuda' if (_HAS_TORCH and torch.cuda.is_available()) else 'cpu'
P = None
try:                        # the model pipeline is only needed for Stage 1 (reconstruction)
    sys.path.insert(0, PIPELINE_DIR)
    import hdeeg2ecg_model as P   # noqa: E402
except Exception:
    pass
SFREQ = getattr(P, 'SFREQ', 256)

REC_DIR, MARK_DIR, GOOD_DIR, BAD_DIR = '01_reconstructed', '02_marked', '03_good', '03_bad'

# ============================================================
# PEAK DETECTORS
# ============================================================
# ------------------------------------------------------------
# R-DECO detector  --  faithful port of peak_detection.m
#   exact ports : adaptive_thresholding, peaks_in_ecg, post_processing,
#                 minute-segmentation driver
#   reconstructed: env_secant, calc_pos_opt  (referenced by peak_detection.m
#                 but never defined in any shared file; standard implementations
#                 matching R-DECO's documented behaviour -- swap in the real .m
#                 files for a byte-exact match)
#   defaults Rpeak{1..5} = {300,100,1,0,0} -> env_ms=300, avgHR=100, postproc on,
#   ectopics off (skipped), not inverted.
# ------------------------------------------------------------
def _env_secant(x, y, view, side):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(y)
    if n < 2:
        return y.copy()
    Vx = [0]
    while Vx[-1] < n - 1:
        v = Vx[-1]; last = min(v + int(view), n - 1)
        if last <= v:
            Vx.append(n - 1); break
        idxs = np.arange(v + 1, last + 1)
        slopes = (y[idxs] - y[v]) / (x[idxs] - x[v])
        k = int(np.argmax(slopes)) if side == 'top' else int(np.argmin(slopes))
        Vx.append(v + 1 + k)
    Vx = np.array(sorted(set(Vx)))
    return np.interp(x, x[Vx], y[Vx])


def _calc_pos_opt(signal, step, typ=1):
    s = np.asarray(signal, float); n = len(s); step = max(1, int(step))
    if n < 3:
        return np.array([], dtype=int)
    pos = [i for i in range(1, n - 1)
           if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] > s[max(0, i - step)]]
    return np.array(pos, dtype=int)


def _adaptive_thresholding(peaks, signal, fs, avgHR):
    peaks = np.asarray(peaks, dtype=int); sig = np.asarray(signal, float); npk = len(peaks)
    if npk < 3:
        return np.array([], int), np.array([], float)
    tt = lambda s: s / fs
    STD = sig.std()
    RR_avg = np.zeros(npk + 2); RR_avg[1] = 60000.0 / avgHR
    pv = sig[peaks]
    peak_avg = 0.7 * (np.median(pv) + np.mean(pv)) / 2.0
    noise_avg = 0.3 * (np.median(pv) + np.mean(pv)) / 2.0
    peak_threshold = noise_avg + 0.25 * (peak_avg - noise_avg); noise_threshold = peak_threshold / 2.0
    R = np.zeros(npk + 1, dtype=int); RR = np.zeros(npk + 1)
    count = 1
    while R[1] == 0:
        if sig[peaks[count - 1]] > noise_threshold:
            R[1] = peaks[count - 1]
        else:
            count += 1
            if count > npk:
                return np.array([], int), np.array([], float)
    pc = 1
    for idx in range(count + 1, npk + 1):
        peak = sig[peaks[idx - 1]]; cur = peaks[idx - 1]; prev = RR_avg[idx - 1]
        RR_lo = max(0.65 * prev, 200.0); RR_hi = min(1.35 * prev, 1600.0)
        if peak >= peak_threshold:
            if peak > 4 * STD:
                peak = 4 * STD
            peak_avg = 0.125 * peak + 0.875 * peak_avg
            pc += 1; R[pc] = cur; RR[pc - 1] = 1000.0 * (R[pc] - R[pc - 1]) / fs
        elif peak <= noise_threshold:
            noise_avg = 0.125 * peak + 0.875 * noise_avg
        else:
            ok = False
            if pc > 1 and (tt(cur) >= RR_lo / 1000.0 + tt(R[pc])) and (tt(cur) <= RR_hi / 1000.0 + tt(R[pc])):
                if idx == npk:
                    ok = True
                else:
                    nxt = sig[peaks[idx]]
                    if nxt < peak_threshold:
                        ok = True
                    elif nxt >= peak_threshold and (tt(peaks[idx]) - tt(cur)) > 0.2:
                        ok = True
            if ok:
                peak_avg = 0.25 * peak + 0.75 * peak_avg
                pc += 1; R[pc] = cur; RR[pc - 1] = 1000.0 * (R[pc] - R[pc - 1]) / fs
            else:
                noise_avg = 0.125 * peak + 0.875 * noise_avg
        if pc > 9:
            RR_avg[idx] = np.mean(RR[pc - 8:pc])
        elif pc > 1:
            RR_avg[idx] = 0.5 * RR_avg[idx - 1] + 0.5 * np.median(RR[1:pc])
        peak_threshold = noise_avg + 0.25 * (peak_avg - noise_avg); noise_threshold = peak_threshold / 2.0
    return R[1:pc + 1].astype(int), RR[1:pc]


def _peaks_in_ecg(signal, R, window):
    sig = np.asarray(signal, float); R = list(np.asarray(R, dtype=int)); window = int(window)
    for i in range(len(R)):
        a = max(0, R[i] - window); seg = sig[a:R[i] + 1]
        if len(seg) == 0:
            continue
        ip = int(np.argmax(seg)); loc = a + ip
        if 0 < loc < len(sig) - 1:
            if sig[loc] >= sig[loc - 1] and sig[loc] >= sig[loc + 1]:
                R[i] = loc
        else:
            R[i] = loc
    return np.array(R, dtype=int)


def _trimmean(x, pct):
    x = np.sort(np.asarray(x, float)); n = len(x); k = int(np.floor(n * pct / 200.0))
    return np.median(x) if 2 * k >= n else float(np.mean(x[k:n - k]))


def _post_processing(all_peaks, R_peak_segm, RR_int_segm, signal, fs):
    a, d, max_k = 0.3, 0.5, 6
    RR = [float(v) for v in RR_int_segm]; L = len(RR)
    large_prob = [0] * L
    R0 = int(R_peak_segm[0]); med = float(np.median(signal)); ap = np.asarray(all_peaks, int)
    RR_ref = _trimmean(RR[:5], 50)
    c = 0
    while c < L - 2:
        if c >= 2:
            k = 1; sum_ref = 0.0; num = 0; wght = [0.5, 0.3, 0.2]
            while (c - k) >= 0 and num < 3 and k < max_k:
                if large_prob[c - k] != 1:
                    sum_ref += wght[num] * RR[c - k]; num += 1
                k += 1
            if num <= 1:
                k = 1; sum_ref = 0.0; num = 0
                while (c - k) >= 0 and num < 3:
                    sum_ref += wght[num] * RR[c - k]; num += 1; k += 1
            if num > 0:
                RR_ref = sum_ref / sum(wght[:num])
        if RR[c] < (1 - a) * RR_ref or RR[c] < 200:
            counter1 = 1; counter2 = 0; test = 1
            RRnew1 = (RR[c] + RR[c - 1]) if c - 1 >= 0 else None
            sumRR = RR[c] + RR[c + counter1]; RRnew2 = sumRR
            while test == 1:
                if c + counter1 < L - 1:
                    if sumRR < (1 - a) * RR_ref or sumRR < 200:
                        counter1 += 1; sumRR += RR[c + counter1]
                    elif abs(sumRR - RR_ref) < a * RR_ref:
                        test2 = 1; counter2 += 1
                        while test2 == 1:
                            if c + counter1 + counter2 < L - 1:
                                sumRR2 = sumRR + RR[c + counter1 + counter2]
                                diff1 = abs(RR_ref - sumRR) / RR_ref; diff2 = abs(RR_ref - sumRR2) / RR_ref
                                if diff1 < diff2:
                                    test = 0; test2 = 0; RRnew2 = sumRR; counter2 -= 1
                                else:
                                    sumRR = sumRR2; counter2 += 1
                                    if c + counter1 + counter2 > L - 1:
                                        test = 0; test2 = 0; RRnew2 = sumRR2
                            else:
                                test = 0; test2 = 0; RRnew2 = sumRR
                    else:
                        test = 0; RRnew2 = sumRR - RR[c + counter1]; counter1 -= 1
                else:
                    test = 0; RRnew2 = sumRR
            if RRnew1 is not None and abs(RRnew1 - RR_ref) < abs(RRnew2 - RR_ref) and abs(RRnew1 - RR_ref) < a * RR_ref:
                RRnew = RRnew1; RR[c - 1] = RRnew1
                del RR[c]; del large_prob[c]; L = len(RR); c -= 1
            else:
                RRnew = RRnew2; RR[c] = RRnew2
                if counter1 >= 1:
                    del RR[c + 1:c + counter1 + counter2 + 1]
                    del large_prob[c + 1:c + counter1 + counter2 + 1]; L = len(RR)
            if c >= 0 and (abs(RRnew - RR_ref) / RR_ref > d or RRnew < 200):
                large_prob[c] = 1
        elif RR[c] > (1 + a) * RR_ref or RR[c] > 1600:
            RR_samples = [v * fs / 1000.0 for v in RR]
            cum = np.cumsum(RR_samples[:c + 1]) + R0
            begin = cum[-2] if c > 0 else R0; end = cum[-1]
            cand = ap[(ap > begin) & (ap < end)]
            test_peak = None
            if len(cand) > 1:
                RR_test = 1000.0 * (cand - begin) / fs
                test_peak = int(cand[int(np.argmin(np.abs(RR_test - RR_ref)))])
            elif len(cand) == 1:
                test_peak = int(cand[0])
            if (test_peak is not None and signal[test_peak] > med
                    and (test_peak - begin) / fs * 1000 > (1 - a) * RR_ref and (test_peak - begin) / fs * 1000 > 200
                    and (end - test_peak) / fs * 1000 > (1 - a) * RR_ref and (end - test_peak) / fs * 1000 > 200):
                i1 = (test_peak - begin) / fs * 1000.0; i2 = (end - test_peak) / fs * 1000.0
                RR = RR[:c] + [i1, i2] + RR[c + 1:]; L = len(RR); large_prob.append(0); c -= 1
            else:
                large_prob[c] = 1
        c += 1
    RR_samples = np.array([v * fs / 1000.0 for v in RR])
    return np.concatenate([[R0], np.round(np.cumsum(RR_samples) + R0)]).astype(int)


def peak_detection(signal, fs=SFREQ, env_ms=300, avgHR=100, postproc=True, inverted=False):
    """Faithful Python port of R-DECO peak_detection.m (single channel).
    Returns sorted unique R-peak sample indices."""
    sig = np.asarray(signal, float)
    if inverted:
        sig = -sig
    env = round(fs * env_ms / 1000.0)
    orig_len = len(sig); segm_length = 60 * fs; nr_orig = orig_len // segm_length
    if nr_orig != 0:
        pad = segm_length - (orig_len - nr_orig * segm_length)
        sig_p = np.concatenate([sig, np.zeros(pad)]); new_len = len(sig_p); nr = new_len // segm_length
        segmented = sig_p.reshape(segm_length, nr, order='F')
    else:
        new_len = orig_len; nr = 1
        segmented = ((sig - sig.mean()) / (sig.std() + 1e-12)).reshape(-1, 1)
    R_peak_ch = [np.array([], int)] * nr
    for j in range(nr):
        if nr > 1:
            if j == 0:
                segm = np.concatenate([segmented[:, 0], segmented[:fs * 5, 1]])
            elif j == nr - 1:
                keep = segmented[:segm_length - (new_len - orig_len), j]
                segm = np.concatenate([segmented[-fs * 5:, j - 1], keep])
            else:
                segm = np.concatenate([segmented[-fs * 5:, j - 1], segmented[:, j], segmented[:fs * 5, j + 1]])
        else:
            segm = segmented[:, 0]
        time = np.arange(1, len(segm) + 1) / fs
        env_segm = _env_secant(time, segm, env, 'top') - _env_secant(time, segm, env, 'bottom')
        fr = fs / 250.0; er = env / (0.15 * fs)
        all_peaks = _calc_pos_opt(env_segm, int(np.floor(20 * fr * er)), 1)
        R = np.array([], int)
        if len(all_peaks) > time[-1] / 2:
            R, RR = _adaptive_thresholding(all_peaks, env_segm, fs, avgHR)
            if postproc and len(RR) > 5:
                try:
                    R = _post_processing(all_peaks, R, RR, env_segm, fs)
                except Exception:
                    pass
        if len(R) > time[-1] / 2:
            R = np.asarray(R, int)
            if nr > 1:
                if j == 0:
                    R = R[R <= segm_length]
                elif j == nr - 1:
                    R = R[R > fs * 5] - fs * 5
                else:
                    R = R[R > fs * 5] - fs * 5; R = R[R <= segm_length]
        else:
            R = np.array([], int)
        R_peak_ch[j] = np.unique(R)
    R_temp = np.concatenate([R_peak_ch[j] + segm_length * j for j in range(nr)]) if nr else np.array([], int)
    R_temp = R_temp[(R_temp >= 0) & (R_temp < orig_len)]
    R_temp = _peaks_in_ecg(sig, R_temp, round(0.065 * fs))
    return np.unique(np.round(R_temp).astype(int))


def rdeco_style_detect(sig, fs=SFREQ, refractory_ms=300, search_ms=100):
    """R-DECO detector (ported from peak_detection.m). Kept under this name so the
    editor seed / re-detect button / SEED_DETECTORS all call the real algorithm.
    refractory_ms maps to R-DECO's envelope window (Rpeak{1}); search_ms is unused
    by R-DECO and retained only for call-signature compatibility."""
    return peak_detection(sig, fs=fs, env_ms=refractory_ms, avgHR=100, postproc=True)


def detect_peaks(sig, fs=SFREQ):
    return np.asarray(P.detect_peaks(sig), dtype=int)


SEED_DETECTORS = {'rdeco': rdeco_style_detect, 'dp': detect_peaks}


# ---- interactive edit helpers ----
def add_peak(peaks, sig, x, fs, search_ms=100):
    sr = max(1, int(search_ms / 1000.0 * fs)); a, b = max(0, x - sr), min(len(sig), x + sr)
    if b <= a:
        return peaks
    p = a + int(np.argmax(sig[a:b]))
    return sorted(peaks + [p]) if p not in peaks else peaks


def delete_peak(peaks, x, fs, tol_ms=120):
    if not peaks:
        return peaks
    arr = np.array(peaks); j = int(np.argmin(np.abs(arr - x)))
    return peaks[:j] + peaks[j + 1:] if abs(peaks[j] - x) <= int(tol_ms / 1000.0 * fs) else peaks


def adjust_peak(peaks, sig, x, fs, search_ms=100):
    if not peaks:
        return peaks
    arr = np.array(peaks); j = int(np.argmin(np.abs(arr - x)))
    sr = max(1, int(search_ms / 1000.0 * fs)); a, b = max(0, x - sr), min(len(sig), x + sr)
    if b <= a:
        return peaks
    newp = a + int(np.argmax(sig[a:b])); rest = peaks[:j] + peaks[j + 1:]
    return sorted(rest + [newp]) if newp not in rest else rest


# ============================================================
# REFERENCE-FREE RHYTHM QC
# ============================================================
SHORT_MS = 400
RR_LO_MS, RR_HI_MS = 500, 1300


def rhythm_qc(pk, fs=SFREQ):
    n = len(pk)
    if n < 6:
        return {'n_peaks': n, 'apparent_hr': float('nan'), 'rr_reg': float('nan'),
                'frac_short': 1.0, 'frac_implaus': 1.0}
    rr = np.diff(pk) / fs * 1000.0
    return {'n_peaks': n, 'apparent_hr': float(60000.0 / np.median(rr)),
            'rr_reg': float(1.0 / (1.0 + rr.std() / rr.mean())),
            'frac_short': float(np.mean(rr < SHORT_MS)),
            'frac_implaus': float(np.mean((rr < RR_LO_MS) | (rr > RR_HI_MS)))}


def is_good(qc, fs_thr, fi_thr, flagged_bad=False):
    if flagged_bad or np.isnan(qc['frac_short']):
        return False
    return qc['frac_short'] <= fs_thr and qc['frac_implaus'] <= fi_thr


# ============================================================
# STAGE FUNCTIONS
# ============================================================
def _read_raw_any(path):
    """Load a subject recording by extension: EEGLAB .set or BioSemi .bdf.
    Returned preloaded with channel order preserved, so the first 128 channels
    are taken as EEG downstream exactly as for .set inputs. Output is always
    written as .set, so a mixed .set/.bdf batch yields a uniform reconstructed set."""
    if Path(path).suffix.lower() == '.bdf':
        return mne.io.read_raw_bdf(str(path), preload=True, verbose=False)
    return mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)


def stage1_reconstruct(subject_file, model, device, out_dir):
    name = Path(subject_file).stem
    raw = _read_raw_any(subject_file)
    raw.resample(SFREQ)  # align to the model rate; resampling preserves amplitude/scale

    # ---- model input: a FILTERED + per-channel z-scored COPY (the model was trained on this).
    #      This copy never reaches the saved file, so reconstruction quality is unchanged. ----
    raw_f = raw.copy().filter(l_freq=5.0, h_freq=50.0, method='iir', verbose=False)
    eeg_f = raw_f.get_data()[:128]
    eeg_norm = (eeg_f - eeg_f.mean(axis=1, keepdims=True)) / (eeg_f.std(axis=1, keepdims=True) + 1e-6)
    T = eeg_norm.shape[1]
    eeg_input = np.column_stack([eeg_norm.T, np.zeros(T, dtype=np.float32)]).astype(np.float32)
    pred_sig, _, _ = P.reconstruct_subject(model, eeg_input, device)

    # ---- output: the ORIGINAL-scale EEG (unfiltered, in Volts as MNE holds it) so the EEGLAB
    #      export restores the original microvolt values, plus the reconstructed ECG rescaled to
    #      the EEG's own amplitude. The previous version saved the z-scored array, which the
    #      exporter multiplied by 1e6 (V->uV) and produced giant values. ----
    eeg_out = raw.get_data()[:128]                       # original values (Volts), resampled only
    L = int(min(len(pred_sig), eeg_out.shape[1]))
    eeg_out = eeg_out[:, :L]
    pred = np.asarray(pred_sig[:L], dtype=float)
    pred = (pred - pred.mean()) / (pred.std() + 1e-12)   # unit-variance reconstruction (arbitrary units)
    scale = float(np.median(np.std(eeg_out, axis=1))) + 1e-12  # match typical EEG channel amplitude (Volts)
    ecg_out = (pred * scale).reshape(1, -1)

    out = np.vstack([eeg_out, ecg_out]).astype(np.float64)  # Volts; EEGLAB export scales V->uV correctly
    ch_names = list(raw.ch_names[:128]) + ['ECG']
    info = mne.create_info(ch_names, SFREQ, ['eeg'] * 128 + ['ecg'])
    out_raw = mne.io.RawArray(out, info, verbose=False)
    try:  # best-effort: carry over the original electrode locations for downstream topography
        mont = raw.get_montage()
        if mont is not None:
            out_raw.set_montage(mont, on_missing='ignore', verbose=False)
    except Exception:
        pass
    out_raw.export(str(Path(out_dir) / f"{name}.set"), fmt='eeglab', overwrite=True, verbose=False)
    return name


def stage2_mark_auto(recon_file, out_dir, method):
    name = Path(recon_file).stem
    raw = mne.io.read_raw_eeglab(recon_file, preload=True, verbose=False)
    ecg = raw.get_data(picks='ECG')[0]
    pk = detect_peaks(ecg, SFREQ)
    raw.set_annotations(mne.Annotations(onset=np.asarray(pk) / SFREQ, duration=np.zeros(len(pk)),
                                        description=['R-peak'] * len(pk)))
    raw.export(str(Path(out_dir) / f"{name}.set"), fmt='eeglab', overwrite=True, verbose=False)
    return name, len(pk)


def stage3_discriminate(marked_file, fs_thr, fi_thr, keep_mode, good_dir, bad_dir):
    name = Path(marked_file).stem
    raw = mne.io.read_raw_eeglab(marked_file, preload=False, verbose=False)
    ann = raw.annotations
    rpeak_sec = [o for o, d in zip(ann.onset, ann.description)
                 if str(d).strip().lower().replace('_', '-') in ('r-peak', 'rpeak')]
    flagged_bad = any('bad' in str(d).lower() for d in ann.description)
    pk = np.round(np.asarray(rpeak_sec) * SFREQ).astype(int)
    qc = rhythm_qc(pk, SFREQ)
    good = is_good(qc, fs_thr, fi_thr, flagged_bad=flagged_bad)
    should_save = (good and keep_mode == 'good') or ((not good) and keep_mode == 'bad')
    saved = 'no'
    if should_save:
        dest = Path(good_dir if good else bad_dir); dest.mkdir(parents=True, exist_ok=True)
        for ext in ('.set', '.fdt'):
            src = Path(marked_file).with_suffix(ext)
            if src.exists():
                shutil.copy(str(src), str(dest / src.name))
        saved = 'yes'
    return {'Subject': name, 'Decision': 'GOOD' if good else 'BAD', 'Saved': saved,
            'flagged_bad': flagged_bad, 'n_peaks': qc['n_peaks'],
            'apparent_HR': round(qc['apparent_hr'], 1) if not np.isnan(qc['apparent_hr']) else np.nan,
            'frac_short': round(qc['frac_short'], 4), 'frac_implaus': round(qc['frac_implaus'], 4)}


# ============================================================
# SHARED LIGHT THEME  (R-DECO lineage, refined)
# ============================================================
UI = 'Segoe UI'
MONO = 'Consolas'
T = {
    'canvas': '#EEF2F6', 'panel': '#F7F9FB', 'paper': '#FFFFFF',
    'ink': '#1B2733', 'mute': '#6B7884', 'faint': '#9AA7B2', 'line': '#DCE3EA',
    'accent': '#2D7DD2', 'accent_hi': '#3F8DDD', 'accent_lo': '#21669F',
    'trace': '#22303C', 'peak': '#E03131', 'peak_hi': '#FF6B6B',
    'band': '#E3F0E7', 'short': '#E8590C',
    'good': '#2F9E44', 'good_hi': '#37B24D', 'bad': '#E03131', 'bad_hi': '#F03E3E',
    'neutral': '#E9EEF3', 'neutral_hi': '#DCE3EA',
}


def _hover(btn, normal, hi):
    btn.bind('<Enter>', lambda _e: btn.configure(bg=hi))
    btn.bind('<Leave>', lambda _e: btn.configure(bg=normal))


def mkbtn(parent, text, command, kind='neutral', **kw):
    palette = {'primary': (T['accent'], T['accent_hi'], 'white'),
               'positive': (T['good'], T['good_hi'], 'white'),
               'danger': (T['bad'], T['bad_hi'], 'white'),
               'neutral': (T['neutral'], T['neutral_hi'], T['ink'])}
    bg, hi, fg = palette.get(kind, palette['neutral'])
    bold = kind in ('primary', 'positive', 'danger')
    b = tk.Button(parent, text=text, command=command, font=(UI, 10, 'bold' if bold else 'normal'),
                  bg=bg, fg=fg, activebackground=hi, activeforeground=fg, relief='flat', bd=0,
                  padx=14, pady=7, cursor='hand2', highlightthickness=0, **kw)
    _hover(b, bg, hi)
    return b


def card(parent, title=None):
    """White card with a 1px hairline border (via padding) and an optional eyebrow title."""
    outer = tk.Frame(parent, bg=T['line'])
    inner = tk.Frame(outer, bg=T['paper'])
    inner.pack(fill='both', expand=True, padx=1, pady=1)
    if title:
        tk.Label(inner, text=title.upper(), bg=T['paper'], fg=T['mute'],
                 font=(UI, 8, 'bold')).pack(anchor='w', padx=12, pady=(9, 0))
    return outer, inner


# ============================================================
# MANUAL R-PEAK EDITOR  (R-DECO style: Add / Adjust(drag) / Delete + RR tachogram)
# ============================================================
class PeakEditor(tk.Toplevel):
    def __init__(self, master, files, out_dir, sfreq, log):
        super().__init__(master)
        self.title("R-peak editor")
        self.geometry("1240x820")
        self.configure(bg=T['canvas'])
        self.files, self.out_dir, self.sfreq, self.log = files, out_dir, sfreq, log
        self.idx = 0
        self.win_sec = tk.StringVar(value='20')
        self.pos = tk.DoubleVar(value=0.0)
        self.mode = tk.StringVar(value='add')
        self.start = 0
        self.end = 0
        self._drag_active = False
        self._drag_other = None
        self._drag_pos = None
        self._mode_btns = {}
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._stop)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self._load(0)

    # ---------------- layout ----------------
    def _build(self):
        self.option_add('*Font', (UI, 10))

        # header / breadcrumb
        head = tk.Frame(self, bg=T['paper'], height=54); head.pack(fill='x'); head.pack_propagate(False)
        self.h_title = tk.Label(head, text="", bg=T['paper'], fg=T['ink'], font=(UI, 15, 'bold'))
        self.h_title.pack(side='left', padx=(18, 0))
        self.h_count = tk.Label(head, text="", bg=T['paper'], fg=T['mute'], font=(UI, 10))
        self.h_count.pack(side='left', padx=(10, 0))
        self.bad_pill = tk.Label(head, text="  MARKED BAD  ", bg=T['bad'], fg='white', font=(UI, 9, 'bold'))
        self.h_peaks = tk.Label(head, text="", bg=T['paper'], fg=T['mute'], font=(UI, 10))
        self.h_peaks.pack(side='right', padx=18)
        tk.Frame(self, bg=T['line'], height=1).pack(fill='x')

        # toolbar
        bar = tk.Frame(self, bg=T['canvas']); bar.pack(fill='x', padx=14, pady=(12, 6))
        self._group(bar, "Correction", self._seg_correction)
        self._group(bar, "Re-detect", self._grp_redetect)
        self._group(bar, "Window", self._grp_window)

        tk.Label(self, bg=T['canvas'], fg=T['faint'], font=(UI, 9), justify='left',
                 text="Add \u2014 click to place a peak     \u00b7     Adjust \u2014 drag a peak to move it     \u00b7     "
                      "Delete \u2014 click a peak     \u00b7     right-click always deletes the nearest peak"
                 ).pack(anchor='w', padx=18, pady=(0, 6))

        # scrubber
        scrub = tk.Frame(self, bg=T['canvas']); scrub.pack(fill='x', padx=18, pady=(0, 6))
        self.slider = ttk.Scale(scrub, from_=0, to=1, variable=self.pos,
                                command=lambda _=None: self._full_draw())
        self.slider.pack(fill='x')

        # plots card
        outer, inner = card(self); outer.pack(fill='both', expand=True, padx=14, pady=(0, 8))
        self.fig = Figure(figsize=(11.6, 5.6), facecolor=T['paper'])
        self.fig.subplots_adjust(left=0.065, right=0.985, top=0.97, bottom=0.085, hspace=0.13)
        gs = self.fig.add_gridspec(3, 1)
        self.ax = self.fig.add_subplot(gs[0:2, 0])
        self.ax_rr = self.fig.add_subplot(gs[2, 0], sharex=self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=inner)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        # action bar
        act = tk.Frame(self, bg=T['canvas']); act.pack(fill='x', padx=14, pady=(0, 12))
        self.bad_btn = mkbtn(act, "\u2691  Mark as bad", self._mark_bad, kind='neutral')
        self.bad_btn.pack(side='left')
        mkbtn(act, "Save & next  \u2192", self._save_next, kind='positive').pack(side='right')
        mkbtn(act, "Skip", self._skip, kind='neutral').pack(side='right', padx=(0, 8))
        mkbtn(act, "Stop", self._stop, kind='neutral').pack(side='right', padx=(0, 8))

    def _group(self, parent, title, builder):
        wrap = tk.Frame(parent, bg=T['canvas']); wrap.pack(side='left', padx=(0, 14))
        tk.Label(wrap, text=title.upper(), bg=T['canvas'], fg=T['faint'],
                 font=(UI, 8, 'bold')).pack(anchor='w', pady=(0, 2))
        box = tk.Frame(wrap, bg=T['line']); box.pack(anchor='w')
        inner = tk.Frame(box, bg=T['paper']); inner.pack(padx=1, pady=1)
        builder(inner)

    def _seg_correction(self, parent):
        for i, (val, lbl) in enumerate([('add', '\uFF0B  Add'), ('adjust', '\u2194  Adjust'),
                                        ('delete', '\u2212  Delete')]):
            if i:
                tk.Frame(parent, bg=T['line'], width=1).pack(side='left', fill='y')
            b = tk.Button(parent, text=lbl, font=(UI, 10, 'bold'), relief='flat', bd=0, padx=15, pady=7,
                          cursor='hand2', highlightthickness=0, command=lambda v=val: self._set_mode(v))
            b.pack(side='left'); self._mode_btns[val] = b
        self._restyle_modes()

    def _set_mode(self, v):
        self.mode.set(v); self._restyle_modes(); self._update_header()

    def _restyle_modes(self):
        for v, b in self._mode_btns.items():
            if v == self.mode.get():
                b.configure(bg=T['accent'], fg='white', activebackground=T['accent'], activeforeground='white')
                b.unbind('<Enter>'); b.unbind('<Leave>')
            else:
                b.configure(bg=T['paper'], fg=T['ink'], activebackground=T['neutral'], activeforeground=T['ink'])
                _hover(b, T['paper'], T['neutral'])

    def _grp_redetect(self, parent):
        for lbl, key in [('R-DECO', 'rdeco'), ('Pipeline', 'dp')]:
            b = tk.Button(parent, text=lbl, font=(UI, 10), relief='flat', bd=0, padx=13, pady=7,
                          cursor='hand2', highlightthickness=0, bg=T['paper'], fg=T['accent'],
                          activebackground=T['neutral'], activeforeground=T['accent_lo'],
                          command=lambda k=key: self._redetect(k))
            b.pack(side='left'); _hover(b, T['paper'], T['neutral'])

    def _grp_window(self, parent):
        for glyph, d in [('\u25C0', -1), ('\u25B6', 1)]:
            b = tk.Button(parent, text=glyph, font=(UI, 10), relief='flat', bd=0, padx=11, pady=7,
                          cursor='hand2', highlightthickness=0, bg=T['paper'], fg=T['ink'],
                          activebackground=T['neutral'], command=lambda dd=d: self._page(dd))
            b.pack(side='left'); _hover(b, T['paper'], T['neutral'])
        tk.Frame(parent, bg=T['line'], width=1).pack(side='left', fill='y')
        tk.Label(parent, text=' width', bg=T['paper'], fg=T['mute'], font=(UI, 9)).pack(side='left')
        e = tk.Entry(parent, textvariable=self.win_sec, width=4, font=(UI, 10), relief='flat',
                     bg=T['panel'], fg=T['ink'], justify='center', highlightthickness=1,
                     highlightbackground=T['line'], highlightcolor=T['accent'])
        e.pack(side='left', padx=5, pady=5, ipady=2); e.bind('<Return>', lambda _=None: self._apply_width())
        tk.Label(parent, text='s ', bg=T['paper'], fg=T['mute'], font=(UI, 9)).pack(side='left')

    # ---------------- data / drawing ----------------
    def _load(self, i):
        self.idx = i
        self.raw = mne.io.read_raw_eeglab(self.files[i], preload=True, verbose=False)
        self.ecg = self.raw.get_data(picks='ECG')[0]
        self.fname = Path(self.files[i]).stem
        self.peaks = list(map(int, rdeco_style_detect(self.ecg, self.sfreq)))
        self.is_bad = False
        self._update_bad_btn()
        self.pos.set(0.0)
        self._apply_width()

    def _win_samples(self):
        try:
            return max(int(float(self.win_sec.get()) * self.sfreq), self.sfreq)
        except ValueError:
            return int(20 * self.sfreq)

    def _apply_width(self):
        self.slider.config(to=max(0, len(self.ecg) - self._win_samples()))
        self._full_draw()

    def _page(self, d):
        step = int(0.8 * self._win_samples())
        self.pos.set(min(max(0.0, self.pos.get() + d * step), float(self.slider.cget('to'))))
        self._full_draw()

    def _style_axes(self):
        for a in (self.ax, self.ax_rr):
            a.set_facecolor(T['paper'])
            for sp in a.spines.values():
                sp.set_color(T['line']); sp.set_linewidth(1.0)
            a.tick_params(colors=T['mute'], labelsize=8, length=3, width=0.8, color=T['line'])
            a.grid(True, color=T['line'], lw=0.7, alpha=0.7)
            a.set_axisbelow(True)

    def _full_draw(self):
        w = self._win_samples()
        self.start = int(self.pos.get()); self.end = min(len(self.ecg), self.start + w)
        s, e = self.start, self.end
        self.ax.clear(); self.ax_rr.clear()
        self._style_axes()
        # ECG trace + peak markers (circles) + drag-highlight ring
        t = np.arange(s, e) / self.sfreq
        self.ax.plot(t, self.ecg[s:e], lw=0.9, color=T['trace'], zorder=2)
        self._mk_line, = self.ax.plot([], [], 'o', ms=6, mfc=T['peak'], mec='white', mew=0.7,
                                      ls='none', zorder=4)
        self._drag_mk, = self.ax.plot([], [], 'o', ms=13, mfc='none', mec=T['accent'], mew=1.8,
                                      ls='none', zorder=5)
        self.ax.set_ylabel('ECG  (a.u.)', color=T['mute'], fontsize=9)
        self.ax.tick_params(labelbottom=False)
        self.ax.set_xlim(s / self.sfreq, max(e, s + 1) / self.sfreq)
        # RR tachogram: plausible band + short-RR line + RR curve (circles)
        self.ax_rr.axhspan(RR_LO_MS, RR_HI_MS, color=T['band'], zorder=0)
        self.ax_rr.axhline(SHORT_MS, color=T['short'], ls=(0, (2, 2)), lw=1.0, alpha=0.85, zorder=1)
        self._rr_line, = self.ax_rr.plot([], [], '-o', color=T['accent'], ms=3.5, lw=1.0,
                                         mfc=T['accent'], mec='white', mew=0.5, zorder=3)
        self.ax_rr.set_ylabel('RR  (ms)', color=T['mute'], fontsize=9)
        self.ax_rr.set_xlabel('Time  (s)', color=T['mute'], fontsize=9)
        self.ax_rr.set_ylim(0, 1800)
        self._refresh_peaks(draw=False)
        self.canvas.draw_idle()

    def _refresh_peaks(self, draw=True):
        s, e = self.start, self.end
        pin = [p for p in self.peaks if s <= p < e]
        self._mk_line.set_data(np.array(pin) / self.sfreq, self.ecg[pin]) if pin else self._mk_line.set_data([], [])
        if self._drag_active and self._drag_pos is not None and s <= self._drag_pos < e:
            self._drag_mk.set_data([self._drag_pos / self.sfreq], [self.ecg[int(self._drag_pos)]])
        else:
            self._drag_mk.set_data([], [])
        if len(self.peaks) >= 2:
            pk = np.array(sorted(self.peaks)); rrv = np.diff(pk) / self.sfreq * 1000.0
            self._rr_line.set_data(pk[1:] / self.sfreq, rrv)
        else:
            self._rr_line.set_data([], [])
        self._update_header()
        if draw:
            self.canvas.draw_idle()

    def _update_header(self):
        self.h_title.config(text=self.fname)
        self.h_count.config(text=f"subject {self.idx + 1} of {len(self.files)}")
        self.h_peaks.config(text=f"{len(self.peaks)} peaks   \u00b7   {self.mode.get().capitalize()} mode")
        self.bad_pill.pack(side='left', padx=(10, 0)) if self.is_bad else self.bad_pill.pack_forget()

    # ---------------- interaction ----------------
    def _nearest(self, x, tol_ms=140):
        if not self.peaks:
            return None
        arr = np.array(self.peaks); j = int(np.argmin(np.abs(arr - x)))
        return j if abs(self.peaks[j] - x) <= int(tol_ms / 1000.0 * self.sfreq) else None

    def _on_press(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        x = int(round(ev.xdata * self.sfreq))
        if ev.button == 3:
            self.peaks = delete_peak(self.peaks, x, self.sfreq); self._refresh_peaks(); return
        if ev.button != 1:
            return
        m = self.mode.get()
        if m == 'add':
            self.peaks = add_peak(self.peaks, self.ecg, x, self.sfreq); self._refresh_peaks()
        elif m == 'delete':
            self.peaks = delete_peak(self.peaks, x, self.sfreq); self._refresh_peaks()
        elif m == 'adjust':
            j = self._nearest(x)
            if j is None:
                return
            self._drag_active = True
            self._drag_other = self.peaks[:j] + self.peaks[j + 1:]
            self._drag_pos = self.peaks[j]
            self.canvas.get_tk_widget().config(cursor='fleur')
            self._refresh_peaks()

    def _on_motion(self, ev):
        if not self._drag_active or ev.inaxes != self.ax or ev.xdata is None:
            return
        x = min(max(0, int(round(ev.xdata * self.sfreq))), len(self.ecg) - 1)
        self._drag_pos = x
        self.peaks = sorted(self._drag_other + [x])
        self._refresh_peaks()

    def _on_release(self, ev):
        if not self._drag_active:
            return
        x = self._drag_pos
        sr = max(1, int(0.10 * self.sfreq)); a, b = max(0, x - sr), min(len(self.ecg), x + sr)
        snapped = a + int(np.argmax(self.ecg[a:b])) if b > a else x
        self.peaks = sorted(self._drag_other) if snapped in self._drag_other else sorted(self._drag_other + [snapped])
        self._drag_active = False; self._drag_other = None; self._drag_pos = None
        self.canvas.get_tk_widget().config(cursor='')
        self._refresh_peaks()

    def _redetect(self, which):
        self.peaks = list(map(int, SEED_DETECTORS[which](self.ecg, self.sfreq)))
        self._full_draw()

    def _mark_bad(self):
        self.is_bad = not self.is_bad; self._update_bad_btn(); self._refresh_peaks()

    def _update_bad_btn(self):
        if self.is_bad:
            self.bad_btn.configure(text="\u2691  Marked bad \u2014 undo", bg=T['bad'], fg='white',
                                   activebackground=T['bad_hi'], activeforeground='white')
            _hover(self.bad_btn, T['bad'], T['bad_hi'])
        else:
            self.bad_btn.configure(text="\u2691  Mark as bad", bg=T['neutral'], fg=T['ink'],
                                   activebackground=T['neutral_hi'], activeforeground=T['ink'])
            _hover(self.bad_btn, T['neutral'], T['neutral_hi'])

    # ---------------- save / navigate ----------------
    def _save_current(self):
        onsets = list(np.asarray(self.peaks) / self.sfreq)
        descs = ['R-peak'] * len(self.peaks)
        if self.is_bad:
            onsets.append(0.0); descs.append('BAD_RECON')
        self.raw.set_annotations(mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descs))
        self.raw.export(str(Path(self.out_dir) / f"{self.fname}.set"), fmt='eeglab', overwrite=True, verbose=False)
        self.log(f"  saved {self.fname}: {len(self.peaks)} peaks" + ("  [BAD]" if self.is_bad else ""))

    def _save_next(self):
        try:
            self._save_current()
        except Exception as e:
            self.log(f"  save failed for {self.fname}: {e}"); return
        self._finish() if self.idx + 1 >= len(self.files) else self._load(self.idx + 1)

    def _skip(self):
        self.log(f"  skipped {self.fname}")
        self._finish() if self.idx + 1 >= len(self.files) else self._load(self.idx + 1)

    def _stop(self):
        self.log("manual editor closed (remaining subjects not marked)."); self.destroy()

    def _finish(self):
        self.log("manual marking complete -> 02_marked/"); self.destroy()


# ============================================================
# GUI
# ============================================================
class ToolboxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG-to-ECG Inference Toolbox")
        self.root.geometry("900x780")
        self.root.configure(bg=T['canvas'])
        self.work_dir = tk.StringVar()
        self.raw_input = tk.StringVar()
        self.raw_mode = tk.StringVar(value='folder')
        self.mark_method = tk.StringVar(value='editor')
        self.mark_in_dir = tk.StringVar()    # optional Stage-2 input override (e.g. a review pool)
        self.mark_out_dir = tk.StringVar()   # optional Stage-2 output override (this rater's folder)
        self.keep_mode = tk.StringVar(value='good')
        self.frac_short_thr = tk.StringVar(value='0.16')    # paper thresholds (calibrated on the selection split)
        self.frac_implaus_thr = tk.StringVar(value='0.29')
        self.raw_files = []
        self.is_running = False
        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        st = ttk.Style()
        try:
            st.theme_use('clam')
        except Exception:
            pass
        st.configure('TNotebook', background=T['canvas'], borderwidth=0, tabmargins=(0, 6, 0, 0))
        st.configure('TNotebook.Tab', background=T['neutral'], foreground=T['mute'],
                     font=(UI, 10, 'bold'), padding=(20, 9), borderwidth=0)
        st.map('TNotebook.Tab', background=[('selected', T['paper'])],
               foreground=[('selected', T['accent'])])
        st.configure('Horizontal.TScale', background=T['canvas'], troughcolor=T['line'], borderwidth=0)
        st.configure('Vertical.TScrollbar', background=T['neutral'], troughcolor=T['canvas'],
                     borderwidth=0, arrowcolor=T['mute'])
        self.C = T

    def _build_ui(self):
        self.root.configure(bg=T['canvas'])

        head = tk.Frame(self.root, bg=T['paper'], height=56); head.pack(fill='x'); head.pack_propagate(False)
        tk.Label(head, text="EEG \u2192 ECG", bg=T['paper'], fg=T['accent'],
                 font=(UI, 15, 'bold')).pack(side='left', padx=(18, 8))
        tk.Label(head, text="Inference Toolbox", bg=T['paper'], fg=T['ink'],
                 font=(UI, 15)).pack(side='left')
        tk.Label(head, text=f" conv_only \u00b7 {DEVICE.upper()} ", bg=T['neutral'], fg=T['mute'],
                 font=(UI, 9, 'bold')).pack(side='right', padx=18, ipady=2)
        tk.Frame(self.root, bg=T['line'], height=1).pack(fill='x')

        wrap = tk.Frame(self.root, bg=T['canvas']); wrap.pack(fill='both', expand=True, padx=16, pady=14)

        wrow = tk.Frame(wrap, bg=T['canvas']); wrow.pack(fill='x', pady=(0, 12))
        tk.Label(wrow, text="Working folder", bg=T['canvas'], fg=T['ink'],
                 font=(UI, 10, 'bold')).pack(side='left')
        ent = tk.Entry(wrow, textvariable=self.work_dir, font=(UI, 10), relief='flat', bg=T['paper'],
                       fg=T['ink'], highlightthickness=1, highlightbackground=T['line'],
                       highlightcolor=T['accent'])
        ent.pack(side='left', fill='x', expand=True, padx=10, ipady=5)
        mkbtn(wrow, "Browse", lambda: self._browse_dir(self.work_dir), kind='neutral').pack(side='right')

        nb = ttk.Notebook(wrap); nb.pack(fill='x', pady=(0, 12))
        self._tab_reconstruct(nb); self._tab_mark(nb); self._tab_discriminate(nb)

        outer, inner = card(wrap, "Activity log"); outer.pack(fill='both', expand=True)
        self.log_box = scrolledtext.ScrolledText(inner, height=11, bg=T['paper'], fg=T['ink'],
                                                  font=(MONO, 9), relief='flat', state='disabled',
                                                  highlightthickness=0, bd=0, insertbackground=T['ink'],
                                                  padx=4, pady=2)
        self.log_box.pack(fill='both', expand=True, padx=10, pady=(4, 10))

        self.status_lbl = tk.Label(self.root, text="Ready", bg=T['canvas'], fg=T['mute'],
                                   font=(UI, 9), anchor='w')
        self.status_lbl.pack(fill='x', padx=18, pady=(0, 8))

    def _panelize(self, nb, title):
        f = tk.Frame(nb, bg=T['paper']); nb.add(f, text=title)
        inner = tk.Frame(f, bg=T['paper']); inner.pack(fill='both', expand=True, padx=20, pady=18)
        return inner

    def _radio(self, parent, val, text):
        return tk.Radiobutton(parent, text=text, variable=None, value=val, bg=T['paper'], fg=T['ink'],
                              selectcolor=T['paper'], activebackground=T['paper'], font=(UI, 10),
                              highlightthickness=0, anchor='w')

    def _tab_reconstruct(self, nb):
        p = self._panelize(nb, "  1  Reconstruct  ")
        tk.Label(p, text="Raw EEG input (128-channel .set or .bdf)", bg=T['paper'], fg=T['ink'],
                 font=(UI, 10, 'bold')).pack(anchor='w')
        for v, lbl in [('folder', 'Folder (searched recursively)'), ('files', 'Individual files')]:
            tk.Radiobutton(p, text=lbl, variable=self.raw_mode, value=v, bg=T['paper'], fg=T['ink'],
                           selectcolor=T['paper'], activebackground=T['paper'], font=(UI, 10),
                           highlightthickness=0, anchor='w').pack(anchor='w', pady=1)
        row = tk.Frame(p, bg=T['paper']); row.pack(fill='x', pady=(8, 2))
        tk.Entry(row, textvariable=self.raw_input, font=(UI, 10), relief='flat', bg=T['panel'], fg=T['ink'],
                 highlightthickness=1, highlightbackground=T['line'], highlightcolor=T['accent']
                 ).pack(side='left', fill='x', expand=True, ipady=4)
        mkbtn(row, "Browse", self._browse_raw, kind='neutral').pack(side='right', padx=(8, 0))
        self.raw_count = tk.Label(p, text="0 files", bg=T['paper'], fg=T['mute'], font=(UI, 9))
        self.raw_count.pack(anchor='w', pady=(2, 0))
        tk.Label(p, text="Reconstructs ECG and writes a 129-channel .set (ECG = channel 129) to 01_reconstructed/.",
                 bg=T['paper'], fg=T['mute'], font=(UI, 9), justify='left').pack(anchor='w', pady=(10, 10))
        mkbtn(p, "Run reconstruction", lambda: self._run(self._do_reconstruct), kind='primary').pack(anchor='w')

    def _tab_mark(self, nb):
        p = self._panelize(nb, "  2  Mark peaks  ")
        tk.Label(p, text="Peak-detection method", bg=T['paper'], fg=T['ink'],
                 font=(UI, 10, 'bold')).pack(anchor='w')
        for v, lbl in [('detectpeaks', 'Automatic \u2014 pipeline detect_peaks (frac-validated)'),
                       ('editor', 'Manual editor \u2014 R-DECO seed + Add / Adjust / Delete')]:
            tk.Radiobutton(p, text=lbl, variable=self.mark_method, value=v, bg=T['paper'], fg=T['ink'],
                           selectcolor=T['paper'], activebackground=T['paper'], font=(UI, 10),
                           highlightthickness=0, anchor='w').pack(anchor='w', pady=1)
        tk.Label(p, text="Reads 01_reconstructed/ and writes R-peak events into each .set \u2192 02_marked/.\n"
                         "The manual editor opens one subject at a time, seeded with R-DECO detection.",
                 bg=T['paper'], fg=T['mute'], font=(UI, 9), justify='left').pack(anchor='w', pady=(10, 10))
        mk_over = tk.Frame(p, bg=T['paper']); mk_over.pack(anchor='w', fill='x', pady=(0, 10))
        tk.Label(mk_over, text="Multi-rater review (optional) \u2014 leave blank to use the working folder above",
                 bg=T['paper'], fg=T['ink'], font=(UI, 9, 'bold')).pack(anchor='w', pady=(0, 4))
        for lbl, var in [("Reconstructions to mark", self.mark_in_dir),
                         ("Save my marked peaks to", self.mark_out_dir)]:
            r = tk.Frame(mk_over, bg=T['paper']); r.pack(anchor='w', fill='x', pady=2)
            tk.Label(r, text=lbl, bg=T['paper'], fg=T['mute'], font=(UI, 9),
                     width=24, anchor='w').pack(side='left')
            tk.Entry(r, textvariable=var, font=(UI, 9), relief='flat', bg=T['paper'],
                     fg=T['ink'], insertbackground=T['ink']).pack(side='left', fill='x', expand=True, ipady=3)
            mkbtn(r, "Browse", lambda v=var: self._browse_dir(v), kind='neutral').pack(side='right', padx=(8, 0))
        mkbtn(p, "Run peak marking", self._on_mark, kind='primary').pack(anchor='w')

    def _tab_discriminate(self, nb):
        p = self._panelize(nb, "  3  Discriminate  ")
        krow = tk.Frame(p, bg=T['paper']); krow.pack(anchor='w')
        tk.Label(krow, text="Save", bg=T['paper'], fg=T['ink'], font=(UI, 10, 'bold')).pack(side='left', padx=(0, 10))
        for v, lbl in [('good', 'Good only'), ('bad', 'Bad only')]:
            tk.Radiobutton(krow, text=lbl, variable=self.keep_mode, value=v, bg=T['paper'], fg=T['ink'],
                           selectcolor=T['paper'], activebackground=T['paper'], font=(UI, 10),
                           highlightthickness=0).pack(side='left', padx=(0, 12))
        trow = tk.Frame(p, bg=T['paper']); trow.pack(anchor='w', pady=(10, 0))
        tk.Label(trow, text="frac_short \u2264", bg=T['paper'], fg=T['ink'], font=(UI, 10)).pack(side='left')
        tk.Entry(trow, textvariable=self.frac_short_thr, width=6, font=(UI, 10), relief='flat', justify='center',
                 bg=T['panel'], fg=T['ink'], highlightthickness=1, highlightbackground=T['line'],
                 highlightcolor=T['accent']).pack(side='left', padx=(4, 16), ipady=2)
        tk.Label(trow, text="frac_implaus \u2264", bg=T['paper'], fg=T['ink'], font=(UI, 10)).pack(side='left')
        tk.Entry(trow, textvariable=self.frac_implaus_thr, width=6, font=(UI, 10), relief='flat', justify='center',
                 bg=T['panel'], fg=T['ink'], highlightthickness=1, highlightbackground=T['line'],
                 highlightcolor=T['accent']).pack(side='left', padx=(4, 0), ipady=2)
        tk.Label(p, text="Reads 02_marked/ peaks \u2192 frac \u2192 Good / Bad (Bad if editor-flagged or frac over\n"
                         "threshold). Copies to 03_good/ or 03_bad/ and writes toolbox_qc_summary.csv.",
                 bg=T['paper'], fg=T['mute'], font=(UI, 9), justify='left').pack(anchor='w', pady=(10, 10))
        mkbtn(p, "Run discrimination", lambda: self._run(self._do_discriminate), kind='primary').pack(anchor='w')

    # ---- helpers ----
    def _log(self, m, end="\n"):
        self.log_box.config(state='normal'); self.log_box.insert('end', m + end)
        self.log_box.see('end'); self.log_box.config(state='disabled'); self.root.update_idletasks()

    def _status(self, m):
        self.status_lbl.config(text=m); self.root.update_idletasks()

    def _browse_dir(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _browse_raw(self):
        if self.raw_mode.get() == 'folder':
            d = filedialog.askdirectory()
            if d:
                self.raw_input.set(d)
                self.raw_files = sorted(p for p in Path(d).glob('**/*')
                                        if p.suffix.lower() in ('.set', '.bdf'))
        else:
            ps = filedialog.askopenfilenames(
                filetypes=[("EEG (.set, .bdf)", "*.set *.bdf"), ("EEGLAB", "*.set"),
                           ("BioSemi BDF", "*.bdf"), ("All", "*.*")])
            if ps:
                self.raw_files = list(ps); self.raw_input.set(f"{len(ps)} files selected")
        self.raw_count.config(text=f"{len(self.raw_files)} files")

    def _workpath(self, sub):
        return Path(self.work_dir.get()) / sub

    def _mark_in(self):
        s = self.mark_in_dir.get().strip()
        if s:
            return Path(s)
        return self._workpath(REC_DIR) if self.work_dir.get().strip() else None

    def _mark_out(self):
        s = self.mark_out_dir.get().strip()
        if s:
            return Path(s)
        return self._workpath(MARK_DIR) if self.work_dir.get().strip() else None

    def _run(self, fn):
        if self.is_running:
            return
        if not self.work_dir.get():
            messagebox.showwarning("Missing", "Choose a working folder"); return
        self.is_running = True; self._status("Running...")

        def inner():
            try:
                fn()
            except Exception as e:
                self._log(f"\n\u274C Error: {e}"); self._status("Error!")
            finally:
                self.is_running = False
        threading.Thread(target=inner, daemon=True).start()

    def _load_model(self):
        self._log("loading conv_only model ...")
        model = P.EEGtoECGModel(latent_dim=P.SELECTED_LATENT_DIM, decoder_schedule=P.SELECTED_DECODER, variant=P.SELECTED_VARIANT)
        ck = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        state = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
        model.load_state_dict({k.replace('model.', ''): v for k, v in state.items()})
        model.eval().to(DEVICE)
        self._log(f"model on {DEVICE.upper()}\n")
        return model

    def _do_reconstruct(self):
        if not (_HAS_TORCH and P is not None):
            self._log("Reconstruction needs PyTorch and the model pipeline, which are not in this build.\n"
                      "This is a rating build \u2014 use Stage 2 (Mark peaks) with the manual editor.")
            self._status("Ready"); return
        if not self.raw_files:
            self._log("no raw files selected"); self._status("Ready"); return
        if not os.path.exists(WEIGHTS_PATH):
            self._log(f"weights not found: {WEIGHTS_PATH}"); self._status("Ready"); return
        out = self._workpath(REC_DIR); out.mkdir(parents=True, exist_ok=True)
        model = self._load_model()
        self._log("=" * 64 + f"\nSTAGE 1: reconstruct {len(self.raw_files)} subject(s)\n" + "=" * 64)
        for i, f in enumerate(self.raw_files, 1):
            self._log(f"[{i}/{len(self.raw_files)}] {Path(f).name} ...", end=" ")
            try:
                self._log(stage1_reconstruct(f, model, DEVICE, out) + " ok")
            except Exception as e:
                self._log(f"FAILED ({e})")
        self._log(f"\n-> {out}"); self._status("Stage 1 done")

    def _on_mark(self):
        if self.is_running:
            return
        recon = self._mark_in()
        if recon is None:
            messagebox.showwarning("Missing", "Choose a working folder, or set 'Reconstructions to mark'"); return
        files = sorted(Path(recon).glob('*.set'))
        if not files:
            self._log(f"no reconstructed .set files in {recon}"); return
        out = self._mark_out()
        if out is None:
            messagebox.showwarning("Missing", "Set 'Save my marked peaks to' (or a working folder)"); return
        out.mkdir(parents=True, exist_ok=True)
        if self.mark_method.get() == 'editor':
            self._log("=" * 64 + f"\nSTAGE 2: manual editor on {len(files)} subject(s)\n" + "=" * 64)
            PeakEditor(self.root, [str(f) for f in files], str(out), SFREQ, log=self._log)
        else:
            self._run(self._do_mark_auto)

    def _do_mark_auto(self):
        recon = self._mark_in(); out = self._mark_out()
        files = sorted(Path(recon).glob('*.set')); method = self.mark_method.get()
        self._log("=" * 64 + f"\nSTAGE 2: mark peaks ({method}) on {len(files)} subject(s)\n" + "=" * 64)
        for i, f in enumerate(files, 1):
            self._log(f"[{i}/{len(files)}] {Path(f).name} ...", end=" ")
            try:
                _, npk = stage2_mark_auto(str(f), out, method); self._log(f"{npk} peaks")
            except Exception as e:
                self._log(f"FAILED ({e})")
        self._log(f"\n-> {out}"); self._status("Stage 2 done")

    def _do_discriminate(self):
        marked = self._workpath(MARK_DIR); files = sorted(marked.glob('*.set'))
        if not files:
            self._log(f"no marked files in {marked} (run Stage 2 first)"); self._status("Ready"); return
        try:
            fs_thr, fi_thr = float(self.frac_short_thr.get()), float(self.frac_implaus_thr.get())
        except ValueError:
            self._log("thresholds must be numbers"); self._status("Ready"); return
        good_dir, bad_dir = self._workpath(GOOD_DIR), self._workpath(BAD_DIR)
        keep = self.keep_mode.get()
        self._log("=" * 64 + f"\nSTAGE 3: discriminate {len(files)} subject(s)  keep={keep.upper()}\n" + "=" * 64)
        results = []
        for i, f in enumerate(files, 1):
            self._log(f"[{i}/{len(files)}] {Path(f).name} ...", end=" ")
            try:
                r = stage3_discriminate(str(f), fs_thr, fi_thr, keep, good_dir, bad_dir)
                results.append(r)
                flag = " (BAD-flagged)" if r['flagged_bad'] else ""
                self._log(f"{r['Decision']}{flag}  HR={r['apparent_HR']} fs={r['frac_short']} fi={r['frac_implaus']}  saved={r['Saved']}")
            except Exception as e:
                self._log(f"FAILED ({e})")
        if results:
            df = pd.DataFrame(results); df.to_csv(self._workpath('toolbox_qc_summary.csv'), index=False)
            self._log("")
            for s, c in df['Decision'].value_counts().items():
                self._log(f"{s:6s}: {c:3d} ({100 * c / len(df):4.1f}%)")
            self._log(f"saved {(df['Saved'] == 'yes').sum()} file(s); summary -> toolbox_qc_summary.csv")
        self._status("Stage 3 done")


if __name__ == '__main__':
    root = tk.Tk()
    app = ToolboxApp(root)
    root.mainloop()