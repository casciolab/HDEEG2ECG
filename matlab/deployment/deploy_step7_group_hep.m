%% deploy_step7_group_hep - Group HEP on the Healthy Brain Network cohort
%
% Shows that heartbeat-evoked potentials are recovered from RECONSTRUCTED
% R-peaks in the deployment cohort, where no ground-truth ECG exists.
%
% Without ground truth there is nothing to compare against, so the evidence is
% built from six strands. Note what each one can and cannot support: only
% strand 2 speaks to IDENTITY. Time-locking and replication are silent on it.
%
%   1. MORPHOLOGY. Grand-average waveform at a priori fronto-central ROIs and
%      its topography. The peak is taken from the REAL waveform, not from the
%      surrogate contrast (see the note at the difference-wave computation).
%
%   2. TRANSITIVE MORPHOLOGY. Agreement with the response measured against a
%      real ECG in our own validated cohort. This replaces comparison to the
%      published literature, which is near-unfalsifiable given how widely HEP
%      latency, polarity and topography vary across studies.
%
%   3. TIME-LOCKING. Cluster-based permutation contrast against a surrogate
%      built by circularly time-shifting each trial. Needs no ECG. Establishes
%      that the response is locked to the reconstructed R-peaks rather than an
%      artifact of averaging. At n in the thousands the p-value sits at the
%      permutation floor, so cluster extent and effect size carry the weight.
%
%   4. EFFECT SIZE, OUT OF SAMPLE. Clusters defined in one half of the cohort,
%      amplitude measured in the other, per cluster and never pooled across
%      polarity. Only possible because the cohort is large enough to split.
%
%   5. SPLIT-HALF REPLICATION of the DIFFERENCE WAVE. Raw-waveform agreement
%      would be near-1 whether or not an HEP is present, so it is reported only
%      as a contrast.
%
%   6. WHAT THIS DOES AND DOES NOT ESTABLISH. Printed explicitly, including the
%      circularity specific to EEG-derived triggers and why the validation
%      cohort is what answers it.
%
% RUNS IN TWO PASSES. Pass 1 reads ~80 GB of epoch files and caches one ERP
% (and one surrogate ERP) per subject; pass 2 does all the statistics from the
% cache. Re-running only redoes pass 2 unless the cache is deleted.
% WARNING: nothing checks that a stale cache matches the current config. Delete
% erp_cache.mat after changing jitterRange or the surrogate construction.
%
% Inputs:  outputs/deployment/<dataset>/deploy_step6_epoch\*.set   (128-channel epochs on reconstructed peaks)
%          outputs/adult_validation/true_rpeaks/adult_step6_epoch/*.set  (reference cohort, true ECG)
% Outputs: outputs/deployment/<dataset>/deploy_step7_group_hep\erp_cache.mat   (pass 1)
%          outputs/deployment/<dataset>/deploy_step7_group_hep\hep_group.mat   (pass 2)
%          hep_validated_vs_cohort / hep_replication / hep_effect_size
%          panel_*.{emf,pdf,png,fig}  - single panels for arranging in PowerPoint
%
% No scalp maps are produced. A cluster's extent depends on the contrast and the
% cluster-forming threshold rather than marking the boundary of a generator, and
% heartbeat-locked activity is recoverable over much of the scalp, so the channel
% set is used only as a fixed a priori ROI and no topography is claimed. The
% panel_topo() and bluewhitered() helpers are retained but uncalled, as are
% panel_waveforms() and label_panels(), so the earlier layouts can be restored
% without rewriting them.

%% CONFIG
clc
clear

fullpath = mfilename('fullpath');
basePath = fileparts(fullpath); % this pipeline's folder (any OS)
addpath(fileparts(basePath));   % for hdeeg2ecg_paths.m
dirs = hdeeg2ecg_paths();       % input, output and toolbox folders (see README)

% Toolboxes - EDIT if these are not already on the path
fieldtripPath = dirs.fieldtrip;
addpath(genpath(dirs.eeglab)) % EEGLAB 2020.0 with its plugins (see README)
if isempty(which('eeglab'))
    error('EEGLAB is not on the MATLAB path. Add it before running.');
end
addpath(genpath(fieldtripPath));
addpath(fullfile(fieldtripPath,'external','eeglab'));
ft_defaults;

epochPath = fullfile(dirs.deploy,'deploy_step6_epoch');
outPath   = fullfile(dirs.deploy,'deploy_step7_group_hep');
if ~exist(outPath,'dir'); mkdir(outPath); end
cacheFile = fullfile(outPath,'erp_cache.mat');

% --- CHANNEL LOCATIONS ---
% The deploy_step6_epoch files carry channel LABELS but no coordinates: X/Y/Z are NaN and
% the spherical/polar fields are 0. They are wiped somewhere in the deploy_step5_remove_ocular_ics
% step - deploy_step1_import and deploy_step2_preprocess still have them - which makes both
% ft_prepare_neighbours (delaunay on non-finite points) and topoplot fail.
%
% Fix: borrow the locations from an upstream file that still has them and match
% by channel label. deploy_step2_preprocess is used because it is the SAME cohort's own montage.
% Any .set with valid coordinates and E1..E128 labels would work; a surrogate-
% study file is a fallback but is a different recording, so prefer this one.
locDonorDir = fullfile(dirs.deploy,'deploy_step2_preprocess');

% --- REFERENCE COHORT (ground-truth-validated) ---
% Epochs from the labelled cohort, locked to TRUE R-peaks from a real ECG. Used
% for a transitive morphology check: comparing this cohort's response to the
% published literature is weak, because HEP morphology varies so much across
% studies that almost any fronto-central deflection matches something. Comparing
% it to a response measured against an actual ECG in our own hands is a specific,
% checkable claim - "the signal recovered at scale is the signal that was
% validated". Set to '' to skip.
%
% Expect a real difference regardless: that cohort is adult, this one paediatric,
% and development changes the HEP. A match is strong evidence; a mismatch is not
% fatal but must be reported as such.
refEpochDir = fullfile(dirs.adult,'true_rpeaks','adult_step6_epoch');

% --- SPATIAL DEFINITION: DATA-DRIVEN, NOT A PRIORI ---
% No a priori ROI is used. Channels come from permutation clusters instead.
%
% The problem that creates, and how it is handled: measuring amplitude in the
% same data that selected the cluster is double-dipping. Two independent masks
% are therefore used, and NEITHER is selected on the subjects it is measured in:
%
%   maskRef  - the significant cluster from the VALIDATED cohort (a different
%              dataset, epoched on true ECG R-peaks). Fully independent of this
%              cohort, so it carries the primary reported waveform, the
%              morphology peak and the split-half replication. Loaded from that
%              analysis' saved results and remapped onto this cohort's axes.
%   clustA/B - this cohort's own clusters, defined in one half and measured in
%              the other, for the within-cohort effect sizes.
%
% Consequence for the manuscript: "fronto-central" becomes a DESCRIPTION of
% where the cluster fell, not confirmation of a prior hypothesis. The a priori
% ROI definition in sec:hep_morphology no longer matches what is computed.
refValidationMat = fullfile(dirs.adult,'adult_step7_hep_validation','validation_results.mat');

% Analysis windows
statLatency = [0.2 0.6];     % s, window for the cluster test
hepWindow   = [0.2 0.6];     % s, window for ROI amplitude / peak measurement

% Surrogate jitter (seconds, magnitude; sign random)
jitterRange = [0.15 0.45];

% Cluster permutation settings
clusterAlpha     = 0.05;
alphaLevel       = 0.025;
minNbChan        = 3;
numRandomization = 1000;     % with n in the thousands, p hits the floor anyway

% Memory guard for the permutation test. FieldTrip holds all subjects x both
% conditions in memory at once, which at the full cohort is ~1 GB before
% randomisation overhead. Set to Inf to use every subject; reduce (e.g. 600) if
% MATLAB runs out of memory - power is not the limiting factor at this n.
permSubset = Inf;

rng(42,'twister');           % reproducible surrogate and split

%% PASS 1 - EXTRACT AND CACHE PER-SUBJECT ERPs
if exist(cacheFile,'file')
    fprintf('Cache found; skipping extraction. Delete %s to force a re-read.\n', cacheFile);
    load(cacheFile,'cache');
else
    epochFiles = dir(fullfile(epochPath,'*.set'));
    nSubj = numel(epochFiles);
    fprintf('Pass 1: extracting ERPs from %d subjects (this reads the whole cohort).\n', nSubj);

    cache = struct();
    cache.subjIDs = cell(1,nSubj);
    cache.nTrials = nan(1,nSubj);
    cache.erpReal = [];   % allocated once dimensions are known
    cache.erpSurr = [];
    cache.times   = [];
    cache.chanlocs = [];

    tic;
    for si = 1:nSubj
        fn = epochFiles(si).name;
        EEGe = pop_loadset('filename', fn, 'filepath', epochPath);

        if isempty(cache.times)
            cache.times    = EEGe.times/1000;    % s
            cache.chanlocs = EEGe.chanlocs;
            nCh = EEGe.nbchan; nT = numel(cache.times);
            cache.erpReal = nan(nCh, nT, nSubj, 'single');
            cache.erpSurr = nan(nCh, nT, nSubj, 'single');
        end

        % Surrogate: destroy R-peak locking by circularly shifting each trial
        nTr = size(EEGe.data,3);
        srate = EEGe.srate;
        surr = EEGe.data;
        for ti = 1:nTr
            jit = (jitterRange(1) + rand*(diff(jitterRange))) * sign(randn);
            surr(:,:,ti) = circshift(EEGe.data(:,:,ti), round(jit*srate), 2);
        end

        cache.erpReal(:,:,si) = single(mean(EEGe.data,3));
        cache.erpSurr(:,:,si) = single(mean(surr,3));
        cache.nTrials(si)     = nTr;

        id = regexp(fn,'NDAR[A-Z0-9]+','match','once');
        if isempty(id); id = fn; end
        cache.subjIDs{si} = id;

        if mod(si,100)==0 || si==nSubj
            fprintf('  %d/%d (%.1f min elapsed)\n', si, nSubj, toc/60);
        end
    end

    fprintf('Saving cache (this is a large file)...\n');
    save(cacheFile,'cache','-v7.3');
end

times    = cache.times;
nSubj    = numel(cache.subjIDs);
nCh      = size(cache.erpReal,1);

% Repair the channel locations (see the note in CONFIG). Done here rather than
% in the cache so an existing cache does not need rebuilding.
chanlocs = borrow_chanlocs(locDonorDir, {cache.chanlocs.labels});
fprintf('\n%d subjects, %d channels, %d samples (%.0f to %.0f ms).\n', ...
    nSubj, nCh, numel(times), times(1)*1000, times(end)*1000);
fprintf('Trials per subject: median %d (range %d-%d)\n', ...
    median(cache.nTrials), min(cache.nTrials), max(cache.nTrials));

%% GRAND AVERAGES AND DIFFERENCE WAVES
gaReal = double(mean(cache.erpReal, 3));
gaSurr = double(mean(cache.erpSurr, 3));

% NOT the HEP. The surrogate is built by circularly shifting each trial WITHIN
% the 900 ms epoch by 150-450 ms, so for any shift under ~400 ms the 200-600 ms
% test window receives genuinely R-peak-locked activity from a neighbouring
% latency. Averaged over random shifts the surrogate converges to gaReal
% convolved with the jitter density K, so this difference is (I - K)*gaReal - a
% high-pass-like FILTERED version of the real ERP.
%
% What that does and does not cost:
%   - The cluster test survives. A nonzero (I - K)*gaReal requires structure that
%     is NOT shift-invariant, i.e. time-locked structure, so leakage makes the
%     locking test CONSERVATIVE rather than liberal. Strand 2 stands.
%   - Latency and amplitude read off this wave do NOT transfer. The peak is the
%     extremum of the filtered wave, displaced from the true HEP peak, and its
%     amplitude is scaled by an unknown factor. Do not compare it to published
%     HEP amplitudes.
% Morphology claims are therefore taken from the REAL waveform below, and the
% difference wave is used only where a contrast against the null is wanted.
gaFiltDiff = gaReal - gaSurr;

hepIdx = find(times >= hepWindow(1) & times <= hepWindow(2));

%% REFERENCE COHORT AND ITS CLUSTER - the independent spatial definition
% Loads the validated cohort's grand average AND the channel x time mask of its
% significant cluster, both remapped onto this cohort's channel order and time
% axis. That mask was selected in a DIFFERENT dataset against a real ECG, so
% measuring in this cohort through it involves no circularity at all.
refGA_interp = []; refTimes = []; refN = 0; refAll_interp = [];
maskRef = false(nCh, numel(times));
if ~isempty(refEpochDir) && exist(refEpochDir,'dir')
    rf = dir(fullfile(refEpochDir,'*.set'));
    refN = numel(rf);
    refSum = []; refLabels = {}; refAll = [];
    for k = 1:refN
        R = pop_loadset('filename', rf(k).name, 'filepath', refEpochDir);
        if isempty(refSum)
            refTimes  = R.times/1000;
            refLabels = {R.chanlocs.labels};
            refSum    = zeros(R.nbchan, numel(refTimes));
            refAll    = zeros(R.nbchan, numel(refTimes), refN);
        end
        dk = mean(R.data,3);
        refSum = refSum + dk;
        refAll(:,:,k) = dk;      % kept per subject, for the standard-error band
    end
    if refN > 0
        refGA = refSum / refN;
        [tf, ix] = ismember({chanlocs.labels}, refLabels);
        if all(tf)
            refGA_interp = interp1(refTimes', refGA(ix,:)', times', 'linear', NaN)';
            refAll_interp = nan(nCh, numel(times), refN);
            for k = 1:refN
                refAll_interp(:,:,k) = ...
                    interp1(refTimes', refAll(ix,:,k)', times', 'linear', NaN)';
            end
        else
            warning('Reference channel labels do not match; transitive check skipped.');
            refN = 0;
        end
    end

    % The cluster mask from that cohort's own permutation test.
    if refN > 0 && exist(refValidationMat,'file')
        V = load(refValidationMat);
        vMask = V.results.roiMask;                       % nChan x nRefTime, logical
        if size(vMask,2) == numel(refTimes)
            vMask = vMask(ix,:);                         % reorder channels to match
            % Nearest-neighbour in time; leave false outside the reference range.
            for j = 1:numel(times)
                if times(j) >= min(refTimes) && times(j) <= max(refTimes)
                    [~, jr] = min(abs(refTimes - times(j)));
                    maskRef(:,j) = vMask(:,jr);
                end
            end
        else
            warning('Saved reference mask has %d time points but the reference epochs have %d; mask not used.', ...
                size(vMask,2), numel(refTimes));
        end
    end
end

% Without the adult outputs, use the group-level adult reference shipped with the
% repository (matlab/reference/adult_hep_reference.mat, made from those outputs by
% make_adult_hep_reference.m): grand average, cluster mask, SEM of the ROI waveform
% and split-half reliability, on the 256-Hz axis of the HBN analysis, in EGI
% HydroCel-128 channel order.
semRefShipped = []; relRefShipped = [];
refFile = fullfile(fileparts(basePath),'reference','adult_hep_reference.mat');
if refN == 0 && exist(refFile,'file') && nCh == 128
    A = load(refFile); A = A.ref;
    refTimes = A.times; refN = A.n;
    refGA_interp = interp1(A.times', A.ga', times', 'linear', NaN)';
    for j = 1:numel(times)
        if times(j) >= min(refTimes) && times(j) <= max(refTimes)
            [~, jr] = min(abs(refTimes - times(j)));
            maskRef(:,j) = A.mask(:,jr);
        end
    end
    semRefShipped = interp1(A.times, A.sem, times, 'linear', NaN);
    relRefShipped = A.reliability;
    fprintf('Adult reference: %s (group level, n = %d)\n', refFile, refN);
end

%% PRIMARY CHANNEL SET - from the reference cluster
% Channels are those implicated anywhere in the reference cluster. The waveform
% is then taken across the WHOLE epoch on those channels (not only inside the
% cluster's time span), so morphology outside the cluster window is visible.
primaryChans = find(any(maskRef,2))';
primarySrc   = 'reference-cohort cluster (independent)';
if isempty(primaryChans)
    % Fallback only. This IS circular for amplitude and latency, and is flagged
    % as such wherever it is reported.
    primaryChans = unique([clustFallbackChans(gaFiltDiff, hepIdx)]);
    primarySrc   = 'FALLBACK: largest-amplitude channels in THIS cohort (circular)';
    warning(['No reference cluster available, so the channel set was selected in this ' ...
             'cohort. Amplitude and latency from it are optimistically biased.']);
end

%% AMPLITUDE SCREEN - applied here, before anything is averaged or tested
% Peak of the channel-averaged RAW waveform inside the analysis window, on the
% a priori channels. Recordings above maxPeakAmp are dropped from every
% subsequent step: grand averages, cluster tests, split halves, effect sizes.
%
% It sits after primaryChans because the criterion is defined on that channel
% set, and inside the analysis window so that the legitimate QRS projection at
% t = 0, which reaches several microvolts in every recording, cannot trigger it.
%
% WHY THIS IS NEEDED, and why it is not a cosmetic exclusion. Twelve of 2,078
% recordings carry channel-averaged amplitudes an order of magnitude outside the
% physiological range for this measure, up to 107 uV against a cohort median of
% 1.1 uV. They barely move the grand average, which is what a mean over two
% thousand recordings is robust to: the peak shifts from -0.913 to -0.912 uV and
% the correlation with the validated cohort from 0.870 to 0.864. What they
% destroy is every quantity carrying a between-subject VARIANCE in its
% denominator. One recording alone accounts for 66% of the variance of the
% per-subject cluster mean, holding d_z at 0.27 while 88.7% of subjects sit on
% the group side of zero - two summaries of one effect that cannot both be
% right. With the screen, d_z is 0.78, skew falls from +24.9 to -0.3 and excess
% kurtosis from +931 to +44.
%
% The distribution stays heavy-tailed even so, so d_z and the sign-based
% proportion still do not coincide exactly (Phi(0.78) = 78% against 88.9%
% observed). That is a property of the data rather than a residual bug: the two
% measure different things, one sensitive to spread and one not, and the script
% reports both.
maxPeakAmp = 10;                                     % uV; Inf disables

rawSubjW = squeeze(mean(double(cache.erpReal(primaryChans,:,:)), 1))';   % nSubj x nT
peakSubj = max(abs(rawSubjW(:,hepIdx)), [], 2);
keepSubj = find(peakSubj < maxPeakAmp);

if isfinite(maxPeakAmp) && numel(keepSubj) < nSubj
    dropSubj = find(peakSubj >= maxPeakAmp);
    fprintf('\nAmplitude screen at %.0f uV (cohort median peak %.2f uV):\n', ...
        maxPeakAmp, median(peakSubj));
    fprintf('  %d of %d recordings excluded from ALL subsequent analyses.\n', ...
        numel(dropSubj), nSubj);
    [~, ord] = sort(peakSubj(dropSubj), 'descend');
    for k = ord(:)'
        fprintf('    %-14s peak %8.1f uV\n', ...
            cache.subjIDs{dropSubj(k)}, peakSubj(dropSubj(k)));
    end

    cache.erpReal = cache.erpReal(:,:,keepSubj);
    cache.erpSurr = cache.erpSurr(:,:,keepSubj);
    cache.subjIDs = cache.subjIDs(keepSubj);
    cache.nTrials = cache.nTrials(keepSubj);
    nSubj = numel(keepSubj);

    % The grand averages above were built from the full set, so rebuild them.
    gaReal     = double(mean(cache.erpReal, 3));
    gaSurr     = double(mean(cache.erpSurr, 3));
    gaFiltDiff = gaReal - gaSurr;
    fprintf('  %d recordings retained.\n', nSubj);
end

wReal = mean(gaReal(primaryChans,:), 1);
wSurr = mean(gaSurr(primaryChans,:), 1);
wDiff = mean(gaFiltDiff(primaryChans,:), 1);

% Morphology peak: from the REAL waveform, the quantity comparable to published
% HEPs, on an independently defined channel set.
[~, relR] = max(abs(wReal(hepIdx)));
peakIdxReal = hepIdx(relR);
peakLatReal = times(peakIdxReal);

% Contrast peak: from the filtered difference wave. Used for the difference
% topographies and the replication statistics, never for a morphology claim.
[~, rel] = max(abs(wDiff(hepIdx)));
peakIdx  = hepIdx(rel);
peakLat  = times(peakIdx);

% Leakage diagnostic: how much time-locked signal the "null" still carries.
leakRatio = max(abs(wSurr(hepIdx))) / max(abs(wReal(hepIdx)));

%% TRANSITIVE MORPHOLOGY - this cohort vs the validated one, same channels
rRefWave = NaN; rRefTopo = NaN; refPeakLat = NaN; refW = []; semRef = [];
if refN > 0 && ~isempty(refGA_interp)
    refW = mean(refGA_interp(primaryChans,:), 1);
    if ~isempty(refAll_interp)
        refSubjW = squeeze(mean(refAll_interp(primaryChans,:,:), 1))';   % refN x nT
        semRef   = std(double(refSubjW), 0, 1, 'omitnan') / sqrt(refN);
    elseif ~isempty(semRefShipped)
        semRef   = semRefShipped;      % shipped reference: group-level SEM
    end
    ok = hepIdx(~isnan(refW(hepIdx)));
    rRefWave = corr(wReal(ok)', refW(ok)');
    [~, rr] = max(abs(refW(ok)));
    refPeakIdx = ok(rr);
    refPeakLat = times(refPeakIdx);
    good = ~isnan(refGA_interp(:,refPeakIdx));
    rRefTopo = corr(gaReal(good,refPeakIdx), refGA_interp(good,refPeakIdx));
end

%% SPLIT THE COHORT
% One half defines where the effect is; the other measures how big it is. This
% removes the circularity of quantifying an effect on a mask chosen from the
% same data, and is only possible because the cohort is large enough for each
% half to stand alone.
%
% The partition is drawn from its OWN random stream, not the global one. Pass 1
% consumes two global draws per trial when building the surrogate, so a global
% randperm here would return a different split depending on whether the cache
% was built in this run or loaded - meaning the split, and every number derived
% from it, could never be reproduced.
splitStream = RandStream('twister','Seed',12345);
perm  = randperm(splitStream, nSubj);
halfA = sort(perm(1:floor(nSubj/2)));
halfB = sort(perm(floor(nSubj/2)+1:end));

%% CLUSTER PERMUTATION - full cohort, and each half separately
cfgN = [];
cfgN.elec = build_elec(chanlocs);
cfgN.method = 'triangulation';
cfgN.compress = 'yes';
cfgN.feedback = 'no';
neighbours = ft_prepare_neighbours(cfgN);

runPerm = @(idx) run_cluster_test(cache, idx, times, chanlocs, neighbours, ...
    statLatency, clusterAlpha, alphaLevel, minNbChan, numRandomization);

fullIdx = 1:nSubj;
if isfinite(permSubset) && permSubset < nSubj
    fullIdx = sort(randsample(splitStream, nSubj, permSubset))';
    fprintf('\nFull-cohort test on a random subset of %d subjects (permSubset).\n', numel(fullIdx));
end

fprintf('\n=== HEP vs surrogate, full cohort (n = %d) ===\n', numel(fullIdx));
statHEP = runPerm(fullIdx);

fprintf('\n=== HEP vs surrogate, half A (n = %d) ===\n', numel(halfA));
statA = runPerm(halfA);

fprintf('\n=== HEP vs surrogate, half B (n = %d) ===\n', numel(halfB));
statB = runPerm(halfB);

%% EFFECT SIZE, MEASURED OUT OF SAMPLE
% Amplitude is averaged over the exact channel x time cluster MASK, not over the
% whole HEP window. A cluster is sign-consistent by construction; a fixed wide
% window is not, so averaging across it cancels the positive and negative phases
% of the difference wave and drives the effect size to zero however strong the
% response actually is.
% Clusters are selected at alphaLevel, NOT at 0.05. With cfg.tail = 0 and
% cfg.correcttail unset, FieldTrip returns one-sided per-tail cluster
% probabilities, so alphaLevel = 0.025 is the manual halving that buys a
% two-tailed FWER of 0.05. Thresholding at 0.05 here would double the effective
% FWER to 0.10 and would admit clusters into the effect-size masks that the
% script's own cfg.alpha rejects.
clustA = get_clusters(statA,   times, {chanlocs.labels}, alphaLevel);
clustB = get_clusters(statB,   times, {chanlocs.labels}, alphaLevel);
clustF = get_clusters(statHEP, times, {chanlocs.labels}, alphaLevel);

% Cross-measured: clusters from A applied to B's subjects, and vice versa.
resBonA = measure_clusters(cache, halfB, clustA);
resAonB = measure_clusters(cache, halfA, clustB);

% Full cohort in-sample, for description only - its effect size is optimistically
% biased because the clusters were selected on these same subjects. Measured on
% fullIdx rather than 1:nSubj so that when permSubset is finite it is not
% evaluated on a superset of the subjects that defined the clusters.
resFull = measure_clusters(cache, fullIdx, clustF);

% FULLY INDEPENDENT: the cluster comes from the validated cohort - a different
% dataset, different subjects, epoched on a real ECG - and is measured across
% every subject here. Nothing about this mask was chosen using these data, so it
% is the least assailable effect size available.
resRef = struct('perCluster',[],'pooledDz',NaN,'pooledAmp',NaN,'pooledCons',NaN);
if any(maskRef(:))
    refClust = struct('mask', maskRef, 'sign', sign(mean(gaFiltDiff(maskRef))), ...
                      'prob', NaN, 'nPoints', nnz(maskRef));
    resRef = measure_clusters(cache, 1:nSubj, refClust);
end

%% SPLIT-HALF REPLICATION - of the DIFFERENCE WAVE
% Correlating the raw waveforms between halves would mostly show that two
% thousand-subject averages of the same cohort resemble each other, which is
% near-guaranteed and says nothing about the HEP. The difference wave is what
% has to replicate.
diffA = double(mean(cache.erpReal(:,:,halfA),3) - mean(cache.erpSurr(:,:,halfA),3));
diffB = double(mean(cache.erpReal(:,:,halfB),3) - mean(cache.erpSurr(:,:,halfB),3));
wDiffA = mean(diffA(primaryChans,:), 1);
wDiffB = mean(diffB(primaryChans,:), 1);

rWaveHalves = corr(wDiffA(hepIdx)', wDiffB(hepIdx)');

% Topography agreement, cross-measured. Using a latency picked from the POOLED
% grand average would make this a within-sample statistic, since both halves
% contributed to choosing it. Each half's own peak latency is used to index the
% OTHER half, and both directions are reported.
[~, ra] = max(abs(wDiffA(hepIdx))); idxA = hepIdx(ra);
[~, rb] = max(abs(wDiffB(hepIdx))); idxB = hepIdx(rb);
rTopoAonB = corr(diffA(:,idxB), diffB(:,idxB));   % latency from B, both topos at it
rTopoBonA = corr(diffA(:,idxA), diffB(:,idxA));   % latency from A
rTopoHalves = mean([rTopoAonB rTopoBonA]);
peakLatA = times(idxA);
peakLatB = times(idxB);

% For reference, the same correlation on the RAW waveforms - expected to be high
% whether or not an HEP is present, and reported only to make that contrast
% explicit.
rawA = double(mean(cache.erpReal(:,:,halfA),3));
rawB = double(mean(cache.erpReal(:,:,halfB),3));
wRawA = mean(rawA(primaryChans,:), 1);
wRawB = mean(rawB(primaryChans,:), 1);
rWaveRaw = corr(wRawA(hepIdx)', wRawB(hepIdx)');

%% CONSOLE SUMMARY
fprintf('\n======= HEP ON THE DEPLOYMENT COHORT (n = %d) =======\n', nSubj);

fprintf('\n1. MORPHOLOGY\n');
fprintf('   Channel set (%d channels), from %s:\n     %s\n', ...
    numel(primaryChans), primarySrc, mat2str(primaryChans));
fprintf('   HEP peak (real waveform):        %.0f ms (%+.3f uV)  <- report THIS for morphology\n', ...
    peakLatReal*1000, wReal(peakIdxReal));
fprintf('   Peak of the real-minus-surrogate contrast: %.0f ms (%+.3f uV)\n', ...
    peakLat*1000, wDiff(peakIdx));
fprintf('   Surrogate leakage: max|surrogate| / max|real| in the window = %.2f\n', leakRatio);
fprintf('      -> The surrogate shifts each trial by only %.0f-%.0f ms inside a %.0f ms epoch, so\n', ...
    jitterRange(1)*1000, jitterRange(2)*1000, (times(end)-times(1))*1000);
fprintf('         it still carries R-peak-locked activity. The contrast is therefore a FILTERED\n');
fprintf('         version of the real ERP, not the isolated HEP: its latency is displaced and\n');
fprintf('         its amplitude scaled by an unknown factor. Quote the real-waveform peak when\n');
fprintf('         comparing to published HEP amplitudes, and treat the contrast amplitude and\n');
fprintf('         the effect sizes below as LOWER BOUNDS.\n');
fprintf('      -> Do NOT summarise the effect by averaging over %.0f-%.0f ms: the wave changes\n', ...
    hepWindow(1)*1000, hepWindow(2)*1000);
fprintf('         sign inside that window, so the average cancels to near zero regardless of\n');
fprintf('         how strong the response is. Use the per-cluster values below.\n');

fprintf('\n2. TRANSITIVE MORPHOLOGY (vs the ground-truth-validated cohort)\n');
if isnan(rRefWave)
    fprintf('   Skipped - reference cohort unavailable.\n');
else
    fprintf('   Reference: %d subjects, epochs locked to TRUE R-peaks from a real ECG\n', refN);
    fprintf('   ROI waveform agreement, %.0f-%.0f ms:  r = %.3f\n', ...
        hepWindow(1)*1000, hepWindow(2)*1000, rRefWave);
    fprintf('   Topography agreement at the reference peak (%.0f ms): r = %.3f\n', ...
        refPeakLat*1000, rRefTopo);
    fprintf('      -> This is the identity argument worth making: the response recovered here\n');
    fprintf('         from reconstructed peaks matches one measured against an actual ECG in\n');
    fprintf('         our own hands. Comparing instead to the published literature would be\n');
    fprintf('         near-unfalsifiable, since HEP latency, polarity and topography vary so\n');
    fprintf('         widely across studies that almost any fronto-central deflection matches.\n');
    fprintf('      -> Caveat to state explicitly: the reference cohort is ADULT and this one\n');
    fprintf('         paediatric, so development predicts some genuine difference. A high r is\n');
    fprintf('         strong; a low one is informative rather than fatal.\n');
end

fprintf('\n3. TIME-LOCKING TO THE RECONSTRUCTED R-PEAKS\n');
fprintf('   Full cohort: %s\n', cluster_phrase(statHEP, clustF, numel(fullIdx)));
fprintf('   Half A:      %s\n', cluster_phrase(statA,   clustA, numel(halfA)));
fprintf('   Half B:      %s\n', cluster_phrase(statB,   clustB, numel(halfB)));
fprintf('   (clusters selected at p < %.3f, the one-sided per-tail threshold that gives a\n', alphaLevel);
fprintf('    two-tailed FWER of %.2f - not at 0.05)\n', 2*alphaLevel);
fprintf('      -> At this n the p-value sits at the permutation floor and is not the\n');
fprintf('         informative quantity; the cluster extent and effect size below are.\n');

fprintf('\n4. EFFECT SIZE, MEASURED OUT OF SAMPLE\n');
fprintf('   Amplitudes are averaged WITHIN each cluster separately. Pooling positive and\n');
fprintf('   negative clusters together would make them cancel against each other.\n');
report_effect('   Cluster from the VALIDATED COHORT, measured in all subjects here', resRef);
report_effect('   Clusters from A, measured in B', resBonA);
report_effect('   Clusters from B, measured in A', resAonB);
report_effect('   Full cohort, IN-SAMPLE (biased, not for reporting)', resFull);
fprintf('      -> The first block is the least assailable: the mask was selected in a\n');
fprintf('         different dataset, against a real ECG, so nothing about it was chosen\n');
fprintf('         using these data. The two cross-measured blocks come next - their\n');
fprintf('         clusters come from subjects who contribute nothing to the measurement.\n');

fprintf('\n5. SPLIT-HALF REPLICATION OF THE DIFFERENCE WAVE (n = %d and %d)\n', numel(halfA), numel(halfB));
fprintf('   Difference-wave agreement, ROI:        r = %.3f\n', rWaveHalves);
fprintf('   Topography, cross-measured latencies:  r = %.3f (A''s peak %.0f ms) / %.3f (B''s peak %.0f ms)\n', ...
    rTopoBonA, peakLatA*1000, rTopoAonB, peakLatB*1000);
fprintf('   (raw waveform agreement, for contrast: r = %.3f)\n', rWaveRaw);
fprintf('      -> The raw-waveform figure is near-1 whether or not an HEP is present, since\n');
fprintf('         two %d-subject averages of the same cohort must resemble each other. Only\n', numel(halfA));
fprintf('         the DIFFERENCE-wave agreement is evidence that the HEP itself replicates.\n');

fprintf('\n6. WHAT THIS DOES AND DOES NOT ESTABLISH\n');
fprintf('   Establishes: a replicable, heartbeat-locked fronto-central response, recovered\n');
fprintf('   from reconstructed peaks whose accuracy was independently validated against a\n');
fprintf('   ground-truth ECG, with morphology matching that validated response.\n');
fprintf('\n   Does NOT establish that the response is an HEP in the strict sense. Time-locking\n');
fprintf('   (strand 3) and replication (strand 5) are silent on identity, and morphology\n');
fprintf('   matching against the literature is close to unfalsifiable given how variable\n');
fprintf('   published HEPs are. Cardiac components are retained in this pipeline, so a\n');
fprintf('   cardiac-field contribution is not excluded either.\n');
fprintf('\n   The circularity that matters for a reconstruction-based HEP: the triggers are\n');
fprintf('   DERIVED FROM THE EEG, so epoching that same EEG on them could in principle\n');
fprintf('   manufacture a time-locked average with no cardiac response at all. No surrogate\n');
fprintf('   test can detect this. What rules it out is the validation cohort, where the\n');
fprintf('   reconstructed peaks were shown to coincide with true ECG R-peaks - so the\n');
fprintf('   triggers mark real cardiac events rather than convenient EEG features. Cite that\n');
fprintf('   explicitly when making the claim; it is load-bearing, not a formality.\n');
fprintf('\n   Suggested claim wording:\n');
fprintf('     "a replicable, heartbeat-locked fronto-central response with the morphology of\n');
fprintf('      the HEP, recovered from reconstructed R-peaks whose accuracy was independently\n');
fprintf('      validated against ground-truth ECG"\n');
fprintf('\n   The strongest remaining step is a brain-behaviour association: covariation with a\n');
fprintf('   phenotypic measure is not explicable by cardiac fields or trigger placement, and\n');
fprintf('   it is where the scientific finding lives. Not computed here.\n');

%% SAVE
hep = struct();
hep.subjIDs   = {cache.subjIDs};
hep.nTrials   = cache.nTrials;
hep.times     = times;
hep.gaReal    = gaReal;
hep.gaSurr    = gaSurr;
hep.primaryChans = primaryChans;
hep.primarySrc   = primarySrc;
hep.maskRef   = maskRef;
hep.resRef    = resRef;
hep.gaFiltDiff  = gaFiltDiff;
hep.leakRatio   = leakRatio;
hep.peakLatReal = peakLatReal;
hep.peakIdxReal = peakIdxReal;
hep.rTopoAonB   = rTopoAonB;
hep.rTopoBonA   = rTopoBonA;
hep.peakLatA    = peakLatA;
hep.peakLatB    = peakLatB;
hep.alphaLevel  = alphaLevel;
hep.refN        = refN;
hep.refW      = refW;
hep.semRef    = semRef;        % for the SEM band in downstream figures
hep.refAll    = refAll_interp;  % per-subject validated ERPs, for split-half reliability
hep.relRef    = relRefShipped;  % their reliability, when only the shipped reference was available
hep.refPeakLat  = refPeakLat;
hep.rRefWave    = rRefWave;
hep.rRefTopo    = rRefTopo;
hep.peakLat   = peakLat;
hep.statHEP   = statHEP;
hep.statA     = statA;
hep.statB     = statB;
hep.clustA    = clustA;
hep.clustB    = clustB;
hep.clustF    = clustF;
hep.resBonA   = resBonA;
hep.resAonB   = resAonB;
hep.resFull   = resFull;
hep.halfA     = halfA;
hep.halfB     = halfB;
hep.diffA     = diffA;
hep.diffB     = diffB;
hep.rWaveHalves = rWaveHalves;
hep.rTopoHalves = rTopoHalves;
hep.rWaveRaw    = rWaveRaw;
save(fullfile(outPath,'hep_group.mat'), 'hep', '-v7.3');

%% FIGURES
% Every panel is written once as a function of an axes handle and rendered both
% as its own figure and as a standalone file, so panels can be rearranged freely
% in PowerPoint without re-plotting.
%
% Export formats per figure:
%   .emf  vector, and the format PowerPoint can ungroup and edit natively
%   .pdf  vector, for LaTeX
%   .png  300 dpi, for quick viewing and for pasting where vector is unwanted
%   .fig  MATLAB, for later tweaking
% All panels are waveforms and export as fully editable vector.

S = fig_style();

% --- Quantities the panels need ---
% Per-subject ROI waveforms, for standard-error bands.
wSubjReal = squeeze(mean(cache.erpReal(primaryChans,:,:),1))';   % nSubj x nT
wSubjSurr = squeeze(mean(cache.erpSurr(primaryChans,:,:),1))';
semReal = std(double(wSubjReal),0,1)/sqrt(nSubj);
semSurr = std(double(wSubjSurr),0,1)/sqrt(nSubj);
semDiff = std(double(wSubjReal)-double(wSubjSurr),0,1)/sqrt(nSubj);

% Time points at which any ROI channel sits inside a significant cluster. Sign
% does not matter here - this marks WHERE the test was significant, it is not an
% amplitude average, so the pos/neg union is legitimate for display.
sigAny = false(nCh, numel(times));
for c = 1:numel(clustF); sigAny = sigAny | clustF(c).mask; end
sigTimeChans = any(sigAny(primaryChans,:),1);

% --- FIGURE 1: the HEP in the validated cohort and in this one ---
% One panel, deliberately. The comparison the manuscript makes is between two
% grand averages measured through the SAME a priori channel set, so putting them
% on one pair of axes is the whole result; a scalp map alongside would invite a
% localisation reading that a cluster-defined channel set cannot support, and the
% surrogate belongs to the significance test rather than to the morphology.
% Nothing on this panel marks a validated-vs-cohort difference: the point is that
% the two share an HEP morphology, and a difference between them is expected on
% sample size and age alone.
f1 = new_fig(S, 6.2, 4.0);
ax1 = axes(f1);
panel_hep_compare(ax1, S, times, wReal, semReal, refW, semRef, refN, nSubj, ...
    primaryChans, peakLatReal, refPeakLat, rRefWave, statLatency);
export_all(f1, outPath, 'hep_validated_vs_cohort');

% --- FIGURE 2: split-half replication ---
f2 = new_fig(S, 6.2, 4.0);
ax2 = axes(f2);
panel_replication(ax2, S, times, wDiffA, wDiffB, numel(halfA), numel(halfB), rWaveHalves);
export_all(f2, outPath, 'hep_replication');

% --- FIGURE 3: out-of-sample effect sizes ---
f3 = new_fig(S, 7.4, 3.4);
tl3 = tiledlayout(f3,1,1,'TileSpacing','compact','Padding','compact');
panel_effectsize(nexttile(tl3), S, resBonA, resAonB);
export_all(f3, outPath, 'hep_effect_size');

% --- Standalone single-panel versions, for free arrangement in PowerPoint ---
standalone(S, outPath, 'panel_hep_compare', 6.2, 4.0, @(ax) ...
    panel_hep_compare(ax, S, times, wReal, semReal, refW, semRef, refN, nSubj, ...
        primaryChans, peakLatReal, refPeakLat, rRefWave, statLatency));
standalone(S, outPath, 'panel_difference', 4.6, 3.2, @(ax) ...
    panel_difference(ax, S, times, wDiff, semDiff, sigTimeChans, peakLat, peakIdx));
standalone(S, outPath, 'panel_replication', 4.6, 3.2, @(ax) ...
    panel_replication(ax, S, times, wDiffA, wDiffB, numel(halfA), numel(halfB), rWaveHalves));
standalone(S, outPath, 'panel_effectsize', 6.4, 3.4, @(ax) ...
    panel_effectsize(ax, S, resBonA, resAonB));

fprintf('\nFigures written to %s\n', outPath);
fprintf('  hep_validated_vs_cohort, hep_replication, hep_effect_size\n');
fprintf('  standalone panels: panel_*.emf for PowerPoint (ungroup to edit)\n');
fprintf('  each as .emf (editable vector) / .pdf (LaTeX) / .png (300 dpi) / .fig\n');
fprintf('  No scalp maps are produced: cluster extent reflects the contrast and the\n');
fprintf('  cluster-forming threshold, not the boundary of a generator, so the channel\n');
fprintf('  set is used only as a fixed ROI. panel_topo() is retained but uncalled.\n');

fprintf('\nDone. Results saved to %s\n', outPath);

%% ===================== FIGURE HELPERS =====================

function ch = clustFallbackChans(gaFiltDiff, hepIdx)
    % Last resort when no independent mask exists: the eight channels with the
    % largest mean |contrast| in the analysis window. Selected in the same data
    % it will be measured in, hence circular - only used so the script still runs.
    m = mean(abs(gaFiltDiff(:,hepIdx)), 2);
    [~, ord] = sort(m, 'descend');
    ch = sort(ord(1:min(8,numel(ord))))';
end

function S = fig_style()
    % One place for every visual choice, so all panels match.
    S = struct();
    S.font    = 'Arial';
    S.fsAxis  = 9;
    S.fsLabel = 10;
    S.fsTitle = 10;
    S.fsPanel = 12;
    S.lw      = 1.7;
    S.lwThin  = 1.2;
    % Okabe-Ito, colour-blind safe.
    S.cReal = [0.10 0.10 0.10];      % near-black
    S.cSurr = [0.55 0.55 0.55];      % grey
    S.cRef  = [0.00 0.62 0.45];      % bluish green
    S.cDiff = [0.00 0.45 0.70];      % blue
    S.cA    = [0.00 0.45 0.70];      % blue
    S.cB    = [0.84 0.37 0.00];      % vermillion
    S.cSig   = [0.20 0.20 0.20];
    S.cShade = [0.91 0.91 0.91];     % significant-latency shading
    S.alpha  = 0.20;                 % SEM band opacity
    % Topography display radius. HydroCel inferior-rim electrodes sit at
    % radius > 0.5, i.e. below the standard head circle (cheek, jaw, neck).
    % Left unrestricted they dominate the colour scale and are interpolated
    % across sparsely, which reads as a large field where the data are weakest.
    % 0.5 confines the map to the scalp proper; those channels remain in the
    % ANALYSIS, only the display is bounded.
    S.plotrad = 0.5;
end

function f = new_fig(S, wIn, hIn)
    f = figure('Color','w','Units','inches','Position',[1 1 wIn hIn], ...
               'PaperPositionMode','auto','InvertHardcopy','off');
    set(f,'DefaultAxesFontName',S.font,'DefaultAxesFontSize',S.fsAxis, ...
          'DefaultTextFontName',S.font);
end

function style_axes(ax, S)
    set(ax,'FontName',S.font,'FontSize',S.fsAxis,'TickDir','out', ...
           'Box','off','LineWidth',0.8,'Layer','top','TickLength',[0.015 0.015]);
end

function band(ax, x, y, e, col, a)
    % Shaded mean +/- SEM, drawn behind the mean line.
    xx = [x(:); flipud(x(:))];
    yy = [y(:)-e(:); flipud(y(:)+e(:))];
    ok = ~isnan(xx) & ~isnan(yy);
    patch(ax, xx(ok), yy(ok), col, 'FaceAlpha', a, 'EdgeColor','none');
end

function mark_sig(ax, S, tms, sigMask)
    % Significant latencies as shaded bands behind the traces, matching the
    % convention used in the validation figures. Call AFTER the y-limits are
    % final; the patches are sent to the back so they never occlude data.
    if ~any(sigMask); return; end
    yl = ylim(ax);
    d  = diff([false sigMask false]);
    on = find(d == 1); off = find(d == -1) - 1;
    for k = 1:numel(on)
        x1 = tms(on(k)); x2 = tms(off(k));
        p = patch(ax, [x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], S.cShade, ...
                  'EdgeColor','none');
        uistack(p, 'bottom');
    end
    ylim(ax, yl);   % patches must not re-trigger autoscaling
end

function panel_waveforms(ax, S, times, wReal, semReal, wSurr, semSurr, ...
                         refW, refN, sigTimeChans, nSubj, primaryChans)
    axes(ax); hold(ax,'on');
    t = times*1000;
    band(ax, t, wSurr, semSurr, S.cSurr, S.alpha);
    band(ax, t, wReal, semReal, S.cReal, S.alpha);
    h = gobjects(0); lab = {};
    hs = plot(ax, t, wSurr, '-', 'Color', S.cSurr, 'LineWidth', S.lwThin);
    hr = plot(ax, t, wReal, '-', 'Color', S.cReal, 'LineWidth', S.lw);
    h(end+1) = hr; lab{end+1} = sprintf('Reconstructed peaks (n = %d)', nSubj);
    h(end+1) = hs; lab{end+1} = 'Surrogate';
    if ~isempty(refW)
        hf = plot(ax, t, refW, '--', 'Color', S.cRef, 'LineWidth', S.lw);
        h(end+1) = hf; lab{end+1} = sprintf('True ECG, validated (n = %d)', refN);
    end
    xline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    yline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    axis(ax,'tight'); yl = ylim(ax); ylim(ax, yl + [-0.12 0.10]*range(yl));
    mark_sig(ax, S, t, sigTimeChans);
    xlabel(ax, 'Time from R-peak (ms)', 'FontSize', S.fsLabel);
    ylabel(ax, 'Amplitude (\muV)', 'FontSize', S.fsLabel);
    title(ax, sprintf('Cluster-defined channels (n = %d)', numel(primaryChans)), ...
        'FontSize', S.fsTitle, 'FontWeight','normal');
    lg = legend(ax, h, lab, 'Location','southeast', 'FontSize', S.fsAxis-1);
    set(lg,'Box','off');
    style_axes(ax, S);
end

function panel_hep_compare(ax, S, times, wReal, semReal, refW, semRef, refN, ...
                           nSubj, primaryChans, peakLatReal, refPeakLat, ...
                           rRefWave, statLatency)
    % The HEP measured in the validated cohort against the HEP measured here,
    % through the same a priori channel set.
    %
    % Both curves carry a standard-error band. That matters more than it usually
    % does: n = 14 against n = 2000-odd means the two are estimated with very
    % different precision, and drawing a band on only the larger cohort - or on
    % neither - would let the reader read agreement into a curve that is simply
    % noisy.
    axes(ax); hold(ax,'on');
    t = times*1000;

    % Analysis window, drawn first so everything else sits on top of it.
    yl0 = [min([wReal(:); refW(:)]) max([wReal(:); refW(:)])];
    pad = 0.35*range(yl0) + eps;
    patch(ax, statLatency([1 2 2 1])*1000, [yl0(1)-pad yl0(1)-pad yl0(2)+pad yl0(2)+pad], ...
          S.cShade, 'EdgeColor','none', 'FaceAlpha',0.55);

    if ~isempty(refW)
        if ~isempty(semRef); band(ax, t, refW, semRef, S.cRef, S.alpha); end
    end
    band(ax, t, wReal, semReal, S.cReal, S.alpha);

    h = gobjects(0); lab = {};
    if ~isempty(refW)
        hf = plot(ax, t, refW, '--', 'Color', S.cRef, 'LineWidth', S.lw);
        h(end+1) = hf;
        lab{end+1} = sprintf('Validated cohort, true ECG (n = %d)', refN);
    end
    hr = plot(ax, t, wReal, '-', 'Color', S.cReal, 'LineWidth', S.lw);
    h(end+1) = hr;
    lab{end+1} = sprintf('This cohort, reconstructed peaks (n = %d)', nSubj);

    xline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    yline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    axis(ax,'tight'); yl = ylim(ax); ylim(ax, yl + [-0.16 0.12]*range(yl));

    % No significance marking here, deliberately. The only cluster test in this
    % script contrasts a cohort against its own surrogate, so a rule drawn under
    % two cohort curves would be read as a between-cohort difference test that
    % was never run. The two cohorts differ in size and in age and are expected
    % to differ in amplitude; what this panel asks is whether the waveform
    % recovered here carries the morphology of an HEP, which is what the overlay
    % and the waveform correlation show.

    xlabel(ax, 'Time from R-peak (ms)', 'FontSize', S.fsLabel);
    ylabel(ax, 'Amplitude (\muV)', 'FontSize', S.fsLabel);
    title(ax, sprintf('Heartbeat-evoked potential, %d a priori channels', ...
        numel(primaryChans)), 'FontSize', S.fsTitle, 'FontWeight','normal');

    % Correlation first: it is the quantity the panel exists to report.
    txt = {};
    if ~isnan(rRefWave)
        txt{end+1} = sprintf('waveform r = %.3f (%.0f-%.0f ms)', rRefWave, ...
                             statLatency(1)*1000, statLatency(2)*1000);
    end
    txt{end+1} = sprintf('peak %.0f ms (this cohort)', peakLatReal*1000);
    if ~isnan(refPeakLat)
        txt{end+1} = sprintf('peak %.0f ms (validated)', refPeakLat*1000);
    end
    txt = strjoin(txt, newline);
    text(ax, 0.975, 0.05, txt, 'Units','normalized', 'HorizontalAlignment','right', ...
        'VerticalAlignment','bottom', 'FontName',S.font, 'FontSize',S.fsAxis-1, ...
        'Color',[0.30 0.30 0.30]);

    lg = legend(ax, h, lab, 'Location','northeast', 'FontSize', S.fsAxis-1);
    set(lg,'Box','off');
    style_axes(ax, S);
end

function panel_difference(ax, S, times, wDiff, semDiff, sigTimeChans, peakLat, peakIdx)
    axes(ax); hold(ax,'on');
    t = times*1000;
    band(ax, t, wDiff, semDiff, S.cDiff, S.alpha);
    plot(ax, t, wDiff, '-', 'Color', S.cDiff, 'LineWidth', S.lw);
    yline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    xline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    axis(ax,'tight'); yl = ylim(ax); ylim(ax, yl + [-0.12 0.10]*range(yl));
    plot(ax, peakLat*1000, wDiff(peakIdx), 'v', 'MarkerSize',5, ...
        'MarkerFaceColor', S.cDiff, 'MarkerEdgeColor','none');
    mark_sig(ax, S, t, sigTimeChans);
    xlabel(ax, 'Time from R-peak (ms)', 'FontSize', S.fsLabel);
    ylabel(ax, 'Reconstructed - surrogate (\muV)', 'FontSize', S.fsLabel);
    title(ax, sprintf('Contrast, peak %.0f ms', peakLat*1000), ...
        'FontSize', S.fsTitle, 'FontWeight','normal');
    style_axes(ax, S);
end

function panel_replication(ax, S, times, dA, dB, nA, nB, r)
    axes(ax); hold(ax,'on');
    t = times*1000;
    plot(ax, t, dA, '-',  'Color', S.cA, 'LineWidth', S.lw);
    plot(ax, t, dB, '--', 'Color', S.cB, 'LineWidth', S.lw);
    yline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    xline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    axis(ax,'tight'); yl = ylim(ax); ylim(ax, yl + [-0.10 0.10]*range(yl));
    xlabel(ax, 'Time from R-peak (ms)', 'FontSize', S.fsLabel);
    ylabel(ax, 'Reconstructed - surrogate (\muV)', 'FontSize', S.fsLabel);
    title(ax, sprintf('Independent halves, r = %.3f', r), ...
        'FontSize', S.fsTitle, 'FontWeight','normal');
    lg = legend(ax, {sprintf('Half A (n = %d)',nA), sprintf('Half B (n = %d)',nB)}, ...
        'Location','southeast','FontSize',S.fsAxis-1);
    set(lg,'Box','off');
    style_axes(ax, S);
end

function panel_topo(ax, S, map, chanlocs, cLim, ttl, showBar)
    axes(ax); cla(ax);
    topoplot(map, chanlocs, 'maplimits', [-cLim cLim], ...
        'electrodes','off', 'numcontour', 4, 'shading','interp', 'whitebk','on', ...
        'plotrad', S.plotrad, 'headrad', S.plotrad);
    title(ax, ttl, 'FontSize', S.fsTitle, 'FontWeight','normal', 'FontName', S.font);
    colormap(ax, bluewhitered(64));
    if showBar
        cb = colorbar(ax, 'eastoutside');
        cb.Label.String = '\muV';
        set(cb,'FontName',S.font,'FontSize',S.fsAxis-1,'Box','off','TickDirection','out');
    end
end

function cm = bluewhitered(n)
    % Diverging blue-white-red, zero-centred so polarity reads correctly.
    if mod(n,2); n = n+1; end
    half = n/2;
    lo = [linspace(0.02,1,half)' linspace(0.19,1,half)' linspace(0.38,1,half)'];
    hi = [linspace(1,0.70,half)' linspace(1,0.09,half)' linspace(1,0.11,half)'];
    cm = [lo; hi];
end

function panel_effectsize(ax, S, resBonA, resAonB)
    % Horizontal bars, grouped by which half supplied the clusters. Clusters are
    % NOT paired across halves - the masks differ, so pairing by rank would imply
    % a correspondence that does not exist.
    axes(ax); hold(ax,'on');
    blocks = {resBonA, resAonB};
    names  = {'Clusters from A, measured in B', 'Clusters from B, measured in A'};
    cols   = {S.cA, S.cB};

    y = 0; yt = []; ytl = {};
    for b = 1:2
        pc = blocks{b}.perCluster;
        for c = 1:numel(pc)
            y = y + 1;
            barh(ax, y, pc(c).dz, 0.68, 'FaceColor', cols{b}, ...
                'EdgeColor','none', 'FaceAlpha', 0.85);
            text(ax, pc(c).dz + 0.03*sign(pc(c).dz), y, ...
                sprintf(' %.0f%%', pc(c).consistentPct), ...
                'FontName',S.font,'FontSize',S.fsAxis-1, ...
                'HorizontalAlignment', ternary(pc(c).dz>0,'left','right'), ...
                'VerticalAlignment','middle','Color',[0.3 0.3 0.3]);
            yt(end+1) = y; %#ok<AGROW>
            ytl{end+1} = sprintf('%s, %d pts', ternary(pc(c).sign>0,'pos','neg'), pc(c).nPoints); %#ok<AGROW>
        end
        % Pooled value for this block, offset slightly for emphasis
        y = y + 1;
        barh(ax, y, blocks{b}.pooledDz, 0.68, 'FaceColor', cols{b}, ...
            'EdgeColor', [0.15 0.15 0.15], 'LineWidth', 1.0);
        text(ax, blocks{b}.pooledDz + 0.03, y, sprintf(' %.0f%%', blocks{b}.pooledCons), ...
            'FontName',S.font,'FontSize',S.fsAxis-1,'Color',[0.3 0.3 0.3], ...
            'VerticalAlignment','middle');
        yt(end+1) = y; %#ok<AGROW>
        ytl{end+1} = 'POOLED'; %#ok<AGROW>
        y = y + 0.8;   % gap between blocks
    end

    xline(ax, 0, '-', 'Color',[0.3 0.3 0.3], 'LineWidth', 0.8);
    set(ax,'YTick',yt,'YTickLabel',ytl,'YDir','reverse');
    ylim(ax, [0.2 y]);
    xlabel(ax, 'Effect size (Cohen''s d_z), out of sample', 'FontSize', S.fsLabel);
    title(ax, 'Per-cluster effect, measured in held-out subjects', ...
        'FontSize', S.fsTitle, 'FontWeight','normal');
    xl = xlim(ax); xlim(ax, xl + [-0.10 0.18]*range(xl));
    text(ax, 0.98, 0.02, 'labels: % of subjects with the group-direction effect', ...
        'Units','normalized','HorizontalAlignment','right', ...
        'FontName',S.font,'FontSize',S.fsAxis-1,'Color',[0.45 0.45 0.45]);
    % Block legend
    hA = patch(ax, NaN, NaN, S.cA, 'EdgeColor','none');
    hB = patch(ax, NaN, NaN, S.cB, 'EdgeColor','none');
    lg = legend(ax, [hA hB], names, 'Location','southeast','FontSize',S.fsAxis-1);
    set(lg,'Box','off');
    style_axes(ax, S);
end

function label_panels(tl, S, letters)
    % Bold lower-case panel letters, journal style.
    axs = findobj(tl.Parent,'Type','axes');
    axs = flipud(axs);
    n = min(numel(letters), numel(axs));
    for k = 1:n
        text(axs(k), -0.16, 1.10, letters{k}, 'Units','normalized', ...
            'FontName',S.font,'FontSize',S.fsPanel,'FontWeight','bold', ...
            'HorizontalAlignment','left','VerticalAlignment','bottom');
    end
end

function standalone(S, outPath, name, wIn, hIn, drawFcn)
    f = new_fig(S, wIn, hIn);
    ax = axes(f);
    drawFcn(ax);
    export_all(f, outPath, name);
    close(f);
end

function export_all(f, outPath, name)
    % EMF is the one PowerPoint can ungroup and edit; PDF for LaTeX; PNG for
    % quick use; .fig to reopen in MATLAB.
    try
        exportgraphics(f, fullfile(outPath,[name '.png']), 'Resolution', 300);
        exportgraphics(f, fullfile(outPath,[name '.pdf']), 'ContentType','vector');
    catch
        print(f, fullfile(outPath,[name '.png']), '-dpng','-r300');
        print(f, fullfile(outPath,[name '.pdf']), '-dpdf','-vector');
    end
    print(f, fullfile(outPath,[name '.emf']), '-dmeta');
    savefig(f, fullfile(outPath,[name '.fig']));
end

%% ===================== LOCAL FUNCTIONS =====================

function chanlocs = borrow_chanlocs(donorDir, wantLabels)
    % Take channel coordinates from an upstream .set that still has them, and
    % return them in the order of wantLabels. Errors loudly rather than silently
    % returning something unusable, since a wrong montage would corrupt both the
    % neighbour structure and every topography.
    d = dir(fullfile(donorDir,'*.set'));
    if isempty(d)
        error('No donor .set found in %s - cannot recover channel locations.', donorDir);
    end
    s = load(fullfile(donorDir, d(1).name), '-mat');
    if isfield(s,'EEG'); E = s.EEG; else; E = s; end

    donorLabels = {E.chanlocs.labels};
    [tf, idx] = ismember(wantLabels, donorLabels);
    if ~all(tf)
        error('Donor %s is missing %d of the required channel labels (e.g. %s).', ...
            d(1).name, sum(~tf), wantLabels{find(~tf,1)});
    end
    chanlocs = E.chanlocs(idx);

    xs = [chanlocs.X]; ys = [chanlocs.Y]; zs = [chanlocs.Z];
    if numel(xs) ~= numel(chanlocs) || any(~isfinite(xs)) || any(~isfinite(ys)) || any(~isfinite(zs))
        error('Donor %s also has missing or non-finite coordinates; pick another donor file.', d(1).name);
    end
    fprintf('Channel locations borrowed from %s (%d channels matched by label).\n', ...
        d(1).name, numel(chanlocs));
end

function stat = run_cluster_test(cache, idx, times, chanlocs, neighbours, ...
                                 statLatency, clusterAlpha, alphaLevel, minNbChan, nRand)
    % Paired cluster permutation, real vs surrogate, for a subset of subjects.
    realFT = erp_to_ft(cache.erpReal, idx, times, chanlocs);
    surrFT = erp_to_ft(cache.erpSurr, idx, times, chanlocs);
    n = numel(idx);

    cfg = [];
    cfg.channel          = 'all';
    cfg.latency          = statLatency;
    cfg.method           = 'montecarlo';
    cfg.statistic        = 'depsamplesT';
    cfg.correctm         = 'cluster';
    cfg.clusteralpha     = clusterAlpha;
    cfg.clusterstatistic = 'maxsum';
    cfg.minnbchan        = minNbChan;
    cfg.neighbours       = neighbours;
    cfg.tail             = 0;
    cfg.clustertail      = 0;
    cfg.alpha            = alphaLevel;
    cfg.computecritval   = 'no';
    cfg.numrandomization = nRand;
    cfg.uvar             = 2;
    cfg.ivar             = 1;
    cfg.design           = [ones(1,n), 2*ones(1,n); 1:n, 1:n];

    stat = ft_timelockstatistics(cfg, realFT{:}, surrFT{:});
end

function clusters = get_clusters(stat, fullTime, fullLabels, alphaThresh)
    % One entry per SIGNIFICANT cluster, each with its own exact channel x time
    % mask mapped onto the full epoch time axis, its polarity and its p-value.
    %
    % Clusters are deliberately NOT merged into a single mask. A cluster is
    % sign-consistent within itself, but a union of positive and negative
    % clusters is not, so averaging amplitude over such a union makes them cancel
    % against one another and drives the measured effect toward zero however
    % strong each cluster is.
    if nargin < 4 || isempty(alphaThresh); alphaThresh = 0.025; end
    clusters = struct('mask',{},'sign',{},'prob',{},'nPoints',{});

    [~, tMap] = ismember(stat.time, fullTime);
    [~, cMap] = ismember(stat.label, fullLabels);
    if any(tMap == 0) || any(cMap == 0)
        error('Could not map the cluster mask back onto the full time/channel axes.');
    end

    add = @(labelmat, c, sgn, p) struct( ...
        'mask', place_mask(labelmat == c, cMap, tMap, numel(fullLabels), numel(fullTime)), ...
        'sign', sgn, 'prob', p, 'nPoints', nnz(labelmat == c));

    if isfield(stat,'posclusters') && ~isempty(stat.posclusters)
        for c = 1:numel(stat.posclusters)
            if stat.posclusters(c).prob < alphaThresh
                clusters(end+1) = add(stat.posclusterslabelmat, c, +1, stat.posclusters(c).prob); %#ok<AGROW>
            end
        end
    end
    if isfield(stat,'negclusters') && ~isempty(stat.negclusters)
        for c = 1:numel(stat.negclusters)
            if stat.negclusters(c).prob < alphaThresh
                clusters(end+1) = add(stat.negclusterslabelmat, c, -1, stat.negclusters(c).prob); %#ok<AGROW>
            end
        end
    end

    % Largest first, so the "strongest cluster" reported is the dominant one.
    if ~isempty(clusters)
        [~, ord] = sort([clusters.nPoints], 'descend');
        clusters = clusters(ord);
    end
end

function m = place_mask(sig, cMap, tMap, nCh, nT)
    m = false(nCh, nT);
    m(cMap, tMap) = sig;
end

function res = measure_clusters(cache, idx, clusters)
    % Per-cluster amplitude and effect size of (real - surrogate) for the given
    % subjects, plus a sign-corrected pooled value across clusters.
    res = struct('perCluster',[],'pooledDz',NaN,'pooledAmp',NaN,'pooledCons',NaN);
    if isempty(clusters); return; end

    nC = numel(clusters);
    n  = numel(idx);
    perSubjPerClust = nan(n, nC);

    for k = 1:n
        r = double(cache.erpReal(:,:,idx(k)));
        s = double(cache.erpSurr(:,:,idx(k)));
        d = r - s;
        for c = 1:nC
            perSubjPerClust(k,c) = mean(d(clusters(c).mask));
        end
    end

    pc = struct('sign',{},'prob',{},'nPoints',{},'amp',{},'dz',{},'consistentPct',{});
    for c = 1:nC
        v = perSubjPerClust(:,c);
        pc(c).sign    = clusters(c).sign;
        pc(c).prob    = clusters(c).prob;
        pc(c).nPoints = clusters(c).nPoints;
        pc(c).amp     = mean(v);
        pc(c).dz      = mean(v)/std(v);
        pc(c).consistentPct = 100*mean(sign(v) == sign(mean(v)));
    end
    res.perCluster = pc;

    % Sign-corrected pooling: flip each cluster into its own polarity before
    % combining, weighted by cluster size, so clusters reinforce rather than
    % cancel. Reported as a single headline number.
    w = [clusters.nPoints] / sum([clusters.nPoints]);
    sgn = [clusters.sign];
    pooledPerSubj = perSubjPerClust .* sgn;        % now all same expected polarity
    pooledPerSubj = pooledPerSubj * w(:);          % size-weighted combination
    res.pooledAmp  = mean(pooledPerSubj);
    res.pooledDz   = mean(pooledPerSubj)/std(pooledPerSubj);
    res.pooledCons = 100*mean(sign(pooledPerSubj) == sign(res.pooledAmp));
end

function report_effect(tag, res)
    fprintf('%s\n', tag);
    if isempty(res.perCluster)
        fprintf('       no significant clusters to measure\n'); return
    end
    fprintf('       %-4s %-8s %-8s %-10s %-7s %s\n', 'sign','p','points','amp(uV)','d_z','consistent');
    for c = 1:numel(res.perCluster)
        p = res.perCluster(c);
        fprintf('       %-4s %-8.4g %-8d %-+10.3f %-7.2f %.1f%%\n', ...
            ternary(p.sign>0,'pos','neg'), p.prob, p.nPoints, p.amp, p.dz, p.consistentPct);
    end
    fprintf('       pooled (sign-corrected, size-weighted): d_z = %.2f (%+.3f uV, %.1f%%)\n', ...
        res.pooledDz, res.pooledAmp, res.pooledCons);
end

function out = ternary(c,a,b)
    if c; out = a; else; out = b; end
end

function elec = build_elec(chanlocs)
    % FieldTrip electrode definition from EEGLAB chanlocs.
    elec = struct();
    elec.label   = {chanlocs.labels}';
    elec.chanpos = [ [chanlocs.X]' [chanlocs.Y]' [chanlocs.Z]' ];
    elec.elecpos = elec.chanpos;
    maxAbs = max(abs(elec.chanpos(:)));
    if maxAbs < 1
        elec.unit = 'm';
    elseif maxAbs < 20
        elec.unit = 'cm';
    else
        elec.unit = 'mm';
    end
end

function ftCells = erp_to_ft(erpArray, useIdx, times, chanlocs)
    % Wrap cached ERPs as minimal FieldTrip timelock structures.
    labels = {chanlocs.labels}';
    elec = struct();
    elec.label = labels;
    elec.chanpos = [ [chanlocs.X]' [chanlocs.Y]' [chanlocs.Z]' ];
    elec.elecpos = elec.chanpos;
    % Units matter to FieldTrip's geometry handling, and this cohort's
    % coordinates are in METRES (head radius ~0.084) rather than the cm EEGLAB
    % more usually carries - so detect rather than assume.
    maxAbs = max(abs(elec.chanpos(:)));
    if maxAbs < 1
        elec.unit = 'm';
    elseif maxAbs < 20
        elec.unit = 'cm';
    else
        elec.unit = 'mm';
    end

    ftCells = cell(1,numel(useIdx));
    for k = 1:numel(useIdx)
        d = struct();
        d.label   = labels;
        d.time    = times;
        d.avg     = double(erpArray(:,:,useIdx(k)));
        d.dimord  = 'chan_time';
        d.elec    = elec;
        ftCells{k} = d;
    end
end

function s = cluster_phrase(stat, clusters, n)
    % Count AND extent are both taken from `clusters`, i.e. from a single
    % threshold. Reading the count from one alpha and the extent from
    % stat.mask (built at FieldTrip's own cfg.alpha) let the two numbers on the
    % same line disagree.
    ps = [];
    if isfield(stat,'posclusters') && ~isempty(stat.posclusters)
        ps = [ps, [stat.posclusters.prob]];
    end
    if isfield(stat,'negclusters') && ~isempty(stat.negclusters)
        ps = [ps, [stat.negclusters.prob]];
    end

    if isempty(clusters)
        if isempty(ps)
            s = sprintf('no candidate clusters formed (n = %d)', n);
        else
            s = sprintf('no significant clusters (smallest p = %.4f of %d candidate(s), n = %d)', ...
                min(ps), numel(ps), n);
        end
        return
    end

    nPos = sum([clusters.sign] > 0);
    nNeg = sum([clusters.sign] < 0);
    s = sprintf('%d significant cluster(s) (%d pos, %d neg), smallest p = %.4g, %d channel x time points, n = %d', ...
        numel(clusters), nPos, nNeg, min([clusters.prob]), sum([clusters.nPoints]), n);
end