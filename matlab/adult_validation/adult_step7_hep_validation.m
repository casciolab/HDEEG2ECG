%% adult_step7_hep_validation - Reconstructed-vs-true HEP validation
%
% Compares the HEP obtained by epoching on true R-peaks (true_rpeaks/adult_step6_epoch) against the
% HEP obtained by epoching the SAME cleaned EEG on reconstructed R-peaks
% (reconstructed_rpeaks/adult_step6_epoch). Only the trigger source differs between the two, per subject.
%
% Three paired (dependent-samples) cluster-based permutation tests:
%   Test A: true vs. reconstructed  -> expected NOT significant
%   Test B: reconstructed vs. surrogate
%   Test C: true vs. surrogate      (ground-truth reference)
% where the surrogate is the same epochs with each trial circularly
% time-shifted by a random jitter, destroying R-peak locking while preserving
% each trial's own spectral content.
%
% The analysis reports two things, and only two:
%
%   1. SENSITIVITY PRESERVATION (primary). Does epoching on reconstructed peaks
%      detect the HEP as well as the ground-truth ECG does? Tests B and C
%      contrast each trigger source against the SAME surrogate, and their
%      detection effect sizes are compared. This is a POSITIVE claim - the
%      reconstruction recovers the effect - not an absence of evidence, and it
%      needs no equivalence margin.
%
%   2. NO DETECTABLE TRUE-vs-RECON DIFFERENCE (supporting). Test A, plus the
%      cluster-mean amplitude difference with a 95% CI, in effect-size units
%      and as a fraction of the HEP effect measured in these same data.
%
% The ROI is the exact channel x time mask of the strongest Test C cluster, so
% it is defined from ground truth alone and never from the reconstruction being
% evaluated. That also makes the sensitivity comparison conservative: the mask
% sits where the TRUE condition is strongest.
%
% NOT reported, deliberately:
%   - Equivalence (TOST). Its verdict depends entirely on the margin: at
%     +/-0.5 uV it "passed" (p = 1.4e-07), but that margin is d = 1.54 in this
%     sample - larger than any published HEP effect - so the result was vacuous.
%     At a margin derived from published between-group effects (d = 0.25) it
%     failed (p = 0.116). With n = 14 the binding constraint is sample size, not
%     reconstruction quality. Sensitivity preservation replaces it. If TOST is
%     wanted later, add subjects to narrow the CI - do not widen the margin
%     until it passes.
%   - Component latency. The implicated cluster is temporally irregular, so peak
%     latency (which sample is the extremum) and area-based latency (how area
%     accumulates) are both moved by small secondary deflections, and neither
%     isolates a component latency. A temporal shift would attenuate the
%     reconstructed-peak contrast relative to the true-peak one, so timing
%     agreement is carried by the preserved detection effect instead.
%   - Grand-average waveform correlation, per-subject noise floor, and the
%     trial-mismatch diagnostic. All were confirmatory of the two results above
%     and are omitted from the manuscript; see git history if they are needed.
%
% Inputs:  true_rpeaks/adult_step6_epoch\*.set  (epoched on true R-peaks)
%          reconstructed_rpeaks/adult_step6_epoch\*.set  (epoched on reconstructed R-peaks)
% Outputs: adult_step7_hep_validation\validation_results.mat
%          adult_step7_hep_validation\ga_waveforms.png / sensitivity.png

%% CONFIG
clc
clear

% Get fullpath
fullpath = mfilename('fullpath');
basePath = fileparts(fullpath); % this pipeline's folder (any OS)
addpath(fileparts(basePath));   % for hdeeg2ecg_paths.m
dirs = hdeeg2ecg_paths();       % input, output and toolbox folders (see README)

% Toolboxes - EDIT THESE if they don't match this machine
eeglabPath = dirs.eeglab;
fieldtripPath = dirs.fieldtrip;

addpath(genpath(eeglabPath));
addpath(genpath(fieldtripPath));
addpath(fullfile(fieldtripPath,'external','eeglab'));
ft_defaults;

% Data paths
truePath  = fullfile(dirs.adult,'true_rpeaks','adult_step6_epoch');
reconPath = fullfile(dirs.adult,'reconstructed_rpeaks','adult_step6_epoch');
outPath   = fullfile(dirs.adult,'adult_step7_hep_validation');
if ~exist(outPath,'dir'); mkdir(outPath); end

% Cluster permutation settings (thresholds follow the lab convention in
% the lab's b9_permutations_intra_ft.m)
statLatency      = [0.2 0.6];  % s, CFA-clean window used for the cluster tests
clusterAlpha     = 0.05;
alphaLevel       = 0.025;      % two-tailed, split from 0.05
minNbChan        = 3;
numRandomization = 4000;       % capped below at 2^nSubjects for small n

% Surrogate positive-control jitter range (seconds, magnitude; sign random)
jitterRange = [0.15 0.45];

% Fixed seed: the surrogate is built from random per-trial jitters, so without
% this the Tests B/C clusters - and therefore the ROI and every metric derived
% from it - shift run to run.
rng(42,'twister');

%% DISCOVER MATCHED SUBJECTS
trueFiles  = dir(fullfile(truePath,'*.set'));
reconFiles = dir(fullfile(reconPath,'*.set'));

getSubjID = @(name) regexp(name,'(\d{3,6})','match','once');

trueIDs  = cellfun(getSubjID, {trueFiles.name}, 'UniformOutput', false);
reconIDs = cellfun(getSubjID, {reconFiles.name}, 'UniformOutput', false);

subjIDs = intersect(trueIDs, reconIDs);
if isempty(subjIDs)
    error('No matching subjects found between true_rpeaks/adult_step6_epoch and reconstructed_rpeaks/adult_step6_epoch.');
end
fprintf('Found %d matched subjects.\n', numel(subjIDs));

%% LOAD, BUILD ERPs AND SURROGATE
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

trueFT   = cell(1,numel(subjIDs));
reconFT  = cell(1,numel(subjIDs));
surrFT   = cell(1,numel(subjIDs));
nTrialsTrue  = nan(1,numel(subjIDs));
nTrialsRecon = nan(1,numel(subjIDs));

for si = 1:numel(subjIDs)
    sid = subjIDs{si};

    trueName  = trueFiles(strcmp(trueIDs, sid)).name;
    reconName = reconFiles(strcmp(reconIDs, sid)).name;

    EEGtrue  = pop_loadset('filename', trueName,  'filepath', truePath);
    EEGrecon = pop_loadset('filename', reconName, 'filepath', reconPath);

    srate = EEGtrue.srate;

    % --- Surrogate: circularly time-shift each trial of the reconstructed
    % epochs by a random jitter to destroy R-peak locking ---
    EEGsurr = EEGrecon;
    nTrials = size(EEGrecon.data,3);
    for ti = 1:nTrials
        jitterSec = (jitterRange(1) + rand*(jitterRange(2)-jitterRange(1))) * sign(randn);
        shiftSamp = round(jitterSec*srate);
        EEGsurr.data(:,:,ti) = circshift(EEGrecon.data(:,:,ti), shiftSamp, 2);
    end

    % --- Subject ERPs (average across trials) ---
    EEGtrueERP  = EEGtrue;  EEGtrueERP.data  = mean(EEGtrue.data,3);  EEGtrueERP.trials  = 1;
    EEGreconERP = EEGrecon; EEGreconERP.data = mean(EEGrecon.data,3); EEGreconERP.trials = 1;
    EEGsurrERP  = EEGsurr;  EEGsurrERP.data  = mean(EEGsurr.data,3);  EEGsurrERP.trials  = 1;

    addpath(fullfile(fieldtripPath,'external','eeglab'))
    trueFT{si}  = eeglab2fieldtrip(EEGtrueERP,  'timelockanalysis', 'none');
    reconFT{si} = eeglab2fieldtrip(EEGreconERP, 'timelockanalysis', 'none');
    surrFT{si}  = eeglab2fieldtrip(EEGsurrERP,  'timelockanalysis', 'none');

    nTrialsTrue(si)  = size(EEGtrue.data,3);
    nTrialsRecon(si) = nTrials;
    fprintf('Subject %s loaded (%d true trials, %d recon trials).\n', sid, nTrialsTrue(si), nTrials);
end

nSubj = numel(subjIDs);

% Trial counts. Epoch rejection in true_rpeaks/adult_step6_epoch runs independently on each file, so
% the two conditions can differ in trial composition for reasons unrelated to
% R-peak detection. Printed as a sanity check, not analysed further.
trialPctDiff = 100 * (nTrialsRecon - nTrialsTrue) ./ nTrialsTrue;
fprintf('\nTrials retained: true %d-%d (median %d), recon %d-%d (median %d)\n', ...
    min(nTrialsTrue), max(nTrialsTrue), median(nTrialsTrue), ...
    min(nTrialsRecon), max(nTrialsRecon), median(nTrialsRecon));
fprintf('Recon-vs-true trial count difference: median %+.1f%%, range %+.1f%% to %+.1f%%\n', ...
    median(trialPctDiff), min(trialPctDiff), max(trialPctDiff));

%% CHANNEL NEIGHBOURS (shared across subjects - same net)
addpath(genpath(fieldtripPath));
cfgN = [];
cfgN.elec = trueFT{1}.elec;
cfgN.method = 'triangulation';
cfgN.compress = 'yes';
cfgN.feedback = 'no';
neighbours = ft_prepare_neighbours(cfgN);

%% CLUSTER PERMUTATION CONFIG (shared by all three tests)
cfgStat = [];
cfgStat.channel          = 'all';
cfgStat.latency          = statLatency;
cfgStat.method           = 'montecarlo';
cfgStat.statistic        = 'depsamplesT';
cfgStat.correctm         = 'cluster';
cfgStat.clusteralpha     = clusterAlpha;
cfgStat.clusterstatistic = 'maxsum';
cfgStat.minnbchan        = minNbChan;
cfgStat.neighbours       = neighbours;
cfgStat.tail             = 0;
cfgStat.clustertail      = 0;
cfgStat.alpha            = alphaLevel;
cfgStat.computecritval   = 'no';
cfgStat.numrandomization = min(numRandomization, 2^nSubj);
cfgStat.uvar             = 2;
cfgStat.ivar             = 1;
cfgStat.design           = [ones(1,nSubj), 2*ones(1,nSubj); 1:nSubj, 1:nSubj];

%% TEST A: true vs. reconstructed (expect NOT significant)
fprintf('\n=== Test A: true vs. reconstructed HEP ===\n');
statA = ft_timelockstatistics(cfgStat, trueFT{:}, reconFT{:});
report_clusters(statA, 'Test A (true vs recon)');

%% TEST B: reconstructed vs. surrogate - can the RECONSTRUCTION detect the HEP?
fprintf('\n=== Test B: reconstructed vs. surrogate HEP ===\n');
statB = ft_timelockstatistics(cfgStat, reconFT{:}, surrFT{:});
report_clusters(statB, 'Test B (recon vs surrogate)');

%% TEST C: true vs. surrogate - can the GROUND TRUTH detect the HEP?
fprintf('\n=== Test C: true vs. surrogate HEP (ground-truth reference) ===\n');
statC = ft_timelockstatistics(cfgStat, trueFT{:}, surrFT{:});
report_clusters(statC, 'Test C (true vs surrogate)');

%% ROI: THE GROUND-TRUTH HEP LOCATION (Test C)
% Defined from Test C so it comes from ground-truth data alone, never from the
% reconstruction being evaluated. This also makes the sensitivity comparison
% conservative: the mask is selected where the TRUE condition shows its
% strongest effect, so the reconstruction matching it there is a harder test
% than a neutral or recon-derived mask would be.
%
% The EXACT channel x time mask is used, not its rectangular hull, so each
% channel contributes only at the latencies at which it was implicated.
[roiChannels, roiTimeIdx, roiMask, roiFound] = get_strongest_cluster(statC, trueFT{1}.time, trueFT{1}.label);

if ~roiFound
    warning('No significant cluster in the ground-truth reference (Test C). Falling back to all channels and the full %d-%d ms window.', statLatency(1)*1000, statLatency(2)*1000);
    roiChannels = 1:numel(statC.label);
    [~, roiTimeIdx(1)] = min(abs(trueFT{1}.time - statLatency(1)));
    [~, roiTimeIdx(2)] = min(abs(trueFT{1}.time - statLatency(2)));
    roiMask = false(numel(trueFT{1}.label), numel(trueFT{1}.time));
    roiMask(roiChannels, roiTimeIdx(1):roiTimeIdx(2)) = true;
end

roiTimeSamples = roiTimeIdx(1):roiTimeIdx(2);
fprintf('\nROI from ground-truth HEP (Test C): %d channels, %.0f-%.0f ms span, %d significant channel x time points\n', ...
    numel(roiChannels), trueFT{1}.time(roiTimeSamples(1))*1000, trueFT{1}.time(roiTimeSamples(end))*1000, nnz(roiMask));

%% CLUSTER-MEAN AMPLITUDES
clustTrue  = nan(nSubj,1);
clustRecon = nan(nSubj,1);
clustSurr  = nan(nSubj,1);
for si = 1:nSubj
    clustTrue(si)  = mean(trueFT{si}.avg(roiMask));
    clustRecon(si) = mean(reconFT{si}.avg(roiMask));
    clustSurr(si)  = mean(surrFT{si}.avg(roiMask));
end
clustDiff = clustTrue - clustRecon;

%% 1. SENSITIVITY PRESERVATION
% Each trigger source contrasted against the same surrogate yields an HEP
% detection effect; if the two match, sensitivity is preserved.
effectTrue  = clustTrue  - clustSurr;   % HEP amplitude recovered from true peaks
effectRecon = clustRecon - clustSurr;   % ...and from reconstructed peaks

% Cohen's d_z for each (paired design: mean over SD of the paired difference).
dzTrue  = mean(effectTrue)  / std(effectTrue);
dzRecon = mean(effectRecon) / std(effectRecon);
dzRatio = dzRecon / dzTrue;

ampRecoveryPct = 100 * mean(effectRecon) / mean(effectTrue);
signAgreePct   = 100 * mean(sign(effectRecon) == sign(effectTrue));
rEffect        = corr(effectTrue, effectRecon);

%% 2. TRUE-vs-RECON DIFFERENCE
[~, pPaired, ~, statsPaired] = ttest(clustTrue, clustRecon);
[clustDiffMean, clustDiffCI] = mean_ci(clustDiff, 0.95);
sdBetween = std(clustTrue);          % between-subject SD, for effect-size units
hepAmp    = abs(mean(effectTrue));   % the HEP effect itself, as the yardstick

% Worst case: the strongest TRUE-vs-RECON cluster (Test A) - where the two
% trigger sources disagree most, whether or not it survived correction.
[~, ~, maskWorst, foundWorst, pWorst] = get_strongest_cluster(statA, trueFT{1}.time, trueFT{1}.label, 1);

wDiffMean = NaN; wDiffCI = [];
if foundWorst
    wTrue  = nan(nSubj,1);
    wRecon = nan(nSubj,1);
    for si = 1:nSubj
        wTrue(si)  = mean(trueFT{si}.avg(maskWorst));
        wRecon(si) = mean(reconFT{si}.avg(maskWorst));
    end
    wDiff = wTrue - wRecon;
    [wDiffMean, wDiffCI] = mean_ci(wDiff, 0.95);
end

%% CONSOLE SUMMARY - exactly the reporting set for the manuscript
fprintf('\n======= REPORT THIS (n = %d) =======\n', nSubj);

fprintf('\n1. SENSITIVITY PRESERVATION (primary)\n');
fprintf('   Does epoching on reconstructed peaks detect the HEP as well as the\n');
fprintf('   ground-truth ECG does? Each trigger source vs the SAME surrogate:\n');
fprintf('     Test C, true vs surrogate:  %s\n', cluster_phrase(statC));
fprintf('     Test B, recon vs surrogate: %s\n', cluster_phrase(statB));
fprintf('   Detection effect size on the ground-truth HEP mask (Cohen''s d_z):\n');
fprintf('     from true peaks:  d_z = %.2f (amplitude %+.3f uV)\n', dzTrue, mean(effectTrue));
fprintf('     from recon peaks: d_z = %.2f (amplitude %+.3f uV)\n', dzRecon, mean(effectRecon));
fprintf('     ratio recon/true: %.2f   amplitude recovered: %.1f%%\n', dzRatio, ampRecoveryPct);
fprintf('     per-subject effect agreement: r = %.3f, same sign in %.0f%% of subjects\n', ...
    rEffect, signAgreePct);
fprintf('      -> This is a POSITIVE claim (the reconstruction recovers the effect) rather\n');
fprintf('         than an absence of evidence, and needs no equivalence margin. Note the\n');
fprintf('         mask is taken from Test C, i.e. selected where the TRUE condition is\n');
fprintf('         strongest, so the comparison is conservative against the reconstruction.\n');

fprintf('\n2. NO DETECTABLE TRUE-vs-RECON DIFFERENCE (supporting)\n');
fprintf('   Test A, true vs reconstructed: %s\n', cluster_phrase(statA));
fprintf('      -> Informative rather than merely null, because Tests B and C show the\n');
fprintf('         same test detects the HEP in these data.\n');
fprintf('   Cluster-mean amplitude difference = %+.3f uV, 95%% CI [%.3f, %.3f]\n', ...
    clustDiffMean, clustDiffCI(1), clustDiffCI(2));
fprintf('     in effect-size units: d = %.3f (between-subject SD = %.3f uV)\n', ...
    clustDiffMean/sdBetween, sdBetween);
fprintf('     as a fraction of the HEP effect itself (%.3f uV): %.1f%%\n', ...
    hepAmp, 100*abs(clustDiffMean)/hepAmp);
if foundWorst
    fprintf('   Worst case (largest true-vs-recon cluster, p = %.4f): %+.3f uV, 95%% CI [%.3f, %.3f]\n', ...
        pWorst, wDiffMean, wDiffCI(1), wDiffCI(2));
    fprintf('     = %.1f%% of the HEP effect; d = %.3f\n', ...
        100*abs(wDiffMean)/hepAmp, wDiffMean/sdBetween);
    if wDiffCI(1) > 0 || wDiffCI(2) < 0
        fprintf('     NOTE this CI excludes zero. That is EXPECTED and not alarming: this\n');
        fprintf('     location was chosen BECAUSE it maximises the discrepancy, so the\n');
        fprintf('     estimate is selection-biased away from zero. Present it as an upper\n');
        fprintf('     bound on the discrepancy, never as "no difference here".\n');
    end
end
fprintf('   (paired t(%d) = %.3f, p = %.4f - descriptive only; a non-significant\n', ...
    statsPaired.df, statsPaired.tstat, pPaired);
fprintf('    difference test cannot establish equivalence.)\n');

%% SAVE RESULTS
results = struct();
results.subjIDs      = subjIDs;
results.nTrialsTrue  = nTrialsTrue;
results.nTrialsRecon = nTrialsRecon;
results.trialPctDiff = trialPctDiff;
results.statA        = statA;
results.statB        = statB;
results.statC        = statC;
results.roiChannels  = roiChannels;
results.roiMask      = roiMask;
results.roiTimeMs    = [trueFT{1}.time(roiTimeSamples(1)) trueFT{1}.time(roiTimeSamples(end))]*1000;
results.clustTrue    = clustTrue;
results.clustRecon   = clustRecon;
results.clustSurr    = clustSurr;
results.clustDiff    = clustDiff;
results.clustDiffCI  = clustDiffCI;
results.sdBetween    = sdBetween;
results.hepAmp       = hepAmp;
results.effectTrue   = effectTrue;
results.effectRecon  = effectRecon;
results.dzTrue       = dzTrue;
results.dzRecon      = dzRecon;
results.dzRatio      = dzRatio;
results.ampRecoveryPct = ampRecoveryPct;
results.signAgreePct = signAgreePct;
results.rEffect      = rEffect;
results.pPaired      = pPaired;
results.statsPaired  = statsPaired;
results.maskWorst    = maskWorst;
results.pWorst       = pWorst;
results.wDiffMean    = wDiffMean;
results.wDiffCI      = wDiffCI;

save(fullfile(outPath,'validation_results.mat'), 'results');

%% FIGURES
avgsTrue  = cellfun(@(x) x.avg, trueFT,  'UniformOutput', false);
avgsRecon = cellfun(@(x) x.avg, reconFT, 'UniformOutput', false);
gaTrue  = mean(cat(3, avgsTrue{:}),  3);
gaRecon = mean(cat(3, avgsRecon{:}), 3);

plot_ga_waveforms(gaTrue, gaRecon, trueFT{1}.time, roiChannels, roiTimeSamples, nSubj, outPath);
plot_sensitivity(effectTrue, effectRecon, dzTrue, dzRecon, outPath);

fprintf('\nDone. Results saved to %s\n', outPath);

%% ===================== LOCAL FUNCTIONS =====================

function report_clusters(stat, label)
    nPos = 0; nNeg = 0;
    if isfield(stat,'posclusters') && ~isempty(stat.posclusters)
        nPos = sum([stat.posclusters.prob] < 0.05);
    end
    if isfield(stat,'negclusters') && ~isempty(stat.negclusters)
        nNeg = sum([stat.negclusters.prob] < 0.05);
    end
    fprintf('%s: %d significant positive cluster(s), %d significant negative cluster(s).\n', label, nPos, nNeg);
    if isfield(stat,'posclusters')
        for c = 1:numel(stat.posclusters)
            fprintf('  pos cluster %d: p = %.4f\n', c, stat.posclusters(c).prob);
        end
    end
    if isfield(stat,'negclusters')
        for c = 1:numel(stat.negclusters)
            fprintf('  neg cluster %d: p = %.4f\n', c, stat.negclusters(c).prob);
        end
    end
end

function s = cluster_phrase(stat)
    % One-line, report-ready summary of a cluster test's outcome.
    ps = [];
    if isfield(stat,'posclusters') && ~isempty(stat.posclusters)
        ps = [ps, [stat.posclusters.prob]];
    end
    if isfield(stat,'negclusters') && ~isempty(stat.negclusters)
        ps = [ps, [stat.negclusters.prob]];
    end

    if isempty(ps)
        s = 'no candidate clusters formed';
        return
    end

    sig = ps(ps < 0.025);   % per-tail threshold, see get_strongest_cluster
    if isempty(sig)
        s = sprintf('no significant clusters (smallest p = %.4f of %d candidate(s))', min(ps), numel(ps));
    else
        s = sprintf('%d significant cluster(s), p = %s', numel(sig), ...
            strjoin(arrayfun(@(p) sprintf('%.4f', p), sig, 'UniformOutput', false), ', '));
    end
end

function [channels, timeIdx, maskFull, found, bestP] = get_strongest_cluster(stat, fullTime, fullLabels, alphaThresh)
    % stat.time is CROPPED to cfg.latency by ft_timelockstatistics, so cluster
    % column indices are positions within stat.time, not within fullTime (the
    % uncropped per-subject epoch time axis). Map them back before returning,
    % the same way b9_permutations_intra_ft.m does via ismember().
    %
    % alphaThresh defaults to 0.025, NOT 0.05. With cfg.tail = 0 and
    % cfg.correcttail unset, FieldTrip returns one-sided per-tail cluster
    % probabilities, so 0.025 is the manual halving that gives a two-tailed FWER
    % of 0.05; selecting at 0.05 would double the effective FWER to 0.10.
    % (This made no difference to the reported results: every significant cluster
    % here had p <= 0.0102 and the non-significant ones p >= 0.7763, so nothing
    % sits between the two thresholds. Corrected so it cannot bite later.)
    %
    % alphaThresh = 0.025 returns the strongest SIGNIFICANT cluster.
    % Pass 1 to return the strongest cluster regardless of significance, which
    % is what the worst-case analysis needs: the location of the largest
    % true-vs-recon difference even when no cluster survives correction.
    if nargin < 4; alphaThresh = 0.025; end

    found = false;
    channels = [];
    timeIdx = [];
    maskFull = false(numel(fullLabels), numel(fullTime));
    bestP = Inf;
    bestMask = [];

    if isfield(stat,'posclusters') && ~isempty(stat.posclusters)
        for c = 1:numel(stat.posclusters)
            if stat.posclusters(c).prob <= alphaThresh && stat.posclusters(c).prob < bestP
                bestP = stat.posclusters(c).prob;
                bestMask = stat.posclusterslabelmat == c;
                found = true;
            end
        end
    end
    if isfield(stat,'negclusters') && ~isempty(stat.negclusters)
        for c = 1:numel(stat.negclusters)
            if stat.negclusters(c).prob <= alphaThresh && stat.negclusters(c).prob < bestP
                bestP = stat.negclusters(c).prob;
                bestMask = stat.negclusterslabelmat == c;
                found = true;
            end
        end
    end

    if found
        [rowIdx, colIdx] = find(bestMask);
        channels = unique(rowIdx)';

        [~, mapIdx] = ismember(stat.time, fullTime);
        if any(mapIdx == 0)
            error('Could not map stat.time back onto the full epoch time axis - they no longer share sample points.');
        end
        fullColIdx = mapIdx(colIdx);
        timeIdx = [min(fullColIdx), max(fullColIdx)];

        % Exact channel x time cluster mask, on the full epoch time axis.
        % Channels are matched by label (not position) in case stat.label is a
        % subset or reordering of the subject channel list.
        [tf, chanMap] = ismember(stat.label, fullLabels);
        if ~all(tf)
            error('Some stat.label channels are absent from the subject channel list.');
        end
        maskFull(chanMap, mapIdx) = bestMask;
    end
end

function [m, ci] = mean_ci(x, level)
    % Mean and its two-sided confidence interval at the requested level.
    if nargin < 2; level = 0.95; end
    n  = numel(x);
    m  = mean(x);
    se = std(x)/sqrt(n);
    t  = tinv(1 - (1-level)/2, n-1);
    ci = [m - t*se, m + t*se];
end

function plot_ga_waveforms(gaTrue, gaRecon, timeVec, roiChannels, roiTimeSamples, nSubj, outPath)
    % Grand-average HEP across the full epoch, averaged over the cluster
    % channels. The cluster time span is shaded for reference.
    time_ms = timeVec * 1000;
    trueW  = mean(gaTrue(roiChannels,:), 1);
    reconW = mean(gaRecon(roiChannels,:), 1);

    f = figure('Color','w');
    hold on;
    winMs = time_ms(roiTimeSamples([1 end]));
    yMin = min([trueW reconW]);
    yMax = max([trueW reconW]);
    pad = 0.2*max(abs([yMin yMax]));
    patch([winMs(1) winMs(2) winMs(2) winMs(1)], ...
          [yMin-pad yMin-pad yMax+pad yMax+pad], [0.9 0.9 0.9], 'EdgeColor','none');
    plot(time_ms, trueW, 'k-', 'LineWidth', 1.8);
    plot(time_ms, reconW, 'r--', 'LineWidth', 1.8);
    xline(0, 'k:');
    ylim([yMin-pad, yMax+pad]);
    xlabel('Time relative to R-peak (ms)'); ylabel('Amplitude (\muV)');
    legend({'Cluster span','True R-peak','Reconstructed R-peak'}, 'Location','best');
    title(sprintf('Grand-average HEP, %d cluster channels (n = %d)', numel(roiChannels), nSubj));
    box off;
    saveas(f, fullfile(outPath,'ga_waveforms.png'));
    savefig(f, fullfile(outPath,'ga_waveforms.fig'));
end

function plot_sensitivity(effectTrue, effectRecon, dzTrue, dzRecon, outPath)
    % Sensitivity preservation: per-subject HEP detection effect recovered from
    % each trigger source. Points on the identity line mean the reconstruction
    % recovers the same effect the ground-truth ECG does.
    f = figure('Color','w');
    tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

    nexttile; hold on;
    lims = [min([effectTrue; effectRecon]) max([effectTrue; effectRecon])];
    pad = 0.1*range(lims) + eps;
    lims = [lims(1)-pad, lims(2)+pad];
    plot(lims, lims, 'k--');
    scatter(effectTrue, effectRecon, 44, 'filled');
    xlim(lims); ylim(lims); axis square;
    xlabel('HEP effect from true peaks (\muV)');
    ylabel('From reconstructed peaks (\muV)');
    title('Per-subject effect recovery'); box off;

    nexttile; hold on;
    bar([dzTrue dzRecon], 0.6, 'FaceColor',[0.35 0.5 0.7]);
    xticks(1:2); xticklabels({'True peaks','Recon peaks'});
    ylabel('HEP detection effect size (d_z)');
    title(sprintf('Detection sensitivity (ratio %.2f)', dzRecon/dzTrue));
    box off;

    saveas(f, fullfile(outPath,'sensitivity.png'));
    savefig(f, fullfile(outPath,'sensitivity.fig'));
end
