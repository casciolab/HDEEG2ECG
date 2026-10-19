%% adult_step8_cfa_check - Is the heartbeat-locked response an HEP, or the cardiac field artifact?
%
% Cardiac ICA components are deliberately RETAINED in this pipeline (see
% true_rpeaks/adult_step5_remove_ocular_ics), because the CFA and the HEP are both locked to the R-peak and
% ICA cannot be relied upon to separate two non-independent sources. The
% consequence is that the R-peak-locked scalp average contains both, and the
% surrogate contrast in adult_step7_hep_validation detects their sum. This script asks how
% much of it can be attributed to cortex rather than to the heart's field.
%
% Three diagnostics, none of which requires removing anything:
%
%   1. TIMING. The CFA can only exist where the ECG is non-zero. The R-peak-
%      locked ECG average is computed and the latency at which it returns to
%      baseline (after the T-wave) is measured. A scalp response peaking after
%      that point cannot be volume-conducted cardiac activity.
%
%   2. TOPOGRAPHY. A single source produces one scalp pattern, merely scaled
%      over time. If the topography at the response latency differs from the
%      topography during the T-wave, two different sources are involved. Report
%      the spatial correlation between them: low correlation favours cortex.
%
%   3. AMPLITUDE DISSOCIATION. If the CFA drives the response, subjects with a
%      larger ECG T-wave should show a larger scalp response. A weak
%      across-subject correlation argues against a purely cardiac explanation.
%
% Inputs:  true_rpeaks/adult_step2_preprocess\*_s2.set   (129 channels, ECG on channel 129, R-peak events)
%          true_rpeaks/adult_step6_epoch\*.set     (128-channel epochs, true R-peaks)
%          adult_step7_hep_validation\validation_results.mat  (for the ROI; optional)
% Outputs: adult_step8_cfa_check\cfa_check.mat
%          adult_step8_cfa_check\cfa_timing.png / cfa_topographies.png

%% CONFIG
clc
clear

fullpath = mfilename('fullpath');
basePath = fileparts(fullpath); % this pipeline's folder (any OS)
addpath(fileparts(basePath));   % for hdeeg2ecg_paths.m
dirs = hdeeg2ecg_paths();       % input, output and toolbox folders (see README)

eeglabPath = dirs.eeglab;
addpath(genpath(eeglabPath));

fzabPath  = fullfile(dirs.adult,'true_rpeaks','adult_step2_preprocess');
epochPath = fullfile(dirs.adult,'true_rpeaks','adult_step6_epoch');
validPath = fullfile(dirs.adult,'adult_step7_hep_validation');
outPath   = fullfile(dirs.adult,'adult_step8_cfa_check');
if ~exist(outPath,'dir'); mkdir(outPath); end

heartChannel = 129;            % ECG channel in true_rpeaks/adult_step2_preprocess
epochWin     = [-0.3 0.6];     % must match true_rpeaks/adult_step6_epoch
rPeakEvent   = 'Rpeak';        % event type in true_rpeaks/adult_step2_preprocess / true_rpeaks/adult_step5_remove_ocular_ics
hepWindow    = [0.2 0.6];      % s, window searched for the scalp response

% Fraction of the peak |ECG| below which the ECG is treated as returned to
% baseline. The first sustained crossing after the T-wave peak defines the
% earliest latency at which a scalp effect cannot be CFA.
ecgBaselineFrac = 0.05;

% Latency after which to search for the T-wave, i.e. how much of the QRS
% complex to skip. The S-wave can extend to roughly 60 ms, so 80 ms is a safe
% default - but CHECK cfa_timing.png: if the marked "T peak" has landed on the
% QRS tail rather than the T-wave, raise this.
qrsSkip = 0.08;

%% ROI - reuse the validation cluster if available
roiChannels = [];
vf = fullfile(validPath,'validation_results.mat');
if exist(vf,'file')
    S = load(vf);
    roiChannels = S.results.roiChannels;
    fprintf('Using the %d-channel ROI from adult_step7_hep_validation.\n', numel(roiChannels));
else
    warning('validation_results.mat not found; falling back to all channels for the scalp waveform.');
end

%% DISCOVER SUBJECTS (those with BOTH an epoch file and an true_rpeaks/adult_step2_preprocess file)
epochFiles = dir(fullfile(epochPath,'*.set'));
fzabFiles  = dir(fullfile(fzabPath,'*.set'));
getID = @(n) regexp(n,'(\d{3,6})','match','once');
epochIDs = cellfun(getID, {epochFiles.name}, 'UniformOutput', false);
fzabIDs  = cellfun(getID, {fzabFiles.name},  'UniformOutput', false);
subjIDs  = intersect(epochIDs, fzabIDs);
nSubj    = numel(subjIDs);
if nSubj == 0
    error('No subjects found with both an true_rpeaks/adult_step6_epoch and an true_rpeaks/adult_step2_preprocess file.');
end
fprintf('Found %d subjects.\n', nSubj);

%% LOAD: ECG R-locked average, and the EEG epoch average
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

ecgAvg  = [];
eegAvg  = [];
chanlocs = [];
timeVec  = [];

for si = 1:nSubj
    sid = subjIDs{si};

    % --- EEG: already epoched on true R-peaks ---
    EEGe = pop_loadset('filename', epochFiles(strcmp(epochIDs,sid)).name, 'filepath', epochPath);
    if isempty(timeVec)
        timeVec  = EEGe.times/1000;      % s
        chanlocs = EEGe.chanlocs;
        eegAvg   = nan(EEGe.nbchan, numel(timeVec), nSubj);
        ecgAvg   = nan(nSubj, numel(timeVec));
    end
    eegAvg(:,:,si) = mean(EEGe.data,3);

    % --- ECG: epoch true_rpeaks/adult_step2_preprocess on the same events and window ---
    EEGf = pop_loadset('filename', fzabFiles(strcmp(fzabIDs,sid)).name, 'filepath', fzabPath);
    if EEGf.nbchan < heartChannel
        error('Subject %s: true_rpeaks/adult_step2_preprocess has only %d channels; expected the ECG on channel %d.', ...
            sid, EEGf.nbchan, heartChannel);
    end
    EEGf = pop_select(EEGf, 'channel', heartChannel);
    EEGf = pop_epoch(EEGf, {rPeakEvent}, epochWin);
    e = mean(EEGf.data,3);
    % Align length to the EEG time axis (epoching can differ by a sample)
    n = min(numel(e), numel(timeVec));
    ecgAvg(si,1:n) = e(1:n);

    fprintf('  %s: %d EEG trials, %d ECG trials\n', sid, size(EEGe.data,3), EEGf.trials);
end

%% GRAND AVERAGES
gaECG = mean(ecgAvg, 1, 'omitnan');
gaEEG = mean(eegAvg, 3);
if isempty(roiChannels); roiChannels = 1:size(gaEEG,1); end
gaROI = mean(gaEEG(roiChannels,:), 1);

%% 1. TIMING - where does the ECG return to baseline?
post = timeVec > qrsSkip;                    % skip the QRS complex itself
absECG = abs(gaECG - mean(gaECG(timeVec < -0.2)));   % baseline-corrected magnitude
[tPeakVal, tPeakRel] = max(absECG(post));
postIdx = find(post);
tPeakIdx = postIdx(tPeakRel);
tPeakLat = timeVec(tPeakIdx);                % T-wave peak latency

% First latency after the T-wave peak at which |ECG| stays below threshold
thr = ecgBaselineFrac * tPeakVal;
safeIdx = NaN;
for k = tPeakIdx:numel(timeVec)
    if all(absECG(k:end) < thr)
        safeIdx = k; break
    end
end
if isnan(safeIdx)
    cfaSafeLat = NaN;
else
    cfaSafeLat = timeVec(safeIdx);
end

% Scalp response peak within the HEP search window
hepIdx = find(timeVec >= hepWindow(1) & timeVec <= hepWindow(2));
[~, rel] = max(abs(gaROI(hepIdx)));
respIdx = hepIdx(rel);
respLat = timeVec(respIdx);

%% 2. TOPOGRAPHY - same source or different?
topoT    = gaEEG(:, tPeakIdx);
topoResp = gaEEG(:, respIdx);
rTopo    = corr(topoT, topoResp);

%% 3. AMPLITUDE DISSOCIATION across subjects
tAmpSubj   = abs(ecgAvg(:, tPeakIdx));
respSubj   = nan(nSubj,1);
for si = 1:nSubj
    respSubj(si) = mean(eegAvg(roiChannels, respIdx, si), 1);
end
[rAmp, pAmp] = corr(tAmpSubj(:), abs(respSubj(:)));

%% CONSOLE SUMMARY
fprintf('\n======= IS IT AN HEP OR THE CFA? (n = %d) =======\n', nSubj);

fprintf('\n1. TIMING\n');
fprintf('   ECG T-wave peak:                 %.0f ms\n', tPeakLat*1000);
if isnan(cfaSafeLat)
    fprintf('   ECG never returns below %.0f%% of the T-wave peak within the epoch.\n', ecgBaselineFrac*100);
    fprintf('      -> NO CFA-free latency exists in this epoch. A cortical interpretation\n');
    fprintf('         of the scalp response cannot be defended on timing grounds alone.\n');
else
    fprintf('   ECG back to <%.0f%% of that peak from:  %.0f ms onward\n', ecgBaselineFrac*100, cfaSafeLat*1000);
end
fprintf('   Scalp response peak (ROI):       %.0f ms\n', respLat*1000);
if ~isnan(cfaSafeLat)
    if respLat >= cfaSafeLat
        fprintf('      -> The response peaks AFTER the ECG has returned to baseline, so it\n');
        fprintf('         cannot be volume-conducted cardiac activity. Supports a cortical origin.\n');
    else
        fprintf('      -> The response peaks WHILE the ECG is still non-zero, so a CFA\n');
        fprintf('         contribution cannot be excluded on timing grounds. Restrict\n');
        fprintf('         interpretation to >= %.0f ms, or report it as a heartbeat-locked\n', cfaSafeLat*1000);
        fprintf('         response rather than an isolated cortical potential.\n');
    end
end

fprintf('\n2. TOPOGRAPHY\n');
fprintf('   Spatial correlation, T-wave topography vs response topography: r = %.3f\n', rTopo);
if abs(rTopo) < 0.5
    fprintf('      -> Distinct scalp patterns, so distinct sources. A single volume-conducted\n');
    fprintf('         cardiac source would give the same topography scaled, not a different one.\n');
else
    fprintf('      -> Similar scalp patterns. Consistent with one dominant source, i.e. the\n');
    fprintf('         response may be largely CFA. Treat a cortical claim cautiously.\n');
end

fprintf('\n3. AMPLITUDE DISSOCIATION\n');
fprintf('   Across subjects, |T-wave amplitude| vs |scalp response|: r = %.3f, p = %.4f\n', rAmp, pAmp);
if abs(rAmp) < 0.5
    fprintf('      -> Weak coupling: subjects with a bigger cardiac field do not show a bigger\n');
    fprintf('         scalp response, which argues against a purely cardiac explanation.\n');
else
    fprintf('      -> Substantial coupling: scalp response scales with the cardiac field, which\n');
    fprintf('         is what CFA contamination looks like.\n');
end

fprintf('\nNOTE none of these three is decisive alone. Timing is the strongest: a response\n');
fprintf('peaking after the ECG has flattened cannot be CFA. Use the printed CFA-free onset\n');
fprintf('to set the interpretation window in the manuscript, rather than a literature guess.\n');

%% SAVE
cfa = struct();
cfa.subjIDs     = {subjIDs};
cfa.timeVec     = timeVec;
cfa.gaECG       = gaECG;
cfa.gaEEG       = gaEEG;
cfa.gaROI       = gaROI;
cfa.roiChannels = roiChannels;
cfa.tPeakLat    = tPeakLat;
cfa.cfaSafeLat  = cfaSafeLat;
cfa.respLat     = respLat;
cfa.rTopo       = rTopo;
cfa.rAmp        = rAmp;
cfa.pAmp        = pAmp;
save(fullfile(outPath,'cfa_check.mat'), 'cfa');

%% FIGURES
f1 = figure('Color','w');
tiledlayout(f1,2,1,'TileSpacing','compact','Padding','compact');

nexttile; hold on;
plot(timeVec*1000, gaECG, 'Color',[0.8 0.2 0.2], 'LineWidth',1.8);
xline(tPeakLat*1000, 'k--', 'T peak');
if ~isnan(cfaSafeLat); xline(cfaSafeLat*1000, 'b--', 'ECG baseline'); end
ylabel('ECG (\muV)'); title('R-peak-locked ECG (grand average)'); box off;

nexttile; hold on;
if ~isnan(cfaSafeLat)
    yl = [min(gaROI) max(gaROI)];
    pad = 0.2*max(abs(yl));
    patch([timeVec(1) cfaSafeLat cfaSafeLat timeVec(1)]*1000, ...
          [yl(1)-pad yl(1)-pad yl(2)+pad yl(2)+pad], [1 0.9 0.9], 'EdgeColor','none');
end
plot(timeVec*1000, gaROI, 'k-', 'LineWidth',1.8);
xline(respLat*1000, 'k:', 'response peak');
xlabel('Time relative to R-peak (ms)'); ylabel('EEG (\muV)');
title(sprintf('Scalp response, %d ROI channels (shaded = CFA possible)', numel(roiChannels)));
box off;
saveas(f1, fullfile(outPath,'cfa_timing.png'));
savefig(f1, fullfile(outPath,'cfa_timing.fig'));

f2 = figure('Color','w');
tiledlayout(f2,1,2,'TileSpacing','compact','Padding','compact');
nexttile; topoplot(topoT, chanlocs); colorbar;
title(sprintf('T-wave topography (%.0f ms)', tPeakLat*1000));
nexttile; topoplot(topoResp, chanlocs); colorbar;
title(sprintf('Response topography (%.0f ms), spatial r = %.2f', respLat*1000, rTopo));
saveas(f2, fullfile(outPath,'cfa_topographies.png'));
savefig(f2, fullfile(outPath,'cfa_topographies.fig'));

fprintf('\nDone. Results saved to %s\n', outPath);
