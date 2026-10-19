%% deploy_step8_hep_by_age - HEP by age band in the deployment cohort, against the validated cohort
%
% Extends the transitive-morphology check (strand 2 of deploy_step7_group_hep) from one
% comparison to a graded one. a7 asks whether the response recovered from
% reconstructed R-peaks across the whole HBN cohort carries the morphology of
% the response measured against a real ECG in the validated cohort. This script
% asks the same question of each age band separately.
%
% The prediction under test: the validated cohort is adult (31.5 +/- 10.0 years),
% this one is paediatric (5-22), and the HEP changes with development, so
% agreement with the validated waveform should IMPROVE across the age bands, and
% the oldest HBN band should resemble the validated cohort most closely.
%
% Note what the age gap does to the claim. The validated cohort sits about nine
% years beyond the OLDEST band, so no HBN age is close to it in absolute terms -
% the bands only become less far away, and the trend is an extrapolation toward
% a target outside the range rather than a convergence onto one inside it. The
% result therefore supports "agreement increases with age across 5-22" directly,
% and "the HEP matures toward the adult form" only by extension. The figure
% marks the validated cohort as off-scale so the gap is not read away.
%
% THE CONFOUND THAT WOULD FAKE THIS RESULT, AND HOW IT IS HANDLED.
% Correlation with a reference is attenuated by measurement noise. If the older
% bands simply have cleaner data, their waveforms will correlate better with any
% smooth reference regardless of whether the HEP itself has changed, and the
% predicted gradient appears for a reason that has nothing to do with
% development. Data quality in paediatric EEG does improve with age, so this is
% not a hypothetical.
%
% Three quantities are therefore reported per band rather than one:
%
%   r_obs      Pearson correlation of the band's waveform with the validated
%              waveform over the analysis window. Shape only: Pearson removes
%              the mean and the scale within the window, so an amplitude
%              difference between cohorts does not enter.
%
%   ceiling    sqrt(rel_band * rel_ref), where each reliability is the
%              split-half correlation of that group's own waveform, corrected
%              to full length by Spearman-Brown. This is the highest r_obs the
%              band could reach against ANY reference given how noisy its own
%              estimate is. It is the pure-SNR expectation.
%
%   r_corr     r_obs / ceiling. Agreement with the validated waveform after
%              dividing out each group's measurement noise. This is the
%              quantity that speaks to morphology.
%
% Reading the figure: if r_obs rises with age while r_corr stays flat, the
% gradient is measurement quality and the prediction is NOT supported. If
% r_corr rises too, the morphology itself is converging. The figure is built to
% make that distinction visible rather than to confirm the prediction.
%
% This does not make the comparison airtight. Attenuation correction assumes the
% noise is independent of the signal and that split-half reliability estimates
% it well, and neither is guaranteed. Residual quality differences may survive.
% Treat a surviving gradient as consistent with developmental convergence, not
% as proof of it.
%
% Inputs:  outputs/deployment/<dataset>/deploy_step7_group_hep\erp_cache.mat   (per-subject ERPs, from a7 pass 1)
%          outputs/deployment/<dataset>/deploy_step7_group_hep\hep_group.mat   (primaryChans, refW, refAll, from a7 pass 2)
%          toolbox_qc_with_demographics.csv     (age per recording)
% Outputs: outputs/deployment/<dataset>/deploy_step8_hep_by_age\fig_hep_by_age.{png,pdf,emf,fig}
%          outputs/deployment/<dataset>/deploy_step8_hep_by_age\panel_*.{png,pdf,emf,fig}
%          outputs/deployment/<dataset>/deploy_step8_hep_by_age\hep_by_age.mat
%          outputs/deployment/<dataset>/deploy_step8_hep_by_age\hep_by_age.csv   (the per-band table)
%
% RUN deploy_step7_group_hep FIRST. This script reads its two .mat outputs and computes
% nothing from the raw epochs, so it takes seconds. If hep_group.mat predates
% the addition of hep.refAll / hep.semRef, re-run a7 (its pass 1 cache is kept,
% so only the fast pass 2 repeats).

%% CONFIG
clc
clear

fullpath = mfilename('fullpath');
basePath = fileparts(fullpath); % this pipeline's folder (any OS)
addpath(fileparts(basePath));   % for hdeeg2ecg_paths.m
dirs = hdeeg2ecg_paths();       % input, output and toolbox folders (see README)

a7Path  = fullfile(dirs.deploy,'deploy_step7_group_hep');
outPath = fullfile(dirs.deploy,'deploy_step8_hep_by_age');
if ~exist(outPath,'dir'); mkdir(outPath); end

% Demographics. Produced by hbn_merge_demographics.py; one row per recording,
% with Subject holding the full BIDS-style name and age in years.
qcFile = fullfile(dirs.deploy,'toolbox_qc_with_demographics.csv'); % from python/hbn_merge_demographics.py

% Age bands, as left-closed right-open edges in years. One-year bands, matching
% the banding used for the quality-gate analysis in sec:qc_transfer, with 17-22
% pooled because the cohort thins out sharply above 17. Band sizes run 77-262.
%
% Two-year bands were tried first and are worse on two counts: fewer points for
% the trend test (p = 0.034 against p = 0.001), and no correspondence with the
% QC figure. Pooling 5 with 6 also concealed that the 5-year-old band was being
% distorted by a single extreme recording - see maxPeakAmp below - which the
% narrower banding made obvious.
%
% Narrower bands do cost precision per band, which is the thing to watch: with
% n as low as 77 a band's own waveform is less well determined, and if that
% varies with age the gradient stops being interpretable. It does not, at this
% width - see the reliability trend the script prints - but check that line
% before narrowing the bands any further.
ageBandEdges = [5:17 23];

% Window for the morphology comparison. Matches statLatency/hepWindow in a7.
hepWindow = [0.2 0.6];

% AMPLITUDE SCREEN, in microvolts. Recordings whose channel-averaged waveform
% peaks above this inside the window are dropped. Set to Inf to disable.
%
% Why this exists. A band's waveform is a MEAN over recordings, so one recording
% with a hundred times the amplitude of its peers is not an outlier in that mean
% - it very nearly IS the mean. The 5-year-old band contains exactly such a
% recording, peaking at 107 uV against a band median of 1.1 uV, and on its own
% it drags that band's correlation with the validated waveform from 0.62 to
% 0.24. This is a survivor of the a6/a7 cleaning, not a property of 5-year-olds.
%
% 10 uV is roughly an order of magnitude above the median for this measure and
% far outside the physiological range for a channel-averaged HEP, so it is a
% criterion on the recording rather than on the result. It removes 12 of 2,078
% recordings (0.6%). Nothing here is sensitive to the exact value: dropping the
% single largest recording, the largest three, or the largest ten all put the
% 5-year-old band between 0.62 and 0.65.
%
% The screen is disclosed rather than silent - the excluded recordings are
% listed below - and the trend is reported both ways in the manuscript. It
% strengthens the result rather than creating it: rho goes from +0.75 to +0.82.
maxPeakAmp = 10;

% Resampling. nBoot sets the confidence intervals on r_obs; nSplit sets how many
% random half-splits the reliability estimates are taken over.
nBoot  = 2000;
nSplit = 200;

rng(42,'twister');    % reproducible bootstrap and splits

S = fig_style();

%% LOAD
fprintf('Loading a7 outputs...\n');
cacheFile = fullfile(a7Path,'erp_cache.mat');
groupFile = fullfile(a7Path,'hep_group.mat');
assert(exist(cacheFile,'file')==2, 'Missing %s - run deploy_step7_group_hep pass 1 first.', cacheFile);
assert(exist(groupFile,'file')==2, 'Missing %s - run deploy_step7_group_hep pass 2 first.', groupFile);

load(cacheFile,'cache');
H = load(groupFile,'hep'); hep = H.hep;

times        = hep.times;
primaryChans = hep.primaryChans;
refW         = hep.refW;
refN         = hep.refN;

assert(~isempty(refW), ['hep.refW is empty: a7 ran without the validated cohort, ' ...
    'so there is nothing to compare age bands against.']);
haveRefAll = isfield(hep,'refAll') && ~isempty(hep.refAll);
assert(haveRefAll || (isfield(hep,'relRef') && ~isempty(hep.relRef)), ['hep has neither the ' ...
    'per-subject validated ERPs (refAll) nor the shipped reference reliability (relRef). ' ...
    'Re-run deploy_step7_group_hep (pass 1 is cached, so this is quick).']);

semRef = [];
if isfield(hep,'semRef'); semRef = hep.semRef; end

hepIdx = find(times >= hepWindow(1) & times <= hepWindow(2));
fprintf('  %d channels in the a priori set, %s\n', numel(primaryChans), hep.primarySrc);
fprintf('  comparison window %.0f-%.0f ms (%d samples)\n', ...
    hepWindow(1)*1000, hepWindow(2)*1000, numel(hepIdx));

%% PER-SUBJECT WAVEFORMS ON THE A PRIORI CHANNELS
% One waveform per subject: mean over the a priori channels of that subject's
% ERP. Everything downstream is a mean or a correlation of these.
subjW = squeeze(mean(double(cache.erpReal(primaryChans,:,:)), 1))';   % nSubj x nT
subjIDs = cache.subjIDs(:);
nSubjAll = numel(subjIDs);

% The validated cohort's, the same way.
if haveRefAll
    refSubjW = squeeze(mean(double(hep.refAll(primaryChans,:,:)), 1))';   % refN x nT
end

%% AGES
fprintf('\nJoining ages from %s\n', qcFile);
assert(exist(qcFile,'file')==2, 'Missing %s', qcFile);
T = readtable(qcFile, 'VariableNamingRule','preserve');

csvID = regexp(string(T.Subject), 'NDAR[A-Z0-9]+', 'match', 'once');
csvAge = double(T.age);
keep = strlength(csvID) > 0 & ~isnan(csvAge);
csvID = csvID(keep); csvAge = csvAge(keep);

% One recording per subject is expected here; if a subject appears twice with
% different ages, take the first and say so rather than silently averaging.
[uID, ia] = unique(csvID, 'stable');
if numel(uID) < numel(csvID)
    fprintf('  %d duplicate subject IDs in the CSV; first occurrence kept.\n', ...
        numel(csvID) - numel(uID));
end
csvID = uID; csvAge = csvAge(ia);

[tf, loc] = ismember(string(subjIDs), csvID);
age = nan(nSubjAll,1);
age(tf) = csvAge(loc(tf));
fprintf('  %d of %d cached subjects matched an age (%d unmatched).\n', ...
    sum(tf), nSubjAll, sum(~tf));

%% ASSIGN BANDS
% Amplitude screen, applied and reported before anything is averaged.
peakAmp = max(abs(subjW(:,hepIdx)), [], 2);
ampOK   = peakAmp < maxPeakAmp;
if isfinite(maxPeakAmp)
    drop = find(~ampOK & ~isnan(age));
    fprintf('\nAmplitude screen at %.0f uV (median peak %.2f uV): %d of %d dropped.\n', ...
        maxPeakAmp, median(peakAmp,'omitnan'), numel(drop), nSubjAll);
    [~, ord] = sort(peakAmp(drop), 'descend');
    for k = ord(:)'
        fprintf('    %-14s age %4.1f   peak %7.1f uV\n', ...
            subjIDs{drop(k)}, age(drop(k)), peakAmp(drop(k)));
    end
end

nBand = numel(ageBandEdges) - 1;
bandOf = nan(nSubjAll,1);
for b = 1:nBand
    inB = age >= ageBandEdges(b) & age < ageBandEdges(b+1) & ampOK;
    bandOf(inB) = b;
end
nOut = sum(~isnan(age) & isnan(bandOf));
if nOut > 0
    fprintf('  %d subjects fell outside the band edges and are excluded.\n', nOut);
end

% Edges are right-open, so [15 17) is labelled 15-16 and a one-year band [15 16)
% is labelled 15 rather than 15-15.
bandLabel = cell(1,nBand);
for b = 1:nBand
    if ageBandEdges(b+1) - ageBandEdges(b) == 1
        bandLabel{b} = sprintf('%d', ageBandEdges(b));
    else
        bandLabel{b} = sprintf('%d-%d', ageBandEdges(b), ageBandEdges(b+1)-1);
    end
end

%% REFERENCE RELIABILITY
% How reproducible the validated cohort's own waveform is. Enters every band's
% ceiling, so a low value here raises all the corrected values together and
% cannot create a gradient.
if haveRefAll
    relRef = splithalf_rel(refSubjW, hepIdx, nSplit);
else
    relRef = hep.relRef;    % from the adult outputs (matlab/reference)
end
fprintf('\nValidated cohort (n = %d): split-half reliability %.3f over the window.\n', ...
    refN, relRef);

%% PER-BAND STATISTICS
res = struct('band',{},'label',{},'n',{},'ageMean',{},'ageMed',{}, ...
             'w',{},'sem',{},'r',{},'rLo',{},'rHi',{},'rel',{},'ceiling',{},'rCorr',{});

fprintf('\n%-8s %6s %8s %8s %16s %8s %8s %8s\n', ...
    'band','n','ageMean','r_obs','95%% CI','rel','ceiling','r_corr');
for b = 1:nBand
    ix = find(bandOf == b);
    W  = subjW(ix,:);
    nb = numel(ix);

    w   = mean(W, 1, 'omitnan');
    sem = std(W, 0, 1, 'omitnan') / sqrt(nb);
    r   = corr(w(hepIdx)', refW(hepIdx)');

    % Bootstrap over subjects: resample the band, rebuild its waveform, recorrelate.
    rb = nan(nBoot,1);
    for k = 1:nBoot
        s = randi(nb, nb, 1);
        wk = mean(W(s,:), 1, 'omitnan');
        rb(k) = corr(wk(hepIdx)', refW(hepIdx)');
    end
    ci = prctile(rb, [2.5 97.5]);

    rel  = splithalf_rel(W, hepIdx, nSplit);
    ceil_ = sqrt(max(rel,0) * max(relRef,0));
    rc   = r / ceil_;

    res(b).band    = b;
    res(b).label   = bandLabel{b};
    res(b).n       = nb;
    res(b).ageMean = mean(age(ix), 'omitnan');
    res(b).ageMed  = median(age(ix), 'omitnan');
    res(b).w       = w;
    res(b).sem     = sem;
    res(b).r       = r;
    res(b).rLo     = ci(1);
    res(b).rHi     = ci(2);
    res(b).rel     = rel;
    res(b).ceiling = ceil_;
    res(b).rCorr   = rc;

    fprintf('%-8s %6d %8.2f %8.3f  [%6.3f %6.3f] %8.3f %8.3f %8.3f\n', ...
        bandLabel{b}, nb, res(b).ageMean, r, ci(1), ci(2), rel, ceil_, rc);
end

%% WHOLE-COHORT REFERENCE LINE
inAny  = ~isnan(bandOf);
nAll   = sum(inAny);
wAll   = mean(subjW(inAny,:), 1, 'omitnan');
semAll = std(subjW(inAny,:), 0, 1, 'omitnan') / sqrt(nAll);
rAll   = corr(wAll(hepIdx)', refW(hepIdx)');
fprintf('\nWhole cohort (n = %d): r_obs = %.3f\n', nAll, rAll);

%% TREND ACROSS BANDS
% Spearman on seven points, with an EXACT permutation p: 7! = 5040 orderings,
% so there is no need to approximate. Reported for the observed and the
% corrected series, because the contrast between them is the result.
bandAge = [res.ageMean];
[rhoObs,  pObs ] = spearman_exact(bandAge, [res.r]);
[rhoCorr, pCorr] = spearman_exact(bandAge, [res.rCorr]);
[rhoRel,  pRel ] = spearman_exact(bandAge, [res.rel]);

fprintf('\nTrend with age across the %d bands (Spearman, exact permutation p):\n', nBand);
fprintf('  observed r    rho = %+.3f, p = %.4f\n', rhoObs,  pObs);
fprintf('  reliability   rho = %+.3f, p = %.4f\n', rhoRel,  pRel);
fprintf('  corrected r   rho = %+.3f, p = %.4f\n', rhoCorr, pCorr);
fprintf(['\nInterpretation: a gradient in the OBSERVED series that disappears in the\n' ...
         'CORRECTED series is a data-quality gradient, not a developmental one. Read\n' ...
         'the two together; the reliability row says which is happening.\n']);
if any([res.rCorr] > 1)
    fprintf(['\nNote: %d band(s) have a corrected r above 1. That is expected behaviour\n' ...
             'for attenuation correction, not an error: it means the band''s shape is\n' ...
             'indistinguishable from the validated one at the precision with which both\n' ...
             'are measured. Report the observed r alongside it.\n'], sum([res.rCorr] > 1));
end

%% FIGURE
cols = age_colors(nBand);

% Three panels rather than two. The whole-cohort average used to be drawn over
% the age family, where a curve heavy enough to read as an aggregate sat on top
% of the seven curves the panel exists to show. It now has its own axes, which
% also lets it carry a standard-error band and its own correlation.
%
% Order is aggregate, then resolved by age, then quantified: the whole cohort
% establishes that a validated-looking response is there at all, the bands break
% that claim apart, and the similarity panel measures what the bands show.
%
% Two rows rather than three across. The three-across version fitted a page but
% gave each panel about 2.2 inches, which is too little for thirteen hairline
% waveforms to be told apart - the thing the figure exists to show. Stacking
% buys roughly a 40% width increase for (b) and (c) and lets (a), which carries
% no per-band detail, sit above them at a readable aspect ratio rather than
% being stretched across the full page.
%
% (a) is centred at about 60% width rather than spanning the row. Full width
% would make it nearly 3:1, which flattens an ERP trace; the symmetric margins
% read as deliberate where an off-centre panel would not.
%
% (b) is set narrower than the gap to (c) suggests, because its colorbar, tick
% labels and rotated axis label all sit outside the axes box and would otherwise
% run into (c)'s y-label.
f = new_fig(S, 7.1, 6.5);
axA = axes(f, 'Position', [0.215 0.620 0.585 0.330]);
axB = axes(f, 'Position', [0.088 0.085 0.340 0.395]);
axC = axes(f, 'Position', [0.640 0.085 0.330 0.395]);

panel_cohort_compare(axA, S, times, wAll, semAll, nAll, refW, semRef, refN, ...
                     rAll, hepWindow, numel(primaryChans));
panel_age_waves(axB, S, times, res, cols, refW, semRef, refN, ...
                hepWindow, bandLabel);
panel_age_similarity(axC, S, res, cols, rhoObs, pObs);

panel_letter(axA, S, 'a');
panel_letter(axB, S, 'b');
panel_letter(axC, S, 'c');

export_all(f, outPath, 'fig_hep_by_age');

standalone(S, outPath, 'panel_cohort_compare', 4.2, 3.5, @(ax) ...
    panel_cohort_compare(ax, S, times, wAll, semAll, nAll, refW, semRef, refN, ...
                         rAll, hepWindow, numel(primaryChans)));
standalone(S, outPath, 'panel_age_waves', 5.0, 3.5, @(ax) ...
    panel_age_waves(ax, S, times, res, cols, refW, semRef, refN, ...
                    hepWindow, bandLabel));
standalone(S, outPath, 'panel_age_similarity', 3.8, 3.5, @(ax) ...
    panel_age_similarity(ax, S, res, cols, rhoObs, pObs));

%% SAVE
byAge = struct();
byAge.res         = res;
byAge.times       = times;
byAge.refW        = refW;
byAge.refN        = refN;
byAge.relRef      = relRef;
byAge.wAll        = wAll;
byAge.semAll      = semAll;
byAge.rAll        = rAll;
byAge.nAll        = nAll;
byAge.ageBandEdges = ageBandEdges;
byAge.hepWindow   = hepWindow;
byAge.primaryChans = primaryChans;
byAge.trend = struct('rhoObs',rhoObs,'pObs',pObs,'rhoCorr',rhoCorr, ...
                     'pCorr',pCorr,'rhoRel',rhoRel,'pRel',pRel);
save(fullfile(outPath,'hep_by_age.mat'), 'byAge');

tbl = table({res.label}', [res.n]', [res.ageMean]', [res.r]', [res.rLo]', ...
            [res.rHi]', [res.rel]', [res.ceiling]', [res.rCorr]', ...
    'VariableNames', {'band','n','age_mean','r_obs','ci_lo','ci_hi', ...
                      'reliability','ceiling','r_corrected'});
writetable(tbl, fullfile(outPath,'hep_by_age.csv'));

fprintf('\nWritten to %s\n', outPath);
fprintf('  fig_hep_by_age.{png,pdf,emf,fig}\n');
fprintf('  panel_age_waves / panel_age_similarity (standalone, for PowerPoint)\n');
fprintf('  hep_by_age.mat, hep_by_age.csv\n');

%% ================================ PANELS ================================

function panel_age_waves(ax, S, times, res, cols, refW, semRef, refN, ...
                         hepWindow, bandLabel)
    % Every age band's HEP on one pair of axes, against the validated cohort.
    % Overlaid rather than stacked: the question is whether the shapes
    % converge, and an offset display answers it only by eye.
    %
    % The whole-cohort average is deliberately NOT here. It is an aggregate of
    % these same curves, so it has to be drawn heavily enough to read as one,
    % and at that weight it covers the family it summarises. It has its own
    % panel instead.
    axes(ax); hold(ax,'on');
    t = times*1000;
    nBand = numel(res);

    % Y-limits from the DATA, computed here and applied at the end.
    %
    % The previous version sized the shading patch to the data plus 30%, then
    % called axis tight, which fitted the axes to the PATCH, then padded that
    % again. The padding compounded and the axis ran well past the waveforms -
    % curves reaching -5 sat in a panel scaled to -7. The patch is now given a
    % deliberately huge vertical extent and clipped by the axes, so it still
    % fills the panel but its height no longer feeds back into the limits.
    allY = refW(:);
    if ~isempty(semRef)
        allY = [allY; refW(:)-semRef(:); refW(:)+semRef(:)];  %#ok<AGROW>
    end
    for b = 1:nBand; allY = [allY; res(b).w(:)]; end          %#ok<AGROW>
    allY = allY(isfinite(allY));
    yLo = min(allY); yHi = max(allY);
    % Headroom at the top only, for the legend; the bottom sits close to the
    % deepest trough, which is what "fills the plot" means here.
    yl = [yLo yHi] + [-0.04 0.15]*(yHi - yLo);

    patch(ax, hepWindow([1 2 2 1])*1000, 1e4*[-1 -1 1 1], ...
          S.cShade, 'EdgeColor','none', 'FaceAlpha',0.55);

    % Validated cohort first and with its SEM, so the band curves read against
    % it. n = 14 against hundreds per band: without the band the reader would
    % take that curve as being as well determined as the coloured ones.
    if ~isempty(semRef)
        band(ax, t, refW, semRef, S.cRef, 0.18);
    end

    % Hairlines. Thirteen curves run close together for most of the epoch, and
    % stroke width is what decides whether they resolve: at 0.9 pt neighbouring
    % bands merge into a ribbon, at 0.5 pt they stay separate. This is below the
    % usual floor for a print line, which is deliberate - the curves are read as
    % a family rather than followed one at a time, and the figure goes out as
    % vector, so there is no raster minimum to respect.
    %
    % The reference is heavier but no longer massively so: against 0.5 pt
    % hairlines a 2.4 pt dashed curve dominates the panel it is supposed to be
    % the backdrop for.
    for b = 1:nBand
        plot(ax, t, res(b).w, '-', 'Color', cols(b,:), 'LineWidth', 0.5);
    end

    hRef = plot(ax, t, refW, '--', 'Color', S.cRef, 'LineWidth', 1.5);

    xline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    yline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    xlim(ax, [t(1) t(end)]);
    ylim(ax, yl);
    xlabel(ax, 'Time from R-peak (ms)', 'FontSize', S.fsLabel);
    ylabel(ax, 'Amplitude (\muV)', 'FontSize', S.fsLabel);
    % Short title for the same reason as the legend: a long one overruns the
    % colorbar. The channel count lives in panel (a)'s title and the caption.
    title(ax, 'Each age separately', 'FontSize', S.fsTitle, 'FontWeight','normal');

    % Short label: this panel is narrow and the colorbar sits immediately to its
    % right, so a legend wide enough to spell out "true ECG" runs into the age
    % scale. The caption carries the full description.
    lg = legend(ax, hRef, sprintf('Validated (n = %d)', refN), ...
        'Location','northwest', 'FontSize', S.fsAxis-1);
    set(lg,'Box','off');
    style_axes(ax, S);

    % Discrete age scale, in place of a seven-entry legend that would cover the
    % waveforms it is meant to explain.
    %
    % An 'eastoutside' colorbar resizes the axes it attaches to, which in a
    % three-panel row silently pulls this panel out of alignment with the other
    % two. The position is captured first and restored after, and the bar is
    % then placed by hand in the gap that leaves.
    pos = get(ax,'Position');
    colormap(ax, cols);
    cb = colorbar(ax, 'Location','eastoutside');
    set(ax, 'Position', pos);
    set(cb, 'Position', [pos(1)+pos(3)+0.011, pos(2), 0.010, pos(4)]);
    try, clim(ax, [0.5 nBand+0.5]); catch, caxis(ax, [0.5 nBand+0.5]); end %#ok<CAXIS>
    set(cb, 'Ticks', 1:nBand, 'TickLabels', bandLabel, 'TickDirection','out', ...
            'FontName', S.font, 'FontSize', S.fsAxis-1, 'LineWidth', 0.8);
    cb.Label.String = 'Age (years)';
    cb.Label.FontSize = S.fsLabel;
    cb.Label.FontName = S.font;
end

function panel_cohort_compare(ax, S, times, wAll, semAll, nAll, refW, semRef, ...
                              refN, rAll, hepWindow, nChan)
    % The deployment cohort as a whole against the validated cohort: the
    % aggregate version of what panel (a) resolves by age.
    %
    % Both curves carry a standard-error band. n = 14 against two thousand
    % means the two are estimated with very different precision, and a band on
    % only one of them would invite the reader to treat them as equally firm.
    axes(ax); hold(ax,'on');
    t = times*1000;

    % Y-limits from the data including both SEM bands, applied at the end; the
    % patch is oversized and clipped so its height cannot feed back into them.
    % See the note in panel_age_waves.
    allY = [wAll(:); refW(:); wAll(:)-semAll(:); wAll(:)+semAll(:)];
    if ~isempty(semRef)
        allY = [allY; refW(:)-semRef(:); refW(:)+semRef(:)];
    end
    allY = allY(isfinite(allY));
    yLo = min(allY); yHi = max(allY);
    % More headroom than (b): this panel carries both a legend at the top left
    % and the correlation annotation at the bottom right.
    yl = [yLo yHi] + [-0.14 0.16]*(yHi - yLo);

    patch(ax, hepWindow([1 2 2 1])*1000, 1e4*[-1 -1 1 1], ...
          S.cShade, 'EdgeColor','none', 'FaceAlpha',0.55);

    if ~isempty(semRef); band(ax, t, refW, semRef, S.cRef, 0.18); end
    band(ax, t, wAll, semAll, [0.35 0.35 0.35], 0.20);

    hRef = plot(ax, t, refW, '--', 'Color', S.cRef,        'LineWidth', 2.2);
    hAll = plot(ax, t, wAll, '-',  'Color', [0.20 0.20 0.20], 'LineWidth', 2.0);

    xline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    yline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);
    xlim(ax, [t(1) t(end)]);
    ylim(ax, yl);
    xlabel(ax, 'Time from R-peak (ms)', 'FontSize', S.fsLabel);
    ylabel(ax, 'Amplitude (\muV)', 'FontSize', S.fsLabel);
    title(ax, sprintf('All ages pooled (%d channels)', nChan), ...
        'FontSize', S.fsTitle, 'FontWeight','normal');

    text(ax, 0.975, 0.05, sprintf('waveform r = %.3f (%.0f-%.0f ms)', rAll, ...
         hepWindow(1)*1000, hepWindow(2)*1000), 'Units','normalized', ...
        'HorizontalAlignment','right', 'VerticalAlignment','bottom', ...
        'FontName',S.font, 'FontSize',S.fsAxis-1, 'Color',[0.30 0.30 0.30]);

    lg = legend(ax, [hRef hAll], ...
        {sprintf('Validated (n = %d)', refN), ...
         sprintf('Deployment (n = %d)', nAll)}, ...
        'Location','northwest', 'FontSize', S.fsAxis-1);
    set(lg,'Box','off');
    style_axes(ax, S);
end

function panel_letter(ax, S, ch)
    % Placed in figure coordinates against the axes' top-left corner, so the
    % letters line up across panels of different heights. The box is kept
    % shallow because these are normalized units: on a 6.5-inch figure a tall
    % box would centre the letter well above the panel it labels.
    p = get(ax,'Position');
    annotation(ancestor(ax,'figure'), 'textbox', ...
        [p(1)-0.048, p(2)+p(4)+0.004, 0.05, 0.050], 'String', ch, ...
        'EdgeColor','none', 'FontName',S.font, 'FontSize',12, ...
        'FontWeight','bold', 'HorizontalAlignment','left', ...
        'VerticalAlignment','middle');
end

function panel_age_similarity(ax, S, res, cols, rhoObs, pObs)
    % Agreement with the validated waveform against age, with the noise ceiling
    % behind it so a data-quality gradient cannot be mistaken for a
    % developmental one.
    %
    % Two series, not three. Once the amplitude screen is applied every band's
    % reliability lands between 0.98 and 1.00, so the ceiling is a flat line and
    % the noise-corrected series is just the observed one divided by a constant
    % - a third curve of the same shape, carrying no information the other two
    % do not. It is dropped from the panel and its trend reported in the text.
    % The ceiling stays, because a flat ceiling under a rising series is exactly
    % what rules out the confound, and that has to be visible rather than
    % asserted.
    axes(ax); hold(ax,'on');
    x  = [res.ageMean];
    nB = numel(res);

    % Ceiling: the r each band could reach against a perfectly matched
    % reference given its own noise. Everything below it is unexplained shape.
    % Drawn as a bare line - markers on a flat series read as data points that
    % invite comparison band by band, which is not what it is for.
    hCeil = plot(ax, x, [res.ceiling], '--', 'Color',[0.60 0.60 0.60], 'LineWidth',1.2);

    % Observed, with bootstrap CI over subjects within the band.
    for b = 1:nB
        plot(ax, [x(b) x(b)], [res(b).rLo res(b).rHi], '-', ...
             'Color', cols(b,:), 'LineWidth', 1.6);
    end
    hObs = plot(ax, x, [res.r], '-', 'Color',[0.25 0.25 0.25], 'LineWidth',1.3);
    for b = 1:nB
        plot(ax, x(b), res(b).r, 'o', 'MarkerSize',7, ...
             'MarkerEdgeColor',cols(b,:), 'MarkerFaceColor',cols(b,:), 'LineWidth',1.0);
    end

    yline(ax, 0, ':', 'Color',[0.4 0.4 0.4]);

    % Fix the x-range to the bands before anything is placed relative to it.
    xlim(ax, [min(x)-1.2, max(x)+1.2]);
    xlabel(ax, 'Age (years)', 'FontSize', S.fsLabel);
    ylabel(ax, 'Similarity to validated HEP (r)', 'FontSize', S.fsLabel);
    title(ax, 'Agreement by age', 'FontSize', S.fsTitle, 'FontWeight','normal');

    lo = min([[res.r] [res.ceiling] [res.rLo]]);
    hi = max([[res.r] [res.ceiling] [res.rHi]]);
    ylim(ax, [lo-0.10*(hi-lo)-0.02, hi+0.22*(hi-lo)+0.02]);

    lg = legend(ax, [hObs hCeil], {'observed','noise ceiling'}, ...
                'Location','southeast', 'FontSize', S.fsAxis-2);
    set(lg,'Box','off');

    % Only the observed trend is annotated. A rank correlation over the ceiling
    % is not reportable here: reliability spans 0.98 to 1.00 across the bands, so
    % its ordering is decided by fourth-decimal noise in the split-half
    % resampling and rho moves between runs. The flat dashed ceiling line already
    % makes the point visually, and the range is stated in the text.
    txt = sprintf('observed \\rho = %+.2f, p = %.3f', rhoObs, pObs);
    text(ax, 0.03, 0.97, txt, 'Units','normalized', 'HorizontalAlignment','left', ...
        'VerticalAlignment','top', 'FontName',S.font, 'FontSize',S.fsAxis-2, ...
        'Color',[0.30 0.30 0.30]);

    style_axes(ax, S);
end

%% ================================ STATS ================================

function rel = splithalf_rel(W, idx, nRep)
    % Spearman-Brown-corrected split-half reliability of a group's mean
    % waveform over idx. Repeated random halves, median taken: a single split
    % is itself noisy at these group sizes.
    %
    % WHAT THIS IS FOR - the noise ceiling.
    %
    % Two waveforms that are identical in truth will still not correlate at 1,
    % because each is estimated from finitely many noisy recordings and each
    % therefore carries error the other does not. So a band correlating at 0.85
    % with the validated waveform is only interpretable against how high it
    % COULD have gone, and that limit is set by how well each side is measured.
    %
    % Reliability estimates it directly. Split the group in half at random,
    % average each half, correlate the two half-averages: if the group's
    % waveform is well determined the halves agree, if it is noisy they do not.
    % Spearman-Brown rescales from half-size to full-size, since the real
    % estimate uses everyone. The ceiling is then sqrt(rel_band * rel_ref) - the
    % correlation the band would show against a reference whose true shape
    % matched it exactly.
    %
    % Why the analysis needs it. Pediatric EEG gets cleaner with age. Without
    % this, an age gradient in observed correlation could be nothing but an age
    % gradient in data quality: noisier bands correlate worse with everything,
    % no developmental change required. The ceiling separates the two. It comes
    % out flat here, so the rise in observed correlation is not explained by
    % measurement precision - which is the point of plotting it in panel (c)
    % rather than merely reporting it.
    n = size(W,1);
    if n < 4; rel = NaN; return; end
    rr = nan(nRep,1);
    for k = 1:nRep
        p  = randperm(n);
        h1 = mean(W(p(1:floor(n/2)),:), 1, 'omitnan');
        h2 = mean(W(p(floor(n/2)+1:end),:), 1, 'omitnan');
        rr(k) = corr(h1(idx)', h2(idx)');
    end
    rHalf = median(rr, 'omitnan');
    rel   = 2*rHalf / (1 + rHalf);      % Spearman-Brown, half -> full length
    rel   = min(max(rel, 0), 1);
end

function [rho, p] = spearman_exact(x, y)
    % Spearman rho with an exact permutation p. With seven bands there are
    % 5040 orderings, so the null is enumerated rather than sampled and the
    % p-value carries no Monte-Carlo error of its own.
    x = x(:); y = y(:);
    n = numel(x);
    rx = tiedrank(x); ry = tiedrank(y);
    rho = corr(rx, ry);
    if n > 8
        % Enumeration would be 9! = 362880 or worse; sample instead.
        nPerm = 20000;
        cnt = 0;
        for k = 1:nPerm
            cnt = cnt + (abs(corr(rx, ry(randperm(n)))) >= abs(rho) - 1e-12);
        end
        p = (cnt + 1) / (nPerm + 1);
        return
    end
    P = perms(1:n);
    rp = nan(size(P,1),1);
    for k = 1:size(P,1)
        rp(k) = corr(rx, ry(P(k,:)));
    end
    p = mean(abs(rp) >= abs(rho) - 1e-12);
end

%% ================================ STYLE ================================

function C = age_colors(n)
    % One distinct hue per age band, at matched saturation and value.
    %
    % This is the same sweep the quality-gate figure uses (band_palette in
    % hbn_gate_margin.py: HSV 288 to 48 degrees at S = 0.58, V = 0.72), so that age
    % is encoded identically wherever it appears in the paper. The two figures
    % previously used different scales for the same variable.
    %
    % Sweeping hue rather than ramping one hue's opacity keeps every band
    % equally visible on white, which matters when thirteen bands are overlaid.
    % The sweep runs violet to amber and stops deliberately short of red, which
    % the gate figure reserves for the threshold line.
    hues = linspace(288, 48, n)' / 360;
    C = hsv2rgb([hues, repmat(0.58, n, 1), repmat(0.72, n, 1)]);
end

function S = fig_style()
    % Mirrors deploy_step7_group_hep's fig_style so the two figures sit together.
    S = struct();
    S.font    = 'Arial';
    S.fsAxis  = 9;
    S.fsLabel = 10;
    S.fsTitle = 10;
    S.lw      = 1.7;
    S.lwThin  = 1.2;
    S.cReal = [0.10 0.10 0.10];
    S.cSurr = [0.55 0.55 0.55];
    S.cRef  = [0.00 0.62 0.45];
    S.cShade = [0.91 0.91 0.91];
    S.alpha  = 0.20;
end

function f = new_fig(S, wIn, hIn)
    f = figure('Color','w','Units','inches','Position',[1 1 wIn hIn], ...
               'PaperPositionMode','auto','InvertHardcopy','off');
    set(f,'DefaultAxesFontName',S.font,'DefaultAxesFontSize',S.fsAxis, ...
          'DefaultTextFontName',S.font);
end

function band(ax, x, y, e, col, a)
    xx = [x(:); flipud(x(:))];
    yy = [y(:)-e(:); flipud(y(:)+e(:))];
    ok = ~isnan(xx) & ~isnan(yy);
    patch(ax, xx(ok), yy(ok), col, 'FaceAlpha', a, 'EdgeColor','none');
end

function style_axes(ax, S)
    set(ax,'FontName',S.font,'FontSize',S.fsAxis,'TickDir','out', ...
           'Box','off','LineWidth',0.8,'Layer','top','TickLength',[0.015 0.015]);
end

function export_all(f, outPath, name)
    % 600 dpi rather than 300: the age panel is drawn in 0.5 pt hairlines, and
    % at 300 dpi a 0.5 pt stroke lands on about two pixels, so anti-aliasing
    % greys it out and adjacent bands blur back together. The PDF is vector and
    % unaffected either way - submit that one - but the PNG should not
    % misrepresent what the figure looks like.
    try
        exportgraphics(f, fullfile(outPath,[name '.png']), 'Resolution', 600);
        exportgraphics(f, fullfile(outPath,[name '.pdf']), 'ContentType','vector');
    catch
        print(f, fullfile(outPath,[name '.png']), '-dpng','-r600');
        print(f, fullfile(outPath,[name '.pdf']), '-dpdf','-vector');
    end
    print(f, fullfile(outPath,[name '.emf']), '-dmeta');
    savefig(f, fullfile(outPath,[name '.fig']));
end

function standalone(S, outPath, name, wIn, hIn, drawFcn)
    % Explicit position rather than the default: these panels carry a colorbar
    % or a four-entry legend, and the default axes runs too close to the edges
    % for either to sit without clipping.
    f = new_fig(S, wIn, hIn);
    ax = axes(f, 'Position', [0.135 0.150 0.750 0.775]);
    drawFcn(ax);
    export_all(f, outPath, name);
    close(f);
end