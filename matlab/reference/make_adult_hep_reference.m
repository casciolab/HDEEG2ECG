function ref = make_adult_hep_reference(validationMat, hepGroupMat, hepByAgeMat, outFile)
%MAKE_ADULT_HEP_REFERENCE Group-level adult HEP reference used by deploy_step7/8.
%   ref = make_adult_hep_reference() builds matlab/reference/adult_hep_reference.mat
%   from three analysis OUTPUTS, without reading any recording or epoch:
%     validationMat  adult_step7_hep_validation/validation_results.mat
%                    (the validated cluster: roiChannels, roiMask)
%     hepGroupMat    deploy_step7_group_hep/hep_group.mat
%                    (the adult reference remapped onto the deployment time axis:
%                    per-subject averages, cluster mask, SEM of the ROI waveform)
%     hepByAgeMat    deploy_step8_hep_by_age/hep_by_age.mat
%                    (split-half reliability of the adult reference)
%   Only group-level quantities are saved: no per-subject waveforms, no IDs.
%   With no arguments the files are taken from the default outputs/ folders
%   (see hdeeg2ecg_paths); pass paths to use files elsewhere.

here = fileparts(mfilename('fullpath'));
if nargin < 4
    addpath(fileparts(here));
    dirs = hdeeg2ecg_paths();
    validationMat = fullfile(dirs.adult, 'adult_step7_hep_validation', 'validation_results.mat');
    hepGroupMat   = fullfile(dirs.deploy, 'deploy_step7_group_hep', 'hep_group.mat');
    hepByAgeMat   = fullfile(dirs.deploy, 'deploy_step8_hep_by_age', 'hep_by_age.mat');
    outFile       = fullfile(here, 'adult_hep_reference.mat');
end

V = load(validationMat, 'results');  V = V.results;
H = load(hepGroupMat, 'hep');        H = H.hep;
B = load(hepByAgeMat, 'byAge');      B = B.byAge;

% --- consistency between the three outputs ---
primary = find(any(H.maskRef, 2))';
assert(isequal(primary, H.primaryChans), 'maskRef and primaryChans disagree in %s', hepGroupMat);
assert(isequal(sort(V.roiChannels), primary), ...
    'Cluster channels differ between the adult validation and the deployment outputs.');
assert(isequal(B.primaryChans, primary) && B.refN == H.refN, ...
    'hep_by_age.mat was not computed from this hep_group.mat.');

ga = mean(double(H.refAll), 3);                               % adult grand average, nChan x nT
w  = mean(ga(primary, :), 1);
assert(max(abs(w - double(H.refW)), [], 'omitnan') < 1e-5, 'Grand average does not reproduce refW.');
subjW = squeeze(mean(double(H.refAll(primary, :, :)), 1))';   % used only for this check
sem = std(subjW, 0, 1, 'omitnan') / sqrt(H.refN);
assert(max(abs(sem - double(H.semRef)), [], 'omitnan') < 1e-6, 'SEM does not reproduce semRef.');

% --- group-level reference ---
ref = struct();
ref.description  = ['Adult heartbeat-evoked potential, epochs locked to the R-peaks of the ' ...
                    'recorded ECG (adult_validation, HEP_TRIGGER = true). Group level only.'];
ref.n            = H.refN;                    % adults
ref.times        = H.times;                   % s, the 256-Hz axis of the HBN analysis
ref.channels     = 'EGI HydroCel-128, native order (channels 1-128)';
ref.ga           = ga;                        % uV, grand average, channels x times
ref.mask         = logical(H.maskRef);        % validated cluster, channels x times
ref.primaryChans = primary;                   % channels anywhere in the cluster
ref.waveform     = w;                         % uV, ROI waveform (mean over primaryChans)
ref.sem          = double(H.semRef);          % uV, SEM of the ROI waveform across adults
ref.reliability  = B.relRef;                  % split-half (Spearman-Brown) reliability
ref.hepWindow    = B.hepWindow;               % s
ref.made         = char(datetime('now', 'Format', 'yyyy-MM-dd'));
ref.sources      = {validationMat; hepGroupMat; hepByAgeMat};
[~, n1, e1] = cellfun(@fileparts, ref.sources, 'UniformOutput', false);
ref.sources      = strcat(n1, e1);           % file names only

save(outFile, 'ref', '-v7');
fprintf('wrote %s\n  n = %d adults, %d channels in the cluster, %d time points (%.0f to %.0f ms)\n', ...
    outFile, ref.n, numel(primary), numel(ref.times), 1000*ref.times(1), 1000*ref.times(end));
end
