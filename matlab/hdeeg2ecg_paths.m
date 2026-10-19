function dirs = hdeeg2ecg_paths()
%HDEEG2ECG_PATHS Folders used by the MATLAB pipelines.
%   Data go in <repo>/inputs and every result is written to <repo>/outputs.
%   To use other locations, or another deployment dataset, set the environment
%   variables below with setenv before running a step, e.g.
%       setenv('HDEEG2ECG_DATASET', 'my_study')
repoRoot = fileparts(fileparts(mfilename('fullpath')));
dirs.inputs    = envOr('HDEEG2ECG_INPUT_DIR',  fullfile(repoRoot, 'inputs'));
dirs.outputs   = envOr('HDEEG2ECG_OUTPUT_DIR', fullfile(repoRoot, 'outputs'));
dirs.eeglab    = envOr('EEGLAB_PATH',    fullfile(repoRoot, 'matlab', 'toolboxes', 'eeglab', 'eeglab2020_0'));
dirs.fieldtrip = envOr('FIELDTRIP_PATH', fullfile(repoRoot, 'matlab', 'toolboxes', 'fieldtrip-20240110'));
dirs.trigger   = envOr('HEP_TRIGGER', 'true');        % adult validation: 'true' or 'reconstructed'
dirs.dataset   = envOr('HDEEG2ECG_DATASET', 'hbn');   % deployment: name of the dataset folder
dirs.adult     = fullfile(dirs.outputs, 'adult_validation');
dirs.deploy    = fullfile(dirs.outputs, 'deployment', dirs.dataset);
end

function v = envOr(name, default)
v = getenv(name);
if isempty(v), v = default; end
end
