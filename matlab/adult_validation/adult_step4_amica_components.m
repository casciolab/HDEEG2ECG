%% adult_step4_amica_components - Save AMICA Component Matrices
%
% This script loads AMICA results and saves the component matrices to a folder
%
%% DEFINING PATHS AND GROUPS
clc % clear CW
clear % clear Workspace

% Get fullpath
fullpath = mfilename('fullpath');

% Path manipulation
basePath = fileparts(fullpath); % this pipeline's folder (any OS)
addpath(fileparts(basePath));   % for hdeeg2ecg_paths.m
dirs = hdeeg2ecg_paths();       % input, output and toolbox folders (see README)

% Add path and open eeglab
addpath(genpath(dirs.eeglab)) % EEGLAB 2020.0 with its plugins (see README)

% Which R-peaks this run is locked to: 'true' (from the recorded ECG) or
% 'reconstructed' (from the HDEEG2ECG toolbox). Run steps 1-6 once with each;
% step 7 compares the two. Choose with setenv('HEP_TRIGGER', ...).
trigger = dirs.trigger;
assert(any(strcmp(trigger, {'true','reconstructed'})), 'HEP_TRIGGER must be ''true'' or ''reconstructed''.');
branchPath = fullfile(dirs.adult,[trigger '_rpeaks']);

%% SAVE AMICA COMPONENT MATRICES
for gi = 1
    % Define paths
    amicaPath = fullfile(branchPath,'adult_step3_amica');
    savePath = fullfile(branchPath,'adult_step4_amica_components');
    
    % Create save directory
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end
    
    % Get subject directories
    cd(amicaPath);
    sDir = dir('*');
    sDir = sDir([sDir.isdir]); % Keep only directories
    sDir = sDir(~ismember({sDir.name},{'.','..'})); % Remove . and ..
    
    % Process each subject
    for si = 1:length(sDir)
        sName = sDir(si).name;
        subjectFolder = fullfile(amicaPath, sName);
        
        fprintf('Processing: %s\n', sName);
        
        try
            % Load AMICA output structures
            mod = loadmodout15(fullfile(subjectFolder));
            
            % Extract actual matrices from the structure
            W = mod.W;  % Unmixing matrix
            S = mod.S;  % Sphering matrix
            A = mod.A;  % Mixing matrix (if available)
            
            % Create subject save folder
            subjectSavePath = fullfile(savePath, sName);
            if ~exist(subjectSavePath, 'dir')
                mkdir(subjectSavePath);
            end
            
            % Save matrices
            save(fullfile(subjectSavePath, 'W.mat'), 'W', '-v7.3');
            save(fullfile(subjectSavePath, 'S.mat'), 'S', '-v7.3');
            if exist('A', 'var')
                save(fullfile(subjectSavePath, 'A.mat'), 'A', '-v7.3');
                fprintf('  Saved W (%dx%d), S (%dx%d), A (%dx%d)\n', ...
                    size(W,1), size(W,2), size(S,1), size(S,2), size(A,1), size(A,2));
            else
                fprintf('  Saved W (%dx%d) and S (%dx%d)\n', ...
                    size(W,1), size(W,2), size(S,1), size(S,2));
            end
            
        catch ME
            warning('Error');
        end
    end
end

fprintf('\nDone!\n');