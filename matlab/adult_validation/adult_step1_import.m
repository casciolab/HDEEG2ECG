%% adult_step1_import - Loading and downsampling.
%
% This code loads and downsamples the data. 
%
% All scripts within this HEP preprocessing repository are intended to be...
% used for HEP analysis but can be modified to best suit your needs.
%
% │﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌ │
% │    ﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎﹎    │
% │  ║    First step of Evoked Potentials analysis   ║    │
% │    ﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊﹊    │
% │                     _ ---~~(~~-_.                     │
% │                   _{         )    )                   │ 
% │                 ,    ) -~~- ( ,- ' )_                 │ 
% │               (    `-,_..`., )--  '_,)                │
% │              (   ` _)  (  -~( -_ `,  }                │
% │              (_-    _ ~_-~~~~`,  ,'  )                │
% │                 `~ -^(     __;-,((()))                │
% │                       ~~~~ {_ -_(())                  │  
% │                               `\  }                   │
% │                                 { }                   │
% │                                                       │
% │   ASCII ART: From: Steven James Walker                │
% │              <swalker1@emerald.tufts.edu>             │
% │                                                       │
%  ﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌

% │﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌│
% │ ● Inputs:  ► Raw EEG data                             │
% │                                                       │
% │ ● Outputs: ► .set file with downsampled data          │
% │                                                       │
% │                                                       │
% │ ● Reqs:    ► Eeglab                                   │
% │            ► Guide folder organization                │
% │                                                       │
%  ﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌ 

% │﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌│                                               
% │ ● Contact: ► matias.fraile@ku.edu                     │
% │            ► carissa.cascio@ku.edu                    │
% │                                                       │
%  ﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌ 

%% LICENSE 
% The license is intended to apply to all codes within this repository.
%
% The MIT License
% 
% Copyright (c) 2011 Dominic Tarr
% 
% Permission is hereby granted, free of charge, 
% to any person obtaining a copy of this software and 
% associated documentation files (the "Software"), to 
% deal in the Software without restriction, including 
% without limitation the rights to use, copy, modify, 
% merge, publish, distribute, sublicense, and/or sell 
% copies of the Software, and to permit persons to whom 
% the Software is furnished to do so, 
% subject to the following conditions:
% 
% The above copyright notice and this permission notice 
% shall be included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
% OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
% ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
% SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
eeglab % Open eeglab

% Define sampling rate
srate = 512;

%% LOAD AND PREPROCESS
% Start group iteration
for gi = 1

    % Define load and save paths
    if strcmp(trigger, 'true')
        loadPath = fullfile(dirs.inputs,'adult','recordings');      % R-peaks of the recorded ECG
    else
        loadPath = fullfile(dirs.adult,'toolbox','03_good');       % toolbox output, same recordings
    end
    savePath = fullfile(branchPath,'adult_step1_import');

    if ~exist(savePath, 'dir')
        mkdir(savePath); % Create save directory if it doesn't exist
    end

    % Get list of subject files (.set)
    sDir = dir(fullfile(loadPath, '*.set'));
    sDir = sDir(~ismember({sDir.name}, {'.', '..'})); % Just in case (redundant here, but harmless)

    % Iterate through subjects
    for si = 1:numel(sDir)

        %% LOAD SUBJECT (SET FILES USED HERE - MODIFY IF NEEDED)
        EEG = pop_loadset('filename', sDir(si).name, 'filepath', loadPath); % Load .set (events already stored in the file)
        EEG = eeg_checkset(EEG);

        % The adult recordings carry two external channels: 129 was not used and 130
        % is the ECG. Dropping 129 puts the ECG on channel 129, where step 2 expects
        % it (the toolbox output already has this layout).
        if EEG.nbchan == 130
            EEG = pop_select(EEG, 'nochannel', 129);
            EEG = eeg_checkset(EEG);
        end

        %% PREPROCESSING STEP: RESAMPLING
        EEG = pop_resample(EEG, srate); % Downsample to 512 Hz
        EEG = eeg_checkset(EEG);

        %% SAVE SET WITH NEW NAME
        % Construct new filename by removing last two parts of original name
        nameParts = strsplit(sDir(si).name, '_'); % CHECK YOUR DELIMITER AND LENGTH
        if numel(nameParts) < 3
            error('Unexpected filename structure for: %s', sDir(si).name);
        end
        newName = [strjoin(nameParts(2:end-1), '_'), '_s1.set'];

        pop_saveset(EEG, 'filename', newName, 'filepath', savePath);
        eeglab redraw;

    end
end