%% adult_step2_preprocess - EEG Preprocessing Script
% Steps: Filter → ZAPLINE → ASR → Bad Channels
%
% This script performs EEG preprocessing in four main steps:
%
% F - Filter:
%     Applies bandpass filtering to remove slow drifts (e.g., below 0.5 Hz)
%     and high-frequency noise (e.g., above 30 Hz). Optional notch filtering
%     can remove line noise (e.g., 60 Hz).
%
% Z - ZAPLINE:
%     Uses the zapline-plus algorithm to adaptively remove line noise
%     and its harmonics without introducing distortion.
%
% A - ASR (Artifact Subspace Reconstruction):
%     Automatically detects and removes transient, high-amplitude artifacts
%     using statistical thresholding (via the clean_artifacts() function).
%
% B - Bad Channels:
%     Identifies and marks persistently bad EEG channels for later interpolation,
%     based on criteria such as flatline duration or correlation with neighbors.

%% DEFINING PATHS AND GROUPS
clc % clear CW
clear % clear Workspace

% Get fullpath
fullpath = mfilename('fullpath');

% Path manipulation
basePath = fileparts(fullpath); % this pipeline's folder (any OS)
addpath(fileparts(basePath));   % for hdeeg2ecg_paths.m
dirs = hdeeg2ecg_paths();       % input, output and toolbox folders (see README)

% Add path of eeglab
addpath(genpath(dirs.eeglab)) % EEGLAB 2020.0 with its plugins (see README)

% Which R-peaks this run is locked to: 'true' (from the recorded ECG) or
% 'reconstructed' (from the HDEEG2ECG toolbox). Run steps 1-6 once with each;
% step 7 compares the two. Choose with setenv('HEP_TRIGGER', ...).
trigger = dirs.trigger;
assert(any(strcmp(trigger, {'true','reconstructed'})), 'HEP_TRIGGER must be ''true'' or ''reconstructed''.');
branchPath = fullfile(dirs.adult,[trigger '_rpeaks']);

% Define heart channel
heartChannel = 129;

%% LOAD AND PREPROCESS
% Start group iteration
for gi = 1

    % Define load and save path
    loadPath = fullfile(branchPath,'adult_step1_import');
    savePath = fullfile(branchPath,'adult_step2_preprocess');

    % Define subjects directory
    cd(loadPath)
    sDir = dir('*.set');
    sDir = sDir(~ismember({sDir.name},{'.','..'})); % Stay only with .set within dir

    % Start iteration through subjects
    for si = 1:size(sDir,1)
        % 87, 208, 223
        %% LOAD SUBJECT
        % Load the subject and redraw the GUI
        EEG = pop_loadset('filename',sDir(si).name,'filepath',loadPath);
        EEG = eeg_checkset( EEG );
        eeglab redraw % Redraw the GUI
        
        % Filtering
        EEG = pop_eegfiltnew( EEG , 30, 0.5); % Low 0.5 and High 30 for HEP (we can later play with this but this numbers are literature guided see Salamone et al. 2021)
        EEG = eeg_checkset( EEG ); % Reload to save within variables
        eeglab redraw;
    
        EEG = pop_clean_rawdata(EEG, ...
            'FlatlineCriterion', 'off', ... 
            'ChannelCriterion', 'off', ... % We already handled channels
            'LineNoiseCriterion', 'off', ... 
            'Highpass', 'off', ... % We already filtered
            'BurstCriterion', 'off', ... 
            'WindowCriterion', 0.25, ... 
            'WindowCriterionTolerances', [-Inf 7],...
            'ignore_channels', heartChannel); 
        
        EEG = eeg_checkset(EEG);
        EEGHeart = EEG.data(heartChannel,:); EEG.data(heartChannel,:) = zeros(length(EEG.data(heartChannel,:)),1)';
        
        % Remove flat channels before running Zapline
        flatChannels = find(std(EEG.data, 0, 2) == 0);  % find channels with zero variance
        EEGClone = EEG;
        if ~isempty(flatChannels)
            warning('Removing flat channels: %s', strjoin({EEG.chanlocs(flatChannels).labels}, ', '))
            EEG.data(flatChannels, :) = [];
            EEG.chanlocs(flatChannels) = [];
            EEG.nbchan = size(EEG.data, 1);
        end

        % Run zapline plus
        EEG = clean_data_with_zapline_plus_eeglab_wrapper(EEG,struct('noisefreqs','line','maxfreq',30,'plotResults', 0,'winSizeCompleteSpectrum',floor(length(EEG.data)/EEG.srate/2)));
        EEG = eeg_checkset( EEG );
        eeglab redraw;
        close all
        
        %% Bad channels
        [EEG,~,~,removed_channels] = clean_artifacts(EEG,'WindowCriterion','off'); % Modified to avoid deleting start and end task events
        EEG = eeg_checkset( EEG ); % Reload to save within variables
        EEG.nbchan = heartChannel;
        

        % Create a logical mask for ALL original channels (1 to heartChannel)
        all_missing_mask = true(1, heartChannel);
        
        % Identify which labels are currently present in the cleaned EEG
        present_labels = {EEG.chanlocs.labels};
        original_labels = {EEGClone.chanlocs(1:heartChannel).labels};
        
        for i = 1:heartChannel
            if any(strcmp(original_labels{i}, present_labels))
                all_missing_mask(i) = false; % It's present, so it's NOT missing
            end
        end
        
        %% 2. Reconstruct the full data matrix (including heart)
        newData = zeros(heartChannel, size(EEG.data, 2));
        current_eeg_idx = 1;
        
        for i = 1:heartChannel
            if ~all_missing_mask(i)
                % This channel exists in the current EEG.data
                % EXCEPT if it's the heartChannel (which we handle separately)
                if i == heartChannel
                     newData(i,:) = EEGHeart;
                else
                     newData(i,:) = EEG.data(current_eeg_idx, :);
                     current_eeg_idx = current_eeg_idx + 1;
                end
            else
                % It's a flat channel or an artifact channel; leave as zeros for now
            end
        end

        % Update EEG structure
        EEG.data = newData;
        EEG.nbchan = heartChannel;
        EEG.chanlocs = EEGClone.chanlocs(1:heartChannel);
        clear newData;
        
        %% 3. Interpolate
        % Now we find the indices of everything that was missing (excluding the heart)
        interp_idx = find(all_missing_mask);
        interp_idx(interp_idx == heartChannel) = []; % NEVER interpolate the heart channel

        if ~isempty(interp_idx)
            fprintf('Interpolating %d channels...\n', length(interp_idx));
            EEG = pop_interp(EEG, interp_idx, 'spherical');
        end
        
        % Final sync
        EEG.data(heartChannel,:) = EEGHeart; % Ensure heart is untouched by interpolation
        EEG = eeg_checkset(EEG);
        EEG = clean_windows(EEG,0.25,[-inf 7]); % Mod to avoid deleting Start and end marks from data.
        
        % Re-referencing
        EEG = pop_reref( EEG , [],'exclude', heartChannel); % Rereference to average channels (modify if using different references)
        EEG = eeg_checkset( EEG ); % Reload to save within variables
        eeglab redraw % Redraw the GUI

        %% SAVE SET AND GO TO THE NEXT ITERATION
        pop_saveset(EEG, [sDir(si).name(1:4),'_s2.set'], savePath) % Save it in savePath
        eeglab redraw; % Redraw the GUI
    end
end