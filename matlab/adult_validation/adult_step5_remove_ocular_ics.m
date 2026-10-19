%% adult_step5_remove_ocular_ics - Remove ICA Eye components
%
% After adding the ICA information to our EEG structure, we will use an
% automatic ICA labeling function to select the eye components found by ICA
% and remove them from the data.

%% DEFINING PATHS AND GROUPS
clc % clear CW
clear % clear Workspace

% Get fullpath
fullpath = mfilename('fullpath');

% Path manipulation
basePath = fileparts(fullpath); % this pipeline's folder (any OS)
addpath(fileparts(basePath));   % for hdeeg2ecg_paths.m
dirs = hdeeg2ecg_paths();       % input, output and toolbox folders (see README)
addpath(genpath(dirs.eeglab)) % EEGLAB 2020.0 with its plugins (see README)

% Which R-peaks this run is locked to: 'true' (from the recorded ECG) or
% 'reconstructed' (from the HDEEG2ECG toolbox). Run steps 1-6 once with each;
% step 7 compares the two. Choose with setenv('HEP_TRIGGER', ...).
trigger = dirs.trigger;
assert(any(strcmp(trigger, {'true','reconstructed'})), 'HEP_TRIGGER must be ''true'' or ''reconstructed''.');
branchPath = fullfile(dirs.adult,[trigger '_rpeaks']);

% Add path and open eeglab

%% LOAD AND PREPROCESS
% Start group iteration
for gi = 1

    % Define load and save path
    loadPath = fullfile(branchPath,'adult_step2_preprocess');
    loadPath2 = fullfile(branchPath,'adult_step3_amica');
    savePath = fullfile(branchPath,'adult_step5_remove_ocular_ics');

    % Define subjects directory
    cd(loadPath2);
    sDir = dir();
    sDir = sDir([sDir(:).isdir]);
    sDir = sDir(~ismember({sDir.name},{'.','..'}));% Stay only with .set within dir


    % Start iteration through subjects
    for si = 1:length(sDir)
        %146, 241, 242, 257, 261
        % Subject Name and load bad channels
        sName = sDir(si).name;
        
        %% LOAD SUBJECT
        % Load the subject and redraw the GUI
        %% LOAD SUBJECT
        % Load the subject and redraw the GUI
        [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; %#ok
        EEG = pop_loadset('filename', [sName, '_s2.set'], 'filepath', loadPath);
        
        % --- Proper Channel Removal ---
        % This handles .data, .chanlocs, .nbchan, and .specicaall in one go
        EEG = pop_select(EEG, 'nochannel', 129);
        
        % Update the dataset in the ALLEEG main structure
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        eeglab redraw;

        % load individual ICA model into EEG structure and modify to fit the 130
        % channels      
        EEG.etc.amica  = loadmodout15([loadPath2, filesep, sName]);

        % ONLY IF RUNNING MORE THAN ONE MODEL TO CHECK WHICH ONE TO USE
        % % compute model probability
        % model_prob = 10 .^ modout.v; % modout.v (#models x #samples)
        % figure, imagesc(model_prob(:,1:10*EEG.srate));
        
        % model_index = 1; % Use first model
        EEG.etc.amica.S = EEG.etc.amica.S(1:EEG.etc.amica.num_pcs, :); % Weirdly, I saw size(S,1) be larger than rank. This process does not hurt anyway.
        EEG.icaweights = EEG.etc.amica.W;
        EEG.icasphere  = EEG.etc.amica.S;
        EEG = eeg_checkset(EEG, 'ica');

        % Label ICA components to later remove
        EEG = iclabel(EEG);
        EEG = eeg_checkset(EEG);
        eeglab redraw

        % Only select eye components and remove them
        EEG = pop_icflag(EEG, [0 0; 0.9 1; 0.9 1; 0 0; 0 0; 0 0; 0 0]); % see function help message
        rejected_comps = find(EEG.reject.gcompreject > 0);
        EEG = pop_subcomp(EEG, rejected_comps);
        EEG = eeg_checkset(EEG);
        eeglab redraw
        
        % Save the subject
        pop_saveset(EEG, [sName,'_s5.set'], savePath) % Save it in savePath
        eeglab redraw; % Redraw the GUI
        
    end
end