%% deploy_step3_amica - Adaptive Mixture Independent Component Analysis
%
% Quick AMICA automatization to remove eye movement components from our signal.
% 
% AMICA is an ML algorythm that fits the best model to specific components
% by using 2nd Order Newton
% (https://en.wikipedia.org/wiki/Newton%27s_method)
%
%
% AMICA Tutorial at: 
% https://sccn.ucsd.edu/githubwiki/files/eeg_nonstationarity_and_amica.pdf
%
% Paper at:
%   Palmer, Jason A., Ken Kreutz-Delgado, and Scott Makeig. ...
%   "AMICA: An adaptive mixture of independent component analyzers with shared components." ...
%   Swartz Center for Computatonal Neursoscience, ...
%   University of California San Diego, Tech. Rep (2012): 1-15.

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

%% LOAD AND PREPROCESS
% Start group iteration
for gi = 1

    % Define load and save path
    loadPath = fullfile(dirs.deploy,'deploy_step2_preprocess');
    savePath = fullfile(dirs.deploy,'deploy_step3_amica');
    
    % Define subjects directory
    cd(loadPath);
    sDir = dir('*.set'); 
    sDir = sDir(~ismember({sDir.name},{'.','..'}));% Stay only with .set within dir
    

    % Start iteration through subjects
    % --- Optimized Params ---
    max_threads = 4;   % 4 threads per subject
    max_iter = 500;    % Usually enough for artifact rejection
    num_models = 1;
    numprocs = 1;      % Keep at 1 if using parfor below
    
    % Start parallel iteration through subjects
    parfor si = 135:length(sDir)
        % 161 NDARLA395AG8
        % Load Subject (Using local EEG variable to avoid parfor conflicts)
        sName = sDir(si).name;
        fprintf('Processing: %s\n', sName);
        localEEG = pop_loadset('filename', sName, 'filepath', loadPath);
        
        % Create Dir
        subjDir = fullfile(savePath, sName(1:end-7));
        if ~exist(subjDir, 'dir'); mkdir(subjDir); end
    
        % Calculate Rank
        dataRank = sum(eig(cov(double(localEEG.data'))) > 1E-6);
    
        % Run AMICA (Silent mode)
        runamica15(localEEG.data(1:128,:), 'num_models', num_models, ...
            'outdir', subjDir, 'numprocs', numprocs, ...
            'max_threads', max_threads, 'max_iter', max_iter, ...
            'pcakeep', dataRank); % verbose 0 saves console time
    end
end
