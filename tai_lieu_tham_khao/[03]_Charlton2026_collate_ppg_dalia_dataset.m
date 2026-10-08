%% =========================================================================
% collate_ppg_dalia_dataset.m
% 
% Collation script for the PPG-DaLiA dataset:
% Converts and combines individual subject data into a unified MATLAB dataset structure.
%
% Author: Dr. Peter H. Charlton
% Department of Public Health and Primary Care, University of Cambridge
% Reference: [3] P. H. Charlton (2026), collate_ppg_dalia_dataset.m
% Project Citation: Do an Cuoi ky AIoT - Nguyen Bach Tung (MSSV: 23110166)
% =========================================================================

function collate_ppg_dalia_dataset()

    fprintf('--- Starting PPG-DaLiA Dataset Collation ---\n');

    %% 1. Setup paths and parameters
    up = setup_up();

    %% 2. Check source data existence
    if ~exist(up.paths.root_folder, 'dir')
        error('Root data folder not found: %s\nPlease download and unpack PPG-DaLiA dataset first.', up.paths.root_folder);
    end

    %% 3. Iterate through all 15 subjects (S1 to S15)
    subject_ids = 1:15;
    collated_data = struct();

    for s_idx = 1:length(subject_ids)
        s_num = subject_ids(s_idx);
        subj_str = sprintf('S%d', s_num);
        mat_file = fullfile(up.paths.root_folder, subj_str, [subj_str, '.mat']);

        fprintf('Processing Subject %s...\n', subj_str);

        if ~exist(mat_file, 'file')
            warning('MAT file %s not found. Run convert_subject_pickle_files_to_mat.py first.', mat_file);
            continue;
        end

        % Load subject data structure
        raw = load(mat_file);

        %% =================================================================
        % CRITICAL SIGNAL AND LABEL DISTINCTION (Section 2.2 of Project):
        % -----------------------------------------------------------------
        % 1. raw.activity (fs = 4 Hz):
        %    Ground truth Human Activity Recognition (HAR) categorical labels.
        %    Classes: 0=transient, 1=sitting, 2=stairs, 3=table soccer,
        %             4=cycling, 5=driving car, 6=lunch, 7=walking, 8=working.
        %    MUST be interpolated using Zero-Order Hold (ZOH) / nearest-neighbor
        %    to avoid fictitious floating-point class labels.
        %
        % 2. raw.label (fs = 0.5 Hz):
        %    Reference ECG Heart Rate (BPM) calculated every 2.0 seconds
        %    from chest ECG R-peak intervals over 8.0-second sliding windows.
        % =================================================================

        % Extract Wrist Sensors (Empatica E4)
        collated_data.(subj_str).wrist.acc = raw.signal.wrist.ACC;       % 32 Hz, 3-axis
        collated_data.(subj_str).wrist.bvp = raw.signal.wrist.BVP;       % 64 Hz, PPG
        collated_data.(subj_str).wrist.eda = raw.signal.wrist.EDA;       % 4 Hz
        collated_data.(subj_str).wrist.temp = raw.signal.wrist.TEMP;     % 4 Hz

        % Extract Chest Sensors (RespiBAN Professional)
        collated_data.(subj_str).chest.ecg = raw.signal.chest.ECG;       % 700 Hz
        collated_data.(subj_str).chest.resp = raw.signal.chest.RESP;     % 700 Hz
        collated_data.(subj_str).chest.acc = raw.signal.chest.ACC;       % 700 Hz

        % Extract Ground Truths
        collated_data.(subj_str).har_activity = raw.activity;            % 4 Hz (HAR Classes 0-8)
        collated_data.(subj_str).ecg_heart_rate = raw.label;             % 0.5 Hz (Reference HR in BPM)

        % Perform Zero-Order Hold (ZOH) alignment of HAR activity labels to 32 Hz ACC
        fs_acc = 32;
        fs_act = 4;
        upsample_factor = fs_acc / fs_act; % factor of 8
        collated_data.(subj_str).har_activity_aligned_32hz = repelem(raw.activity, upsample_factor);

        fprintf('  Done %s: %d ACC samples, %d HAR labels.\n', ...
            subj_str, length(raw.signal.wrist.ACC), length(raw.activity));
    end

    %% 4. Save consolidated collation file
    save_path = fullfile(up.paths.save_folder, 'ppg_dalia_collated.mat');
    save(save_path, 'collated_data', '-v7.3');
    fprintf('Successfully saved collated dataset to %s\n', save_path);

end

function up = setup_up()
    % Setup path configurations
    up.paths.root_folder = fullfile(pwd, 'data', 'raw', 'PPG_DaLiA');
    up.paths.save_folder = fullfile(pwd, 'data', 'processed');
    if ~exist(up.paths.save_folder, 'dir')
        mkdir(up.paths.save_folder);
    end
end
