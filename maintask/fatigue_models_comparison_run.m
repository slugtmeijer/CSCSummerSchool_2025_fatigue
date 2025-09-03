%% Wrapper to run fatigue_models_comparison
% Summer school Sept 2025
% written in Matlab 2019b

clear
rng default % resets the randomisation seed to ensure results are reproducible

cd '/Users/selmalugtmeijer/Library/CloudStorage/OneDrive-UniversityofBirmingham/Birmingham/courses/CSC2025_SummerSchool/data/';
savedir = '/Users/selmalugtmeijer/Library/CloudStorage/OneDrive-UniversityofBirmingham/Birmingham/courses/CSC2025_SummerSchool/';

nmodels = 4; % how many models are we comparing

%% experiment information
prac_length = 18; %how many practice trials
NFT_length = 75; % how many trials in the NFT part of the experiment
FT_length = 180; % how many trials in the fatigue part of the experiment

%% What files need to be loaded?
SubjectArray = [2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 18 19 21 22 23 24 25 27 28 29 31 32 33 34 35 36 37 38 39 40 41]; % HC 41 ceiling
filename = {'test_HC'} ;  
session = {'_HC.mat'};        

n = length(SubjectArray);

% create array for variables per subject
s = struct('choice_fat_reward', [], 'choice_fat_effort', [], 'choice_fat_choice', [], 'k', [], 'beta', []);
n_rest = zeros(1,n);

% Expand s to all subs
s = repmat(s, n, 1);

rest = zeros(1,FT_length);
work = zeros(1,FT_length);
miss = zeros(1,FT_length);

% load file parameters (discounting and beta) from pretask 
FilePath = fullfile([savedir,  'results_pre/model_results_pretask_HC.mat']);
load(FilePath);

% loop over subjects, load subject data, extract relevant variables
for j = 1:n
    try
        fullFilePath = fullfile(pwd, [num2str(SubjectArray(j)) session{1}]);
        load(fullFilePath);

        % Initialize arrays to store the data
        choice_fat_reward = zeros(1, FT_length);
        choice_fat_effort = zeros(1, FT_length);
        choice_fat_choice = zeros(1, FT_length);
        choice_fat_outcome = zeros(1, FT_length); % not taken into account for modelling

        % Extract data from the struct array
        for i = 1:FT_length
            idx = prac_length + NFT_length + i;
            choice_fat_effort(i) = result.data(idx).effortLevel;
            choice_fat_reward(i) = result.data(idx).stake/2+1; % from credits to reward levels
            choice_fat_choice(i) = result.data(idx).choice;
            % count per trial index which reponse was given across subjects
            if result.data(idx).choice == 0
                rest(i) = rest(i) + 1;
            elseif result.data(idx).choice == 1
                work(i) = work(i) + 1;
            elseif result.data(idx).choice == 2 
                miss(i) = miss(i) + 1;
            end    
        end

        % Assign the extracted data to the subject's struct
        s(j).choice_fat_reward = choice_fat_reward;
        s(j).choice_fat_effort = choice_fat_effort;
        s(j).choice_fat_choice = choice_fat_choice;

        % Find misses per subject (2)
        n_miss(j) = length(find(s(j).choice_fat_choice==2));

        % Misses / not responded in time (choice 2) is scored as rest
        s(j).choice_fat_choice(s(j).choice_fat_choice == 2) = 0;

        % Assign parameters from pretask to subject's struct
        s(j).k = model_results_pre.k(j);
        s(j).beta = model_results_pre.beta(j);

        % Find subjects that never rest (0) / always work
        n_rest(j) = length(find(s(j).choice_fat_choice==0));

    catch err
        disp(['Error in subject ', num2str(j), ': ', err.message]);
    end
end

%%
%     % plot some data to decide whether to discart the first trial for modeling
%     figure; 
%     plot(miss(1:5)); % plot just first 5 trials
%     title('Misses per trial index - EC');
%     xlabel('Trial Index fatigue-task');
%     xticks(1:5);
%     ylabel('Count of misses across subjects');
%     yticks(0:6);
%     ylim([0 6]);
%     grid on; % Add a grid for better readability

%% find subjects with ceiling effect (accept all work offers)
ceiling = find(n_rest == 0);
sub_id_ceiling = SubjectArray(ceiling);
all_indices = 1:length(s);
sub_id_notceiling = SubjectArray(setdiff(all_indices, ceiling));
s1 = s(setdiff(all_indices, ceiling));

%% run model fits
% subjects without ceiling effect
fat_model_results = fatigue_models_comparison(s1); 

% subjects with ceiling effect (always work)
s2 = s(ceiling);
if ~isempty(s2)
    fat_model_results_ceiling = fatigue_models_comparison_ceiling(s2);
    % Create a new struct combining both ceiling and not-ceiling
    fat_model_results_combined = [fat_model_results, fat_model_results_ceiling];
else
    disp('Skipping model comparison: no subjects with ceiling effect.');
end

% Define fields that need to be concatenated (excluding first 3 fields)
fields_to_concat = {'all_nll', 'all_p', 'prob', 'Value_work', 'STfatigue', 'LTfatigue', 'Fatigue', 'aic', 'likelihood', 'params', 'bic'};

% Define fields that need to be concatenated (excluding first 3 fields)
for i = 1:length(fields_to_concat)
    current_field = fields_to_concat{i};
    fat_model_results_combined_HC.(current_field) = vertcat(fat_model_results.(current_field), fat_model_results_ceiling.(current_field));
end

% get IDs from non ceiling and ceiling effect subjects per group in
% order they were ran though model fitting
fat_model_results_combined_HC.ReorderedSubIDs = [sub_id_notceiling, sub_id_ceiling];

%% Create matrix with all model parameters
param_matrix = zeros(n, 7); % 7 parameters across 4 models
subject_ids = fat_model_results_combined_HC.ReorderedSubIDs(:); % Get subject IDs

for i = 1:n
    param_cell = fat_model_results_combined_HC.params(i,:); % 1x4 cell
    param_matrix(i, :) = [ ...
        param_cell{1}(:)' ... % 3 from Model 1
        param_cell{2}(:)' ... % 2 from Model 2
        param_cell{3}(:)' ... % 1 from Model 3
        param_cell{4}(:)' ... % 1 from Model 4
    ];
end

% Create table
T = array2table(param_matrix, ...
    'VariableNames', {'M1_STr', 'M1_STf', 'M1_LF', ...
                      'M2_STr', 'M2_STf', ...
                      'M3_LF', 'M4_k'});
T.SubID = subject_ids(:); % Add subject ID

final_table = T;

% Reorder columns
final_table = final_table(:, [{'SubID'}, final_table.Properties.VariableNames(1:7)]);
% Reorder so subjects with ceiling are not at the end of each group
final_table = sortrows(final_table, {'SubID'});

%% Save data

% Save model results
filename = [savedir, 'results_main/model_results_fatTask_HC'];
save(filename,'fat_model_results_combined_HC')

% Save table
filename = [savedir, 'results_main/fatigue_model_parameters.mat'];
save(filename, 'final_table');
csv_filename = strrep(filename, '.mat', '.csv');
writetable(final_table, csv_filename);

% Save workspace
filename = [savedir, 'results_main/fatTask_fit_results_', date];
save([filename, '.mat']);