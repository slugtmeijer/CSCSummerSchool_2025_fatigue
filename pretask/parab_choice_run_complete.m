%% Wrapper to run parab_choice to get values for disc (k) and beta from the pretask (NFT)
% code for summer school Sept 2025 - finished
% Selma Lugtmeijer
% written in MATLAB 2019b

clear
rng default % resets the randomisation seed to ensure results are reproducible

cd '/Users/selmalugtmeijer/Library/CloudStorage/OneDrive-UniversityofBirmingham/Birmingham/courses/CSC2025_SummerSchool_complete/data/';
savedir = '/Users/selmalugtmeijer/Library/CloudStorage/OneDrive-UniversityofBirmingham/Birmingham/courses/CSC2025_SummerSchool_complete/results_pre/';

%% experiment information
prac_length = 18; %how many practice trials
NFT_length = 75; % how many trials in the Not Fatigue Task (NFT) part of the experiment

% What files need to be loaded?
SubjectArray = [2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 21 22 23 24 25 27 28 29 31 32 33 34 35 36 37 38 39 40];
gr = {'_HC.mat'};
n = length(SubjectArray);

% Create array for variables per subject
s = struct('choice_only_reward', [], 'choice_only_effort', [], 'choice_only_choice', []);
n_rest = zeros(1,n);
% Expand s to all subs
s = repmat(s, n, 1);

% Initialize variables. For each trial how many people choose which option
rest = zeros(1,NFT_length);
work = zeros(1,NFT_length);
miss = zeros(1,NFT_length);

%% Loop over subjects, load subject data, extract relevant variables
for j = 1:n
    try
        fullFilePath = fullfile(pwd, [num2str(SubjectArray(j)) gr{1}]);
        load(fullFilePath);
        
        % Initialize arrays to store the data
        choice_only_reward = zeros(1, NFT_length);
        choice_only_effort = zeros(1, NFT_length);
        choice_only_choice = zeros(1, NFT_length);

        % Extract data from the struct array
        for i = 1:NFT_length
            idx = prac_length + i;
            choice_only_effort(i) = result.data(idx).effortLevel;
            choice_only_reward(i) = result.data(idx).stake / 2 + 1; % to get levels of reward (2-6) instead of credits (2-10), rest is 1 is level 1
            choice_only_choice(i) = result.data(idx).choice;
            % count per trial index which reponse was given across subjects
            if result.data(idx).choice == 0
                rest(i) = rest(i) + 1;
            elseif result.data(idx).choice == 1
                work(i) = work(i) + 1;
            else 
                miss(i) = miss(i) + 1;
            end    
        end

        % Remove rows where choice_only_choice is 2 (missed/no response)
        valid_indices = choice_only_choice ~= 2;
        choice_only_reward = choice_only_reward(valid_indices);
        choice_only_effort = choice_only_effort(valid_indices);
        choice_only_choice = choice_only_choice(valid_indices);
        
        % Assign the extracted data to the subject's struct
        s(j).choice_only_reward = choice_only_reward;
        s(j).choice_only_effort = choice_only_effort;
        s(j).choice_only_choice = choice_only_choice;

        % Find subjects that always always work (1) - per subject how often
        % rest
        n_rest(j) = length(find(s(j).choice_only_choice==0));

    catch err
        disp(['Error in iteration ', num2str(j), ': ', err.message]);
    end
end

%% run parab_choice function to get model parameters
model_results_pre = parab_choice(s);

%% replace ceiling effect parameters and save parameters
% Subjects that always choose to work have no variance to explain so get k
% close to 0 and a beta of 1 as effort/reward levels don't predict choice
% Mathematical explanation:
% Vrest = 1
% Veffort >= Vrest (value of any work option is higher than rest)
% Veffort = Rmin - k * Emax^2 (parabolic discounting for worst-case
% scenario, for Veffort = 1)
% Where is Veffort 1?
% kmin = (Rmin - 1) / Emax^2
% Rmin = lowest effort reward = 2; Emax = max effort = 6
% (2-1)/6^2 = .0278 (minimum possible k value where someone would still
% choose the worst work option)
% Temperature parameter/beta = 1 (moderate randomness), as there's no data 
% to inform what the "true" sensitivity should be.

for i = 1:length(n_rest)
    if n_rest(i) == 0
        model_results_pre.params{i} = {[.0278, 1]}; 
        model_results_pre.param_matrix(i, :) = [.0278, 1];  
    end
end

%% save data
model_results_pre.SubjectArray = SubjectArray;
model_results_pre.n_rest = n_rest;

% save model results so it can be used for the main task
filename = [savedir, 'model_results_pretask_HC'];

save(filename,'model_results_pre')

%%
% plot some data to get insight in responses per trial across subjects e.g.
% misses across subjects per trial number
figure; 
plot(miss); 
title('Misses per trial index across subjects');
xlabel('Trial Index pre-task');
ylabel('Count of misses');
grid on; % Add a grid for better readability

% plot k and beta values