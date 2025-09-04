%% Function for the fatigue experiment in order to fit different (fatigue) models to choice data (accept or reject work offer) and compare them (Müller, Klein-Flügge, Manohar, Husain, & Apps, 2021, Nature Communications)
%%% This script models the short term effects of fatigue
%%% and re-energising and also the long-term effects of fatigue over time.

% The function uses negative log-likelihood and fminsearch, with participant data
% stored in/taken from "s".

% SL This function fits the models with all fatigue parameter values set to
% .001
% correction for subjects that always chose to work on the main task
% force fit models with 0.001 for all fatigue parameters
% so here all fatigue parameters are set to a fixed value close to zero as
% apparently fatigue does not explain variance in the data
% select only data of subjects with ceiling effect and run forced model fit

%% modelling

function fat_model_results_ceiling = fatigue_models_comparison_ceiling(s2)

% Go to folder with script
% cd .. 

% How many subjects are you modelling?
num_subs = max(size(s2));

% How many models do you want to compare?
num_models = 4;

% What are the names of the models?
model_names = {'fat_full','fat_ST','fat_LT','est_disc'};
% 1) Short Term and Long Term fatigue effects
% 2) Only Short Term fatigue
% 3) Only Long Term fatigue
% 4) Calculate k and beta from main task (don't use pretask), no fatigue

% Define the variables included in the model
% p - parameters estimated and fit to each subjects' data: k, beta (main
% task)
% R - trial by trial offered reward
% E - trial by trial offered effort
% choice - what they chose on each trial (needs to be 1 = work, 0 = rest)
% Missed trials are treated as a rest
% k (initial discount variable from pretask)
% beta (initial softmax/temperature variable from pretask)

models = {
    @(p,R,E,choice,base,STfat,LTfat,Fat,k,beta) prob_fat_full(p,R,E,choice,base,STfat,LTfat,Fat,k,beta);
    @(p,R,E,choice,base,STfat,LTfat,Fat,k,beta) prob_fat_ST(p,R,E,choice,base,STfat,LTfat,Fat,k,beta);
    @(p,R,E,choice,base,STfat,LTfat,Fat,k,beta) prob_fat_LT(p,R,E,choice,base,STfat,LTfat,Fat,k,beta);
    @(p,R,E,choice,base,STfat,LTfat,Fat,k,beta) prob_est_disc(p,R,E,choice,base,STfat,LTfat,Fat,k,beta);
    };

% How many parameters are you estimating in each model?
% These are listed in the order they are
model_params = [3 2 1 1];

% Name of some of the outcome variables
fat_model_results_ceiling.models       = models;
fat_model_results_ceiling.model_names  = model_names;
fat_model_results_ceiling.model_params = model_params;

%% Loop over subjects.
for j = 1:num_subs
    
    % load in each subjects variables for the experiment
    R = s2(j).choice_fat_reward; %offered reward
    E = s2(j).choice_fat_effort; % offered effort
    choice = s2(j).choice_fat_choice; % always 1 (work) in this case because these are subjects with ceiling effect
    
    k = s2(j).k;
    beta = s2(j).beta;
     
    % Give the fatigue variabels values (these should add up to 1
    % in this model because they multiply the discounting parameter. So 1 means
    % no effect of fatigue.
    STfat(1) = 0.5;
    LTfat(1) = 0.5;
    Fat(1) = 1;

    %what is the value of the baseline
    base = 1;
    
    %% Loop over models
    
    for z = 1:num_models      
        
        %which models are we going to run?
        pfunc = (models{z});
        
        % Instead of optimization, set fixed parameters based on number of parameters in model
        if model_params(z) == 1
            pkBestIter = 0.001;    
        elseif model_params(z) == 2
            pkBestIter = [0.001, 0.001];
        elseif model_params(z) == 3
            pkBestIter = [0.001, 0.001, 0.001];
        end

        % Calculate negative log likelihood with fixed parameters
        neg_likelihood_func = @(p) -sum(log(eps+pfunc(p,R,E,choice,base,STfat,LTfat,Fat,k,beta)));
        nll(z) = neg_likelihood_func(pkBestIter);  % Store nll for each model
        p{z} = pkBestIter;  % Store parameters for each model
        
        % Store other results
        fat_model_results_ceiling.all_nll(j,z) = nll(z);
        fat_model_results_ceiling.all_p{j,z} = pkBestIter;    
    
        % Calculate model probabilities and other metrics
        [fat_model_results_ceiling.prob{j,z}, fat_model_results_ceiling.Value_work{j,z}, ...
         fat_model_results_ceiling.STfatigue{j,z}, fat_model_results_ceiling.LTfatigue{j,z}, ...
         fat_model_results_ceiling.Fatigue{j,z}] = pfunc(pkBestIter,R,E,choice,base,STfat,LTfat,Fat,k,beta);
    end

    % After z loop - model fits using all models' nll values
    likelihood = nll;
    fat_model_results_ceiling.likelihood(j,:) = likelihood;

    num_trials = length(E);  % how many choices this subject made
    aic = 2*model_params + 2*likelihood;  
    bic = 2*likelihood + model_params .* log(num_trials); 

    fat_model_results_ceiling.aic(j,:) = aic;
    fat_model_results_ceiling.bic(j,:) = bic;

    fat_model_results_ceiling.params(j,:) = p; 
end


%% From here below are some possible models of the data.

% k = initial discounting parameter k from pretask
% beta = initial beta from pretask

% model 1 - full model
function [prob, Value_work, STfatigue, LTfatigue, Fatigue] = prob_fat_full(p,R,E,choice,base,STfat,LTfat,Fat,k,beta)
% p(1) = Short-term Effect of Rest
% p(2) = Short-term effect of effort
% p(3) = Long-term fatigue effect

STfat(1) = 0.5;
LTfat(1) = 0.5;
Fat(1) = 1;
val_work(1) = 0;

% It runs through chocies and updates the values of STFAT and LTFAT to up
% and down-weight the value of working on each trial
for t = 1:length(choice)
    val_work(t) = (R(t)) - (Fat(t)*k*(E(t).^2)); % ^ parabolic
    % to compare, this is the formula for discounting (k) in the pretask:
    % val = reward-(k.*effort.^2);
    % so here the current fatigue influences the discounting and in turn
    % the value of the work offer on that trial

    if choice(t) == 1
        STfat(t+1) = STfat(t) + (p(2)*E(t)) - (p(1)*0.57); % .57 is proportion of rest if you should the work option
        LTfat(t+1) = LTfat(t)+ p(3)*E(t);
    elseif choice(t) == 0
        STfat(t+1) = STfat(t) - (p(1)*1);
        LTfat(t+1) = LTfat(t);
    end
    
    if STfat(t+1) < 0.5 % SL assumption is that you don't get less fatigued
        STfat(t+1) = 0.5; % resting might reduce STfat but not overall fat
    else
        STfat(t+1) = STfat(t+1);
    end
    Fat(t+1) = STfat(t+1) + LTfat(t+1); % these summed should always be > 1, you only get more fatigued
    
end

%SoftMax

probs =  exp(val_work.*beta)./(exp(base*beta) + exp(beta.*val_work)); 
probs(~choice) =  1 - probs(~choice);

prob = probs';
Value_work = val_work';
STfatigue = STfat';
LTfatigue = LTfat';
Fatigue = Fat';

% model 2 - two different terms for STfat (effort(p2) and rest (p1)) no
% LTfat - SL in this model p1 and p2 where coded the other way around in 
% the TM script, I've switched this to be consistent
function [prob, Value_work, STfatigue, LTfatigue, Fatigue] = prob_fat_ST(p,R,E,choice,base,STfat,LTfat,Fat,k,beta)

STfat(1) = 1;
Fat(1) = 1;
val_work(1) = 0;

for t = 1:length(choice)
    val_work(t) = R(t) - (Fat(t)*k*(E(t).^2));
    
    if choice(t) == 1
        STfat(t+1) = STfat(t) + (p(2)*E(t)) - (p(1)*.57);
    elseif choice(t) == 0
        STfat(t+1) = STfat(t) - (p(1)*1);
    end
    
    if STfat(t+1) < 1
        STfat(t+1) = 1;
    else
        STfat(t+1) = STfat(t+1);
    end
    Fat(t+1) = STfat(t+1);
    
end

probs =  exp(val_work.*beta)./(exp(base*beta) + exp(beta.*val_work)); 
probs(~choice) =  1 - probs(~choice);

prob = probs';
Value_work = val_work';
STfatigue = STfat';
LTfatigue = LTfat';
Fatigue = Fat';

% model 3 - only LTfat - current fatigue plus (or not) effect of fatigue
% based on current effort
function [prob, Value_work, STfatigue, LTfatigue, Fatigue] = prob_fat_LT(p,R,E,choice,base,STfat,LTfat,Fat,k,beta)

LTfat(1) = 1;
Fat(1) = 1;
val_work(1) = 0;

for t = 1:length(choice)
    val_work(t) = R(t) - (Fat(t)*k*(E(t).^2));
    
    if choice(t) == 1
        LTfat(t+1) = LTfat(t)+ p(1)*E(t);
    elseif choice(t) == 0
        LTfat(t+1) = LTfat(t);
    end
    
    Fat(t+1) =  LTfat(t+1);
    
end

probs =  exp(val_work.*beta)./(exp(base*beta) + exp(beta.*val_work)); 
probs(~choice) =  1 - probs(~choice);

prob = probs';
Value_work = val_work';
STfatigue = STfat';
LTfatigue = LTfat';
Fatigue = Fat';

% model 4 - instead of using k from pretask reestimate - no fatigue
% parameter in this model
function [prob, Value_work, STfatigue, LTfatigue, Fatigue] = prob_est_disc(p,R,E,choice,base,STfat,LTfat,Fat,k,beta)

val_work(1) = 0;

for t = 1:length(choice)
    val_work(t) = R(t) - (p(1)*(E(t).^2)); % same function as pretask
end

probs =  exp(val_work.*beta)./(exp(base*beta) + exp(beta.*val_work)); 


probs(~choice) =  1 - probs(~choice);

prob = probs';
Value_work = val_work';
STfatigue = STfat';
LTfatigue = LTfat';
Fatigue = Fat';
