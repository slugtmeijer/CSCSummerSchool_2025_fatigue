%% Function for the fatigue experiment in order to fit different (fatigue) models to choice data (accept or reject work offer) and compare them (Müller, Klein-Flügge, Manohar, Husain, & Apps, 2021, Nature Communications)
% Adapted for summer school 2025 by Selma Lugtmeijer
%
% This script models the short term effects of fatigue and re-energising 
% and also the long-term effects of fatigue over time.
% The function uses negative log-likelihood and fminsearch, with participant data
% stored in/taken from "s".
%
% ratio rest to work/effort trials = 0.57:1 
% .57 being the proportion of the trial that is also rest in the work choice
% 11.5s for a trial: 4s for a choice, 1s to get ready, 5s for effort, 1s outcome, 0.5s ITI 
% so as a proportion that would be 0.57 (6.5/11.5s rest on work trials for 11.5 rest on rest trials)


%% modelling

function fat_model_results = fatigue_models_comparison(s1)

% Go to folder with script
% cd .. 

% How many subjects are you modelling?
num_subs = max(size(s1));

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
fat_model_results.models       = models;
fat_model_results.model_names  = model_names;
fat_model_results.model_params = model_params;

%% Loop over subjects
for j = 1:num_subs   
        
    % load in each subjects variables for the experiment
    R = s1(j).choice_fat_reward; %offered reward
    E = s1(j).choice_fat_effort; % offered effort
    choice = s1(j).choice_fat_choice; % 0 for rest and miss, 1 for work
    
    k = s1(j).k; % k pretask
    beta = s1(j).beta; % beta pretask 
    
    % Give the fatigue variabels some starting values (these should add up to 1
    % in this model because they multiply the discounting parameter. So 1 means
    % no effect of fatigue. Just to initialize variables, they are
    % specified in each model function too.
    STfat(1) = 0.5;
    LTfat(1) = 0.5;
    Fat(1) = 1;

    % Reward for rest
    base = 1;
       
    %% Loop over models
    
    for z = 1:num_models
        
        % Which model are we going to run?
        pfunc = (models{z});
        
        % Use a negative log likelihood function. We are aiming to minimise the
        % distance between the choices made by subjects and the prediction of
        % the model
        neg_likelihood_func = @(p) -sum(log(eps+pfunc(p,R,E,choice,base,STfat,LTfat,Fat,k,beta))) ;  % adds a small value 'eps' to prevent taking the log of zero
        p{z}=[]; % store optimized parameters for k and beta main task from the model
        nll(z)=inf; % In optimization problems, we often want to keep track of the best solution found so far. Starting with infinity ensures that any valid solution will be considered better than the initial state
        nllBestIter = inf;

        % 100 interations with random starting points for possible values
        % of parameters
        for ii=1:100
            % clear variables between iterations
            clear startp
            clear constrained_nll

            large_penalty = 1e20; % Large enough to strongly discourage constraint violations, in this case parameters should not be negative
            
            if model_params(z) == 1
                startp = rand*3; % fatigue values tend to be below 3 based on Müller
                constrained_nll = @(p) neg_likelihood_func(p) + (p(1)<0)*large_penalty; % only constained is not to go below 0 as that would not be meaningful               
            elseif model_params(z) == 2
                startp = [rand*3,rand*3];
                constrained_nll = @(p) neg_likelihood_func(p) + (p(1)<0)*large_penalty + (p(2)<0)*large_penalty;
            elseif model_params(z) == 3
                startp = [rand*3,rand*3,rand*3];
                constrained_nll = @(p) neg_likelihood_func(p) + (p(1)<0)*large_penalty + (p(2)<0)*large_penalty + (p(3)<0)*large_penalty;
            end
        
            % Begin the search for the best parameters that minimise the distance
            % between the probability of choices on each trial and the values from the
            % model.
            
            [pk, nllk] = fminsearch( constrained_nll  , startp, ...  % find best params pk with their nll
                optimset('MaxFunEvals',10000,'MaxIter',10000) );
            % Is this iteration's fit better than previous iterations?
            if nllk<nllBestIter
                nllBestIter = nllk;
                pkBestIter = pk;
            end
        end    
        
        % After all iterations, is this model's best fit better than before?
        if nllBestIter<nll(z)  % is this better than previous estimates? Save for each model
            nll(z)=nllBestIter; p{z}=pkBestIter; % if so, update the 'best' estimate    
        end
        
        fat_model_results.all_nll(j,z) = nllBestIter;
        fat_model_results.all_p{j,z}  = pkBestIter;    

        % for each model, store probabilities of each trial
        [ fat_model_results.prob{j,z}, fat_model_results.Value_work{j,z}, fat_model_results.STfatigue{j,z}, fat_model_results.LTfatigue{j,z}, fat_model_results.Fatigue{j,z} ] =   pfunc(p{z},R,E,choice,base,STfat,LTfat,Fat,k,beta);
        
   end % next model
    
    % model fits based on best iteration
    likelihood = nll;
    fat_model_results.likelihood(j,:) = likelihood;

    num_trials = length(E);  % how many choices this subject made
    aic = 2*model_params + 2*likelihood;  
    bic = 2*likelihood + model_params .* log(num_trials); 

    fat_model_results.aic(j,:) = aic;
    fat_model_results.bic(j,:) = bic;

    fat_model_results.params(j,:) = p;
  
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
