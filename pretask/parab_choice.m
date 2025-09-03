%% Function to fit the parabolic effort discounting model to the choice data (accept or reject work offer) in Müller, Husain, & Apps (2022, Scientific Reports)

% The code is adapted from the function to fit the parabolic effort discounting model to the Pre-Task choice data in Müller, Klein-Flügge, Manohar, Husain, & Apps (2021, Nature Communications).

% The function uses negative log-likelihood and fminsearch, with participant data
% stored in/taken from "s"

% 02/06/2025 updated by SL - changed starting points from rand to random between min and max
% and changed beta max

%%

function model_results = parab_choice(s)


num_subs = max(size(s));
%num_models = 1; just for info, running only 1 model

model_names = {'parabolic_discount'};


models = { 
@(p,reward,effort,chosen,base) prob_para(p,reward,effort,chosen,base);

};

model_params = 2; % k and beta

% set bounds for k and beta
k_min = .0278; % kmin = (Rmin - Rrest) / Emax^2 - see explanation parab_choice_run
k_max = 1.5; % based on script for max k
beta_min = .01; % > 0
beta_max = 5; % based on visual inspection of the data 

for j = 1:num_subs
       
    reward = s(j).choice_only_reward;
    effort = s(j).choice_only_effort;
    chosen = s(j).choice_only_choice;
    
    base = 1; % reward for rest
    large_penalty = 1e20; % Large enough to strongly discourage constraint violations
    
    for z =  1 % here this is a 'loop' over 1 model but could be more  if you want to check linear and hyperbolic
        pfunc = (models{z});
        neg_likelihood_func = @(p) -sum(log(eps+pfunc(p,reward,effort,chosen,base))) ;
        p{z}=[]; 
        nll(z)=inf;
        
        % use different random starting values on 50 interations
        for k=1:50
          constrained_nll = @(p) neg_likelihood_func(p) + (p(1)<k_min)*large_penalty + (p(1)>k_max)*large_penalty + (p(2)<beta_min)*large_penalty + (p(2)>beta_max)*large_penalty;
          % 2 random starting points as we are estimating 2 values in this
          % model
          startp(1) = k_min + rand * (k_max - k_min);
          startp(2) = beta_min + rand * (beta_max - beta_min);

          [pk, nllk] = fminsearch( constrained_nll  , startp, ...  
            optimset('MaxFunEvals',10000,'MaxIter',10000) ); % find best params 
       
          if nllk<nll(z)  % is this better than previous estimates?
            nll(z)=nllk; p{z}=pk; % if so, update the 'best' estimate
          end
          model_results.all_nll(j,z,k) = nllk;
          model_results.all_p{j,z,k}  = pk;
        end
        
        % for each model, store probabilities of each trial  
        model_results.prob{j,z} =   pfunc(p{z},reward,effort,chosen,base);
        
        % to get some insight in model fit and parameters - sub is nr of
        % subject not subject ID from SubjectArray
        fprintf('Sub %d | k = %.4f | beta = %.4f | NLL = %.4f | minP = %.4f | maxP = %.4f\n', ...
        j, p{z}(1), p{z}(2), nll(z), min(model_results.prob{j,z}), max(model_results.prob{j,z}));
    end 

    model_results.likelihood(j,:) = nll; % LIKELIHOOD (SUBJECT, MODEL)
    model_results.params{j,:} = p; % PARAMS {SUBJECT, MODEL} ( PARAMETER_NUMBER )
  
    % Compute AIC and BIC
    num_trials = length(effort); % number of trials used in the fit
    aic = 2*model_params + 2*nll;
    bic = 2*nll + model_params.*log(num_trials);

    model_results.aic(j) = aic;
    model_results.bic(j) = bic;
end

param_matrix = zeros(num_subs, 2); % Preallocate

for j = 1:num_subs
    param_matrix(j, :) = model_results.params{j}{1}; % [k, beta]
end

model_results.param_matrix = param_matrix; % k and beta parameters
model_results.k = param_matrix(:, 1);    % k values
model_results.beta = param_matrix(:, 2); % beta values
model_results.models       = models;
model_results.model_names  = model_names;
model_results.model_params = model_params;


% function for parabolic discounting, optionally add functions for hyperbolic and linear 

function prob = prob_para(p,reward,effort,chosen,base)
% p(1) = single discount parameter k
% p(2) = softmax beta

k = p(1);
beta = p(2);
val = reward-(k.*effort.^2); % value of work option
prob =  exp(val.*beta)./(exp(base*beta) + exp(beta.*val)); % If participant chose work (choice=1): probability = prob
prob(~chosen) =  1 - prob(~chosen); % If participant chose rest (choice=0): probability = 1 - prob
prob = prob(:); % per trial