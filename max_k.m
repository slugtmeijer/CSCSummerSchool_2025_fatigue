minEffort = 2; % minimum effort level (rest = 1)
maxReward = 6; % maximum reward level (rest = 1)

% maximum k calculated as the discount rate that means the
% maximum reward and minimum effort has a value of 'low'
low = 0; % the best work option has exactly zero utility 
% - i.e., someone would be indifferent between the best work option (effort=2, reward=5) and rest (value=1)
maxKp = (maxReward - low) ./ (minEffort.^2); % parabolic
maxKl = (maxReward - low) ./ (minEffort); % linear
maxKh = ((maxReward * (1/low)) - 1) / minEffort; % hyperbolic
if maxKh == Inf
  maxKh = max(maxKp, maxKl);
end