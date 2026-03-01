function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Initialization of variables %
[m, n] = size(X);
X_norm = zeros(m, n);
mu = zeros(1, n);
sigma = zeros(1, n);

% Calculate the mean of each feature %
mu = mean(X);

% Calculate the standard deviation of each feature %
sigma = std(X);

X_norm = zeros(size(X));

% Normalize the features in X %
X_norm = (X - mu) ./ sigma;

% ============================================================

end