function [eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [eigenval, eigenvec, order] = myPCA(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order
%

% Useful values
[m, n] = size(X);

% Make sure each feature from the data is zero mean
X_centered = X - mean(X);

% ====================== YOUR CODE HERE ======================

% Compute the covariance matrix
covariance_matrix = (1 / m) * (X_centered' * X_centered);

% Compute the eigenvectors and eigenvalues of the covariance matrix
[eigenvec, eigenval] = eig(covariance_matrix);

% Sort eigenvalues in descending order
[eigenval, order] = sort(diag(eigenval), 'descend');
eigenvec = eigenvec(:, order);

% =========================================================================

end
