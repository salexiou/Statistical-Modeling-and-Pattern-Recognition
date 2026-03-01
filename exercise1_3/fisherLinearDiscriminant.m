function v = fisherLinearDiscriminant(X1, X2)
    % Compute the mean vectors for each class
    mu1 = mean(X1);
    mu2 = mean(X2);

    % Compute the within-class scatter matrices
    S1 = cov(X1);
    S2 = cov(X2);

    % Compute the within-class scatter matrix
    Sw = (S1 + S2)/2;

    % Compute the optimal direction for maximum class separation
    v = inv(Sw) * (mu1 - mu2)';

    % Normalize the vector to have unit norm
    v = v / norm(v);
end
