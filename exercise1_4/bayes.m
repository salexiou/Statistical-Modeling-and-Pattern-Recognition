%% Initialization
clear all; close all; clc

% Parameters of the distributions
mu1 = [2 3];
sigma1 = [2 0.5; 0.5 1];

mu2 = [4 4];
sigma2 = [1.5 -0.3; -0.3 0.8];

% Prior probabilities
P_w1_values = [0.1, 0.25, 0.5, 0.75, 0.9];

% Colors for the decision boundaries
colors = {'r', 'g', 'b', 'm', 'c'};

% Create a grid for the x values
[x, y] = meshgrid(-1:0.1:6.5, -1:0.1:6.5);
X = [x(:) y(:)];

% Calculate the probability densities
F1 = mvnpdf(X, mu1, sigma1);
F2 = mvnpdf(X, mu2, sigma2);

F1 = reshape(F1, size(x));
F2 = reshape(F2, size(x));

% Plot the contour lines
figure;
hold on;
contour(x, y, F1, 10, 'LineColor', 'r');
contour(x, y, F2, 10, 'LineColor', 'b');
grid on;
xlabel('x');
ylabel('y');
title('Contour lines onto x-y plane'); 

% Calculate and plot decision boundaries for each prior probability
for i = 1:length(P_w1_values)
    P_w1 = P_w1_values(i);
    P_w2 = 1 - P_w1;
    
    % Discriminant functions
    g1 = -1/2 * ((x - mu1(1)).^2 / sigma1(1, 1) + (y - mu1(2)).^2 / sigma1(2, 2)) ...
        + log(P_w1) - 1/2 * log(det(sigma1));
    g2 = -1/2 * ((x - mu2(1)).^2 / sigma2(1, 1) + (y - mu2(2)).^2 / sigma2(2, 2)) ...
        + log(P_w2) - 1/2 * log(det(sigma2));
    
    % Decision boundary
    contour(x, y, g1 - g2, [0 0], 'LineWidth', 2, 'LineColor', colors{i}, ...
        'DisplayName', sprintf('P(\\omega_1) = %.2f', P_w1));
end
legend show;
xlabel('x_1');
ylabel('x_2');
title('Decision Boundaries for Different Prior Probabilities');
hold off;



fprintf('Program paused. Press enter to continue.\n');
pause
clc; close all; clear all;


% Parameters of the distributions
mu1 = [2 3];
mu2 = [4 4];
sigma2 = sigma1 = [1.2 0.4; 0.4 1.2];

% Prior probabilities
P_w1_values = [0.1, 0.25, 0.5, 0.75, 0.9];

% Colors for the decision boundaries
colors = {'r', 'g', 'b', 'm', 'c'};

% Create a grid for the x values
[x, y] = meshgrid(-1:0.1:6.5, -1:0.1:6.5);
X = [x(:) y(:)];

% Calculate the probability densities
F1 = mvnpdf(X, mu1, sigma1);
F2 = mvnpdf(X, mu2, sigma2);

F1 = reshape(F1, size(x));
F2 = reshape(F2, size(x));

% Plot the contour lines
figure;
hold on;
contour(x, y, F1, 10, 'LineColor', 'r');
contour(x, y, F2, 10, 'LineColor', 'b');
grid on;
xlabel('x');
ylabel('y');
title('Contour lines onto x-y plane');

% Calculate and plot decision boundaries for each prior probability
for i = 1:length(P_w1_values)
    P_w1 = P_w1_values(i);
    P_w2 = 1 - P_w1;
    
    % Discriminant functions
    g1 = -0.5 * ((x - mu1(1)).^2 / sigma1(1, 1) + (y - mu1(2)).^2 / sigma1(2, 2)) ...
        + log(P_w1);
    g2 = -0.5 * ((x - mu2(1)).^2 / sigma2(1, 1) + (y - mu2(2)).^2 / sigma2(2, 2)) ...
        + log(P_w2);
    
    % Decision boundary
    contour(x, y, g1 - g2, [0 0], 'LineWidth', 2, 'LineColor', colors{i}, ...
        'DisplayName', sprintf('P(\\omega_1) = %.2f', P_w1));
end

legend show;
xlabel('x_1');
ylabel('x_2');
title('Decision Boundaries for Different Prior Probabilities');
hold off;

title('Contour lines of the distributions P(x|\omega_1) and P(x|\omega_2)');
xlabel('x_1');
ylabel('x_2');
legend('P(x|\omega_1)', 'P(x|\omega_2)', 'Location', 'northeast');
grid on;
hold off;






