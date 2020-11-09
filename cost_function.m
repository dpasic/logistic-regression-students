function [J, gradient] = cost_function(x, y, theta, lambda)
    % number of examples
    m = size(x, 1);
    % compute hypothesis
    h = sigmoid(x * theta);

    % we should not regularize the parameter theta_zero
    theta_cut = theta(2:end, 1);
    % compute regularization parameter
    regularization_param = (lambda / (2 * m)) * (theta_cut' * theta_cut);

    % compute J(theta)
    J = (1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h)) + regularization_param;
    % compute derivative of J(theta) -> gradient step
    gradient = (1 / m) * (x' * (h - y)) + regularization_param;
end