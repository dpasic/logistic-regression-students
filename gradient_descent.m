function [theta, J, exit_flag] = gradient_descent(x, y, theta, lambda)
    % set options for the minimize function
    % 'GradObj': 'on' -> provide gradient to the implementation
    % 'MaxIter': 100 -> max number of the algorithm iterations
    options = optimset('GradObj', 'on', 'MaxIter', 100);

    % function minimization uses pointer to cost_function with theta and options
    [theta, J, exit_flag] = fminunc(@(theta)(cost_function(x, y, theta, lambda)), theta, options);
end